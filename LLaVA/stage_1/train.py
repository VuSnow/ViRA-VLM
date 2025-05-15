import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datasets import load_dataset
from tqdm import tqdm
from peft import get_peft_model, LoraConfig, TaskType
from huggingface_hub import login
from LLaVA.model.vision_encoder_model.EVA_Clip_02 import EVA02VisionTower
from LLaVA.model.language_model.LLaVA_SeaLLM import LLaVA_seaLLMs
from LLaVA.fussion_modules.Cross_Attention import CrossAttention
from LLaVA.stage_1.Captioning import CaptionGenerating
from LLaVA.stage_1.Dataloader import ImageCaptionDataset
from LLaVA.model.metrics import calculate_token_accuracy, calculate_bleu_scores, calculate_rouge_scores
import sys
import os
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# Thêm thư mục gốc project vào sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT_DIR)


def train(model, dataloader, optimizer, device, epochs=100, accumulation_steps=2, save_dir='./saved_models'):
    scaler = torch.amp.GradScaler(
        enabled=(model.llm.dtype != torch.float32))
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        processed_batches = 0

        batch_accuracies = []
        epoch_references = []
        epoch_hypotheses = []

        optimizer.zero_grad()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for i, (image_tensor, prompt_texts, description_texts) in enumerate(pbar):
            image_tensor = image_tensor.to(
                device=device, dtype=model.llm.dtype)
            with torch.amp.autocast(enabled=(model.llm.dtype != torch.float32), dtype=model.llm.dtype):
                outputs = model(image_tensor, prompt_texts, description_texts)
                loss = outputs.loss / accumulation_steps

            if torch.isnan(loss):
                logger.warning(
                    f"NaN loss detected in epoch {epoch+1}, batch {i}. Skipping optimizer step.")
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            if outputs.logits is not None:
                labels = model.llm.tokenizer(description_texts, return_tensors="pt",
                                             padding=True, truncation=True, max_length=model.llm.max_seq_length)["input_ids"].to(device)
                accuracy = calculate_token_accuracy(
                    outputs.logits, labels, model.llm.tokenizer.pad_token_id)
                batch_accuracies.append(accuracy)

            if (i + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if not torch.isnan(loss):
                total_loss += loss.item() * accumulation_steps
                processed_batches += 1

            # Thu thập sample generation nhỏ mỗi 10 batch
            if i % 10 == 0:
                model.eval()
                with torch.no_grad():
                    for img, ref in zip(image_tensor[:2], description_texts[:2]):
                        pred = model.generate_caption(img)
                        epoch_references.append(ref)
                        epoch_hypotheses.append(pred)
                model.train()

            del image_tensor, prompt_texts, description_texts
            torch.cuda.empty_cache()

            for param_group in optimizer.param_groups:
                lr = param_group['lr']

            avg_loss = total_loss / processed_batches if processed_batches > 0 else 0
            avg_acc = sum(batch_accuracies) / \
                len(batch_accuracies) if batch_accuracies else 0

            pbar.set_postfix(loss=f"{loss.item()*accumulation_steps:.4f}",
                             avg_loss=f"{avg_loss:.4f}",
                             avg_acc=f"{avg_acc:.4f}",
                             lr=f"{lr:.6f}")
        # Tính BLEU, ROUGE cuối epoch
        if len(epoch_references) > 0:
            bleu_scores = calculate_bleu_scores(
                [[r] for r in epoch_references], epoch_hypotheses)
            rouge_scores = calculate_rouge_scores(
                epoch_references, epoch_hypotheses)
        else:
            bleu_scores = {"BLEU-1": 0, "BLEU-2": 0, "BLEU-3": 0, "BLEU-4": 0}
            rouge_scores = {"ROUGE-1": 0, "ROUGE-2": 0, "ROUGE-L": 0}

        logger.info(f"[Epoch {epoch+1}] Avg Loss: {avg_loss:.4f}")
        logger.info(f"[Epoch {epoch+1}] Avg Token Accuracy: {avg_acc:.4f}")
        logger.info(f"[Epoch {epoch+1}] BLEU Scores: {bleu_scores}")
        logger.info(f"[Epoch {epoch+1}] ROUGE Scores: {rouge_scores}")

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), save_path)
        logger.info(f"Model saved at {save_path}")

        llm_path = os.path.join(save_dir, f"llm_epoch_{epoch+1}.pt")
        torch.save(model.llm.state_dict(), llm_path)
        logger.info(f"LLM saved at {llm_path}")

        vision_encoder_path = os.path.join(
            save_dir, f"vision_encoder_epoch_{epoch+1}.pt")
        torch.save(model.vision_encoder.state_dict(), vision_encoder_path)
        logger.info(f"Vision Encoder saved at {vision_encoder_path}")

        cross_attention_path = os.path.join(
            save_dir, f"cross_attention_epoch_{epoch+1}.pt")
        torch.save(model.cross_attention.state_dict(), cross_attention_path)
        logger.info(f"Cross Attention saved at {cross_attention_path}")

        torch.cuda.empty_cache()


def main():
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        device_map = "balanced" if gpu_count > 1 else "auto"
        device = torch.device("cuda")
    else:
        device_map = "cpu"
        device = torch.device("cpu")

    logger.info("Initializing Vision Encoder...")
    vision_encoder = EVA02VisionTower(
        unfreeze_vision_tower=False, delay_load=False)
    llm = LLaVA_seaLLMs(device_map=device_map, requires_grad=True)

    logger.info("Applying LoRA configuration...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        # or try "k_proj", "out_proj", etc.
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    if not llm.is_loaded:
        llm.load_model()

    llm.model = get_peft_model(llm.model, lora_config)
    logger.info("LoRA applied. Trainable parameters:")
    llm.model.print_trainable_parameters()

    compute_dtype = llm.dtype
    logger.info(f"Using compute dtype: {compute_dtype}")

    vision_encoder = vision_encoder.to(device=device, dtype=compute_dtype)

    logger.info("Initializing Cross Attention...")
    cross_attention = CrossAttention(
        query_dim=llm.embed_dim,
        key_dim=vision_encoder.embed_dims,
        hidden_dim=llm.embed_dim
    )
    cross_attention = cross_attention.to(device=device, dtype=compute_dtype)

    logger.info("Initializing Captioning Model...")
    model = CaptionGenerating(vision_encoder, cross_attention, llm)

    logger.info("Loading dataset...")
    login()
    dataset = load_dataset("5CD-AI/Viet-LAION-Gemini-VQA")
    first_100 = [x for _, x in zip(range(100), dataset["train"])]
    images = [item['image'] for item in first_100]
    descriptions = [item['description'] for item in first_100]
    logger.info(f"Loaded {len(images)} valid samples.")

    custom_dataset = ImageCaptionDataset(
        images, descriptions, vision_encoder.transform)
    dataloader = DataLoader(custom_dataset, batch_size=16,
                            shuffle=True, num_workers=os.cpu_count() // 2, pin_memory=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=1e-5)
    logger.info(
        f"Optimizer AdamW created with LR: {1e-5}. Optimizing {len(trainable_params)} tensors.")

    logger.info("Starting training...")
    train(model, dataloader, optimizer, device, epochs=10,
          accumulation_steps=2)
    logger.info("Training finished.")


if __name__ == "__main__":
    main()
