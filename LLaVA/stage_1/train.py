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
import sys
import os
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# Thêm thư mục gốc project vào sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT_DIR)


def calculate_accuracy(predictions, labels):
    _, predicted = torch.max(predictions, 1)  # Lấy chỉ số có giá trị cao nhất
    correct = (predicted == labels).sum().item()
    accuracy = correct / labels.size(0)
    return accuracy


def train(model, dataloader, optimizer, device, epochs=100, accumulation_steps=2, save_dir='./saved_models'):
    model.train()
    scaler = torch.cuda.amp.GradScaler(
        enabled=(model.llm.dtype == torch.float32))

    for epoch in range(epochs):
        total_loss = 0
        processed_batches = 0
        optimizer.zero_grad()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for i, (image_tensor, prompt_texts, description_texts) in enumerate(pbar):
            with torch.cuda.amp.autocast(enabled=(model.llm.dtype != torch.float32), dtype=model.llm.dtype):
                loss = model(image_tensor, prompt_texts, description_texts)
                loss = loss / accumulation_steps

            if torch.isnan(loss):
                logger.warning(
                    f"NaN loss detected in epoch {epoch+1}, batch {i}. Skipping optimizer step.")
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
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

            # Tính accuracy
            accuracy = calculate_accuracy(
                model(image_tensor), description_texts)
            total_accuracy += accuracy

            for param_group in optimizer.param_groups:
                lr = param_group['lr']

            pbar.set_postfix(loss=f"{loss.item()*accumulation_steps:.4f}",
                             avg_loss=f"{total_loss / processed_batches:.4f}" if processed_batches > 0 else "N/A",
                             accuracy=f"{total_accuracy / processed_batches:.4f}",
                             lr=f"{lr:.6f}")

        avg_epoch_loss = total_loss / processed_batches if processed_batches > 0 else 0
        avg_epoch_accuracy = total_accuracy / \
            processed_batches if processed_batches > 0 else 0
        logger.info(f"[Epoch {epoch+1}] Avg Loss: {avg_epoch_loss:.4f}")
        logger.info(
            f"[Epoch {epoch+1}] Avg Loss: {avg_epoch_loss:.4f}, Avg Accuracy: {avg_epoch_accuracy:.4f}, Learning Rate: {lr:.6f}")

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
