import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from peft import get_peft_model, LoraConfig, TaskType
from huggingface_hub import login
from model.vision_encoder_model.EVA_Clip_02 import EVA02VisionTower
from model.language_model.LLaVA_SeaLLM import LLaVA_seaLLMs
from fussion_modules.Cross_Attention import CrossAttention
from stage_1.Captioning import CaptionGenerating
from stage_1.Dataloader import ImageCaptionDataset


def train(model, dataloader, optimizer, device, epochs=100):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for image_tensor, prompt_texts, caption_texts in pbar:
            image_tensor = image_tensor.to(device)
            loss = model(image_tensor, prompt_texts, caption_texts)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        print(
            f"[Epoch {epoch+1}] Avg Loss: {total_loss / len(dataloader):.4f}")


def main():
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        device_map = "balanced" if gpu_count > 1 else "auto"
        device = torch.device("cuda")
    else:
        device_map = "cpu"
        device = torch.device("cpu")

    vision_encoder = EVA02VisionTower(unfreeze_vision_tower=False)
    llm = LLaVA_seaLLMs(device_map=device_map, requires_grad=False)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        # or try "k_proj", "out_proj", etc.
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    llm.model = get_peft_model(llm.model, lora_config)
    llm.model.print_trainable_parameters()
    cross_attention = CrossAttention(
        query_dim=llm.embed_dim,
        key_dim=vision_encoder.embed_dims,
        hidden_dim=llm.embed_dim
    )
    model = CaptionGenerating(vision_encoder, cross_attention, llm).to(device)

    login()
    dataset = load_dataset("5CD-AI/Viet-LAION-Gemini-VQA", split="train[:100]")
    images = [item['image'] for item in dataset]
    descriptions = [item['description'] for item in dataset]
    custom_dataset = ImageCaptionDataset(images, descriptions, vision_encoder)
    dataloader = DataLoader(custom_dataset, batch_size=16, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=1e-5)

    train(model, dataloader, optimizer, device, epochs=10)
