from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tqdm import tqdm
from peft import get_peft_model, LoraConfig, TaskType
from huggingface_hub import login
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from LLaVA.model.vision_encoder_model.EVA_Clip_02 import EVA02VisionTower
from LLaVA.model.language_model.LLaVA_SeaLLM import LLaVA_seaLLMs
from LLaVA.model.fussion_modules.Cross_Attention import CrossAttention
from LLaVA.stage1_description.model import CaptionGenerating
from LLaVA.stage1_description.dataset import ImageCaptionDataset
from LLaVA.model.metrics import calculate_token_accuracy, calculate_bleu_scores, calculate_rouge_scores
import sys
import os
import logging
import numpy as np
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT_DIR)

class VLMTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_references = []
        self.epoch_hypotheses = []
        self.best_metric = float('inf')
        self.best_model_path = None
        
    def compute_loss(self, model, inputs, return_outputs=False):
        image_tensor = inputs["image_tensor"]
        prompt_texts = inputs["prompt_texts"]
        description_texts = inputs["description_texts"]
        
        outputs = model(image_tensor, prompt_texts, description_texts)
        loss = outputs.loss
        
        if return_outputs:
            return loss, outputs
        return loss
    
    def log_metrics(self, split, metrics, epoch=None):
        if split == "train":
            # Tính BLEU và ROUGE scores
            if len(self.epoch_references) > 0:
                bleu_scores = calculate_bleu_scores(
                    [[r] for r in self.epoch_references], 
                    self.epoch_hypotheses
                )
                rouge_scores = calculate_rouge_scores(
                    self.epoch_references, 
                    self.epoch_hypotheses
                )
                metrics.update(bleu_scores)
                metrics.update(rouge_scores)
            
            # Reset references và hypotheses cho epoch tiếp theo
            self.epoch_references = []
            self.epoch_hypotheses = []
            
            logger.info(f"[Epoch {epoch}] Metrics: {metrics}")
        
        super().log_metrics(split, metrics, epoch)
    
    def training_step(self, model, inputs):
        loss = super().training_step(model, inputs)
        
        # Thu thập sample generation mỗi 10 batch
        if self.state.global_step % 10 == 0:
            model.eval()
            with torch.no_grad():
                for img, ref in zip(inputs["image_tensor"][:2], inputs["description_texts"][:2]):
                    pred = model.generate_caption(img)
                    self.epoch_references.append(ref)
                    self.epoch_hypotheses.append(pred)
            model.train()
            
        return loss

    def _save_checkpoint(self, model, trial, metrics=None):
        # Override save checkpoint to only save best model
        if metrics is not None and "eval_loss" in metrics:
            current_metric = metrics["eval_loss"]
            if current_metric < self.best_metric:
                self.best_metric = current_metric
                # Save best model
                if self.best_model_path is not None:
                    # Remove previous best model
                    if os.path.exists(self.best_model_path):
                        os.remove(self.best_model_path)
                
                # Save new best model
                self.best_model_path = os.path.join(
                    self.args.output_dir, 
                    "best_model"
                )
                self.save_model(self.best_model_path)
                logger.info(f"New best model saved with eval_loss: {current_metric:.4f}")

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
        unfreeze_vision_tower=False, 
        delay_load=False
    )
    llm = LLaVA_seaLLMs(device_map=device_map, requires_grad=True)

    logger.info("Applying LoRA configuration...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
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

    # Initialize Cross Attention
    logger.info("Initializing Cross Attention...")
    cross_attention = CrossAttention(
        query_dim=llm.embed_dim,
        key_dim=vision_encoder.embed_dims,
        hidden_dim=llm.embed_dim
    )
    cross_attention = cross_attention.to(device=device, dtype=compute_dtype)

    # Initialize Model
    logger.info("Initializing Captioning Model...")
    model = CaptionGenerating(vision_encoder, cross_attention, llm)

    # Load dataset
    logger.info("Loading dataset...")
    login()
    dataset = load_dataset("5CD-AI/Viet-LAION-Gemini-VQA")
    dataset = dataset["train"].select(range(100000)) 
    logger.info(f"Loaded {len(dataset)} samples from dataset")

    images = [item['image'] for item in dataset]
    descriptions = [item['description'] for item in dataset]

    train_images, val_images, train_descriptions, val_descriptions = train_test_split(
        images, 
        descriptions,
        test_size=0.2,  
        random_state=42  
    )

    logger.info(f"Split dataset into {len(train_images)} training samples and {len(val_images)} validation samples")

    # Create datasets
    train_custom_dataset = ImageCaptionDataset(
        train_images, 
        train_descriptions, 
        vision_encoder.transform
    )
    
    val_custom_dataset = ImageCaptionDataset(
        val_images,
        val_descriptions,
        vision_encoder.transform
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./saved_models",
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        weight_decay=0.01,
        warmup_steps=500,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="no",  # We'll handle saving in our custom trainer
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=(compute_dtype != torch.float32),
        report_to="none",
    )

    # Initialize trainer with early stopping
    trainer = VLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_custom_dataset,
        eval_dataset=val_custom_dataset,
        data_collator=lambda x: {
            "image_tensor": torch.stack([item["image_tensor"] for item in x]),
            "prompt_texts": [item["prompt_texts"] for item in x],
            "description_texts": [item["description_texts"] for item in x]
        },
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.01
            )
        ]
    )

    # Start training
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training finished.")

    # Load best model
    if trainer.best_model_path is not None:
        logger.info(f"Loading best model from {trainer.best_model_path}")
        trainer.load_model(trainer.best_model_path)
    else:
        logger.warning("No best model was saved during training!")

if __name__ == "__main__":
    main()
