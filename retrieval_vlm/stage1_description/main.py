from transformers import AutoTokenizer
from retrieval_vlm.stage1_description.dataset.image_caption_dataset import ImageCaptionDataset
from retrieval_vlm.stage1_description.model.generating_caption import GeneratingCaption
from retrieval_vlm.stage1_description.trainer.train import Train
from retrieval_vlm.stage1_description.config.model_config import GeneratingCaptionConfig
from datasets import load_dataset
from easydict import EasyDict
from torchvision import transforms
import argparse
import yaml
import os


def main():
    parser = argparse.ArgumentParser(description="Training description generating model")
    parser.add_argument("--config", type=str, default="./config/config.yaml", help="Path to the config file")
    parser.add_argument("--dataset", type=str, default="5CD-AI/Viet-OCR-VQA-flash2", help="Dataset name")
    args = parser.parse_args()

    # Load config
    config = EasyDict(yaml.safe_load(open(args.config, "r")))
    model_config = GeneratingCaptionConfig(
        vision_encoder_config=config.vision_encoder_config,
        cross_attention_config=config.cross_attention_config,
        language_model=config.language_model,
        lora=config.lora,
        train_config=config.train_config,
    )
    # Load tokenizer and transform
    tokenizer = AutoTokenizer.from_pretrained(config.language_model.name)
    transform = transforms.Compose([
        transforms.Resize(
            config.vision_encoder_config.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(config.vision_encoder_config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.vision_encoder_config.mean, std=config.vision_encoder_config.std)
    ])
    # Load dataset
    dataset = load_dataset(args.dataset)
    # Split dataset
    split_dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
    train_data = split_dataset['train']
    test_data = split_dataset['test']
    # Create captioning train dataset
    captioning_train_dataset = ImageCaptionDataset(
        dataset=train_data,
        transform=transforms,
        tokenizer=tokenizer
    )
    # Create captioning eval dataset
    captioning_eval_dataset = ImageCaptionDataset(
        dataset=test_data,
        transform=transforms,
        tokenizer=tokenizer
    )
    

    # Load model
    model = GeneratingCaption(model_config)

    # Create trainer
    trainer = Train(
        model=model,
        train_dataset=captioning_train_dataset,
        tokenizer=tokenizer,
        transform=transform,
        config=model_config,
        eval_dataset=captioning_eval_dataset
    )

    # Train model
    trainer.train()

    # # Evaluate model
    # trainer.evaluate()

    # # Save model
    # trainer.save_model()

if __name__ == "__main__":
    main()
