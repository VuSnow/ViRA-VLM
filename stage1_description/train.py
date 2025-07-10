from datasets import disable_progress_bar
from transformers import AutoTokenizer, TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
from torchvision import transforms
from datasets import load_dataset
from easydict import EasyDict
from stage1_description.dataset.description_dataset import DescriptionDataset
from stage1_description.utils.metrics import build_compute_metrics, preprocess_logits_for_metrics
from stage1_description.utils.data_collator import DataCollator
from stage1_description.model.description_model import DescriptionModel, DescriptionModelConfig
from stage1_description.utils.utils import print_trainable_parameters, check_image_validity
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import Dataset
from huggingface_hub import login
from dotenv import load_dotenv
from datasets import load_from_disk
import yaml
import argparse
import os
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"


def main():
    load_dotenv()
    choice_dataset = ["5CD-AI/Viet-LAION-Gemini-VQA", "5CD-AI/Viet-Geometry-VQA",
                      "5CD-AI/Viet-Sketches-VQA", "5CD-AI/Viet-Doc-VQA-II"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str,
                        default="/models/stage1_description/configs/config.yaml")
    parser.add_argument("--dataset_name", type=str, required=True,
                        choices=choice_dataset, default="5CD-AI/Viet-LAION-Gemini-VQA")
    parser.add_argument("--num_samples", type=int, required=True, default=1000)
    parser.add_argument("--split_ratio", type=float,
                        required=True, default=0.1)
    parser.add_argument("--seed", type=int, required=True, default=42)
    parser.add_argument(
        "--freeze_llm", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--freeze_vision",
                        type=lambda x: x.lower() == "true", default=False)

    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        configs = EasyDict(yaml.safe_load(f))

    model_config = DescriptionModelConfig(**configs)
    model = DescriptionModel(model_config)

    if args.freeze_llm:
        for param in model.llm.parameters():
            param.requires_grad = False

    if args.freeze_vision:
        for param in model.vision.parameters():
            param.requires_grad = False

    dtype = model.llm.dtype
    model = model.to(dtype=dtype)
    print_trainable_parameters(model)

    transform = transforms.Compose([
        transforms.Resize(
            configs.eva_vision_model.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(configs.eva_vision_model.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=configs.eva_vision_model.mean,
                             std=configs.eva_vision_model.std)
    ])
    tokenizer = AutoTokenizer.from_pretrained(configs.language_model.name)

    login(os.getenv("HF_TOKEN"))
    dataset = load_from_disk(
        "/home/user05/dungvm/stage1_description/dataset/viet_laion_gemini_vqa_cleaned_v2")
    print(f"Dataset loaded from {dataset}. Number of samples: {len(dataset)}")
    dataset = dataset.select(range(args.num_samples))
    # dataset = dataset.filter(check_image_validity, num_proc=4, batched=False)
    split_dataset = dataset.train_test_split(
        test_size=args.split_ratio, seed=args.seed)

    train_dataset = DescriptionDataset(
        split_dataset["train"], transform=transform)
    eval_dataset = DescriptionDataset(
        split_dataset["test"], transform=transform)

    args_trainer = TrainingArguments(
        output_dir=configs.training.output_dir,
        overwrite_output_dir=configs.training.overwrite_output_dir,
        eval_strategy=configs.training.eval_strategy,
        per_device_train_batch_size=configs.training.per_device_train_batch_size,
        per_device_eval_batch_size=configs.training.per_device_eval_batch_size,
        gradient_accumulation_steps=configs.training.gradient_accumulation_steps,
        eval_accumulation_steps=configs.training.eval_accumulation_steps,
        learning_rate=float(configs.training.learning_rate),
        weight_decay=configs.training.weight_decay,
        optim=configs.training.optim,
        num_train_epochs=configs.training.num_train_epochs,
        warmup_steps=configs.training.warmup_steps,
        logging_dir=configs.training.logging_dir,
        logging_strategy=configs.training.logging_strategy,
        logging_first_step=configs.training.logging_first_step,
        logging_steps=configs.training.logging_steps,
        save_strategy=configs.training.save_strategy,
        save_total_limit=configs.training.save_total_limit,
        save_safetensors=configs.training.save_safetensors,
        seed=args.seed,
        bf16=configs.training.bf16,
        dataloader_num_workers=configs.training.dataloader_num_workers,
        disable_tqdm=False,
        load_best_model_at_end=configs.training.load_best_model_at_end,
        metric_for_best_model=configs.training.metric_for_best_model,
        greater_is_better=configs.training.greater_is_better,
        report_to=["tensorboard"],
        remove_unused_columns=configs.training.remove_unused_columns,
        max_grad_norm=1.0,
        warmup_ratio=0.05,
    )
    print("Initialize collator")
    data_collator = DataCollator(
        tokenizer=tokenizer,
        dtype=dtype,
        add_eos_token=True,
    )
    compute_metrics_fn = build_compute_metrics(tokenizer)

    print("Initialize trainer")
    trainer = Trainer(
        model=model,
        args=args_trainer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,  # Sửa lại compute metrics
        # Thêm earlystopping
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    print("Start Training !!!")
    trainer.train()


if __name__ == "__main__":
    main()
