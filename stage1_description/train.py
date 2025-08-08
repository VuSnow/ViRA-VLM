from datasets import disable_progress_bar
from transformers import AutoTokenizer, TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
from transformers.trainer_utils import get_last_checkpoint
from torchvision import transforms
from easydict import EasyDict
from stage1_description.dataset.description_dataset import DescriptionDataset
from stage1_description.utils.metrics import build_compute_metrics, preprocess_logits_for_metrics
from stage1_description.utils.data_collator import DataCollator
from stage1_description.model.description_model import DescriptionModel, DescriptionModelConfig
from stage1_description.utils.utils import print_trainable_parameters
from huggingface_hub import login
from dotenv import load_dotenv
from datasets import load_from_disk
import yaml
import argparse
import os
import torch
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"
print(f"torch available: {torch.cuda.is_available()}")


def main():
    torch.autograd.set_detect_anomaly(True)
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

    print(f"1. Loading configs from {args.config_path}")
    with open(args.config_path, "r") as f:
        configs = EasyDict(yaml.safe_load(f))

    print(f"2. Loading transform")
    transform = transforms.Compose([
        transforms.Resize(
            configs.eva_vision_model.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(configs.eva_vision_model.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=configs.eva_vision_model.mean,
                             std=configs.eva_vision_model.std)
    ])

    print(f"3. Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(configs.language_model.name)

    print(f"4. Adding special tokens")
    print(f"Len before adding special tokens: {len(tokenizer)}")
    special_tokens_dict = {}
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = "<|im_bos|>"
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = "<|im_eos|>"
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = "<|im_pad|>"
    if tokenizer.sep_token is None:
        special_tokens_dict["sep_token"] = "<|im_sep|>"
    if tokenizer.mask_token is None:
        special_tokens_dict["mask_token"] = "<|im_mask|>"
    if special_tokens_dict:
        tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Len after adding special tokens: {len(tokenizer)}")

    print(f"5. Loading model config")
    model_config = DescriptionModelConfig(**configs)

    print(f"6. Resizing model embedding size")
    model_config.vocab_size = len(tokenizer)
    model = DescriptionModel(model_config)
    model.llm.resize_token_embeddings(len(tokenizer))
    model.llm.tie_weights()
    # model.gradient_checkpointing_enable()
    print(f"Model embedding size resized to: {len(tokenizer)}")

    print(f"7. Freezing model parameters")
    if args.freeze_llm:
        for param in model.llm.parameters():
            param.requires_grad = False
    if args.freeze_vision:
        for param in model.vision.parameters():
            param.requires_grad = False
    print(f"8. Converting model to {model.llm.dtype} dtype")
    dtype = model.llm.dtype
    # model = model.to(dtype=torch.bfloat16)
    print(f"8.1. Model dtype: {model.llm.dtype}")
    print_trainable_parameters(model)

    print(f"9. Logging in Hugging Face")
    login(os.getenv("HF_TOKEN"))
    dataset = load_from_disk(
        "/home/user05/dungvm/stage1_description/dataset/viet_laion_gemini_vqa_cleaned_v2")
    print(f"Dataset loaded from {dataset}. Number of samples: {len(dataset)}")
    dataset = dataset.select(range(args.num_samples))
    split_dataset = dataset.train_test_split(
        test_size=args.split_ratio, seed=args.seed)

    print(f"10. Splitting dataset into train and eval")
    train_dataset = DescriptionDataset(
        split_dataset["train"], transform=transform)
    eval_dataset = DescriptionDataset(
        split_dataset["test"], transform=transform)

    print(f"11. Initializing training arguments")
    args_trainer = TrainingArguments(
        output_dir=configs.training.output_dir,
        overwrite_output_dir=configs.training.overwrite_output_dir,
        eval_strategy=configs.training.eval_strategy,
        per_device_train_batch_size=configs.training.per_device_train_batch_size,
        per_device_eval_batch_size=configs.training.per_device_eval_batch_size,
        gradient_accumulation_steps=configs.training.gradient_accumulation_steps,
        eval_accumulation_steps=configs.training.eval_accumulation_steps,
        learning_rate=float(configs.training.learning_rate),
        lr_scheduler_type="cosine",
        weight_decay=configs.training.weight_decay,
        optim=configs.training.optim,
        num_train_epochs=configs.training.num_train_epochs,
        logging_dir=configs.training.logging_dir,
        logging_strategy=configs.training.logging_strategy,
        logging_first_step=configs.training.logging_first_step,
        logging_steps=configs.training.logging_steps,
        save_strategy=configs.training.save_strategy,
        save_total_limit=configs.training.save_total_limit,
        save_safetensors=configs.training.save_safetensors,
        seed=args.seed,
        bf16=configs.training.bf16,
        fp16=configs.training.fp16,
        dataloader_num_workers=configs.training.dataloader_num_workers,
        disable_tqdm=configs.training.disable_tqdm,
        load_best_model_at_end=configs.training.load_best_model_at_end,
        metric_for_best_model=configs.training.metric_for_best_model,
        greater_is_better=configs.training.greater_is_better,
        report_to=["tensorboard"],
        remove_unused_columns=configs.training.remove_unused_columns,
        max_grad_norm=configs.training.max_grad_norm,
        warmup_ratio=configs.training.warmup_ratio,
        ddp_find_unused_parameters=True,
        # gradient_checkpointing=True,
    )
    print(f"12. Initializing data collator")
    data_collator = DataCollator(
        tokenizer=tokenizer,
        dtype=dtype,
        add_eos_token=True,
    )
    print(f"13. Initializing compute metrics function")
    compute_metrics_fn = build_compute_metrics(tokenizer)

    print(f"14. Initializing trainer")
    trainer = Trainer(
        model=model,
        args=args_trainer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        tokenizer=tokenizer
    )
    print(f"15. Getting trained model")
    trained_model = trainer.model

    print(f"16. Checking if input embeddings and output head weights are tied")
    is_tied = (trained_model.llm.get_input_embeddings().weight.data_ptr(
    ) == trained_model.llm.get_output_embeddings().weight.data_ptr())
    print(f"Are the input embeddings and output head weights tied? {is_tied}")

    if not is_tied:
        print("Warning: Weights are NOT tied after loading. There might be an issue.")
    else:
        print(
            "Confirmation: Weights are correctly tied. The warning can be safely ignored.")

    print(f"17. Getting last checkpoint")
    last_checkpoint = None
    if os.path.isdir(configs.training.output_dir):
        last_checkpoint = get_last_checkpoint(configs.training.output_dir)
        print(f"Last checkpoint: {last_checkpoint}")
    print(f"18. Starting training")
    trainer.train()
    print(
        f"After training start - model dtype: {next(model.parameters()).dtype}")
    # tokenizer.save_pretrained(configs.training.output_dir)
    # model.save_pretrained(configs.training.output_dir)


if __name__ == "__main__":
    main()
