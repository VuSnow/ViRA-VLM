from transformers import AutoTokenizer
import torch
from typing import Dict, List


class DataCollator:
    def __init__(self, tokenizer, dtype=torch.bfloat16, add_eos_token=True):
        self.tokenizer = tokenizer
        self._dtype = dtype
        self.add_eos_token = add_eos_token

        if self.tokenizer.padding_side != "left":
            self.tokenizer.padding_side = "left"

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        # Stack images
        images_tensor = torch.stack([item['image_tensor']
                                    for item in batch]).to(self._dtype)
        prompts = [item['prompt'] for item in batch]
        labels = [item['label'] for item in batch]

        answers_text = [
            f"{prompt} {self.tokenizer.sep_token} <|im_start|> {label} <|im_end|>{self.tokenizer.eos_token}" for prompt, label in zip(prompts, labels)]
        answers_encoding = self.tokenizer(
            answers_text,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = answers_encoding.input_ids
        attention_mask = answers_encoding.attention_mask
        labels = input_ids.clone()

        sep_token_id = self.tokenizer.sep_token_id
        for i in range(labels.shape[0]):
            sep_idx = (labels[i] == sep_token_id).nonzero(as_tuple=True)[0]
            if sep_idx.numel() > 0:
                last_prompt_idx = sep_idx[0].item()
                labels[i, :last_prompt_idx + 1] = -100

        return {
            "image_tensor": images_tensor,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
