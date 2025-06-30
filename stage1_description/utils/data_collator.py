from transformers import AutoTokenizer
import torch
from typing import Dict, List


class DataCollator:
    def __init__(self, tokenizer: AutoTokenizer, dtype: torch.dtype = torch.bfloat16, add_eos_token: bool = True):
        self.tokenizer = tokenizer
        self._dtype = dtype
        self.add_eos_token = add_eos_token

        if self.tokenizer.padding_side == "right":
            print("Warning: tokenizer's padding_side is 'right'. Changing to 'left' for batch generation consistency.")
            self.tokenizer.padding_side = "left"

    def __call__(self, batch: List[Dict]):
        images_tensor = torch.stack([item['image_tensor']
                                    for item in batch]).to(self._dtype)
        prompts = [item['prompt'] for item in batch]
        labels = [item['label'] for item in batch]

        eos_token_str = self.tokenizer.eos_token if self.add_eos_token else ""
        full_texts = [prompt + label + eos_token_str for prompt,
                      label in zip(prompts, labels)]

        full_text_encodings = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        prompt_encodings = self.tokenizer(
            prompts,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

        labels_ids = full_text_encodings['input_ids'].clone()

        for i in range(len(batch)):
            prompt_len = prompt_encodings.attention_mask[i].sum()
            labels_ids[i, :prompt_len] = -100

        labels_ids[labels_ids == self.tokenizer.pad_token_id] = -100

        return {
            "image_tensor": images_tensor,
            "input_ids": full_text_encodings['input_ids'],
            "attention_mask": full_text_encodings['attention_mask'],
            "labels": labels_ids,
        }
