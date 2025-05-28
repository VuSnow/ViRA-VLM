from typing import List, Dict, Any
import torch

class VQADataCollator:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Stack các trường bắt buộc cho vision-language
        batch = {
            "input_ids": torch.stack([f["input_ids"] for f in features], dim=0),           # [B, seq_len]
            "attention_mask": torch.stack([f["attention_mask"] for f in features], dim=0), # [B, seq_len]
            "image_tensor": torch.stack([f["image_tensor"] for f in features], dim=0),     # [B, C, H, W]
            "labels": torch.stack([f["labels"] for f in features], dim=0)                  # [B, seq_len]
        }
        return batch
