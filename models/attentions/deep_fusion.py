import torch
import torch.nn as nn
from typing import Optional
from models.attentions.fusion_block import FusionBlock


class DeepFusion(nn.Module):
    def __init__(
        self,
        n_layers: int,
        vision_dim: int,
        llm_dim: int,
        n_heads: int = 8,
        ffn_multiplier: int = 4,
        dropout: float = 0.1,
        add_positional: bool = False,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            FusionBlock(
                llm_dim=llm_dim,
                vision_dim=vision_dim,
                num_heads=n_heads,
                ffn_multiplier=ffn_multiplier,
                dropout=dropout,
                add_positional=add_positional,
                max_seq_len=max_seq_len,
            )
            for _ in range(n_layers)
        ])

    def forward(self, vision_input: torch.Tensor, llm_input: torch.Tensor, vision_mask: Optional[torch.Tensor] = None, llm_mask: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None, need_weights: bool = False):
        vision_attn_weights = None
        llm_attn_weights = None
        for layer in self.layers:
            result = layer(
                vision_input=vision_input,
                llm_input=llm_input,
                vision_mask=vision_mask,
                llm_mask=llm_mask,
                need_weights=need_weights,
            )
            vision_input = result["vision_output"]
            llm_input = result["llm_output"]
            if need_weights:
                vision_attn_weights = result["vision_attn_weights"]
                llm_attn_weights = result["llm_attn_weights"]

        return {
            "vision_output": vision_input,
            "llm_output": llm_input,
            "vision_attn_weights": vision_attn_weights,
            "llm_attn_weights": llm_attn_weights,
        }
