import torch
import torch.nn as nn
from typing import Optional
from easydict import EasyDict
from models.attentions.self_attention import SelfAttention
from models.attentions.cross_attention import CrossAttention


class FusionBlock(nn.Module):
    def __init__(
        self,
        llm_dim: int,
        vision_dim: int,
        num_heads: int = 8,
        ffn_multiplier: int = 4,
        dropout: float = 0.1,
        add_positional: bool = False,
        hidden_dim: Optional[int] = None,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.llm_dim = llm_dim
        self.vision_dim = vision_dim
        self.hidden_dim = hidden_dim or llm_dim
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.ffn_multiplier = ffn_multiplier
        self.num_heads = num_heads
        self.add_positional = add_positional

        self.vision_self_attn = SelfAttention(
            query_dim=self.vision_dim,
            hidden_dim=self.vision_dim,
            num_heads=self.num_heads,
            ffn_multiplier=self.ffn_multiplier,
            dropout=self.dropout,
            add_positional=self.add_positional,
            max_seq_len=self.max_seq_len,
        )

        self.llm_self_attn = SelfAttention(
            query_dim=self.llm_dim,
            hidden_dim=self.llm_dim,
            num_heads=self.num_heads,
            ffn_multiplier=self.ffn_multiplier,
            dropout=self.dropout,
            add_positional=self.add_positional,
            max_seq_len=self.max_seq_len,
        )

        self.vision_cross_llm = CrossAttention(
            query_dim=self.vision_dim,
            kv_dim=self.llm_dim,
            hidden_dim=self.vision_dim,
            num_heads=self.num_heads,
            ffn_multiplier=self.ffn_multiplier,
            dropout=self.dropout,
            add_positional=self.add_positional,
            max_seq_len=self.max_seq_len,
        )

        self.llm_cross_vision = CrossAttention(
            query_dim=self.llm_dim,
            kv_dim=self.vision_dim,
            hidden_dim=self.llm_dim,
            num_heads=self.num_heads,
            ffn_multiplier=self.ffn_multiplier,
            dropout=self.dropout,
            add_positional=self.add_positional,
            max_seq_len=self.max_seq_len,
        )

        self.vision_norm = nn.LayerNorm(self.vision_dim)
        self.llm_norm = nn.LayerNorm(self.llm_dim)
        self.vision_ffn = nn.Sequential(
            nn.Linear(in_features=self.vision_dim,
                      out_features=self.vision_dim * self.ffn_multiplier),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(in_features=self.vision_dim *
                      self.ffn_multiplier, out_features=self.vision_dim),
            nn.Dropout(self.dropout),
        )
        self.llm_ffn = nn.Sequential(
            nn.Linear(in_features=self.llm_dim,
                      out_features=self.llm_dim * self.ffn_multiplier),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(in_features=self.llm_dim * self.ffn_multiplier,
                      out_features=self.llm_dim),
            nn.Dropout(self.dropout),
        )

    def forward(
        self,
        vision_input: torch.Tensor,
        llm_input: torch.Tensor,
        vision_mask: Optional[torch.Tensor] = None,
        llm_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ):
        vision_sa = self.vision_self_attn(
            query=vision_input, key_padding_mask=vision_mask)
        llm_sa = self.llm_self_attn(query=llm_input, key_padding_mask=llm_mask)

        # Bi-directional cross-attention
        vision_cross = self.vision_cross_llm(
            query=vision_sa["output"],
            key_value=llm_sa["output"],
            kv_mask=llm_mask,
            need_weights=need_weights,
        )
        llm_cross = self.llm_cross_vision(
            query=llm_sa["output"],
            key_value=vision_sa["output"],
            kv_mask=vision_mask,
            need_weights=need_weights,
        )

        # Residual + LayerNorm
        vision_fused = self.vision_norm(vision_input + vision_cross["output"])
        llm_fused = self.llm_norm(llm_input + llm_cross["output"])

        # FeedForward + Residual + LayerNorm
        vision_out = self.vision_norm(
            vision_fused + self.vision_ffn(vision_fused))
        llm_out = self.llm_norm(llm_fused + self.llm_ffn(llm_fused))

        return {
            "vision_output": vision_out,
            "llm_output": llm_out,
            "vision_attn_weights": vision_cross["attn_weights"] if need_weights else None,
            "llm_attn_weights": llm_cross["attn_weights"] if need_weights else None,
        }
