import torch
import torch.nn as nn
from typing import Optional
from easydict import EasyDict


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        kv_dim: int,
        num_heads: int = 8,
        ffn_multiplier: int = 4,
        dropout: float = 0.1,
        add_positional: bool = False,
        hidden_dim: Optional[int] = None,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.query_dim = query_dim
        self.kv_dim = kv_dim
        self.hidden_dim = hidden_dim or query_dim
        self.dropout = dropout
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.add_positional = add_positional
        self.ffn_multiplier = ffn_multiplier

        self.proj_q = (
            nn.Linear(in_features=self.query_dim, out_features=self.hidden_dim)
            if self.query_dim != self.hidden_dim else nn.Identity()
        )

        self.query_norm = nn.LayerNorm(self.hidden_dim)
        self.kv_norm = nn.LayerNorm(self.hidden_dim)

        self.proj_k = (
            nn.Linear(in_features=self.kv_dim, out_features=self.hidden_dim)
            if self.kv_dim != self.hidden_dim else nn.Identity()
        )

        if self.add_positional:
            self.query_pos_embedding = nn.Parameter(
                torch.zeros(1, self.max_seq_len, self.hidden_dim))
            self.kv_pos_embedding = nn.Parameter(
                torch.zeros(1, self.max_seq_len, self.hidden_dim))

        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            kdim=self.hidden_dim,
            vdim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True,
        )

        self.layer_norm1 = nn.LayerNorm(self.hidden_dim)
        self.layer_norm2 = nn.LayerNorm(self.hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(in_features=self.hidden_dim,
                      out_features=self.hidden_dim * self.ffn_multiplier),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(in_features=self.hidden_dim *
                      self.ffn_multiplier, out_features=self.hidden_dim),
            nn.Dropout(self.dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        kv_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ):
        if not isinstance(query, torch.Tensor):
            raise TypeError(
                f"Expected query to be a Tensor, but got {type(query)}")

        if not isinstance(key_value, torch.Tensor):
            raise TypeError(
                f"Expected key_value to be a Tensor, but got {type(key_value)}")

        query_proj = self.query_norm(self.proj_q(query))
        kv_proj = self.kv_norm(self.proj_k(key_value))

        if self.add_positional:
            lq = query_proj.size(1)
            lkv = kv_proj.size(1)
            query_proj = query_proj + self.query_pos_embedding[:, :lq, :]
            kv_proj = kv_proj + self.kv_pos_embedding[:, :lkv, :]

        if kv_mask is not None:
            kv_mask = kv_mask.to(dtype=torch.bool)

        # print("kv_mask", kv_mask.shape, kv_mask.dtype, kv_mask)
        # print("query_proj", query_proj.shape, torch.isnan(query_proj).any())
        # print("kv_proj", kv_proj.shape, torch.isnan(kv_proj).any())
        # print("Mask all masked:", kv_mask.all(dim=1))
        # print("NaN in query_proj:", torch.isnan(query_proj).any())
        # print("NaN in kv_proj:", torch.isnan(kv_proj).any())
        attn_weights = None
        if need_weights:
            attn_output, attn_weights = self.attention(
                query=query_proj,
                key=kv_proj,
                value=kv_proj,
                key_padding_mask=kv_mask,
                need_weights=True,
            )
            # print("attn_weights shape:", attn_weights.shape)
            # print("attn_weights stats: min", attn_weights.min().item(),
            #     "max", attn_weights.max().item(),
            #     "mean", attn_weights.mean().item(),
            #     "std", attn_weights.std().item())
            # print("attn_weights[0, :5, :5]:", attn_weights[0, :5, :5])
        else:
            attn_output, _ = self.attention(
                query=query_proj,
                key=kv_proj,
                value=kv_proj,
                key_padding_mask=kv_mask,
                need_weights=False,
            )

        x = self.layer_norm1(attn_output + query_proj)
        ffn_output = self.ffn(x)
        output = self.layer_norm2(ffn_output + x)
        return output, attn_weights if need_weights else None

    @property
    def dim(self):
        return self.hidden_dim
