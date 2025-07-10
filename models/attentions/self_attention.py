import torch
import torch.nn as nn
from typing import Optional
from easydict import EasyDict


class SelfAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        num_heads: int = 8,
        ffn_multiplier: int = 4,
        dropout: float = 0.1,
        add_positional: bool = False,
        hidden_dim: Optional[int] = None,
        max_seq_len: int = 2048
    ):
        super().__init__()
        self.query_dim = query_dim
        self.hidden_dim = hidden_dim or query_dim
        self.dropout = dropout
        self.num_heads = num_heads
        self.ffn_multiplier = ffn_multiplier
        self.add_positional = add_positional
        self.max_seq_len = max_seq_len

        if self.add_positional:
            self.pos_embedding = nn.Parameter(
                torch.zeros(1, self.max_seq_len, self.hidden_dim))

        self.proj = (
            nn.Linear(in_features=self.query_dim, out_features=self.hidden_dim)
            if self.query_dim != self.hidden_dim else nn.Identity()
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
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
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ):
        if not isinstance(query, torch.Tensor):
            raise TypeError(
                f"Expected query to be a Tensor, but got {type(query)}")

        query_proj = self.proj(query)

        if self.add_positional:
            seq_len = query.size(1)
            query_proj = query_proj + self.pos_embedding[:, :seq_len, :]

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.to(dtype=torch.bool)

        attn_weights = None
        if need_weights:
            attn_output, attn_weights = self.attention(
                query=query_proj,
                key=query_proj,
                value=query_proj,
                key_padding_mask=key_padding_mask,
                need_weights=True,
            )
        else:
            attn_output, _ = self.attention(
                query=query_proj,
                key=query_proj,
                value=query_proj,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )

        x = self.layer_norm1(attn_output + query_proj)
        ffn_output = self.ffn(x)
        output = self.layer_norm2(ffn_output + x)

        return output, attn_weights if need_weights else None

    @property
    def dim(self):
        return self.hidden_dim
