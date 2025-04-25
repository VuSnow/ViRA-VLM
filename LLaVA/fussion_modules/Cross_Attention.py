import torch
from torch import nn


class CrossAttention(nn.Module):
    def __init__(self, query_dim=1792, key_dim=768, hidden_dim=768, num_heads=8, ffn_multiplier=4, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.ffn_multiplier = ffn_multiplier
        self.dropout = dropout

        self.query_proj = nn.Linear(self.query_dim, self.hidden_dim)
        self.key_proj = nn.Linear(self.key_dim, self.hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * self.ffn_multiplier),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * self.ffn_multiplier, self.hidden_dim),
            nn.Dropout(self.dropout)
        )
        self.layer_norm1 = nn.LayerNorm(self.hidden_dim)
        self.layer_norm2 = nn.LayerNorm(self.hidden_dim)

    def forward(self, query_embeds, key_embeds):
        """
        Forward pass for the CrossAttention module.

        Args:
            query_embeds (torch.Tensor): Text embeddings of shape (batch_size, seq_len, query_dim).
            key_embeds (torch.Tensor): Image features of shape (batch_size, num_patches, key_dim).

        Returns:
            torch.Tensor: Output of the cross-attention layer.
        """
        query_proj = self.query_proj(query_embeds)
        key_proj = self.key_proj(key_embeds)

        # Multi-head Attention
        attn_output, _ = self.attention(
            query=query_proj,
            key=key_proj,
            value=key_proj
        )

        # Residual connection between attention output and query proj
        attn_output = self.layer_norm1(attn_output + query_proj)

        # feed forward
        ffn_output = self.ffn(attn_output)

        # residual connection between feed forward output and attention output
        output = self.layer_norm2(ffn_output + attn_output)

        return output
