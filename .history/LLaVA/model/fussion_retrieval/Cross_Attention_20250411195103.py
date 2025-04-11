import torch
from torch import nn


class CrossAttention(nn.Module):
    def __init__(self, vision_dim=1792, text_dim=768, hidden_dim=768, num_heads=8, ffn_multiplier=4, dropout=0.1):
        """
        Initialize the CrossAttention module.  
        """
        super(CrossAttention, self).__init__()
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.ffn_multiplier = ffn_multiplier
        self.dropout = dropout

        self.vision_proj = nn.Linear(self.vision_dim, self.hidden_dim)
        self.text_proj = nn.Linear(self.text_dim, self.hidden_dim)
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

    def forward(self, text_embeds, image_feats):
        """
        Forward pass for the CrossAttention module.

        Args:
            text_embeds (torch.Tensor): Text embeddings of shape (batch_size, seq_len, text_dim).
            image_feats (torch.Tensor): Image features of shape (batch_size, num_patches, vision_dim).

        Returns:
            torch.Tensor: Output of the cross-attention layer.
        """
        # Project text and image features to the same dimension
        # # transform [B, 768] -> [B, T, 768]
        text_proj = self.text_proj(
            text_embeds
        ).squeeze(1)
        image_proj = self.vision_proj(
            image_feats)              # [B, 256, 1792]

        # Cross-attention
        attn_output, _ = self.attention(
            query=text_proj,
            key=image_proj,
            value=image_proj
        )

        # Residual connection and layer normalization
        attn_output = self.layer_norm1(attn_output + text_proj)

        # Feed-forward network
        ffn_output = self.ffn(attn_output)

        # Residual connection and layer normalization
        output = self.layer_norm2(ffn_output + attn_output)

        return output.squeeze(1)  # [B, 768] to retrive the text embedding
