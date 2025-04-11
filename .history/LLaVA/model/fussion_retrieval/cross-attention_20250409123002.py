import torch
from torch import nn


class RetrievalCrossAttention(nn.Module):
    def __init__(self, vision_dim=1792, text_dim=768, hidden_dim=1024, num_heads=8, ffn_multiplier=4, dropout=0.1):
        """
        Initialize the RetrievalCrossAttention module.  
        """
        super(RetrievalCrossAttention, self).__init__()
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
