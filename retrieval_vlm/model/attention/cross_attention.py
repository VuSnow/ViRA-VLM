import torch
from torch import nn
from easydict import EasyDict
from typing import Optional


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        hidden_dim: int,
        config: EasyDict = None,
        dtype: torch.dtype = torch.float32
    ):
        super(CrossAttention, self).__init__()
        self._validate_config(config)
        self.config = config
        self.dtype = dtype
        
        self.hidden_dim = hidden_dim

        # Projection layers
        self.query_proj = nn.Linear(query_dim, hidden_dim).to(dtype=self.dtype)
        self.key_proj = nn.Linear(key_dim, hidden_dim).to(dtype=self.dtype)
        self.value_proj = nn.Linear(key_dim, hidden_dim).to(dtype=self.dtype)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
            batch_first=True
        ).to(dtype=self.dtype)

        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_dim).to(dtype=self.dtype)
        self.layer_norm2 = nn.LayerNorm(hidden_dim).to(dtype=self.dtype)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * self.config.ffn_multiplier),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden_dim * self.config.ffn_multiplier, hidden_dim),
            nn.Dropout(self.config.dropout)
        ).to(dtype=self.dtype)

        # Initialize last attention weights
        self.last_attn_weights: Optional[torch.Tensor] = None
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
        attn_mask: torch.Tensor = None,
        need_weights: bool = False
    ):   
        query = query.to(dtype=self.dtype)
        key = key.to(dtype=self.dtype)
        if value is not None:
            value = value.to(dtype=self.dtype)

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.to(dtype=torch.bool)
        if attn_mask is not None:
            attn_mask = attn_mask.to(dtype=torch.bool)
    
        query_proj = self.query_proj(query)
        key_proj = self.key_proj(key)
        value_proj = self.value_proj(key)
    
        attn_output, attn_weights = self.attention(
            query=query_proj,
            key=key_proj,
            value=value_proj,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=need_weights
        )
        attn_output = self.layer_norm1(attn_output + query_proj)
        ffn_output = self.ffn(attn_output)
        output = self.layer_norm2(ffn_output + attn_output)
        self.last_attn_weights = attn_weights
    
        if need_weights:
            return output, attn_weights
        return output


    def _check_input_types(self, query, key, value, key_padding_mask, attn_mask):
        """Helper method to check input types"""
        if not isinstance(query, torch.Tensor):
            raise TypeError(f"Expected query to be a Tensor, got {type(query)}")
        
        if not isinstance(key, torch.Tensor):
            raise TypeError(f"Expected key to be a Tensor, got {type(key)}")
        
        if value is not None and not isinstance(value, torch.Tensor):
            raise TypeError(f"Expected value to be a Tensor, got {type(value)}")
        
        if key_padding_mask is not None and not isinstance(key_padding_mask, torch.Tensor):
            raise TypeError(f"Expected key_padding_mask to be a Tensor, got {type(key_padding_mask)}")
        
        if attn_mask is not None and not isinstance(attn_mask, torch.Tensor):
            raise TypeError(f"Expected attn_mask to be a Tensor, got {type(attn_mask)}")
        
    def _validate_config(self, config) -> None:
        """Validate the configuration"""
        required_fields = ['num_heads', 'ffn_multiplier', 'dropout']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Config must contain {field}")
    
    @property
    def attention_weights(self):
        """Get the last computed attention weights"""
        return self.last_attn_weights

    @property
    def dim(self):
        return self.hidden_dim