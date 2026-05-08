import torch.nn as nn
import torch
import logging
from typing import Optional, Tuple
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

logger = logging.getLogger(__name__)


class Qwen2DecoderLayerWithCrossAttn(Qwen2DecoderLayer):
    def __init__(self, config, layer_idx, cross_attn_module):
        super().__init__(config, layer_idx)
        self.cross_attn_module = cross_attn_module
        self.cross_attn_ln = nn.LayerNorm(config.hidden_size)
        self.cross_attn_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.rms_norm_eps)
        dropout = getattr(config, 'attention_dropout', 0.1)
        self.cross_attn_dropout = nn.Dropout(p=dropout)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor,
                                            torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        vision_emb = kwargs.get("vision_emb", None)
        vision_mask = kwargs.get("vision_mask", None)

        # 1. Self-Attention Block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        self_attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

        hidden_states = residual + self_attn_outputs[0]
        other_outputs_from_self_attn = self_attn_outputs[1:]

        # 2. Cross-Attention Block
        if vision_emb is not None:
            if vision_emb.dtype != hidden_states.dtype:
                vision_emb = vision_emb.to(hidden_states.dtype)
            residual = hidden_states
            hidden_states = self.cross_attn_layernorm(hidden_states)

            cross_attn_out, _ = self.cross_attn_module(
                query=hidden_states,
                key_value=vision_emb,
                kv_mask=vision_mask,
                need_weights=False,
            )
            if torch.isnan(cross_attn_out).any() or torch.isinf(cross_attn_out).any():
                logger.warning(
                    f"Cross-Attention output is nan/inf at layer {self.layer_idx}, "
                    f"vision_emb: {vision_emb.shape}, hidden: {hidden_states.shape}")
                raise ValueError("Cross-Attention output is nan")
            hidden_states = residual + self.cross_attn_dropout(cross_attn_out)

        # 3. MLP Block (Feed-Forward Network)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return (hidden_states,) + other_outputs_from_self_attn
