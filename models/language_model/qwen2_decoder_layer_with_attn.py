from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
import torch.nn as nn


class Qwen2DecoderLayerWithCrossAttn(Qwen2DecoderLayer):
    def __init__(self, config, layer_idx, cross_attn_module):
        super().__init__(config, layer_idx)
        self.cross_attn_module = cross_attn_module
        self.cross_attn_ln = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        fused_emb=None,
        fused_mask=None,
        **kwargs
    ):
        # Forward chuẩn (self-attn, FFN, ...); trả tuple
        outputs = super().forward(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs
        )
        hidden_states = outputs[0]

        if fused_emb is not None:
            cross_attn_out = self.cross_attn_module(
                query=hidden_states,
                key_value=fused_emb,
                kv_mask=fused_mask,
            )
            if isinstance(cross_attn_out, dict):
                cross_attn_out = cross_attn_out["output"]
            hidden_states = hidden_states + cross_attn_out
            hidden_states = self.cross_attn_ln(hidden_states)

        return (hidden_states,) + outputs[1:]
