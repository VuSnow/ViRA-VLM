import torch
import torch.nn as nn
import logging
from transformers import AutoModelForCausalLM, AutoConfig, AutoModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from easydict import EasyDict
from models.language_model.qwen2_decoder_layer_with_attn import Qwen2DecoderLayerWithCrossAttn
from models.attentions.cross_attention import CrossAttention
logger = logging.getLogger(__name__)


class SeaLLMs(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        base_llm_config = AutoConfig.from_pretrained(
            self.config['name'],
            output_hidden_states=True,
            output_attentions=True,
        )
        # base_llm_config.tie_word_embeddings = True

        # Load the model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['name'],
            config=base_llm_config,
            ignore_mismatched_sizes=True,
        )
        self.model.tie_weights()

        print("Loaded AutoModelForCausalLM:")
        print(self.model)

        # Replace the default decoder layer with the custom one
        inject_layers = self.config['inject_layers']
        num_layers = self.model.config.num_hidden_layers
        if num_layers < 2 * inject_layers:
            raise ValueError(
                f"Number of layers ({num_layers}) must be at least 2 * inject_layers ({2 * inject_layers})")
        print(
            f"--- Inject cross-attention in first {inject_layers} layers ---")
        first_inject_layers = list(range(0, inject_layers))
        for i in first_inject_layers:
            cross_attn_module_first = CrossAttention(
                query_dim=self.model.config.hidden_size,
                kv_dim=self.model.config.hidden_size,
                hidden_dim=self.model.config.hidden_size,
                num_heads=self.config['num_heads'],
                ffn_multiplier=self.config['ffn_multiplier'],
                dropout=self.config['dropout'],
                add_positional=self.config['add_positional'],
                max_seq_len=self.model.config.max_position_embeddings,
            )
            self.model.model.layers[i] = Qwen2DecoderLayerWithCrossAttn(
                config=self.model.config,
                layer_idx=i,
                cross_attn_module=cross_attn_module_first,
            )

        print(f"--- Inject cross-attention in last {inject_layers} layers ---")
        last_inject_layers = list(
            range(num_layers - inject_layers, num_layers))
        for i in last_inject_layers:
            cross_attn_module_last = CrossAttention(
                query_dim=self.model.config.hidden_size,
                kv_dim=self.model.config.hidden_size,
                hidden_dim=self.model.config.hidden_size,
                num_heads=self.config['num_heads'],
                ffn_multiplier=self.config['ffn_multiplier'],
                dropout=self.config['dropout'],
                add_positional=self.config['add_positional'],
                max_seq_len=self.model.config.max_position_embeddings,
            )
            self.model.model.layers[i] = Qwen2DecoderLayerWithCrossAttn(
                config=self.model.config,
                layer_idx=i,
                cross_attn_module=cross_attn_module_last,
            )

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @property
    def dim(self) -> int:
        """
            Get the dimension of the output tensor.
        """
        return self.model.model.config.hidden_size

    @property
    def dtype(self) -> torch.dtype:
        return self.model.dtype

    @property
    def device(self) -> torch.device:
        return self.model.device
