import torch
import torch.nn as nn
import logging
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from easydict import EasyDict
from models.language_model.qwen2_decoder_layer_with_attn import Qwen2DecoderLayerWithCrossAttn
from models.attentions.cross_attention import CrossAttention
logger = logging.getLogger(__name__)


class SeaLLMs(nn.Module):
    def __init__(self, config: EasyDict):
        super().__init__()
        self.config = config

        # Decide device map based on available hardware
        self.device_map = self._decide_device_map()

        # Set compute dtype
        try:
            self.compute_dtype = torch.bfloat16
            _ = torch.zeros(1, dtype=torch.bfloat16)
            logger.info("Using bfloat16 for LLM")
        except RuntimeError:
            self.compute_dtype = torch.float32
            logger.warning(
                "bfloat16 not supported, falling back to float32. This might be less stable.")

        # Load the model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.name,
            device_map=self.device_map,
            torch_dtype=self.compute_dtype,
            output_hidden_states=True,
            output_attentions=True,
        )

        # Replace the default decoder layer with the custom one
        inject_layers = self.config.inject_layers

        num_layers = self.model.model.config.num_hidden_layers
        inject_layers = list(range(num_layers - inject_layers, num_layers))
        for i in inject_layers:
            cross_attn_module = CrossAttention(
                query_dim=self.model.model.config.hidden_size,
                kv_dim=self.model.model.config.hidden_size,
                hidden_dim=self.model.model.config.hidden_size,
                num_heads=self.config.num_heads,
                ffn_multiplier=self.config.ffn_multiplier,
                dropout=self.config.dropout,
                add_positional=self.config.add_positional,
                max_seq_len=self.model.model.config.max_position_embeddings,
            )
            self.model.model.layers[i] = Qwen2DecoderLayerWithCrossAttn(
                config=self.model.model.config,
                layer_idx=i,
                cross_attn_module=cross_attn_module,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        fused_emb: torch.Tensor = None,
        fused_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        **kwargs
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            fused_emb=fused_emb,
            fused_mask=fused_mask,
            labels=labels,
            **kwargs,
        )

        # Trường hợp outputs là tuple (thường do custom decoder layer)
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
            logits = self.model.lm_head(hidden_states)
            return CausalLMOutputWithCrossAttentions(
                loss=None,
                logits=logits,
                hidden_states=None,
                attentions=None,
                cross_attentions=None,
            )

        # Trường hợp outputs là dict hoặc CausalLMOutput
        logits = outputs.logits
        if logits is None:
            if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                hidden_states = outputs.hidden_states[-1]  # Lấy layer cuối
            else:
                raise ValueError(
                    "Cannot compute logits: missing hidden_states.")
            logits = self.model.lm_head(hidden_states)

        return CausalLMOutputWithCrossAttentions(
            loss=outputs.loss if hasattr(outputs, "loss") else None,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(
                outputs, "hidden_states") else None,
            attentions=outputs.attentions if hasattr(
                outputs, "attentions") else None,
            cross_attentions=None,
        )

    def _decide_device_map(self):
        """Decide device map based on available hardware"""
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count >= 2:
                logger.info(
                    f"Found {gpu_count} GPUs. Using 'balanced' device map.")
                return "balanced"
            else:
                logger.info("Found 1 GPU. Using 'auto' device map.")
                return "auto"
        else:
            logger.info("No CUDA GPUs found. Using 'cpu' device map.")
            return "cpu"

    @property
    def dim(self) -> int:
        """
            Get the dimension of the output tensor.
        """
        return self.model.config.hidden_size

    @property
    def dtype(self) -> torch.dtype:
        return self.model.dtype

    @property
    def device(self) -> torch.device:
        return self.model.device
