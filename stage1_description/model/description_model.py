# Description Model
import torch
import torch.nn as nn
import logging
from easydict import EasyDict
from models.attentions.cross_attention import CrossAttention
from models.vision_encoder.eva_clip import EvaClip
from models.language_model.seallms import SeaLLMs
from models.attentions.deep_fusion import DeepFusion
from models.attentions.self_attention import SelfAttention
from models.language_model.qwen2_decoder_layer_with_attn import Qwen2DecoderLayerWithCrossAttn
from transformers import PreTrainedModel, PretrainedConfig, GenerationMixin, AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, CausalLMOutputWithPast
from typing import Optional, Tuple, Dict, Any
from collections import OrderedDict

logger = logging.getLogger(__name__)


class DescriptionModelConfig(PretrainedConfig):
    model_type = "description_model"

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, dict):
                v = EasyDict(v)
            setattr(self, k, v)
        super().__init__(**kwargs)


class DescriptionModel(PreTrainedModel, GenerationMixin):
    config_class = DescriptionModelConfig

    def __init__(self, config: DescriptionModelConfig):
        super().__init__(config)

        print(f"--- Initializing DescriptionModel in __init__ ---")
        self.config = config

        # Initialize language model
        print(f"--- Initializing language model in __init__ ---")
        llm_config_params = self.config.language_model
        base_llm_config = AutoConfig.from_pretrained(
            llm_config_params['name'],
        )
        if hasattr(base_llm_config, 'vocab_size'):
            base_llm_config.vocab_size = self.config.vocab_size
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_config_params['name'],
            config=base_llm_config,
            ignore_mismatched_sizes=True
        )
        self.llm.tie_weights()
        inject_layers = llm_config_params['inject_layers']
        num_layers = self.llm.config.num_hidden_layers
        if num_layers < 2 * inject_layers:
            raise ValueError(
                f"Number of layers ({num_layers}) must be at least 2 * inject_layers ({2 * inject_layers})")
        print(
            f"--- Inject cross-attention in first {inject_layers} layers ---")
        first_inject_layers = list(range(0, inject_layers))
        for i in first_inject_layers:
            cross_attn_module_first = CrossAttention(
                query_dim=self.llm.config.hidden_size,
                kv_dim=self.llm.config.hidden_size,
                hidden_dim=self.llm.config.hidden_size,
                num_heads=llm_config_params['num_heads'],
                ffn_multiplier=llm_config_params['ffn_multiplier'],
                dropout=llm_config_params['dropout'],
                add_positional=llm_config_params['add_positional'],
                max_seq_len=self.llm.config.max_position_embeddings,
            )
            self.llm.model.layers[i] = Qwen2DecoderLayerWithCrossAttn(
                config=self.llm.config,
                layer_idx=i,
                cross_attn_module=cross_attn_module_first,
            )

        print(f"--- Inject cross-attention in last {inject_layers} layers ---")
        last_inject_layers = list(
            range(num_layers - inject_layers, num_layers))
        for i in last_inject_layers:
            cross_attn_module_last = CrossAttention(
                query_dim=self.llm.config.hidden_size,
                kv_dim=self.llm.config.hidden_size,
                hidden_dim=self.llm.config.hidden_size,
                num_heads=llm_config_params['num_heads'],
                ffn_multiplier=llm_config_params['ffn_multiplier'],
                dropout=llm_config_params['dropout'],
                add_positional=llm_config_params['add_positional'],
                max_seq_len=self.llm.config.max_position_embeddings,
            )
            self.llm.model.layers[i] = Qwen2DecoderLayerWithCrossAttn(
                config=self.llm.config,
                layer_idx=i,
                cross_attn_module=cross_attn_module_last,
            )

        print(f"--- Initializing vision encoder ---")
        self.vision = EvaClip(config=self.config.eva_vision_model)

        print(f"--- Initializing vision projection ---")
        self.vision_proj = nn.Linear(
            in_features=self.vision.dim,
            out_features=self.llm.config.hidden_size,
        )

        print(f"--- Initializing modality embedding for vision and language ---")
        self.modality_embedding = nn.Embedding(
            num_embeddings=2,
            embedding_dim=self.llm.config.hidden_size,
        )

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

        if isinstance(module, PreTrainedModel):
            pass

    def post_init(self):
        self.llm.tie_weights()

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.llm.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.llm.set_output_embeddings(new_embeddings)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_tensor: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        past_key_values: Optional[list] = None,
        return_dict: Optional[bool] = True,
        **kwargs,
    ):
        if image_tensor is not None:
            vision_features = self.vision(image_tensor)
            assert vision_features.requires_grad, "vision_features does not require grad"
            vision_emb = self.vision_proj(vision_features)
            assert vision_emb.requires_grad, "vision_emb does not require grad"
            batch_size, num_patches, _ = vision_emb.shape
            # mod_emb = self.modality_embedding(
            #     torch.zeros((batch_size, num_patches), dtype=torch.long, device=vision_emb.device)
            # )
            # vision_emb = vision_emb + mod_emb
            vision_mask = torch.zeros(
                (batch_size, num_patches),
                dtype=torch.bool,
                device=vision_emb.device,
            )
        else:
            vision_emb = None
            vision_mask = None

        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=use_cache,
            past_key_values=past_key_values,
            vision_emb=vision_emb,
            vision_mask=vision_mask,
        )
        if self.training:
            return CausalLMOutputWithPast(loss=outputs.loss)

        return outputs

    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        model_kwargs["past_key_values"] = outputs.past_key_values

        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        if model_kwargs.get("image_tensor", None) is not None:
            model_kwargs["image_tensor"] = model_kwargs["image_tensor"]

        return model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor,
                                              torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:

        if past_key_values:
            input_ids = input_ids[:, -1:]

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
            "image_tensor": kwargs.get("image_tensor", None),
        }
        return model_inputs

    def save_pretrained(
        self,
        save_directory: str,
        state_dict: Optional[dict] = None,
        **kwargs,
    ):
        if state_dict is None:
            state_dict = self.state_dict()
        super().save_pretrained(save_directory, state_dict=state_dict, **kwargs)

    @property
    def llm_dim(self):
        return self.llm.config.hidden_size
