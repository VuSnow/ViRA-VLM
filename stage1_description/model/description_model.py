# Description Model
import torch
import torch.nn as nn
import logging
from easydict import EasyDict
from models.vision_encoder.eva_clip import EvaClip
from models.language_model.seallms import SeaLLMs
from models.attentions.deep_fusion import DeepFusion
from models.attentions.self_attention import SelfAttention
from transformers import PreTrainedModel, PretrainedConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from typing import Optional

logger = logging.getLogger(__name__)


class DescriptionModelConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, dict):
                v = EasyDict(v)
            setattr(self, k, v)
        super().__init__(**kwargs)


class DescriptionModel(PreTrainedModel, GenerationMixin):
    def __init__(self, config: DescriptionModelConfig):
        super().__init__(config)
        self.config = config
        # self.task_type = self.config.task_type
        # self.mode = self.config.mode

        # Initialize language model
        self.llm = SeaLLMs(config=self.config.language_model)

        # Initialize vision encoder
        self.vision = EvaClip(config=self.config.eva_vision_model)

        # Deep fusion
        self.deep_fusion = DeepFusion(
            n_layers=self.config.deep_fusion.n_layers,
            vision_dim=self.vision.dim,
            llm_dim=self.llm.dim,
            n_heads=self.config.deep_fusion.n_heads,
            ffn_multiplier=self.config.deep_fusion.ffn_multiplier,
            dropout=self.config.deep_fusion.dropout,
            add_positional=self.config.deep_fusion.add_positional,
        )

        # Vision projection
        self.vision_proj = nn.Linear(
            in_features=self.vision.dim,
            out_features=self.llm.dim,
        )

        # Self attention of concatenated embeddings
        self.self_attn = SelfAttention(
            query_dim=self.llm.dim,
            num_heads=self.config.self_attention.num_heads,
            ffn_multiplier=self.config.self_attention.ffn_multiplier,
            dropout=self.config.self_attention.dropout,
            add_positional=self.config.self_attention.add_positional,
            max_seq_len=self.llm.model.config.max_position_embeddings,
        )

        # Modality embedding for vision and language
        self.modality_embedding = nn.Embedding(
            num_embeddings=2,
            embedding_dim=self.llm.dim,
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.llm.dim)

    def forward(
        self,
        image_tensor: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        **kwargs,
    ):
        # 1. Get features embeddings from image tensor
        vision_features = self.vision(image_tensor)
        # 2. Create vision attention mask
        batch_size, num_patches, _ = vision_features.shape
        if attention_mask is not None:
            vision_mask = torch.ones(
                (batch_size, num_patches), dtype=attention_mask.dtype, device=attention_mask.device)
        else:
            vision_mask = None

        # 3. Get question embeddings from input_ids
        question_embeddings = self.llm.model.get_input_embeddings()(input_ids)

        # 4. Deep fusion dictionary output
        outputs = self.deep_fusion(
            vision_input=vision_features,
            llm_input=question_embeddings,
            vision_mask=vision_mask,
            llm_mask=attention_mask,
        )

        # 5. Project vision embeddings to language space
        outputs['vision_output'] = self.vision_proj(outputs['vision_output'])

        # 6. Add modality embeddings
        outputs['vision_output'] = outputs['vision_output'] + \
            self.modality_embedding(torch.tensor(
                [0], device=vision_features.device))
        outputs['llm_output'] = outputs['llm_output'] + \
            self.modality_embedding(torch.tensor(
                [1], device=question_embeddings.device))

        # 7. Concatenate vision and language embeddings, mask
        fused_embeddings = torch.cat(
            [outputs['vision_output'], outputs['llm_output']], dim=1)
        if attention_mask is not None:
            fused_mask = torch.cat([vision_mask, attention_mask], dim=1)
        else:
            fused_mask = None

        # 8. Self attention of concatenated embeddings
        fused_embeddings = self.self_attn(
            query=fused_embeddings,
            key_padding_mask=fused_mask,
        )
        # 9. Layer normalization
        fused_embeddings['output'] = self.layer_norm(
            fused_embeddings['output'])

        # 10. Get llm output
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            fused_emb=fused_embeddings['output'],
            fused_mask=fused_mask,
            labels=labels,
            return_dict=return_dict,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            **kwargs,
        )
        if outputs.logits is None:
            print("===> logits is None after self.llm() in DescriptionModel")

        if self.training:
            # Chế độ training: Chỉ cần loss. Trả về logits=None để tiết kiệm bộ nhớ.
            # Trainer sẽ chỉ lấy thuộc tính 'loss' từ đối tượng trả về này.
            # Các giá trị khác như hidden_states, attentions cũng có thể đặt là None
            # để tiết kiệm thêm một chút nếu chúng không cần thiết cho backward pass.
            return CausalLMOutputWithCrossAttentions(
                loss=outputs.loss,
                logits=None,
                past_key_values=None,
                hidden_states=None,
                attentions=None,
            )
        else:  # self.training is False (chế độ eval hoặc inference)
            # Chế độ eval/inference: Cần cả loss và logits.
            # Trainer sẽ lấy 'loss' để tính eval_loss và 'logits' để tính metrics.
            return CausalLMOutputWithCrossAttentions(
                loss=outputs.loss,
                logits=outputs.logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states if output_hidden_states else None,
                attentions=outputs.attentions if output_attentions else None,
            )
