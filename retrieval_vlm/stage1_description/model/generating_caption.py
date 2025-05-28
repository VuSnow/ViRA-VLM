from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import logging
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from retrieval_vlm.stage1_description.config.model_config import GeneratingCaptionConfig
from retrieval_vlm.model.vision_encoder_model.eva_clip import EvaClip
from retrieval_vlm.model.attention.cross_attention import CrossAttention
from retrieval_vlm.model.language_model.seallms import SeaLLMs
from transformers import PreTrainedModel, GenerationMixin

logger = logging.getLogger(__name__)

class GeneratingCaption(PreTrainedModel, GenerationMixin):
    def __init__(self, config: GeneratingCaptionConfig):
        super(GeneratingCaption, self).__init__(config)
        self.config = config

        self.llm = SeaLLMs(config.llm_config)
        self._dtype = self.llm.dtype

        self.vision_encoder = EvaClip(self.config.vision_encoder_config).to(self._dtype)
        for param in self.vision_encoder.parameters():
            param.requires_grad = self.config.vision_encoder_config.get("unfreeze", False)
        
        self.cross_attention = CrossAttention(
            query_dim=self.llm.dim,
            key_dim=self.vision_encoder.dim,
            hidden_dim=self.llm.dim,
            config=self.config.cross_attention_config,
            dtype=self._dtype
        )

        if self.config.lora and self.config.lora.get("r", 0) > 0:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r = self.config.lora.get("r", 0),
                lora_alpha = self.config.lora.get("lora_alpha", 1),
                lora_dropout = self.config.lora.get("lora_dropout", 0),
                target_modules = self.config.lora.get("target_modules", ["all"]),
                bias = "none",
                task_type = "CAUSAL_LM"
            )
            self.llm = get_peft_model(self.llm.model, lora_config)
            self.vision_encoder = get_peft_model(self.vision_encoder, lora_config)
            self.cross_attention = get_peft_model(self.cross_attention, lora_config)

    def forward(
        self, 
        input_ids: torch.Tensor, 
        image_tensor: torch.Tensor, 
        attention_mask: torch.Tensor, 
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        **kwargs
    ):
        # Chỉ giữ lại việc chuyển đổi dtype cho các tensor cần dtype cụ thể
        print("\n1. Converting input tensors to appropriate dtypes:")
        input_ids = input_ids.to(dtype=torch.long)
        image_tensor = image_tensor.to(dtype=self._dtype)
        attention_mask = attention_mask.to(dtype=torch.float32) if attention_mask is not None else None
        if labels is not None:
            labels = labels.to(dtype=torch.long)

        print("\n4. Processing Vision:")
        features_embedding = self.vision_encoder(image_tensor)
        features_embedding = features_embedding.to(dtype=self._dtype)

        print("\n5. Processing Question:")
        question_inputs = self.llm.model.get_input_embeddings()(input_ids)
        
        print("\n6. Performing Cross-Attention:")
        fused = self.cross_attention(
            query=question_inputs,
            key=features_embedding,
            value=features_embedding,
            need_weights=True
        )

        print("\n7. Performing final logits calculation:")
        logits = self.llm.model.lm_head(fused)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        print(f"\n8. Loss: {loss}")
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            # past_key_values=None,
            hidden_states=fused if output_hidden_states else None,
            attentions=self.cross_attention.attention_weights if output_attentions else None
        )
    
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_tensor: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if attention_mask is not None:
                attention_mask = attention_mask[:, -1].unsqueeze(-1)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_tensor": image_tensor,
            "past_key_values": past_key_values,
            **kwargs
        }
    
    def get_encoder(self):
        """Return vision encoder"""
        return self.vision_encoder

    def get_decoder(self):
        """Return language model decoder"""
        return self.llm

    def _reorder_cache(self, past_key_values, beam_idx):
        """
        Reorder cache for beam search
        Args:
            past_key_values: Tuple of key-value pairs from previous layers
            beam_idx: Index of the selected beam
        Returns:
            New tuple of past_key_values reordered
        """
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past

    def _prepare_attention_mask_for_generation(
        self,
        input_ids: torch.Tensor,
        pad_token_id: int,
        eos_token_id: int,
    ) -> torch.LongTensor:
        """
        Prepare attention mask for generation
        Args:
            input_ids: Input token ids
            pad_token_id: ID of padding token
            eos_token_id: ID of end-of-sequence token
        Returns:
            Attention mask tensor
        """
        attention_mask = torch.ones_like(input_ids)
        if pad_token_id is None:
            raise ValueError("pad_token_id must be defined in config")
        
        # Mark padding positions
        attention_mask = attention_mask.masked_fill(input_ids == pad_token_id, 0)
        
        # Mark positions after EOS token
        if eos_token_id is not None:
            eos_mask = input_ids == eos_token_id
            eos_mask = torch.cumsum(eos_mask, dim=-1) > 0
            attention_mask = attention_mask.masked_fill(eos_mask, 0)
        
        return attention_mask

    def _prepare_model_inputs(
        self,
        inputs: Dict[str, torch.Tensor],
        bos_token_id: Optional[int] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for model
        Args:
            inputs: Dictionary containing input tensors
            bos_token_id: ID of beginning-of-sequence token
            model_kwargs: Additional parameters for model
        Returns:
            Dictionary containing prepared inputs
        """
        if model_kwargs is None:
            model_kwargs = {}
        
        # Add BOS token if needed
        if bos_token_id is not None and "input_ids" in inputs:
            input_ids = inputs["input_ids"]
            if input_ids.shape[1] == 0:
                input_ids = torch.full(
                    (input_ids.shape[0], 1),
                    bos_token_id,
                    dtype=torch.long,
                    device=input_ids.device,
                )
                inputs["input_ids"] = input_ids
        
        # Update attention mask
        if "attention_mask" in inputs:
            attention_mask = inputs["attention_mask"]
            if attention_mask.shape[1] == 0:
                attention_mask = torch.ones_like(inputs["input_ids"])
                inputs["attention_mask"] = attention_mask
        
        return inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: CausalLMOutputWithCrossAttentions,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
    ) -> Dict[str, Any]:
        """
        Update model_kwargs for generation
        Args:
            outputs: Output from forward pass
            model_kwargs: Dictionary containing model parameters
            is_encoder_decoder: Flag indicating if model is encoder-decoder
        Returns:
            Dictionary model_kwargs updated
        """
        # Update past_key_values
        if "past_key_values" in outputs:
            model_kwargs["past_key_values"] = outputs.past_key_values
        
        # Update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            if attention_mask is not None and attention_mask.shape[1] == 1:
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, torch.ones_like(attention_mask)], dim=-1
                )
        
        return model_kwargs