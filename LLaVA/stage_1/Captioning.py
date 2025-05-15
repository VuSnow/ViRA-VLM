import torch
import torch.nn as nn
import logging
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
logger = logging.getLogger(__name__)


class CaptionGenerating(nn.Module):
    def __init__(self, vision_encoder, cross_attention, llm):
        super(CaptionGenerating, self).__init__()
        self.vision_encoder = vision_encoder
        self.cross_attention = cross_attention
        self.llm = llm
        self._device = next(self.llm.model.parameters()).device

    def forward(self, image_tensor, questions, descriptions):
        target_dtype = self.llm.dtype
        image_tensor = image_tensor.to(device=self._device, dtype=target_dtype)

        image_features = self.vision_encoder(image_tensor)  # [B, 256, 1792]

        if not torch.isfinite(image_features).all():
            logger.error(
                "[DEBUG] image_features has NaN or Inf after vision_encoder")
            return torch.tensor(0.0, device=image_tensor._device, requires_grad=True)
        image_features = image_features.to(dtype=target_dtype)

        question_inputs = self.llm.tokenize(questions)
        with torch.no_grad():
            question_embeds = self.llm.embeddings(
                question_inputs)  # [B, T_q, D]

        if not torch.isfinite(question_embeds).all():
            logger.error(
                "[DEBUG] question_embeds has NaN or Inf after llm.embeddings")
            return torch.tensor(0.0, device=image_tensor.device, requires_grad=True)

        fused = self.cross_attention(
            # [B, T_q, D]
            question_embeds.to(dtype=target_dtype),
            image_features.to(dtype=target_dtype)
        )
        if not torch.isfinite(fused).all():
            logger.error("[DEBUG] fused has NaN or Inf after cross_attention")
            return torch.tensor(0.0, device=image_tensor.device, requires_grad=True)
        fused = fused.to(dtype=target_dtype)

        caption_inputs = self.llm.tokenizer(
            descriptions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.llm.max_seq_length
        )
        caption_inputs = {k: v.to(self._device)
                          for k, v in caption_inputs.items()}

        target_ids = caption_inputs["input_ids"].to(
            dtype=torch.long)

        if (target_ids != self.llm.tokenizer.pad_token_id).sum().item() == 0:
            logger.warning(
                "[WARNING] All padding in captions after tokenization, skipping batch")
            return torch.tensor(0.0, device=image_tensor.device, requires_grad=True)

        caption_input_ids = target_ids[:, :-1]
        caption_target_ids = target_ids[:, 1:]

        caption_embeds = self.llm.model.get_input_embeddings()(caption_input_ids)
        caption_embeds = caption_embeds.to(dtype=target_dtype)

        if not torch.isfinite(caption_embeds).all():
            print("[DEBUG] caption_embeds has NaN or Inf")
            return torch.tensor(0.0, device=fused.device, requires_grad=True)

        inputs_embeds = torch.cat([fused, caption_embeds], dim=1)
        if not torch.isfinite(inputs_embeds).all():
            logger.error(
                "[WARNING] inputs_embeds has NaN or Inf after concatenation")
            return torch.tensor(0.0, device=inputs_embeds.device, requires_grad=True)

        labels = torch.cat([
            torch.full((target_ids.size(0), fused.size(1)), -100,
                       dtype=torch.long, device=self._device),
            caption_target_ids
        ], dim=1)

        fused_mask = torch.ones((fused.size(0), fused.size(
            1)), dtype=torch.long, device=self._device)
        caption_mask = caption_inputs["attention_mask"][:, 1:].to(
            dtype=torch.long, device=self._device)
        attention_mask = torch.cat(
            [fused_mask, caption_mask], dim=1).to(dtype=torch.long)

        outputs = self.llm.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

        loss = outputs.loss
        logits = outputs.logits

        if loss is None:
            logger.error(
                "[ERROR] Loss calculation failed, model output did not contain loss.")
            return torch.tensor(0.0, device=self._device, requires_grad=True)

        if not torch.isfinite(loss):
            logger.error(
                f"[ERROR] Calculated loss is NaN or Inf: {loss.item()}")
            return torch.tensor(0.0, device=self._device, requires_grad=True)

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

    def generate_caption(self, image_tensor, prompt="Miêu tả chi tiết bức ảnh này từ hành động, vật thể cho đến cảm giác hoặc bối cảnh. Bao gồm mọi yếu tố như màu sắc, ánh sáng, bố cục, các đối tượng và sự tương tác giữa chúng, cảm xúc hoặc thông điệp mà bức ảnh truyền tải. Hãy chú ý đến cả các yếu tố nhỏ mà có thể mang lại chiều sâu cho bức ảnh."):
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():
            # Get device where the module is
            primary_device = next(self.parameters()).device
            llm_dtype = self.llm.dtype  # Get compute dtype (e.g., bfloat16)

            image_tensor = image_tensor.unsqueeze(0).to(
                device=primary_device, dtype=llm_dtype)
            image_features = self.vision_encoder(
                image_tensor)  # [1, 256, K_dim]
            image_features = image_features.to(dtype=llm_dtype)  # Ensure dtype

            if not torch.isfinite(image_features).all():
                logger.error("[Generate] image_features has NaN/Inf")
                return "Error generating caption due to vision encoder issues."

            question_inputs = self.llm.tokenize(prompt)
            question_embeds = self.llm.embeddings(
                question_inputs)  # [1, T_q, D]
            question_embeds = question_embeds.to(
                dtype=llm_dtype)  # Ensure dtype

            if not torch.isfinite(question_embeds).all():
                logger.error("[Generate] question_embeds has NaN/Inf")
                return "Error generating caption due to LLM embedding issues."

            fused = self.cross_attention(
                question_embeds, image_features)  # [1, T_q, D]
            fused = fused.to(dtype=llm_dtype)

            if not torch.isfinite(fused).all():
                logger.error("[Generate] fused has NaN/Inf")
                return "Error generating caption due to cross-attention issues."

            model_generate_kwargs = {
                "inputs_embeds": fused,
                "max_new_tokens": 512,
                "eos_token_id": self.llm.tokenizer.eos_token_id,
                "pad_token_id": self.llm.tokenizer.pad_token_id  # Also set pad token id
            }

            output_ids = self.llm.model.generate(**model_generate_kwargs)
            return self.llm.tokenizer.decode(output_ids[0], skip_special_tokens=True)
