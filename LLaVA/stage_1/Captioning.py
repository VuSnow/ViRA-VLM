import torch
import torch.nn as nn
from LLaVA.model.vision_encoder_model.EVA_Clip_02 import EVA02VisionTower
from LLaVA.model.language_model.LLaVA_SeaLLM import LLaVA_seaLLMs
from LLaVA.fussion_modules.Cross_Attention import CrossAttention


class CaptionGenerating(nn.Module):
    def __init__(self, vision_encoder, cross_attention, llm):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.cross_attention = cross_attention
        self.llm = llm

    def forward(self, image_tensor, questions, descriptions):
        """
        Args:
            image_tensor (torch.Tensor): [B, 3, H, W] - raw images
            questions (List[str]): e.g., ["Mô tả chi tiết bức ảnh:"]
            descriptions (List[str]): ground truth caption
        Returns:
            torch.Tensor: loss
        """
        # 1. Extract image features
        image_features = self.vision_encoder(image_tensor)  # [B, 256, 1792]

        # 2. Encode questions
        # includes input_ids + attention_mask
        question_inputs = self.llm.tokenize(questions)
        with torch.no_grad():
            question_embeds = self.llm.embeddings(
                question_inputs)  # [B, T_q, D]
        question_embeds = question_embeds.to(
            dtype=next(self.llm.model.parameters()).dtype)

        # 3. Fuse with image
        fused = self.cross_attention(
            question_embeds, image_features)  # [B, T_q, D]

        # 4. Tokenize target captions
        caption_inputs = self.llm.tokenizer(
            descriptions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.llm.max_seq_length
        ).to(fused.device)

        target_ids = caption_inputs["input_ids"].to(
            dtype=torch.long, device=fused.device)

        # input: fused + caption[:, :-1]
        caption_input_ids = target_ids[:, :-1]
        caption_embeds = self.llm.model.get_input_embeddings()(
            caption_input_ids
        ).to(dtype=next(self.llm.model.parameters()).dtype)

        inputs_embeds = torch.cat([fused, caption_embeds], dim=1)

        # label: -100 for prefix + caption[:, 1:]
        caption_target_ids = target_ids[:, 1:]
        labels = torch.cat([
            torch.full((target_ids.size(0), fused.size(1)), -100,
                       dtype=torch.long, device=fused.device),
            caption_target_ids
        ], dim=1)  # ignore prefix in loss

        # attention_mask
        fused_mask = torch.ones((fused.size(0), fused.size(
            1)), dtype=torch.long, device=fused.device)
        caption_mask = caption_inputs["attention_mask"][:, 1:]
        attention_mask = torch.cat([fused_mask, caption_mask], dim=1)
        attention_mask = attention_mask.to(dtype=inputs_embeds.dtype)

        # 8. Forward pass through LLM
        outputs = self.llm.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )

        return outputs.loss

    def generate_caption(model, image_tensor, prompt="Mô tả chi tiết bức ảnh:"):
        model.eval()
        with torch.no_grad():
            image_tensor = image_tensor.unsqueeze(
                0).to(next(model.parameters()).device)
            image_features = model.vision_encoder(image_tensor)
            question_inputs = model.llm.tokenize(prompt)
            question_embeds = model.llm.embeddings(question_inputs)
            question_embeds = question_embeds.to(dtype=image_features.dtype)
            fused = model.cross_attention(question_embeds, image_features)
            # Bắt buộc thêm dòng này
            fused = fused.to(dtype=next(model.llm.model.parameters()).dtype)

            output_ids = model.llm.model.generate(
                inputs_embeds=fused,
                max_new_tokens=256,
                eos_token_id=model.llm.tokenizer.eos_token_id
            )
            return model.llm.tokenizer.decode(output_ids[0], skip_special_tokens=True)
