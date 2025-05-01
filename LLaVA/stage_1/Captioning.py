import torch
import torch.nn as nn
from model.vision_encoder_model.EVA_Clip_02 import EVA02VisionTower
from model.language_model.LLaVA_SeaLLM import LLaVA_seaLLMs
from fussion_modules.Cross_Attention import CrossAttention


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

        target_ids = caption_inputs["input_ids"]  # [B, T_d]
        caption_embeds = self.llm.model.get_input_embeddings()(
            target_ids[:, :-1])  # [B, T_d-1, D]

        # 5. Prepare input embeddings
        inputs_embeds = torch.cat(
            [fused, caption_embeds], dim=1)  # [B, T_q + T_d -1, D]

        # 6. Construct attention_mask
        fused_mask = torch.ones((fused.size(0), fused.size(
            1)), dtype=torch.long, device=fused.device)
        caption_mask = caption_inputs["attention_mask"][:, 1:]  # [B, T_d-1]
        attention_mask = torch.cat(
            [fused_mask, caption_mask], dim=1)  # [B, T_q + T_d - 1]

        # 7. Mask prefix in loss
        labels = target_ids.clone()
        prefix_len = fused.size(1)
        labels[:, :prefix_len] = -100  # ignore prefix in loss

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
            fused = model.cross_attention(question_embeds, image_features)

            output_ids = model.llm.model.generate(
                inputs_embeds=fused,
                max_new_tokens=256,
                eos_token_id=model.llm.tokenizer.eos_token_id
            )
            return model.llm.tokenizer.decode(output_ids[0], skip_special_tokens=True)
