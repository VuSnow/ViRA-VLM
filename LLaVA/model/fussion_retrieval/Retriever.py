import torch
from torch import nn
from LLaVA.model.embedding_model import SentenceEmbeddingRetrieval
from LLaVA.model.fussion_retrieval import CrossAttention
from LLaVA.model.vision_encoder_model import EVA02VisionTower


class Retriever(nn.Module):
    def __init__(self,
                 vision_encoder=None,
                 text_encoder=None,
                 cross_attention=None,
                 hidden_dim=768,
                 embedding_path=None,
                 metadata_path=None,
                 top_k=5
                 ):
        super(Retriever, self).__init__()
        self.vision_model = vision_encoder if vision_encoder else EVA02VisionTower()
        self.text_encoder = text_encoder if text_encoder else SentenceEmbeddingRetrieval()
    # def forward(self, questions, images):
    #     """
    #     Forward pass for the Retriever module.

    #     Args:
    #         text_embeds (torch.Tensor): Text embeddings of shape (batch_size, seq_len, text_dim).
    #         image_feats (torch.Tensor): Image features of shape (batch_size, num_patches, vision_dim).

    #     Returns:
    #         torch.Tensor: Output of the cross-attention layer.
    #     """
    #     # encode image
    #     # (batch_size, 1792, 256) - (batch_size, vision_dim, num_patches)
    #     image_feats = self.vision_model(images)
    #     # (batch_size, 256, 1792) - (batch_size, , num_patches, vision_dim)
    #     image_feats = image_feats.transpose(1, 2)

    #     # encode text
    #     # (batch_size, 768) - (batch_size, text_dim)
    #     text_embeds = self.text_encoder(questions)
    #     # (batch_size, 1, 768) - (batch_size, seq_len, text_dim)
    #     text_embeds = text_embeds.unsqueeze(1)

    #     fused = self.cross_attention(
    #         text_embeds=text_embeds.to(self.vision_model.device),
    #         image_feats=image_feats.to(self.vision_model.device)
    #     )

    #     return fused.squeeze(1)

    # def retrieve(self, questions, images, top_k=5):
    #     # Encode the questions and images
    #     fused = self.forward(questions, images)
