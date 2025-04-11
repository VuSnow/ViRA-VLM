import torch
from torch import nn
from LLaVA.model.embedding_model import EmbeddingModel
from LLaVA.model.fussion_retrieval import RetrievalCrossAttention
from LLaVA.model.vision_encoder_model import EVA02VisionTower


class Retriever(nn.Module):
    def __init__(self,
                 vision_encoder=None,
                 text_encoder=None,
                 hidden_dim=1024,
                 num_heads=8,
                 num_layers=2,
                 dropout=0.1,
                 use_vision=True
                 ):
        super(Retriever, self).__init__()
        self.use_vision = use_vision
        self.vision_model = vision_encoder if not vision_encoder else EVA02VisionTower()
        self.vision_dim = self.vision_model.embed_dims

        self.embedding_model = text_encoder if not text_encoder else EmbeddingModel()
        self.text_dim = self.embedding_model.embed_dims

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.cross_attention = RetrievalCrossAttention(
            vision_dim=self.vision_dim,
            text_dim=self.text_dim,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout
        )

    def forward(self, questions, images):
        """
        Forward pass for the Retriever module.

        Args:
            text_embeds (torch.Tensor): Text embeddings of shape (batch_size, seq_len, text_dim).
            image_feats (torch.Tensor): Image features of shape (batch_size, num_patches, vision_dim).

        Returns:
            torch.Tensor: Output of the cross-attention layer.
        """
        # encode image
        # (batch_size, 1792, 256) - (batch_size, vision_dim, num_patches)
        image_feats = self.vision_model(images)
        # (batch_size, 256, 1792) - (batch_size, , num_patches, vision_dim)
        image_feats = image_feats.transpose(1, 2)

        # encode text
        # (batch_size, 768) - (batch_size, text_dim)
        text_embeds = self.text_encoder(questions)
        # (batch_size, 1, 768) - (batch_size, seq_len, text_dim)
        text_embeds = text_embeds.unsqueeze(1)

        fused = self.cross_attention(
            text_embeds=text_embeds.to(self.vision_model.device),
            image_feats=image_feats.to(self.vision_model.device)
        )

        return fused.squeeze(1)
