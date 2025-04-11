import torch
from torch import nn
from LLaVA.model.embedding_model import EmbeddingModel
from LLaVA.model.fussion_retrieval import RetrievalCrossAttention
from LLaVA.model.vision_encoder_model import EVA02VisionTower


class Retriever(nn.Module):
    def __init__(self, vision_dim=1792, text_dim=768, hidden_dim=1024,
                 num_heads=8, num_layers=2, dropout=0.1, use_vision=True):
        super(Retriever, self).__init__()
        self.use_vision = use_vision
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.vision_model = EVA02VisionTower()
        self.embedding_model = EmbeddingModel()
        self.cross_attention = RetrievalCrossAttention(
            vision_dim=self.vision_dim,
            text_dim=self.text_dim,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout
        )
