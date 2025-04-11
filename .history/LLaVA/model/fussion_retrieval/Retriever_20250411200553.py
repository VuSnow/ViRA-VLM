import torch
import os
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
                 embedding_path='/workspace/Vi-VLM/data/corpus_embedding',
                 metadata_path='/workspace/Vi-VLM/data/corpus_embedding',
                 top_k=5
                 ):
        super(Retriever, self).__init__()
        self.vision_model = vision_encoder if vision_encoder else EVA02VisionTower()
        self.text_encoder = text_encoder if text_encoder else SentenceEmbeddingRetrieval()
        self.cross_attention = cross_attention if cross_attention else CrossAttention()
        self.hidden_dim = hidden_dim
        self.embedding_path = embedding_path
        self.metadata_path = metadata_path
        self.top_k = top_k

    def load_embedding(self):
        if torch.cuda.is_available():
            tensor_file = os.path.join(
                self.embedding_path, 'wiki_embeddings.pt')
            if os.path.isfile(tensor_file):
                embeddings = torch.load(tensor_file, map_location='cuda')
                self.use_gpu = True
