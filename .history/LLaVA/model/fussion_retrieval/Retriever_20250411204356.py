import torch
import os
import numpy as np
import faiss
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
        self.use_gpu = False
        self.index = None
        self.embeddings = None

    def load_embedding(self):
        if torch.cuda.is_available():
            tensor_file = os.path.join(
                self.embedding_path, 'wiki_embeddings.pt')
            if os.path.isfile(tensor_file):
                embeddings = torch.load(tensor_file, map_location='cuda')
                self.use_gpu = True
            else:
                raise FileNotFoundError(
                    f"Embedding file {tensor_file} not found.")
        else:
            numpy_file = os.path.join(
                self.embedding_path, 'wiki_embeddings.npy')
            if os.path.isfile(numpy_file):
                embeddings = np.load(numpy_file).astype(np.float32)
            else:
                raise FileNotFoundError(
                    f"Embedding file {numpy_file} not found.")

        self.embedding_dim = embeddings.shape[1]
        return embeddings

    def build_index(self):
        if not self.
