import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize


class SentenceEmbeddingRetrieval(nn.Module):
    def __init__(self, model_name="dangvantuan/vietnamese-embedding", device=None, is_loaded=False):
        super(SentenceEmbeddingRetrieval, self).__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_model.to(self._device)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.max_seq_length = self.embedding_model.max_seq_length
