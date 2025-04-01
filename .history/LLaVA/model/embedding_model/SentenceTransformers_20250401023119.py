import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize


class SentenceEmbeddingRetrieval(nn.Module):
    def __init__(self, model_name="dangvantuan/vietnamese-embedding", device=None, is_loaded=False):
        super(SentenceEmbeddingRetrieval, self).__init__()
        self.is_loaded = is_loaded
        self.model_name = model_name
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        if not self.is_loaded:
            self.load_model()

    def load_model(self, device_map=None):
        if self.is_loaded:
            print(f'{self.model_name} is already loaded, skipping.')
            return

        self.embedding_model = SentenceTransformer(self.model_name)
        self.embedding_model = self.embedding_model.to(self._device)
        self.embedding_model.requires_grad_(False)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.max_seq_length = self.embedding_model.max_seq_length
        self.is_loaded = True

    # @torch.no_grad()
    # def encode(self, )
