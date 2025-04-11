import torch
import os
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize
from py_vncorenlp import VnCoreNLP
import py_vncorenlp


class SentenceEmbeddingRetrieval(nn.Module):
    def __init__(self,
                 model_name="dangvantuan/vietnamese-embedding",
                 device=None,
                 batch_size=32,
                 is_loaded=False
                 ):
        super(SentenceEmbeddingRetrieval, self).__init__()
        self.is_loaded = is_loaded
        self.model_name = model_name
        self.batch_size = batch_size
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
        self.embedding_model.eval()
        self.embedding_model.requires_grad_(False)
        self.is_loaded = True

    @torch.no_grad()
    def forward(self, segmented_texts: list[str], tokenize_text=True):
        if isinstance(segmented_texts, str):
            segmented_texts = [segmented_texts]

        return self.embedding_model.encode(
            segmented_texts,
            convert_to_tensor=True,
            batch_size=self.batch_size,
            device=self._device
        )

    @property
    def embed_dims(self):
        return self.embedding_model.get_sentence_embedding_dimension()

    @property
    def max_seq_length(self):
        return self.embedding_model.max_seq_length
