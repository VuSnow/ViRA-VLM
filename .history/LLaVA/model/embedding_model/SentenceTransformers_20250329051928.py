import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize

sentences = ["Hà Nội là thủ đô của Việt Nam", "Đà Nẵng là thành phố du lịch"]
tokenizer_sent = [tokenize(sent) for sent in sentences]

model = SentenceTransformer('dangvantuan/vietnamese-embedding')
embeddings = model.encode(tokenizer_sent)
print(embeddings)


class SentenceEmbeddingRetrieval(nn.Module):
    def __init__(self, model_name="dangvantuan/vietnamese-embedding", device=None):
        super(SentenceEmbeddingRetrieval, self).__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_model.to(self._device)
