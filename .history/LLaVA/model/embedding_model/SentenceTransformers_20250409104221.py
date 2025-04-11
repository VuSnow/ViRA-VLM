import torch
import os
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize
from py_vncorenlp import VnCoreNLP
import py_vncorenlp


class SentenceEmbeddingRetrieval(nn.Module):
    def __init__(self, model_name="dangvantuan/vietnamese-embedding", tokenizer_path="/workspace/Vi-VLM-TTDN/modules/vncorenlp", device=None, batch_size=32, is_loaded=False):
        super(SentenceEmbeddingRetrieval, self).__init__()
        self.is_loaded = is_loaded
        self.model_name = model_name
        self.batch_size = batch_size
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self.tokenizer_path = tokenizer_path
        if not self.is_loaded:
            self.load_model()

    def load_model(self, device_map=None):
        if self.is_loaded:
            print(f'{self.model_name} is already loaded, skipping.')
            return

        self.embedding_model = SentenceTransformer(self.model_name)
        self.embedding_model = self.embedding_model.to(self._device)
        self.embedding_model.requires_grad_(False)
        # self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.max_seq_length = self.embedding_model.max_seq_length

        os.makedirs(self.tokenizer_path, exist_ok=True)
        if VnCoreNLP is None:
            raise ImportError(
                "VnCoreNLP is not installed. Please intall it with `pip install py_vncorenlp`")
        if not os.path.isabs(self.tokenizer_path):
            raise FileNotFoundError(
                f"This is not absolute path of VnCoreNLP model. Please insert the absolute path of folder containing VnCoreNLP model.")
        if not os.path.isfile(os.path.join(self.tokenizer_path, "VnCoreNLP-1.2.jar")):
            print("Downloading VnCoreNLP model...")
            py_vncorenlp.download_model(save_dir=self.tokenizer_path)
        self.segmenter = VnCoreNLP(
            save_dir=self.tokenizer_path, annotators=["wseg"])
        self.is_loaded = True

    def segment_text(self, raw_text: str):
        return " ".join(self.segmenter.word_segment(raw_text))

    @torch.no_grad()
    def encode(self, questions, tokenize_text=True):
        if isinstance(questions, str):
            questions = [questions]

        segmented_questions = [self.segment_text(
            question) for question in questions]

        embeddings = self.embedding_model.encode(
            segmented_questions,
            convert_to_tensor=True,
            batch_size=self.batch_size
        )

        return

    @property
    def embed_dims(self):
        return self.embedding_model.get_sentence_embedding_dimension()
