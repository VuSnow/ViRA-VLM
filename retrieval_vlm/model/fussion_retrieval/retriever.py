import torch
import os
import numpy as np
import faiss
from torch import nn
from retrieval_vlm.model.embedding_model import SentenceEmbeddingRetrieval
from retrieval_vlm.model.attention.cross_attention import CrossAttention
from retrieval_vlm.model.vision_encoder_model.eva_clip import EvaClip


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
        # init vision model
        self.vision_model = vision_encoder if vision_encoder else EvaClip()
        self.vision_dim = self.vision_model.embed_dims

        # init text model
        self.text_encoder = text_encoder if text_encoder else SentenceEmbeddingRetrieval()
        self.text_dim = self.text_encoder.embed_dims
        self.hidden_dim = hidden_dim

        # init cross attention layer
        if cross_attention is None:
            cross_attention = CrossAttention(
                vision_dim=self.vision_dim,
                text_dim=self.text_dim,
                hidden_dim=self.hidden_dim
            )
        else:
            cross_attention.vision_dim = self.vision_dim
            cross_attention.text_dim = self.text_dim
            cross_attention.hidden_dim = hidden_dim

        self.embedding_path = embedding_path
        self.metadata_path = metadata_path
        self.top_k = top_k
        self.use_gpu = False
        self.index = None
        self.embeddings = None
        if self.embeddings is None:
            self.embeddings = self.load_embedding()
        if self.index is None:
            self.index = self.build_index()

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
        index = faiss.IndexFlatL2(self.embedding_dim)
        if self.use_gpu:
            ngpus = faiss.get_num_gpus()
            print(f"Using {ngpus} GPUs")
            index = faiss.index_cpu_to_all_gpus(index)
            index.add(self.embeddings.contiguous())
        else:
            index.add(self.embeddings)
        return index

    def forward(self, segmented_texts, image_feats):
        """
        Forward pass for the Retriever module.

        Args:
            segmented_texts (list[str]): List of segmented texts.
            image_feats (torch.Tensor): Image features of shape (batch_size, num_patches, vision_dim).

        Returns:
            torch.Tensor: Output of the cross-attention layer.
        """
