import os
import json
import pickle
import numpy as np 
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
    
try:
    from py_vncorenlp import VnCoreNLP
    import py_vncorenlp
except ImportError:
    VnCoreNLP = None

class WikiCorpusProcessor():
    def __init__(self,
                 json_dir: str = "/workspace/Vi-VLM-TTDN/data/wiki_corpus/extracted",
                 embedding_model_name: str = "dangvantuan/vietnamese-embedding",
                 segmenter_name: str = "VnCoreNLP",
                 vncorenlp_path: str = "/workspace/Vi-VLM-TTDN/modules/vncorenlp",
                 chunk_size: int = 8,
                 batch_size: int = 64,
                 is_loaded = False):
        self.json_dir = json_dir
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.segmenter_name = segmenter_name
        self.vncorenlp_path = vncorenlp_path
        self.segmenter = None
        self.is_loaded = is_loaded
        self.batch_size = batch_size
        
        self.embedder = SentenceTransformer(self.embedding_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
        
        if not self.is_loaded:
            if VnCoreNLP is None:
                raise ImportError("VnCoreNLP is not installed. Please intall it with `pip install py_vncorenlp`")
            if not os.path.isabs(self.vncorenlp_path):
                raise FileNotFoundError(f"This is not absolute path of VnCoreNLP model. Please insert the absolute path of folder containing VnCoreNLP model.")
            py_vncorenlp.download_model(save_dir=self.vncorenlp_path)
            self.segmenter = VnCoreNLP(save_dir=self.vncorenlp_path, annotators=["wseg"])
            self.is_loaded = True
            
        self.corpus = []
        self.chunks = []
        self.metadata = []
        self.embeddings = None
        
        # self.load_json()
        
    def load_json(self):
        all_docs = []
        for root, dirs, files in os.walk(self.json_dir):
            for filename in tqdm(files):
                file_path = os.path.join(root, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            if isinstance(data, dict):
                                all_docs.append(data)
                        except json.JSONDecodeError as e:
                            print(f"[WARNING] Skipping malformed line in {filename}: {e}")

        self.corpus = all_docs
        print(f"The keys in each article dictionary{self.corpus[2].keys()}")
        print(f"The number of articles: {len(self.corpus)}")
    
    def segment_text(self, raw_text: str):
        if self.segmenter is None:
            raise ImportError("the segmenter is None. Check the model path again")
        return self.segmenter.word_segment(raw_text) # return a list of segmented sentences
    
    def chunk_by_sentence(self, segmented_sentences: list, overlap: int = 6):
        assert 0 <= overlap < self.chunk_size
        step = self.chunk_size - overlap
        chunks = []
        last_index = 0
        for i in range(0, len(segmented_sentences) - self.chunk_size + 1, step):
            chunk = " ".join(segmented_sentences[i:i+self.chunk_size])
            chunks.append(chunk)
            last_index = i + step       
        if last_index < len(segmented_sentences):
            chunks.append(" ".join(segmented_sentences[-self.chunk_size:]))
        return chunks    
            
    def embed_chunks(self, chunks: list[str]) -> np.ndarray:
        embeddings = []
        for i in tqdm(range(0, len(chunks), self.batch_size), desc="Embedding chunks"):
            batch = chunks[i:i + self.batch_size]
            batch_embeddings = self.embedder.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
            embeddings.append(batch_embeddings)
        return np.vstack(embeddings).astype("float32")
    
    def save_embeddings(self, path: str):
        if self.embeddings is None:
            raise ValueError("No embeddings to save.")
        np.save(path, self.embeddings)
        
    def save_metadata(self, metadata_path: str):
        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
            
    def run(self, embedding_path: str, metadata_path: str):
        print("[1] Loading JSON corpus...")
        self.load_json()

        print("[2] Segmenting, chunking, and collecting metadata...")
        
        
test = WikiCorpusProcessor()
test.run(embedding_path="./outputs/wiki_embeddings.npy", metadata_path="./outputs/wiki_metadata.pkl")
    
print(len(test.chunks))
        
        
        
        
