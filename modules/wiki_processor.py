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
                 chunk_size: int = 256,
                 stride: int = 128,
                 is_loaded = False):
        self.json_dir = json_dir
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.stride = stride
        self.segmenter_name = segmenter_name
        self.vncorenlp_path = vncorenlp_path
        self.segmenter = None
        self.is_loaded = is_loaded
        
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
        
        self.load_json()
        
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
        return self.segmenter.word_segment(raw_text)
    
    # def chunk_text(self, text: str):
        
print(WikiCorpusProcessor())
        
        
        

        
        
        
        
