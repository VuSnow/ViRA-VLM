import os
import json
import pickle
import numpy as np 
import argparse
import torch
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
                 json_path: str = "/workspace/Vi-VLM-TTDN/data/wiki_corpus/saved_json/outputs.json",
                 embedding_path: str = "/workspace/Vi-VLM-TTDN/data/outputs/wiki_embeddings.pt",
                 metadata_path: str = "/workspace/Vi-VLM-TTDN/data/outputs/wiki_metadata.pkl",
                 embedding_model_name: str = "dangvantuan/vietnamese-embedding",
                 segmenter_name: str = "VnCoreNLP",
                 vncorenlp_path: str = "/workspace/Vi-VLM-TTDN/modules/vncorenlp",
                 batch_size: int = 32,
                 is_loaded: bool = False):
        self.json_path = json_path
        self.embedding_path = embedding_path
        self.metadata_path = metadata_path
        self.embedding_model_name = embedding_model_name
        self.segmenter_name = segmenter_name
        self.vncorenlp_path = vncorenlp_path
        self.is_loaded = is_loaded
        self.batch_size = batch_size
        
        self.max_token = 128
        self.overlap_ratio = 0.3
        self.overlap_tokens = int(self.max_token * self.overlap_ratio)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder = SentenceTransformer(self.embedding_model_name, device=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
        
        self.segmenter = None
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
        
    def load_data(self):
        if not os.path.isfile(self.json_path):
            raise FileNotFoundError(f"File not found: {self.json_path}")
        
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("File JSON không chứa danh sách bài viết (list of dicts).")
            
            for article in tqdm(data, desc="Loading articles"):
                self.corpus.append(article)
            print(f"Loaded {len(self.corpus)} articles from {self.json_path}")
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e} in file {self.json_path}")
        
        print(f"The number of empty text in self.corpus: {len([article for article in self.corpus if not article['text']])}")
        print(f"Keys in each article information in self.corpus: {self.corpus[0].keys()}")
    
    def segment_text(self, raw_text: str):
        try:
            return " ".join(self.segmenter.word_segment(raw_text))
        except UnicodeDecodeError as e:
            print(f"UnicodeDecodeError at segment_text(): {e}")
            return ""
        except Exception as e:
            print(f"General error in segment_text(): {e}")
            return ""
    
    def chunk_text_by_token(self, text:str):
        # This function returns a list of passages that have the number of tokens < max_token_length
        input_ids = self.tokenizer(text, return_attention_mask=False, return_token_type_ids=False)["input_ids"]
        chunks = []
        step = self.max_token - self.overlap_tokens
        
        for i in range(0, len(input_ids) - self.max_token + 1, step):
            
            window = input_ids[i : i + self.max_token]
            chunk_text = self.tokenizer.decode(window, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            chunks.append(chunk_text)
            
        if len(input_ids) > 0 and (len(input_ids) - self.max_token) % step != 0:
            last_window = input_ids[-self.max_token:]
            chunk_text = self.tokenizer.decode(last_window, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            if chunk_text not in chunks:
                chunks.append(chunk_text)
                
        return chunks
    
    def embed_chunks(self, chunks: list[str]):
        embeddings = []
        
        for i in tqdm(range(0, len(chunks), self.batch_size), desc="Embedding chunks:..."):
            batch = chunks[i : i + self.batch_size]
            batch_embeddings = self.embedder.encode(
                batch, 
                convert_to_tensor=True, 
                normalize_embeddings=True, 
                batch_size=self.batch_size,
                device=self.device
            )
            embeddings.append(batch_embeddings)
            
        return torch.cat(embeddings, dim=0)
    
    def save_embeddings(self):
        if self.embeddings is None:
            raise ValueError("No embeddings to save.")
        
        os.makedirs(os.path.dirname(self.embedding_path), exist_ok=True)
        torch.save(self.embeddings, self.embedding_path)
        print(f"Saved torch tensor embeddings to {self.embedding_path}")
        print(f"Embedding tensor shape: {self.embeddings.shape}")
        
        np_path = self.embedding_path.replace(".pt", ".npy")
        np.save(np_path, self.embeddings.cpu().numpy())
        print(f"Saved numpy embeddings to {np_path}")
        print(f"Embedding numpy shape: {self.embeddings.cpu().numpy().shape}")
        
    def save_metadata(self):
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"Saved metadata to {self.metadata_path}")
        
    def run(self):
        print("[1] Loading Wiki Corpus from json file...")
        self.load_data()
        
        print("[2] Chunking by token and collecting metadata...")
        for doc in tqdm(self.corpus, desc="Processing documents"):
            title = doc["title"]
            doc_id = int(doc["id"])
            text = doc["text"]
            if not text.strip():
                continue
            
            segmented_text = self.segment_text(text)
            chunks = self.chunk_text_by_token(segmented_text)
            for idx, chunk in enumerate(chunks):
                self.chunks.append(chunk)
                self.metadata.append({
                    "title": title,
                    "doc_id": doc_id,
                    "chunk_id": idx,
                    "chunk_text": chunk
                })
        print(f"Total chunks: {len(self.chunks)}")
        
        print("[3] Embedding chunks...")
        self.embeddings = self.embed_chunks(self.chunks)
        
        print("[4] Saving data...")
        self.save_embeddings()
        self.save_metadata()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default="/workspace/Vi-VLM-TTDN/data/wiki_corpus/saved_json/outputs.json", help="The path of combined json file")
    parser.add_argument("--embedding_path", type=str, default="/workspace/Vi-VLM-TTDN/data/outputs/wiki_embeddings.pt", help="The path to save pt embeddings file")
    parser.add_argument("--metadata_path", type=str, default="/workspace/Vi-VLM-TTDN/data/outputs/wiki_metadata.pkl", help="The path to save metadata pickle file")
    parser.add_argument("--vncore_path", type=str, default="/workspace/Vi-VLM-TTDN/modules/vncorenlp", help="The path to save vncorenlp model")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    args = parser.parse_args()
    
    processor = WikiCorpusProcessor(
        json_path=args.json_path,
        embedding_path=args.embedding_path,
        metadata_path=args.metadata_path,
        vncorenlp_path=args.vncore_path,
        batch_size=args.batch_size
    )
    processor.run()
    
if __name__ == "__main__":
    main()
