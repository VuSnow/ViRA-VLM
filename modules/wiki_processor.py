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
                 batch_size: int = 64,
                 is_loaded = False):
        self.json_dir = json_dir
        self.embedding_model_name = embedding_model_name
        self.segmenter_name = segmenter_name
        self.vncorenlp_path = vncorenlp_path
        self.segmenter = None
        self.is_loaded = is_loaded
        self.batch_size = batch_size
        
        self.max_token = 128
        self.overlap_ratio = 0.3
        self.overlap_tokens = int(self.max_token * self.overlap_ratio)
        
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

        
        self.corpus = all_docs[0:2000]
        print(f"The keys in each article dictionary{self.corpus[2].keys()}")
        print(f"The number of articles: {len(self.corpus)}")
    
    def segment_text(self, raw_text: str) -> str:
        if self.segmenter is None:
            raise ImportError("the segmenter is None. Check the model path again")
        # because segmenter return a list of segmented sentence in text so we have to join them
        # return a list of segmented sentences
        segmented_list = self.segmenter.word_segment(raw_text)
        return " ".join(segmented_list)
    
    def chunk_text_by_token(self, text: str):
        input_ids = self.tokenizer(text, return_attention_mask=False, return_token_type_ids=False)["input_ids"]
        chunks = []
        
        i = 0
        while i < len(input_ids):
            window = input_ids[i : i + self.max_token]
            chunk_text = self.tokenizer.decode(window, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            chunks.append(chunk_text)
            i += (self.max_token - self.overlap_tokens)
            # i += (self.max_token - self.overlap_tokens)
        
        return chunks
    
    def embed_chunks(self, chunks: list[str]):
        embeddings = []
        for i in tqdm(range(0, len(chunks), self.batch_size), desc="Embedding chunks..."):
            batch = chunks[i : i + self.batch_size]
            batch_embeddings = self.embedder.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
            embeddings.append(batch_embeddings)
            
        return np.vstack(embeddings).astype("float32")
    
    def save_embeddings(self, path: str):
        if self.embeddings is None:
            raise ValueError("No embeddings to save.")
        np.save(path, self.embeddings)

    def save_metadata(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.metadata, f)
            
    def run(self, embedding_path: str, metadata_path: str):
        print("[1] Loading corpus...")
        self.load_json()
        skipped_empty_docs = 0
        skipped_empty_chunks = 0
        total_chunks = 0

        print("[2] Segmenting, chunking, and collecting metadata...")
        for doc in tqdm(self.corpus, desc="Processing documents"):
            doc_id = doc.get("id", "")
            title = doc.get("title", f"doc_{doc_id}")
            text = doc.get("text", "")
            if not text or len(text) < 10:
                skipped_empty_docs += 1
                print(f"Not text or len < 10, doc_id: {doc_id}")
                continue

        #     segmented = self.segment_text(text)
        #     chunks = self.chunk_text_by_token(segmented)
        #     for idx, chunk in enumerate(chunks):
        #         if chunk and chunk.strip(): # Thêm kiểm tra chunk không rỗng
        #             self.chunks.append(chunk)
        #             self.metadata.append({
        #                 "title": title,
        #                 "doc_id": doc_id,
        #                 "chunk_id": idx,
        #                 "chunk_text": chunk
        #             })
        #             total_chunks += 1
        #         else:
        #             skipped_empty_chunks += 1
        #             print(f"[WARNING] Skipping empty chunk from doc_id {doc_id}, title '{title}'") 
        
        # print(f"[INFO] Skipped {skipped_empty_docs} empty docs")
        # print(f"[INFO] Skipped {skipped_empty_chunks} empty chunks")
        # print(f"[INFO] Total processed chunks: {total_chunks}")

        # print("[3] Embedding chunks...")
        # self.embeddings = self.embed_chunks(self.chunks)

        # print("[4] Saving data...")
        # self.save_embeddings(embedding_path)
        # self.save_metadata(metadata_path)

        # print(f"✅ Done. Total chunks: {len(self.chunks)}")
        
        
test = WikiCorpusProcessor()
test.run(embedding_path="/workspace/Vi-VLM-TTDN/outputs/wiki_embeddings.npy", metadata_path="/workspace/Vi-VLM-TTDN/outputs/wiki_metadata.pkl")
    
# print(len(test.chunks))
        
        
        
        
