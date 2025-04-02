import json
import os
import argparse
from tqdm import tqdm

class WikiCombination():
    def __init__(self, extracted_dir="/workspace/Vi-VLM-TTDN/data/wiki_corpus/extracted", output_dir="/workspace/Vi-VLM-TTDN/data/wiki_corpus/saved_json", file_name="outputs.json"):
        self.extracted_dir = extracted_dir
        self.output_dir = output_dir
        self.file_name = os.path.join(self.output_dir, file_name)
        self.all_docs = []
        
    def load_all_json(self):
        for root, dir, files in os.walk(self.extracted_dir):
            for filename in tqdm(files):
                filepath = os.path.join(root, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    for idx, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            if data.get("text", "").strip():  # Chỉ thêm nếu text KHÔNG rỗng
                                self.all_docs.append(data)
                        except json.JSONDecodeError as e:
                            print(f"Lỗi tại dòng {idx}: {e}")
                            print(f"Nội dung dòng: {line}, file: {filename}")
                            break
    
    def save_combined_file(self):
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.file_name, "w", encoding="utf-8") as f:
            json.dump(self.all_docs, f, ensure_ascii=False, indent=2)
            
        print(f"The number of non-empty article: {len(self.all_docs)}")
            
def main():
    parser = argparse.ArgumentParser(description="Combining all json files in wiki dumps")
    parser.add_argument("--extracted_dir", type=str, required=False, default="/workspace/Vi-VLM-TTDN/data/wiki_corpus/extracted", help="The folder contains all extracted json files")
    parser.add_argument("--output_dir", type=str, required=False, default="/workspace/Vi-VLM-TTDN/data/wiki_corpus/saved_json", help="The folder you want to save the combined file")
    parser.add_argument("--file_name", type=str, required=False, default="outputs.json", help="The name of json file")
    
    args = parser.parse_args()
    combiner = WikiCombination(extracted_dir=args.extracted_dir, output_dir=args.output_dir, file_name=args.file_name)
    combiner.load_all_json()
    combiner.save_combined_file()
    
if __name__ == "__main__":
    main()                        