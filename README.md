# Wiki Corpus Embedding Pipeline

## 1. Wiki Corpus
You can download the latest Vietnamese Wikipedia corpus from [dumps.wikimedia.org](https://dumps.wikimedia.org/viwiki/latest/) using the following command:
```bash
wget -P ./data/wiki_corpus https://dumps.wikimedia.org/viwiki/latest/viwiki-latest-pages-articles.xml.bz2
```
Replace ./data/wiki_corpus with your desired path for saving the .xml.bz2 file.

## 2. Extracting the corpus
After downloading, you need to extract the raw wiki content using `wikiextractor`:
```bash
pip install wikiextractor
```
**Note:** Make sure you are using `Python ≤ 3.10` to avoid compatibility issues with wikiextractor. 

**Example command**:
```bash
wikiextractor path/to/viwiki-latest-pages-articles.xml.bz2 -o path/to/saved/folder/wiki_corpus/extracted --json
```

## 3. Combining all the extracted json files into 1 file:
After extracting corpus using `wikiextractor`, each file in `data/wiki_corpus/extracted/` contains multiple lines, where each line is a JSON object representing one Wikipedia article.
To combine all these JSON lines into a single JSON array (and discard entries with empty `text` fields), run the following command:
```bash
python path/to/wiki_corpus_combine.py \
    --extracted_dir data/wiki_corpus/extracted \
    --output_dir data/wiki_corpus/saved_json \
    --file_name outputs.json
```
After the script finishes, you will get a single file in:
```bash
data/wiki_corpus/saved_json/outputs.json
```
The file contains an array of cleaned Wikipedia articles in Vietnamese, ready for downstream processing.

## 4. Run the processor
After combining all the extracted json files, run the following command:
```bash
python path/to/wiki_processor.py \
    --json_path data/wiki_corpus/saved_json/outputs.json \
    --embedding_path data/outputs/wiki_embeddings.npy \
    --metadata_path data/outputs/wiki_metadata.pkl \
    --vncore_path modules/vncorenlp \
    --batch_size 32
```
The above command: 
- Loads all articles from the combined JSON file 
- Segments Vietnamese text using VnCoreNLP
- Chunks long texts into passages (max 128 tokens, overlap 30%)
- embeds each chunk using a pretrained Sentence-BERT model [dangvantuan/vietnamese-embedding]
- Save:
    - `wiki_embeddings.npy`: chunk embeddings [float32]
    - `wiki_metadata.pkl`: metadata for each chunk (`title`, `doc_id`, `chunk_id`, `chunk_text`)


