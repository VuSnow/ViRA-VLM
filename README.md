# Resources

## Wiki Corpus
You can download the latest Vietnamese Wikipedia corpus from [dumps.wikimedia.org](https://dumps.wikimedia.org/viwiki/latest/) using the following command:
```
wget -P ./data/wiki_corpus https://dumps.wikimedia.org/viwiki/latest/viwiki-latest-pages-articles.xml.bz2
```
Replace ./data/wiki_corpus with your desired path for saving the .xml.bz2 file.

## Extracting the corpus
After downloading, you need to extract the raw wiki content using `wikiextractor`:
```
pip install wikiextractor
```
**Note:** Make sure you are using `Python ≤ 3.10` to avoid compatibility issues with wikiextractor. 

**Example command**:
```
wikiextractor data/wiki_corpus/viwiki-latest-pages-articles.xml.bz2 -o data/wiki_corpus/extracted --json
```

## Combining all the extracted json files into 1 file:
After extracting corpus using `wikiextractor`, each file in `data/wiki_corpus/extracted/` contains multiple lines, where each line is a JSON object representing one Wikipedia article.
To combine all these JSON lines into a single JSON array (and discard entries with empty `text` fields), run the following command:
```
python data/wiki_corpus_combine.py \
    --extracted_dir data/wiki_corpus/extracted \
    --output_dir data/wiki_corpus/saved_json \
    --file_name outputs.json
```
After the script finishes, you will get a single file in:
```
data/wiki_corpus/saved_json/outputs.json
```
The file contains an array of cleaned Wikipedia articles in Vietnamese, ready for downstream processing.
