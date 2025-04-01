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
