# ViRA-VLM — Vietnamese Retrieval-Augmented Vision-Language Model

A Vision-Language Model (VLM) for Vietnamese, built for the **VLSP 2025 MLQA-TSR** shared task (Multimodal Legal Question Answering for Traffic Sign Recognition). The model generates detailed Vietnamese image descriptions and supports retrieval-augmented QA over a Vietnamese legal/Wikipedia corpus.

## Architecture Overview

```
Image ──► EVA-CLIP (eva02_base_patch14_448) ──► Vision Projection ──► ┐
                                                                      ├──► Cross-Attention Injection ──► SeaLLMs-v3-1.5B (Qwen2) ──► Output
Text Prompt ──► Tokenizer ──► Token Embeddings ──────────────────────► ┘
```

### Key Components

| Component | Description |
|---|---|
| **Vision Encoder** | [EVA-CLIP](https://github.com/baaivision/EVA) (`eva02_base_patch14_448`), pretrained on ImageNet-22k. Extracts patch features at 448×448 resolution. |
| **Language Model** | [SeaLLMs-v3-1.5B](https://huggingface.co/SeaLLMs/SeaLLMs-v3-1.5B), a Qwen2-based multilingual LLM optimized for Southeast Asian languages. |
| **Vision-Language Fusion** | Cross-attention layers injected into the **first N** and **last N** decoder layers of the LLM, allowing text hidden states to attend to visual features. |
| **Deep Fusion** | Multi-layer bidirectional fusion blocks with self-attention + cross-attention for both modalities. |

### Training Pipeline (Stage 1 — Image Description)

The model is trained to generate detailed Vietnamese descriptions from images using the [5CD-AI/Viet-LAION-Gemini-VQA](https://huggingface.co/datasets/5CD-AI/Viet-LAION-Gemini-VQA) dataset.

- **Evaluation metrics**: BLEU, ROUGE-1, ROUGE-L, METEOR
- **Optimizer**: AdamW with cosine LR schedule, warmup ratio 0.05
- **Early stopping**: patience = 3 epochs on ROUGE-L
- **Hardware**: NVIDIA A100 GPUs via SLURM

## Project Structure

```
ViRA-VLM/
├── configs/
│   └── configs.yaml              # Model & training hyperparameters
├── models/
│   ├── attentions/
│   │   ├── self_attention.py      # Multi-head self-attention with optional positional encoding
│   │   ├── cross_attention.py     # Multi-head cross-attention (query attends to key-value)
│   │   ├── fusion_block.py        # Bi-directional fusion: vision↔language cross-attention
│   │   └── deep_fusion.py         # Stacked FusionBlocks for multi-layer deep fusion
│   ├── language_model/
│   │   ├── seallms.py             # SeaLLMs wrapper with cross-attention layer injection
│   │   └── qwen2_decoder_layer_with_attn.py  # Custom Qwen2 decoder layer with cross-attention
│   └── vision_encoder/
│       └── eva_clip.py            # EVA-CLIP vision encoder via timm
├── modules/
│   ├── wiki_processor.py          # Vietnamese Wikipedia corpus → chunked embeddings pipeline
│   ├── embed_indexer.py           # Embedding index builder (WIP)
│   └── utils.py                   # Image validation, checkpoint finder
├── stage1_description/
│   ├── model/
│   │   └── description_model.py   # Main VLM: EVA-CLIP + cross-attn injected SeaLLMs
│   ├── dataset/
│   │   ├── description_dataset.py # Dataset wrapper for image-description pairs
│   │   └── dataset_cleaning.py    # Data cleaning for Viet-LAION-Gemini-VQA
│   ├── utils/
│   │   ├── data_collator.py       # Collator: tokenizes prompts + labels, stacks images
│   │   ├── metrics.py             # BLEU / ROUGE / METEOR evaluation
│   │   └── utils.py               # print_trainable_parameters, checkpoint utils
│   ├── train.py                   # Training entry point (HuggingFace Trainer)
│   └── inference.py               # Inference / generation script
├── run_training.sh                # SLURM job script for training
├── run_inference.sh               # SLURM job script for inference
├── requirements.txt
└── pyproject.toml
```

## Installation

**Requirements**: Python ≥ 3.12, CUDA-enabled GPU

```bash
pip install -r requirements.txt
```

Main dependencies: `torch`, `transformers`, `timm`, `accelerate`, `peft`, `sentence-transformers`, `datasets`, `evaluate`, `py-vncorenlp`, `faiss-cpu`.

## Usage

### 1. Training (Stage 1 — Image Description)

```bash
python -m stage1_description.train \
    --config_path configs/configs.yaml \
    --dataset_name "5CD-AI/Viet-LAION-Gemini-VQA" \
    --num_samples 800000 \
    --split_ratio 0.1 \
    --seed 42 \
    --freeze_llm False \
    --freeze_vision False
```

Or submit via SLURM:
```bash
sbatch run_training.sh
```

**Arguments**:
| Argument | Description | Default |
|---|---|---|
| `--config_path` | Path to YAML config file | required |
| `--dataset_name` | HuggingFace dataset name | `5CD-AI/Viet-LAION-Gemini-VQA` |
| `--num_samples` | Number of samples to use | `1000` |
| `--split_ratio` | Train/eval split ratio | `0.1` |
| `--freeze_llm` | Freeze LLM parameters | `False` |
| `--freeze_vision` | Freeze vision encoder | `False` |

### 2. Inference

```bash
python -m stage1_description.inference
```

The inference script loads a trained checkpoint and generates Vietnamese descriptions for input images using sampling (top-p=0.9, temperature=0.7, max 512 tokens).

### 3. Wiki Corpus Embedding Pipeline

Build a retrieval index from Vietnamese Wikipedia for downstream RAG:

#### 3.1 Download Wikipedia dump
```bash
wget -P ./data/wiki_corpus https://dumps.wikimedia.org/viwiki/latest/viwiki-latest-pages-articles.xml.bz2
```

#### 3.2 Extract with wikiextractor
> **Note**: Requires Python ≤ 3.10 for wikiextractor compatibility.

```bash
pip install wikiextractor
wikiextractor path/to/viwiki-latest-pages-articles.xml.bz2 \
    -o data/wiki_corpus/extracted --json
```

#### 3.3 Combine extracted JSON files
```bash
python modules/wiki_corpus_combine.py \
    --extracted_dir data/wiki_corpus/extracted \
    --output_dir data/wiki_corpus/saved_json \
    --file_name outputs.json
```

#### 3.4 Process: segment → chunk → embed
```bash
python -m modules.wiki_processor \
    --json_path data/wiki_corpus/saved_json/outputs.json \
    --embedding_path data/outputs/wiki_embeddings.pt \
    --metadata_path data/outputs/wiki_metadata.pkl \
    --vncore_path modules/vncorenlp \
    --batch_size 128
```

This pipeline:
1. Segments Vietnamese text using **VnCoreNLP**
2. Chunks text into passages (max 128 tokens, 30% overlap)
3. Embeds each chunk using [dangvantuan/vietnamese-embedding](https://huggingface.co/dangvantuan/vietnamese-embedding)
4. Saves `wiki_embeddings.npy` (chunk vectors) and `wiki_metadata.pkl` (title, doc_id, chunk_id, chunk_text)

## Configuration

All model and training hyperparameters are defined in [`configs/configs.yaml`](configs/configs.yaml):

- **Vision**: EVA-CLIP model name, image size (448), feature selection (patch/cls_patch)
- **Language Model**: SeaLLMs-v3-1.5B, number of injected cross-attention layers (2)
- **Deep Fusion**: number of fusion layers, heads, FFN multiplier
- **Training**: batch size, learning rate, epochs, gradient accumulation, scheduler, etc.

## License

TBD
