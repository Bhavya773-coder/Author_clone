# Author Voice Pipeline

An end-to-end pipeline to build a **Gujarati Author Twin AI** of Shahbuddin Rathod — a famous Gujarati humorist and philosopher. The system extracts structured knowledge from 20 books, builds a semantic retrieval layer, and generates fine-tuning data to train a local LLM to speak in the author's distinct voice.

---

## Project Phases

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | OCR + Text Extraction from 20 scanned PDFs | ✅ Complete |
| Phase 1.5 | Corpus Cleaning, Normalisation & Chunking | ✅ Complete |
| Phase 2a | LLM-Assisted Structured Knowledge Extraction (Ollama) | ✅ Complete |
| Phase 2b | Stylometric Profiling (sentence length, TTR, punctuation) | ✅ Complete |
| Phase 3 | Vector Embedding & Semantic RAG (ChromaDB) | ✅ Complete |
| Phase 4 | Fine-Tuning Data Construction (SFT + DPO pairs) | 🔄 In Progress |
| Phase 5 | QLoRA Fine-tuning | 🔜 Upcoming |
| Phase 6 | Persona Orchestrator (Retrieval + Generation + Checking) | 🔜 Upcoming |

---

## Project Structure

```
author_voice_pipeline/
├── scripts/
│   ├── clean_corpus.py            # Phase 1.5: OCR cleaning and chunking
│   ├── extract_knowledge.py       # Phase 2a: Knowledge extraction via Ollama
│   ├── stylometric_profile.py     # Phase 2b: Compute author style fingerprint
│   ├── index_corpus.py            # Phase 3: Embed corpus into local ChromaDB
│   ├── retrieve.py                # Phase 3: CLI semantic search tool
│   ├── generate_tuning_data.py    # Phase 4: SFT/DPO training pair generation
│   └── verify_data.py             # Phase 4: Stylometric verification of training data
├── requirements.txt
└── ...

data/
├── style_profile.json             # Quantitative stylometric fingerprint
├── style_report.md                # Human-readable style analysis
├── knowledge/                     # 2,634 structured Markdown knowledge records
│   └── book-XX/
│       └── book-XX_cleaned_NNNN.md
├── tuning/                        # Generated fine-tuning datasets
│   ├── sft_data.jsonl             # Supervised Fine-Tuning pairs
│   └── dpo_data.jsonl             # DPO preference pairs
└── vector_db/                     # Local ChromaDB vector index (5,316 vectors)
```

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/Bhavya773-coder/Author_clone.git
cd Author_clone
```

### 2. Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Ollama and pull the model
```bash
# Install Ollama from https://ollama.com
ollama pull qwen2.5:3b
```

---

## Usage

### Run Knowledge Extraction
```bash
python -X utf8 scripts/extract_knowledge.py --cleaned-dir ../data/cleaned --knowledge-dir ../data/knowledge
```

### Build Vector Index
```bash
python -X utf8 scripts/index_corpus.py --cleaned-dir ../data/cleaned --knowledge-dir ../data/knowledge --db-dir ../data/vector_db
```

### Semantic Search
```bash
python -X utf8 scripts/retrieve.py --query "હાસ્ય" --top-k 5
```

### English Book Knowledge QA Model

The system supports training a dedicated **English Book Knowledge QA Model** trained to answer questions about the books, stories, philosophy, and anecdotes strictly in **English**.

#### 1. Generate English SFT Training Dataset
```bash
python -X utf8 scripts/generate_english_tuning_data.py --knowledge-dir knowledge --output-dir tuning
```

#### 2. Fine-Tune English Model (QLoRA)
```bash
python -X utf8 scripts/train_english.py --data-dir tuning --output-dir models/book-qa-english-v1
```

#### 3. Chat with English Book QA Model (with Cross-Lingual RAG)
```bash
# Interactive Chat
python -X utf8 scripts/chat_english.py --adapter-dir models/book-qa-english-v1

# Single Query Test (Ollama backend)
python -X utf8 scripts/chat_english.py --use-ollama --query "What is Master Saheb's philosophy on life?"
```

---

## Key Stylometric Findings
| Metric | Value |
|--------|-------|
| Mean sentence length | 8.95 words |
| Dialogue/Quote density | 401.2 per 10k words |
| Exclamation mark density | 33.7 per 10k words |
| Vocabulary diversity (TTR) | 28.97% |

---

## Tech Stack
- **OCR**: Tesseract + EasyOCR
- **Knowledge Extraction**: Ollama (`qwen2.5:3b`)
- **Embeddings**: `intfloat/multilingual-e5-base` (via sentence-transformers)
- **Vector DB**: ChromaDB (persistent local storage)
- **Fine-Tuning**: Hugging Face PEFT + TRL (QLoRA 4-bit NF4)

