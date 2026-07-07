Here's the README:

---

```markdown
# KG-RAG: Knowledge Graph-Augmented Retrieval-Augmented Generation for ISRO Domain Question Answering

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-In%20Development-orange)

> A hybrid RAG system that automatically constructs a domain-specific knowledge graph
> from unstructured ISRO mission documents and integrates it with FAISS dense retrieval
> for accurate, hallucination-reduced question answering — running entirely on local
> consumer-grade hardware with zero cloud dependency.

---

## Research Context

This project is developed as part of the **ISRO Bharatiya Antariksh Hackathon 2025 (BAH-02)**
and serves as the Capstone Project for the M.Tech in Artificial Intelligence and Data Science
programme at Alliance School of Advanced Computing, Alliance University (2025–2027).

**Target Publication:** ICNLP 2027 (IEEE Xplore / Scopus)

---

## System Architecture

┌─────────────────────┐
                    │   ISRO Public Docs  │
                    │   (isro.gov.in)     │
                    └─────────┬───────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │   Firecrawl API     │
                    │  Web + PDF scrape   │
                    └─────────┬───────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │    Text Chunks      │
                    │ 512 tok, stride 128 │
                    └────────┬────────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
   ┌─────────────────────┐      ┌─────────────────────┐
   │     spaCy NER       │      │   MiniLM-L6-v2      │
   │  + Dep. Parsing     │      │  (384-dim encoder)  │
   └──────────┬──────────┘      └──────────┬──────────┘
              │                             │
              ▼                             ▼
   ┌─────────────────────┐      ┌─────────────────────┐
   │    NetworkX KG      │      │    FAISS Index      │
   │ 4.2K nodes/11.5K   │      │     Flat L2         │
   │      edges          │      │                     │
   └──────────┬──────────┘      └──────────┬──────────┘
              │                             │
              │      ┌──────────────┐       │
              │      │  User Query  │       │
              │      └──────┬───────┘       │
              │             │               │
              │      ┌──────┴───────┐       │
              │      │  Query NER  │        │
              │      └──────┬───────┘       │
              │             │               │
              ▼             ▼               ▼
   ┌─────────────────────────────────────────────┐
   │            Hybrid Retrieval                 │
   │   1-hop KG neighbours + top-5 FAISS chunks  │
   └─────────────────────┬───────────────────────┘
                         │
                         ▼
           ┌─────────────────────────┐
           │  Mistral-7B-Instruct    │
           │     Q4_K_M via Ollama   │
           │   (local, zero-cost)    │
           └─────────────┬───────────┘
                         │
                         ▼
           ┌─────────────────────────┐
           │     Generated Answer    │
           └─────────────────────────┘

---

## Key Features

- Automatic KG construction from raw unstructured ISRO documents — no pre-built structured KB required
- Hybrid retrieval combining structured KG one-hop neighbourhood expansion with dense FAISS passage retrieval
- Fully local inference using Mistral-7B-Instruct Q4\_K\_M via Ollama — zero cloud dependency
- ISRO-QA: a curated benchmark of 200 domain-specific question-answer pairs across three difficulty tiers
- Evaluated using RAGAS metrics — Faithfulness, Answer Relevancy, Context Precision, Context Recall
- Deployable React frontend for interactive question answering

---

## Results (Preliminary)

| System             | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|--------------------|--------------|------------------|-------------------|----------------|
| BM25 + LLM         | 0.62         | 0.59             | —                 | —              |
| Vanilla RAG        | 0.71         | 0.68             | —                 | —              |
| GraphRAG           | —            | —                | —                 | —              |
| **KG-RAG (ours)**  | **0.84**     | **0.81**         | —                 | —              |

*Full results to be updated after experimental evaluation.*

---

## Project Structure

```
KG-RAG-ISRO/
│
├── data/
│   ├── raw/              # Scraped markdown files from isro.gov.in
│   ├── chunks/           # Preprocessed chunks (JSON)
│   ├── kg/               # NetworkX graph files
│   └── benchmark/        # ISRO-QA benchmark (JSON)
│
├── src/
│   ├── scraper/          # Firecrawl data collection scripts
│   ├── preprocessing/    # Document cleaning and chunking
│   ├── kg_builder/       # spaCy NER + relation extraction + NetworkX
│   ├── indexer/          # MiniLM embeddings + FAISS index builder
│   ├── retriever/        # Hybrid retrieval pipeline
│   ├── generator/        # Ollama API integration + prompt templates
│   ├── baselines/        # BM25, Vanilla RAG, GraphRAG implementations
│   └── evaluation/       # RAGAS evaluation scripts
│
├── frontend/             # React chat UI
│
├── notebooks/            # Experiments and analysis notebooks
│
├── paper/                # LaTeX source files for IEEE paper
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.10+
- NVIDIA GPU with 4GB+ VRAM (tested on RTX 3050 4GB)
- [Ollama](https://ollama.ai) installed and running
- Node.js 18+ (for frontend)

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/KG-RAG-ISRO.git
cd KG-RAG-ISRO
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

### 4. Pull Mistral model via Ollama

```bash
ollama pull mistral:7b-instruct-q4_K_M
```

### 5. Set up environment variables

```bash
cp .env.example .env
# Add your Firecrawl API key to .env
```

### 6. Install frontend dependencies

```bash
cd frontend
npm install
```

---

## Usage

### Step 1 — Collect documents

```bash
python src/scraper/crawl.py
```

### Step 2 — Preprocess and chunk

```bash
python src/preprocessing/chunk.py
```

### Step 3 — Build knowledge graph

```bash
python src/kg_builder/build_kg.py
```

### Step 4 — Build FAISS index

```bash
python src/indexer/build_index.py
```

### Step 5 — Run the QA system

```bash
python src/retriever/query.py --question "What is the primary payload of Chandrayaan-2?"
```

### Step 6 — Launch frontend

```bash
cd frontend
npm start
```

### Step 7 — Run evaluation

```bash
python src/evaluation/evaluate.py --benchmark data/benchmark/isro_qa.json
```

---

## ISRO-QA Benchmark

The ISRO-QA benchmark consists of 200 manually curated question-answer pairs:

| Tier      | Type                 | Count   |
|-----------|----------------------|---------|
| 1         | Factoid              | 100     |
| 2         | Multi-hop relational | 60      |
| 3         | Timeline reasoning   | 40      |
| **Total** |                      | **200** |

The benchmark JSON is located at `data/benchmark/isro_qa.json`.

---

## Tech Stack

| Component       | Tool                         |
|-----------------|------------------------------|
| Web scraping    | Firecrawl API                |
| NER + parsing   | spaCy `en_core_web_lg`       |
| Knowledge graph | NetworkX 3.x                 |
| Embeddings      | `all-MiniLM-L6-v2`           |
| Vector index    | FAISS-CPU                    |
| LLM             | Mistral-7B-Instruct Q4\_K\_M |
| LLM serving     | Ollama                       |
| Evaluation      | RAGAS                        |
| Frontend        | React                        |

---

## Team

| Name                | Role                                             | Institution|
|---------------------|--------------------------------------------------|------------|
| Nandana Narayan Das | KG pipeline, retrieval, evaluation, paper        | ASAC       |
| Gowri Kannan        | Data collection, generation, baselines, frontend | VJCET      |

---

## Citation

If you use this work or the ISRO-QA benchmark, please cite:

```bibtex
@inproceedings{narayandas2027kgrag,
  title     = {KG-RAG: Knowledge Graph-Augmented Retrieval-Augmented Generation
               for ISRO Domain Question Answering on Resource-Constrained Hardware},
  author    = {Narayan Das, Nandana and Kannan, Gowri},
  booktitle = {Proceedings of the International Conference on Natural Language Processing (ICNLP)},
  year      = {2027}
}
```

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## Acknowledgements

This work is conducted under the ISRO Bharatiya Antariksh Hackathon 2025 (BAH-02) framework.
We thank Alliance School of Advanced Computing, Alliance University for academic support.
```

---

Copy this into a file called `README.md` in the root of the repo. Want anything added or changed?
