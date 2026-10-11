# Data Directory

This directory contains the datasets, corpus files, benchmarks, human-annotation sheets, and experimental results for the ISRO Knowledge Graph RAG project.

## Authoritative Corpus vs Tracked Artifacts

Contrary to older documentation, **key data artifacts are actively tracked in Git** to maintain an immutable audit trail and support zero-setup mock runs.

### Tracked in Git
- **`data/corpus/`** — **The authoritative Phase 1 corpus.** Contains 33 Tier-1 official ISRO/ISSDC sources, 1,006 chunks (~65.6k words), 123 entities, and 134 provenance-linked triples.
  - `document_registry.json`: Registry of official sources with SHA-256 checksums and URLs.
  - `chunks.jsonl`: Semantic section/page-aware chunks with metadata.
  - `entities.jsonl`: Extracted and normalized ISRO entities.
  - `triples.jsonl`: Provenance-grounded facts with source document, chunk, URL, and excerpt.
  - `fact_annotations.jsonl`: Manually verified source claims with chunk locators.
  - `ontology.json`, `relations.json`: Entity and relation schema definitions.
  - Reproducibility reports: `extraction_report.json`, `chunking_report.json`, `validation_report.json`, etc.
  - For full details, see [`data/corpus/README.md`](corpus/README.md).
- **`data/relation_aware_benchmark/`** — The frozen Relation-Aware QA v2 benchmark (`relation_aware_qa_v2.json`, 72 questions) used for the pre-registered corrected rerun.
- **`data/relational_benchmark/`** — Relational QA v1 benchmark (`relational_qa_v1.json`, 62 questions).
- **`data/benchmark/`** — The canonical 200-question benchmark (20-dev / 180-test split) from early project phases.
- **`data/annotations/`** — Evaluation templates, rater guidelines, completed human evaluation workbooks, and spot-checks. See [`data/annotations/README.md`](annotations/README.md).
- **`data/results/`** — Frozen experimental outputs, generation logs, and analysis summaries across all project phases. See [`data/results/README.md`](results/README.md).
- **`data/source_manifest.csv`**, **`data/abstention/`**, **`data/documents/`** — Supplemental manifests and documentation.

### Gitignored (Local Scratch & Cache)
Per `.gitignore`, large or machine-generated intermediate caches are excluded from version control:
- `data/raw/`: Raw downloaded documents from web crawling.
- `data/chunks/`: Intermediate scratch chunks from historical crawler pipelines.
- `data/kg/`: Legacy NetworkX graph pickles (`kg.gpickle`).
- `data/index/`: Local FAISS dense vector index (`faiss.index`, `chunks.jsonl`).
- `data/cleaned/`: Intermediate text cleaning scratch.

---

## Reproducing the Authoritative Corpus

The current system relies on the audited Phase 1 corpus pipeline in `src/corpus/`. To reproduce the corpus from scratch from the repository root:

```bash
# 1. Download official Tier-1 documents registered in document_registry.json
python -m src.corpus.download_documents

# 2. Extract text and structural records from downloaded documents
python -m src.corpus.extract_documents

# 3. Clean whitespace and remove repeated margin headers
python -m src.corpus.clean_documents

# 4. Chunk text into semantic section-aware units (up to 350 words)
python -m src.corpus.chunk_documents

# 5. Build KG-ready entities and triples from fact annotations
python -m src.corpus.build_kg_ready

# 6. Analyze relation coverage and graph statistics
python -m src.corpus.coverage_analysis

# 7. Validate corpus integrity and provenance constraints
python -m src.corpus.validate_corpus
```

> **Note on Fact Annotations:** `fact_annotations.jsonl` contains human-reviewed source claims. Rebuilding KG-ready outputs validates these facts against the source documents rather than generating facts automatically from an unverified LLM.

---

## Building the Local Dense Retrieval Index

To run the live pipeline locally with dense FAISS retrieval:

```bash
python -m src.indexer.build_index
```

This encodes the chunks into 384-dimensional dense vectors using `all-MiniLM-L6-v2` and writes the FAISS FlatL2 index to `data/index/faiss.index`.
