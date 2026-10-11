# KG-RAG: Knowledge-Graph-Augmented RAG for ISRO/ISSDC Question Answering

![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![LLM](https://img.shields.io/badge/LLM-Mistral--7B%20Q4%20(local)-purple)
![Status](https://img.shields.io/badge/Human%20validation-pending-orange)

A local question-answering system over official ISRO and ISSDC documents. It
combines a provenance-linked knowledge graph with dense retrieval and a 4-bit
Mistral-7B model served by Ollama, on resource-constrained hardware.

**Research question:** can a provenance-preserving, relation-aware knowledge
graph improve question answering over official ISRO/ISSDC documents?

This is an M.Tech (AI & Data Science) capstone at Alliance School of Advanced
Computing, Alliance University (2025–2027), developed under the ISRO Bharatiya
Antariksh Hackathon 2025 (BAH-02).

---

## Current status (October 2026)

| | |
|---|---|
| Latest result | Pre-registered corrected rerun: relation-aware KG-RAG beats vanilla dense RAG on correctness |
| Caveat | Scores are AI-drafted and reviewed by one human (99.2% unchanged) |
| In progress | Independent human spot-check on 20 random questions (pre-registered, §14) |
| Paper | Not yet updated with the corrected results; kept locally, outside git |

### Headline result

Pre-registered rerun on the frozen relation-aware benchmark (60 KG-required
questions), with every prompt guaranteed to fit the model's context window:

| System | Correctness (1–5) | Completeness | Groundedness | Relevance | Unsupported claims |
|---|---:|---:|---:|---:|---:|
| Vanilla dense RAG | 3.22 | 3.12 | 3.77 | 3.98 | 43.3% |
| **A: relation-aware KG-RAG** | **4.38** | **4.33** | **4.42** | **4.63** | **16.7%** |
| C: two-stage KG-grounded | 4.15 | 4.12 | 4.10 | 4.40 | 28.3% |

- **A vs Vanilla:** correctness +1.17, 95% CI [0.72, 1.62], Holm-adjusted p < .001. Decision: **SUPPORTED**.
- **C vs Vanilla:** correctness +0.93, 95% CI [0.42, 1.43], Holm-adjusted p = .0011. Decision: **SUPPORTED**.
- **C vs A:** −0.23, 95% CI [−0.60, 0.12], not significant. The two-stage step adds nothing over relation-aware retrieval.
- The effect holds with duplicate questions removed (n = 45) and with direct-fact questions excluded (n = 50).

**How far this can be trusted:** the scores come from one rater who reviewed
a model-drafted sheet, so they are effectively an AI judge's verdict. The
references were derived from KG triples, which may favour KG systems that
reuse triple wording. Until the independent spot-check passes, report this as
*AI-drafted, human-reviewed* and *not yet independently validated*. See
[`reports/preregistration_corrected_rerun.md`](reports/preregistration_corrected_rerun.md) §12–§14.

---

## What changed: the context-truncation bug

Every generation run before October 2026 used Ollama with `num_ctx=2048` and
a 1,500-**word** context budget. Real prompts were often 2,400–3,900 model
tokens. Ollama silently kept only about the **last** 1,027 tokens and dropped
the beginning, which is exactly where the KG evidence sat. In some runs up to
78% of the prompt never reached the model.

This affected every earlier experiment, including the 180-question benchmark,
Aditya-L1, relational v1, Phase 7, the relation-aware run and the two-stage
experiments. Their negative or mixed results describe systems that mostly
never saw their KG evidence.

**Fix** (commit `7cb2d2a`):
- [`src/generator/token_budget.py`](src/generator/token_budget.py) counts exact
  model tokens with the tokenizer extracted from the model's own GGUF file
  ([`config/tokenizer/`](config/tokenizer/)). Its counts match Ollama's
  `prompt_eval_count` exactly.
- Every call through `generate_with_metrics` fits the prompt to
  `num_ctx − num_predict`. It trims the end of the context (lowest-priority
  evidence), never the question, and logs `prompt_tokens_expected`,
  `context_trimmed` and `ollama_truncated`.
- KG generation budgets now use model tokens, and the structured prompts fit
  their evidence before wrapping.

---

## Experimental history

All earlier artifacts are kept unchanged as an audit trail. The results
listed under "earlier experiments" were produced **under truncation**.

| Stage | Benchmark | Outcome |
|---|---|---|
| Canonical benchmark | 180 test questions | No KG advantage (ROUGE-L: BM25 .292, Vanilla .278, KG-RAG .274). An audit later found 0 of 180 questions strictly require the KG |
| Aditya-L1 targeted | 36 researcher-built questions | KG-RAG +.094 ROUGE-L (p = .005); selection-biased, not independent |
| Relational v1 | 62 questions (26 KG-required) | No KG advantage; traced to an evaluation/KG mismatch, then fixed |
| Phase 7 corrected structured KG | 26 KG-required, human-rated | Better on all 4 dimensions descriptively; nothing significant |
| Relation-aware retrieval | 72 questions × 7 systems | All 60 KG paths recovered; human evaluation NOT SUPPORTED |
| Two-stage generation | 60 KG-required, 13 held out | Lexical gains; held-out human evaluation NOT SUPPORTED |
| **Corrected rerun (pre-registered)** | **72 questions × 3 systems** | **KG systems beat Vanilla; see above** |

Known limitations of the relation-aware benchmark v2, which is frozen and
documented rather than edited:
- 45 unique texts among its 60 KG-required items;
- 5 questions labelled both KG-required and not;
- set-valued questions with a single reference answer;
- references derived from the same KG the retriever queries;
- held-out leakage: 6 of 13 held-out questions repeat development texts.

---

## Corpus and knowledge graph

- 33 Tier-1 official sources (24 ISRO, 9 ISSDC); 1,006 chunks; about 65.6k words.
- 123 entities and 134 provenance-linked triples. Each triple carries its source
  document, chunk, URL and excerpt.
- Relations include `HAS_PAYLOAD`, `HAS_OBJECTIVE`, `OBSERVES`, `STUDIES`,
  `DEVELOPED_BY`, `LAUNCHED_BY`, `LAUNCHED_FROM` and `LAUNCHED_ON`.
- Missions: Aditya-L1, AstroSat, Chandrayaan-1/2/3, Gaganyaan, Mars Orbiter Mission.

Canonical files: [`data/corpus/chunks.jsonl`](data/corpus/chunks.jsonl),
[`data/corpus/triples.jsonl`](data/corpus/triples.jsonl),
[`data/corpus/entities.jsonl`](data/corpus/entities.jsonl).

---

## Systems

All three use the same model, settings and 1,500-token evidence budget.

- **Vanilla dense RAG:** MiniLM embeddings and FAISS retrieve text chunks; one
  context-only generation call.
- **A: relation-aware KG-RAG** ([`src/retriever/relation_aware.py`](src/retriever/relation_aware.py)):
  detects entities, the requested relation, its direction and the hop depth;
  retrieves 1–2-hop provenance-linked KG paths, scored by dense, lexical,
  entity, relation and provenance signals. KG evidence comes first, then
  dense text; one generation call.
- **C: two-stage KG-grounded** ([`src/generator/kg_grounded_generator.py`](src/generator/kg_grounded_generator.py)):
  the same evidence as A. The model first writes an evidence plan, then
  answers from the verified KG facts and the plan only.

---

## Reproducing the corrected rerun

Prerequisites: Python 3.11, [Ollama](https://ollama.com) with
`mistral:7b-instruct-q4_K_M`, and `pip install -r requirements.txt`.

```bash
ollama pull mistral:7b-instruct-q4_K_M
```

```bash
python -m src.evaluation.run_corrected_rerun --dry-run
```

```bash
python -m src.evaluation.run_corrected_rerun
```

```bash
python -m src.evaluation.analyze_corrected_rerun
```

- The dry run builds and fits every prompt without calling the model.
- The full run refuses to start unless the pre-registration is approved and
  committed. It checks the benchmark and frozen-retrieval hashes and records
  per-call integrity: expected token count equals evaluated token count.
- Outputs go to `data/results/corrected_rerun/`.

Spot-check (pending): an independent rater scores
`data/annotations/corrected_rerun_spotcheck.csv` from scratch and saves it as
`corrected_rerun_spotcheck_filled.csv`. Then:

```bash
python -m src.evaluation.analyze_corrected_rerun_spotcheck
```

Run the tests:

```bash
pytest -q
```

### Reproducibility notes

- `.gitattributes` stores `data/**` byte-for-byte, so a fresh clone reproduces
  every recorded SHA-256 (benchmark v2 `6c7f3600…`, frozen retrieval
  `e13e4c82…`, corrected rerun `2d9b6f47…`). Don't remove it.
- Large local artifacts (`data/raw`, `data/chunks`, `data/index`, `data/kg`,
  `data/cleaned`) and `paper/` are not tracked.
- [`src/generator/extract_tokenizer.py`](src/generator/extract_tokenizer.py)
  regenerates the tokenizer if the Ollama model changes; it needs `gguf`.

---

## HTTP API for frontends

`src/api` serves the relation-aware KG-RAG system (system A) over HTTP, so a
frontend can be built in any framework. The guide, TypeScript types and
OpenAPI spec are in [`docs/frontend/`](docs/frontend/README.md).

```bash
python -m src.api --mock
```

Mock mode needs only git-tracked files. Live mode (`python -m src.api`)
needs Ollama and the local index; it reproduces system A's rerun answers
exactly. Interactive docs are at http://127.0.0.1:8000/docs.

---

## Repository layout

```
config/tokenizer/        Tokenizer extracted from the Ollama GGUF (exact token counts)
data/
  corpus/                Canonical Phase-1 corpus: chunks, entities, triples, sources
  benchmark/             200-question canonical benchmark and splits
  relational_benchmark/  Relational QA v1 (62 questions)
  relation_aware_benchmark/  Relation-aware QA v2 (72 questions, frozen)
  annotations/           Human-evaluation packages and filled sheets
  results/               Every experiment's outputs and manifests (frozen)
    corrected_rerun/     Pre-registered rerun: generations, analysis, spot-check
reports/                 Audits, construction reports, pre-registration
src/
  corpus/                Collection, extraction, cleaning, chunking, KG assembly
  retriever/             Dense, KG, hybrid, and relation-aware retrieval
  generator/             Ollama client, prompts, token budgeting, KG-grounded generation
  baselines/             BM25 + LLM, vanilla RAG, GraphRAG, LightRAG
  evaluation/            Runners, analyses, and human-evaluation package builders
tests/                   pytest suite
src/api/                 HTTP API for frontends (FastAPI)
docs/frontend/           Frontend guide, TypeScript types, OpenAPI spec
app.py, app_v2.py        Streamlit demo apps
```

---

## Team

| Name | Role | Institution |
|---|---|---|
| Nandana Narayan Das | KG pipeline, retrieval, evaluation, paper | Alliance School of Advanced Computing, Alliance University |
| Gowri Kannan | Data collection, generation, baselines, frontend | VJCET |

## Citation

```bibtex
@inproceedings{narayandas2027kgrag,
  title     = {KG-RAG: Knowledge Graph-Augmented Retrieval-Augmented Generation
               for ISRO Domain Question Answering on Resource-Constrained Hardware},
  author    = {Narayan Das, Nandana and Kannan, Gowri},
  booktitle = {Proceedings of the International Conference on Natural Language Processing (ICNLP)},
  year      = {2027}
}
```

## License

MIT. See `LICENSE`.

## Acknowledgements

Conducted under the ISRO Bharatiya Antariksh Hackathon 2025 (BAH-02)
framework. We thank Alliance School of Advanced Computing, Alliance
University, for academic support.
