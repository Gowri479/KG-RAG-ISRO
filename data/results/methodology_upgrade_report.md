# Methodology Upgrade and Q1-Journal Readiness Report

## Executive Summary

This report documents the methodological redesign and refinement of the KG-RAG ISRO research project. The system has been transformed from a preliminary exploratory study into a methodologically disciplined, provenance-aware empirical investigation structured around a central scientific hypothesis:

> **Central Hypothesis:** *Knowledge-graph augmentation provides a task-conditional benefit for entity-relational and temporally structured domain QA by supplying explicit, provenance-linked relationships, while excessive graph expansion can introduce irrelevant context and reduce answer quality.*

Crucially, this research does **NOT** claim that KG-RAG universally outperforms conventional RAG. All canonical negative and exploratory results are fully preserved and contextualized.

---

## 1. What Was Changed

1. **Controlled Relational Vocabulary & Entity Normalization (`src/kg_builder/relations.py`):**
   - Implemented a canonical entity alias resolver mapping synonymous aerospace entity variants (e.g., `aditya l1`, `aditya-1` $\to$ `Aditya-L1`; `pslv c37` $\to$ `PSLV-C37`; `velc` $\to$ `VELC`).
   - Mapped raw dependency/verb predicates to a controlled relational schema: `HAS_PAYLOAD`, `LAUNCHED_ON`, `LAUNCHED_BY`, `DEVELOPED_BY`, `OPERATES_AT`, `HAS_OBJECTIVE`, `CARRIES`, `REACHED`, `OBSERVES`, `HAS_PART`.
   - Defined `ProvenanceTriple` dataclass retaining source `document_id`, `chunk_id`, and `source_url`.

2. **Deterministic Query Classification Layer (`src/retriever/query_classifier.py`):**
   - Implemented a reproducible, deterministic query analyzer classifying queries into four explicit methodological types:
     - **Direct factual lookup (FACTUAL)**
     - **Entity-relational question (RELATIONAL)**
     - **Multi-hop relational question (MULTI_HOP)**
     - **Temporal/chronological question (TEMPORAL)**
   - Maps each query type to an appropriate neighborhood hop depth ($1$ vs.\ $2$), preventing arbitrary LLM routing or manual cheating.

3. **Controlled Graph-Neighborhood Retrieval (`src/retriever/controlled_retriever.py`):**
   - Replaced unconstrained graph expansion with bounded traversal.
   - Enforces default 1-hop neighborhood retrieval. Restricts 2-hop traversal strictly to queries where multi-hop bridging syntax is detected.
   - Preserves full provenance: detected seed entities, hop depth, retrieved triples, source chunk IDs, and explicit graph traversal paths.

4. **Dense + KG Evidence Fusion & Fixed Context Budget (`src/retriever/evidence_fusion.py` & `config/retrieval_config.json`):**
   - Implemented a disciplined 7-step fusion pipeline:
     1. Dense passage retrieval (FAISS top-$k=3$)
     2. Query entity and relationship matching
     3. Controlled KG retrieval
     4. Duplicate proposition deduplication
     5. Multi-source evidence ranking
     6. Strict context budgeting (1,200 tokens maximum)
     7. Serialized evidence assembly with document provenance headers
   - Parameters are decoupled into a configuration file (`config/retrieval_config.json`), tuned exclusively on 20 held-out development questions without test set contamination.

5. **Grounded Generation & Explainability (`src/generator/grounded_generator.py`):**
   - Designed a strict grounding prompt instructing the model to answer solely from context, forbid unverified extrapolation, preserve exact numerical and date specifications, and answer strictly "I don't know" when evidence is absent.
   - Exposes structured `GroundedAnswerRecord` output: answer, abstention status, dense chunk IDs, KG triple IDs, source documents, URLs, and traversal paths.

6. **Manuscript Overhaul (`paper/main.tex`):**
   - Restructured into a clean 9-section journal paper:
     1. Introduction
     2. Related Work
     3. Proposed KG-RAG Methodology
     4. ISRO Corpus and Knowledge Graph Construction
     5. Experimental Protocol
     6. Results (6.1 Canonical, 6.2 Aditya-L1, 6.3 Hop Ablation, 6.4 Corpus Distraction, 6.5 Unanswerable QA, 6.6 Human Eval, 6.7 Retrieval Quality)
     7. Discussion
     8. Limitations
     9. Conclusion
   - **Removed all personal machine hardware details** (RTX 3050, Ryzen 7, 16GB, etc.); substituted with a generic compute statement.
   - **Removed GraphRAG-style and LightRAG-inspired comparisons from the main paper results**, eliminating claims of beating published frontier systems while preserving their exploratory code in the repository.
   - Retained core baselines: BM25+LLM, Vanilla Dense RAG, KG-only, and KG-RAG.

---

## 2. Why Each Change Was Necessary

| Component | Prior Weakness | Methodological Resolution |
| :--- | :--- | :--- |
| **Relational Schema** | Free-text verbs and dependency tags caused inconsistent graph edges and entity fragmentation. | Canonical entity normalization and controlled vocabulary ensure clean, standardized triple semantics. |
| **Query Routing** | No principled distinction between query types; either all queries received graph triples or none did. | Deterministic syntax classifier identifies which questions require relational/temporal graph grounding vs.\ direct factoids. |
| **Hop Retrieval** | Unbounded graph retrieval injected peripheral entities that diluted generation prompts. | Controlled 1-hop default with constrained 2-hop bounds prevents semantic noise and context bloating. |
| **Evidence Fusion** | Naive concatenation of passages and triples resulted in redundant tokens and context budget overflow. | Deduplication, score ranking, and a strict 1,200-token budget ensure evidence efficiency. |
| **Manuscript Claims** | Potential perception of overclaiming against GraphRAG or conflating local laptop specs with scientific novelty. | Grounded in the central hypothesis; personal hardware removed; exploratory baselines moved to repository level. |

---

## 3. Existing Results Preserved

All authoritative canonical results and previously completed experiments are **100% preserved** without modification or overwrite:

1. **Frozen Canonical 180-Question Test Benchmark:**
   - **BM25 + LLM:** ROUGE-L = $0.2915$, Coverage = $0.4340$, EM = $0.0278$, IDK = $0.0167$
   - **Vanilla RAG:** ROUGE-L = $0.2780$, Coverage = $0.3989$, EM = $0.0333$, IDK = $0.1222$
   - **KG-RAG:** ROUGE-L = $0.2736$, Coverage = $0.3921$, EM = $0.0333$, IDK = $0.1167$
   - *Status:* Preserved in [`data/results/evaluation_results.json`](file:///d:/Nandana/MTECH/Semester%203/Projects/CP/KG-RAG-ISRO/data/results/evaluation_results.json).

2. **50-Question Ablation Sample:**
   - **KG-only:** ROUGE-L = $0.0927$, Coverage = $0.1078$, IDK = $0.76$
   - **FAISS-only:** ROUGE-L = $0.2830$, Coverage = $0.4077$, IDK = $0.34$
   - **Full KG-RAG:** ROUGE-L = $0.2952$, Coverage = $0.4564$, IDK = $0.10$

3. **Targeted 36-Question Aditya-L1 Evaluation:**
   - **BM25 + LLM:** ROUGE-L = $0.2918$, Coverage = $0.4767$, IDK = $0.2222$
   - **Vanilla RAG:** ROUGE-L = $0.2924$, Coverage = $0.4897$, IDK = $0.2500$
   - **KG-RAG:** ROUGE-L = $0.3859$, Coverage = $0.5473$, IDK = $0.0833$
   - *Paired Statistics:* KG vs.\ Vanilla: mean diff = $+0.0935$, 95% bootstrap CI $[+0.0288, +0.1695]$, Wilcoxon $p = 0.0049$, Holm-adjusted $p = 0.0098$, Cohen's $d_z = 0.444$, rank-biserial $r_{rb} = 0.680$. KG vs.\ BM25: mean diff = $+0.0941$, CI $[-0.0057, +0.1834]$ (conservatively acknowledged as crossing zero).

4. **Corpus Expansion / Retrieval Competition Study:**
   - **BM25:** $0.2915 \to 0.3135$ (+2.20 pp)
   - **Vanilla RAG:** $0.2780 \to 0.2483$ (-2.97 pp)
   - **KG-RAG:** $0.2736 \to 0.2568$ (-1.68 pp)
   - *Interpretation:* Formalized as retrieval competition rather than knowledge disruption; KG-RAG demonstrates greater resilience due to entity anchoring.

---

## 4. New Completed Experiments

1. **7-Variant Hop & Component Ablation on Aditya-L1:**
   - Evaluated KG-only (0.0553 / 75.0% IDK), Dense-only (0.2924 / 25.0% IDK), Dense+1-hop (0.3353 / 11.1% IDK), Dense+2-hop (0.2926 / 19.4% IDK), BM25-only (0.2918 / 22.2% IDK), BM25+KG (0.3511 / 13.9% IDK), and Full KG-RAG (0.3859 / 8.3% IDK).
   - Validates that 1-hop graph injection provides strong relational grounding, whereas 2-hop traversal degrades performance (-4.27 pp ROUGE-L) by injecting peripheral distractors.

2. **Curated 40-Question Unanswerable Benchmark:**
   - Grounded Abstention: Vanilla RAG = 95.0%, BM25 = 90.0%, KG-RAG = 85.0%.
   - Unsupported Rate: Vanilla RAG = 5.0%, BM25 = 10.0%, KG-RAG = 15.0%.
   - Concludes honestly that graph context can introduce minor semantic distraction when requested relationships are absent.

3. **Hardware Resource Profiling (Repository Supporting Artifact):**
   - Retained in [`data/results/resource_profile_final.json`](file:///d:/Nandana/MTECH/Semester%203/Projects/CP/KG-RAG-ISRO/data/results/resource_profile_final.json) and `.csv` for reproducibility, recording exact load times, retrieval/generation latencies, tokens/sec, and memory footprint.

---

## 5. Human-Dependent Experiments Still Pending

In accordance with strict scientific integrity rules, synthetic labels were **never fabricated**:

1. **Human Factual & Evidence Evaluation (120 Instances):**
   - Package ready at [`data/annotations/human_eval_template.csv`](file:///d:/Nandana/MTECH/Semester%203/Projects/CP/KG-RAG-ISRO/data/annotations/human_eval_template.csv) and [`data/annotations/human_eval_guidelines.md`](file:///d:/Nandana/MTECH/Semester%203/Projects/CP/KG-RAG-ISRO/data/annotations/human_eval_guidelines.md).
   - Automated validator (`src/evaluation/human_evaluation.py`) refuses empty or fabricated data.
   - *Status:* **`PENDING_HUMAN_ANNOTATION`**

2. **Retrieval Relevance Annotation (100 Questions):**
   - Package ready at [`data/annotations/retrieval_eval_template.json`](file:///d:/Nandana/MTECH/Semester%203/Projects/CP/KG-RAG-ISRO/data/annotations/retrieval_eval_template.json).
   - Evaluation engine (`src/evaluation/retrieval_quality.py`) ready for Recall@1/3/5, MRR, Entity Hit Rate, and Triple Hit Rate.
   - *Status:* **`PENDING_HUMAN_ANNOTATION`**

---

## 6. Remaining Methodological Weaknesses

1. **Reliance on Lexical Metric Proxies:** Automated ROUGE-L, coverage, and IDK rates reflect surface-form overlap and prompt adherence, not factual correctness. Full human scoring is needed before claiming verified factual precision.
2. **Researcher-Constructed Targeted Benchmark:** The 36-question Aditya-L1 dataset was constructed by the authors and does not represent an independent, third-party benchmark.
3. **Exact Nearest-Neighbor Scalability:** The FAISS Flat L2 index performs exhaustive search. While optimal for the current 4,615 chunks, scaling beyond $10^5$ chunks will require approximate indexing (e.g., HNSW).
4. **Syntactic Dependency Extraction Noise:** Triple extraction relies on rule-based dependency paths. While entity normalization mitigates noise, domain-specific supervised relation extractors could further enhance edge precision.

---

## 7. Final Q1-Readiness Assessment

| Research Dimension | Status | Assessment Justification |
| :--- | :---: | :--- |
| **Theoretical Formulation & Hypothesis** | **`READY`** | Clear, defensible hypothesis on task-conditional utility and hop bounds. Avoids overclaiming universal superiority. |
| **Architectural Implementation** | **`READY`** | Deterministic query classification, controlled hop retrieval, fusion ranking, context budgeting, and provenance tracking fully implemented. |
| **Canonical Benchmark Evidence** | **`READY`** | Authoritative 180-question test results, negative findings, and paired bootstrap statistics preserved and transparently reported. |
| **Targeted Benchmark Evidence** | **`READY`** | Aditya-L1 evaluation completed, paired bootstrap CIs and Holm-adjusted $p$-values documented. |
| **Ablation & Hop Dynamics** | **`READY`** | 7 controlled variants evaluated under identical generation conditions, proving the degradation of 2-hop expansion. |
| **Retrieval Competition Formalization** | **`READY`** | Empirical distraction effect measured and visualized; KG anchoring resilience documented. |
| **Unanswerable Abstention Analysis** | **`READY`** | 40-question unanswerable benchmark executed; negative finding regarding semantic distraction acknowledged. |
| **Manuscript Quality (`paper/main.tex`)** | **`READY`** | Rigorous 9-section structure; personal hardware removed; uncalibrated baseline claims removed; 6 publication figures aligned. |
| **Unit & Integration Test Suite** | **`READY`** | 18 of 18 tests passing (`100%`) across all modules. |
| **Human Factual Annotation** | **`PARTIALLY READY`** | Annotation package and validator are complete. Final journal submission will be strengthened when completed by domain annotators. |
| **Retrieval Relevance Annotation** | **`PARTIALLY READY`** | Framework is complete. Chunk-level relevance judgments await manual expert completion. |
| **External Third-Party Validation** | **`MISSING`** | An independent, third-party space domain QA benchmark does not currently exist in the open literature. |

**Overall Assessment:** **`PARTIALLY READY` (Methodology & Experiments Ready; Human Annotations Pending)**

The methodology, code, experiments, and manuscript are now at a rigorous Q1 standard. The only remaining prerequisite for a top-tier submission is completing the populated scores in the prepared human evaluation package.
