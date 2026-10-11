# Results Directory Index (`data/results/`)

This directory contains the frozen outputs, experiment checkpoints, manifests, statistical tests, and diagnostic reports across all phases of the ISRO Knowledge Graph RAG research.

> **Research Integrity Policy:** Old experiment outputs are **never deleted**, even if results were negative or produced prior to bug fixes. They provide an immutable, verifiable audit trail of the research methodology and its evolution.

---

## Authoritative Current Result: Pre-Registered Corrected Rerun

| Artifact Path | Description | Runner Script | Interpretation |
|---|---|---|---|
| [`corrected_rerun/generation_results.jsonl`](corrected_rerun/generation_results.jsonl) | 72 questions × 3 systems (V, A, C) generations with exact token counts | `src.evaluation.run_corrected_rerun` | Authoritative generation outputs with context-truncation bug eliminated. Every prompt fit within Ollama's 2048 context window. |
| [`corrected_rerun/analysis_summary.json`](corrected_rerun/analysis_summary.json) | Statistical analysis of AI-drafted, rater-1-reviewed evaluations | `src.evaluation.analyze_corrected_rerun` | **System A beats Vanilla (+1.17 correctness, Holm p < .001).** System C beats Vanilla (+0.93, p = .0011). Two-stage C adds no significant gain over relation-aware A. |
| [`corrected_rerun/spotcheck_analysis.json`](corrected_rerun/spotcheck_analysis.json) | Independent human spot-check analysis (20 random questions, rater IS) | `src.evaluation.analyze_corrected_rerun_spotcheck` | **FAILED pre-registered agreement threshold** (correctness Krippendorff's α = 0.64 < 0.667; groundedness α = 0.15). Directions agree (A − Vanilla ≈ +1.00), but inter-rater reliability was not independently established to required rigor. |
| [`corrected_rerun/spotcheck2_analysis.json`](corrected_rerun/spotcheck2_analysis.json) | Exploratory second independent spot-check analysis | `src.evaluation.analyze_corrected_rerun_spotcheck` | Exploratory check agreeing with rater 1 (α = 0.84), but only α = 0.58 with the first independent checker. Confirms direction (+1.00), but inter-rater agreement varies by rater. |

---

## Provenance of `ragas_scores.json`

- **File:** [`data/results/ragas_scores.json`](ragas_scores.json)
- **Provenance & Clarification:** Despite the filename containing `"ragas"`, this file does **not** contain metrics from the automated RAGAS library (such as LLM-as-a-judge faithfulness or answer relevancy). Instead, it records an early exploratory baseline evaluating lexical metrics (`rouge_l`, `answer_coverage`, `exact_match`, `idk_rate`) across all 200 questions of the initial candidate benchmark before the 20-dev / 180-test split was formalized.
- **Summary in File:**
  - `bm25` (200 questions): ROUGE-L 0.2862, coverage 0.4291, IDK rate 2.5%
  - `vanilla_rag` (200 questions): ROUGE-L 0.2667, coverage 0.3881, IDK rate 14.0%
  - `kgrag` (200 questions): ROUGE-L 0.2658, coverage 0.3841, IDK rate 10.0%
- **Status in Research Narrative:** This is a legacy exploratory artifact. It is superseded by the formal 180-question test set evaluation (`evaluation_results.json`) and the relation-aware benchmarks (`relational_qa_v1/`, `relation_aware/`, and `corrected_rerun/`).

---

## Historical Experiments (Produced Under Context Truncation)

All experiments run before October 2026 used Ollama with `num_ctx=2048` and a 1,500-word prompt budget. Because prompts frequently reached 2,400–3,900 tokens, Ollama silently discarded the earliest ~1,000+ tokens (which contained the KG evidence). These results are preserved as evidence of the truncation anomaly:

### 1. Canonical Benchmark (180 Test Questions)
- **Directory / Files:** `evaluation_results.json`, `baseline_results.json`, `statistical_tests.json`, `tier_comparison_data.json`
- **Runner:** `src.evaluation.evaluate_pipeline`
- **Findings:** No KG advantage on surface lexical metrics (ROUGE-L: BM25 .292, Vanilla .278, KG-RAG .274). Later audit showed 0 of 180 questions strictly required multi-hop KG traversal.

### 2. Aditya-L1 Targeted Benchmark (36 Questions)
- **Files:** `aditya_l1_benchmark_results.json`, `aditya_l1_benchmark_checkpoint.jsonl`, `aditya_l1_benchmark_report.md`
- **Runner:** `src.evaluation.run_aditya_l1_benchmark`
- **Findings:** KG-RAG +.094 ROUGE-L over Vanilla (p = .005); researcher-constructed questions introduced selection bias.

### 3. Relational QA v1 (62 Questions)
- **Directory:** `relational_qa_v1/`
- **Runner:** `src.evaluation.evaluate_relational_qa`
- **Findings:** 26 KG-required questions. No significant KG advantage due to prompt truncation and evaluation mismatch.

### 4. Phase 7 Corrected Structured KG (26 Questions)
- **Directory:** `final_structured_kg_eval/`
- **Runner:** `src.evaluation.evaluate_structured_kg`
- **Findings:** Descriptively superior on four dimensions, but statistically insignificant under truncation.

### 5. Relation-Aware Retrieval (72 Questions × 7 Systems)
- **Directory:** `relation_aware/`
- **Runner:** `src.evaluation.run_relation_aware_experiment`
- **Findings:** Successfully recovered all 60 KG paths; initial human evaluation hypothesis not supported under prompt truncation.

### 6. Two-Stage Generation Fix (60 KG-Required, 13 Held-Out)
- **Directory:** `generation_fix/`
- **Runner:** `src.generator.kg_grounded_generator`
- **Findings:** Lexical gains; held-out human evaluation not supported under truncation.

---

## Ablation, Diagnostic, and Profiling Artifacts

- **Ablation Studies:**
  - `ablation_results.json`: 50-question ablation (KG-only vs FAISS-only vs Full KG-RAG).
  - `component_ablation_results.json`, `hop_ablation_results.json`: 1-hop vs 2-hop retrieval contributions.
  - `topk_sensitivity.json`: Performance sensitivity across retrieval depth k.
- **Diagnostics:**
  - `kg_diagnostic_26.json`, `kg_diagnostic_audit.md`, `kg_diagnostic_report.md`: Detailed audits of triple coverage.
  - `kg_relevance_analysis.json`: Evaluates whether retrieved triples directly match question entities.
- **Resource Profiling:**
  - `resource_profile.csv`, `resource_profile.json`, `resource_profile_final.json`: Peak GPU VRAM usage (~4.1 GB), latency profiles, and CPU memory footprints demonstrating execution feasibility on consumer hardware.
- **Abstention & Robustness:**
  - `abstention_evaluation.json`, `unanswerable_eval_results.json`, `unanswerable_eval_report.md`: Rates of "I don't know" when questions lack evidence.
  - `paraphrase_robustness.json`: Retrieval stability under paraphrased phrasing.
