# Aditya-L1 Benchmark Evaluation Report: Knowledge Graph Benefit Analysis

**Date**: 2026-10-05 20:25:26 UTC
**Dataset**: 36-question targeted Aditya-L1 benchmark (`data/benchmark/aditya_l1_optional_qa.json`)
**Backends**: Local Ollama (`mistral:7b-instruct-q4_K_M`), deterministic temp=0.1, max_tokens=1500 context

## 1. Overall System Results (N=36)

| System | ROUGE-L | Ref Token Coverage | Exact Match | IDK Rate |
| :--- | :---: | :---: | :---: | :---: |
| **bm25_llm** | 0.2918 | 0.4767 | 0.0000 | 0.2222 |
| **vanilla_rag** | 0.2924 | 0.4897 | 0.0000 | 0.2500 |
| **kg_rag** | 0.3859 | 0.5473 | 0.0000 | 0.0833 |

## 2. Targeted Reasoning Type Breakdown

| Reasoning Type | N | BM25 ROUGE-L | Vanilla ROUGE-L | KG-RAG ROUGE-L | KG vs Van Diff |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Type A: Direct fact lookup | 6 | 0.1821 | 0.2092 | 0.3564 | +0.1472 |
| Type B: One explicit relationship | 19 | 0.2684 | 0.2620 | 0.3322 | +0.0702 |
| Type C: Multiple related entities / multi-hop | 2 | 0.3264 | 0.2479 | 0.2475 | -0.0004 |
| Type D: Temporal relationship | 9 | 0.4065 | 0.4218 | 0.5497 | +0.1279 |

## 3. Paired Statistical Tests (KG-RAG vs Baselines)

- **ROUGE-L**:
  - KG-RAG vs Vanilla RAG: Mean Diff = +0.0935, 95% Bootstrap CI = (0.0288, 0.1695)
  - KG-RAG vs BM25: Mean Diff = +0.0941, 95% Bootstrap CI = (-0.0057, 0.1834)
- **Reference Token Coverage**:
  - KG-RAG vs Vanilla RAG: Mean Diff = +0.0577, 95% Bootstrap CI = (-0.0477, 0.1757)
  - KG-RAG vs BM25: Mean Diff = +0.0707, 95% Bootstrap CI = (-0.049, 0.185)

## 4. KG Path & Evidence Retrieval Analysis

- Total Aditya-L1 Questions: 36
- Questions matching KG entity nodes: 28 (77.8%)
- Questions retrieving non-empty serialized KG triples: 27 (75.0%)

## 5. Limitations & Scientific Caveats

1. **Sample Size**: N=36 provides an indicative benchmark for the newly added domain, but statistical power for fine-grained significance testing is limited.
2. **Lexical Proxy**: Evaluation relies on lexical overlap metrics (ROUGE-L and reference token coverage), which can penalize semantically valid answers phrased differently.
3. **Local LLM Ceiling**: When the KG supplies accurate triples, generation failures (hallucination or generic abstention by 7B quantized model) occasionally bottleneck end-to-end performance.
