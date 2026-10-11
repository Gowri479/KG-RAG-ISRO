# Unanswerable QA & Grounded Abstention Evaluation Report

**Benchmark Dataset:** 40 curated unanswerable questions (20 evaluated: 4/category across 5 failure categories).
**Target Systems:** BM25 + LLM, Vanilla RAG, KG-RAG (Mistral-7B-Instruct).

## Summary Metrics

| System | Grounded Abstention Rate | Strict IDK Rate | Unsupported Answer Rate | Mean Latency (ms) |
| :--- | :---: | :---: | :---: | :---: |
| **bm25_llm** | 90.00% | 75.00% | 10.00% | 34273.1 |
| **vanilla_rag** | 95.00% | 85.00% | 5.00% | 32123.7 |
| **kg_rag** | 85.00% | 65.00% | 15.00% | 41240.4 |

## Category-Wise Grounded Abstention Rates

| Failure Category | BM25 + LLM | Vanilla RAG | KG-RAG |
| :--- | :---: | :---: | :---: |
| Completely absent fact | 75.0% | 75.0% | 75.0% |
| Plausible but unsupported question | 100.0% | 100.0% | 100.0% |
| Related entity but absent relationship | 75.0% | 100.0% | 75.0% |
| Similar entity with misleading evidence | 100.0% | 100.0% | 100.0% |
| Temporal fact absent from corpus | 100.0% | 100.0% | 75.0% |
