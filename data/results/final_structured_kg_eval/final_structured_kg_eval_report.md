# Final structured KG human-quality validation

## Experimental objective

This preregistered validation compares vanilla dense RAG, corrected
provenance-preserving structured KG-RAG, and BM25 + LLM on the frozen
26-question `kg_required=YES` subset. It tests whether corrected structured
KG evidence improves human-rated factual quality. No human scores have been
assigned or analyzed.

## Frozen identity and protocol

- Benchmark SHA-256: `7f5daf5419b0a4cb70e5c3d5eba021a8a39118b5a0fd235ce19b81f75324ffce`
- Questions: 26
- Generations: 78
- Model: `mistral:7b-instruct-q4_K_M`
- Options: `{"num_ctx": 2048, "num_predict": 150, "temperature": 0.1}`
- Blind randomization seed: `20261007` (recorded only in the manifest)

## Systems

1. Vanilla RAG
2. Corrected Structured KG-RAG
3. BM25 + LLM

## Preliminary lexical diagnostics

These descriptive metrics are not factual-quality conclusions.

| System | ROUGE-L | Reference-token coverage | Exact match | IDK rate |
|---|---:|---:|---:|---:|
| vanilla_rag | 0.116641 | 0.292949 | 0.000000 | 0.076923 |
| corrected_structured_kg_rag | 0.203808 | 0.501282 | 0.000000 | 0.000000 |
| bm25_llm | 0.111743 | 0.280128 | 0.000000 | 0.000000 |

## Corrected KG retrieval and provenance

- Required paths handled: 25/26 (0.961538)
- Selected paths with provenance: 26/26 (1.000000)

## Files created

- `data/results/final_structured_kg_eval/generation_results.jsonl`
- `data/results/final_structured_kg_eval/preliminary_metrics.json`
- `data/results/final_structured_kg_eval/run_manifest.json`
- `data/results/final_structured_kg_eval/traces/`
- `data/annotations/final_structured_kg_human_eval.csv`
- `data/annotations/final_structured_kg_human_eval_guidelines.md`
- `data/results/final_structured_kg_eval/final_structured_kg_eval_report.md`

Previous benchmark, corpus, Phase 4, Phase 5, Phase 6, and paper artifacts
were not overwritten by this experiment.

**Human factual-quality conclusions are pending blinded annotation.**
