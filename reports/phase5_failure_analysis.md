# Phase 5 — KG-RAG failure analysis

## 1. Objective

Diagnose the frozen Phase 4 KG-required evaluation using all 26 questions and saved traces only. No benchmark, corpus, model, prompt, retrieval setting, or prior result was changed.

## 2. Frozen benchmark verification

- Benchmark SHA-256: `7f5daf5419b0a4cb70e5c3d5eba021a8a39118b5a0fd235ce19b81f75324ffce` (expected hash matched)
- KG_REQUIRED=YES: 26
- Phase 4 system-question rows: 186
- All three systems represented: verified

## 3–6. Retrieval and pipeline diagnosis

| Diagnostic | Rate |
|---|---:|
| Complete KG path retrieved | 0.0 |
| Partial KG path retrieved | 0.0 |
| Required path not retrieved | 0.0 |
| KG path not determinable | 1.0 |
| Supporting excerpt found in saved text/context | 0.653846 |
| Supporting excerpt not found | 0.346154 |

The saved traces expose `retrieved_kg_triples` and `retrieved_kg_paths`, but the latter is empty and there are no source-linked canonical path IDs. Dense IDs also belong to the pilot index rather than the frozen corpus namespace. Therefore path retrieval is **not determinable**, not silently treated as zero.

## 7. Retrieval success versus answer success

A valid retrieval-success/answer-outcome 2×2 cannot be computed because complete KG path retrieval is not observable in the saved traces. Lexical overlap is retained only as a proxy and is not factual correctness.

## 8. Relation-type analysis

The machine-readable report contains category/relation breakdowns for all 26 questions. `MULTI_RELATION` is descriptive only because its subgroup is tiny.

## 9. KG-RAG versus Vanilla

All 26 case-by-case comparisons are in `failure_analysis.json`. They use ROUGE-L and coverage only as lexical proxies. They do not establish factual superiority or failure.

## 10. Context dilution

The traces record context word counts, dense evidence counts, KG triple counts, path counts, and duplicate/unique evidence counts. However, they do not support a controlled causal claim that augmentation caused dilution. **Context dilution is not established.**

## 11. Human evaluation

The 62-question, 3-system blinded package exists and contains 186 answer instances. Its scoring fields remain blank and status is `PENDING`.

## 12. Limitations

- No source-linked KG path IDs were saved.
- Query classification and relation matching fields are unavailable.
- Pilot-index chunk IDs cannot be compared directly to frozen-corpus chunk IDs.
- ROUGE-L/token coverage are lexical proxies, not correctness judgments.
- No human annotations are available.

## 13. Final decision

**D — NO CLEAR SINGLE BOTTLENECK**

The traces show a substantial instrumentation limitation and do not establish whether the dominant problem is retrieval, fusion/context, or generation/evidence use. A stronger causal diagnosis requires independently instrumented future runs, but no rerun or tuning is performed in Phase 5.

## Recommended next step

Complete independent human evaluation first. If further engineering is approved afterward, add trace instrumentation in a separate controlled phase before changing system behavior; do not reinterpret the current frozen results as a path-retrieval experiment.
