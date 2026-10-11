# Frozen relational QA evaluation

## 1. Objective

Evaluate the frozen 62-question relational benchmark using the existing BM25 + LLM, Vanilla RAG, and KG-RAG paths without tuning or benchmark changes.

## 2. Frozen benchmark verification

- Benchmark SHA-256: `7f5daf5419b0a4cb70e5c3d5eba021a8a39118b5a0fd235ce19b81f75324ffce`
- Questions: 62; system-question evaluations: 186
- Freeze/count verification: passed before evaluation.

## 3–4. Configuration and systems

- Model: `mistral:7b-instruct-q4_K_M`; options: `{'num_predict': 150, 'temperature': 0.1, 'num_ctx': 2048}`.
- Systems: BM25 + LLM, Vanilla RAG, KG-RAG.
- Existing retrieval ranking, top-k, one-hop KG expansion, prompt, and context budget were preserved.

## 5. Overall results

| System | N | ROUGE-L | Coverage | Exact match | IDK rate |
|---|---:|---:|---:|---:|---:|
| BM25 + LLM | 62 | 0.133216 | 0.34735 | 0.0 | 0.0 |
| Vanilla RAG | 62 | 0.107625 | 0.296838 | 0.0 | 0.129032 |
| KG-RAG | 62 | 0.100404 | 0.283781 | 0.0 | 0.096774 |

## 6. Primary KG_REQUIRED=YES result

`kg_rag` minus `vanilla_rag`: mean difference **-0.018373**, median **0.0**, 95% bootstrap CI **[-0.068196, 0.010827]**, Wilcoxon p **1.0**, Cohen's dz **-0.152782**, positive/negative/ties **2/2/22**.

## 7–8. Secondary and BM25 comparisons

KG_REQUIRED=NO: mean KG-RAG minus Vanilla difference **0.000832**, CI **[-0.005052, 0.008245]**, Wilcoxon p **0.9527650219907529**.
KG_REQUIRED=YES: KG-RAG minus BM25: **-0.030805**; Vanilla minus BM25: **-0.012432**.

## 9. Retrieval-level results

Retrieval metrics are trace-level diagnostics, not answer factuality metrics. Complete path hits are conservative because the existing KG retriever exposes one-hop serialized triples but no source-linked path IDs.

```json
{
  "bm25_llm": {
    "n": 62,
    "supporting_chunk_hit_rate": 0,
    "supporting_triple_hit_rate": 0,
    "complete_supporting_path_hit_rate": 0,
    "path_metric_note": "Existing KG retriever exposes serialized one-hop triples but no source-linked path IDs; complete path hits are conservatively recorded as false, never inferred."
  },
  "vanilla_rag": {
    "n": 62,
    "supporting_chunk_hit_rate": 0,
    "supporting_triple_hit_rate": 0,
    "complete_supporting_path_hit_rate": 0,
    "path_metric_note": "Existing KG retriever exposes serialized one-hop triples but no source-linked path IDs; complete path hits are conservatively recorded as false, never inferred."
  },
  "kg_rag": {
    "n": 62,
    "supporting_chunk_hit_rate": 0.483871,
    "supporting_triple_hit_rate": 0,
    "complete_supporting_path_hit_rate": 0,
    "path_metric_note": "Existing KG retriever exposes serialized one-hop triples but no source-linked path IDs; complete path hits are conservatively recorded as false, never inferred."
  }
}
```

## 10. Human evaluation

An annotation-ready blinded package is generated separately and remains `PENDING`; no human scores are fabricated.

## 11–14. Subgroups, relations, matched controls, and errors

Category/relation/match-group results are stored in the machine-readable aggregate. Small MULTI_RELATION groups are descriptive only. Error examples follow the predeclared paired-score extreme selection rule and do not establish causal failure categories automatically.

## 15–18. Statistical interpretation and limitations

ROUGE-L, coverage, exact match, and IDK are lexical/output metrics and are not factuality judgments. Retrieval traces are limited by the existing APIs; no ranking or generation behavior was changed to create them. Human factual evaluation is pending. The benchmark has 26 primary items, but uncertainty and retrieval instrumentation limitations must be considered.

## Final decision

Primary result: mean KG-RAG minus Vanilla difference **-0.018373** with CI **[-0.068196, 0.010827]** and p **1.0**. The final gate is **C. INSUFFICIENT EVIDENCE** under the predefined rules; this is not a universal RAG conclusion.
