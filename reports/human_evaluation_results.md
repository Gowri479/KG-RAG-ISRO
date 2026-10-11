# Phase 5C — Completed blinded human evaluation

## 1. Verification

- Evaluation scope: **FULL_186**
- Questions: 62; scored answer instances: 186
- Benchmark SHA-256: `7f5daf5419b0a4cb70e5c3d5eba021a8a39118b5a0fd235ce19b81f75324ffce`
- Missing scores: 0
- Submitted scores were preserved unchanged.

## 2. Actual scoring scale

The supplied CSV uses a **1–5 ordinal scale** for correctness, completeness, groundedness, and relevance, plus 0/1 unsupported-claim values. This is a deviation from the originally proposed 0–2 rubric. No score conversion was performed.

## 3. Per-question blinded mapping

The original package uses a deterministic per-question shuffle with seed `20261007`; there is no single global A/B/C mapping. The exact mapping for every question is stored in `human_evaluation_analysis.json`.

| System | Correctness mean | Completeness mean | Groundedness mean | Relevance mean | Unsupported-claim rate |
|---|---:|---:|---:|---:|---:|
| BM25 + LLM | 2.385 | 2.385 | 3.615 | 3.346 | 0.346 |
| Vanilla RAG | 2.115 | 2.154 | 3.615 | 3.308 | 0.346 |
| KG-RAG | 1.962 | 2.000 | 3.154 | 3.115 | 0.462 |

## 4. Primary KG-RAG versus Vanilla RAG comparison

| Dimension | Mean diff | Median diff | + / − / tie | Bootstrap 95% CI | Wilcoxon p | Cohen dz |
|---|---:|---:|---:|---|---:|---:|
| correctness | -0.154 | 0.000 | 0 / 1 / 25 | [-0.461538, 0.0] | None | None |
| completeness | -0.154 | 0.000 | 0 / 1 / 25 | [-0.461538, 0.0] | None | None |
| groundedness | -0.462 | 0.000 | 0 / 3 / 23 | [-0.923077, 0.0] | 0.083265 | -0.354144 |
| relevance | -0.192 | 0.000 | 0 / 4 / 22 | [-0.384615, -0.038462] | 0.058782 | -0.391294 |

Unsupported-claim rate on YES: KG-RAG **0.462**; Vanilla RAG **0.346**.

## 5. Human summaries

Full per-system summaries for all 62 questions, YES=26, and NO=36 are stored in the JSON artifact. The supplied score dimensions are reported on their original 1–5 scale.

## 6. Human versus lexical results

Phase 4 ROUGE-L on YES had KG-RAG minus Vanilla mean difference −0.018373, CI [−0.068196, 0.010827], and Wilcoxon p=1.0. The human analysis is more directly relevant to factual quality, but it is a single-annotator, researcher-constructed evaluation. Agreement or contradiction with ROUGE must therefore be described cautiously rather than treating ROUGE as factual correctness.

## 7. Methodological caveats

- Primary N=26 questions.
- One completed annotation file is available; inter-annotator agreement is not calculated.
- The benchmark is researcher-constructed, not independent external validation.
- Unsupported-claim judgments are human judgments under the supplied rubric.
- A/B/C system labels are shuffled per question; the full mapping is recorded in the JSON.

## 8. Conclusion

**B — NO** — based on the submitted human scores and the predeclared paired comparisons. This conclusion is limited to the completed single-annotator evaluation and does not justify universal claims about KG-RAG.

The paper was not modified. Whether to retain the KG-RAG contribution should be decided after considering these human results together with the Phase 4 lexical results and the documented trace limitations.
