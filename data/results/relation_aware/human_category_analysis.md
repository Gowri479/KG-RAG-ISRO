# Final category-level human analysis

## Integrity verification

- 60 questions and 180 system-level evaluations verified.
- Three systems per question; no duplicate or missing question rows.
- Correctness, completeness, groundedness, and relevance scores are valid 1–5 values.
- Unsupported-claim flags are valid 0/1 values.
- Benchmark SHA-256: `6c7f3600995c0091de73d20d2f02953cb414023df281f7178069f9f81b432b64`

## Overall human results

| System | Correctness | Completeness | Groundedness | Relevance | Unsupported claims |
|---|---:|---:|---:|---:|---:|
| Vanilla Dense RAG | 3.067 | 2.733 | 3.867 | 4.150 | 0.200 |
| Corrected Structured KG-RAG | 2.983 | 2.683 | 3.850 | 4.083 | 0.217 |
| Relation-Aware KG-RAG | 2.783 | 2.633 | 3.583 | 3.967 | 0.350 |

## Category-level results

| Category | N | C Vanilla | C Relation | ΔC | Comp Vanilla | Comp Relation | ΔComp | Ground Vanilla | Ground Relation | ΔGround | Rel Vanilla | Rel Relation | ΔRel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DIRECT_CONTROL | 10 | 3.300 | 3.400 | 0.100 | 3.500 | 3.500 | 0.000 | 4.400 | 4.500 | 0.100 | 4.200 | 4.200 | 0.000 |
| MULTI_RELATION | 10 | 4.300 | 4.000 | -0.300 | 3.300 | 3.000 | -0.300 | 4.000 | 3.700 | -0.300 | 5.000 | 5.000 | 0.000 |
| SINGLE_RELATION | 30 | 2.533 | 2.433 | -0.100 | 2.367 | 2.333 | -0.033 | 3.633 | 3.733 | 0.100 | 3.800 | 3.733 | -0.067 |
| TWO_HOP_RELATION | 10 | 3.200 | 2.000 | -1.200 | 2.500 | 2.300 | -0.200 | 3.900 | 2.100 | -1.800 | 4.300 | 3.400 | -0.900 |

## Unsupported-claim rates

| Category | N | Vanilla | Corrected KG-RAG | Relation-Aware KG-RAG |
|---|---:|---:|---:|---:|
| DIRECT_CONTROL | 10 | 0.100 | 0.100 | 0.100 |
| MULTI_RELATION | 10 | 0.000 | 0.000 | 0.000 |
| SINGLE_RELATION | 30 | 0.300 | 0.367 | 0.367 |
| TWO_HOP_RELATION | 10 | 0.200 | 0.100 | 0.900 |

The Relation-Aware unsupported-claim rate is especially concentrated in TWO_HOP_RELATION questions.

## Task-conditional interaction

The relational-task versus DIRECT_CONTROL interaction is reported for each human dimension with bootstrap intervals, permutation p-values, and pooled effect sizes in `human_category_analysis.json`.

## Existing corrected KG-RAG comparison

Relation-Aware KG-RAG versus Existing Corrected Structured KG-RAG paired category statistics are included in `human_category_statistics.json`; the newer retriever is not assumed to be better.

## Automated versus human correlations

Correlations between frozen Relation-Aware ROUGE/coverage and human correctness, completeness, and groundedness are exploratory. They show positive associations with correctness/completeness but little association with groundedness and do not validate lexical metrics as factual measures.

Unsupported-claim rates, paired statistics, Holm-adjusted category tests, existing corrected KG-RAG comparisons, balanced question examples, and automated/human correlations are in `human_category_analysis.json` and `human_category_statistics.json`.

## Final scientific decision

**C. NOT SUPPORTED**

Relation-Aware KG-RAG is descriptively below Vanilla on all four overall human dimensions, and category/task-conditional analyses do not establish a consistent statistically credible improvement on intended relational categories.

## Recommendation

Do not present Relation-Aware KG-RAG as a demonstrated factual-quality improvement. If retained, describe the result as a negative or diagnostic finding and preserve the retrieval/provenance contribution separately from end-to-end answer quality.

No paper file or frozen experiment artifact was modified.
