# Relation-aware human evaluation analysis

- Questions: 60
- Candidate annotations: 180
- Submitted CSV SHA-256: `f9f9173e64dfc45c955f280c6cfd4e96ccfc3127cecdaa9ecd271b9863164769`

## System means and medians

| System | Correctness mean | Completeness mean | Groundedness mean | Relevance mean | Unsupported claim rate |
|---|---:|---:|---:|---:|---:|
| Vanilla Dense RAG | 3.067 | 2.733 | 3.867 | 4.150 | 0.200 |
| Corrected Structured KG-RAG | 2.983 | 2.683 | 3.850 | 4.083 | 0.217 |
| Relation-Aware KG-RAG | 2.783 | 2.633 | 3.583 | 3.967 | 0.350 |

## Pairwise tests

Paired differences, bootstrap intervals, Wilcoxon tests, wins/losses/ties, and Holm-adjusted p-values are in `human_evaluation_analysis.json`.

These results are from human annotation, but interpretation remains conservative until category-level and task-conditional review is complete.
