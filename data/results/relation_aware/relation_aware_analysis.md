# Relation-aware experiment analysis

- Frozen generation rows: 504 (72 questions x 7 systems)
- Benchmark SHA-256: `6c7f3600995c0091de73d20d2f02953cb414023df281f7178069f9f81b432b64`
- No human evaluation was performed. Automated metrics are lexical diagnostics only.

## System-level lexical diagnostics

| System | ROUGE-L mean | Coverage mean | Exact match | IDK rate |
|---|---:|---:|---:|---:|
| vanilla_dense_rag | 0.098672 | 0.271528 | 0.000000 | 0.069444 |
| corrected_structured_kg_rag | 0.093931 | 0.280556 | 0.000000 | 0.069444 |
| relation_aware_kg_rag | 0.123176 | 0.315075 | 0.000000 | 0.027778 |
| relation_aware_kg_only | 0.417537 | 0.810747 | 0.069444 | 0.013889 |
| relation_aware_kg_source | 0.128640 | 0.316117 | 0.000000 | 0.027778 |
| relation_aware_kg_dense | 0.134232 | 0.340654 | 0.000000 | 0.000000 |
| relation_aware_kg_dense_bm25 | 0.089770 | 0.291285 | 0.000000 | 0.000000 |

## Conclusion

**PARTIAL**

The generation package is structurally valid and the relation-aware traces recover the required paths with preserved provenance. The automated results are descriptive lexical diagnostics; they cannot establish factual-quality improvement. Human evaluation is required before any factual-quality conclusion.

## Primary comparison

Relation-Aware KG-RAG minus Vanilla Dense RAG is reported separately for KG-required and KG-not-required questions, including paired bootstrap intervals, Wilcoxon tests, wins/losses/ties, and Holm-adjusted p-values in `statistical_results.json`.

The difference-in-differences test asks whether the required-subset paired improvement exceeds the control-subset paired improvement; its bootstrap interval and permutation p-value are recorded in `statistical_results.json`.

## Categories, relations, and ablation

Category-level results, supported-relation results, all A-G pairwise lexical contrasts, and context/path/provenance summaries are recorded in `relation_aware_analysis.json`.

## Context and evidence associations

Spearman associations between context tokens, retrieved path count, provenance count, and lexical metrics are descriptive only; they do not establish that larger context or more paths cause better answers.

## Retrieval diagnostics

- Required-path recovery: 1.000
- Valid provenance: 1.000
- Relation-intent set-match rate: 0.967
- Nonempty entity detection rate: 1.000

Detailed system, category, relation, ablation, statistical, correlation, and trace-supported failure results are in `relation_aware_analysis.json` and `question_level_results.csv`.
