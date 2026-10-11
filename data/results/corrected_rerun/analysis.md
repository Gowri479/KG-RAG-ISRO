# Corrected-rerun analysis (pre-registered)

Raters: 1. Integrity: {'rows': 216, 'rows_failing_integrity': 0, 'rows_trimmed_by_window_guard': 0, 'retried_rows': 0}.
Agreement (Krippendorff's alpha): single rater; not reported

## Primary: human correctness, 60 KG-required items (Holm over 2 tests)

| Comparison | Mean diff | 95% CI | Median | dz | W/L/T | Wilcoxon p | Holm p | Decision |
|---|---|---|---|---|---|---|---|---|
| C_TWO_STAGE_vs_V_VANILLA | +0.933 | [0.417, 1.433] | +0.500 | 0.450064 | 30/11/19 | 0.0011 | 0.0011 | SUPPORTED |
| A_CURRENT_vs_V_VANILLA | +1.167 | [0.717, 1.617] | +0.000 | 0.649554 | 29/6/25 | 0.0000 | 0.0000 | SUPPORTED |

## Means (primary set)

| System | correctness | completeness | groundedness | relevance | unsupported |
|---|---|---|---|---|---|
| A_CURRENT | 4.383 | 4.333 | 4.417 | 4.633 | 16.7% |
| C_TWO_STAGE | 4.150 | 4.117 | 4.100 | 4.400 | 28.3% |
| V_VANILLA | 3.217 | 3.117 | 3.767 | 3.983 | 43.3% |

Secondary, sensitivity and automated results are exploratory; see analysis.json.
