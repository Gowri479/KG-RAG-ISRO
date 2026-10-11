# Held-out human evaluation analysis

## Scope and integrity

- 13 held-out questions and 39 candidate evaluations were decoded through the private mapping.
- No generation, retrieval, prompt, benchmark, or annotation step was rerun.
- Input CSV SHA-256: `df1df327c4e0ac84a3736a2e2e01c2458482086dab790743fbda1ed5e65dde3a`

## Human mean scores

| System | Correctness | Completeness | Groundedness | Relevance | Unsupported claims |
|---|---:|---:|---:|---:|---:|
| Vanilla Dense RAG | 3.308 | 2.846 | 3.846 | 4.231 | 15.4% |
| Current Relation-Aware KG-RAG | 2.615 | 2.231 | 3.538 | 3.769 | 30.8% |
| Two-Stage KG-Grounded Generation | 3.077 | 2.692 | 3.308 | 4.231 | 30.8% |

## Paired results

### C_TWO_STAGE_minus_A_VANILLA

| Dimension | Mean Δ | Median Δ | Bootstrap 95% CI | Wilcoxon p | Holm p | dz | W/L/T |
|---|---:|---:|---|---:|---:|---:|---|
| correctness | -0.231 | 0.000 | [-1.231, 0.769] | 0.8750 | 1.0000 | -0.123 | 2/3/8 |
| completeness | -0.154 | 0.000 | [-1.077, 0.769] | 0.8750 | 1.0000 | -0.087 | 2/3/8 |
| groundedness | -0.538 | 0.000 | [-1.308, 0.077] | 0.3750 | 1.0000 | -0.387 | 1/3/9 |
| relevance | 0.000 | 0.000 | [-0.615, 0.615] | 1.0000 | 1.0000 | 0.000 | 2/2/9 |

Unsupported claims: C rate 0.308, comparator rate 0.154; difference 15.4 percentage points; C improves/worsens/unchanged = 0/2/11.

### C_TWO_STAGE_minus_B_RELATION_AWARE

| Dimension | Mean Δ | Median Δ | Bootstrap 95% CI | Wilcoxon p | Holm p | dz | W/L/T |
|---|---:|---:|---|---:|---:|---:|---|
| correctness | 0.462 | 0.000 | [0.000, 1.231] | 0.5000 | 1.0000 | 0.385 | 2/0/11 |
| completeness | 0.462 | 0.000 | [0.000, 1.231] | 0.5000 | 1.0000 | 0.385 | 2/0/11 |
| groundedness | -0.231 | 0.000 | [-1.000, 0.308] | 0.7500 | 1.0000 | -0.177 | 1/2/10 |
| relevance | 0.462 | 0.000 | [0.000, 0.923] | 0.2500 | 1.0000 | 0.526 | 3/0/10 |

Unsupported claims: C rate 0.308, comparator rate 0.308; difference 0.0 percentage points; C improves/worsens/unchanged = 1/1/11.

## Category observations

### DIRECT_CONTROL (N=2)
Descriptive only; category sample sizes are too small for reliable inference.
- Vanilla Dense RAG: correctness 5.000, completeness 5.000, groundedness 5.000, relevance 5.000, unsupported 0.0%
- Current Relation-Aware KG-RAG: correctness 3.000, completeness 3.000, groundedness 5.000, relevance 4.000, unsupported 0.0%
- Two-Stage KG-Grounded Generation: correctness 3.000, completeness 3.000, groundedness 5.000, relevance 4.000, unsupported 0.0%

### MULTI_RELATION (N=4)
Descriptive only; category sample sizes are too small for reliable inference.
- Vanilla Dense RAG: correctness 4.250, completeness 3.250, groundedness 4.000, relevance 5.000, unsupported 0.0%
- Current Relation-Aware KG-RAG: correctness 4.000, completeness 3.000, groundedness 3.750, relevance 5.000, unsupported 0.0%
- Two-Stage KG-Grounded Generation: correctness 4.000, completeness 3.000, groundedness 3.750, relevance 5.000, unsupported 0.0%

### SINGLE_RELATION (N=6)
Descriptive only; category sample sizes are too small for reliable inference.
- Vanilla Dense RAG: correctness 2.000, completeness 1.833, groundedness 3.333, relevance 3.333, unsupported 33.3%
- Current Relation-Aware KG-RAG: correctness 1.833, completeness 1.667, groundedness 3.167, relevance 3.000, unsupported 50.0%
- Two-Stage KG-Grounded Generation: correctness 2.833, completeness 2.667, groundedness 2.833, relevance 4.000, unsupported 50.0%

### TWO_HOP_RELATION (N=1)
Descriptive only; category sample sizes are too small for reliable inference.
- Vanilla Dense RAG: correctness 4.000, completeness 3.000, groundedness 4.000, relevance 5.000, unsupported 0.0%
- Current Relation-Aware KG-RAG: correctness 1.000, completeness 1.000, groundedness 2.000, relevance 3.000, unsupported 100.0%
- Two-Stage KG-Grounded Generation: correctness 1.000, completeness 1.000, groundedness 1.000, relevance 3.000, unsupported 100.0%

## Automated versus human results

C improved held-out ROUGE-L and coverage over A descriptively, but the automated Wilcoxon p-values were 0.097656 and 0.125. The human comparison must not be interpreted as validated merely because lexical metrics improved.

## Decision

**NOT SUPPORTED**

The primary C-versus-A comparison is descriptively worse on correctness, completeness, and groundedness, tied on relevance, and has a higher unsupported-claim rate. The C-versus-B comparison is mixed and not statistically conclusive. The two-stage method therefore does not demonstrate improved human-rated factual quality.

## Paper implication

Do not replace the proposed final method solely on this human evaluation. The two-stage architecture may be presented as a promising generation-bottleneck direction, with limitations and without a superiority claim. `paper/main.tex` was not modified.
