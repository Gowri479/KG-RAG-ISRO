# Final human-evaluation results

**Decision: B. PARTIAL**

Human scores were analyzed unchanged from the submitted completed CSV. No generation was rerun and no human scores were fabricated.

## System means

| System | Correctness | Completeness | Groundedness | Relevance | Unsupported claims |
|---|---:|---:|---:|---:|---:|
| Vanilla RAG | 2.115 | 2.154 | 3.346 | 3.154 | 0.385 |
| Corrected Structured KG-RAG | 2.962 | 2.962 | 3.500 | 3.615 | 0.385 |
| BM25 + LLM | 2.269 | 2.269 | 3.692 | 3.308 | 0.308 |

## Corrected KG-RAG minus Vanilla RAG

| Metric | Mean difference | Median | SD | 95% CI | Wilcoxon p | Holm p | Cohen dz | KG/Vanilla/Tie |
|---|---:|---:|---:|---|---:|---:|---:|---|
| correctness | 0.846 | 0.000 | 2.257 | [-0.038462, 1.730769] | 0.117116 | 0.434788 | 0.374853 | 8/2/16 |
| completeness | 0.808 | 0.000 | 2.209 | [-0.038462, 1.653846] | 0.151855 | 0.434788 | 0.365568 | 8/2/16 |
| groundedness | 0.154 | 0.000 | 1.974 | [-0.576923, 0.884615] | 0.67266 | 0.67266 | 0.077949 | 5/4/17 |
| relevance | 0.462 | 0.000 | 1.449 | [-0.115385, 1.038462] | 0.108697 | 0.434788 | 0.318609 | 9/4/13 |

## Comparison with the old KG-RAG result

The frozen Phase 5 old KG-RAG result on the same subset had correctness difference -0.154, groundedness difference -0.462, and unsupported-claim rates 0.462 (old KG-RAG) versus 0.346 (Vanilla). The corrected pipeline has correctness difference 0.846, groundedness difference 0.154, and equal unsupported-claim rates of 0.385 and 0.385. The direction therefore changed for groundedness but not for correctness.

## Automated versus human results

The corrected KG improved the preliminary lexical diagnostics over Vanilla (ROUGE-L 0.203808 vs 0.116641; reference-token coverage 0.501282 vs 0.292949). Human results agree only on groundedness direction: groundedness was higher for corrected KG, while correctness, completeness, and relevance were not higher. Lexical improvement therefore does not establish factual-quality improvement.

## Interpretation

The corrected KG system is evaluated here using human correctness and groundedness as the primary evidence. The result is B. PARTIAL: correctness, completeness, groundedness, and relevance were all higher descriptively, but confidence intervals were wide and no Holm-adjusted primary metric was significant. This is not a clear overall human-rated factual-quality advantage.

## Limitations

This is a 26-question researcher-constructed benchmark with one annotation per answer. Results should not be generalized beyond this controlled evaluation. One annotation was available per answer, so inter-annotator agreement cannot be estimated.

The previous Phase 5 human evaluation remains frozen and was not combined with these scores.
