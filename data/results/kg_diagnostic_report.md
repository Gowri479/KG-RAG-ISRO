# KG diagnostic report

- Questions: 26; generations: 130

| Diagnostic | Rate |
|---|---:|
| path_exists | 1.000000 |
| retrieved | 0.961538 |
| provenance_available | 0.884615 |
| in_final_context | 0.961538 |
| answer_correct_lexical_proxy | 0.615385 |

## Failure categories

```json
{
  "RETRIEVAL_SUCCESS": 13,
  "GENERATION_FAILURE": 10,
  "PROVENANCE_FAILURE": 2,
  "RETRIEVAL_FAILURE": 1
}
```

## Variant results

| Variant | ROUGE-L | Coverage | IDK rate | Context tokens |
|---|---:|---:|---:|---:|
| DENSE_ONLY | 0.116553 | 0.292949 | 0.076923 | 1218.5 |
| KG_ONLY | 0.343512 | 0.871795 | 0.000000 | 266.3 |
| KG_PLUS_SOURCE | 0.170786 | 0.496154 | 0.000000 | 1325.3 |
| DENSE_PLUS_KG_STRUCTURED | 0.197776 | 0.498077 | 0.000000 | 1500.0 |
| DENSE_PLUS_KG_UNSTRUCTURED | 0.101179 | 0.257692 | 0.038462 | 1249.8 |

## Dense-only versus structured KG

- rouge_l: mean difference 0.081223; median 0.000000; CI [0.007223, 0.162266]; Wilcoxon 0.0867637960488504; Cohen dz 0.39096733355552626; +/−/ties 8/5/13.
- reference_token_coverage: mean difference 0.205128; median 0.000000; CI [0.022436, 0.394231]; Wilcoxon 0.03868271133845152; Cohen dz 0.41198538411195784; +/−/ties 8/3/15.

## Per-question diagnostic table

| Question | KG path exists | Retrieved | Provenance | In final context | Answer correct (lexical proxy) | Failure category |
|---|---|---|---|---|---|---|
| rqa_037 | True | True | True | True | True | RETRIEVAL_SUCCESS |
| rqa_038 | True | True | True | True | True | RETRIEVAL_SUCCESS |
| rqa_039 | True | True | True | True | True | RETRIEVAL_SUCCESS |
| rqa_040 | True | True | True | True | False | GENERATION_FAILURE |
| rqa_041 | True | True | True | True | False | GENERATION_FAILURE |
| rqa_042 | True | True | True | True | False | GENERATION_FAILURE |
| rqa_043 | True | True | True | True | True | RETRIEVAL_SUCCESS |
| rqa_044 | True | True | True | True | False | GENERATION_FAILURE |
| rqa_045 | True | True | True | True | True | RETRIEVAL_SUCCESS |
| rqa_046 | True | True | True | True | False | GENERATION_FAILURE |
| rqa_047 | True | True | True | True | True | RETRIEVAL_SUCCESS |
| rqa_048 | True | True | True | True | True | RETRIEVAL_SUCCESS |
| rqa_049 | True | True | True | True | True | RETRIEVAL_SUCCESS |
| rqa_050 | True | True | True | True | True | RETRIEVAL_SUCCESS |
| rqa_051 | True | True | True | True | False | GENERATION_FAILURE |
| rqa_052 | True | True | True | True | True | RETRIEVAL_SUCCESS |
| rqa_053 | True | True | True | True | False | GENERATION_FAILURE |
| rqa_054 | True | True | True | True | True | RETRIEVAL_SUCCESS |
| rqa_055 | True | True | True | True | False | GENERATION_FAILURE |
| rqa_056 | True | True | True | True | False | GENERATION_FAILURE |
| rqa_057 | True | True | True | True | True | RETRIEVAL_SUCCESS |
| rqa_058 | True | True | True | True | False | GENERATION_FAILURE |
| rqa_059 | True | True | True | True | True | RETRIEVAL_SUCCESS |
| rqa_060 | True | True | False | True | True | PROVENANCE_FAILURE |
| rqa_061 | True | False | False | False | True | RETRIEVAL_FAILURE |
| rqa_062 | True | True | False | True | True | PROVENANCE_FAILURE |

## Decision

**A. TECHNICAL_FAILURE_IDENTIFIED**

The original pipeline queried a noisy pilot graph, used one-hop surface-form lookup, and discarded source-linked path identifiers. In contrast, the isolated canonical pipeline found 25/26 required paths, retained provenance for 23/26 selected required paths, and placed 25/26 required paths into the structured final context. Dense-plus-structured KG improved lexical ROUGE-L by 0.081223 and reference-token coverage by 0.205128 versus Dense-only. This identifies a technical pipeline failure, but does not establish a final human-factual advantage; the original negative Phase 4 result remains unchanged.
