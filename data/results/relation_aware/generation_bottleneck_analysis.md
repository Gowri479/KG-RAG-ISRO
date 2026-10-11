# Generation bottleneck diagnostic

## Key observation

KG-only substantially exceeds hybrid lexical scores, but these are automated lexical diagnostics rather than factual-quality evidence.

## Trace audit

- Audited 504 frozen rows and 60 KG-required questions.
- No answers were regenerated.
- Evidence composition fields are marked unavailable when a context block is absent.

## Current prompt audit

The current prompt requires context-only answering, but does not prioritize KG facts, define conflict resolution, preserve relation direction explicitly, or instruct provenance use.

## Metric sanity check

Reference answers are canonical triple objects, and KG-required paths point to those same canonical triples. This is legitimate for evidence grounding, but it creates direct lexical overlap between reference answers and KG evidence. It can inflate lexical diagnostics and does not replace human evaluation.

## Most-supported explanation

**F. MIXED/UNCERTAIN**

Prompt failure and evidence dilution are plausible. The frozen traces do not establish whether heterogeneous evidence conflicts semantically, and the benchmark/KG lexical relationship is a confound for ROUGE and coverage. A small prompt-only diagnostic is recommended.

Detailed composition, correlations, per-question gap examples, and prompt audit are in `generation_bottleneck_analysis.json` and `evidence_composition.csv`.

## Small prompt-only diagnostic

- Fixed subset: first 18 KG-required questions by question ID.
- Retrieval contexts: frozen Relation-Aware KG-RAG contexts.
- P1 is the current prompt; P2 adds KG priority; P3 separates structured evidence instructions.

| Variant | ROUGE-L mean | Coverage mean | IDK rate |
|---|---:|---:|---:|
| P1_current | 0.124566 | 0.298611 | 0.000000 |
| P2_kg_priority | 0.115307 | 0.293519 | 0.111111 |
| P3_structured_evidence | 0.133517 | 0.362037 | 0.000000 |

P3 was descriptively highest on this small diagnostic, while P2 was below P1. This is exploratory and is not a benchmark or evidence of factual-quality improvement. The complete paired statistics are in `prompt_diagnostic_results.json`.
