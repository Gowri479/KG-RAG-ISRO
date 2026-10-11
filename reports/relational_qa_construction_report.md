# Relational QA v1 construction report

## Scope and freeze gate

This is the construction/validation stage of Phase 3. It creates a separate
researcher-annotated benchmark from the frozen Phase 1 corpus and KG. No
system answers, retrieval traces, lexical scores, prior benchmark outputs, or
human evaluation results were consulted during question construction. No model
was run and no prompt, retriever, model configuration, corpus, paper, or
existing benchmark was modified.

The primary benchmark is answerable-only. Temporal reasoning is not a family:
one incidental item follows the single explicit `PRECEDED_BY` edge from
Chandrayaan-3 to Chandrayaan-2.

## Benchmark inventory

- Benchmark: `data/relational_benchmark/relational_qa_v1.json`
- Questions: **62**
- Provenance rows: **62**
- `KG_REQUIRED=YES`: **26**
- `KG_REQUIRED=NO`: **36**
- Categories: **DIRECT_CONTROL 15**, **SINGLE_RELATION 21**,
  **TWO_HOP_RELATION 23**, **MULTI_RELATION 3**
- Matched pairs: **15** two-item groups, each containing one YES and one NO
- Annotation agreement: not available; single-researcher construction
- Benchmark SHA-256:
  `7f5daf5419b0a4cb70e5c3d5eba021a8a39118b5a0fd235ce19b81f75324ffce`

## Mission distribution

| Mission | Questions |
|---|---:|
| Aditya-L1 | 13 |
| AstroSat | 4 |
| Chandrayaan-1 | 15 |
| Chandrayaan-2 | 7 |
| Chandrayaan-3 | 16 |
| Gaganyaan | 2 |
| Mars Orbiter Mission | 5 |

The distribution is intentionally not uniform: it follows the available
provenance-backed facts and avoids inventing coverage for Gaganyaan or other
missions.

## Relation distribution

Counts below count relation labels attached to questions; multi-relation
questions can contribute more than one label.

| Relation | Question labels |
|---|---:|
| HAS_PAYLOAD | 32 |
| HAS_OBJECTIVE | 19 |
| OBSERVES | 11 |
| DEVELOPED_BY | 14 |
| LAUNCHED_BY | 6 |
| LAUNCHED_ON | 6 |
| LED_BY | 1 |
| OPERATED_BY | 1 |
| PRECEDED_BY | 1 |

The benchmark contains 26 YES items based on connected multi-edge paths:
23 two-edge questions and 3 multi-relation questions. All path edges have
provenance and are connected by canonical entity IDs. Source chunks are
checked exactly; the validator found zero invalid paths or provenance errors.

## Construction protocol

For each item, the supporting triple/path was selected first, then the
question was paraphrased and the answer/provenance metadata were recorded.
`DIRECT_CONTROL` and `SINGLE_RELATION` items are controls with
`KG_REQUIRED=NO`. `TWO_HOP_RELATION` and `MULTI_RELATION` items require
joining explicit KG edges and have `KG_REQUIRED=YES`.

The benchmark excludes one otherwise plausible Chandrayaan-2 OHRC item
because the frozen corpus's stored supporting excerpt does not occur in its
cited chunk. The corpus was not repaired or altered for this benchmark.

## Validation and tests

- Relational benchmark validator: **passed**
- Questions/unique IDs: **62/62**
- Provenance rows: **62**
- Validation errors: **0**
- Full pytest suite: **27 passed**
- `git diff --check`: **passed**
- Canonical benchmark/results/paper protected-path diff: **no changes**

## Evaluation status

No BM25+LLM, Vanilla RAG, or KG-RAG evaluation has been run. No retrieval
instrumentation, human factual evaluation, statistical testing, or final
support/no-support decision is appropriate until this frozen artifact has
been reviewed and approved. This construction report must not be interpreted
as evidence that KG augmentation improves QA.
