# Relational QA v1: construction and annotation guidelines

## Purpose and scope

This benchmark is a separate, evidence-first evaluation set for relationship-intensive QA over the frozen Phase 1 ISRO corpus and KG. It does not replace or modify the canonical 180-question benchmark or the separate 36-question Aditya-L1 benchmark. No question was chosen, rewritten, or retained using system answers, retrieval traces, lexical scores, or other evaluation outcomes.

The benchmark is stored under `data/relational_benchmark/`, rather than inside the frozen `data/benchmark/` directory. Its fact basis is the current provenance-linked KG and its authoritative source chunks. A source fact is usable only when the cited triple, source document, chunk, URL, and supporting excerpt agree. The OHRC payload annotation was excluded from the benchmark because its stored excerpt does not occur in its cited chunk; the frozen corpus was not edited.

## Categories

- **DIRECT_CONTROL**: an ordinary factual answer directly stated in a single source chunk, with no graph traversal needed.
- **SINGLE_RELATION**: one controlled KG relation supports the answer. The answer can be retrieved from a single source chunk, so `KG_REQUIRED=NO`.
- **TWO_HOP_RELATION**: a two-edge, connected path is necessary to resolve the relation asked in the question. Every edge has provenance, and the path's edges are not all stated in one source chunk.
- **MULTI_RELATION**: the question combines multiple connected facts (including a three-edge chain or multiple branches from one payload). Each required fact has provenance; no complete answer is present in one cited chunk.

## KG_REQUIRED decision

- **YES** only where the question's information requirement calls for following/joining two or more explicit, connected KG edges. Every edge must be in the supporting path(s), and the evidence must not collapse to one source chunk that states the whole queried relation.
- **NO** where one source chunk states the answer directly, even if the KG also represents that fact.

Labels are assigned from the question and evidence structure only, never from model performance. A question does not become KG-required merely because graph retrieval could help. These are operational task labels, not claims that a language model could never infer the answer from one retrieved passage or prior knowledge.

## Construction protocol

For each item, the researcher selected provenance-backed KG triple(s)/path(s) first, checked the source chunk and excerpt, then wrote a paraphrased question, reference answer, and acceptable answer forms. The item records its controlled relation type(s), mission, category, KG-required label, match group (when applicable), supporting triples and paths, source document(s), URLs, chunks, reasoning requirement, and answerability.

Questions must not quote a source sentence as a question or include the reference answer verbatim in the question. Answer variants are limited to canonical names and source-supported aliases; the evaluator should accept a semantically equivalent sourced answer, not an unsupported expansion. A named mission's payload question asks for one payload, not an exhaustive list. Direct-answer controls are intentionally paired with relational questions when topic and evidence allow it; `match_group_id` identifies these pairs. Matched pairs are not additional independent facts.

Temporal reasoning is not a benchmark family. There is one incidental three-edge question using the sole explicit `PRECEDED_BY` edge, alongside payload and objective edges. No order is inferred from dates.

## Provenance and answerability

Every primary-set item is `ANSWERABLE`. A multi-edge item is valid only when each edge has provenance and the cited edges connect through canonical entity IDs. Source excerpts must be found in the exact cited corpus chunk. Source documents and URLs must match the document registry. Corpus absence is not evidence that a fact is false, so no unanswerable item is included in this primary set.

## Review, ambiguity, and limitations

This is single-researcher construction. No independent second reviewer was available for this run; therefore there is no inter-annotator agreement statistic and no Cohen's kappa claim. A future independent review should check category, `KG_REQUIRED`, reference-answer validity, and provenance validity before model evaluation or publication. Resolve disagreements against the frozen source chunk, not system outputs.

Some scientific relation descriptions are paraphrases of source-stated objectives/observations. They must not be read as claims that a historical discovery occurred. The benchmark is deliberately limited by current KG coverage, uneven mission documentation, and a small number of multi-hop motifs. The curated QA set establishes a transparent test condition, not evidence that KG-RAG is useful.

## Freeze and evaluation separation

The benchmark validator checks IDs, schema, categories, controlled relations, missions, cited triples and paths, provenance, exact source chunks/excerpts, duplicate wording, match groups, answer fields, and the KG-required/category constraints. A SHA-256 record is generated only after validation. The benchmark and provenance JSONL must remain unchanged after freeze. System answers are to be generated only after the freeze record exists; no post-evaluation question edits are permitted.
