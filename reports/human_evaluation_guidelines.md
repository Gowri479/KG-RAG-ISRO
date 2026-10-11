# Human evaluation guidelines

## Purpose

Evaluate the 62 frozen relational QA questions using the blinded systems in
`data/results/relational_qa_v1/human_evaluation_blinded.csv`. Do not use
ROUGE-L or token overlap as a substitute for factual judgment.

## Scoring

Score each anonymized answer independently:

- **Correctness (0–2):** 0 = incorrect or unsupported answer; 1 = partially
  correct or materially incomplete; 2 = correct answer.
- **Completeness (0–2):** 0 = required relationship is absent; 1 = some
  required entities/edges are present; 2 = all requested relationship parts
  are answered.
- **Groundedness (0–2):** 0 = answer is not supported by the supplied
  evidence; 1 = evidence supports only part of the answer; 2 = answer is
  supported by the supplied evidence.
- **Relevance (0–2):** 0 = does not answer the question; 1 = partly relevant
  or includes substantial irrelevant material; 2 = directly answers the
  question.
- **Unsupported claim:** `YES` if the answer asserts a claim not supported by
  the supplied evidence, otherwise `NO`.

Use the frozen benchmark reference answer and the displayed evidence, while
allowing harmless formatting, date, acronym, and alias variants. For
multi-hop questions, correctness requires the requested end answer and
completeness requires the relevant relationship chain to be represented.

Leave a score blank if the answer cannot be judged from the supplied evidence,
and explain why in the notes field. Do not infer the hidden system identity.

## Analysis policy

The package is currently `PENDING`. No scores may be fabricated. If one
annotator completes the package, report descriptive means and paired
comparisons only; do not calculate inter-annotator agreement. Agreement
statistics are appropriate only after independent annotations from at least
two annotators are available.
