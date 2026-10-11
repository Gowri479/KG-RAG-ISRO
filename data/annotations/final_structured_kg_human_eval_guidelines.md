# Blinded human evaluation guidelines

## Scope

Evaluate all 78 answer instances independently. The three systems are
identified only as A, B, and C. Do not infer or discuss their identities.
Judge each answer against the question, reference answer, and available
authoritative evidence. Do not use ROUGE, token overlap, or answer length as
a substitute for factual judgment.

## Scoring

For correctness, completeness, groundedness, and relevance use the 1–5 scale:

- 1 = very poor or incorrect
- 2 = mostly incorrect, substantially incomplete, or weakly supported
- 3 = partially correct, complete, supported, or relevant
- 4 = mostly correct, complete, supported, and relevant
- 5 = fully correct, complete, clearly grounded, and directly relevant

For unsupported claim, enter `YES` when the answer contains a factual claim
not supported by the supplied evidence or reference answer; otherwise enter
`NO`. Leave a score blank rather than guessing when the answer cannot be
judged, and record the reason separately if applicable.

Score each question independently. Do not select a winner before all scores
are recorded. Harmless wording, acronym, date-format, and alias differences
should not be penalized. For multi-hop questions, correctness requires the
requested end answer and completeness requires the relevant relationship to
be answered.

Human factual-quality conclusions are pending blinded annotation.
