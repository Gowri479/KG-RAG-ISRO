# Relation-aware blinded human evaluation guidelines

## Scope

Evaluate exactly 60 KG_REQUIRED questions and 180 candidate answers. The
candidate systems are blinded as A, B, and C independently for each question.
Do not infer, discuss, or attempt to recover system identities. Do not use
ROUGE, coverage, answer length, filenames, or previous experiment results.

Read each question and its three candidate answers independently. Judge each
answer against the question and the supplied reference/evidence package
available to the evaluator. Do not reward lexical overlap alone and do not
penalize harmless wording, acronym, date-format, or alias differences.

## Scores

Correctness:
- 1 = substantially incorrect
- 2 = mostly incorrect / major factual problems
- 3 = partially correct
- 4 = mostly correct
- 5 = fully correct

Completeness:
- 1 = misses essentially all required information
- 2 = major omissions
- 3 = partial coverage
- 4 = mostly complete
- 5 = complete

Groundedness:
- 1 = largely unsupported
- 2 = weakly supported
- 3 = partially grounded
- 4 = well grounded
- 5 = fully grounded in supplied evidence

Relevance:
- 1 = largely irrelevant
- 2 = substantial irrelevant content
- 3 = partially relevant
- 4 = mostly relevant
- 5 = directly relevant

Unsupported claim:
- 0 = no materially unsupported claim
- 1 = at least one materially unsupported claim

Score all three candidates for every question. Do not select a winner before
all scoring is complete. Leave a score blank only if the answer genuinely
cannot be judged, and record the reason outside the CSV.
