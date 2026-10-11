# Corrected-rerun blinded human evaluation

Pre-registered protocol: `reports/preregistration_corrected_rerun.md` §8.
Score independently. Do not discuss items with the other rater until both
files are complete. Do not open `data/results/corrected_rerun/`.

For each question you get the reference answer, the acceptable answers and an
**evidence pack** (supporting facts and official source text). It is the same
for all three candidates (A, B, C). Candidate letters are shuffled per
question; do not try to identify systems.

## Scores (1-5)

- **Correctness**: 1 substantially incorrect, 2 mostly incorrect,
  3 partially correct, 4 mostly correct, 5 fully correct.
  Some questions have several correct answers (e.g. "Which organization
  developed a payload carried by Aditya-L1?"). The reference is **one
  example**: any answer the evidence pack supports as a member of the correct
  set counts as correct. Do not penalise wording, acronym, alias or date-format
  differences.
- **Completeness**: 1 misses essentially all required information,
  3 partial, 5 complete.
- **Groundedness**: judged against the evidence pack. 1 largely unsupported,
  3 partially supported, 5 fully supported. A correct "I don't know" makes no
  claim and is fully grounded, but scores low on correctness and completeness.
- **Relevance**: 1 largely irrelevant, 3 partially relevant, 5 directly relevant.

## Unsupported claim (0/1)

1 if the answer makes at least one material claim that the evidence pack does
not support, otherwise 0.

Fill every score cell. If an answer genuinely cannot be judged, still score it
and note the reason outside the CSV.
