# Pre-registration: corrected-context rerun

Status: **APPROVED 2026-10-08** by the project owner, before any generation.
Raters: two or more independent raters confirmed (see §8). The commit hash of the approved version is recorded in
the run manifest, and any later change to this file is reported as a deviation.

## 1. Why this rerun exists

Every earlier generation run used Ollama `num_ctx=2048` with a 1,500-word
context. Prompts were often 2.4k–3.9k model tokens, and Ollama silently kept
only the last ~1,027, dropping the KG evidence placed first. Commit `7cb2d2a`
fixes this. The rerun answers one question: **with evidence that actually
reaches the generator, does KG-grounded generation improve human-judged answer
quality over vanilla dense RAG?**

All earlier results stay in the repository and are reported as as-implemented
results under truncation. This rerun does not replace them.

## 2. What is held fixed

- **Benchmark:** `relation_aware_qa_v2.json`, SHA-256 `6c7f3600…2b64`,
  unchanged. No questions, references or labels are edited.
- **Retrieval:** frozen. Contexts come from
  `data/results/relation_aware/generation_results.jsonl`, SHA-256 `e13e4c82…635c`.
  No retrieval is rerun.
- **Prompts and system prompts:** exactly as at commit `7cb2d2a`. No tuning.
- **Model:** `mistral:7b-instruct-q4_K_M`, temperature 0.1, `num_predict` 150,
  `num_ctx` 2048, `seed` 20261008.
- **Evidence budget:** at most 1,500 model tokens, **identical for all three
  systems**, and always within the window. Each context is trimmed to the
  budget from its tail (lowest-priority evidence) before sending. Dry run
  (`--dry-run`, no model calls): largest prompt 1,676 tokens against a
  1,890-token limit; no prompt needed trimming by the window guard.
- **Runner:** `src/evaluation/run_corrected_rerun.py`. It refuses to generate
  while this file is a draft or uncommitted, and while any source code is
  uncommitted.
- **One run.** No regeneration of any row for any reason except a logged
  transport failure (HTTP error or timeout), retried once and reported.

## 3. Systems (3)

| Label | System | Context |
|---|---|---|
| V | Vanilla dense RAG | frozen `vanilla_dense_rag` context |
| A | Relation-aware KG-RAG (A_CURRENT) | frozen `relation_aware_kg_rag` context |
| C | Two-stage KG-grounded (C_TWO_STAGE) | frozen KG facts + dense text, plan then answer |

These match the earlier human evaluations, so corrected and truncated results
compare directly. No other systems are generated.

## 4. Analysis sets

- **Primary:** the 60 `KG_REQUIRED = YES` items, as frozen.
- **Reported, not tested:** the 12 `KG_REQUIRED = NO` items, generated for
  completeness and summarised descriptively.
- **Sensitivity analyses (pre-specified, no inference beyond what is listed):**
  1. Deduplicated: one item per unique question text (45 items). The kept item
     is the lowest question ID.
  2. Excluding the 10 `DIRECT_CONTROL` items labelled `KG_REQUIRED = YES` (50 items).
  3. The 13 held-out items, described only. This set is too small to test, and
     it leaked (6 texts duplicated in development; 4 items used in the prompt
     diagnostic).
- **Not performed:** the KG_REQUIRED YES-vs-NO difference-in-differences. Five
  identical questions carry both labels, so the contrast is invalid.

## 5. Endpoints

**Primary:** human **correctness** (1–5), paired by question:
- C vs V
- A vs V

**Secondary:** completeness, groundedness, relevance (1–5); unsupported-claim
rate (0/1); C vs A on all endpoints; ROUGE-L, reference coverage and
"I don't know" rate (automated, reported as secondary only).

## 6. Statistical plan

- Paired Wilcoxon signed-rank test, two-sided; zero differences dropped
  (Wilcoxon method). Unsupported-claim rate: exact McNemar test.
- Holm correction across the **2 primary tests** at α = 0.05. Secondary tests
  are Holm-corrected within their own family and labelled exploratory.
- For every comparison, report mean and median difference, 10,000-resample
  paired bootstrap 95% CI (seed 20261008), Cohen's dz, and wins/losses/ties.
- **Power:** with n = 60 pairs, the primary tests have roughly 80% power for
  dz ≈ 0.41 at the Holm-adjusted α. Smaller true effects may be missed; a
  non-significant result is reported as "not detected", never as "no effect".

## 7. Decision rule (fixed now)

For each KG system (C, A) against V:
- **SUPPORTED:** the Holm-adjusted primary test has p < .05 with a positive
  mean difference, **and** the unsupported-claim rate is not higher than V's
  by a statistically significant margin (McNemar p < .05).
- **HARMFUL:** the Holm-adjusted primary test has p < .05 with a negative
  mean difference.
- **NOT SUPPORTED:** anything else. The CI is reported, and no equivalence is
  claimed.

The result is reported whatever it is. No further variants, prompts or reruns
follow from this outcome within this study.

## 8. Human evaluation protocol

- **Blinding:** candidates shuffled per question with seed 20261008; the
  mapping is kept outside the rater files.
- **Shown to raters:** the question, the reference answer and acceptable
  answers, and a **shared source-evidence pack**: the text of the item's
  `supporting_chunks` and its `supporting_triples`. The pack is identical for
  all three candidates, so it does not reveal system identity.
- **Correctness for set-valued questions:** an answer naming any member of the
  correct set that the evidence pack supports counts as correct. The single
  reference is shown as one example, not the only answer.
- **Groundedness and unsupported claims** are judged against the evidence pack.
- **Raters:** at least 2 independent raters, scoring all items. Agreement is
  reported as Krippendorff's α (ordinal) per dimension. The primary analysis
  uses the per-item mean of the raters. *If only one rater is available, this
  is stated as a limitation and no agreement figure is claimed.*
- Raters do not see automated metrics, system names or earlier results.

## 9. Integrity checks (run automatically, reported)

- For every model call: `prompt_eval_count` equals `prompt_tokens_expected`,
  and `ollama_truncated` is false. Any failure is reported, never hidden.
- The benchmark and frozen-context hashes match §2.
- Every candidate in the rater CSV matches its generation by hash.

## 10. Reporting

Report all three systems on all endpoints, all sensitivity sets, the
integrity-check results and every deviation from this plan. Report the
truncated (earlier) and corrected (this) results side by side.

## 11. Clarifications made before any rating or result was seen

Recorded 2026-10-08 while generation was running and before any generated
answer, metric or rating had been inspected. They fill gaps in §6–§8 and do
not change any endpoint, comparison or decision rule.

1. **Unsupported-claim aggregation across raters:** an item counts as
   containing an unsupported claim if **any** rater flags it (conservative).
   The 1–5 scales use the rater mean, as in §8.
2. **"I don't know" answers:** they make no claim, so they are fully grounded
   and carry no unsupported claim, but score low on correctness and
   completeness. This is written into the rater guidelines.
3. **Wilcoxon computation:** `scipy.stats.wilcoxon`, `zero_method="wilcox"`,
   default `method` (exact for small samples without ties, normal
   approximation otherwise). If every difference is zero, p = 1.
4. **Sensitivity sets (§4):** report means, mean differences, bootstrap CIs and
   wins/losses/ties only, with no p-values. The deduplicated set keeps the
   lowest question ID per unique question text.
5. **Secondary family (§6):** all non-primary human comparisons (C vs V, A vs V
   on the other four endpoints; C vs A on all five) form one Holm family.
   Automated metrics are reported with uncorrected p-values, labelled
   exploratory.
6. **Package presentation:** questions appear in benchmark order; candidate
   letters are shuffled per question; each rater receives an identical CSV.
7. **Code:** analysis is `src/evaluation/analyze_corrected_rerun.py` and the
   package builder is `src/evaluation/build_corrected_rerun_human_package.py`,
   both committed before any rating exists.

## 12. Deviation: single human rater

Recorded 2026-10-08, after generation and before any human rating existed.

- **Change:** the human evaluation uses **one** human rater instead of the two
  or more confirmed at approval. §8 already provides for this case: the
  analysis uses that rater's scores directly, the single rater is stated as a
  limitation, and no agreement figure is reported. Endpoints, comparisons,
  tests and the decision rule (§5–§7) are unchanged.
- **Reason:** a second independent human rater was not available.
- **Excluded material:** a fully pre-filled rater-1 sheet drafted by a language
  model was received and is **not** used as, or as a starting point for, human
  ratings. The human rater scores the blank package file
  (`data/annotations/corrected_rerun_human_eval_rater1.csv`) without seeing
  any model-generated scores. The model draft may only appear as a separately
  labelled, non-preregistered AI-judge comparison after the human analysis is
  final.
- **Unused file:** `corrected_rerun_human_eval_rater2.csv` remains in the
  package but is not part of the analysis.

## 13. Provenance of the rater-1 scores (recorded before analysis)

Recorded 2026-10-08, before the analysis script was run on any rating.

- **Scores analysed:** `data/annotations/corrected_rerun_human_eval_rater1_filled.csv`
  (SHA-256 `7d216c83…3914`). The project owner states that they rated it.
- **Starting point:** the rater worked from a fully pre-filled sheet drafted by
  a language model, kept for audit as
  `data/annotations/corrected_rerun_rater1_model_draft_AUDIT_ONLY.csv`
  (SHA-256 `d16b79b8…8fed`). The model and its inputs are not recorded.
- **Overlap:** 1,071 of 1,080 score cells (99.2%) in the analysed file equal
  the model draft. 9 cells differ, across 7 of 72 questions.
- **Consequence for interpretation:** this deviates from §12, which required
  scoring the blank file without seeing model scores. The rating is
  model-anchored: a single human reviewed and adopted model-drafted scores.
  It is not an independent human evaluation. Every report of these results
  must state this, e.g. "AI-drafted scores reviewed by one human rater
  (99.2% unchanged)".
- The analysis plan (§5–§7, §11) is otherwise unchanged.

## 14. Independent human spot-check (fixed before any spot-check rating)

Recorded 2026-10-08 after the primary analysis (§13) and before any
spot-check score exists. Purpose: test whether the model-anchored rater-1
scores agree with an independent human.

- **Sample:** 20 of the 60 primary questions, drawn with
  `random.Random(20261022).sample` from the sorted question IDs. The list is in
  `data/results/corrected_rerun/spotcheck_selection.json`, giving 60 candidate
  answers. It includes rakg_064, one of the three partially unblinded items
  (§13); it was not replaced.
- **Rater:** a person who has not seen the model draft, the rater-1 scores,
  any analysis output or `data/results/corrected_rerun/`. They score the
  blank `data/annotations/corrected_rerun_spotcheck.csv` from scratch using
  the same guidelines, and save it as `corrected_rerun_spotcheck_filled.csv`.
- **Analysis:** `src/evaluation/analyze_corrected_rerun_spotcheck.py`.
  Krippendorff's α between the checker and rater 1 per dimension (ordinal;
  nominal for unsupported claims), exact agreement, and both raters'
  A−V and C−V correctness differences with bootstrap CIs on the 20 questions.
- **Validation criterion:** PASSED only if correctness ordinal α ≥ 0.667
  **and** the checker's A−V and C−V mean correctness differences have the same
  sign as rater 1's. Otherwise FAILED.
- **Reporting:** the outcome is reported either way. If PASSED, the primary
  results may be described as AI-drafted scores validated against an
  independent human on a random subset. If FAILED, the primary results must
  be reported as unvalidated, with the disagreement shown.

## 15. Spot-check presentation format (recorded before any spot-check rating)

Recorded 2026-10-08, before the kit was given to the evaluator.

- The evaluator receives `KG-RAG_Spotcheck_Scoring.xlsx` (one row per answer,
  dropdowns, locked source text) plus a PDF of the guidelines, built by
  `src/evaluation/build_spotcheck_evaluator_kit.py` from the blank
  `corrected_rerun_spotcheck.csv`. The CSV is included as a backup. Reason:
  Excel can change the encoding of a UTF-8 CSV on save, which would alter the
  hash-checked answer text.
- `src/evaluation/import_spotcheck_workbook.py` converts the returned workbook
  into `corrected_rerun_spotcheck_filled.csv`. It checks every answer against
  the package and every score against the allowed values. §14's analysis is
  unchanged.
- The scoring rubric is word-for-word the one rater 1 received. Additions are
  procedural only: independence rules (no AI tools, no discussion, no other
  ratings), filling instructions, and one sentence stating that 2 and 4 mark
  levels between the described anchors.
