# Frozen canonical benchmark KG-relevance analysis

## 1. Objective

Assess whether the frozen canonical QA set contains a sufficiently large, evidence-supported subset whose question structure naturally corresponds to the frozen Phase 1.6 KG. This is an analysis of existing answers; it is not a new benchmark run and is not a universal RAG-vs-KG-RAG comparison.

## 2. Frozen benchmark definition

- Source bank: `data/benchmark/isro_qa.json` (200 total records; 20 development records).
- Evaluated split: `data/benchmark/test_ids.json`; exactly 180 frozen test IDs.
- Per-question saved output source: `data/results/baseline_results.json`.
- Filtered output rows: 180; every test question and reference answer was matched by ID.
- Existing aggregate parity: all stored BM25 + LLM, Vanilla RAG, and KG-RAG ROUGE-L, reference-token coverage, exact-match, and IDK metrics were reproduced from the saved answer strings.
- The benchmark item itself has no `mission` field; annotation mission scope is extracted from the question text only. Questions that name multiple missions retain a `multiple: ...` value; unrelated questions are `not mission-specific`.
- The targeted 36-question Aditya-L1 stress test was not merged or analyzed.

## 3–6. Annotation methodology and taxonomy

Each test question was assigned one primary type, one KG-relevance label, and one KG-required label from question wording and the frozen relation vocabulary/triples. System output and score were not used to assign labels. See `data/annotations/canonical_180_kg_relevance_guidelines.md` for detailed rules and examples.

The annotation is a single-researcher classification, not inter-annotator agreement. Multi-relation means combining evidence items; two-hop is reserved for a linked mission→payload→attribute/target chain. A relationship-looking question is not labeled KG-multi-hop merely because it mentions two entities.

## 7. Annotation counts

**Question type**

| Type | N |
|---|---:|
| DIRECT_FACT | 54 |
| ATTRIBUTE | 32 |
| SINGLE_RELATION | 28 |
| TWO_HOP_RELATION | 10 |
| MULTI_RELATION | 56 |
| OTHER | 0 |

**KG relevance**

| Label | N |
|---|---:|
| KG_NOT_RELEVANT | 141 |
| KG_POTENTIALLY_USEFUL | 27 |
| KG_RELATIONALLY_NATURAL | 9 |
| KG_MULTI_HOP | 3 |

**KG required**

| Label | N |
|---|---:|
| YES | 0 |
| NO | 172 |
| UNCERTAIN | 8 |

## 8. Overall saved-output results

| System | N | Mean ROUGE-L | Median ROUGE-L | Mean ref-token coverage | Exact match | IDK rate |
|---|---:|---:|---:|---:|---:|---:|
| BM25 + LLM | 180 | 0.2915 | 0.2174 | 0.4340 | 0.0278 | 0.0167 |
| Vanilla RAG | 180 | 0.2780 | 0.2319 | 0.3989 | 0.0333 | 0.1222 |
| KG-RAG | 180 | 0.2736 | 0.2222 | 0.3921 | 0.0333 | 0.1167 |

These reproduce, after rounding, the frozen aggregate artifact: BM25 + LLM ROUGE-L 0.2915, coverage 0.4340, exact match 0.0278, IDK 0.0167; Vanilla RAG ROUGE-L 0.2780, coverage 0.3989, exact match 0.0333, IDK 0.1222; KG-RAG ROUGE-L 0.2736, coverage 0.3921, exact match 0.0333, IDK 0.1167.

## 9. KG-relevance groups

| KG relevance | N | BM25 + LLM mean ROUGE-L | Vanilla mean ROUGE-L | KG-RAG mean ROUGE-L |
|---|---:|---:|---:|---:|
| KG_NOT_RELEVANT | 141 | 0.3046 | 0.2842 | 0.2870 |
| KG_POTENTIALLY_USEFUL | 27 | 0.2637 | 0.2613 | 0.2167 |
| KG_RELATIONALLY_NATURAL | 9 | 0.1532 | 0.2019 | 0.2040 |
| KG_MULTI_HOP | 3 | 0.3452 | 0.3662 | 0.3662 |

The separate KG_RELATIONALLY_NATURAL (N=9) and KG_MULTI_HOP (N=3) groups are each below N=10; interpret them descriptively. The combined relational subset N=12 is exploratory and does not replace the primary KG_REQUIRED=YES test.

## 10. KG-required vs not required

| KG required | N | BM25 + LLM mean ROUGE-L | Vanilla mean ROUGE-L | KG-RAG mean ROUGE-L |
|---|---:|---:|---:|---:|
| YES | 0 | — | — | — |
| NO | 172 | 0.2908 | 0.2767 | 0.2754 |
| UNCERTAIN | 8 | 0.3079 | 0.3060 | 0.2339 |

## 11. Question-type results

| Question type | N | BM25 + LLM mean ROUGE-L | Vanilla mean ROUGE-L | KG-RAG mean ROUGE-L |
|---|---:|---:|---:|---:|
| ATTRIBUTE | 32 | 0.2243 | 0.2602 | 0.2132 |
| DIRECT_FACT | 54 | 0.4026 | 0.3614 | 0.3690 |
| MULTI_RELATION | 56 | 0.2529 | 0.2404 | 0.2432 |
| OTHER | 0 | — | — | — |
| SINGLE_RELATION | 28 | 0.2252 | 0.1970 | 0.2120 |
| TWO_HOP_RELATION | 10 | 0.3093 | 0.3226 | 0.2942 |

## 12. Tier and mission results

| Tier | N | BM25 + LLM mean ROUGE-L | Vanilla mean ROUGE-L | KG-RAG mean ROUGE-L |
|---|---:|---:|---:|---:|
| Tier 1 | 90 | 0.3220 | 0.3223 | 0.3080 |
| Tier 2 | 54 | 0.2587 | 0.2306 | 0.2246 |
| Tier 3 | 36 | 0.2647 | 0.2387 | 0.2611 |

Mission groups with N≥5:

| Mission label | N | BM25 + LLM | Vanilla RAG | KG-RAG |
|---|---:|---:|---:|---:|
| Aditya-L1 | 8 | 0.3218 | 0.3862 | 0.3777 |
| Chandrayaan-1 | 7 | 0.4071 | 0.3685 | 0.3974 |
| Chandrayaan-2 | 9 | 0.2925 | 0.3429 | 0.2647 |
| Chandrayaan-3 | 16 | 0.3395 | 0.3362 | 0.3008 |
| Gaganyaan | 5 | 0.2255 | 0.2268 | 0.2131 |
| Mars Orbiter Mission | 7 | 0.1620 | 0.1679 | 0.1758 |
| not mission-specific | 112 | 0.2884 | 0.2700 | 0.2618 |

Smaller mission groups remain in the JSON as descriptive counts and scores; they are not interpreted inferentially.

The primary types `TWO_HOP_RELATION` (N=10) and `OTHER` (N=0) are also shown with exact N. No separate hypothesis tests are run for taxonomy/tier/mission slices.

## 13–15. Primary comparison, secondary tests, effect sizes, and intervals

### Primary: KG-RAG vs Vanilla RAG on KG_REQUIRED=YES

N=0. Descriptive only; insufficient sample size for reliable inferential comparison.

- KG-RAG vs Vanilla RAG: mean ROUGE-L None; median None; Vanilla mean None; median None; mean paired difference None; median difference None; 95% bootstrap CI None; Wilcoxon raw p None; Holm-adjusted p None; Cohen's dz None.
- KG-RAG vs BM25 + LLM (secondary): mean paired difference None; 95% bootstrap CI None; Wilcoxon raw p None; Holm-adjusted p None.

The subset has 0 items; this is below the predeclared minimum N=10. No Wilcoxon test, effect-size claim, or bootstrap interval is used to make an inferential claim for this group.

### Secondary: KG_REQUIRED=NO

N=172. Exploratory inference only; interpret with the stated Holm correction.

- KG-RAG vs Vanilla RAG: mean ROUGE-L 0.275447; median 0.222222; Vanilla mean 0.276733; median 0.222222; mean paired difference -0.001286; median difference 0.0; 95% bootstrap CI [-0.025461, 0.022519]; Wilcoxon raw p 0.5564315298194373; Holm-adjusted p 1.0; Cohen's dz -0.008036.
- KG-RAG vs BM25 + LLM (secondary): mean paired difference -0.015328; 95% bootstrap CI [-0.041493, 0.011907]; Wilcoxon raw p 0.28833319824080383; Holm-adjusted p 0.86499959.

### Relational subset: KG_RELATIONALLY_NATURAL + KG_MULTI_HOP

N=12. Exploratory inference only; interpret with the stated Holm correction.

- KG-RAG vs Vanilla RAG: mean ROUGE-L 0.244566; median 0.222222; Vanilla mean 0.242988; median 0.222222; mean paired difference 0.001578; median difference 0.0; 95% bootstrap CI [-0.042124, 0.03719]; Wilcoxon raw p 0.8125; Holm-adjusted p 1.0; Cohen's dz 0.021248.
- KG-RAG vs BM25 + LLM (secondary): mean paired difference 0.043405; 95% bootstrap CI [0.008764, 0.077216]; Wilcoxon raw p 0.0673828125; Holm-adjusted p 0.26953125.

Subgroup p-values are exploratory and Holm-adjusted over the testable secondary comparisons in the JSON. The absent/empty primary YES group is not replaced by a more favorable subgroup. Cohen's dz is mean paired difference divided by sample SD; rank-biserial correlation is also included in the machine-readable summaries.

## 16. Failure-pattern / evidence audit

- The stored canonical per-question artifact contains answer strings, but no retrieved chunks, retrieval contexts, graph-expansion traces, or evidence ranking. Therefore it cannot establish whether KG retrieved useful evidence, whether dense RAG retrieved the same evidence, or whether context became noisy.
- On the 12 relationally natural/multi-hop items, KG-RAG and Vanilla RAG return identical answer strings for 7 items; stored IDK counts are 1 for KG-RAG and 0 for Vanilla RAG. Similarity does not identify the retrieval cause.
- The graph-specific annotation found 0 relationally natural/multi-hop questions without a cited matching edge set (see per-question reasons). Some cited paths are partial: a payload objective is not a mission result, and an observation target edge is not proof that a historical discovery occurred.
- The five most unfavorable paired ROUGE-L differences in the relationally natural/multi-hop subset are listed below as lexical-score examples only. They are not factual-error judgments.

- `isro_071` (KG_RELATIONALLY_NATURAL): KG-RAG 0.0000, Vanilla 0.2000. Question: What is the PSLV-C57 mission?
- `isro_107` (KG_RELATIONALLY_NATURAL): KG-RAG 0.3529, Vanilla 0.3600. Question: What launch vehicle was used for Aditya-L1 and what orbit did it achieve?
- `isro_007` (KG_RELATIONALLY_NATURAL): KG-RAG 0.2222, Vanilla 0.2222. Question: What launch vehicle carried Mangalyaan?
- `isro_035` (KG_RELATIONALLY_NATURAL): KG-RAG 0.2222, Vanilla 0.2222. Question: What launch vehicle carried Chandrayaan-1?
- `isro_043` (KG_RELATIONALLY_NATURAL): KG-RAG 0.1333, Vanilla 0.1333. Question: What launch vehicle was used for Chandrayaan-3?

Selected stored answer-text examples (output inspection only):

These fixed illustrative examples are not a random sample or an exhaustive factuality audit; displayed strings are truncated.

- `isro_071`: KG-RAG ROUGE-L 0.0000, Vanilla 0.2000. Reference: “PSLV-C57 was the launch mission that carried Aditya-L1 into orbit.” KG-RAG: “I don't know.” Vanilla: “The PSLV-C57 mission is not mentioned in the provided context.”
- `isro_098`: KG-RAG ROUGE-L 0.2759, Vanilla 0.2222. Reference: “SHAPE (Spectro-polarimetry of HAbitable Planet Earth) studies Earth's spectral and polarimetric properties from lunar orbit.” KG-RAG: “The SHAPE payload on Chandrayaan-3 studies the shape and topography of the lunar surface.” Vanilla: “The SHAPE payload on Chandrayaan-3 studies the shape of the lunar surface.”
- `isro_130`: KG-RAG ROUGE-L 0.4286, Vanilla 0.4286. Reference: “The Suprathermal and Energetic Particle Spectrometer (STEPS), part of the ASPEX payload on Aditya-L1, studies solar energetic particles. ASPEX was developed by...” KG-RAG: “The payload on Aditya-L1 that studies solar energetic particles is the Solar Low Energy X-ray Spectrometer (SoLEXS). It was developed by the U Rao Satellite Cen...” Vanilla: “The payload on Aditya-L1 that studies solar energetic particles is the Solar Low Energy X-ray Spectrometer (SoLEXS). It was developed by the U Rao Satellite Cen...”
- `isro_153`: KG-RAG ROUGE-L 0.4091, Vanilla 0.4091. Reference: “The Imaging Infrared Spectrometer (IIRS) on the Chandrayaan-2 orbiter studies lunar surface mineralogy across 250 spectral bands, including the 3-micrometre abs...” KG-RAG: “The payload on the Chandrayaan-2 orbiter that studies the Moon's surface mineralogy is the Alpha Particle X-ray Spectrometer (APXS).” Vanilla: “The payload on the Chandrayaan-2 orbiter that studies the Moon's surface mineralogy is the Alpha Particle X-ray Spectrometer (APXS).”
- `isro_156`: KG-RAG ROUGE-L 0.2609, Vanilla 0.2609. Reference: “Chandrayaan-1 carried instruments from NASA (Moon Mineralogy Mapper), ESA, and other international partners alongside Indian instruments — 11 scientific instrum...” KG-RAG: “The ISRO mission that carried instruments from NASA and ESA is not mentioned in the provided context.” Vanilla: “The ISRO mission that carried instruments from NASA and ESA is not mentioned in the provided context.”

ROUGE-L and token coverage measure lexical overlap, not factual correctness. No hallucination, faithfulness, or factuality conclusion is drawn.

## 17–18. Interpretation and limitations

The frozen benchmark has 12 KG-relationally-natural/multi-hop questions, but **0** meet the strict KG_REQUIRED=YES rule. Thus it cannot test the primary conditional-benefit hypothesis. The analysis uses one annotator; the benchmark's mission field was absent and conservatively derived from question text; graph coverage is incomplete for some benchmark claims; retrieval contexts are unavailable; and automatic lexical metrics do not adjudicate truth.

No questions were added, removed, or rewritten. The 36-question Aditya-L1 evaluation remains separate. No new QA benchmark was run.

## 19. Decision for next phase

**D. FROZEN BENCHMARK INSUFFICIENT**

The primary `KG_REQUIRED=YES` subset is empty. Do not use observed outcomes to relabel questions or revise the canonical benchmark. A future hypothesis test requires a separately approved protocol and independently designed relational benchmark; no such work is started here.
