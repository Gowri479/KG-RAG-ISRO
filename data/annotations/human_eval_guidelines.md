# Human Factual Evaluation Guidelines for Space Agency Domain QA

## Overview

This document provides explicit annotation guidelines for evaluating generated answers from Question Answering (QA) systems on the Indian Space Research Organisation (ISRO) domain corpus. The evaluation covers outputs produced by:
1. **BM25 + LLM** (Sparse passage retrieval + Mistral-7B)
2. **Vanilla RAG** (Dense FAISS retrieval + Mistral-7B)
3. **KG-RAG** (Hybrid Knowledge Graph + Dense retrieval + Mistral-7B)

Each response is evaluated on four independent, 3-point ordinal dimensions (0, 1, 2).

---

## Evaluation Dimensions & Scoring Rubric

### 1. Correctness (Factuality with Respect to Domain Truth)
Measures whether the stated facts in the answer are accurate against verified ground truth (reference answer and official ISRO documentation).

- **0 (Incorrect / Unsupported):** The answer makes factually false statements, attributes wrong dates, incorrect missions, wrong launch vehicles, or invents non-existent technical specifications.
- **1 (Partially Correct):** The primary entity or concept is correctly identified, but contains minor inaccuracies (e.g., slightly off date or secondary attribute error), or combines a correct statement with an unverified claim.
- **2 (Completely Correct):** All factual claims in the response are entirely accurate and agree with official ground truth. If the reference is unanswerable or the context provides zero evidence, a clear abstention ("I don't know.") is scored as 2.

### 2. Completeness (Information Coverage)
Measures the extent to which the response satisfies all sub-questions or necessary elements demanded by the prompt.

- **0 (Incomplete / Non-informative):** Fails to provide the key factual entity or target of the query (or says "I don't know" when reliable evidence exists).
- **1 (Partially Complete):** Answers part of the question (e.g., provides the launch year but misses the vehicle; or lists two out of four payloads).
- **2 (Fully Complete):** Provides all requisite information requested by the query without omitting critical domain details.

### 3. Faithfulness to Retrieved Evidence (Groundedness)
Measures whether every claim in the generated answer is directly traceable to and supported by the retrieved context provided to the model.

- **0 (Unfaithful / Hallucinated):** The model asserts facts that are NOT supported by the retrieved context, or contradicts facts explicitly stated in the context (parametric hallucination).
- **1 (Partially Faithful):** The core statement is derived from context, but includes extraneous speculation or extrapolations not directly present in the context.
- **2 (Completely Faithful):** Every statement in the answer is strictly substantiated by the retrieved context snippet. If context is empty or lacks evidence, correctly stating "I don't know" is scored as 2.

### 4. Relevance (Conciseness and Focus)
Measures how directly the response addresses the user prompt without irrelevant tangents or boilerplate filler.

- **0 (Irrelevant / Off-topic):** The answer drifts onto unrelated missions, provides irrelevant trivia, or fails to address the subject of the prompt.
- **1 (Partially Relevant):** Addresses the question but includes excessive conversational fluff, repetitive excerpts, or tangential explanations.
- **2 (Directly Relevant):** Directly, concisely, and cleanly answers the specific question asked.

---

## Annotation Procedure

1. **Dual Independent Annotation:**
   - Two domain-literate annotators should independently score all instances in `data/annotations/human_eval_template.csv`.
   - Annotators must NOT communicate during scoring.
   - Evaluator IDs must be recorded in the `evaluator_id` column.

2. **Inter-Annotator Agreement:**
   - Evaluated using **Cohen's kappa ($\kappa$)** for pairs of annotators on ordinal scales.
   - For multiple annotators, compute **Krippendorff's alpha ($\alpha$)**.
   - Minimum acceptable agreement threshold: $\kappa \ge 0.60$ (substantial agreement).
   - Discrepancies are resolved by a third arbiter to produce the final consensus set.

3. **Strict Scientific Rules:**
   - Synthetic, automated, or simulated human ratings are strictly prohibited.
   - Rows marked `PENDING` must remain untouched until genuine human assessment is completed.
