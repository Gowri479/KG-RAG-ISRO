# Retrieval-Level Quality Evaluation

**Status:** `PENDING_HUMAN_ANNOTATION`

Retrieval-level evaluation requires manually verified chunk/passage relevance judgments and answer support IDs. Synthetic labels are strictly prohibited.

- **Candidate Question Set:** 100 items in `data\annotations\retrieval_eval_template.json`
- **Target Systems:** BM25, Dense (FAISS), KG, and Hybrid
- **Target Metrics:** Recall@1, Recall@3, Recall@5, MRR, Entity Hit Rate, Triple Hit Rate
