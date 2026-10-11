"""Retrieval Quality Evaluation Framework for BM25, Dense, KG, and Hybrid retrieval.

Computes:
- BM25: Recall@1, Recall@3, Recall@5, MRR
- Dense (FAISS): Recall@1, Recall@3, Recall@5, MRR
- KG: Entity hit rate, Triple hit rate, Answer-path hit rate
- Hybrid: Recall@1, Recall@3, Recall@5, MRR

Strictly refuses to fabricate chunk relevance judgments. Reports PENDING status
when ground truth relevance labels are unpopulated.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines.bm25_llm import _load_chunks, _load_bm25, _tokenize
from src.retriever.faiss_retriever import _load_index, _load_model, _query_keywords
from src.retriever.kg_retriever import _load_graph, get_kg_context
from src.retriever.hybrid import _extract_entities

TEMPLATE_PATH = ROOT / "data" / "annotations" / "retrieval_eval_template.json"
RESULTS_PATH = ROOT / "data" / "results" / "retrieval_metrics.json"
REPORT_MD = ROOT / "data" / "results" / "retrieval_quality_pending.md"


def compute_recall_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    hits = len(set(retrieved_ids[:k]) & gold_ids)
    return hits / len(gold_ids)


def compute_mrr(retrieved_ids: list[str], gold_ids: set[str]) -> float:
    if not gold_ids:
        return 0.0
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in gold_ids:
            return 1.0 / rank
    return 0.0


def check_kg_hits(question: str, gold_entities: set[str], gold_triples: set[tuple[str, str, str]]) -> dict:
    G = _load_graph()
    extracted = _extract_entities(question)
    entity_hits = len(set(extracted) & gold_entities) / len(gold_entities) if gold_entities else 0.0

    context = get_kg_context(extracted)
    triple_hits = 0
    if gold_triples:
        for s, r, o in gold_triples:
            pattern = f"{s.lower()} {r.replace('_', ' ').lower()} {o.lower()}"
            if pattern in context.lower():
                triple_hits += 1
        triple_hit_rate = triple_hits / len(gold_triples)
    else:
        triple_hit_rate = 0.0

    # Answer path hit: connected path exists in G between any pair of extracted entities
    path_hit = 0.0
    if len(extracted) >= 2 and G.number_of_nodes() > 0:
        e1, e2 = extracted[0], extracted[1]
        if e1 in G and e2 in G:
            import networkx as nx
            if nx.has_path(G.to_undirected(), e1, e2):
                path_hit = 1.0

    return {
        "entity_hit_rate": entity_hits,
        "triple_hit_rate": triple_hit_rate,
        "answer_path_hit_rate": path_hit,
    }


def evaluate_retrieval(dataset: list[dict]) -> dict:
    chunks = _load_chunks()
    bm25_index = _load_bm25()
    faiss_index = _load_index()
    emb_model = _load_model()

    bm25_r1, bm25_r3, bm25_r5, bm25_mrr = [], [], [], []
    dense_r1, dense_r3, dense_r5, dense_mrr = [], [], [], []
    hybrid_r1, hybrid_r3, hybrid_r5, hybrid_mrr = [], [], [], []
    kg_ent_hits, kg_trp_hits, kg_path_hits = [], [], []

    for item in dataset:
        q = item["question"]
        gold_chunks = set(item.get("relevant_chunk_ids", [])) | set(item.get("answer_support_chunk_ids", []))
        gold_ents = set(item.get("relevant_kg_entities", []))
        gold_triples = {tuple(t) for t in item.get("relevant_kg_triples", [])}

        # BM25 retrieval
        scores = bm25_index.get_scores(_tokenize(q))
        bm25_top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]
        bm25_top_ids = [str(i) for i in bm25_top]

        bm25_r1.append(compute_recall_at_k(bm25_top_ids, gold_chunks, 1))
        bm25_r3.append(compute_recall_at_k(bm25_top_ids, gold_chunks, 3))
        bm25_r5.append(compute_recall_at_k(bm25_top_ids, gold_chunks, 5))
        bm25_mrr.append(compute_mrr(bm25_top_ids, gold_chunks))

        # Dense retrieval
        import numpy as np
        vec = emb_model.encode([q], convert_to_numpy=True, normalize_embeddings=True)
        _, indices = faiss_index.search(np.asarray(vec, dtype=np.float32), 10)
        dense_top_ids = [str(i) for i in indices[0] if 0 <= i < len(chunks)]

        dense_r1.append(compute_recall_at_k(dense_top_ids, gold_chunks, 1))
        dense_r3.append(compute_recall_at_k(dense_top_ids, gold_chunks, 3))
        dense_r5.append(compute_recall_at_k(dense_top_ids, gold_chunks, 5))
        dense_mrr.append(compute_mrr(dense_top_ids, gold_chunks))

        # Hybrid retrieval (RRF combination)
        rrf_scores = {}
        for rank, cid in enumerate(bm25_top_ids, start=1):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (60 + rank))
        for rank, cid in enumerate(dense_top_ids, start=1):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (60 + rank))
        hybrid_top_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:10]

        hybrid_r1.append(compute_recall_at_k(hybrid_top_ids, gold_chunks, 1))
        hybrid_r3.append(compute_recall_at_k(hybrid_top_ids, gold_chunks, 3))
        hybrid_r5.append(compute_recall_at_k(hybrid_top_ids, gold_chunks, 5))
        hybrid_mrr.append(compute_mrr(hybrid_top_ids, gold_chunks))

        # KG hit rates
        kg_res = check_kg_hits(q, gold_ents, gold_triples)
        kg_ent_hits.append(kg_res["entity_hit_rate"])
        kg_trp_hits.append(kg_res["triple_hit_rate"])
        kg_path_hits.append(kg_res["answer_path_hit_rate"])

    return {
        "BM25": {
            "Recall@1": round(mean(bm25_r1), 4),
            "Recall@3": round(mean(bm25_r3), 4),
            "Recall@5": round(mean(bm25_r5), 4),
            "MRR": round(mean(bm25_mrr), 4),
        },
        "Dense": {
            "Recall@1": round(mean(dense_r1), 4),
            "Recall@3": round(mean(dense_r3), 4),
            "Recall@5": round(mean(dense_r5), 4),
            "MRR": round(mean(dense_mrr), 4),
        },
        "Hybrid": {
            "Recall@1": round(mean(hybrid_r1), 4),
            "Recall@3": round(mean(hybrid_r3), 4),
            "Recall@5": round(mean(hybrid_r5), 4),
            "MRR": round(mean(hybrid_mrr), 4),
        },
        "KG": {
            "entity_hit_rate": round(mean(kg_ent_hits), 4),
            "triple_hit_rate": round(mean(kg_trp_hits), 4),
            "answer_path_hit_rate": round(mean(kg_path_hits), 4),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality across BM25, Dense, KG, and Hybrid")
    parser.add_argument("--input", type=Path, default=TEMPLATE_PATH, help="Path to retrieval annotation template JSON")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Template not found at {args.input}")
        return

    data = json.loads(args.input.read_text(encoding="utf-8"))
    annotated = [
        item for item in data
        if item.get("annotation_status", "").upper() == "ANNOTATED"
        and (item.get("relevant_chunk_ids") or item.get("relevant_kg_entities"))
    ]

    if not annotated:
        reason = (
            "Retrieval-level evaluation requires manually verified chunk/passage relevance judgments "
            "and answer support IDs. Synthetic labels are strictly prohibited."
        )
        report = {
            "experiment": "retrieval_quality",
            "status": "pending_human_annotation",
            "reason": reason,
            "candidate_questions": len(data),
            "systems": ["BM25", "Dense (FAISS)", "KG", "Hybrid"],
            "metrics": {
                "BM25": ["Recall@1", "Recall@3", "Recall@5", "MRR"],
                "Dense": ["Recall@1", "Recall@3", "Recall@5", "MRR"],
                "KG": ["entity_hit_rate", "triple_hit_rate", "answer_path_hit_rate"],
                "Hybrid": ["Recall@1", "Recall@3", "Recall@5", "MRR"],
            },
            "annotation_template": str(args.input.relative_to(ROOT)),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        RESULTS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
        REPORT_MD.write_text(
            f"# Retrieval-Level Quality Evaluation\n\n"
            f"**Status:** `PENDING_HUMAN_ANNOTATION`\n\n"
            f"{reason}\n\n"
            f"- **Candidate Question Set:** {len(data)} items in `{args.input.relative_to(ROOT)}`\n"
            f"- **Target Systems:** BM25, Dense (FAISS), KG, and Hybrid\n"
            f"- **Target Metrics:** Recall@1, Recall@3, Recall@5, MRR, Entity Hit Rate, Triple Hit Rate\n",
            encoding="utf-8",
        )
        print(f"[PENDING] Retrieval quality framework initialized. Saved status to {RESULTS_PATH}")
        return

    print(f"Found {len(annotated)} verified annotated questions. Evaluating retrieval...")
    metrics = evaluate_retrieval(annotated)
    out = {
        "experiment": "retrieval_quality",
        "status": "COMPLETED",
        "n_annotated_questions": len(annotated),
        "metrics": metrics,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    RESULTS_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[SUCCESS] Computed retrieval metrics saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
