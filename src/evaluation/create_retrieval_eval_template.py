"""Generate retrieval quality annotation template for 100 questions.

Selects 70 canonical test questions (stratified across Tiers 1-3) and 30 Aditya-L1
questions, with ground-truth fields left empty for human relevance annotation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.analysis_utils import BASELINE_PATH, TEST_IDS_PATH, load_json

ADITYA_BENCHMARK = ROOT / "data" / "benchmark" / "aditya_l1_optional_qa.json"
OUTPUT_PATH = ROOT / "data" / "annotations" / "retrieval_eval_template.json"


def main() -> None:
    test_ids = set(load_json(TEST_IDS_PATH))
    canonical_rows = [r for r in load_json(BASELINE_PATH) if r.get("id") in test_ids]

    t1 = [r for r in canonical_rows if r.get("tier") == 1][:35]
    t2 = [r for r in canonical_rows if r.get("tier") == 2][:20]
    t3 = [r for r in canonical_rows if r.get("tier") == 3][:15]

    aditya_items = load_json(ADITYA_BENCHMARK)[:30]

    items = []
    for r in t1 + t2 + t3:
        items.append({
            "question_id": r["id"],
            "question": r["question"],
            "reference_answer": r["reference_answer"],
            "tier": f"Tier {r.get('tier')} canonical",
            "source_document_id": "",
            "source_url": "",
            "relevant_chunk_ids": [],
            "relevant_kg_entities": [],
            "relevant_kg_triples": [],
            "answer_support_chunk_ids": [],
            "annotation_status": "PENDING",
        })

    for r in aditya_items:
        items.append({
            "question_id": r.get("id") or r.get("question_id"),
            "question": r["question"],
            "reference_answer": r.get("answer") or r.get("reference_answer", ""),
            "tier": "Aditya-L1 targeted",
            "source_document_id": r.get("source_document_id", ""),
            "source_url": r.get("source_url", ""),
            "relevant_chunk_ids": [],
            "relevant_kg_entities": [],
            "relevant_kg_triples": [],
            "answer_support_chunk_ids": [],
            "annotation_status": "PENDING",
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(items, indent=2), encoding="utf-8")
    print(f"Saved {len(items)} retrieval evaluation question templates to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
