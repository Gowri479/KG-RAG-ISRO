"""Generate stratified human evaluation package for Q1 journal study.

Selects 40 balanced questions across canonical test (Tiers 1, 2, 3) and Aditya-L1
targeted benchmark, yielding 120 question-system instances for evaluation across:
- BM25 + LLM
- Vanilla RAG
- KG-RAG

Strictly sets annotation fields to blank and annotation_status to PENDING.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines.bm25_llm import retrieve_context as bm25_retrieve
from src.baselines.vanilla_rag import retrieve_context as vanilla_retrieve
from src.evaluation.analysis_utils import BASELINE_PATH, TEST_IDS_PATH, is_idk, load_json
from src.retriever.hybrid import retrieve as hybrid_retrieve

ADITYA_ANSWERS_PATH = ROOT / "data" / "results" / "answers_aditya_l1.json"
OUTPUT_DIR = ROOT / "data" / "annotations"
OUTPUT_CSV = OUTPUT_DIR / "human_eval_template.csv"


def select_questions() -> list[dict]:
    test_ids = set(load_json(TEST_IDS_PATH))
    canonical_rows = [r for r in load_json(BASELINE_PATH) if r.get("id") in test_ids]

    t1 = [r for r in canonical_rows if r.get("tier") == 1]
    t2 = [r for r in canonical_rows if r.get("tier") == 2]
    t3 = [r for r in canonical_rows if r.get("tier") == 3]

    # Select balanced diverse samples from each tier:
    # Include disagreement, IDK cases, and non-zero/zero ROUGE
    def pick_diverse(tier_rows: list[dict], count: int) -> list[dict]:
        disagreement = [
            r for r in tier_rows
            if is_idk(r.get("kgrag_answer")) != is_idk(r.get("bm25_answer"))
            or is_idk(r.get("kgrag_answer")) != is_idk(r.get("vanilla_rag_answer"))
        ]
        agreements = [r for r in tier_rows if r not in disagreement]
        selected = disagreement[:count // 2]
        remaining = count - len(selected)
        selected.extend(agreements[:remaining])
        if len(selected) < count:
            selected.extend([r for r in tier_rows if r not in selected][:count - len(selected)])
        return selected

    selected_t1 = pick_diverse(t1, 10)
    selected_t2 = pick_diverse(t2, 10)
    selected_t3 = pick_diverse(t3, 10)

    # Aditya-L1 questions (10 questions)
    aditya_data = load_json(ADITYA_ANSWERS_PATH)
    aditya_q_map = {}
    for item in aditya_data:
        qid = item["question_id"]
        if qid not in aditya_q_map:
            aditya_q_map[qid] = {
                "id": qid,
                "question": item["question"],
                "reference_answer": item["reference_answer"],
                "category": item.get("category", "aditya_targeted"),
                "tier": "Aditya-L1",
                "answers": {},
                "contexts": {},
            }
        sys_name = item["system"]
        aditya_q_map[qid]["answers"][sys_name] = item.get("generated_answer", "")
        aditya_q_map[qid]["contexts"][sys_name] = item.get("retrieved_context_snippet", "")

    aditya_list = list(aditya_q_map.values())[:10]

    all_selected = []
    # Process canonical rows
    for r in selected_t1 + selected_t2 + selected_t3:
        all_selected.append({
            "id": r["id"],
            "question": r["question"],
            "reference_answer": r["reference_answer"],
            "category": f"Tier {r.get('tier')} canonical",
            "tier": str(r.get("tier")),
            "answers": {
                "bm25_llm": r.get("bm25_answer", ""),
                "vanilla_rag": r.get("vanilla_rag_answer", ""),
                "kg_rag": r.get("kgrag_answer", ""),
            },
            "contexts": {
                "bm25_llm": bm25_retrieve(r["question"], top_k=3),
                "vanilla_rag": vanilla_retrieve(r["question"], top_k=3),
                "kg_rag": hybrid_retrieve(r["question"], passage_limit=3, max_tokens=1500),
            },
        })

    all_selected.extend(aditya_list)
    return all_selected


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    questions = select_questions()

    fields = [
        "question_id",
        "question",
        "reference_answer",
        "system",
        "generated_answer",
        "retrieved_context",
        "category",
        "difficulty_tier",
        "annotation_status",
        "correctness",
        "completeness",
        "faithfulness",
        "relevance",
        "evaluator_id",
        "notes",
    ]

    rows = []
    systems = ["bm25_llm", "vanilla_rag", "kg_rag"]

    for q in questions:
        for sys_name in systems:
            ans = q["answers"].get(sys_name, "")
            ctx = q["contexts"].get(sys_name, "")
            # Truncate context for CSV storage if excessive
            if len(ctx) > 2000:
                ctx = ctx[:2000] + "... [truncated]"
            rows.append({
                "question_id": q["id"],
                "question": q["question"],
                "reference_answer": q["reference_answer"],
                "system": sys_name,
                "generated_answer": ans,
                "retrieved_context": ctx,
                "category": q["category"],
                "difficulty_tier": q["tier"],
                "annotation_status": "PENDING",
                "correctness": "",
                "completeness": "",
                "faithfulness": "",
                "relevance": "",
                "evaluator_id": "",
                "notes": "",
            })

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} annotation instances across {len(questions)} questions at {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
