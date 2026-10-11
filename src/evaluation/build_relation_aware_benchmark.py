"""Build the isolated relation-aware benchmark from canonical KG evidence."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRIPLES = ROOT / "data" / "corpus" / "triples.jsonl"
OUT = ROOT / "data" / "relation_aware_benchmark"
SEED = 20261008


def load() -> list[dict]:
    return [json.loads(line) for line in TRIPLES.read_text(encoding="utf8").splitlines() if line.strip()]


def template(triple: dict) -> tuple[str, str]:
    subject, object_, relation = triple["subject"], triple["object"], triple["relation"]
    if relation == "HAS_PAYLOAD":
        return f"Which payload is carried by {subject}?", "SINGLE_RELATION"
    if relation == "DEVELOPED_BY":
        return f"Which organization developed {subject}?", "SINGLE_RELATION"
    if relation == "HAS_OBJECTIVE":
        return f"What is the objective of {subject}?", "SINGLE_RELATION"
    if relation == "OBSERVES":
        return f"What does {subject} observe?", "SINGLE_RELATION"
    if relation == "STUDIES":
        return f"What does {subject} study?", "SINGLE_RELATION"
    if relation == "LAUNCHED_ON":
        return f"When was {subject} launched?", "DIRECT_CONTROL"
    if relation == "LAUNCHED_BY":
        return f"Which launch vehicle launched {subject}?", "SINGLE_RELATION"
    if relation == "LAUNCHED_FROM":
        return f"Where was {subject} launched from?", "SINGLE_RELATION"
    if relation == "ORBITS":
        return f"Where does {subject} orbit?", "SINGLE_RELATION"
    if relation == "OPERATED_BY":
        return f"Which organization operates {subject}?", "SINGLE_RELATION"
    if relation == "LED_BY":
        return f"Which organization leads {subject}?", "SINGLE_RELATION"
    if relation == "PRECEDED_BY":
        return f"Which mission preceded {subject}?", "SINGLE_RELATION"
    raise ValueError(relation)


def row(number: int, triple: dict, category: str, required: str) -> dict:
    return {
        "question_id": f"rakg_{number:03d}",
        "question": template(triple)[0],
        "category": category,
        "kg_required": required,
        "relation_type": [triple["relation"]],
        "mission": triple.get("mission"),
        "reference_answer": triple["object"],
        "acceptable_answers": [triple["object"]],
        "supporting_triples": [triple["triple_id"]],
        "supporting_paths": [[triple["triple_id"]]],
        "supporting_chunks": [triple.get("source_chunk_id")],
        "source_document": triple.get("source_document"),
        "source_url": triple.get("source_url"),
        "selection_rule": "All canonical triples in supported relation families, ordered by relation and triple ID; controls are deterministic launch/date facts.",
        "system_result_contamination": False,
    }


def main() -> None:
    triples = sorted(load(), key=lambda x: (x["relation"], x["triple_id"]))
    by_subject = {}
    for triple in triples:
        by_subject.setdefault(triple["subject_id"], []).append(triple)
    questions = []
    direct = [x for x in triples if x["relation"] in {"LAUNCHED_ON", "LAUNCHED_BY", "LAUNCHED_FROM", "ORBITS"}][:10]
    questions.extend(row(i + 1, triple, "DIRECT_CONTROL", "YES") for i, triple in enumerate(direct))
    single = [x for x in triples if x["relation"] in {"HAS_PAYLOAD", "DEVELOPED_BY", "HAS_OBJECTIVE", "OBSERVES", "STUDIES", "OPERATED_BY", "LED_BY", "PRECEDED_BY"}][:30]
    questions.extend(row(len(questions) + i + 1, triple, "SINGLE_RELATION", "YES") for i, triple in enumerate(single))
    two_hop = []
    for first in triples:
        if first["relation"] != "HAS_PAYLOAD":
            continue
        for second in by_subject.get(first["object_id"], []):
            if second["relation"] in {"OBSERVES", "STUDIES"}:
                two_hop.append((first, second))
    for i, (first, second) in enumerate(two_hop[:10], len(questions) + 1):
        questions.append({
            **row(i, second, "TWO_HOP_RELATION", "YES"),
            "question": f"What does a payload carried by {first['subject']} {second['relation'].lower()}?",
            "relation_type": ["HAS_PAYLOAD", second["relation"]],
            "supporting_triples": [first["triple_id"], second["triple_id"]],
            "supporting_paths": [[first["triple_id"], second["triple_id"]]],
            "supporting_chunks": [first.get("source_chunk_id"), second.get("source_chunk_id")],
        })
    multi = []
    for first in triples:
        if first["relation"] != "HAS_PAYLOAD":
            continue
        for second in by_subject.get(first["object_id"], []):
            if second["relation"] == "DEVELOPED_BY":
                multi.append((first, second))
    for i, (first, second) in enumerate(multi[:10], len(questions) + 1):
        questions.append({
            **row(i, second, "MULTI_RELATION", "YES"),
            "question": f"Which organization developed a payload carried by {first['subject']}?",
            "relation_type": ["HAS_PAYLOAD", "DEVELOPED_BY"],
            "supporting_triples": [first["triple_id"], second["triple_id"]],
            "supporting_paths": [[first["triple_id"], second["triple_id"]]],
            "supporting_chunks": [first.get("source_chunk_id"), second.get("source_chunk_id")],
        })
    controls = [x for x in triples if x["relation"] in {"LAUNCHED_ON", "LAUNCHED_BY"}][:12]
    questions.extend(row(len(questions) + i + 1, triple, "DIRECT_CONTROL", "NO") for i, triple in enumerate(controls))
    rng = random.Random(SEED)
    rng.shuffle(questions)
    for i, item in enumerate(questions, 1):
        item["question_id"] = f"rakg_{i:03d}"
        item["split"] = "development" if (i % 5) else "evaluation"
    OUT.mkdir(parents=True, exist_ok=True)
    benchmark = {"benchmark_id": "isro_relation_aware_qa_v2", "schema_version": "2.0", "seed": SEED, "questions": questions}
    path = OUT / "relation_aware_qa_v2.json"
    path.write_text(json.dumps(benchmark, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "benchmark_sha256": digest,
        "total_questions": len(questions),
        "split_counts": Counter(item["split"] for item in questions),
        "category_counts": Counter(item["category"] for item in questions),
        "required_counts": Counter(item["kg_required"] for item in questions),
        "construction": "canonical triples only; no answer/system outputs inspected",
        "seed": SEED,
    }
    (OUT / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2, default=dict) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
