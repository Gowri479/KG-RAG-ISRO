"""Audit the submitted sheet and rebuild the authoritative blinded sheet."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "data" / "relational_benchmark" / "relational_qa_v1.json"
GEN = ROOT / "data" / "results" / "final_structured_kg_eval" / "generation_results.jsonl"
MANIFEST = ROOT / "data" / "results" / "final_structured_kg_eval" / "run_manifest.json"
OUT = ROOT / "data" / "results" / "final_structured_kg_eval"
ANNOTATIONS = ROOT / "data" / "annotations"
SUBMITTED = Path(r"D:\Users\NANS\Downloads\final_structured_kg_human_eval_filled_01.csv")
V2 = ANNOTATIONS / "final_structured_kg_human_eval_v2.csv"
GUIDELINES = ANNOTATIONS / "final_structured_kg_human_eval_v2_guidelines.md"
AUDIT_JSON = OUT / "human_csv_integrity_audit.json"
AUDIT_MD = OUT / "human_csv_integrity_audit.md"
SYSTEMS = ("vanilla_rag", "corrected_structured_kg_rag", "bm25_llm")
SCORE_FIELDS = ("correctness", "completeness", "groundedness", "relevance", "unsupported_claim")


def answer_hash(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value.strip()).casefold()
    return hashlib.sha256(normalized.encode("utf8")).hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf8"))
    seed = manifest["blind_randomization_seed"]
    questions = [q for q in json.loads(BENCHMARK.read_text(encoding="utf8"))["questions"] if q["kg_required"] == "YES"]
    generated = {}
    for line in GEN.read_text(encoding="utf8").splitlines():
        if line.strip():
            row = json.loads(line)
            key = (row["question_id"], row["system"])
            if key in generated:
                raise ValueError(f"Duplicate generation row: {key}")
            generated[key] = row
    if len(questions) != 26 or len(generated) != 78:
        raise ValueError("Original generation package is incomplete")
    rng = random.Random(seed)
    mapping = {}
    expected = {}
    verification = []
    for question in questions:
        order = list(SYSTEMS)
        rng.shuffle(order)
        mapping[question["question_id"]] = dict(zip(("A", "B", "C"), order))
        for label, system in mapping[question["question_id"]].items():
            row = generated[(question["question_id"], system)]
            expected[(question["question_id"], label)] = row
        verification.append({
            "question_id": question["question_id"],
            **{
                f"{label}_system": mapping[question["question_id"]][label]
                for label in ("A", "B", "C")
            },
            **{
                f"{label}_answer_sha256": answer_hash(expected[(question["question_id"], label)]["answer"])
                for label in ("A", "B", "C")
            },
        })
    submitted = []
    if SUBMITTED.exists():
        with SUBMITTED.open(encoding="utf8-sig", newline="") as handle:
            submitted = list(csv.DictReader(handle))
    submitted_keys = {}
    row_audit = []
    for row in submitted:
        key = (row.get("question_id", ""), row.get("blind_system_id", ""))
        status = "UNKNOWN"
        if key in submitted_keys:
            status = "DUPLICATE"
        elif key not in expected:
            status = "UNKNOWN"
        elif row.get("candidate_answer", "") == expected[key]["answer"]:
            status = "EXACT_MATCH"
        else:
            status = "MISMATCH"
        submitted_keys[key] = submitted_keys.get(key, 0) + 1
        row_audit.append({
            "question_id": key[0],
            "blind_system_id": key[1],
            "status": status,
            "submitted_answer_sha256": answer_hash(row.get("candidate_answer", "")),
            "original_answer_sha256": answer_hash(expected[key]["answer"]) if key in expected else None,
        })
    ANNOTATIONS.mkdir(parents=True, exist_ok=True)
    fields = ["question_id", "blind_system_id", "question", "reference_answer", "candidate_answer", *SCORE_FIELDS]
    with V2.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for question in questions:
            for label in ("A", "B", "C"):
                row = expected[(question["question_id"], label)]
                writer.writerow({
                    "question_id": question["question_id"],
                    "blind_system_id": label,
                    "question": question["question"],
                    "reference_answer": question["reference_answer"],
                    "candidate_answer": row["answer"],
                    **{field: "" for field in SCORE_FIELDS},
                })
    GUIDELINES.write_text(
        """# Blinded human evaluation guidelines (v2)

Evaluate the candidate answer shown for the question and record only the five
requested scores. Do not attempt to identify which retrieval system produced
the answer.

Use the 1–5 scale for correctness, completeness, groundedness, and relevance:

- 1 = very poor or incorrect
- 2 = mostly incorrect, substantially incomplete, or weakly supported
- 3 = partially correct, complete, supported, or relevant
- 4 = mostly correct, complete, supported, and relevant
- 5 = fully correct, complete, clearly grounded, and directly relevant

Enter `1` for unsupported claim when the answer contains a factual claim not
supported by the supplied evidence or reference answer; otherwise enter `0`.
Judge every answer independently. Do not use lexical overlap or ROUGE as a
substitute for factual judgment, and do not infer hidden system identities.
""",
        encoding="utf8",
    )
    audit = {
        "submitted_file": str(SUBMITTED),
        "original_generation_count": len(generated),
        "question_count": len(questions),
        "blind_randomization_seed": seed,
        "submitted_row_count": len(submitted),
        "submitted_status_counts": {
            status: sum(row["status"] == status for row in row_audit)
            for status in ("EXACT_MATCH", "MISMATCH", "MISSING", "DUPLICATE", "UNKNOWN")
        },
        "verification_table": verification,
        "submitted_rows": row_audit,
        "scores_transferred": 0,
        "authoritative_rebuilt_file": str(V2),
    }
    AUDIT_JSON.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf8")
    counts = audit["submitted_status_counts"]
    lines = [
        "# Human CSV integrity audit",
        "",
        f"- Original answers verified: {len(generated)}",
        f"- Submitted rows inspected: {len(submitted)}",
        f"- Randomization seed: `{seed}`",
        "",
        "## Submitted-row status counts",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    lines.extend(f"| {key} | {value} |" for key, value in counts.items())
    submitted_note = (
        "No submitted human scores were transferred because the submitted file was not available at audit time."
        if not submitted
        else "No submitted human scores were transferred because the submitted rows contain candidate-answer mismatches."
    )
    lines += [
        "",
        "Candidate answers in the authoritative v2 file were generated only from the original checkpoint.",
        submitted_note,
        "",
        "## Verification table",
        "",
        "| Question | A system | A hash | B system | B hash | C system | C hash |",
        "|---|---|---|---|---|---|---|",
    ]
    lines.extend(
        f"| {row['question_id']} | {row['A_system']} | `{row['A_answer_sha256'][:12]}` | "
        f"{row['B_system']} | `{row['B_answer_sha256'][:12]}` | {row['C_system']} | `{row['C_answer_sha256'][:12]}` |"
        for row in verification
    )
    AUDIT_MD.write_text("\n".join(lines) + "\n", encoding="utf8")
    print(json.dumps({"rebuilt_rows": 78, "audit_counts": counts, "v2": str(V2)}, indent=2))


if __name__ == "__main__":
    main()
