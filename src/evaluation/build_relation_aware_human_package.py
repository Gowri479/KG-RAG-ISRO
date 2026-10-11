"""Build and validate the blinded relation-aware human-evaluation package."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "data" / "relation_aware_benchmark" / "relation_aware_qa_v2.json"
GENERATIONS = ROOT / "data" / "results" / "relation_aware" / "generation_results.jsonl"
ANNOTATIONS = ROOT / "data" / "annotations"
RESULTS = ROOT / "data" / "results" / "relation_aware"
CSV_PATH = ANNOTATIONS / "relation_aware_human_eval.csv"
GUIDELINES = ANNOTATIONS / "relation_aware_human_eval_guidelines.md"
MAPPING = RESULTS / "relation_aware_human_eval_mapping.json"
MANIFEST = RESULTS / "human_eval_manifest.json"
AUDIT = RESULTS / "human_eval_integrity_audit.json"
SEED = 20261008
SYSTEMS = (
    "vanilla_dense_rag",
    "corrected_structured_kg_rag",
    "relation_aware_kg_rag",
)
SCORE_FIELDS = (
    "A_correctness", "B_correctness", "C_correctness",
    "A_completeness", "B_completeness", "C_completeness",
    "A_groundedness", "B_groundedness", "C_groundedness",
    "A_relevance", "B_relevance", "C_relevance",
    "A_unsupported_claim", "B_unsupported_claim", "C_unsupported_claim",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_hash(answer: str) -> str:
    normalized = re.sub(r"\s+", " ", (answer or "").strip().casefold())
    return hashlib.sha256(normalized.encode("utf8")).hexdigest()


def load_rows() -> tuple[list[dict], dict[tuple[str, str], dict]]:
    questions = json.loads(BENCHMARK.read_text(encoding="utf8"))["questions"]
    questions = [question for question in questions if question["kg_required"] == "YES"]
    rows = [
        json.loads(line)
        for line in GENERATIONS.read_text(encoding="utf8").splitlines()
        if line.strip()
    ]
    indexed = {(row["question_id"], row["system"]): row for row in rows}
    expected = {(question["question_id"], system) for question in questions for system in SYSTEMS}
    if len(questions) != 60 or len(rows) != 504 or set(indexed) != {
        (row["question_id"], row["system"]) for row in rows
    }:
        raise RuntimeError("Frozen benchmark or generation package has unexpected integrity")
    if not expected.issubset(indexed):
        raise RuntimeError("Required question/system answer is missing")
    return questions, indexed


def build() -> dict:
    questions, indexed = load_rows()
    rng = random.Random(SEED)
    annotation_rows = []
    mapping = {}
    for question in questions:
        labels = ["A", "B", "C"]
        rng.shuffle(labels)
        mapping[question["question_id"]] = {}
        candidates = {}
        for label, system in zip(labels, SYSTEMS):
            answer = indexed[(question["question_id"], system)]["answer"]
            candidates[label] = answer
            mapping[question["question_id"]][label] = {
                "system": system,
                "normalized_answer_sha256": normalized_hash(answer),
            }
        annotation_rows.append({
            "question_id": question["question_id"],
            "question": question["question"],
            "candidate_A": candidates["A"],
            "candidate_B": candidates["B"],
            "candidate_C": candidates["C"],
            **{field: "" for field in SCORE_FIELDS},
        })
    ANNOTATIONS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf8") as handle:
        fields = list(annotation_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(annotation_rows)
    MAPPING.write_text(json.dumps({
        "warning": "Keep this file separate from the annotator-facing CSV.",
        "randomization_seed": SEED,
        "systems": list(SYSTEMS),
        "mapping": mapping,
    }, indent=2) + "\n", encoding="utf8")
    GUIDELINES.write_text(
        """# Relation-aware blinded human evaluation guidelines

## Scope

Evaluate exactly 60 KG_REQUIRED questions and 180 candidate answers. The
candidate systems are blinded as A, B, and C independently for each question.
Do not infer, discuss, or attempt to recover system identities. Do not use
ROUGE, coverage, answer length, filenames, or previous experiment results.

Read each question and its three candidate answers independently. Judge each
answer against the question and the supplied reference/evidence package
available to the evaluator. Do not reward lexical overlap alone and do not
penalize harmless wording, acronym, date-format, or alias differences.

## Scores

Correctness:
- 1 = substantially incorrect
- 2 = mostly incorrect / major factual problems
- 3 = partially correct
- 4 = mostly correct
- 5 = fully correct

Completeness:
- 1 = misses essentially all required information
- 2 = major omissions
- 3 = partial coverage
- 4 = mostly complete
- 5 = complete

Groundedness:
- 1 = largely unsupported
- 2 = weakly supported
- 3 = partially grounded
- 4 = well grounded
- 5 = fully grounded in supplied evidence

Relevance:
- 1 = largely irrelevant
- 2 = substantial irrelevant content
- 3 = partially relevant
- 4 = mostly relevant
- 5 = directly relevant

Unsupported claim:
- 0 = no materially unsupported claim
- 1 = at least one materially unsupported claim

Score all three candidates for every question. Do not select a winner before
all scoring is complete. Leave a score blank only if the answer genuinely
cannot be judged, and record the reason outside the CSV.
""",
        encoding="utf8",
    )
    return validate()


def validate() -> dict:
    questions, indexed = load_rows()
    with CSV_PATH.open(newline="", encoding="utf8") as handle:
        rows = list(csv.DictReader(handle))
    mapping = json.loads(MAPPING.read_text(encoding="utf8"))["mapping"]
    expected_ids = {question["question_id"] for question in questions}
    candidate_hash_matches = 0
    mapping_errors = []
    for row in rows:
        qid = row["question_id"]
        for label in ("A", "B", "C"):
            info = mapping[qid][label]
            if normalized_hash(row[f"candidate_{label}"]) != info["normalized_answer_sha256"]:
                mapping_errors.append((qid, label))
            else:
                candidate_hash_matches += 1
    score_blanks = all(not row[field] for row in rows for field in SCORE_FIELDS)
    audit = {
        "questions": len(rows),
        "unique_questions": len({row["question_id"] for row in rows}),
        "candidate_answers": len(rows) * 3,
        "triplets": sum(all(row.get(f"candidate_{label}") is not None for label in ("A", "B", "C")) for row in rows),
        "duplicate_question_ids": len(rows) - len({row["question_id"] for row in rows}),
        "missing_question_ids": sorted(expected_ids - {row["question_id"] for row in rows}),
        "candidate_hash_matches": candidate_hash_matches,
        "candidate_hash_mismatches": mapping_errors,
        "system_label_leakage": any(system in CSV_PATH.read_text(encoding="utf8") for system in SYSTEMS),
        "blank_score_fields": score_blanks,
        "generation_results_sha256": digest(GENERATIONS),
        "benchmark_sha256": digest(BENCHMARK),
        "mapping_sha256": digest(MAPPING),
        "valid": (
            len(rows) == 60
            and len({row["question_id"] for row in rows}) == 60
            and candidate_hash_matches == 180
            and not mapping_errors
            and score_blanks
            and not any(system in CSV_PATH.read_text(encoding="utf8") for system in SYSTEMS)
        ),
    }
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf8")
    if not audit["valid"]:
        raise RuntimeError(f"Human package integrity failed: {audit}")
    manifest = {
        "benchmark_sha256": digest(BENCHMARK),
        "generation_results_sha256": digest(GENERATIONS),
        "question_count": 60,
        "candidate_answer_count": 180,
        "system_count": 3,
        "systems": ["A", "B", "C"],
        "randomization_seed": SEED,
        "rubric_version": "relation-aware-human-v1-1-to-5-plus-binary-unsupported-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": {
            "benchmark": str(BENCHMARK.relative_to(ROOT)),
            "generation_results": str(GENERATIONS.relative_to(ROOT)),
            "annotator_csv": str(CSV_PATH.relative_to(ROOT)),
            "guidelines": str(GUIDELINES.relative_to(ROOT)),
            "private_mapping": str(MAPPING.relative_to(ROOT)),
        },
        "validation": audit,
        "scores_analyzed": False,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")
    return audit


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
