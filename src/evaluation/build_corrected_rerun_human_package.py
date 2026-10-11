"""Build the blinded rater package for the pre-registered corrected rerun (§8).

Each rater gets an identical CSV with, per question: the reference and
acceptable answers, a shared source-evidence pack (supporting triples and the
text of supporting chunks), and the three blinded candidates.

    python -m src.evaluation.build_corrected_rerun_human_package --raters 2
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "data" / "relation_aware_benchmark" / "relation_aware_qa_v2.json"
CHUNKS = ROOT / "data" / "corpus" / "chunks.jsonl"
TRIPLES = ROOT / "data" / "corpus" / "triples.jsonl"
RERUN = ROOT / "data" / "results" / "corrected_rerun"
RESULTS = RERUN / "generation_results.jsonl"
RUN_MANIFEST = RERUN / "run_manifest.json"
MAPPING = RERUN / "human_eval_mapping.json"
PACKAGE_MANIFEST = RERUN / "human_eval_package_manifest.json"
ANNOTATIONS = ROOT / "data" / "annotations"
GUIDELINES = ANNOTATIONS / "corrected_rerun_human_eval_guidelines.md"
SEED = 20261008
SYSTEMS = ("V_VANILLA", "A_CURRENT", "C_TWO_STAGE")
LABELS = ("A", "B", "C")
DIMENSIONS = ("correctness", "completeness", "groundedness", "relevance", "unsupported_claim")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def answer_hash(text: str) -> str:
    return sha256_bytes(" ".join((text or "").casefold().split()).encode())


def rater_csv(index: int) -> Path:
    return ANNOTATIONS / f"corrected_rerun_human_eval_rater{index}.csv"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]


def evidence_pack(question: dict, chunks: dict[str, dict], triples: dict[str, dict]) -> str:
    lines = ["SUPPORTING FACTS:"]
    for triple_id in question["supporting_triples"]:
        t = triples[triple_id]
        lines.append(f"- {t['subject']} -> {t['relation']} -> {t['object']}")
    lines.append("")
    lines.append("SOURCE TEXT:")
    for chunk_id in question["supporting_chunks"]:
        c = chunks[chunk_id]
        lines.append(f"[{c['document_id']}] {' '.join(c['text'].split())}")
    return "\n".join(lines)


def main(raters: int) -> None:
    manifest = json.loads(RUN_MANIFEST.read_text(encoding="utf8"))
    if manifest["rows"] != manifest["expected_rows"] or manifest["integrity_failures"]:
        raise RuntimeError("Rerun is incomplete or has integrity failures; package not built")
    questions = json.loads(BENCHMARK.read_text(encoding="utf8"))["questions"]
    chunks = {c["chunk_id"]: c for c in load_jsonl(CHUNKS)}
    triples = {t["triple_id"]: t for t in load_jsonl(TRIPLES)}
    answers = {(r["question_id"], r["system"]): r["answer"] for r in load_jsonl(RESULTS)}

    rng = random.Random(SEED)
    rows, mapping = [], {}
    for question in questions:
        qid = question["question_id"]
        labels = list(LABELS)
        rng.shuffle(labels)
        label_to_system = dict(zip(labels, SYSTEMS))
        mapping[qid] = {
            "label_to_system": label_to_system,
            "label_hashes": {label: answer_hash(answers[(qid, system)]) for label, system in label_to_system.items()},
        }
        row = {
            "question_id": qid,
            "question": question["question"],
            "reference_answer": question["reference_answer"],
            "acceptable_answers": " | ".join(question["acceptable_answers"]),
            "evidence_pack": evidence_pack(question, chunks, triples),
        }
        for label in LABELS:
            row[f"candidate_{label}"] = answers[(qid, label_to_system[label])]
        for dimension in DIMENSIONS:
            for label in LABELS:
                row[f"{label}_{dimension}"] = ""
        rows.append(row)

    ANNOTATIONS.mkdir(parents=True, exist_ok=True)
    for index in range(1, raters + 1):
        with rater_csv(index).open("w", encoding="utf8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    GUIDELINES.write_text(GUIDELINES_TEXT, encoding="utf8")
    MAPPING.write_text(json.dumps({"seed": SEED, "systems": list(SYSTEMS), "mapping": mapping}, indent=2) + "\n", encoding="utf8")
    PACKAGE_MANIFEST.write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "questions": len(rows),
        "candidates": len(rows) * len(SYSTEMS),
        "raters": raters,
        "rater_files": [str(rater_csv(i).relative_to(ROOT)) for i in range(1, raters + 1)],
        "generation_results_sha256": sha256_bytes(RESULTS.read_bytes()),
        "mapping_sha256": sha256_bytes(MAPPING.read_bytes()),
        "warning": "Raters must not open the mapping file.",
    }, indent=2) + "\n", encoding="utf8")
    print(f"built {raters} rater files with {len(rows)} questions")


GUIDELINES_TEXT = """# Corrected-rerun blinded human evaluation

Pre-registered protocol: `reports/preregistration_corrected_rerun.md` §8.
Score independently. Do not discuss items with the other rater until both
files are complete. Do not open `data/results/corrected_rerun/`.

For each question you get the reference answer, the acceptable answers and an
**evidence pack** (supporting facts and official source text). It is the same
for all three candidates (A, B, C). Candidate letters are shuffled per
question; do not try to identify systems.

## Scores (1-5)

- **Correctness**: 1 substantially incorrect, 2 mostly incorrect,
  3 partially correct, 4 mostly correct, 5 fully correct.
  Some questions have several correct answers (e.g. "Which organization
  developed a payload carried by Aditya-L1?"). The reference is **one
  example**: any answer the evidence pack supports as a member of the correct
  set counts as correct. Do not penalise wording, acronym, alias or date-format
  differences.
- **Completeness**: 1 misses essentially all required information,
  3 partial, 5 complete.
- **Groundedness**: judged against the evidence pack. 1 largely unsupported,
  3 partially supported, 5 fully supported. A correct "I don't know" makes no
  claim and is fully grounded, but scores low on correctness and completeness.
- **Relevance**: 1 largely irrelevant, 3 partially relevant, 5 directly relevant.

## Unsupported claim (0/1)

1 if the answer makes at least one material claim that the evidence pack does
not support, otherwise 0.

Fill every score cell. If an answer genuinely cannot be judged, still score it
and note the reason outside the CSV.
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raters", type=int, default=2)
    main(parser.parse_args().raters)
