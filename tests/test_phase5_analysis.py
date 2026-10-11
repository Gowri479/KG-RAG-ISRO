import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "data" / "relational_benchmark" / "relational_qa_v1.json"
CHECKPOINT = ROOT / "data" / "results" / "relational_qa_v1" / "per_question_results.jsonl"


def test_phase5_freeze_and_trace_invariants():
    assert hashlib.sha256(BENCHMARK.read_bytes()).hexdigest() == (
        "7f5daf5419b0a4cb70e5c3d5eba021a8a39118b5a0fd235ce19b81f75324ffce"
    )
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf8"))
    assert len(benchmark["questions"]) == 62
    assert sum(q["kg_required"] == "YES" for q in benchmark["questions"]) == 26
    rows = [json.loads(line) for line in CHECKPOINT.read_text(encoding="utf8").splitlines() if line.strip()]
    assert len(rows) == 186
    assert len({(row["question_id"], row["system"]) for row in rows}) == 186


def test_phase5_outputs_cover_all_required_questions():
    path = ROOT / "data" / "results" / "relational_qa_v1" / "failure_analysis.json"
    if not path.exists():
        return
    result = json.loads(path.read_text(encoding="utf8"))
    assert len(result["per_question_diagnosis"]) == 26
    assert result["final_decision"] in {
        "A — RETRIEVAL BOTTLENECK",
        "B — EVIDENCE-FUSION / CONTEXT BOTTLENECK",
        "C — GENERATION / EVIDENCE-USE BOTTLENECK",
        "D — NO CLEAR SINGLE BOTTLENECK",
    }
