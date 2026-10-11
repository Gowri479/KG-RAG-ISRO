"""Small prompt-only diagnostic using frozen relation-aware KG-RAG contexts."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import requests

from src.evaluation.evaluate_relational_qa import metrics
from src.generator.ollama_api import MODEL_NAME, OLLAMA_TIMEOUT, SYSTEM_PROMPT, _clean_response
from src.generator.token_budget import EVIDENCE, fit_evidence

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "data" / "results" / "relation_aware"
GENERATIONS = RESULTS / "generation_results.jsonl"
BENCHMARK = ROOT / "data" / "relation_aware_benchmark" / "relation_aware_qa_v2.json"
OUTPUT = RESULTS / "prompt_diagnostic_results.jsonl"
OPTIONS = {"temperature": 0.1, "num_predict": 150, "num_ctx": 2048}
PROMPTS = {
    "P1_current": (
        "CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\n"
        "ANSWER (based only on the context above):"
    ),
    "P2_kg_priority": (
        "CONTEXT:\n{context}\n\n"
        "GENERATION INSTRUCTIONS:\n"
        "This is a relationship-intensive question. The structured KG evidence contains "
        "canonical relational facts and should be prioritized over other context when answering. "
        "Use only supplied evidence, preserve the direction of every relation, and do not combine "
        "unsupported facts. If the KG evidence does not answer the question, use the other supplied "
        "evidence only when it is consistent. If the evidence is insufficient, answer exactly: I don't know.\n\n"
        "QUESTION:\n{question}\n\nANSWER:"
    ),
    "P3_structured_evidence": (
        "EVIDENCE BLOCKS:\n"
        "The context contains labeled KG FACTS, TEXTUAL EVIDENCE, and SOURCE/PROVENANCE. "
        "Answer only from these blocks. Prefer KG FACTS for the requested relation, use TEXTUAL "
        "EVIDENCE for clarification, and use SOURCE/PROVENANCE to verify grounding. Preserve "
        "subject-to-object relation direction and do not infer a reversed relationship. "
        "If blocks conflict, prefer the canonical KG FACT only when it directly states the requested "
        "relation; otherwise answer exactly: I don't know.\n\n"
        "{context}\n\nQUESTION:\n{question}\n\nANSWER:"
    ),
}


def generate(prompt: str) -> tuple[str, dict]:
    payload = {
        "model": MODEL_NAME,
        "system": SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": False,
        "keep_alive": -1,
        "options": OPTIONS,
    }
    started = time.perf_counter()
    response = requests.post("http://127.0.0.1:11434/api/generate", json=payload, timeout=OLLAMA_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    answer = data.get("response", "") if isinstance(data, dict) else ""
    return _clean_response(answer), {
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "prompt_tokens": data.get("prompt_eval_count") if isinstance(data, dict) else None,
        "generated_tokens": data.get("eval_count") if isinstance(data, dict) else None,
    }


def main() -> None:
    questions = json.loads(BENCHMARK.read_text(encoding="utf8"))["questions"]
    selected = sorted(
        (question for question in questions if question["kg_required"] == "YES"),
        key=lambda question: question["question_id"],
    )[:18]
    frozen = {
        json.loads(line)["question_id"]: json.loads(line)
        for line in GENERATIONS.read_text(encoding="utf8").splitlines()
        if line.strip()
    }
    existing = {}
    if OUTPUT.exists():
        existing = {
            (row["question_id"], row["prompt_variant"]): row
            for row in (json.loads(line) for line in OUTPUT.read_text(encoding="utf8").splitlines() if line.strip())
        }
    with OUTPUT.open("a", encoding="utf8", newline="\n") as handle:
        for question in selected:
            base = frozen[question["question_id"]]
            context = base["context"]
            for variant, template in PROMPTS.items():
                key = (question["question_id"], variant)
                if key in existing:
                    continue
                user_template = template.replace("{question}", question["question"]).replace("{context}", EVIDENCE)
                prompt, fit = fit_evidence(SYSTEM_PROMPT, user_template, context, OPTIONS)
                answer, telemetry = generate(prompt)
                telemetry.update(fit)
                row = {
                    "question_id": question["question_id"],
                    "question": question["question"],
                    "reference_answer": question["reference_answer"],
                    "prompt_variant": variant,
                    "selection_rule": "First 18 KG-required questions by question_id, selected before diagnostic outcomes.",
                    "retrieval_source_system": "relation_aware_kg_rag",
                    "retrieval_context_sha256": hashlib.sha256(context.encode("utf8")).hexdigest(),
                    "context_token_count": len(context.split()),
                    "prompt": prompt,
                    "answer": answer,
                    "metrics": metrics(answer, question["reference_answer"]),
                    "generation_configuration": OPTIONS,
                    "telemetry": telemetry,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                existing[key] = row
                print(f"checkpointed {len(existing)}/54 {key}", flush=True)
    manifest = {
        "questions": [question["question_id"] for question in selected],
        "question_count": len(selected),
        "variants": list(PROMPTS),
        "expected_rows": 54,
        "retrieval_frozen": True,
        "retrieval_source": "frozen generation_results.jsonl relation_aware_kg_rag context",
        "model": MODEL_NAME,
        "options": OPTIONS,
        "benchmark_sha256": hashlib.sha256(BENCHMARK.read_bytes()).hexdigest(),
        "generation_results_sha256": hashlib.sha256(GENERATIONS.read_bytes()).hexdigest(),
        "not_a_benchmark": True,
    }
    (RESULTS / "prompt_diagnostic_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
