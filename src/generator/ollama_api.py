"""Ollama REST API integration for local model generation."""

from __future__ import annotations

import re
import time
import warnings

import requests

from src.generator.prompt import SYSTEM_PROMPT, build_user_prompt
from src.generator.token_budget import EVIDENCE, fit_evidence, with_context_window

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "mistral:7b-instruct-q4_K_M"
OLLAMA_TIMEOUT = (5, 180)
UNKNOWN = "I don't know."
FILLER_MARKERS = (
    "don't hesitate",
    "glad",
    "happy to help",
    "let me know",
    "please provide",
    "additional context",
    "let me help",
    "for me to help",
    "more information",
)


def _clean_response(response_text: str) -> str:
    """Remove common answer labels and reject conversational non-answers."""
    answer = response_text.strip()
    if not answer or any(marker in answer.lower() for marker in FILLER_MARKERS):
        return UNKNOWN

    answer = re.sub(r"^(?:answer\s*:\s*|response\s*:\s*)", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"^(?:based on the context(?: above)?[,]?\s*)", "", answer, flags=re.IGNORECASE)
    answer = answer.strip()
    return answer or UNKNOWN


def generate(query: str, context: str, options: dict | None = None, system_prompt: str | None = None) -> str:
    """Ask Ollama to answer from the supplied retrieval context only."""
    answer, _ = generate_with_metrics(query, context, options=options, system_prompt=system_prompt)
    return answer


def generate_with_metrics(
    query: str,
    context: str,
    options: dict | None = None,
    system_prompt: str | None = None,
) -> tuple[str, dict]:
    """Generate with timing and Ollama telemetry while preserving the normal output."""
    started = time.perf_counter()
    metrics = {
        "prompt_construction_ms": 0.0,
        "request_latency_ms": 0.0,
        "total_generation_ms": 0.0,
        "prompt_chars": 0,
        "prompt_tokens": None,
        "generated_tokens": None,
        "eval_duration_ns": None,
        "prompt_eval_duration_ns": None,
        "status": "unknown",
        "failure": None,
    }
    if not query or not query.strip():
        metrics["status"] = "empty_query"
        return UNKNOWN, metrics

    context_text = (context or "").strip()

    if not context_text:
        metrics["status"] = "empty_context"
        return UNKNOWN, metrics

    prompt_started = time.perf_counter()
    system = system_prompt or SYSTEM_PROMPT
    options = with_context_window(options)
    # Fit the context to the window ourselves; Ollama would otherwise drop its head.
    user_prompt, fit = fit_evidence(system, build_user_prompt(EVIDENCE, query), context_text, options)
    metrics.update(fit)
    metrics["prompt_construction_ms"] = round((time.perf_counter() - prompt_started) * 1000, 4)
    metrics["prompt_chars"] = len(user_prompt)
    payload = {
        "model": MODEL_NAME,
        "system": system,
        "prompt": user_prompt,
        "stream": False,
        "keep_alive": -1,
        "options": options,
    }

    try:
        request_started = time.perf_counter()
        response = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        metrics["request_latency_ms"] = round((time.perf_counter() - request_started) * 1000, 4)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            metrics["prompt_tokens"] = data.get("prompt_eval_count")
            metrics["ollama_truncated"] = _ollama_truncated(metrics["prompt_tokens"], metrics["prompt_tokens_expected"])
            metrics["generated_tokens"] = data.get("eval_count")
            metrics["eval_duration_ns"] = data.get("eval_duration")
            metrics["prompt_eval_duration_ns"] = data.get("prompt_eval_duration")
            if "response" in data and isinstance(data["response"], str):
                metrics["status"] = "ok"
                return _clean_response(data["response"]), _finish_metrics(metrics, started)
            if isinstance(data.get("message"), dict):
                answer = data["message"].get("content", "")
                if answer:
                    metrics["status"] = "ok"
                    return _clean_response(answer), _finish_metrics(metrics, started)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and isinstance(item.get("response"), str):
                    metrics["status"] = "ok"
                    return _clean_response(item["response"]), _finish_metrics(metrics, started)
        metrics["status"] = "invalid_response"
    except Exception as exc:
        metrics["status"] = "error"
        metrics["failure"] = type(exc).__name__

    return UNKNOWN, _finish_metrics(metrics, started)


def _ollama_truncated(observed: int | None, expected: int) -> bool:
    """Flag a prompt Ollama evaluated only partially despite pre-fitting."""
    if observed is None or observed >= expected - 2:
        return False
    warnings.warn(
        f"Ollama evaluated {observed} of {expected} prompt tokens; the prompt was truncated.",
        RuntimeWarning,
        stacklevel=3,
    )
    return True


def _finish_metrics(metrics: dict, started: float) -> dict:
    metrics["total_generation_ms"] = round((time.perf_counter() - started) * 1000, 4)
    return metrics


if __name__ == "__main__":
    print(generate("What is ISRO?", "ISRO is the Indian Space Research Organisation."))
