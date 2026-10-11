import json
from pathlib import Path

import pytest

from src.generator import kg_grounded_generator, ollama_api
from src.generator.kg_grounded_generator import KGFact, generate_condition
from src.generator.prompt import SYSTEM_PROMPT, build_user_prompt
from src.generator.token_budget import (
    EVIDENCE,
    ContextWindowError,
    count_tokens,
    fit_evidence,
    prompt_tokens,
    trim_to_tokens,
    window_limit,
)

ROOT = Path(__file__).resolve().parents[1]
OPTIONS = {"temperature": 0.1, "num_predict": 150, "num_ctx": 2048}
QUESTION = "Which organization developed SUIT?"


def _long_context():
    return "TOP-RANKED EVIDENCE: SUIT was developed by IUCAA. " + "filler passage text. " * 2000 + "LAST PASSAGE."


def test_prompt_count_matches_ollama_for_frozen_prompt():
    # Ollama reported prompt_eval_count=4578 for this prompt at num_ctx=8192.
    rows = (ROOT / "data/results/relation_aware/generation_results.jsonl").read_text(encoding="utf8").splitlines()
    row = next(
        r for r in map(json.loads, rows)
        if r["system"] == "relation_aware_kg_rag" and r["question_id"] == "rakg_005"
    )
    assert prompt_tokens(SYSTEM_PROMPT, row["final_prompt"]) == 4578


def test_trim_keeps_head_within_limit():
    trimmed = trim_to_tokens(_long_context(), 100)
    assert trimmed.startswith("TOP-RANKED EVIDENCE")
    assert "LAST PASSAGE" not in trimmed
    assert count_tokens(trimmed) <= 100


def test_fit_evidence_fits_window_and_keeps_question():
    user, report = fit_evidence(SYSTEM_PROMPT, build_user_prompt(EVIDENCE, QUESTION), _long_context(), OPTIONS)
    assert prompt_tokens(SYSTEM_PROMPT, user) <= window_limit(OPTIONS)
    assert QUESTION in user and "TOP-RANKED EVIDENCE" in user
    assert report["context_trimmed"] and report["context_tokens_dropped"] > 0


def test_short_context_is_untouched():
    user, report = fit_evidence(SYSTEM_PROMPT, build_user_prompt(EVIDENCE, QUESTION), "SUIT was developed by IUCAA.", OPTIONS)
    assert "SUIT was developed by IUCAA." in user
    assert not report["context_trimmed"]


def test_fixed_prompt_larger_than_window_raises():
    with pytest.raises(ContextWindowError):
        fit_evidence(SYSTEM_PROMPT, build_user_prompt(EVIDENCE, QUESTION * 1000), "x", OPTIONS)


class _Response:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


def test_generate_never_sends_oversized_prompt(monkeypatch):
    sent = {}

    def fake_post(url, json, timeout):
        sent.update(json)
        return _Response({"response": "IUCAA", "prompt_eval_count": prompt_tokens(json["system"], json["prompt"])})

    monkeypatch.setattr(ollama_api.requests, "post", fake_post)
    answer, metrics = ollama_api.generate_with_metrics(QUESTION, _long_context(), options=OPTIONS)
    assert answer == "IUCAA"
    assert sent["options"]["num_ctx"] == 2048
    assert prompt_tokens(sent["system"], sent["prompt"]) <= window_limit(OPTIONS)
    assert "TOP-RANKED EVIDENCE" in sent["prompt"] and sent["prompt"].rstrip().endswith("ANSWER (based only on the context above):")
    assert metrics["context_trimmed"] and not metrics["ollama_truncated"]


def test_generate_sets_explicit_window_when_options_missing(monkeypatch):
    sent = {}
    monkeypatch.setattr(ollama_api.requests, "post", lambda url, json, timeout: sent.update(json) or _Response({"response": "ok"}))
    ollama_api.generate_with_metrics(QUESTION, "short context")
    assert sent["options"]["num_ctx"] == 2048


def test_ollama_side_truncation_is_flagged(monkeypatch):
    monkeypatch.setattr(ollama_api.requests, "post", lambda url, json, timeout: _Response({"response": "x", "prompt_eval_count": 10}))
    with pytest.warns(RuntimeWarning):
        _, metrics = ollama_api.generate_with_metrics(QUESTION, "SUIT was developed by IUCAA. " * 20, options=OPTIONS)
    assert metrics["ollama_truncated"]


@pytest.mark.parametrize("condition", ["A_CURRENT", "B_STRUCTURED", "C_TWO_STAGE"])
def test_kg_conditions_fit_window_and_keep_kg_facts(monkeypatch, condition):
    calls = []

    def fake_generate(question, context, options=None, system_prompt=None):
        system = system_prompt or SYSTEM_PROMPT
        user, report = fit_evidence(system, build_user_prompt(EVIDENCE, question), context, options)
        calls.append((system, user, report))
        return "IUCAA", {"plan": None}

    monkeypatch.setattr(kg_grounded_generator, "generate_with_metrics", fake_generate)
    facts = [KGFact("path:1", "SUIT", "DEVELOPED_BY", "IUCAA", ("doc",), ("chunk",), ("url",))]
    current = "[RELATION-AWARE KG EVIDENCE] SUIT DEVELOPED_BY IUCAA " + _long_context()
    generate_condition(condition, QUESTION, current, facts, OPTIONS, 1, _long_context())
    for system, user, report in calls:
        assert prompt_tokens(system, user) <= window_limit(OPTIONS)
        assert not report["context_trimmed"], "evidence must be fitted before the window guard"
        assert user.count(QUESTION) >= 1
        assert "IUCAA" in user
