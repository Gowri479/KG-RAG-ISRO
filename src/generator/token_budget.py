"""Exact prompt-token budgeting for the local Ollama model.

Ollama silently truncates a prompt longer than ``num_ctx`` and keeps its TAIL,
so evidence placed first in the context (KG facts, top-ranked passages) is the
part that disappears. Every prompt must therefore be fitted to the window
before it is sent, counting real model tokens rather than whitespace words.

Token counts use the tokenizer extracted from the model's own GGUF file
(``src/generator/extract_tokenizer.py``) and the Ollama template
``[INST] {system} {prompt} [/INST]``. Counts match Ollama's
``prompt_eval_count`` exactly.
"""

from __future__ import annotations

import math
import os
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TOKENIZER_PATH = Path(
    os.environ.get(
        "KG_RAG_TOKENIZER",
        ROOT / "config" / "tokenizer" / "mistral-7b-instruct-q4_K_M" / "tokenizer.json",
    )
)
DEFAULT_NUM_CTX = 2048
DEFAULT_RESPONSE_RESERVE = 256
SAFETY_MARGIN = 8
# Ollama counts BOS plus one extra token beyond the encoded template.
TEMPLATE_OVERHEAD = 2
EVIDENCE = "\x00EVIDENCE\x00"
# Worst observed ratio on this corpus is ~2.1 characters per token.
FALLBACK_CHARS_PER_TOKEN = 2.0


class ContextWindowError(ValueError):
    """Raised when the fixed parts of a prompt alone exceed the context window."""


@lru_cache(maxsize=1)
def _tokenizer():
    try:
        from tokenizers import Tokenizer

        return Tokenizer.from_file(str(TOKENIZER_PATH))
    except Exception as exc:  # pragma: no cover - depends on local files
        warnings.warn(
            f"Model tokenizer unavailable ({exc}); using a conservative "
            f"{FALLBACK_CHARS_PER_TOKEN} chars/token estimate.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def count_tokens(text: str) -> int:
    """Count model tokens in ``text`` without special tokens."""
    if not text:
        return 0
    tokenizer = _tokenizer()
    if tokenizer is None:
        return math.ceil(len(text) / FALLBACK_CHARS_PER_TOKEN)
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


def prompt_tokens(system: str, user: str) -> int:
    """Tokens Ollama will evaluate for this system and user prompt."""
    return TEMPLATE_OVERHEAD + count_tokens(f"[INST] {system} {user} [/INST]")


def window_limit(options: dict[str, Any] | None) -> int:
    """Maximum prompt tokens that fit beside the reserved response tokens."""
    options = options or {}
    num_ctx = int(options.get("num_ctx") or DEFAULT_NUM_CTX)
    num_predict = options.get("num_predict")
    reserve = int(num_predict) if num_predict is not None and int(num_predict) > 0 else DEFAULT_RESPONSE_RESERVE
    return num_ctx - reserve - SAFETY_MARGIN


def with_context_window(options: dict[str, Any] | None) -> dict[str, Any]:
    """Return options with an explicit ``num_ctx`` so the window is never implicit."""
    resolved = dict(options or {})
    resolved.setdefault("num_ctx", DEFAULT_NUM_CTX)
    return resolved


def trim_to_tokens(text: str, max_tokens: int) -> str:
    """Keep the head of ``text`` within ``max_tokens``, cutting at a whitespace boundary."""
    if max_tokens <= 0 or not text:
        return ""
    tokenizer = _tokenizer()
    if tokenizer is None:
        cut = int(max_tokens * FALLBACK_CHARS_PER_TOKEN)
        if len(text) <= cut:
            return text
    else:
        encoding = tokenizer.encode(text, add_special_tokens=False)
        if len(encoding.ids) <= max_tokens:
            return text
        cut = encoding.offsets[max_tokens][0]
    head = text[:cut]
    boundary = max(head.rfind(" "), head.rfind("\n"))
    if boundary >= int(len(head) * 0.8):
        head = head[:boundary]
    head = head.rstrip()
    while head and count_tokens(head) > max_tokens:
        head = head[: int(len(head) * 0.95)].rstrip()
    return head


def evidence_budget(
    system: str,
    user_template: str,
    options: dict[str, Any] | None,
    cap: int | None = None,
) -> int:
    """Tokens available for the ``EVIDENCE`` slot of ``user_template``."""
    if EVIDENCE not in user_template:
        raise ValueError("user_template must contain the EVIDENCE placeholder")
    overhead = prompt_tokens(system, user_template.replace(EVIDENCE, ""))
    available = window_limit(options) - overhead
    if available <= 0:
        raise ContextWindowError(
            f"Fixed prompt needs {overhead} tokens; window allows {window_limit(options)}."
        )
    return min(available, cap) if cap is not None else available


def fit_evidence(
    system: str,
    user_template: str,
    evidence: str,
    options: dict[str, Any] | None,
    cap: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """Fill ``user_template`` with as much of ``evidence`` (head first) as fits."""
    budget = evidence_budget(system, user_template, options, cap)
    full = count_tokens(evidence)
    fitted = trim_to_tokens(evidence, budget)
    user = user_template.replace(EVIDENCE, fitted)
    while prompt_tokens(system, user) > window_limit(options) and fitted:
        budget -= SAFETY_MARGIN
        fitted = trim_to_tokens(fitted, budget)
        user = user_template.replace(EVIDENCE, fitted)
    used = count_tokens(fitted)
    return user, {
        "num_ctx": int(with_context_window(options)["num_ctx"]),
        "context_tokens_full": full,
        "context_tokens_used": used,
        "context_tokens_dropped": max(0, full - used),
        "context_trimmed": used < full,
        "prompt_tokens_expected": prompt_tokens(system, user),
    }
