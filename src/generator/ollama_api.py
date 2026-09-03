"""Ollama REST API integration for local model generation."""

from __future__ import annotations

import re

import requests

from src.generator.prompt import SYSTEM_PROMPT, build_user_prompt

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral:7b-instruct-q4_K_M"
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


def generate(query: str, context: str) -> str:
    """Ask Ollama to answer from the supplied retrieval context only."""
    if not query or not query.strip():
        return UNKNOWN

    context_text = (context or "").strip()

    if not context_text:
        return UNKNOWN

    payload = {
        "model": MODEL_NAME,
        "system": SYSTEM_PROMPT,
        "prompt": build_user_prompt(context_text, query),
        "stream": False,
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            if "response" in data and isinstance(data["response"], str):
                return _clean_response(data["response"])
            if isinstance(data.get("message"), dict):
                answer = data["message"].get("content", "")
                if answer:
                    return _clean_response(answer)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and isinstance(item.get("response"), str):
                    return _clean_response(item["response"])
    except Exception:
        pass

    return UNKNOWN


if __name__ == "__main__":
    print(generate("What is ISRO?", "ISRO is the Indian Space Research Organisation."))

