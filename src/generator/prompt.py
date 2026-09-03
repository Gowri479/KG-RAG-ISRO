"""Prompt templates for context-grounded question answering."""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a factual QA assistant. You must answer ONLY using the context "
    "provided below. Do not use any prior knowledge. Do not be conversational. "
    "If the answer is not in the context, respond with exactly: I don't know. "
    "Never say anything else if the answer is not found."
)


def build_user_prompt(context: str, question: str) -> str:
    """Build the user prompt with explicit context and answer boundaries."""
    return (
        f"CONTEXT:\n{context.strip()}\n\n"
        f"QUESTION:\n{question.strip()}\n\n"
        "ANSWER (based only on the context above):"
    )

