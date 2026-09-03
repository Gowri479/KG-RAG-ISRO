"""Entity extraction utilities for graph construction."""

from __future__ import annotations

import logging
import re
from typing import Any

import spacy

logger = logging.getLogger(__name__)

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:  # pragma: no cover - fallback for local installs without model
    logger.warning("spaCy model 'en_core_web_sm' not found; using fallback entity heuristics.")
    nlp = spacy.blank("en")


def _fallback_entities(text: str) -> list[dict[str, Any]]:
    """Rule-based fallback for common mission, organisation, and place names used in ISRO docs."""
    entities: list[dict[str, Any]] = []
    patterns = [
        (re.compile(r"\bChandrayaan[- ]?\d+(?:\.[A-Za-z0-9-]+)?\b", re.I), "MISSION"),
        (re.compile(r"\bGaganyaan\b|\bAditya[- ]?L1\b|\bXPoSat\b|\bNISAR\b|\bPSLV\b|\bGSLV\b|\bLVM3\b", re.I), "MISSION"),
        (re.compile(r"\bISRO\b|\bIndian Space Research Organisation\b|\bNASA\b|\bESA\b|\bIN-SPACe\b", re.I), "ORG"),
    ]

    for pattern, label in patterns:
        for match in pattern.finditer(text):
            value = match.group(0).strip()
            if not value:
                continue
            if value.lower() in {"chandrayaan", "gaganyaan", "aditya", "l1", "xposat", "nisar", "pslv", "gslv", "lvm3"}:
                continue
            entities.append({
                "text": value,
                "label": label,
                "start": match.start(),
                "end": match.end(),
            })

    place_candidates = re.finditer(r"\b(?:from|in|at|near|from\s+the|from\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text)
    for match in place_candidates:
        value = match.group(1).strip()
        if not value:
            continue
        if value.lower() in {"was", "launched", "by", "the", "from", "in", "at", "near"}:
            continue
        if value.lower() in {"isro", "chandrayaan", "gaganyaan", "aditya", "xposat", "nisar", "pslv", "gslv", "lvm3"}:
            continue
        entities.append({
            "text": value,
            "label": "PLACE",
            "start": match.start(1),
            "end": match.end(1),
        })

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for item in entities:
        key = (item["text"].lower(), item["start"], item["end"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    return sorted(deduped, key=lambda item: item["start"])


def extract_entities(text: str) -> list[dict[str, Any]]:
    """Return a list of recognized entities with label, text, and span metadata."""
    if not text or not text.strip():
        return []

    doc = nlp(text)
    entities: list[dict[str, Any]] = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text.strip(),
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
        })

    if entities:
        return sorted(entities, key=lambda item: item["start"])

    return _fallback_entities(text)


if __name__ == "__main__":
    sample = "Chandrayaan-3 was launched by ISRO from Sriharikota."
    print(extract_entities(sample))

