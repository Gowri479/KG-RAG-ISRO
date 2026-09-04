"""Combine KG and vector retrieval into a single context string."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import spacy
except ImportError:  # pragma: no cover
    spacy = None

try:
    from src.retriever.faiss_retriever import get_passage_context
    from src.retriever.kg_retriever import get_kg_context
except ImportError:  # pragma: no cover
    from retriever.faiss_retriever import get_passage_context
    from retriever.kg_retriever import get_kg_context


def _entity_fallback(query: str) -> list[str]:
    pattern = r"\b[A-Z][A-Za-z0-9-]+(?:\s+[A-Z][A-Za-z0-9-]+)*\b|\b[A-Z]{2,}\b"
    candidates = re.findall(pattern, query)
    return [candidate.strip() for candidate in candidates if candidate.strip()][:10]


def _query_keywords(query: str) -> list[str]:
    return [word.lower() for word in query.split() if len(word) > 3]


def _extract_entities(query: str) -> list[str]:
    if not query:
        return []

    if spacy is None:
        return _entity_fallback(query)

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        try:
            nlp = spacy.blank("en")
        except Exception:
            return _entity_fallback(query)

    doc = nlp(query)
    entities = [ent.text.strip() for ent in getattr(doc, "ents", []) if ent.text.strip()]
    if entities:
        return entities[:10]
    return _entity_fallback(query)


def retrieve(query: str, passage_limit: int = 5, max_tokens: int = 4000) -> str:
    """Merge KG and vector-context evidence into a single retrieval string."""
    if not query or not query.strip():
        return ""

    keywords = _query_keywords(query)
    entities = _extract_entities(query)
    entities = [
        entity for entity in entities
        if any(keyword in entity.lower() for keyword in keywords)
    ]
    kg_context = get_kg_context(entities)
    passage_context = get_passage_context(query, top_k=passage_limit)

    combined_parts = []
    if kg_context.strip():
        combined_parts.append(kg_context.strip())
    if passage_context.strip():
        combined_parts.append(passage_context.strip())

    merged = "\n\n".join(combined_parts)
    tokens = merged.split()
    if len(tokens) > max_tokens:
        merged = " ".join(tokens[:max_tokens])
    return merged.strip()


if __name__ == "__main__":
    print(retrieve("What is ISRO?"))

