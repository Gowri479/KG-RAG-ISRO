"""Deterministic, reproducible query classification layer for ISRO domain QA.

Classifies incoming queries into four explicit methodological types:
A. Direct factual lookup (FACTUAL)
B. Entity-relational question (RELATIONAL)
C. Multi-hop relational question (MULTI_HOP)
D. Temporal/chronological question (TEMPORAL)

Used to analyze task-conditional retrieval utility and determine appropriate
graph neighborhood traversal bounds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class QueryType(str, Enum):
    FACTUAL = "Direct factual lookup"
    RELATIONAL = "Entity-relational question"
    MULTI_HOP = "Multi-hop relational question"
    TEMPORAL = "Temporal/chronological question"


@dataclass(frozen=True)
class QueryClassification:
    query: str
    query_type: QueryType
    suggested_hop_depth: int
    rationale: str
    detected_markers: list[str]


# Temporal & chronological indicators
TEMPORAL_PATTERNS = [
    r"\bwhen\b",
    r"\bwhat year\b",
    r"\bwhat date\b",
    r"\bwhich date\b",
    r"\bwhich year\b",
    r"\bhow long\b",
    r"\bduration\b",
    r"\btimeline\b",
    r"\bchronolog\w*\b",
    r"\border of\b",
    r"\bbefore\b",
    r"\bafter\b",
    r"\blaunch date\b",
    r"\binsertion date\b",
    r"\bhow many (?:days|months|years|hours|minutes)\b",
]

# Multi-hop relational indicators (bridging phrases, multi-step relationships)
MULTI_HOP_PATTERNS = [
    r"\bwhich (?:mission|spacecraft) launched (?:the )?(?:payload|instrument) that\b",
    r"\bwho was the (?:chairman|director|head) when\b",
    r"\bwhich center developed (?:the )?(?:instrument|payload) onboard\b",
    r"\bwhat is the connection between\b",
    r"\bhow does .* relate to .*\b",
    r"\bwhich launch vehicle carried .* to .*\b",
    r"\bwhat payload on .* observes .*\b",
    r"\bwhich payload developed by .* is onboard\b",
    r"\bwhere was the launch vehicle that carried .* developed\b",
]

# Entity-relational indicators (single-step relation query between entities)
RELATIONAL_PATTERNS = [
    r"\bwhich (?:payload|instrument|sensor)\b",
    r"\bwhat (?:payload|instrument|sensor)\b",
    r"\bwhich (?:launch vehicle|rocket|booster)\b",
    r"\bwhat (?:launch vehicle|rocket)\b",
    r"\bwho (?:developed|built|designed|manufactured|fabricated)\b",
    r"\bwhich (?:center|organization|institute|agency) (?:developed|built|operates)\b",
    r"\bwhat is the objective of\b",
    r"\bwhat does .* observe\b",
    r"\bwhich orbit\b",
    r"\bwhere is .* located\b",
    r"\bwho is the (?:director|chairman|leader|principal investigator)\b",
    r"\bwhat are the (?:payloads|instruments) of\b",
    r"\bwhat does .* carry\b",
]


def classify_query(query: str) -> QueryClassification:
    """Classify a question deterministically based on syntactic and semantic markers."""
    if not query or not query.strip():
        return QueryClassification(
            query=query,
            query_type=QueryType.FACTUAL,
            suggested_hop_depth=1,
            rationale="Empty query default",
            detected_markers=[],
        )

    q_lower = query.strip().lower()

    # 1. Check Temporal
    temp_matches = [p for p in TEMPORAL_PATTERNS if re.search(p, q_lower)]
    if temp_matches:
        return QueryClassification(
            query=query,
            query_type=QueryType.TEMPORAL,
            suggested_hop_depth=1,
            rationale="Contains explicit temporal/chronological markers",
            detected_markers=temp_matches,
        )

    # 2. Check Multi-Hop
    multi_matches = [p for p in MULTI_HOP_PATTERNS if re.search(p, q_lower)]
    if multi_matches:
        return QueryClassification(
            query=query,
            query_type=QueryType.MULTI_HOP,
            suggested_hop_depth=2,
            rationale="Contains multi-hop compositional bridging syntax",
            detected_markers=multi_matches,
        )

    # Check for presence of bridging conjunctions with multiple entity references
    entity_refs = re.findall(r"\b[A-Z][A-Za-z0-9-]+(?:\s+[A-Z][A-Za-z0-9-]+)*\b", query)
    if len(entity_refs) >= 3 and any(w in q_lower for w in ["and", "with", "between", "through"]):
        return QueryClassification(
            query=query,
            query_type=QueryType.MULTI_HOP,
            suggested_hop_depth=2,
            rationale="Multiple distinct named entity references with relational conjunction",
            detected_markers=["multi_entity_conjunction"],
        )

    # 3. Check Single-Step Relational
    rel_matches = [p for p in RELATIONAL_PATTERNS if re.search(p, q_lower)]
    if rel_matches:
        return QueryClassification(
            query=query,
            query_type=QueryType.RELATIONAL,
            suggested_hop_depth=1,
            rationale="Queries direct entity relationship (payload, vehicle, developer, orbit)",
            detected_markers=rel_matches,
        )

    # 4. Default to Direct Factual Lookup
    return QueryClassification(
        query=query,
        query_type=QueryType.FACTUAL,
        suggested_hop_depth=1,
        rationale="Standard direct factual lookup",
        detected_markers=["default_factual"],
    )
