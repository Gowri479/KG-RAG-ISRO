"""Structured, provenance-aware generation over frozen relation-aware evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any

from src.generator.ollama_api import UNKNOWN, generate_with_metrics
from src.generator.prompt import SYSTEM_PROMPT, build_user_prompt
from src.generator.token_budget import EVIDENCE, count_tokens, evidence_budget, trim_to_tokens

# Maximum evidence per prompt in model tokens; the context window may lower it further.
TOKEN_BUDGET = 1500
STRUCTURED_SYSTEM_PROMPT = (
    "You answer factual questions using only supplied evidence. KG FACTS are verified "
    "relational evidence. Preserve entity names and relation direction exactly. Do not "
    "invent relations. Dense evidence must not override an explicit verified KG fact "
    "unless a documented provenance conflict exists. If evidence is insufficient, answer "
    "exactly: I don't know."
)
PLAN_SYSTEM_PROMPT = (
    "You are an evidence planner. Produce only a compact answer plan from the supplied "
    "verified KG facts. Preserve relation direction and entity names. Do not add facts. "
    "Use exactly these headings: SUPPORTED FACTS, ANSWER PLAN, UNSUPPORTED CLAIMS."
)


@dataclass(frozen=True)
class KGFact:
    path_id: str
    subject: str
    relation: str
    object: str
    source_documents: tuple[str, ...]
    source_chunks: tuple[str, ...]
    source_urls: tuple[str, ...]
    confidence: str = "verified"


@dataclass(frozen=True)
class GenerationTrace:
    condition: str
    kg_fact_count: int
    kg_path_count: int
    provenance_count: int
    kg_token_count: int
    dense_token_count: int
    bm25_token_count: int
    total_token_count: int
    conflict_classification: str
    used_fact_path_ids: tuple[str, ...]
    ignored_fact_path_ids: tuple[str, ...]
    hop_depth: int
    abstained: bool
    plan: str = ""
    unsupported_claim_proxy: bool = False


def _evidence_budget(system: str, question: str, inner_template: str, options: dict[str, Any]) -> int:
    """Evidence tokens that fit once generate_with_metrics wraps ``inner_template``."""
    return evidence_budget(system, build_user_prompt(inner_template, question), options, cap=TOKEN_BUDGET)


def extract_dense_text(context: str) -> str:
    marker = "[DENSE EVIDENCE]"
    if marker not in context:
        return ""
    return context.split(marker, 1)[1].strip()


def facts_from_paths(paths: list[dict[str, Any]]) -> list[KGFact]:
    facts: list[KGFact] = []
    for path in paths:
        nodes = list(path.get("nodes", []))
        relations = list(path.get("relations", []))
        for index, relation in enumerate(relations):
            if index + 1 >= len(nodes):
                continue
            facts.append(
                KGFact(
                    path_id=str(path.get("path_id", "")),
                    subject=str(nodes[index]),
                    relation=str(relation),
                    object=str(nodes[index + 1]),
                    source_documents=tuple(path.get("source_documents", [])),
                    source_chunks=tuple(path.get("source_chunks", [])),
                    source_urls=tuple(path.get("source_urls", [])),
                )
            )
    return facts


def serialize_kg_facts(facts: list[KGFact]) -> str:
    blocks: list[str] = []
    for index, fact in enumerate(facts, 1):
        provenance = ", ".join(fact.source_chunks) or ", ".join(fact.source_documents) or "unavailable"
        source = ", ".join(fact.source_urls) or "unavailable"
        blocks.append(
            f"KG FACT {index}\n"
            f"Path: {fact.path_id}\n"
            f"Entity: {fact.subject}\n"
            f"Relation: {fact.relation}\n"
            f"Target: {fact.object}\n"
            f"Direction: {fact.subject} -> {fact.relation} -> {fact.object}\n"
            f"Source: {source}\n"
            f"Provenance: {provenance}\n"
            f"Confidence: {fact.confidence}"
        )
    return "\n\n".join(blocks)


def classify_conflict(facts: list[KGFact], dense_text: str) -> str:
    if not facts:
        return "INSUFFICIENT_EVIDENCE"
    dense_norm = dense_text.casefold()
    verified_hits = sum(fact.object.casefold() in dense_norm for fact in facts)
    if verified_hits:
        return "NO_CONFLICT"
    relation_words = {
        "HAS_PAYLOAD": ("payload", "onboard", "carried"),
        "DEVELOPED_BY": ("developed", "designed", "built"),
        "OBSERVES": ("observes", "observe", "measures", "study"),
        "LAUNCHED_ON": ("launched", "launch date"),
    }
    has_relation = any(
        any(term in dense_norm for term in relation_words.get(fact.relation, (fact.relation.casefold(),)))
        for fact in facts
    )
    if has_relation:
        return "DENSE_MORE_SPECIFIC"
    return "KG_MORE_SPECIFIC"


def build_structured_context(facts: list[KGFact], dense_text: str, budget: int = TOKEN_BUDGET) -> tuple[str, dict[str, int]]:
    """Build KG-first evidence within ``budget`` model tokens; dense text gets the remainder."""
    kg_block = serialize_kg_facts(facts)
    dense_block = dense_text.strip()
    header = "KG FACTS:\n"
    separator = "\n\nTEXTUAL EVIDENCE:\n"
    fixed_tokens = count_tokens(header) + count_tokens(separator)
    bounded_kg = trim_to_tokens(kg_block, max(0, budget - fixed_tokens))
    remaining = max(0, budget - fixed_tokens - count_tokens(bounded_kg))
    bounded_dense = trim_to_tokens(dense_block, remaining)
    context = trim_to_tokens(f"{header}{bounded_kg}{separator}{bounded_dense}".strip(), budget)
    stats = {
        "kg_token_count": count_tokens(bounded_kg),
        "dense_token_count": count_tokens(bounded_dense),
        "bm25_token_count": 0,
        "total_token_count": count_tokens(context),
    }
    return context, stats


def _unsupported_proxy(answer: str, facts: list[KGFact]) -> bool:
    if not answer or answer.casefold().strip() == UNKNOWN.casefold():
        return False
    supported = " ".join(f"{fact.subject} {fact.relation} {fact.object}" for fact in facts).casefold()
    answer_tokens = set(re.findall(r"[a-z0-9-]+", answer.casefold()))
    supported_tokens = set(re.findall(r"[a-z0-9-]+", supported))
    return bool(answer_tokens - supported_tokens) and len(answer_tokens - supported_tokens) > max(2, len(answer_tokens) // 2)


def generate_condition(
    condition: str,
    question: str,
    current_context: str,
    facts: list[KGFact],
    options: dict[str, Any],
    hop_depth: int,
    dense_text: str,
) -> tuple[str, GenerationTrace, dict[str, Any]]:
    """Generate one condition without retrieval or evidence changes."""
    if condition == "A_CURRENT":
        context = trim_to_tokens(current_context, _evidence_budget(SYSTEM_PROMPT, question, EVIDENCE, options))
        answer, telemetry = generate_with_metrics(question, context, options=options)
        stats = {"kg_token_count": 0, "dense_token_count": count_tokens(context), "bm25_token_count": 0, "total_token_count": count_tokens(context)}
        trace = GenerationTrace(condition, len(facts), len({f.path_id for f in facts}), len({c for f in facts for c in f.source_chunks}), **stats, conflict_classification=classify_conflict(facts, dense_text), used_fact_path_ids=(), ignored_fact_path_ids=tuple(sorted({f.path_id for f in facts})), hop_depth=hop_depth, abstained=answer.casefold().strip() == UNKNOWN.casefold(), unsupported_claim_proxy=_unsupported_proxy(answer, facts))
        return answer, trace, telemetry

    # Evidence is fitted before wrapping so the window guard never cuts the inner question.
    if condition == "B_STRUCTURED":
        template = (
            f"{EVIDENCE}\n\nQUESTION:\n{question}\n\n"
            "Answer only from the evidence above. Preserve relation direction. ANSWER:"
        )
        context, stats = build_structured_context(facts, dense_text, _evidence_budget(STRUCTURED_SYSTEM_PROMPT, question, template, options))
        user = template.replace(EVIDENCE, context)
        answer, telemetry = generate_with_metrics(question, user, options=options, system_prompt=STRUCTURED_SYSTEM_PROMPT)
        plan = ""
    elif condition == "C_TWO_STAGE":
        plan_template = f"{EVIDENCE}\n\nQUESTION:\n{question}\n\nANSWER PLAN:"
        context, stats = build_structured_context(facts, dense_text, _evidence_budget(PLAN_SYSTEM_PROMPT, question, plan_template, options))
        plan_prompt = plan_template.replace(EVIDENCE, context)
        plan, plan_telemetry = generate_with_metrics(question, plan_prompt, options=options, system_prompt=PLAN_SYSTEM_PROMPT)
        final_template = (
            f"VERIFIED KG FACTS:\n{EVIDENCE}\n\n"
            f"ANSWER PLAN:\n{plan}\n\nQUESTION:\n{question}\n\n"
            "Write the final answer using only the verified facts and plan. ANSWER:"
        )
        fact_budget = _evidence_budget(STRUCTURED_SYSTEM_PROMPT, question, final_template, options)
        final_prompt = final_template.replace(EVIDENCE, trim_to_tokens(serialize_kg_facts(facts), fact_budget))
        answer, telemetry = generate_with_metrics(question, final_prompt, options=options, system_prompt=STRUCTURED_SYSTEM_PROMPT)
        telemetry = {"plan": plan_telemetry, "final": telemetry}
    else:
        raise ValueError(f"Unknown generation condition: {condition}")
    trace = GenerationTrace(condition, len(facts), len({f.path_id for f in facts}), len({c for f in facts for c in f.source_chunks}), **stats, conflict_classification=classify_conflict(facts, dense_text), used_fact_path_ids=tuple(sorted({f.path_id for f in facts})), ignored_fact_path_ids=(), hop_depth=hop_depth, abstained=answer.casefold().strip() == UNKNOWN.casefold(), plan=plan, unsupported_claim_proxy=_unsupported_proxy(answer, facts))
    return answer, trace, telemetry
