"""Pre-registered analysis of the corrected rerun (preregistration §4-§7, §9).

    python -m src.evaluation.analyze_corrected_rerun
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median, stdev

import numpy as np
from scipy.stats import binomtest, wilcoxon

from src.evaluation.build_corrected_rerun_human_package import (
    ANNOTATIONS,
    BENCHMARK,
    DIMENSIONS,
    MAPPING,
    RERUN,
    RESULTS,
    SEED,
    answer_hash,
    load_jsonl,
)

OUTPUT_JSON = RERUN / "analysis.json"
OUTPUT_MD = RERUN / "analysis.md"
SCORES = ("correctness", "completeness", "groundedness", "relevance")
PRIMARY = (("C_TWO_STAGE", "V_VANILLA"), ("A_CURRENT", "V_VANILLA"))
SECONDARY_PAIRS = PRIMARY + (("C_TWO_STAGE", "A_CURRENT"),)
BOOTSTRAP = 10_000
ALPHA = 0.05


# ---------- statistics ----------

def holm(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    adjusted, running = {}, 0.0
    for rank, (key, p) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * p))
        adjusted[key] = running
    return adjusted


def wilcoxon_p(diffs: list[float]) -> float:
    if not any(diffs):
        return 1.0
    return float(wilcoxon(diffs, zero_method="wilcox", alternative="two-sided").pvalue)


def mcnemar_p(treatment: list[int], control: list[int]) -> tuple[float, int, int]:
    worse = sum(t == 1 and c == 0 for t, c in zip(treatment, control))
    better = sum(t == 0 and c == 1 for t, c in zip(treatment, control))
    if worse + better == 0:
        return 1.0, worse, better
    return float(binomtest(worse, worse + better, 0.5).pvalue), worse, better


def bootstrap_ci(diffs: list[float]) -> list[float]:
    values = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(SEED)
    means = values[rng.integers(0, len(values), size=(BOOTSTRAP, len(values)))].mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 6), round(float(np.percentile(means, 97.5)), 6)]


def describe(diffs: list[float]) -> dict:
    sd = stdev(diffs) if len(diffs) > 1 else 0.0
    return {
        "n": len(diffs),
        "mean_diff": round(mean(diffs), 6),
        "median_diff": round(median(diffs), 6),
        "ci95": bootstrap_ci(diffs),
        "dz": round(mean(diffs) / sd, 6) if sd else None,
        "wins_losses_ties": [sum(d > 0 for d in diffs), sum(d < 0 for d in diffs), sum(d == 0 for d in diffs)],
    }


def krippendorff_alpha(units: list[list[float | None]], metric: str = "ordinal") -> float | None:
    """Krippendorff's alpha; ``units`` holds one list of rater values per unit (None = missing)."""
    pairable = [[v for v in unit if v is not None] for unit in units]
    pairable = [unit for unit in pairable if len(unit) >= 2]
    values = sorted({v for unit in pairable for v in unit})
    if len(values) < 2:
        return None
    index = {v: i for i, v in enumerate(values)}
    coincidence = np.zeros((len(values), len(values)))
    for unit in pairable:
        m = len(unit)
        for i, a in enumerate(unit):
            for j, b in enumerate(unit):
                if i != j:
                    coincidence[index[a], index[b]] += 1 / (m - 1)
    n_c = coincidence.sum(axis=1)
    total = n_c.sum()

    def delta(c: int, k: int) -> float:
        if metric == "nominal":
            return float(c != k)
        if metric == "interval":
            return (values[c] - values[k]) ** 2
        lo, hi = min(c, k), max(c, k)
        return (n_c[lo:hi + 1].sum() - (n_c[c] + n_c[k]) / 2) ** 2

    d = np.array([[delta(c, k) for k in range(len(values))] for c in range(len(values))])
    observed = (coincidence * d).sum()
    expected = (np.outer(n_c, n_c) * d).sum() / (total - 1)
    return float(1 - observed / expected) if expected else None


# ---------- loading ----------

def load_ratings() -> tuple[dict, list[Path]]:
    mapping = json.loads(MAPPING.read_text(encoding="utf8"))["mapping"]
    files = sorted(ANNOTATIONS.glob("corrected_rerun_human_eval_rater*_filled.csv"))
    if not files:
        raise RuntimeError("No filled rater files (corrected_rerun_human_eval_rater*_filled.csv)")
    ratings: dict = {}  # (qid, system) -> dimension -> [rater values]
    for path in files:
        rows = list(csv.DictReader(path.open(encoding="utf8")))
        if sorted(r["question_id"] for r in rows) != sorted(mapping):
            raise RuntimeError(f"{path.name}: question set does not match the package")
        for row in rows:
            entry = mapping[row["question_id"]]
            for label, system in entry["label_to_system"].items():
                if answer_hash(row[f"candidate_{label}"]) != entry["label_hashes"][label]:
                    raise RuntimeError(f"{path.name}: candidate hash mismatch {row['question_id']} {label}")
                cell = ratings.setdefault((row["question_id"], system), {d: [] for d in DIMENSIONS})
                for dimension in DIMENSIONS:
                    raw = row[f"{label}_{dimension}"].strip()
                    # Spreadsheet exports write whole numbers as "5.0"; fractional scores are invalid.
                    value = int(float(raw)) if raw and float(raw).is_integer() else None
                    allowed = (0, 1) if dimension == "unsupported_claim" else (1, 2, 3, 4, 5)
                    if value not in allowed:
                        raise RuntimeError(f"{path.name}: invalid {dimension}={value} for {row['question_id']} {label}")
                    cell[dimension].append(value)
    return ratings, files


def item_scores(ratings: dict) -> dict:
    """Per-item score: rater mean for 1-5 scales; unsupported = 1 if any rater flags it."""
    return {
        key: {**{d: mean(v[d]) for d in SCORES}, "unsupported_claim": int(any(v["unsupported_claim"]))}
        for key, v in ratings.items()
    }


def analysis_sets(questions: list[dict]) -> dict[str, list[str]]:
    required = [q for q in questions if q["kg_required"] == "YES"]
    seen, dedup = set(), []
    for q in sorted(required, key=lambda q: q["question_id"]):
        text = " ".join(q["question"].casefold().split())
        if text not in seen:
            seen.add(text)
            dedup.append(q["question_id"])
    return {
        "primary_kg_required": [q["question_id"] for q in required],
        "kg_not_required": [q["question_id"] for q in questions if q["kg_required"] == "NO"],
        "sensitivity_deduplicated": dedup,
        "sensitivity_excluding_direct_control": [q["question_id"] for q in required if q["category"] != "DIRECT_CONTROL"],
        "heldout_descriptive": [q["question_id"] for q in required if q["split"] == "evaluation"],
    }


# ---------- analysis ----------

def system_means(scores: dict, ids: list[str]) -> dict:
    systems = sorted({s for _, s in scores})
    return {
        s: {d: round(mean(scores[(q, s)][d] for q in ids), 6) for d in SCORES + ("unsupported_claim",)}
        for s in systems
    }


def compare(scores: dict, ids: list[str], treatment: str, control: str, dimension: str, test: bool) -> dict:
    if dimension == "unsupported_claim":
        t = [scores[(q, treatment)][dimension] for q in ids]
        c = [scores[(q, control)][dimension] for q in ids]
        result = {"rate_treatment": round(mean(t), 6), "rate_control": round(mean(c), 6), "rate_diff": round(mean(t) - mean(c), 6)}
        if test:
            p, worse, better = mcnemar_p(t, c)
            result.update({"mcnemar_p": p, "treatment_worse": worse, "treatment_better": better})
        return result
    diffs = [scores[(q, treatment)][dimension] - scores[(q, control)][dimension] for q in ids]
    result = describe(diffs)
    if test:
        result["wilcoxon_p"] = wilcoxon_p(diffs)
    return result


def decision(primary: dict, unsupported: dict) -> str:
    if primary["holm_p"] < ALPHA and primary["mean_diff"] > 0:
        higher_unsupported = unsupported["rate_diff"] > 0 and unsupported["mcnemar_p"] < ALPHA
        return "NOT SUPPORTED (unsupported claims significantly higher)" if higher_unsupported else "SUPPORTED"
    if primary["holm_p"] < ALPHA and primary["mean_diff"] < 0:
        return "HARMFUL"
    return "NOT SUPPORTED"


def automated(ids: list[str]) -> dict:
    rows = {(r["question_id"], r["system"]): r for r in load_jsonl(RESULTS)}
    def value(q, s, metric):
        m = rows[(q, s)]["metrics"]
        return float(m[metric]) if metric in m else float(rows[(q, s)]["answer"].strip().casefold() == "i don't know.")
    metric_names = [k for k in rows[(ids[0], "V_VANILLA")]["metrics"] if isinstance(rows[(ids[0], "V_VANILLA")]["metrics"][k], (int, float))]
    out = {"means": {}, "comparisons": {}}
    for s in ("V_VANILLA", "A_CURRENT", "C_TWO_STAGE"):
        out["means"][s] = {m: round(mean(value(q, s, m) for q in ids), 6) for m in metric_names}
        out["means"][s]["idk_rate"] = round(mean(rows[(q, s)]["answer"].strip().casefold() == "i don't know." for q in ids), 6)
    for t, c in SECONDARY_PAIRS:
        for m in metric_names:
            diffs = [value(q, t, m) - value(q, c, m) for q in ids]
            out["comparisons"][f"{t}_vs_{c}:{m}"] = {**describe(diffs), "wilcoxon_p": wilcoxon_p(diffs)}
    return out


def integrity_summary() -> dict:
    rows = load_jsonl(RESULTS)
    return {
        "rows": len(rows),
        "rows_failing_integrity": sum(not (r["integrity"]["all_status_ok"] and r["integrity"]["all_prompt_counts_match"] and not r["integrity"]["any_ollama_truncated"]) for r in rows),
        "rows_trimmed_by_window_guard": sum(r["integrity"]["any_context_trimmed"] for r in rows),
        "retried_rows": sum(bool(r["integrity"].get("retried")) for r in rows),
    }


def main() -> None:
    questions = json.loads(BENCHMARK.read_text(encoding="utf8"))["questions"]
    ratings, files = load_ratings()
    scores = item_scores(ratings)
    sets = analysis_sets(questions)
    primary_ids = sets["primary_kg_required"]

    agreement = None
    if len(files) >= 2:
        agreement = {
            d: krippendorff_alpha([v[d] for v in ratings.values()], "nominal" if d == "unsupported_claim" else "ordinal")
            for d in DIMENSIONS
        }

    primary = {f"{t}_vs_{c}": compare(scores, primary_ids, t, c, "correctness", test=True) for t, c in PRIMARY}
    for key, adjusted in holm({k: v["wilcoxon_p"] for k, v in primary.items()}).items():
        primary[key]["holm_p"] = adjusted

    secondary = {}
    for t, c in SECONDARY_PAIRS:
        for d in SCORES + ("unsupported_claim",):
            if (t, c) in PRIMARY and d == "correctness":
                continue
            secondary[f"{t}_vs_{c}:{d}"] = compare(scores, primary_ids, t, c, d, test=True)
    secondary_p = {k: v.get("wilcoxon_p", v.get("mcnemar_p")) for k, v in secondary.items()}
    for key, adjusted in holm(secondary_p).items():
        secondary[key]["holm_p_within_secondary"] = adjusted

    decisions = {
        f"{t}_vs_{c}": decision(primary[f"{t}_vs_{c}"], secondary[f"{t}_vs_{c}:unsupported_claim"])
        for t, c in PRIMARY
    }

    sensitivity = {}
    for name in ("sensitivity_deduplicated", "sensitivity_excluding_direct_control", "heldout_descriptive"):
        ids = sets[name]
        sensitivity[name] = {
            "n": len(ids),
            "means": system_means(scores, ids),
            "correctness_diffs": {f"{t}_vs_{c}": describe([scores[(q, t)]["correctness"] - scores[(q, c)]["correctness"] for q in ids]) for t, c in PRIMARY},
        }

    result = {
        "preregistration": "reports/preregistration_corrected_rerun.md",
        "raters": len(files),
        "rater_files": [f.name for f in files],
        "integrity": integrity_summary(),
        "agreement_krippendorff_alpha": agreement,
        "set_sizes": {k: len(v) for k, v in sets.items()},
        "primary_means": system_means(scores, primary_ids),
        "primary": primary,
        "decisions": decisions,
        "secondary_exploratory": secondary,
        "kg_not_required_descriptive": system_means(scores, sets["kg_not_required"]),
        "sensitivity": sensitivity,
        "automated_secondary": automated(primary_ids),
    }
    OUTPUT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    OUTPUT_MD.write_text(render(result), encoding="utf8")
    print(json.dumps({"decisions": decisions, "primary": {k: {x: v[x] for x in ("mean_diff", "ci95", "wilcoxon_p", "holm_p")} for k, v in primary.items()}}, indent=2))


def render(r: dict) -> str:
    lines = [
        "# Corrected-rerun analysis (pre-registered)",
        "",
        f"Raters: {r['raters']}. Integrity: {r['integrity']}.",
        f"Agreement (Krippendorff's alpha): {r['agreement_krippendorff_alpha'] or 'single rater; not reported'}",
        "",
        "## Primary: human correctness, 60 KG-required items (Holm over 2 tests)",
        "",
        "| Comparison | Mean diff | 95% CI | Median | dz | W/L/T | Wilcoxon p | Holm p | Decision |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for key, v in r["primary"].items():
        lines.append(f"| {key} | {v['mean_diff']:+.3f} | [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] | {v['median_diff']:+.3f} | {v['dz']} | {'/'.join(map(str, v['wins_losses_ties']))} | {v['wilcoxon_p']:.4f} | {v['holm_p']:.4f} | {r['decisions'][key]} |")
    lines += ["", "## Means (primary set)", "", "| System | " + " | ".join(SCORES) + " | unsupported |", "|---|" + "---|" * 5]
    for s, m in r["primary_means"].items():
        lines.append(f"| {s} | " + " | ".join(f"{m[d]:.3f}" for d in SCORES) + f" | {m['unsupported_claim']:.1%} |")
    lines += ["", "Secondary, sensitivity and automated results are exploratory; see analysis.json.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
