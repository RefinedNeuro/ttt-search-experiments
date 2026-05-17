"""Auto-analysis for v1.1: per-condition pass rates, pairwise solve-sets, oracle unions,
diversity metrics, and the §6 verdict matrix.
"""
from __future__ import annotations

import itertools
import json
import os
from collections import Counter


def _normalized_levenshtein(a: str, b: str) -> float:
    """Returns Levenshtein distance / max(len(a), len(b)). 0 = identical, 1 = totally different."""
    if a == b:
        return 0.0
    if not a or not b:
        return 1.0
    # Iterative DP, O(min(m,n)) memory
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(cur[j-1] + 1, prev[j] + 1, prev[j-1] + cost)
        prev = cur
    return prev[-1] / max(len(a), len(b))


def _solve_set(records: list[dict]) -> set[str]:
    return {r["task_id"] for r in records if r["any_passed"]}


def _mean_pairwise_distance(candidates: list[dict]) -> float:
    codes = [c["code"] for c in candidates]
    if len(codes) < 2:
        return 0.0
    dists = [
        _normalized_levenshtein(codes[i], codes[j])
        for i, j in itertools.combinations(range(len(codes)), 2)
    ]
    return sum(dists) / len(dists)


def analyze(detailed_path: str, output_path: str) -> str:
    """Load detailed.json, run analysis, write analysis.txt. Returns the analysis text."""
    with open(detailed_path) as f:
        raw = json.load(f)

    # raw is a list of {task_id, condition, candidates, selected_index, any_passed}
    by_cond: dict[str, list[dict]] = {}
    for r in raw:
        by_cond.setdefault(r["condition"], []).append(r)

    conditions = list(by_cond.keys())
    total_problems = len(next(iter(by_cond.values())))

    lines: list[str] = []
    lines.append("TTT-Search POC v1.1 — Auto-Analysis")
    lines.append("=" * 50)
    lines.append("")
    lines.append(f"Total problems: {total_problems}")
    lines.append(f"Conditions: {', '.join(conditions)}")
    lines.append("")

    # ── §3.3.1 Per-condition pass rates ──
    lines.append("─" * 50)
    lines.append("1. Per-condition pass rates")
    lines.append("─" * 50)
    lines.append(f"{'Condition':<25} {'Pass':>10} {'Rate':>8}")
    for c in conditions:
        passed = sum(1 for r in by_cond[c] if r["any_passed"])
        n = len(by_cond[c])
        lines.append(f"{c:<25} {passed:>5}/{n:<4} {passed/n*100:>7.1f}%")
    lines.append("")

    # ── §3.3.2 Pairwise solve-set comparison ──
    lines.append("─" * 50)
    lines.append("2. Pairwise solve-set comparison")
    lines.append("─" * 50)
    solve_sets = {c: _solve_set(by_cond[c]) for c in conditions}

    for a, b in itertools.combinations(conditions, 2):
        sa, sb = solve_sets[a], solve_sets[b]
        both = sa & sb
        only_a = sa - sb
        only_b = sb - sa
        neither = total_problems - len(sa | sb)
        sym_diff = len(only_a) + len(only_b)
        lines.append(f"\n  {a} vs {b}:")
        lines.append(f"    Both solve:      {len(both):>4}")
        lines.append(f"    Only {a:<18} {len(only_a):>4}")
        lines.append(f"    Only {b:<18} {len(only_b):>4}")
        lines.append(f"    Neither:         {neither:>4}")
        lines.append(f"    Symmetric diff:  {sym_diff:>4}")
        lines.append(f"    Union (oracle):  {len(sa | sb):>4}  ({len(sa | sb)/total_problems*100:.1f}%)")
    lines.append("")

    # ── §3.3.3 Oracle union of all conditions ──
    lines.append("─" * 50)
    lines.append("3. Oracle union — best possible if we could pick any condition per problem")
    lines.append("─" * 50)
    all_union: set[str] = set()
    for s in solve_sets.values():
        all_union |= s
    lines.append(f"Union of all {len(conditions)} conditions: {len(all_union)}/{total_problems} "
                 f"({len(all_union)/total_problems*100:.1f}%)")
    lines.append("")

    # ── §3.3.4 Diversity within each multi-candidate condition ──
    lines.append("─" * 50)
    lines.append("4. Diversity within each condition (mean pairwise normalized Levenshtein)")
    lines.append("─" * 50)
    lines.append(f"{'Condition':<25} {'Mean dist':>10} {'All-identical problems':>25}")
    for c in conditions:
        if len(by_cond[c][0]["candidates"]) < 2:
            lines.append(f"{c:<25} {'n/a':>10} {'n/a (single candidate)':>25}")
            continue
        dists = [_mean_pairwise_distance(r["candidates"]) for r in by_cond[c]]
        mean_dist = sum(dists) / len(dists)
        n_identical = sum(
            1 for r in by_cond[c]
            if len({cand["code"] for cand in r["candidates"]}) == 1
        )
        lines.append(f"{c:<25} {mean_dist:>10.3f} {n_identical:>5}/{len(by_cond[c]):<4} all identical")
    lines.append("")

    # ── §3.3.4 Per-source-label win contribution ──
    lines.append("─" * 50)
    lines.append("5. Per-source-label first-pass contributions")
    lines.append("─" * 50)
    for c in conditions:
        if len(by_cond[c][0]["candidates"]) < 2:
            continue
        first_pass_labels: Counter = Counter()
        any_pass_labels: Counter = Counter()
        for r in by_cond[c]:
            passing = [cand for cand in r["candidates"] if cand["passed"]]
            if passing:
                # First passing candidate by index
                first_passing = min(passing, key=lambda c: c["index"])
                first_pass_labels[first_passing["source_label"]] += 1
                for cand in passing:
                    any_pass_labels[cand["source_label"]] += 1
        lines.append(f"\n  {c}:")
        all_labels = sorted(set(any_pass_labels) | set(first_pass_labels))
        if not all_labels:
            lines.append(f"    (no passes)")
        for label in all_labels:
            lines.append(
                f"    {label:<15} first-pass={first_pass_labels[label]:>3} "
                f"any-pass={any_pass_labels[label]:>3}"
            )
    lines.append("")

    # ── §6 Verdict matrix ──
    lines.append("─" * 50)
    lines.append("6. Verdict (§6 of v1.1 design doc)")
    lines.append("─" * 50)

    verdict = _verdict(by_cond, solve_sets)
    lines.append(verdict)
    lines.append("")

    text = "\n".join(lines)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(text)
    return text


def _count_passed(records: list[dict]) -> int:
    return sum(1 for r in records if r["any_passed"])


def _verdict(by_cond: dict, solve_sets: dict) -> str:
    """Apply the §6 verdict matrix to ttt_lowT vs base_temp_lowT (and high-T mirror)."""
    needed = {"ttt_lowT", "base_temp_lowT"}
    if not needed.issubset(by_cond):
        return f"VERDICT: cannot evaluate — missing conditions {needed - set(by_cond)}."

    out = []

    for ttt_cond, base_cond, label in [
        ("ttt_lowT", "base_temp_lowT", "Low temperature (T=0.3) — PRIMARY"),
        ("ttt_highT", "base_temp_highT", "High temperature (T=0.8) — secondary"),
    ]:
        if ttt_cond not in by_cond or base_cond not in by_cond:
            out.append(f"\n  [{label}] skipped — condition missing.")
            continue

        ttt_set = solve_sets[ttt_cond]
        base_set = solve_sets[base_cond]
        n_ttt = len(ttt_set)
        n_base = len(base_set)
        only_ttt = len(ttt_set - base_set)
        only_base = len(base_set - ttt_set)
        D = only_ttt + only_base
        delta = n_ttt - n_base

        out.append(f"\n  [{label}]")
        out.append(f"    ttt={n_ttt}, base={n_base}, delta={delta:+d}")
        out.append(f"    only_ttt={only_ttt}, only_base={only_base}, symmetric_diff D={D}")

        # Verdict per §6
        if D >= 10 and delta >= -2:
            out.append(f"    → H1 + H2 HOLD: priming creates real diversity. Worth pursuing v2.")
        elif D < 10 and abs(delta) <= 3:
            out.append(f"    → H3 HOLDS: priming is decorative. Temperature alone explained v1.")
        elif delta <= -5:
            out.append(f"    → TTT ACTIVELY HARMS: warm-up is interfering. Approach may be wrong.")
        else:
            out.append(f"    → AMBIGUOUS: D={D}, delta={delta:+d}. v1.2 needed.")

    return "\n".join(out)
