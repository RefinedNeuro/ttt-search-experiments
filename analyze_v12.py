#!/usr/bin/env python3
"""v1.2 analysis: noise floor (Q1) + compute-matched ensemble comparison (Q2).

Reads two detailed.json files:
  - v1.1 results (must contain: ttt_lowT, ttt_highT, base_temp_lowT, base_temp_highT)
  - v1.2 results (must contain: base_temp_lowT_seedB, base_temp_highT_seedB)

Writes results/analysis_v12.txt with the verdict.
"""
from __future__ import annotations

import argparse
import json
import os


# ── Decision-rule thresholds (locked in §4/§5 of the v1.2 design doc) ──
Q1_REAL_GAP = 8          # D_ttt >= D_noise + 8  -> priming is real
Q1_NOISE_BAND = 5        # |D_ttt - D_noise| <= 5 -> priming is noise
Q1_COLLAPSE_GAP = 5      # D_ttt <= D_noise - 5  -> priming reduces diversity
Q2_MIXED_HELPS = 4       # mixed_temp_n8 >= max_pure + 4  -> mixed helps
Q2_TIE_BAND = 2          # |mixed_temp - max_pure| <= 2 -> mixed doesn't help


def load_records(path: str) -> dict[str, dict[str, dict]]:
    """Returns nested dict: condition -> task_id -> record."""
    with open(path) as f:
        raw = json.load(f)
    out: dict[str, dict[str, dict]] = {}
    for r in raw:
        out.setdefault(r["condition"], {})[r["task_id"]] = r
    return out


def solve_set(condition_records: dict[str, dict]) -> set[str]:
    return {tid for tid, r in condition_records.items() if r["any_passed"]}


def passed_in_any_candidate(record_list: list[dict]) -> bool:
    """Given a list of records for the same task_id across multiple conditions,
    return True iff any candidate in any record passed."""
    for r in record_list:
        for c in r["candidates"]:
            if c["passed"]:
                return True
    return False


def union_pass_rate(conditions: list[str], all_recs: dict[str, dict[str, dict]]) -> tuple[int, set[str]]:
    """Pass count and solved set when we pool candidates across multiple conditions
    and ask 'any candidate passes'."""
    task_ids = set()
    for c in conditions:
        if c in all_recs:
            task_ids.update(all_recs[c].keys())

    solved: set[str] = set()
    for tid in task_ids:
        recs = [all_recs[c][tid] for c in conditions if c in all_recs and tid in all_recs[c]]
        if passed_in_any_candidate(recs):
            solved.add(tid)
    return len(solved), solved


def q1_verdict(d_ttt: int, d_noise: int) -> tuple[str, str]:
    """Returns (short_label, explanation)."""
    if d_ttt >= d_noise + Q1_REAL_GAP:
        return ("priming is real", f"D_ttt={d_ttt} >= D_noise={d_noise} + {Q1_REAL_GAP}")
    if abs(d_ttt - d_noise) <= Q1_NOISE_BAND:
        return ("priming is noise", f"|D_ttt={d_ttt} - D_noise={d_noise}| <= {Q1_NOISE_BAND}")
    if d_ttt <= d_noise - Q1_COLLAPSE_GAP:
        return ("priming reduces diversity", f"D_ttt={d_ttt} <= D_noise={d_noise} - {Q1_COLLAPSE_GAP}")
    return ("ambiguous", f"D_ttt={d_ttt}, D_noise={d_noise} (between bands)")


def q2_mixed_vs_pure_verdict(mixed_temp: int, best_pure: int) -> tuple[str, str]:
    if mixed_temp >= best_pure + Q2_MIXED_HELPS:
        return ("mixed helps", f"mixed_temp_n8={mixed_temp} >= best_pure={best_pure} + {Q2_MIXED_HELPS}")
    if abs(mixed_temp - best_pure) <= Q2_TIE_BAND:
        return ("mixed doesn't help", f"|mixed_temp_n8={mixed_temp} - best_pure={best_pure}| <= {Q2_TIE_BAND}")
    return ("ambiguous", f"mixed_temp_n8={mixed_temp}, best_pure={best_pure} (between bands)")


def final_verdict(q1_low: str, q1_high: str,
                  q2_mixed_vs_pure: str,
                  mixed_temp: int, mixed_ttt_temp: int) -> str:
    """Apply the §2/§10 branch logic. Low-T is canonical when they disagree."""
    q1_canonical = q1_low  # low-T is the cleaner signal per design doc §4

    if q1_canonical == "priming is real":
        if q2_mixed_vs_pure == "mixed helps":
            # Does TTT add measurable problems on top of mixed-temp?
            if mixed_ttt_temp >= mixed_temp + Q2_MIXED_HELPS:
                return "VERDICT: Build ensemble system, keep TTT as one source."
            else:
                return "VERDICT: Build ensemble system, drop TTT."
        else:
            return ("VERDICT: TTT has real signal but ensembling at n=8 doesn't capture "
                    "it — needs design work.")
    elif q1_canonical == "priming is noise":
        if q2_mixed_vs_pure == "mixed helps":
            return "VERDICT: Build ensemble system, drop TTT."
        else:
            return "VERDICT: Whole direction dead. Just sample more."
    elif q1_canonical == "priming reduces diversity":
        # Active harm. Best move is to drop TTT regardless of Q2.
        if q2_mixed_vs_pure == "mixed helps":
            return "VERDICT: Build ensemble system, drop TTT."
        else:
            return "VERDICT: Whole direction dead. Just sample more."
    else:
        # Ambiguous Q1 → can't reach a clean verdict
        return ("VERDICT: TTT has real signal but ensembling at n=8 doesn't capture "
                "it — needs design work.")


def main() -> None:
    parser = argparse.ArgumentParser(description="v1.2 analysis script")
    parser.add_argument("--v11", default="results/v11_full/detailed.json",
                        help="Path to v1.1 detailed.json")
    parser.add_argument("--v12", default="results/v12_full/detailed.json",
                        help="Path to v1.2 detailed.json (with seedB conditions)")
    parser.add_argument("--out", default="results/v12_full/analysis_v12.txt",
                        help="Output path for the v1.2 analysis text")
    args = parser.parse_args()

    if not os.path.exists(args.v11):
        raise SystemExit(f"v1.1 results not found: {args.v11}")
    if not os.path.exists(args.v12):
        raise SystemExit(f"v1.2 results not found: {args.v12}")

    v11 = load_records(args.v11)
    v12 = load_records(args.v12)

    # Merge: v1.1 + v1.2 conditions in a single dict
    all_recs = {**v11, **v12}

    required = ["ttt_lowT", "ttt_highT", "base_temp_lowT", "base_temp_highT",
                "base_temp_lowT_seedB", "base_temp_highT_seedB"]
    missing = [c for c in required if c not in all_recs]
    if missing:
        raise SystemExit(f"Missing required conditions: {missing}")

    # ── Per-condition pass rates (sanity check) ──
    solve_sets = {c: solve_set(all_recs[c]) for c in all_recs}

    # ── Q1: noise floor + TTT-vs-base symmetric differences ──
    def sym_diff(a: str, b: str) -> tuple[int, int, int]:
        only_a = len(solve_sets[a] - solve_sets[b])
        only_b = len(solve_sets[b] - solve_sets[a])
        return only_a, only_b, only_a + only_b

    _, _, D_ttt_lowT = sym_diff("ttt_lowT", "base_temp_lowT")
    _, _, D_ttt_highT = sym_diff("ttt_highT", "base_temp_highT")
    _, _, D_noise_lowT = sym_diff("base_temp_lowT", "base_temp_lowT_seedB")
    _, _, D_noise_highT = sym_diff("base_temp_highT", "base_temp_highT_seedB")

    q1_lowT, q1_lowT_why = q1_verdict(D_ttt_lowT, D_noise_lowT)
    q1_highT, q1_highT_why = q1_verdict(D_ttt_highT, D_noise_highT)

    # ── Q2: 5 compute-matched ensemble unions (n=8 candidates) ──
    pure_lowT_n8, pure_lowT_set = union_pass_rate(
        ["base_temp_lowT", "base_temp_lowT_seedB"], all_recs)
    pure_highT_n8, pure_highT_set = union_pass_rate(
        ["base_temp_highT", "base_temp_highT_seedB"], all_recs)
    mixed_temp_n8, mixed_temp_set = union_pass_rate(
        ["base_temp_lowT", "base_temp_highT"], all_recs)
    mixed_ttt_temp_n8, mixed_ttt_temp_set = union_pass_rate(
        ["ttt_lowT", "base_temp_highT"], all_recs)
    mixed_ttt_only_n8, mixed_ttt_only_set = union_pass_rate(
        ["ttt_lowT", "ttt_highT"], all_recs)

    best_pure = max(pure_lowT_n8, pure_highT_n8)
    q2_short, q2_why = q2_mixed_vs_pure_verdict(mixed_temp_n8, best_pure)

    # ── Build report ──
    lines: list[str] = []
    lines.append("TTT-Search POC v1.2 — Auto-Analysis")
    lines.append("=" * 50)
    lines.append("")

    # Sources
    lines.append(f"v1.1 results: {args.v11}")
    lines.append(f"v1.2 results: {args.v12}")
    lines.append("")

    # Pass rates table
    lines.append("Per-condition pass rates:")
    table = [
        ("base_greedy",            "base_greedy"),
        ("base_temp_lowT  (seedA, T=0.3)",   "base_temp_lowT"),
        ("base_temp_lowT_seedB (T=0.3)",     "base_temp_lowT_seedB"),
        ("base_temp_highT (seedA, T=0.8)",   "base_temp_highT"),
        ("base_temp_highT_seedB (T=0.8)",    "base_temp_highT_seedB"),
        ("ttt_lowT  (gen T=0.3)",            "ttt_lowT"),
        ("ttt_highT (gen T=0.8)",            "ttt_highT"),
    ]
    total = max(len(all_recs[c]) for _, c in table if c in all_recs)
    for label, c in table:
        if c not in all_recs:
            continue
        passed = len(solve_sets[c])
        n = len(all_recs[c])
        lines.append(f"  {label:<36} {passed:>3}/{n}  ({passed/n*100:.1f}%)")
    lines.append("")

    # Q1 section
    lines.append("=" * 50)
    lines.append("Q1: Noise floor vs TTT-induced divergence")
    lines.append("=" * 50)
    lines.append("")
    lines.append("Symmetric differences:")
    lines.append(f"  D_ttt_lowT   = sym_diff(ttt_lowT,  base_temp_lowT)        = {D_ttt_lowT}")
    lines.append(f"  D_noise_lowT = sym_diff(base_temp_lowT, base_temp_lowT_seedB) = {D_noise_lowT}")
    lines.append(f"  D_ttt_highT  = sym_diff(ttt_highT, base_temp_highT)       = {D_ttt_highT}")
    lines.append(f"  D_noise_highT= sym_diff(base_temp_highT, base_temp_highT_seedB) = {D_noise_highT}")
    lines.append("")
    lines.append(f"Q1 verdict @ T=0.3 (PRIMARY): {q1_lowT}")
    lines.append(f"  rule: {q1_lowT_why}")
    lines.append(f"Q1 verdict @ T=0.8 (secondary): {q1_highT}")
    lines.append(f"  rule: {q1_highT_why}")
    lines.append("")

    # Q2 section
    lines.append("=" * 50)
    lines.append("Q2: Compute-matched (n=8) ensemble combinations")
    lines.append("=" * 50)
    lines.append("")
    lines.append(f"  pure_lowT_n8        = lowT_seedA UNION lowT_seedB      = {pure_lowT_n8}/{total}  ({pure_lowT_n8/total*100:.1f}%)")
    lines.append(f"  pure_highT_n8       = highT_seedA UNION highT_seedB    = {pure_highT_n8}/{total}  ({pure_highT_n8/total*100:.1f}%)")
    lines.append(f"  mixed_temp_n8       = lowT_seedA UNION highT_seedA     = {mixed_temp_n8}/{total}  ({mixed_temp_n8/total*100:.1f}%)")
    lines.append(f"  mixed_ttt_temp_n8   = ttt_lowT  UNION highT_seedA      = {mixed_ttt_temp_n8}/{total}  ({mixed_ttt_temp_n8/total*100:.1f}%)")
    lines.append(f"  mixed_ttt_only_n8   = ttt_lowT  UNION ttt_highT        = {mixed_ttt_only_n8}/{total}  ({mixed_ttt_only_n8/total*100:.1f}%)")
    lines.append("")
    lines.append(f"Best pure = max(pure_lowT_n8, pure_highT_n8) = {best_pure}")
    lines.append(f"Q2 (mixed vs best pure): {q2_short}")
    lines.append(f"  rule: {q2_why}")
    lines.append(f"  delta(mixed_ttt_temp - mixed_temp) = {mixed_ttt_temp_n8 - mixed_temp_n8:+d}")
    lines.append("")

    # Final verdict line
    verdict = final_verdict(q1_lowT, q1_highT, q2_short, mixed_temp_n8, mixed_ttt_temp_n8)
    lines.append("=" * 50)
    lines.append("Final verdict")
    lines.append("=" * 50)
    lines.append("")
    lines.append(verdict)
    lines.append("")

    text = "\n".join(lines)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(text)
    print(text)
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
