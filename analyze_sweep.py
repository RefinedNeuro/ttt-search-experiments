#!/usr/bin/env python3
"""Sampling-dynamics sweep analysis.

Combines v1.1 + v1.2 + sweep candidate pools, groups by temperature, and
computes pass@N curves, compile-failure rates, within-pool diversity,
saturation analysis, and a compute-optimal (N, T) table.

Q1-Q4 verdicts are produced per the design doc §6 thresholds.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from collections import defaultdict


# ── Decision-rule thresholds from the sweep design doc §6 ──
TEMP_TIE_BAND = 2          # |pass@16(A) - pass@16(B)| <= 2  -> tied (pick lower T)
PLATEAU_DELTA = 1          # pass@N - pass@N/2 < 1 problem  -> plateaued at N
GARBAGE_RATE = 0.05        # compile fail rate > 5%  -> "garbage onset"


# Map condition name -> (temperature, list_of_seeds_used)
# This lets the analyzer pool candidates across v1.1 / v1.2 / sweep files.
CONDITION_INFO = {
    "base_temp_lowT":         (0.3, "v1.1 seedA"),
    "base_temp_lowT_seedB":   (0.3, "v1.2 seedB"),
    "sweep_T03":              (0.3, "sweep extra"),
    "sweep_T05":              (0.5, "sweep"),
    "base_temp_highT":        (0.8, "v1.1 seedA"),
    "base_temp_highT_seedB":  (0.8, "v1.2 seedB"),
    "sweep_T08":              (0.8, "sweep extra"),
    "sweep_T10":              (1.0, "sweep"),
    "sweep_T12":              (1.2, "sweep"),
}


try:
    import Levenshtein as _lev
    _HAVE_C_LEV = True
except ImportError:
    _HAVE_C_LEV = False


def _normalized_levenshtein(a: str, b: str) -> float:
    if a == b:
        return 0.0
    if not a or not b:
        return 1.0
    if _HAVE_C_LEV:
        return _lev.distance(a, b) / max(len(a), len(b))
    # Pure-Python fallback (slow on long strings).
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


def load_records(paths: list[str]) -> list[dict]:
    """Concat all detailed.json records from the given paths."""
    out: list[dict] = []
    for p in paths:
        if not os.path.exists(p):
            print(f"(skipping missing file: {p})")
            continue
        with open(p) as f:
            out.extend(json.load(f))
    return out


def pool_by_temperature(records: list[dict]) -> dict[float, dict[str, list[dict]]]:
    """Return temperature -> task_id -> list of candidate dicts.

    Only conditions in CONDITION_INFO are included; conditions outside it
    (e.g., base_greedy, ttt_*) are filtered out so the temperature pool is
    clean.
    """
    pools: dict[float, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        cond = r["condition"]
        if cond not in CONDITION_INFO:
            continue
        temperature, _label = CONDITION_INFO[cond]
        for cand in r["candidates"]:
            pools[temperature][r["task_id"]].append(cand)
    return pools


def empirical_pass_at_n(pools_per_temp: dict[str, list[dict]], n: int) -> tuple[int, int]:
    """For each task, take the first n candidates from its pool and count problems
    where any of those n passed. Returns (passed_count, total_problems)."""
    passed = 0
    total = 0
    for task_id, candidates in pools_per_temp.items():
        if len(candidates) < n:
            # Skip if this task doesn't have enough candidates at this temperature
            continue
        total += 1
        first_n = candidates[:n]
        if any(c["passed"] for c in first_n):
            passed += 1
    return passed, total


def unbiased_pass_at_n(pools_per_temp: dict[str, list[dict]], n: int) -> float:
    """Unbiased pass@n estimator from the HumanEval paper:
        pass@n = 1 - C(M-c, n) / C(M, n)
    where M is the number of candidates and c is the number that passed.
    Averaged across problems.
    """
    estimates = []
    for task_id, candidates in pools_per_temp.items():
        M = len(candidates)
        if M < n:
            continue
        c = sum(1 for cand in candidates if cand["passed"])
        if M - c < n:
            # All n must include at least one passing candidate
            estimates.append(1.0)
        else:
            # 1 - C(M-c, n) / C(M, n) = 1 - prod((M-c-i)/(M-i)) for i in 0..n-1
            ratio = 1.0
            for i in range(n):
                ratio *= (M - c - i) / (M - i)
            estimates.append(1.0 - ratio)
    return sum(estimates) / len(estimates) if estimates else 0.0


def compile_failure_rate(pools_per_temp: dict[str, list[dict]]) -> float:
    """Fraction of candidates whose `error` starts with 'SyntaxError' or contains
    'entry_point' (the two compile-time pre-flight failures from the verifier)."""
    total = 0
    fails = 0
    for candidates in pools_per_temp.values():
        for c in candidates:
            total += 1
            err = c.get("error") or ""
            if err.startswith("SyntaxError") or "entry_point" in err and "not defined" in err:
                fails += 1
    return fails / total if total else 0.0


def mean_pairwise_levenshtein(pools_per_temp: dict[str, list[dict]], n_cap: int = 16) -> float:
    """Mean pairwise normalized Levenshtein distance across up to n_cap candidates
    per problem, averaged across problems."""
    per_problem = []
    for candidates in pools_per_temp.values():
        codes = [c["code"] for c in candidates[:n_cap]]
        if len(codes) < 2:
            continue
        dists = [
            _normalized_levenshtein(codes[i], codes[j])
            for i, j in itertools.combinations(range(len(codes)), 2)
        ]
        per_problem.append(sum(dists) / len(dists))
    return sum(per_problem) / len(per_problem) if per_problem else 0.0


def saturation_n(passN_table: dict[int, int], n_values: list[int]) -> int | None:
    """Return the smallest n in n_values where pass@n - pass@(n//2) < PLATEAU_DELTA.
    Skip n=1 (no half value). Returns None if no plateau in range."""
    for n in n_values:
        half = n // 2
        if half < 1 or half not in passN_table:
            continue
        delta = passN_table[n] - passN_table[half]
        if delta < PLATEAU_DELTA:
            return n
    return None


def make_plots(pass_at_n_table: dict, n_values: list[int], temperatures: list[float],
               out_dir: str) -> bool:
    """Optional. Returns True if plots were made, False if matplotlib unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return False

    os.makedirs(out_dir, exist_ok=True)

    # Plot 1: pass@N curves per temperature
    fig, ax = plt.subplots(figsize=(8, 5))
    for T in temperatures:
        if T not in pass_at_n_table:
            continue
        ys = [pass_at_n_table[T].get(n, 0) for n in n_values]
        ax.plot(n_values, ys, marker="o", label=f"T={T}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(n_values)
    ax.set_xticklabels(n_values)
    ax.set_xlabel("N (number of samples)")
    ax.set_ylabel("Pass count (out of 164)")
    ax.set_title("pass@N curves per temperature (Qwen3.5-0.8B, HumanEval)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "plot_passN_curves.png"), dpi=120)
    plt.close(fig)

    # Plot 2: heatmap of pass count over (N, T)
    fig, ax = plt.subplots(figsize=(7, 4))
    grid = np.array([[pass_at_n_table.get(T, {}).get(n, 0) for n in n_values] for T in temperatures])
    im = ax.imshow(grid, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(n_values)))
    ax.set_xticklabels(n_values)
    ax.set_yticks(range(len(temperatures)))
    ax.set_yticklabels([f"T={T}" for T in temperatures])
    ax.set_xlabel("N")
    ax.set_ylabel("Temperature")
    ax.set_title("Pass count over (N, T) surface")
    for i in range(len(temperatures)):
        for j in range(len(n_values)):
            ax.text(j, i, str(int(grid[i, j])), ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(im, ax=ax, label="Pass count")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "plot_surface.png"), dpi=120)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep analyzer")
    parser.add_argument("--v11", default="results/v11_full/detailed.json")
    parser.add_argument("--v12", default="results/v12_full/detailed.json")
    parser.add_argument("--sweep", default="results/sweep/detailed.json")
    parser.add_argument("--out", default="results/sweep/sweep_analysis.txt")
    parser.add_argument("--plots-dir", default="results/sweep/sweep_plots")
    args = parser.parse_args()

    records = load_records([args.v11, args.v12, args.sweep])
    if not records:
        raise SystemExit("No records loaded from any input file.")

    pools = pool_by_temperature(records)
    temperatures = sorted(pools.keys())
    n_values = [1, 2, 4, 8, 12, 16]

    # ── Compute the tables ──
    pass_at_n_emp: dict[float, dict[int, int]] = {}
    pass_at_n_unb: dict[float, dict[int, float]] = {}
    total_problems: dict[float, int] = {}
    compile_fail: dict[float, float] = {}
    diversity: dict[float, float] = {}
    candidates_per_problem: dict[float, dict[str, int]] = {}

    for T in temperatures:
        pool = pools[T]
        candidates_per_problem[T] = {tid: len(cs) for tid, cs in pool.items()}
        # Use the minimum candidate count across problems as the effective N cap
        min_n = min(candidates_per_problem[T].values()) if candidates_per_problem[T] else 0

        passes_at_n: dict[int, int] = {}
        unbs_at_n: dict[int, float] = {}
        for n in n_values:
            if n > min_n:
                continue
            passed, total = empirical_pass_at_n(pool, n)
            passes_at_n[n] = passed
            unbs_at_n[n] = unbiased_pass_at_n(pool, n)
            total_problems[T] = total
        pass_at_n_emp[T] = passes_at_n
        pass_at_n_unb[T] = unbs_at_n
        compile_fail[T] = compile_failure_rate(pool)
        diversity[T] = mean_pairwise_levenshtein(pool, n_cap=16)

    # ── Q1: temperature peak at N=16 ──
    pass_at_16 = {T: pass_at_n_emp[T].get(16, -1) for T in temperatures}
    valid_at_16 = {T: v for T, v in pass_at_16.items() if v >= 0}
    if valid_at_16:
        best_count = max(valid_at_16.values())
        peak_candidates = sorted(T for T, v in valid_at_16.items() if v >= best_count - TEMP_TIE_BAND)
        # Tie-break: prefer lower temperature
        q1_T = peak_candidates[0]
        q1_count = valid_at_16[q1_T]
        q1_total = total_problems.get(q1_T, 164)
        q1_line = f"Q1 (Temperature peak at N=16): T={q1_T}, pass {q1_count}/{q1_total} ({q1_count/q1_total*100:.1f}%)"
    else:
        q1_line = "Q1 (Temperature peak at N=16): no temperature reached N=16"

    # ── Q2: plateau per temperature ──
    q2_lines = ["Q2 (Plateau per T):"]
    for T in temperatures:
        sat = saturation_n(pass_at_n_emp[T], n_values)
        if sat is None:
            q2_lines.append(f"    T={T} plateaus at N=no plateau in range (pass@16 - pass@8 >= {PLATEAU_DELTA})")
        else:
            q2_lines.append(f"    T={T} plateaus at N={sat}")

    # ── Q3: garbage onset ──
    safe_temps = sorted(T for T, rate in compile_fail.items() if rate <= GARBAGE_RATE)
    if safe_temps:
        q3_T = safe_temps[-1]
        q3_line = f"Q3 (Highest safe temperature, compile rate <={GARBAGE_RATE*100:.0f}%): T={q3_T} (rate={compile_fail[q3_T]*100:.1f}%)"
    else:
        q3_line = f"Q3 (Highest safe temperature, compile rate <={GARBAGE_RATE*100:.0f}%): none — even the lowest temperature exceeded the threshold"

    # ── Q4: compute-optimal (N, T) per budget ──
    q4_lines = ["Q4 (Compute-optimal):"]
    for budget in [4, 8, 16]:
        best = None
        for T in temperatures:
            for n in n_values:
                if n > budget:
                    break
                if n not in pass_at_n_emp[T]:
                    continue
                count = pass_at_n_emp[T][n]
                # Tie break: prefer higher N (more reliable)
                key = (count, n)
                if best is None or key > best[0]:
                    best = (key, T, n)
        if best:
            (count, _), T, n = best
            total = total_problems.get(T, 164)
            q4_lines.append(f"    Budget {budget}: (N={n}, T={T}) -> {count}/{total} ({count/total*100:.1f}%)")
        else:
            q4_lines.append(f"    Budget {budget}: no data")

    # ── Build the report ──
    lines: list[str] = []
    lines.append("Sampling-Dynamics Sweep — Auto-Analysis")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Inputs combined: {args.v11}, {args.v12}, {args.sweep}")
    lines.append(f"Temperatures   : {temperatures}")
    lines.append(f"N values       : {n_values}")
    lines.append("")
    lines.append("Candidates per problem by temperature (min across problems):")
    for T in temperatures:
        min_n = min(candidates_per_problem[T].values()) if candidates_per_problem[T] else 0
        max_n = max(candidates_per_problem[T].values()) if candidates_per_problem[T] else 0
        n_probs = len(candidates_per_problem[T])
        lines.append(f"    T={T}: {n_probs} problems, candidates per problem min={min_n} max={max_n}")
    lines.append("")

    # Pass@N table (empirical)
    lines.append("-" * 60)
    lines.append("Pass@N (empirical, first-N candidates)")
    lines.append("-" * 60)
    header = f"  {'N':>4}  " + "  ".join(f"T={T}".rjust(10) for T in temperatures)
    lines.append(header)
    for n in n_values:
        row_cells = [f"{n:>4}"]
        for T in temperatures:
            v = pass_at_n_emp[T].get(n)
            if v is None:
                cell = "  -"
            else:
                total = total_problems.get(T, 164)
                cell = f"{v:>3}/{total} ({v/total*100:>4.1f}%)"
            row_cells.append(cell.rjust(10))
        lines.append("  " + "  ".join(row_cells))
    lines.append("")

    # Pass@N table (unbiased estimator)
    lines.append("-" * 60)
    lines.append("Pass@N (HumanEval-paper unbiased estimator)")
    lines.append("-" * 60)
    lines.append(f"  {'N':>4}  " + "  ".join(f"T={T}".rjust(10) for T in temperatures))
    for n in n_values:
        row_cells = [f"{n:>4}"]
        for T in temperatures:
            v = pass_at_n_unb[T].get(n)
            cell = "  -" if v is None else f"{v*100:>6.1f}%"
            row_cells.append(cell.rjust(10))
        lines.append("  " + "  ".join(row_cells))
    lines.append("")

    # Compile failure rates
    lines.append("-" * 60)
    lines.append("Compile-failure rate per temperature")
    lines.append("-" * 60)
    for T in temperatures:
        lines.append(f"    T={T}: {compile_fail[T]*100:>5.2f}% ({'OK' if compile_fail[T] <= GARBAGE_RATE else 'GARBAGE'})")
    lines.append("")

    # Diversity
    lines.append("-" * 60)
    lines.append("Within-pool diversity (mean pairwise normalized Levenshtein, n_cap=16)")
    lines.append("-" * 60)
    for T in temperatures:
        lines.append(f"    T={T}: {diversity[T]:.3f}")
    lines.append("")

    # Saturation
    lines.append("-" * 60)
    lines.append("Saturation analysis (delta = pass@N - pass@N/2)")
    lines.append("-" * 60)
    for T in temperatures:
        row = [f"    T={T}: "]
        for n in n_values[1:]:
            half = n // 2
            if half in pass_at_n_emp[T] and n in pass_at_n_emp[T]:
                d = pass_at_n_emp[T][n] - pass_at_n_emp[T][half]
                row.append(f"pass@{n}-pass@{half}={d:+d}")
        lines.append("  ".join(row))
    lines.append("")

    # Compute-optimal table (re-render for the report)
    lines.append("-" * 60)
    lines.append("Compute-optimal (N, T) per budget (best pass@N with N <= budget)")
    lines.append("-" * 60)
    for line in q4_lines[1:]:
        lines.append(line)
    lines.append("")

    # Final answers
    lines.append("=" * 60)
    lines.append("Final answers")
    lines.append("=" * 60)
    lines.append("")
    lines.append(q1_line)
    lines.extend(q2_lines)
    lines.append(q3_line)
    lines.extend(q4_lines)
    lines.append("")

    text = "\n".join(lines)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(text)
    print(text)
    print(f"\nWritten to {args.out}")

    # Plots (optional)
    if make_plots(pass_at_n_emp, n_values, temperatures, args.plots_dir):
        print(f"Plots written to {args.plots_dir}/")
    else:
        print("(matplotlib unavailable — skipping plots)")


if __name__ == "__main__":
    main()
