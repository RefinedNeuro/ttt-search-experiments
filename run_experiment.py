#!/usr/bin/env python3
"""TTT-Search POC v1.1 — experiment runner with 5 conditions and per-variant logging."""
from __future__ import annotations

import os

# Disable TileLang backend — nvcc/TVM not available in this environment.
os.environ.setdefault("FLA_TILELANG", "0")

import argparse
import hashlib
import json
import random
import time
from datetime import datetime

import numpy as np
import torch


# v1.1 condition configurations. v1.2 adds seedB variants for the noise-floor measurement.
CONDITIONS = {
    "base_greedy": {
        "type": "greedy",
    },
    "base_temp_lowT": {
        "type": "temperature",
        "temperature": 0.3,
        "top_p": 0.95,
        "seeds": [42, 43, 44, 45],
    },
    "base_temp_highT": {
        "type": "temperature",
        "temperature": 0.8,
        "top_p": 0.95,
        "seeds": [42, 43, 44, 45],
    },
    "ttt_lowT": {
        "type": "ttt",
        "generation_temperature": 0.3,
        "top_p": 0.95,
    },
    "ttt_highT": {
        "type": "ttt",
        "generation_temperature": 0.8,
        "top_p": 0.95,
    },
    # ── v1.2: noise-floor conditions ──────────────────────────────────────
    "base_temp_lowT_seedB": {
        "type": "temperature",
        "temperature": 0.3,
        "top_p": 0.95,
        "seeds": [46, 47, 48, 49],
    },
    "base_temp_highT_seedB": {
        "type": "temperature",
        "temperature": 0.8,
        "top_p": 0.95,
        "seeds": [46, 47, 48, 49],
    },
    # ── Sampling-dynamics sweep ───────────────────────────────────────────
    # Together with v1.1 (seeds 42-45) and v1.2 (seeds 46-49) these give 16
    # candidates per problem at T=0.3 and T=0.8, and 16 fresh candidates at
    # T=0.5, 1.0, 1.2. Source labels are seed_NN so the analyzer can recover
    # temperature from condition name and seed from candidate.
    "sweep_T03": {
        "type": "temperature",
        "temperature": 0.3,
        "top_p": 0.95,
        "seeds": [50, 51, 52, 53, 54, 55, 56, 57],
    },
    "sweep_T05": {
        "type": "temperature",
        "temperature": 0.5,
        "top_p": 0.95,
        "seeds": [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57],
    },
    "sweep_T08": {
        "type": "temperature",
        "temperature": 0.8,
        "top_p": 0.95,
        "seeds": [50, 51, 52, 53, 54, 55, 56, 57],
    },
    "sweep_T10": {
        "type": "temperature",
        "temperature": 1.0,
        "top_p": 0.95,
        "seeds": [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57],
    },
    "sweep_T12": {
        "type": "temperature",
        "temperature": 1.2,
        "top_p": 0.95,
        "seeds": [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57],
    },
}


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def run_condition(
    name: str, config: dict, model, tokenizer, problems: list, seed: int,
    checkpoint_path: str | None = None,
) -> list[dict]:
    """Run one condition over all problems. Returns list of v1.1 schema records."""
    from ttt_search.conditions import base_greedy, base_temperature, ttt_search

    results = []
    total = len(problems)
    for idx, (task_id, prompt, test_code, entry_point) in enumerate(problems, 1):
        print(f"  [{name}] {idx}/{total} {task_id} ...", flush=True)

        if config["type"] == "greedy":
            result = base_greedy.generate(model, tokenizer, prompt, test_code, entry_point, seed=seed)
        elif config["type"] == "temperature":
            result = base_temperature.generate(
                model, tokenizer, prompt, test_code, entry_point,
                temperature=config["temperature"], top_p=config["top_p"],
                seeds=config.get("seeds"),
            )
        elif config["type"] == "ttt":
            result = ttt_search.generate(
                model, tokenizer, prompt, test_code, entry_point,
                generation_temperature=config["generation_temperature"], top_p=config["top_p"],
            )
        else:
            raise ValueError(f"Unknown condition type: {config['type']}")

        record = {
            "task_id": task_id,
            "condition": name,
            "candidates": result["candidates"],
            "selected_index": result["selected_index"],
            "any_passed": result["any_passed"],
        }
        results.append(record)

        status = "PASS" if record["any_passed"] else "FAIL"
        n_pass = sum(1 for c in result["candidates"] if c["passed"])
        print(f"    -> {status} ({n_pass}/{len(result['candidates'])} candidates passed)", flush=True)

        if checkpoint_path:
            with open(checkpoint_path, "w") as f:
                json.dump(results, f, indent=2)

    return results


def write_summary(
    all_results: dict[str, list[dict]],
    num_problems: int,
    seed: int,
    runtime_seconds: float,
    model_name: str,
    priming_hashes: dict[str, str],
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Flatten all condition results into detailed.json (v1.1 schema)
    flat = [r for cond_results in all_results.values() for r in cond_results]
    with open(os.path.join(output_dir, "detailed.json"), "w") as f:
        json.dump(flat, f, indent=2)

    lines = [
        "TTT-Search POC v1.1 — Results",
        "============================",
        f"Model:          {model_name}",
        f"Problems:       HumanEval[0:{num_problems}]",
        f"Seed:           {seed}",
        f"Total runtime:  {runtime_seconds/60:.1f} minutes",
        f"Timestamp:      {datetime.utcnow().isoformat()}Z",
        "",
        "Hyperparameters:",
        "  LoRA r=4, target_modules=[q_proj,v_proj], lora_alpha=8, lora_dropout=0.0",
        "  TTT: steps=8, lr=1e-4, AdamW",
        "  base_temp_lowT:   T=0.3, top_p=0.95, n=4 (seeds 42-45)",
        "  base_temp_highT:  T=0.8, top_p=0.95, n=4 (seeds 42-45)",
        "  ttt_lowT:         generation T=0.3, top_p=0.95, n=4 priming texts",
        "  ttt_highT:        generation T=0.8, top_p=0.95, n=4 priming texts",
        "",
        "Priming text SHA-256:",
    ]
    for fname, digest in priming_hashes.items():
        lines.append(f"  {fname}: {digest}")
    lines += ["", f"{'Condition':<25} {'Pass Rate':<12} {'Pass Count'}", "-" * 55]

    for cond_name, cond_results in all_results.items():
        n = len(cond_results)
        passed = sum(1 for r in cond_results if r["any_passed"])
        rate = passed / n * 100 if n else 0.0
        lines.append(f"{cond_name:<25} {rate:>5.1f}%       {passed}/{n}")

    summary = "\n".join(lines) + "\n"
    path = os.path.join(output_dir, "summary.txt")
    with open(path, "w") as f:
        f.write(summary)
    print("\n" + summary)
    print(f"Results written to {output_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="TTT-Search POC v1.1")
    parser.add_argument("--num-problems", type=int, default=164)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--conditions", type=str,
        default=",".join(CONDITIONS.keys()),
        help="Comma-separated condition names",
    )
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--skip-analysis", action="store_true",
                        help="Skip running analysis.py after the experiment")
    args = parser.parse_args()

    set_seeds(args.seed)
    requested = [c.strip() for c in args.conditions.split(",")]
    for c in requested:
        if c not in CONDITIONS:
            raise SystemExit(f"Unknown condition '{c}'. Choices: {list(CONDITIONS)}")

    from ttt_search.benchmark import iter_problems
    problems = list(iter_problems(n=args.num_problems))
    print(f"Loaded {len(problems)} problems.")

    from ttt_search.loader import get_model_and_tokenizer, MODEL_NAME
    print(f"Loading model {MODEL_NAME} ...")
    model, tokenizer = get_model_and_tokenizer()
    print("Model loaded.")

    priming_dir = os.path.join(os.path.dirname(__file__), "priming")
    priming_files = ["iterative.txt", "recursive.txt", "edge_cases.txt", "builtins.txt"]
    priming_hashes = {}
    for fname in priming_files:
        p = os.path.join(priming_dir, fname)
        if os.path.exists(p):
            priming_hashes[fname] = sha256_file(p)

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoints_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    all_results: dict[str, list[dict]] = {}
    t0 = time.time()

    for cond in requested:
        print(f"\n=== Condition: {cond} ===")
        set_seeds(args.seed)
        ckpt = os.path.join(checkpoints_dir, f"{cond}.json")
        all_results[cond] = run_condition(
            cond, CONDITIONS[cond], model, tokenizer, problems, args.seed, ckpt,
        )

    runtime = time.time() - t0
    write_summary(all_results, args.num_problems, args.seed, runtime, MODEL_NAME,
                  priming_hashes, args.output_dir)

    if not args.skip_analysis:
        print("\nRunning auto-analysis ...")
        from ttt_search.analysis import analyze
        analyze(
            os.path.join(args.output_dir, "detailed.json"),
            os.path.join(args.output_dir, "analysis.txt"),
        )
        with open(os.path.join(args.output_dir, "analysis.txt")) as f:
            print("\n" + f.read())


if __name__ == "__main__":
    main()
