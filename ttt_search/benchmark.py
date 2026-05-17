"""Loads HumanEval problems and yields (task_id, prompt, test, entry_point)."""
from __future__ import annotations

import json
import os
from typing import Iterator


def _load_humaneval() -> list[dict]:
    # Try the human_eval package first
    try:
        from human_eval.data import read_problems
        problems = read_problems()
        return list(problems.values())
    except ImportError:
        pass

    # Fall back to a local JSON file
    local = os.path.join(os.path.dirname(__file__), "..", "HumanEval.jsonl")
    if os.path.exists(local):
        with open(local) as f:
            return [json.loads(line) for line in f if line.strip()]

    raise RuntimeError(
        "HumanEval not found. Install via: pip install human-eval  "
        "or place HumanEval.jsonl next to this package."
    )


def iter_problems(n: int = 20, start: int = 0) -> Iterator[tuple[str, str, str, str]]:
    """Yield (task_id, prompt, test, entry_point) for `n` problems starting at `start`."""
    problems = _load_humaneval()
    for prob in problems[start : start + n]:
        yield (
            prob["task_id"],
            prob["prompt"],
            prob["test"],
            prob["entry_point"],
        )
