"""Helpers for chat-template formatting and response extraction."""
from __future__ import annotations

import re
import torch


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
# Splits on any opening fence marker
_FENCE_SPLIT_RE = re.compile(r"```(?:python)?\s*\n?", re.IGNORECASE)
# Strips a closing fence at the end of a block
_CLOSE_FENCE_RE = re.compile(r"\n?```\s*$")


def format_prompt(tokenizer, problem_prompt: str) -> torch.Tensor:
    """Wrap the HumanEval prompt in a chat message and return input_ids."""
    messages = [
        {
            "role": "user",
            "content": (
                "Complete the following Python function. "
                "Return ONLY the complete function code in a Python code block.\n\n"
                f"```python\n{problem_prompt.strip()}\n```"
            ),
        }
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    )
    return input_ids


def format_priming(tokenizer, text: str, max_length: int = 512) -> torch.Tensor:
    """Tokenize priming text as plain causal-LM input."""
    return tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )["input_ids"]


def extract_code(raw: str, original_prompt: str) -> str:
    """
    Extract function code from a model response robustly.
    Handles: thinking tags, multiple fenced blocks (model self-corrections),
    truncated responses (no closing fence).

    When the model generates multiple code blocks ("Wait, let me fix..."),
    we take the LAST block — models consistently self-correct in later blocks.
    """
    text = _THINK_RE.sub("", raw).strip()

    fn_name = _extract_fn_name(original_prompt)

    # Split on any fence opener — parts[0] is pre-fence text, parts[1::2] are code blocks
    parts = _FENCE_SPLIT_RE.split(text)
    if len(parts) > 1:
        # Collect all code blocks (everything after each fence opener)
        code_blocks = []
        for block in parts[1:]:
            block = _CLOSE_FENCE_RE.sub("", block).strip()
            if block:
                code_blocks.append(block)

        if code_blocks:
            # Prefer the last block that contains the function definition
            if fn_name:
                fn_pat = re.compile(rf"^\s*def\s+{re.escape(fn_name)}\s*\(", re.MULTILINE)
                for block in reversed(code_blocks):
                    if fn_pat.search(block):
                        return block
            # Fall back to the last non-empty block
            return code_blocks[-1]

    # No fence found — use the raw text as-is if it has the function definition
    if fn_name and re.search(rf"^\s*def\s+{re.escape(fn_name)}\s*\(", text, re.MULTILINE):
        return text.strip()

    # Last resort: model returned just the body — append to original prompt
    return original_prompt.rstrip() + "\n" + _ensure_indented(text, 4)


def _extract_fn_name(prompt: str) -> str | None:
    m = re.search(r"def\s+(\w+)\s*\(", prompt)
    return m.group(1) if m else None


def _ensure_indented(text: str, spaces: int) -> str:
    """Indent lines that are not already indented and are non-empty."""
    pad = " " * spaces
    result = []
    for line in text.splitlines():
        if line.strip() and not line.startswith(" ") and not line.startswith("\t"):
            result.append(pad + line)
        else:
            result.append(line)
    return "\n".join(result)
