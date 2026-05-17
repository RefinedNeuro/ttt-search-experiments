"""ttt_search: parameterized generation temperature. Warm up on priming text, generate, verify.

v1.1: returns per-variant data; generation_temperature is now a parameter.
"""
from __future__ import annotations

import os
import time
import torch
from torch.optim import AdamW
from ..lora_utils import reset_lora_weights
from ..prompt_utils import format_prompt, format_priming, extract_code
from ..verifier import verify

PRIMING_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "priming")
PRIMING_FILES = ["iterative.txt", "recursive.txt", "edge_cases.txt", "builtins.txt"]

TTT_STEPS = 8
TTT_LR = 1e-4


def _load_priming_texts() -> list[tuple[str, str]]:
    """Returns [(label_stem, text), ...]."""
    out = []
    for fname in PRIMING_FILES:
        path = os.path.join(PRIMING_DIR, fname)
        with open(path) as f:
            out.append((os.path.splitext(fname)[0], f.read()))
    return out


def _warmup(model, tokenizer, priming_text: str) -> None:
    model.train()
    lora_params = [
        p for n, p in model.named_parameters()
        if ("lora_A" in n or "lora_B" in n) and p.requires_grad
    ]
    optimizer = AdamW(lora_params, lr=TTT_LR)

    input_ids = format_priming(tokenizer, priming_text).to(model.device)

    for _ in range(TTT_STEPS):
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=input_ids)
        outputs.loss.backward()
        optimizer.step()

    model.eval()


def generate(
    model, tokenizer, prompt: str, test_code: str, entry_point: str,
    generation_temperature: float = 0.3, top_p: float = 0.95,
) -> dict:
    priming_texts = _load_priming_texts()
    candidates = []

    input_ids = format_prompt(tokenizer, prompt).to(model.device)

    for i, (label, priming_text) in enumerate(priming_texts):
        reset_lora_weights(model)
        _warmup(model, tokenizer, priming_text)

        torch.manual_seed(42 + i)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=768,
                do_sample=True,
                temperature=generation_temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_time = time.time() - t0

        new_tokens = out[0][input_ids.shape[1]:]
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
        code = extract_code(raw, prompt)
        passed, error = verify(code, test_code, entry_point)
        candidates.append({
            "index": i,
            "source_label": label,
            "code": code,
            "passed": passed,
            "error": error,
            "generation_seconds": gen_time,
        })

    # Selection rule: first passing candidate, falling back to index 0
    selected = 0
    for c in candidates:
        if c["passed"]:
            selected = c["index"]
            break

    return {
        "candidates": candidates,
        "selected_index": selected,
        "any_passed": any(c["passed"] for c in candidates),
    }
