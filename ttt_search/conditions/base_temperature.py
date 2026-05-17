"""base_temperature: parameterized temperature, best-of-4, LoRA zeroed.

v1.1: returns per-variant data including full code, pass status, generation time.
v1.2: seeds are now a parameter so we can run a second seed set for the noise floor.
"""
from __future__ import annotations

import time
import torch
from ..lora_utils import zero_lora_weights
from ..prompt_utils import format_prompt, extract_code
from ..verifier import verify

DEFAULT_SEEDS = [42, 43, 44, 45]


def generate(
    model, tokenizer, prompt: str, test_code: str, entry_point: str,
    temperature: float = 0.8, top_p: float = 0.95,
    seeds: list[int] | None = None,
) -> dict:
    zero_lora_weights(model)
    seeds = seeds or DEFAULT_SEEDS

    input_ids = format_prompt(tokenizer, prompt).to(model.device)

    # Batched generation. Seed the RNG with seeds[0] so the same seed list always
    # produces the same set of candidates. (Each token is sampled stochastically
    # using shared GPU RNG state, so per-candidate seeds aren't directly used,
    # but the seeds-derived RNG seeding makes the run reproducible.)
    torch.manual_seed(seeds[0])
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=768,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=len(seeds),
            pad_token_id=tokenizer.eos_token_id,
        )
    batch_time = time.time() - t0

    prompt_len = input_ids.shape[1]
    candidates = []
    for i in range(out.shape[0]):
        new_tokens = out[i][prompt_len:]
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
        code = extract_code(raw, prompt)
        passed, error = verify(code, test_code, entry_point)
        candidates.append({
            "index": i,
            "source_label": f"seed_{seeds[i]}",
            "code": code,
            "passed": passed,
            "error": error,
            "generation_seconds": batch_time,
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
