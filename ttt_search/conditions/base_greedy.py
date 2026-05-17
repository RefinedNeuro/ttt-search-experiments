"""base_greedy: temperature=0, single attempt, LoRA zeroed."""
from __future__ import annotations

import time
import torch
from ..lora_utils import zero_lora_weights
from ..prompt_utils import format_prompt, extract_code
from ..verifier import verify


def generate(model, tokenizer, prompt: str, test_code: str, entry_point: str, seed: int = 42) -> dict:
    zero_lora_weights(model)

    input_ids = format_prompt(tokenizer, prompt).to(model.device)
    torch.manual_seed(seed)

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=768,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_time = time.time() - t0

    new_tokens = out[0][input_ids.shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
    code = extract_code(raw, prompt)
    passed, error = verify(code, test_code, entry_point)

    return {
        "candidates": [{
            "index": 0,
            "source_label": "greedy",
            "code": code,
            "passed": passed,
            "error": error,
            "generation_seconds": gen_time,
        }],
        "selected_index": 0,
        "any_passed": passed,
    }
