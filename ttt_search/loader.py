"""Loads the base model once; all conditions share the same instance."""
from __future__ import annotations

import torch

_model = None
_tokenizer = None

LORA_CONFIG = dict(
    r=4,
    target_modules=["q_proj", "v_proj"],
    lora_alpha=8,
    lora_dropout=0.0,
    bias="none",
)

# Candidate tags in preference order. The first one that loads successfully is used.
# Verified available as of 2026-05: unsloth/Qwen3.5-0.8B (instruct, no pre-quantized 4bit variant)
# Unsloth will quantize to 4-bit NF4 on load via load_in_4bit=True.
_MODEL_CANDIDATES = [
    "unsloth/Qwen3.5-0.8B",                 # confirmed available (design doc target)
    "unsloth/Qwen3-0.6B-unsloth-bnb-4bit",  # fallback
]

MODEL_NAME: str = _MODEL_CANDIDATES[0]  # updated after successful load


def get_model_and_tokenizer():
    global _model, _tokenizer, MODEL_NAME
    if _model is not None:
        return _model, _tokenizer

    from unsloth import FastLanguageModel

    # BF16, no 4-bit: bitsandbytes dequantization is the dominant cost for this model size.
    # 0.8B params * 2 bytes = 1.6 GB, fits easily on 24 GB GPUs.
    # The design doc spec'd 4-bit for 8 GB GPUs; deviation is documented in results/summary.txt.
    last_err = None
    for tag in _MODEL_CANDIDATES:
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=tag,
                max_seq_length=2048,
                dtype=torch.bfloat16,
                load_in_4bit=False,
            )
            MODEL_NAME = tag
            print(f"Loaded model: {tag}")
            break
        except Exception as e:
            print(f"Could not load {tag}: {e}")
            last_err = e
    else:
        raise RuntimeError(f"All model candidates failed. Last error: {last_err}")

    model = FastLanguageModel.get_peft_model(
        model,
        **LORA_CONFIG,
        use_gradient_checkpointing="unsloth",  # required for Qwen3.5 GatedDeltaRule backward
        random_state=42,
    )

    # Qwen3.5-0.8B is a VLM — its processor treats plain text as image URLs.
    # Use the inner text-only tokenizer for all text operations.
    text_tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer

    _model = model
    _tokenizer = text_tokenizer
    return _model, _tokenizer
