"""Utilities for resetting LoRA weights between variants."""
from __future__ import annotations

import torch


def zero_lora_weights(model) -> None:
    """Zero all LoRA A/B matrices so generation falls back to the base model."""
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            with torch.no_grad():
                param.zero_()


def reset_lora_weights(model) -> None:
    """Re-initialize LoRA A with kaiming_uniform and B with zeros (standard PEFT init)."""
    import math
    for name, param in model.named_parameters():
        if "lora_A" in name:
            with torch.no_grad():
                torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
        elif "lora_B" in name:
            with torch.no_grad():
                param.zero_()
