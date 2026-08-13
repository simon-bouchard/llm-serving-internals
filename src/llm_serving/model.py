from __future__ import annotations

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


def resolve_device(device: str | None) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dtype(dtype: torch.dtype | None, device: str) -> torch.dtype:
    if dtype is not None:
        return dtype
    return torch.float16 if device == "cuda" else torch.float32


def load_model_and_tokenizer(
    model_name: str,
    device: str | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    resolved_device = resolve_device(device)
    resolved_dtype = resolve_dtype(dtype, resolved_device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=resolved_dtype
    )
    model.to(resolved_device)  # pyright: ignore[reportArgumentType] -- transformers stub misresolves .to()
    model.eval()

    return model, tokenizer
