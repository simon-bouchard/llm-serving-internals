from __future__ import annotations

import random

import torch


def make_prompt(tokenizer, input_len: int, seed: int) -> torch.Tensor:
    rng = random.Random(seed)
    special_ids = set(tokenizer.all_special_ids)
    vocab_size = tokenizer.vocab_size

    token_ids: list[int] = []
    while len(token_ids) < input_len:
        candidate = rng.randrange(vocab_size)
        if candidate not in special_ids:
            token_ids.append(candidate)

    return torch.tensor([token_ids], dtype=torch.long)
