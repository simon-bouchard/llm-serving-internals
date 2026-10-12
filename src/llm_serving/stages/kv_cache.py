from __future__ import annotations

import time

import torch

from llm_serving.metrics import RequestMetrics
from llm_serving.stages.base import register_stage


def _sync(device: torch.device) -> None:
    """Block until queued GPU work finishes, so perf_counter() reads completion time."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def decode(
    model, input_ids: torch.Tensor, output_len: int, use_cache: bool
) -> tuple[torch.Tensor, float, list[float]]:
    """Greedy-decode output_len tokens with a hand-written loop.

    With use_cache, each step feeds only the new token plus the HF cache object; without it,
    each step re-feeds the full sequence. Every call passes logits_to_keep=1, as generate() does,
    so lm_head runs on the last position only. Returns (tokens, start, per-token timestamps).
    """
    if output_len < 1:
        raise ValueError(f"output_len must be >= 1, got {output_len}")

    device = input_ids.device
    _sync(device)
    start = time.perf_counter()

    with torch.no_grad():
        outputs = model(input_ids, use_cache=use_cache, logits_to_keep=1)
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        _sync(device)
        timestamps = [time.perf_counter()]

        cache = outputs.past_key_values if use_cache else None
        sequence = torch.cat([input_ids, next_token], dim=-1)
        generated = [next_token]

        for _ in range(output_len - 1):
            if use_cache:
                outputs = model(next_token, past_key_values=cache, use_cache=True, logits_to_keep=1)
                cache = outputs.past_key_values
            else:
                outputs = model(sequence, use_cache=False, logits_to_keep=1)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            _sync(device)
            timestamps.append(time.perf_counter())

            sequence = torch.cat([sequence, next_token], dim=-1)
            generated.append(next_token)

    return torch.cat(generated, dim=-1), start, timestamps


class ManualDecodeStage:
    """Stage 2: hand-written greedy decode loop, with or without the KV cache."""

    def __init__(self, use_cache: bool) -> None:
        self.use_cache = use_cache

    def generate_one(
        self, model, tokenizer, input_ids: torch.Tensor, output_len: int
    ) -> RequestMetrics:
        input_ids = input_ids.to(model.device)
        _, start, timestamps = decode(model, input_ids, output_len, self.use_cache)

        return RequestMetrics(
            request_id=0,
            prompt_tokens=input_ids.shape[-1],
            output_tokens=len(timestamps),
            ttft_s=timestamps[0] - start,
            total_latency_s=timestamps[-1] - start,
            inter_token_latencies_s=[t2 - t1 for t1, t2 in zip(timestamps, timestamps[1:])],
        )


register_stage("kv_cache", ManualDecodeStage(use_cache=True))
register_stage("no_cache", ManualDecodeStage(use_cache=False))
