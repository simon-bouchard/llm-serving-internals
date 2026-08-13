from __future__ import annotations

import time

import torch
from transformers.generation.streamers import BaseStreamer

from llm_serving.metrics import RequestMetrics
from llm_serving.stages.base import register_stage


class TimestampStreamer(BaseStreamer):
    """Records one timestamp per generated token.

    generate() unconditionally calls put() once with the full prompt before
    generation starts (transformers/generation/utils.py), so the first put()
    call is a prompt echo, not a generated token, and must not be timed.
    """

    def __init__(self) -> None:
        self.timestamps: list[float] = []
        self._skipped_prompt_put = False

    def put(self, value) -> None:
        if not self._skipped_prompt_put:
            self._skipped_prompt_put = True
            return
        self.timestamps.append(time.perf_counter())

    def end(self) -> None:
        pass


class NaiveStage:
    def generate_one(
        self, model, tokenizer, input_ids: torch.Tensor, output_len: int
    ) -> RequestMetrics:
        streamer = TimestampStreamer()
        input_ids = input_ids.to(model.device)
        prompt_tokens = input_ids.shape[-1]

        start = time.perf_counter()
        with torch.no_grad():
            model.generate(
                input_ids,
                min_new_tokens=output_len,
                max_new_tokens=output_len,
                do_sample=False,
                streamer=streamer,
                pad_token_id=tokenizer.pad_token_id,
            )

        timestamps = streamer.timestamps
        ttft_s = timestamps[0] - start
        total_latency_s = timestamps[-1] - start
        inter_token_latencies_s = [t2 - t1 for t1, t2 in zip(timestamps, timestamps[1:])]

        return RequestMetrics(
            request_id=0,
            prompt_tokens=prompt_tokens,
            output_tokens=len(timestamps),
            ttft_s=ttft_s,
            total_latency_s=total_latency_s,
            inter_token_latencies_s=inter_token_latencies_s,
        )


register_stage("naive", NaiveStage())
