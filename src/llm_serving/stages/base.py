from __future__ import annotations

from typing import Protocol

import torch

from llm_serving.metrics import RequestMetrics


class Stage(Protocol):
    def generate_one(
        self, model, tokenizer, input_ids: torch.Tensor, output_len: int
    ) -> RequestMetrics: ...


STAGES: dict[str, Stage] = {}


def register_stage(name: str, stage: Stage) -> None:
    STAGES[name] = stage
