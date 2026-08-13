import torch

from llm_serving.metrics import RequestMetrics
from llm_serving.stages.base import STAGES, register_stage


class FakeStage:
    def generate_one(self, model, tokenizer, input_ids, output_len):
        return RequestMetrics(
            request_id=0,
            prompt_tokens=input_ids.shape[-1],
            output_tokens=output_len,
            ttft_s=0.01,
            total_latency_s=0.05,
            inter_token_latencies_s=[0.01, 0.01],
        )


def test_register_stage_adds_to_registry():
    register_stage("fake_test_stage", FakeStage())
    assert "fake_test_stage" in STAGES


def test_registered_stage_generate_one_returns_request_metrics():
    register_stage("fake_test_stage_2", FakeStage())
    result = STAGES["fake_test_stage_2"].generate_one(
        None, None, torch.zeros(1, 8, dtype=torch.long), 4
    )
    assert result.output_tokens == 4
