import torch

from llm_serving.stages.base import STAGES, register_stage


def test_register_stage_adds_to_registry(fake_stage):
    register_stage("fake_test_stage", fake_stage)
    assert "fake_test_stage" in STAGES


def test_registered_stage_generate_one_returns_request_metrics(fake_stage):
    register_stage("fake_test_stage_2", fake_stage)
    result = STAGES["fake_test_stage_2"].generate_one(
        None, None, torch.zeros(1, 8, dtype=torch.long), 4
    )
    assert result.output_tokens == 4
