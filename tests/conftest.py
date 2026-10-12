import pytest
import torch

from llm_serving.metrics import RequestMetrics
from llm_serving.model import load_model_and_tokenizer

TINY_MODEL_NAME = "yujiepan/qwen2-tiny-random"


class FakeModel:
    """Model stand-in exposing only parameters(), for device/dtype introspection."""

    def parameters(self):
        return iter([torch.zeros(1, dtype=torch.float32, device="cpu")])


class FakeTokenizer:
    """Tokenizer stand-in with the attributes make_prompt and the stages read."""

    pad_token_id = 0
    vocab_size = 100
    all_special_ids = [0]


class FakeStage:
    """Stage stub returning fixed metrics and counting generate_one() calls."""

    def __init__(self) -> None:
        self.calls = 0

    def generate_one(self, model, tokenizer, input_ids, output_len) -> RequestMetrics:
        self.calls += 1
        return RequestMetrics(
            request_id=0,
            prompt_tokens=input_ids.shape[-1],
            output_tokens=output_len,
            ttft_s=0.01,
            total_latency_s=0.05,
            inter_token_latencies_s=[0.01] * max(output_len - 1, 0),
        )


@pytest.fixture
def fake_model() -> FakeModel:
    return FakeModel()


@pytest.fixture
def fake_tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


@pytest.fixture
def fake_stage() -> FakeStage:
    return FakeStage()


@pytest.fixture(scope="session")
def tiny_model_and_tokenizer():
    """Real tiny Qwen2 model on CPU; downloaded once per machine, loaded once per run."""
    return load_model_and_tokenizer(TINY_MODEL_NAME, device="cpu")


@pytest.fixture(scope="session")
def tiny_model(tiny_model_and_tokenizer):
    return tiny_model_and_tokenizer[0]


@pytest.fixture(scope="session")
def tiny_tokenizer(tiny_model_and_tokenizer):
    return tiny_model_and_tokenizer[1]
