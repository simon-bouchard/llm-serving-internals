import torch

from llm_serving.stages.base import STAGES
from llm_serving.stages.naive import TimestampStreamer, greedy_generate


class FakeGenerateModel:
    """Records generate() kwargs and appends output_len copies of token 7 to the prompt."""

    def __init__(self) -> None:
        self.kwargs: dict = {}

    def generate(self, input_ids, **kwargs):
        self.kwargs = kwargs
        new_tokens = torch.full((input_ids.shape[0], kwargs["max_new_tokens"]), 7, dtype=torch.long)
        return torch.cat([input_ids, new_tokens], dim=-1)


def test_greedy_generate_returns_only_new_tokens(fake_tokenizer):
    model = FakeGenerateModel()
    prompt = torch.tensor([[1, 2, 3]])

    tokens = greedy_generate(model, fake_tokenizer, prompt, output_len=4)

    assert tokens.tolist() == [[7, 7, 7, 7]]


def test_greedy_generate_overrides_model_sampling_defaults(fake_tokenizer):
    model = FakeGenerateModel()

    greedy_generate(model, fake_tokenizer, torch.tensor([[1, 2, 3]]), output_len=4)

    assert model.kwargs["do_sample"] is False
    assert model.kwargs["repetition_penalty"] == 1.0
    assert model.kwargs["min_new_tokens"] == 4
    assert model.kwargs["max_new_tokens"] == 4


def test_timestamp_streamer_skips_the_first_put_call():
    # generate() calls put() once with the full prompt before generation starts.
    streamer = TimestampStreamer()
    streamer.put("prompt_echo")

    assert streamer.timestamps == []


def test_timestamp_streamer_records_one_timestamp_per_put_call_after_the_first():
    streamer = TimestampStreamer()
    streamer.put("prompt_echo")
    streamer.put("token_a")
    streamer.put("token_b")
    streamer.put("token_c")

    assert len(streamer.timestamps) == 3


def test_timestamp_streamer_timestamps_are_increasing():
    streamer = TimestampStreamer()
    streamer.put("prompt_echo")
    streamer.put("a")
    streamer.put("b")

    assert streamer.timestamps[1] >= streamer.timestamps[0]


def test_naive_stage_registers_itself():
    assert "naive" in STAGES
