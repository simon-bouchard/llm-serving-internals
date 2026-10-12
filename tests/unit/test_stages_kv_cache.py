from types import SimpleNamespace

import pytest
import torch

from llm_serving.stages import kv_cache
from llm_serving.stages.base import STAGES
from llm_serving.stages.kv_cache import ManualDecodeStage, decode


class FakeCausalLM:
    """Records each forward call; greedily predicts (last input token + 1) mod vocab."""

    device = torch.device("cpu")
    vocab_size = 50

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], bool, bool]] = []

    def __call__(self, input_ids, past_key_values=None, use_cache=False):
        self.calls.append((tuple(input_ids.shape), past_key_values is not None, use_cache))
        batch, seq = input_ids.shape
        logits = torch.zeros(batch, seq, self.vocab_size)
        next_ids = (input_ids[:, -1] + 1) % self.vocab_size
        logits[torch.arange(batch), -1, next_ids] = 1.0
        return SimpleNamespace(logits=logits, past_key_values=object() if use_cache else None)


PROMPT = torch.tensor([[3, 4, 5]])


def test_decode_with_cache_feeds_only_the_new_token():
    model = FakeCausalLM()

    tokens, _, _ = decode(model, PROMPT, output_len=4, use_cache=True)

    assert tokens.tolist() == [[6, 7, 8, 9]]
    assert [shape for shape, _, _ in model.calls] == [(1, 3), (1, 1), (1, 1), (1, 1)]
    assert [has_cache for _, has_cache, _ in model.calls] == [False, True, True, True]


def test_decode_without_cache_refeeds_the_full_sequence():
    model = FakeCausalLM()

    tokens, _, _ = decode(model, PROMPT, output_len=4, use_cache=False)

    assert tokens.tolist() == [[6, 7, 8, 9]]
    assert [shape for shape, _, _ in model.calls] == [(1, 3), (1, 4), (1, 5), (1, 6)]
    assert all(not has_cache and not use for _, has_cache, use in model.calls)


def test_decode_returns_one_ordered_timestamp_per_token():
    _, start, timestamps = decode(FakeCausalLM(), PROMPT, output_len=4, use_cache=True)

    assert len(timestamps) == 4
    assert start <= timestamps[0]
    assert timestamps == sorted(timestamps)


def test_decode_output_len_one_runs_prefill_only():
    model = FakeCausalLM()

    tokens, _, timestamps = decode(model, PROMPT, output_len=1, use_cache=True)

    assert tokens.tolist() == [[6]]
    assert len(model.calls) == 1
    assert len(timestamps) == 1


@pytest.mark.parametrize("output_len", [0, -1])
def test_decode_rejects_non_positive_output_len(output_len):
    with pytest.raises(ValueError):
        decode(FakeCausalLM(), PROMPT, output_len=output_len, use_cache=True)


@pytest.mark.parametrize("use_cache", [True, False])
def test_decode_syncs_before_every_timestamp(monkeypatch, use_cache):
    sync_calls: list[torch.device] = []
    monkeypatch.setattr(kv_cache, "_sync", sync_calls.append)

    decode(FakeCausalLM(), PROMPT, output_len=4, use_cache=use_cache)

    assert len(sync_calls) == 5  # start + prefill + 3 decode steps


@pytest.mark.parametrize("use_cache", [True, False])
def test_generate_one_reports_consistent_metrics(fake_tokenizer, use_cache):
    stage = ManualDecodeStage(use_cache=use_cache)

    result = stage.generate_one(FakeCausalLM(), fake_tokenizer, PROMPT, output_len=4)

    assert result.prompt_tokens == 3
    assert result.output_tokens == 4
    assert len(result.inter_token_latencies_s) == 3
    assert 0 <= result.ttft_s <= result.total_latency_s


def test_generate_one_output_len_one_has_no_itls(fake_tokenizer):
    result = ManualDecodeStage(use_cache=True).generate_one(
        FakeCausalLM(), fake_tokenizer, PROMPT, output_len=1
    )

    assert result.inter_token_latencies_s == []
    assert result.ttft_s == result.total_latency_s


def test_stages_register_with_matching_cache_flags():
    kv_stage = STAGES["kv_cache"]
    no_cache_stage = STAGES["no_cache"]
    assert isinstance(kv_stage, ManualDecodeStage) and kv_stage.use_cache is True
    assert isinstance(no_cache_stage, ManualDecodeStage) and no_cache_stage.use_cache is False
