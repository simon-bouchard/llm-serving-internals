import pytest
import torch

from llm_serving.stages.kv_cache import ManualDecodeStage, decode
from llm_serving.stages.naive import greedy_generate
from llm_serving.workload import make_prompt

PROMPT_LEN = 16
OUTPUT_LEN = 12
SEED = 0


@pytest.fixture(scope="module")
def prompt(tiny_tokenizer) -> torch.Tensor:
    return make_prompt(tiny_tokenizer, PROMPT_LEN, seed=SEED)


@pytest.fixture(scope="module")
def reference_tokens(tiny_model, tiny_tokenizer, prompt) -> torch.Tensor:
    return greedy_generate(tiny_model, tiny_tokenizer, prompt, OUTPUT_LEN)


def test_reference_output_contains_a_repeated_token(reference_tokens):
    # A repeat is what exposed the repetition-penalty mismatch; keep it covered.
    assert reference_tokens.unique().numel() < OUTPUT_LEN


@pytest.mark.parametrize("use_cache", [True, False])
def test_decode_matches_naive_greedy_generate(tiny_model, prompt, reference_tokens, use_cache):
    tokens, _, _ = decode(tiny_model, prompt, OUTPUT_LEN, use_cache=use_cache)

    assert torch.equal(tokens, reference_tokens)


@pytest.mark.parametrize("use_cache", [True, False])
def test_generate_one_on_real_model_reports_all_tokens(
    tiny_model, tiny_tokenizer, prompt, use_cache
):
    result = ManualDecodeStage(use_cache=use_cache).generate_one(
        tiny_model, tiny_tokenizer, prompt, OUTPUT_LEN
    )

    assert result.output_tokens == OUTPUT_LEN
    assert len(result.inter_token_latencies_s) == OUTPUT_LEN - 1
