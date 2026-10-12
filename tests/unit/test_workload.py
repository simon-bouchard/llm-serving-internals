import torch

from llm_serving.workload import make_prompt


class StubTokenizer:
    vocab_size = 10
    all_special_ids = [0, 1]


def test_make_prompt_returns_correct_shape():
    prompt = make_prompt(StubTokenizer(), input_len=16, seed=0)
    assert prompt.shape == (1, 16)
    assert prompt.dtype == torch.long


def test_make_prompt_excludes_special_tokens():
    prompt = make_prompt(StubTokenizer(), input_len=50, seed=0)
    token_ids = prompt[0].tolist()
    assert not (set(token_ids) & set(StubTokenizer.all_special_ids))


def test_make_prompt_is_reproducible_with_same_seed():
    first = make_prompt(StubTokenizer(), input_len=16, seed=42)
    second = make_prompt(StubTokenizer(), input_len=16, seed=42)
    assert torch.equal(first, second)


def test_make_prompt_differs_with_different_seed():
    first = make_prompt(StubTokenizer(), input_len=16, seed=1)
    second = make_prompt(StubTokenizer(), input_len=16, seed=2)
    assert not torch.equal(first, second)
