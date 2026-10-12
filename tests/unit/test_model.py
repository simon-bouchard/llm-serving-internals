import torch

from llm_serving.model import resolve_device, resolve_dtype


def test_resolve_device_returns_explicit_value_unchanged():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"


def test_resolve_device_auto_detects(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device(None) == "cuda"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device(None) == "cpu"


def test_resolve_dtype_returns_explicit_value_unchanged():
    assert resolve_dtype(torch.bfloat16, "cuda") == torch.bfloat16


def test_resolve_dtype_auto_selects_fp16_on_cuda():
    assert resolve_dtype(None, "cuda") == torch.float16


def test_resolve_dtype_auto_selects_fp32_on_cpu():
    assert resolve_dtype(None, "cpu") == torch.float32
