# Scaffold + Naive-Generate Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scaffold `llm-serving-internals` as an installable `src/`-layout package and implement the first benchmark stage (naive `model.generate()` baseline) behind a single unified benchmarking driver, so stages 2–4 and the vLLM leg can slot into the same harness later.

**Architecture:** A `src/llm_serving` package holds all reusable logic (model loading, synthetic workload generation, the metrics/aggregation harness, and a `stages` sub-package where each generation strategy registers itself under a name). One CLI (`llm_serving.cli`, exposed as `benchmarks/run.py`) drives every stage identically. Each numbered experiment directory under `benchmarks/` gets a thin wrapper that shells out to that one driver with fixed args, plus a `notes.md` write-up template.

**Tech Stack:** Python 3.11, `uv` (package mode, src layout, hatchling backend), `torch`, `transformers`, `numpy`, `pytest` (dev), `ruff` + `pyright` (already configured).

## Global Constraints

- No `accelerate` dependency — everything here is single-process, single-device.
- dtype auto-selection: fp16 on CUDA, fp32 on CPU. Never bf16 (Pascal/Turing — the 1060 and
  the T4 — both lack native bf16 support; that's Ampere+).
- Tokenizer pad-token fallback: `tokenizer.pad_token = tokenizer.eos_token` whenever the
  tokenizer has no pad token (true for Qwen2.5).
- Every stage decodes a fixed length: `min_new_tokens == max_new_tokens == output_len`, EOS
  early-stop disabled, so requests within a run are directly comparable.
- Percentiles computed via `numpy.percentile` (default linear interpolation).
- TTFT/ITL are captured with a synchronous custom `BaseStreamer` subclass
  (`TimestampStreamer`) whose `put()` just records `time.perf_counter()` — not
  `TextIteratorStreamer`, which threads for a reason (concurrent text iteration) that doesn't
  apply here.
- Synthetic prompts: random token ids sampled from `range(vocab_size)`, rejecting and
  resampling any id in `tokenizer.all_special_ids`, seeded for reproducibility. No text
  round-trip, no chat template.
- Numbered experiment directories (`01_naive_generate`, etc.) are not valid Python module
  names — their `run.py` invokes the shared driver via
  `subprocess.run([sys.executable, ...], check=True)`, never via `import`.
- Default model: `Qwen/Qwen2.5-0.5B-Instruct`, always overridable via `--model`.
- Pure-logic tests (aggregation math, prompt sampling, device/dtype resolution, CLI wiring)
  must run fast with no model download, using stub objects and `monkeypatch` where a real
  model/tokenizer would otherwise be needed. Tests that require a real model (does
  `generate_one()` actually produce correct tokens end-to-end) are explicitly **out of
  scope for this plan** — deferred to when stage 2 is designed, since that's when
  cross-stage output correctness first needs checking. Manual verification steps (documented
  in the tasks below, not part of the automated `pytest` suite) stand in for that here.
- **Commits are deferred**: do not commit after each task. The user wants one bundled commit
  once the whole baseline pipeline works end-to-end — that happens at the end of Task 7. If
  executing this plan via subagent-driven-development, skip that skill's usual per-task
  commit; use inline execution (`superpowers:executing-plans`) instead, or explicitly
  suppress per-task commits if using the subagent flow.

---

### Task 1: Package scaffold + `llm_serving.metrics`

**Files:**
- Modify: `pyproject.toml`
- Create: `src/llm_serving/__init__.py`
- Create: `src/llm_serving/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces: `RequestMetrics(request_id: int, prompt_tokens: int, output_tokens: int, ttft_s: float, total_latency_s: float, inter_token_latencies_s: list[float])`
- Produces: `StageSummary` dataclass (see field list in Step 3 below)
- Produces: `summarize(requests: list[RequestMetrics], *, stage: str, model: str, device: str, dtype: str, batch_size: int, input_len: int, output_len: int, wall_clock_s: float) -> StageSummary`
- Produces: `print_table(summary: StageSummary) -> None`
- Produces: `write_requests_csv(requests: list[RequestMetrics], path: Path) -> None`
- Produces: `write_summary_json(summary: StageSummary, path: Path) -> None`

- [ ] **Step 1: Switch `pyproject.toml` to package mode and add dependencies**

Add these two tables to `pyproject.toml` (anywhere after `[project]` is fine):

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/llm_serving"]
```

Then run:

```bash
uv add numpy
uv add --dev pytest
```

- [ ] **Step 2: Create the package skeleton and sync**

Create `src/llm_serving/__init__.py` (empty file is fine).

Run: `uv sync`
Expected: no errors; `.venv` now has `llm-serving-internals` installed editable.

Verify: `uv run python -c "import llm_serving; print(llm_serving.__file__)"`
Expected: prints a path under `src/llm_serving/__init__.py`.

- [ ] **Step 3: Write the failing tests for `metrics.py`**

Create `tests/test_metrics.py`:

```python
from pathlib import Path

import pytest

from llm_serving.metrics import RequestMetrics, summarize, write_requests_csv, write_summary_json


def make_request(request_id, total_latency_s, ttft_s, output_tokens=4):
    itls = [total_latency_s / output_tokens] * (output_tokens - 1)
    return RequestMetrics(
        request_id=request_id,
        prompt_tokens=16,
        output_tokens=output_tokens,
        ttft_s=ttft_s,
        total_latency_s=total_latency_s,
        inter_token_latencies_s=itls,
    )


def test_summarize_computes_latency_percentiles():
    requests = [make_request(i, total_latency_s=0.1 * (i + 1), ttft_s=0.01) for i in range(10)]

    summary = summarize(
        requests,
        stage="naive",
        model="test-model",
        device="cpu",
        dtype="float32",
        batch_size=1,
        input_len=16,
        output_len=4,
        wall_clock_s=1.0,
    )

    assert summary.latency_p50_ms == pytest.approx(550.0, rel=1e-3)
    assert summary.latency_min_ms == pytest.approx(100.0, rel=1e-3)
    assert summary.latency_max_ms == pytest.approx(1000.0, rel=1e-3)


def test_summarize_computes_throughput():
    requests = [make_request(i, total_latency_s=0.5, ttft_s=0.05) for i in range(5)]

    summary = summarize(
        requests,
        stage="naive",
        model="test-model",
        device="cpu",
        dtype="float32",
        batch_size=1,
        input_len=16,
        output_len=4,
        wall_clock_s=2.0,
    )

    assert summary.request_throughput == pytest.approx(2.5)
    assert summary.token_throughput == pytest.approx(10.0)  # 5 requests * 4 output tokens / 2s


def test_summarize_handles_single_request_with_no_inter_token_latencies():
    request = RequestMetrics(
        request_id=0,
        prompt_tokens=16,
        output_tokens=1,
        ttft_s=0.02,
        total_latency_s=0.02,
        inter_token_latencies_s=[],
    )

    summary = summarize(
        [request],
        stage="naive",
        model="test-model",
        device="cpu",
        dtype="float32",
        batch_size=1,
        input_len=16,
        output_len=1,
        wall_clock_s=0.02,
    )

    assert summary.itl_p50_ms == 0.0
    assert summary.itl_mean_ms == 0.0


def test_summarize_raises_on_empty_requests():
    with pytest.raises(ValueError):
        summarize(
            [],
            stage="naive",
            model="test-model",
            device="cpu",
            dtype="float32",
            batch_size=1,
            input_len=16,
            output_len=4,
            wall_clock_s=1.0,
        )


def test_write_requests_csv_and_summary_json_roundtrip(tmp_path: Path):
    requests = [make_request(0, total_latency_s=0.2, ttft_s=0.02)]
    summary = summarize(
        requests,
        stage="naive",
        model="test-model",
        device="cpu",
        dtype="float32",
        batch_size=1,
        input_len=16,
        output_len=4,
        wall_clock_s=0.2,
    )

    csv_path = tmp_path / "requests.csv"
    json_path = tmp_path / "summary.json"
    write_requests_csv(requests, csv_path)
    write_summary_json(summary, json_path)

    assert csv_path.exists()
    assert json_path.exists()
    assert "naive" in json_path.read_text()
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm_serving.metrics'`

- [ ] **Step 5: Implement `metrics.py`**

Create `src/llm_serving/metrics.py`:

```python
from __future__ import annotations

import csv
import json
import statistics
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


@dataclass
class RequestMetrics:
    request_id: int
    prompt_tokens: int
    output_tokens: int
    ttft_s: float
    total_latency_s: float
    inter_token_latencies_s: list[float]


@dataclass
class StageSummary:
    stage: str
    model: str
    device: str
    dtype: str
    batch_size: int
    input_len: int
    output_len: int
    num_requests: int
    wall_clock_s: float

    latency_p50_ms: float
    latency_p90_ms: float
    latency_p99_ms: float
    latency_mean_ms: float
    latency_min_ms: float
    latency_max_ms: float

    ttft_p50_ms: float
    ttft_p90_ms: float
    ttft_mean_ms: float

    itl_p50_ms: float
    itl_mean_ms: float

    request_throughput: float
    token_throughput: float
    timestamp: str


def summarize(
    requests: list[RequestMetrics],
    *,
    stage: str,
    model: str,
    device: str,
    dtype: str,
    batch_size: int,
    input_len: int,
    output_len: int,
    wall_clock_s: float,
) -> StageSummary:
    if not requests:
        raise ValueError("summarize() requires at least one RequestMetrics")

    latencies_ms = np.array([r.total_latency_s * 1000 for r in requests])
    ttfts_ms = np.array([r.ttft_s * 1000 for r in requests])
    all_itls_ms = [itl * 1000 for r in requests for itl in r.inter_token_latencies_s]

    total_output_tokens = sum(r.output_tokens for r in requests)

    return StageSummary(
        stage=stage,
        model=model,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        input_len=input_len,
        output_len=output_len,
        num_requests=len(requests),
        wall_clock_s=wall_clock_s,
        latency_p50_ms=round(float(np.percentile(latencies_ms, 50)), 3),
        latency_p90_ms=round(float(np.percentile(latencies_ms, 90)), 3),
        latency_p99_ms=round(float(np.percentile(latencies_ms, 99)), 3),
        latency_mean_ms=round(float(np.mean(latencies_ms)), 3),
        latency_min_ms=round(float(np.min(latencies_ms)), 3),
        latency_max_ms=round(float(np.max(latencies_ms)), 3),
        ttft_p50_ms=round(float(np.percentile(ttfts_ms, 50)), 3),
        ttft_p90_ms=round(float(np.percentile(ttfts_ms, 90)), 3),
        ttft_mean_ms=round(float(np.mean(ttfts_ms)), 3),
        itl_p50_ms=round(float(np.percentile(all_itls_ms, 50)), 3) if all_itls_ms else 0.0,
        itl_mean_ms=round(float(np.mean(all_itls_ms)), 3) if all_itls_ms else 0.0,
        request_throughput=round(len(requests) / wall_clock_s, 3),
        token_throughput=round(total_output_tokens / wall_clock_s, 3),
        timestamp=datetime.now(UTC).isoformat(),
    )


def print_table(summary: StageSummary) -> None:
    print(f"\nStage:    {summary.stage}")
    print(f"Model:    {summary.model}")
    print(f"Device:   {summary.device} ({summary.dtype})")
    print(f"Requests: {summary.num_requests}  Batch size: {summary.batch_size}")
    print(f"Input len: {summary.input_len}  Output len: {summary.output_len}\n")

    print(f"{'Latency (ms)':<14}{'p50':>10}{'p90':>10}{'p99':>10}{'mean':>10}{'min':>10}{'max':>10}")
    print(
        f"{'':<14}{summary.latency_p50_ms:>10.2f}{summary.latency_p90_ms:>10.2f}"
        f"{summary.latency_p99_ms:>10.2f}{summary.latency_mean_ms:>10.2f}"
        f"{summary.latency_min_ms:>10.2f}{summary.latency_max_ms:>10.2f}"
    )

    print(f"\n{'TTFT (ms)':<14}{'p50':>10}{'p90':>10}{'mean':>10}")
    print(f"{'':<14}{summary.ttft_p50_ms:>10.2f}{summary.ttft_p90_ms:>10.2f}{summary.ttft_mean_ms:>10.2f}")

    print(f"\n{'ITL (ms)':<14}{'p50':>10}{'mean':>10}")
    print(f"{'':<14}{summary.itl_p50_ms:>10.2f}{summary.itl_mean_ms:>10.2f}")

    print(f"\nThroughput: {summary.request_throughput:.2f} req/s, {summary.token_throughput:.2f} tok/s")


def write_requests_csv(requests: list[RequestMetrics], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["request_id", "prompt_tokens", "output_tokens", "ttft_s", "total_latency_s", "mean_itl_s"]
        )
        for r in requests:
            mean_itl = statistics.mean(r.inter_token_latencies_s) if r.inter_token_latencies_s else 0.0
            writer.writerow(
                [r.request_id, r.prompt_tokens, r.output_tokens, r.ttft_s, r.total_latency_s, mean_itl]
            )


def write_summary_json(summary: StageSummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(asdict(summary), f, indent=2)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: 5 passed

- [ ] **Step 7: Lint and type-check**

Run: `uv run ruff check src/ tests/` and `uv run pyright src/llm_serving/metrics.py`
Expected: no errors. Fix any before moving on.

---

### Task 2: `llm_serving.model`

**Files:**
- Create: `src/llm_serving/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: nothing from earlier tasks
- Produces: `resolve_device(device: str | None) -> str`
- Produces: `resolve_dtype(dtype: torch.dtype | None, device: str) -> torch.dtype`
- Produces: `load_model_and_tokenizer(model_name: str, device: str | None = None, dtype: torch.dtype | None = None) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]`

- [ ] **Step 1: Add `torch` and `transformers` dependencies**

```bash
uv add torch transformers
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_model.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm_serving.model'`

- [ ] **Step 4: Implement `model.py`**

Create `src/llm_serving/model.py`:

```python
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


def resolve_device(device: str | None) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dtype(dtype: torch.dtype | None, device: str) -> torch.dtype:
    if dtype is not None:
        return dtype
    return torch.float16 if device == "cuda" else torch.float32


def load_model_and_tokenizer(
    model_name: str,
    device: str | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    resolved_device = resolve_device(device)
    resolved_dtype = resolve_dtype(dtype, resolved_device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=resolved_dtype)
    model = model.to(resolved_device)
    model.eval()

    return model, tokenizer
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_model.py -v`
Expected: 5 passed

- [ ] **Step 6: Manual verification of `load_model_and_tokenizer` (real download, not automated)**

This laptop is CPU-only, so this step confirms the loading path works, not real
performance. Run:

```bash
uv run python -c "
from llm_serving.model import load_model_and_tokenizer
model, tokenizer = load_model_and_tokenizer('Qwen/Qwen2.5-0.5B-Instruct')
print(next(model.parameters()).device, next(model.parameters()).dtype)
print(tokenizer.pad_token)
"
```

Expected: downloads the model (first run only, ~1GB), then prints `cpu torch.float32` and a
non-`None` pad token. This confirms the auto device/dtype resolution and pad-token fallback
work against a real model, without needing this in the automated test suite.

- [ ] **Step 7: Lint and type-check**

Run: `uv run ruff check src/ tests/` and `uv run pyright src/llm_serving/model.py`
Expected: no errors.

---

### Task 3: `llm_serving.workload`

**Files:**
- Create: `src/llm_serving/workload.py`
- Test: `tests/test_workload.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (works against any object exposing `.vocab_size` and `.all_special_ids`, e.g. a real tokenizer or a stub)
- Produces: `make_prompt(tokenizer, input_len: int, seed: int) -> torch.Tensor` — shape `(1, input_len)`, dtype `torch.long`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_workload.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_workload.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm_serving.workload'`

- [ ] **Step 3: Implement `workload.py`**

Create `src/llm_serving/workload.py`:

```python
from __future__ import annotations

import random

import torch


def make_prompt(tokenizer, input_len: int, seed: int) -> torch.Tensor:
    rng = random.Random(seed)
    special_ids = set(tokenizer.all_special_ids)
    vocab_size = tokenizer.vocab_size

    token_ids: list[int] = []
    while len(token_ids) < input_len:
        candidate = rng.randrange(vocab_size)
        if candidate not in special_ids:
            token_ids.append(candidate)

    return torch.tensor([token_ids], dtype=torch.long)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_workload.py -v`
Expected: 4 passed

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff check src/ tests/` and `uv run pyright src/llm_serving/workload.py`
Expected: no errors.

---

### Task 4: `llm_serving.stages.base`

**Files:**
- Create: `src/llm_serving/stages/__init__.py`
- Create: `src/llm_serving/stages/base.py`
- Test: `tests/test_stages_base.py`

**Interfaces:**
- Consumes: `RequestMetrics` from Task 1 (`llm_serving.metrics`)
- Produces: `Stage` protocol with `generate_one(self, model, tokenizer, input_ids: torch.Tensor, output_len: int) -> RequestMetrics`
- Produces: `STAGES: dict[str, Stage]` (module-level registry)
- Produces: `register_stage(name: str, stage: Stage) -> None`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_stages_base.py`:

```python
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
    result = STAGES["fake_test_stage_2"].generate_one(None, None, torch.zeros(1, 8, dtype=torch.long), 4)
    assert result.output_tokens == 4
```

Note: these tests intentionally never call `STAGES.clear()` — `STAGES` is process-global and
shared with `naive.py`'s self-registration (Task 5) and the CLI tests (Task 6). Clearing it
would make those other tests order-dependent and flaky. Always register under a unique key
here instead.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_stages_base.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm_serving.stages'`

- [ ] **Step 3: Implement `stages/base.py`**

Create `src/llm_serving/stages/__init__.py` (empty file).

Create `src/llm_serving/stages/base.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_stages_base.py -v`
Expected: 2 passed

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff check src/ tests/` and `uv run pyright src/llm_serving/stages/base.py`
Expected: no errors.

---

### Task 5: `llm_serving.stages.naive`

**Files:**
- Create: `src/llm_serving/stages/naive.py`
- Test: `tests/test_stages_naive.py`

**Interfaces:**
- Consumes: `RequestMetrics` (Task 1), `Stage`/`register_stage`/`STAGES` (Task 4)
- Produces: `TimestampStreamer` (a `transformers.generation.streamers.BaseStreamer` subclass with a `.timestamps: list[float]` attribute)
- Produces: `NaiveStage` (implements `Stage`), registered into `STAGES["naive"]` as an import-time side effect

- [ ] **Step 1: Write the failing tests**

Create `tests/test_stages_naive.py`:

```python
from llm_serving.stages.base import STAGES
from llm_serving.stages.naive import TimestampStreamer

import llm_serving.stages.naive  # noqa: F401 - import triggers self-registration


def test_timestamp_streamer_records_one_timestamp_per_put_call():
    streamer = TimestampStreamer()
    streamer.put("token_a")
    streamer.put("token_b")
    streamer.put("token_c")

    assert len(streamer.timestamps) == 3


def test_timestamp_streamer_timestamps_are_increasing():
    streamer = TimestampStreamer()
    streamer.put("a")
    streamer.put("b")

    assert streamer.timestamps[1] >= streamer.timestamps[0]


def test_naive_stage_registers_itself():
    assert "naive" in STAGES
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_stages_naive.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm_serving.stages.naive'`

- [ ] **Step 3: Implement `stages/naive.py`**

Create `src/llm_serving/stages/naive.py`:

```python
from __future__ import annotations

import time

import torch
from transformers.generation.streamers import BaseStreamer

from llm_serving.metrics import RequestMetrics
from llm_serving.stages.base import register_stage


class TimestampStreamer(BaseStreamer):
    def __init__(self) -> None:
        self.timestamps: list[float] = []

    def put(self, value) -> None:
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
```

Note: `request_id` is always `0` here — the stage has no notion of which request number it
is. Task 6's `run_benchmark()` overwrites it with the real sequential id via
`dataclasses.replace()` after calling `generate_one()`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_stages_naive.py -v`
Expected: 3 passed

- [ ] **Step 5: Manual verification of `NaiveStage.generate_one` against a real model (not automated)**

```bash
uv run python -c "
from llm_serving.model import load_model_and_tokenizer
from llm_serving.workload import make_prompt
from llm_serving.stages.naive import NaiveStage

model, tokenizer = load_model_and_tokenizer('Qwen/Qwen2.5-0.5B-Instruct')
input_ids = make_prompt(tokenizer, input_len=16, seed=0)
result = NaiveStage().generate_one(model, tokenizer, input_ids, output_len=8)
print(result)
"
```

Expected: prints a `RequestMetrics` with `output_tokens=8`, `ttft_s > 0`,
`total_latency_s >= ttft_s`, and `len(inter_token_latencies_s) == 7`. This confirms the
streamer/timing wiring works against the real model without adding a slow, network-dependent
case to the automated suite (correctness-of-output tests are deferred, per Global
Constraints).

- [ ] **Step 6: Lint and type-check**

Run: `uv run ruff check src/ tests/` and `uv run pyright src/llm_serving/stages/naive.py`
Expected: no errors.

---

### Task 6: `llm_serving.cli`

**Files:**
- Create: `src/llm_serving/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `load_model_and_tokenizer` (Task 2), `make_prompt` (Task 3), `STAGES` (Task 4), `NaiveStage`/self-registration (Task 5), `summarize`/`print_table`/`write_requests_csv`/`write_summary_json`/`StageSummary` (Task 1)
- Produces: `build_arg_parser() -> argparse.ArgumentParser`
- Produces: `run_benchmark(args: argparse.Namespace) -> StageSummary`
- Produces: `main(argv: list[str] | None = None) -> None`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cli.py`:

```python
from pathlib import Path

import torch

from llm_serving import cli
from llm_serving.metrics import RequestMetrics
from llm_serving.stages.base import STAGES


class FakeModel:
    def parameters(self):
        return iter([torch.zeros(1, dtype=torch.float32, device="cpu")])


class FakeTokenizer:
    pad_token_id = 0


class FakeStage:
    def __init__(self):
        self.calls = 0

    def generate_one(self, model, tokenizer, input_ids, output_len):
        self.calls += 1
        return RequestMetrics(
            request_id=0,
            prompt_tokens=input_ids.shape[-1],
            output_tokens=output_len,
            ttft_s=0.01,
            total_latency_s=0.02,
            inter_token_latencies_s=[0.01],
        )


def test_build_arg_parser_defaults():
    parser = cli.build_arg_parser()
    args = parser.parse_args(["--stage", "naive", "--output-dir", "/tmp/out"])

    assert args.model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert args.input_len == 128
    assert args.output_len == 128
    assert args.num_requests == 10
    assert args.num_warmup == 2
    assert args.seed == 0


def test_run_benchmark_excludes_warmup_requests_from_summary(monkeypatch, tmp_path: Path):
    fake_stage = FakeStage()
    STAGES["fake_cli_stage"] = fake_stage
    monkeypatch.setattr(cli, "load_model_and_tokenizer", lambda model_name: (FakeModel(), FakeTokenizer()))

    parser = cli.build_arg_parser()
    args = parser.parse_args(
        [
            "--stage", "fake_cli_stage",
            "--output-dir", str(tmp_path),
            "--num-requests", "3",
            "--num-warmup", "2",
        ]
    )

    summary = cli.run_benchmark(args)

    assert fake_stage.calls == 5  # 2 warmup + 3 timed
    assert summary.num_requests == 3
    assert (tmp_path / "requests.csv").exists()
    assert (tmp_path / "summary.json").exists()


def test_run_benchmark_assigns_sequential_request_ids(monkeypatch, tmp_path: Path):
    fake_stage = FakeStage()
    STAGES["fake_cli_stage_2"] = fake_stage
    monkeypatch.setattr(cli, "load_model_and_tokenizer", lambda model_name: (FakeModel(), FakeTokenizer()))

    parser = cli.build_arg_parser()
    args = parser.parse_args(
        [
            "--stage", "fake_cli_stage_2",
            "--output-dir", str(tmp_path),
            "--num-requests", "3",
            "--num-warmup", "0",
        ]
    )

    summary = cli.run_benchmark(args)
    assert summary.num_requests == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm_serving.cli'`

- [ ] **Step 3: Implement `cli.py`**

Create `src/llm_serving/cli.py`:

```python
from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import llm_serving.stages.naive  # noqa: F401 - registers the "naive" stage
from llm_serving.metrics import StageSummary, print_table, summarize, write_requests_csv, write_summary_json
from llm_serving.model import load_model_and_tokenizer
from llm_serving.stages.base import STAGES
from llm_serving.workload import make_prompt


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified LLM serving benchmark driver")
    parser.add_argument("--stage", required=True, choices=sorted(STAGES.keys()))
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--num-requests", type=int, default=10)
    parser.add_argument("--num-warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def run_benchmark(args: argparse.Namespace) -> StageSummary:
    stage = STAGES[args.stage]
    model, tokenizer = load_model_and_tokenizer(args.model)
    device = str(next(model.parameters()).device)
    dtype = str(next(model.parameters()).dtype).removeprefix("torch.")

    for i in range(args.num_warmup):
        input_ids = make_prompt(tokenizer, args.input_len, seed=args.seed + i)
        stage.generate_one(model, tokenizer, input_ids, args.output_len)

    requests = []
    start = time.perf_counter()
    for i in range(args.num_requests):
        input_ids = make_prompt(tokenizer, args.input_len, seed=args.seed + args.num_warmup + i)
        result = stage.generate_one(model, tokenizer, input_ids, args.output_len)
        requests.append(replace(result, request_id=i))
    wall_clock_s = time.perf_counter() - start

    summary = summarize(
        requests,
        stage=args.stage,
        model=args.model,
        device=device,
        dtype=dtype,
        batch_size=1,
        input_len=args.input_len,
        output_len=args.output_len,
        wall_clock_s=wall_clock_s,
    )

    print_table(summary)
    write_requests_csv(requests, args.output_dir / "requests.csv")
    write_summary_json(summary, args.output_dir / "summary.json")

    return summary


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_benchmark(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: 3 passed

- [ ] **Step 5: Run the full test suite together**

Run: `uv run pytest -v`
Expected: all tests from Tasks 1–6 pass together (registry-sharing tests in particular —
confirms Task 4's note about not calling `STAGES.clear()` was correct).

- [ ] **Step 6: Lint and type-check**

Run: `uv run ruff check src/ tests/` and `uv run pyright src/llm_serving/cli.py`
Expected: no errors.

---

### Task 7: Wire up `benchmarks/`, verify end-to-end, and commit

**Files:**
- Create: `benchmarks/run.py`
- Create: `benchmarks/01_naive_generate/run.py`
- Create: `benchmarks/01_naive_generate/notes.md`

**Interfaces:**
- Consumes: `llm_serving.cli.main` (Task 6)

- [ ] **Step 1: Create the shared driver entrypoint**

Create `benchmarks/run.py`:

```python
#!/usr/bin/env python
from llm_serving.cli import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create the stage-1 experiment wrapper**

Create `benchmarks/01_naive_generate/run.py`:

```python
#!/usr/bin/env python
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent

if __name__ == "__main__":
    subprocess.run(
        [
            sys.executable,
            str(HERE.parent / "run.py"),
            "--stage", "naive",
            "--model", "Qwen/Qwen2.5-0.5B-Instruct",
            "--input-len", "128",
            "--output-len", "128",
            "--num-requests", "10",
            "--num-warmup", "2",
            "--seed", "0",
            "--output-dir", str(HERE / "results"),
        ],
        check=True,
    )
```

- [ ] **Step 3: Create the notes.md write-up template**

Create `benchmarks/01_naive_generate/notes.md`:

```markdown
# Experiment 01 — Naive `generate()` Baseline

## What changed
First benchmark stage: calling `model.generate()` as-is (no manual KV cache, no batching),
single request at a time. This is the reference point every later stage (manual KV cache,
static batching, continuous batching) gets compared against.

## Hypothesis
_Fill in before running on the 1060: what do you expect the naive baseline's latency and
throughput profile to look like, and why?_

## How tests were run

```bash
python benchmarks/01_naive_generate/run.py
```

Model: Qwen/Qwen2.5-0.5B-Instruct | Input length: 128 | Output length: 128 | Requests: 10 (+2 warmup)

## Results

_Paste the console table / summary.json contents here after running on the 1060._

## Why

_Fill in after seeing the results._

## Key takeaways

_Fill in after seeing the results._

## Hardware
- GPU: GTX 1060 3GB (Pascal, compute 6.1)
- CPU/RAM: _fill in_
```

- [ ] **Step 4: Manual end-to-end smoke test on this (CPU) laptop**

This confirms the whole pipeline runs correctly, using tiny lengths so it finishes quickly
on CPU — it is not a real benchmark run (that happens later on the 1060) and its output is
written to the scratchpad directory, not into `benchmarks/01_naive_generate/results/`:

```bash
uv run python benchmarks/run.py \
  --stage naive \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --input-len 16 \
  --output-len 16 \
  --num-requests 2 \
  --num-warmup 1 \
  --seed 0 \
  --output-dir /tmp/claude-1000/-home-simon-documents-llm-serving-internals/be37945b-3ae7-453d-b110-f3f0cae1c0cb/scratchpad/naive_smoke_test
```

Expected: prints the console table (latency/TTFT/ITL/throughput blocks) and writes
`requests.csv` + `summary.json` into that scratchpad directory. Inspect both files to confirm
they contain 2 rows / one summary with `num_requests: 2`.

Do **not** run `benchmarks/01_naive_generate/run.py` itself yet — its fixed args (128/128/10
requests) are tuned for a real 1060 run, and running it here would populate
`benchmarks/01_naive_generate/results/` with CPU-derived numbers that could later be
mistaken for real benchmark data. Leave that directory unpopulated; the user runs the
wrapper for real on the 1060 and fills in `notes.md` afterward.

- [ ] **Step 5: Run the full test suite, lint, and type-check one more time**

```bash
uv run pytest -v
uv run ruff check src/ tests/ benchmarks/
uv run pyright
```

Expected: all tests pass, no lint errors, no type errors.

- [ ] **Step 6: Commit everything**

This is the single bundled commit for the whole baseline — design doc, `CLAUDE.md` updates,
and all code from Tasks 1–7 (per the user's explicit preference to defer commits until the
baseline works end-to-end):

```bash
git add CLAUDE.md docs/superpowers/ pyproject.toml uv.lock src/ tests/ benchmarks/
git commit -m "$(cat <<'EOF'
Scaffold llm-serving-internals and add naive-generate baseline

Unified benchmarking harness (llm_serving.metrics/cli) shared by every
future stage, plus the naive model.generate() baseline as stage 1.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
git status --short
```

Expected: clean working tree after the commit (aside from anything intentionally left
untracked).
