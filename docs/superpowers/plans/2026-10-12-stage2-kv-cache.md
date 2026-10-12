# Stage 2: Manual KV Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a hand-written greedy decode loop (with and without KV cache) as benchmark stages
`kv_cache` / `no_cache`, make stage 1 pure greedy, record per-position ITL for every stage, and
set up experiment `02_kv_cache`.

**Architecture:** A shared `decode()` function in `llm_serving.stages.kv_cache` runs prefill plus
one forward pass per output token, toggling only whether the HF cache object is fed back in. Two
`ManualDecodeStage` instances register under `kv_cache` and `no_cache`. `llm_serving.metrics`
gains a per-position ITL aggregation that the shared driver writes for every stage.

**Tech Stack:** Python 3.11+, PyTorch 2.5.1 (cu121), transformers 5.15, numpy, pytest, uv, ruff,
pyright.

**Spec:** `docs/superpowers/specs/2026-08-13-stage2-kv-cache-design.md`

## Global Constraints

- Run everything through `uv run` (`uv run pytest`, `uv run ruff`, `uv run pyright`). Never pip.
- Do not touch the `torch` pin in `pyproject.toml` (`[tool.uv.sources]` / `pytorch-cu121`).
- Line length 100; ruff lint rules `E, F, I, UP`; pyright clean.
- `ruff format --check` only on files this plan creates. The existing tree has pre-existing
  format drift (paired `"--flag", value` arg lists, `model.py`, docs markdown); leave it alone
  and keep the paired-arg style when editing those files.
- Never import inside functions or methods; module-level imports only.
- Docstrings on classes and functions; inline comments only when necessary, max 2-3 lines.
- No emojis in code or files.
- Tiny test model: `yujiepan/qwen2-tiny-random`, loaded on CPU (fp32).
- Benchmark args for 01/02 are fixed: `Qwen/Qwen2.5-0.5B-Instruct`, `--input-len 128`,
  `--output-len 128`, `--num-requests 10`, `--num-warmup 2`, `--seed 0`.
- `StageSummary` / `summary.json` schema must not change.
- Git: commits only at the two milestones marked below, and only after the user approves each
  one. No Co-Authored-By or other attribution lines in commit messages.

## Review Focus

- `output_len == 1` (prefill only): `decode()` makes exactly one forward call, ITLs are empty,
  `itl_by_position()` returns `[]`, and `print_table()` skips the position block. Pinned in
  Tasks 2 and 3.
- `output_len <= 0`: `decode()` raises `ValueError` before touching the model. Pinned in Task 3.
- CUDA async timing: a GPU sync must precede every `perf_counter()` read. Not observable on CPU,
  so Task 3 pins it by counting `_sync` calls (`output_len + 1`).
- Ragged ITL lengths across requests (stage 4 will produce these): NaN padding and per-position
  `n` counts must be correct. Pinned in Task 2.
- Model `generation_config` defaults leaking into the greedy path (the repetition-penalty bug):
  pinned by Task 1's kwargs test and Task 4's repeated-token integration test.

---

### Task 1: Test restructure and pure greedy stage 1

**Files:**
- Move: `tests/test_*.py` -> `tests/unit/` (all 6 files)
- Create: `tests/conftest.py`
- Modify: `tests/unit/test_cli.py`, `tests/unit/test_stages_base.py` (use shared fixtures)
- Modify: `src/llm_serving/stages/naive.py`
- Test: `tests/unit/test_stages_naive.py`

**Interfaces:**
- Produces: `greedy_generate(model, tokenizer, input_ids: torch.Tensor, output_len: int,
  streamer: BaseStreamer | None = None) -> torch.Tensor` in `llm_serving.stages.naive`,
  returning only the new token ids, shape `[batch, output_len]`.
- Produces fixtures in `tests/conftest.py`: `fake_model`, `fake_tokenizer`, `fake_stage`
  (function-scoped), `tiny_model_and_tokenizer`, `tiny_model`, `tiny_tokenizer`
  (session-scoped). `FakeStage` has a `calls: int` counter and returns
  `output_len - 1` ITLs of 0.01s.

- [ ] **Step 1: Move the existing tests**

```bash
mkdir -p tests/unit tests/integration
git mv tests/test_metrics.py tests/test_model.py tests/test_workload.py \
  tests/test_stages_base.py tests/test_stages_naive.py tests/test_cli.py tests/unit/
```

- [ ] **Step 2: Create `tests/conftest.py`**

```python
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
```

- [ ] **Step 3: Rewrite `tests/unit/test_stages_base.py` to use the shared stub**

```python
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
```

- [ ] **Step 4: Rewrite `tests/unit/test_cli.py` to use the shared stubs**

```python
from pathlib import Path

from llm_serving import cli
from llm_serving.stages.base import STAGES


def test_build_arg_parser_defaults():
    parser = cli.build_arg_parser()
    args = parser.parse_args(["--stage", "naive", "--output-dir", "/tmp/out"])

    assert args.model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert args.input_len == 128
    assert args.output_len == 128
    assert args.num_requests == 10
    assert args.num_warmup == 2
    assert args.seed == 0


def test_run_benchmark_excludes_warmup_requests_from_summary(
    monkeypatch, tmp_path: Path, fake_model, fake_tokenizer, fake_stage
):
    STAGES["fake_cli_stage"] = fake_stage
    monkeypatch.setattr(
        cli, "load_model_and_tokenizer", lambda model_name: (fake_model, fake_tokenizer)
    )

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


def test_run_benchmark_assigns_sequential_request_ids(
    monkeypatch, tmp_path: Path, fake_model, fake_tokenizer, fake_stage
):
    STAGES["fake_cli_stage_2"] = fake_stage
    monkeypatch.setattr(
        cli, "load_model_and_tokenizer", lambda model_name: (fake_model, fake_tokenizer)
    )

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

- [ ] **Step 5: Run the moved suite to confirm the restructure broke nothing**

Run: `uv run pytest tests/unit -v`
Expected: all existing tests PASS (same count as before the move).

- [ ] **Step 6: Write failing tests for `greedy_generate` in `tests/unit/test_stages_naive.py`**

Replace the import block at the top and append the new tests (existing streamer tests stay):

```python
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
```

```python
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
```

- [ ] **Step 7: Run them to verify they fail**

Run: `uv run pytest tests/unit/test_stages_naive.py -v`
Expected: collection error, `ImportError: cannot import name 'greedy_generate'`.

- [ ] **Step 8: Implement `greedy_generate` and route `NaiveStage` through it**

In `src/llm_serving/stages/naive.py`, add after `TimestampStreamer`:

```python
def greedy_generate(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    output_len: int,
    streamer: BaseStreamer | None = None,
) -> torch.Tensor:
    """Pure greedy generate(), returning only the newly generated token ids.

    repetition_penalty is forced to 1.0 because generate() applies the model's
    generation_config penalty even with do_sample=False.
    """
    with torch.no_grad():
        output = model.generate(
            input_ids,
            min_new_tokens=output_len,
            max_new_tokens=output_len,
            do_sample=False,
            repetition_penalty=1.0,
            streamer=streamer,
            pad_token_id=tokenizer.pad_token_id,
        )
    return output[:, input_ids.shape[-1] :]
```

Replace the `with torch.no_grad(): model.generate(...)` block in `NaiveStage.generate_one` with:

```python
        start = time.perf_counter()
        greedy_generate(model, tokenizer, input_ids, output_len, streamer=streamer)
```

- [ ] **Step 9: Run the unit suite**

Run: `uv run pytest tests/unit -v`
Expected: all PASS.

---

### Task 2: Per-position ITL in `llm_serving.metrics` and the driver

**Files:**
- Modify: `src/llm_serving/metrics.py`
- Modify: `src/llm_serving/cli.py` (compute, print and write per-position ITL)
- Test: `tests/unit/test_metrics.py`, `tests/unit/test_cli.py`

**Interfaces:**
- Consumes: `RequestMetrics`, `StageSummary`, `summarize` (existing).
- Produces:
  - `@dataclass class PositionITL: position: int; itl_p50_ms: float; itl_mean_ms: float; n: int`
  - `itl_by_position(requests: list[RequestMetrics]) -> list[PositionITL]`
  - `write_itl_by_position_csv(rows: list[PositionITL], path: Path) -> None`, columns
    `position, itl_p50_ms, itl_mean_ms, n`
  - `display_positions(output_len: int) -> list[int]`
  - `print_table(summary: StageSummary, position_itls: list[PositionITL]) -> None`
    (new required second parameter)

- [ ] **Step 1: Write the failing tests**

Update the import in `tests/unit/test_metrics.py`:

```python
from llm_serving.metrics import (
    PositionITL,
    RequestMetrics,
    display_positions,
    itl_by_position,
    print_table,
    summarize,
    write_itl_by_position_csv,
    write_requests_csv,
    write_summary_json,
)
```

Append:

```python
def make_request_with_itls(request_id, itls):
    return RequestMetrics(
        request_id=request_id,
        prompt_tokens=16,
        output_tokens=len(itls) + 1,
        ttft_s=0.01,
        total_latency_s=0.01 + sum(itls),
        inter_token_latencies_s=itls,
    )


def make_summary(output_len):
    return summarize(
        [make_request(0, total_latency_s=0.2, ttft_s=0.02)],
        stage="naive",
        model="test-model",
        device="cpu",
        dtype="float32",
        batch_size=1,
        input_len=16,
        output_len=output_len,
        wall_clock_s=0.2,
    )


def test_itl_by_position_aggregates_across_requests():
    requests = [
        make_request_with_itls(0, [0.010, 0.020, 0.030]),
        make_request_with_itls(1, [0.030, 0.040, 0.050]),
    ]

    rows = itl_by_position(requests)

    assert [r.position for r in rows] == [1, 2, 3]
    assert [r.itl_p50_ms for r in rows] == pytest.approx([20.0, 30.0, 40.0])
    assert [r.itl_mean_ms for r in rows] == pytest.approx([20.0, 30.0, 40.0])
    assert [r.n for r in rows] == [2, 2, 2]


def test_itl_by_position_handles_ragged_lengths():
    requests = [
        make_request_with_itls(0, [0.010, 0.020, 0.030]),
        make_request_with_itls(1, [0.030, 0.040, 0.050]),
        make_request_with_itls(2, [0.020]),
    ]

    rows = itl_by_position(requests)

    assert [r.n for r in rows] == [3, 2, 2]
    assert rows[0].itl_p50_ms == pytest.approx(20.0)
    assert rows[0].itl_mean_ms == pytest.approx(20.0)
    assert rows[1].itl_mean_ms == pytest.approx(30.0)


def test_itl_by_position_returns_empty_without_itls():
    assert itl_by_position([make_request_with_itls(0, [])]) == []


def test_write_itl_by_position_csv(tmp_path: Path):
    rows = [PositionITL(position=1, itl_p50_ms=20.0, itl_mean_ms=21.0, n=2)]
    path = tmp_path / "nested" / "itl_by_position.csv"

    write_itl_by_position_csv(rows, path)

    lines = path.read_text().splitlines()
    assert lines[0] == "position,itl_p50_ms,itl_mean_ms,n"
    assert lines[1] == "1,20.0,21.0,2"


def test_display_positions():
    assert display_positions(128) == [1, 32, 64, 127]
    assert display_positions(2) == [1]
    assert display_positions(1) == []


def test_print_table_shows_itl_by_position(capsys):
    rows = [PositionITL(position=p, itl_p50_ms=25.0, itl_mean_ms=25.0, n=1) for p in range(1, 4)]

    print_table(make_summary(output_len=4), rows)

    out = capsys.readouterr().out
    assert "ITL by pos" in out
    assert "25.00" in out


def test_print_table_skips_itl_by_position_without_rows(capsys):
    print_table(make_summary(output_len=1), [])

    assert "ITL by pos" not in capsys.readouterr().out
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/unit/test_metrics.py -v`
Expected: collection error, `ImportError: cannot import name 'PositionITL'`.

- [ ] **Step 3: Implement**

In `src/llm_serving/metrics.py`, add after `StageSummary`:

```python
@dataclass
class PositionITL:
    """ITL at one decode position, aggregated across requests.

    position k is the gap between output token k and k+1 (1-based).
    """

    position: int
    itl_p50_ms: float
    itl_mean_ms: float
    n: int
```

Add after `summarize()`:

```python
def itl_by_position(requests: list[RequestMetrics]) -> list[PositionITL]:
    """Aggregate ITL per decode position; NaN padding keeps ragged lengths from skewing it."""
    lengths = [len(r.inter_token_latencies_s) for r in requests]
    max_len = max(lengths, default=0)
    if max_len == 0:
        return []

    itls_ms = np.full((len(requests), max_len), np.nan)
    for row, (request, length) in enumerate(zip(requests, lengths)):
        itls_ms[row, :length] = np.asarray(request.inter_token_latencies_s) * 1000

    p50 = np.nanmedian(itls_ms, axis=0)
    mean = np.nanmean(itls_ms, axis=0)
    counts = np.sum(~np.isnan(itls_ms), axis=0)

    return [
        PositionITL(
            position=k + 1,
            itl_p50_ms=round(float(p50[k]), 3),
            itl_mean_ms=round(float(mean[k]), 3),
            n=int(counts[k]),
        )
        for k in range(max_len)
    ]


def display_positions(output_len: int) -> list[int]:
    """Positions shown in the console ITL row: first, quarter, half, last."""
    candidates = [1, output_len // 4, output_len // 2, output_len - 1]
    return sorted({p for p in candidates if 1 <= p <= output_len - 1})
```

Change `print_table`'s signature and insert the position block between the ITL block and the
throughput line:

```python
def print_table(summary: StageSummary, position_itls: list[PositionITL]) -> None:
```

```python
    print(f"\n{'ITL (ms)':<14}{'p50':>10}{'mean':>10}")
    print(f"{'':<14}{summary.itl_p50_ms:>10.2f}{summary.itl_mean_ms:>10.2f}")

    by_position = {row.position: row for row in position_itls}
    positions = [p for p in display_positions(summary.output_len) if p in by_position]
    if positions:
        headers = "".join(f"{p:>10}" for p in positions)
        values = "".join(f"{by_position[p].itl_p50_ms:>10.2f}" for p in positions)
        print(f"\n{'ITL by pos':<14}{headers}")
        print(f"{'(p50 ms)':<14}{values}")
```

Add after `write_requests_csv()`:

```python
def write_itl_by_position_csv(rows: list[PositionITL], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["position", "itl_p50_ms", "itl_mean_ms", "n"])
        writer.writerows([r.position, r.itl_p50_ms, r.itl_mean_ms, r.n] for r in rows)
```

- [ ] **Step 4: Run the metrics tests**

Run: `uv run pytest tests/unit/test_metrics.py -v`
Expected: all PASS.

- [ ] **Step 5: Write the failing driver test**

Append to `tests/unit/test_cli.py` (keep the paired-arg style used in this file):

```python
def test_run_benchmark_writes_itl_by_position_csv(
    monkeypatch, tmp_path: Path, fake_model, fake_tokenizer, fake_stage
):
    STAGES["fake_cli_stage_3"] = fake_stage
    monkeypatch.setattr(
        cli, "load_model_and_tokenizer", lambda model_name: (fake_model, fake_tokenizer)
    )

    parser = cli.build_arg_parser()
    args = parser.parse_args(
        [
            "--stage", "fake_cli_stage_3",
            "--output-dir", str(tmp_path),
            "--output-len", "4",
            "--num-requests", "2",
            "--num-warmup", "0",
        ]
    )

    cli.run_benchmark(args)

    lines = (tmp_path / "itl_by_position.csv").read_text().splitlines()
    assert lines[0] == "position,itl_p50_ms,itl_mean_ms,n"
    assert len(lines) == 4  # header + positions 1..3
```

- [ ] **Step 6: Run the driver tests to verify they fail**

Run: `uv run pytest tests/unit/test_cli.py -v`
Expected: FAIL with `TypeError: print_table() missing 1 required positional argument:
'position_itls'` (every `run_benchmark` test).

- [ ] **Step 7: Wire it into `cli.py`**

Replace the metrics import:

```python
from llm_serving.metrics import (
    StageSummary,
    itl_by_position,
    print_table,
    summarize,
    write_itl_by_position_csv,
    write_requests_csv,
    write_summary_json,
)
```

Replace the output block at the end of `run_benchmark`:

```python
    position_itls = itl_by_position(requests)

    print_table(summary, position_itls)
    write_requests_csv(requests, args.output_dir / "requests.csv")
    write_summary_json(summary, args.output_dir / "summary.json")
    write_itl_by_position_csv(position_itls, args.output_dir / "itl_by_position.csv")
```

- [ ] **Step 8: Run the unit suite**

Run: `uv run pytest tests/unit -v`
Expected: all PASS.

---

### Task 3: `kv_cache` / `no_cache` stages

**Files:**
- Create: `src/llm_serving/stages/kv_cache.py`
- Test: `tests/unit/test_stages_kv_cache.py`

**Interfaces:**
- Consumes: `RequestMetrics` (`llm_serving.metrics`), `register_stage` (`llm_serving.stages.base`).
- Produces:
  - `decode(model, input_ids: torch.Tensor, output_len: int, use_cache: bool)
    -> tuple[torch.Tensor, float, list[float]]` returning `(tokens [batch, output_len],
    start, timestamps)` with `len(timestamps) == output_len`.
  - `class ManualDecodeStage(use_cache: bool)` with `.use_cache` and
    `.generate_one(model, tokenizer, input_ids, output_len) -> RequestMetrics`.
  - Registry entries `"kv_cache"` (`use_cache=True`) and `"no_cache"` (`use_cache=False`).
  - Module-level `_sync(device: torch.device) -> None`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_stages_kv_cache.py`:

```python
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
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/unit/test_stages_kv_cache.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'llm_serving.stages.kv_cache'`.

- [ ] **Step 3: Implement `src/llm_serving/stages/kv_cache.py`**

```python
from __future__ import annotations

import time

import torch

from llm_serving.metrics import RequestMetrics
from llm_serving.stages.base import register_stage


def _sync(device: torch.device) -> None:
    """Block until queued GPU work finishes, so perf_counter() reads completion time."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def decode(
    model, input_ids: torch.Tensor, output_len: int, use_cache: bool
) -> tuple[torch.Tensor, float, list[float]]:
    """Greedy-decode output_len tokens with a hand-written loop.

    With use_cache, each step feeds only the new token plus the HF cache object; without it,
    each step re-feeds the full sequence. Returns (tokens, start, per-token timestamps).
    """
    if output_len < 1:
        raise ValueError(f"output_len must be >= 1, got {output_len}")

    device = input_ids.device
    _sync(device)
    start = time.perf_counter()

    with torch.no_grad():
        outputs = model(input_ids, use_cache=use_cache)
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        _sync(device)
        timestamps = [time.perf_counter()]

        cache = outputs.past_key_values if use_cache else None
        sequence = torch.cat([input_ids, next_token], dim=-1)
        generated = [next_token]

        for _ in range(output_len - 1):
            if use_cache:
                outputs = model(next_token, past_key_values=cache, use_cache=True)
                cache = outputs.past_key_values
            else:
                outputs = model(sequence, use_cache=False)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            _sync(device)
            timestamps.append(time.perf_counter())

            sequence = torch.cat([sequence, next_token], dim=-1)
            generated.append(next_token)

    return torch.cat(generated, dim=-1), start, timestamps


class ManualDecodeStage:
    """Stage 2: hand-written greedy decode loop, with or without the KV cache."""

    def __init__(self, use_cache: bool) -> None:
        self.use_cache = use_cache

    def generate_one(
        self, model, tokenizer, input_ids: torch.Tensor, output_len: int
    ) -> RequestMetrics:
        input_ids = input_ids.to(model.device)
        _, start, timestamps = decode(model, input_ids, output_len, self.use_cache)

        return RequestMetrics(
            request_id=0,
            prompt_tokens=input_ids.shape[-1],
            output_tokens=len(timestamps),
            ttft_s=timestamps[0] - start,
            total_latency_s=timestamps[-1] - start,
            inter_token_latencies_s=[t2 - t1 for t1, t2 in zip(timestamps, timestamps[1:])],
        )


register_stage("kv_cache", ManualDecodeStage(use_cache=True))
register_stage("no_cache", ManualDecodeStage(use_cache=False))
```

- [ ] **Step 4: Run the stage tests**

Run: `uv run pytest tests/unit/test_stages_kv_cache.py -v`
Expected: all PASS.

---

### Task 4: Integration correctness test

**Files:**
- Create: `tests/integration/test_kv_cache_correctness.py`

**Interfaces:**
- Consumes: `tiny_model`, `tiny_tokenizer` fixtures (Task 1); `greedy_generate` (Task 1);
  `decode`, `ManualDecodeStage` (Task 3); `make_prompt` (`llm_serving.workload`).

- [ ] **Step 1: Write the test**

```python
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
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/integration -v`
Expected: 5 PASS. The model is already in the local HF cache (`~/.cache/huggingface/hub/`); on a
fresh machine the first run downloads ~16MB.

If `test_reference_output_contains_a_repeated_token` fails, the design-time probe (seed 0,
16 prompt tokens, 12 output tokens produced `7539` three times) no longer reproduces; try
`SEED` values 1-5 until greedy output has a repeat, and note the change in the report.

- [ ] **Step 3: Sanity-check that the test would have caught the original bug**

Temporarily change `repetition_penalty=1.0` to `repetition_penalty=1.05` in `greedy_generate`,
run `uv run pytest tests/integration -v`, confirm both `test_decode_matches_naive_greedy_generate`
cases FAIL, then revert the change and confirm they PASS again.

---

### Task 5: Register the new stages with the driver

**Files:**
- Modify: `src/llm_serving/cli.py`
- Test: `tests/unit/test_cli.py`

**Interfaces:**
- Consumes: `llm_serving.stages.kv_cache` registration (Task 3).

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_cli.py`:

```python
def test_arg_parser_accepts_manual_decode_stages():
    parser = cli.build_arg_parser()

    for stage in ("kv_cache", "no_cache"):
        args = parser.parse_args(["--stage", stage, "--output-dir", "/tmp/out"])
        assert args.stage == stage
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/unit/test_cli.py::test_arg_parser_accepts_manual_decode_stages -v`
Expected: FAIL, `SystemExit: 2` from argparse (`invalid choice: 'kv_cache'`), unless another
test module imported `kv_cache` first; run the single test as shown to avoid that.

- [ ] **Step 3: Add the registration import**

In `src/llm_serving/cli.py`, above the existing `naive` import:

```python
import llm_serving.stages.kv_cache  # noqa: F401 - registers "kv_cache" and "no_cache"
```

- [ ] **Step 4: Run the full suite plus lint and type checks**

Run: `uv run pytest -v`
Expected: all unit and integration tests PASS.

Run:
`uv run ruff check . && uv run ruff format --check src/llm_serving/stages/kv_cache.py tests/conftest.py tests/unit/test_stages_kv_cache.py tests/integration/test_kv_cache_correctness.py && uv run pyright`
Expected: no errors. Fix anything reported before moving on.

---

### Task 6: Experiment 02 scaffolding and docs

**Files:**
- Create: `benchmarks/02_kv_cache/run.py`
- Create: `benchmarks/02_kv_cache/notes.md`
- Modify: `CLAUDE.md` (project structure tree)
- Modify: `README.md` (status, stages table, project structure, usage)

- [ ] **Step 1: Create `benchmarks/02_kv_cache/run.py`**

```python
#!/usr/bin/env python
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
STAGES_TO_RUN = ("naive", "kv_cache", "no_cache")

if __name__ == "__main__":
    for stage in STAGES_TO_RUN:
        subprocess.run(
            [
                sys.executable,
                str(HERE.parent / "run.py"),
                "--stage", stage,
                "--model", "Qwen/Qwen2.5-0.5B-Instruct",
                "--input-len", "128",
                "--output-len", "128",
                "--num-requests", "10",
                "--num-warmup", "2",
                "--seed", "0",
                "--output-dir", str(HERE / "results" / stage),
            ],
            check=True,
        )
```

- [ ] **Step 2: Create `benchmarks/02_kv_cache/notes.md` with the pre-run sections**

````markdown
# Experiment 02 — Manual KV Cache

## What changed
Replaced `model.generate()` with a hand-written greedy decode loop (`llm_serving.stages.kv_cache`).
The loop runs one prefill pass over the prompt, then one forward pass per output token, and is
registered as two stages that differ only in whether the KV cache is reused:

- `kv_cache` feeds just the newly generated token plus the cache object returned by the previous
  step, so each step attends over cached keys/values instead of recomputing them.
- `no_cache` re-feeds the full sequence (prompt + everything generated so far) every step, with
  caching disabled, so each step recomputes attention for every prior position.

`generate()` already uses a KV cache by default, so `naive` vs. `kv_cache` compares HF's cached loop
against ours; `no_cache` is the control that shows what the cache itself buys. The loop exists
mainly because stages 3-4 need it: batching and continuous batching require stepping several
sequences through a loop we control.

Two measurement fixes landed alongside this stage:

- **Pure greedy stage 1.** `generate()` applies the model's `repetition_penalty` (1.1 for
  Qwen2.5-0.5B-Instruct) even with `do_sample=False`. `naive` now passes `repetition_penalty=1.0`,
  so all three stages do identical logits work and produce identical tokens (checked by an
  integration test). Experiment 01 was rerun with this fix.
- **Explicit GPU sync.** CUDA kernel launches are asynchronous, so the manual loop calls
  `torch.cuda.synchronize()` before every timestamp; otherwise it would time when work was queued,
  not when it finished. `generate()` already syncs every step through its stopping-criteria check.

A new `itl_by_position.csv` (ITL p50/mean per decode position, across requests) is written for
every stage, since the no-cache effect is in the shape of ITL over a request, not its mean.

## Hypothesis

1. **`kv_cache` ≈ `naive`** within a few percent on throughput and ITL. Same mechanism and the same
   logits work; any gap reflects `generate()`'s remaining per-step bookkeeping (logits-processor
   list, stopping-criteria checks, streamer calls) vs. the bare loop.
2. **TTFT is the same across all three stages**: each does one identical 128-token prefill pass.
3. **`no_cache` ITL climbs with position**, starting near the 128-token prefill cost (~55ms in
   experiment 01) rather than ~25ms, because every step is effectively a fresh prefill over
   128 + k tokens. The climb is likely sublinear at first: a 0.5B model at these sequence lengths is
   probably still kernel-launch/overhead-bound rather than attention-FLOP-bound.

## How tests were run

```bash
python benchmarks/02_kv_cache/run.py
```

Runs `naive`, `kv_cache` and `no_cache` back to back in one session, each writing to
`results/<stage>/`.

Model: Qwen/Qwen2.5-0.5B-Instruct | Input length: 128 | Output length: 128 | Requests: 10 (+2 warmup)
| Greedy decoding, repetition penalty disabled

## Results

## Why

## Key takeaways

## Hardware
- GPU: GTX 1060 3GB (Pascal, compute 6.1)
- CPU: Intel Core i7-6700 (8 threads) @ 4.00 GHz
- RAM: 15.54 GiB, DDR4 @ 2133 MT/s
- Driver: NVIDIA 535.309.01, CUDA 12.2
````

- [ ] **Step 3: Update `CLAUDE.md`'s project-structure tree**

Replace the fenced tree under "Project structure & benchmarking convention" with:

```
llm-serving-internals/
├── src/
│   └── llm_serving/
│       ├── model.py         # load_model_and_tokenizer(): device/dtype selection, pad-token fixup
│       ├── workload.py      # synthetic fixed-length random-token prompt generation
│       ├── metrics.py       # shared benchmarking "utils": RequestMetrics, StageSummary, PositionITL, percentiles, CSV/JSON/console output
│       ├── cli.py           # benchmark driver: args, warmup, timed requests, output writers
│       └── stages/
│           ├── base.py      # Stage protocol: generate_one(model, tok, input_ids, output_len) -> RequestMetrics
│           ├── naive.py     # stage 1 "naive": greedy_generate() (pure greedy generate(), repetition penalty off)
│           └── kv_cache.py  # stage 2: decode() hand-written loop, registered as "kv_cache" and "no_cache"
├── benchmarks/
│   ├── run.py                # the one real driver: --stage, --model, --input-len, --output-len, --num-requests, --output-dir
│   ├── 01_naive_generate/
│   │   ├── run.py            # thin wrapper: subprocess-invokes benchmarks/run.py with this experiment's fixed args
│   │   ├── notes.md          # write-up: What changed / Hypothesis / How tests were run / Results / Why / Takeaways / Hardware
│   │   └── results/          # requests.csv, summary.json, itl_by_position.csv, committed
│   ├── 02_kv_cache/          # runs naive, kv_cache, no_cache; results/<stage>/
│   └── 0N_.../                # one dir per stage, added as each stage lands
└── tests/
    ├── conftest.py           # shared fakes (fake_model/tokenizer/stage) + session-scoped tiny-model fixtures
    ├── unit/                 # fast, CPU-only, no model download
    └── integration/          # downloads yujiepan/qwen2-tiny-random once; decode-vs-generate() correctness
```

- [ ] **Step 4: Update `README.md`**

Status line:

```markdown
> Work in progress. Stage 1 (baseline) is complete; stage 2 (manual KV cache) is implemented and
> awaiting its benchmark run.
```

Stages table row 2:

```markdown
| 2 | Manual KV cache (hand-written decode loop, with a no-cache control) | GTX 1060 | In progress |
```

Project structure block:

```
src/llm_serving/
  model.py        model/tokenizer loading, device and dtype selection
  workload.py     synthetic fixed-length prompt generation
  metrics.py      RequestMetrics, StageSummary, per-position ITL, CSV/JSON/console output
  cli.py          benchmark driver arguments
  stages/
    base.py       Stage protocol
    naive.py      stage 1
    kv_cache.py   stage 2 (kv_cache and no_cache)
benchmarks/
  run.py                  the single benchmark driver
  01_naive_generate/      run.py (fixed args), notes.md, results/
  02_kv_cache/            run.py (three stages), notes.md, results/<stage>/
tests/
  unit/                   CPU-only, no model download
  integration/            downloads a ~16MB random-weight Qwen2 once, checks decode correctness
```

Replace the "Run the tests" snippet:

````markdown
Run the tests:

```bash
uv run pytest tests/unit         # fast, no download
uv run pytest tests/integration  # downloads a tiny test model on first run
```
````

In the Methodology bullet "One benchmarking path for every stage", change "output as CSV, JSON
and a console summary" to "output as CSV (per request and per decode position), JSON and a
console summary".

- [ ] **Step 5: Final verification**

Run: `uv run pytest -v && uv run ruff check . && uv run pyright`
Expected: all PASS, no lint/type errors. (`benchmarks/02_kv_cache/run.py` keeps the paired-arg
style of `01`, so it is deliberately not format-checked.)

- [ ] **Step 6: Milestone commit (ask the user first)**

Show the user `git status` and the proposed message, and commit only after approval:

```bash
git add -A src tests benchmarks/02_kv_cache CLAUDE.md README.md docs/superpowers
git commit -m "feat: stage 2 manual decode loop (kv_cache/no_cache), per-position ITL, pure greedy stage 1"
```

---

### Task 7: Benchmark runs on the 1060 and write-ups (user-driven)

Claude has no access to the desktop. This task is a handoff, then write-up work once results
come back.

- [ ] **Step 1: User runs both experiments on the GTX 1060 desktop**

```bash
uv sync
uv run pytest tests/unit
uv run python benchmarks/01_naive_generate/run.py
uv run python benchmarks/02_kv_cache/run.py
```

The user brings back the four console outputs and the `results/` directories (via git from the
desktop, or by copying them).

- [ ] **Step 2: Update experiment 01's notes for the rerun**

In `benchmarks/01_naive_generate/notes.md`:
- "How tests were run": append `| Greedy decoding, repetition penalty disabled` to the
  model/length line, plus one sentence that the original run applied the model's default
  `repetition_penalty=1.1`, and was rerun pure greedy so later stages compare like for like.
- "Results": replace the console block with the new output.
- "Why" / "Key takeaways": update the numbers (TTFT, ITL, tok/s, the ~40GB/s / ~20% of peak
  bandwidth estimate); note if the penalty's removal moved ITL measurably.

- [ ] **Step 3: Fill experiment 02's Results / Why / Key takeaways**

- Results: the three console blocks, plus the `no_cache` per-position ITL at positions 1, 32,
  64, 127 vs. `kv_cache` (from the console rows; cite `itl_by_position.csv` for the full curve).
- Why: check each of the three hypotheses against the data, explicitly saying which held and
  which did not. For hypothesis 3, report the actual growth shape (linear, sublinear, flat-then-
  rising) from the CSV.
- Key takeaways: what the cache buys at this sequence length, and what that implies for stage 3
  (batching), including that cache memory will be the constraint on a 3GB card.

- [ ] **Step 4: Update `README.md` results**

Refresh the stage 1 table with the rerun numbers, add a stage 2 results subsection
(three-row comparison table plus one paragraph), and set stage 2's status to `Done`.

- [ ] **Step 5: Milestone commit (ask the user first)**

```bash
git add benchmarks/01_naive_generate benchmarks/02_kv_cache README.md
git commit -m "feat: experiment 02 results, rerun experiment 01 as pure greedy"
```
