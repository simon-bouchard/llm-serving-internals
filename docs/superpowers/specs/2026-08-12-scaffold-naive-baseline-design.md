# Project scaffold + naive-generate baseline: design

Date: 2026-08-12

## Goal

Scaffold `llm-serving-internals` and implement the first benchmark stage (naive
`model.generate()` baseline), with a benchmarking harness designed up front to be reused,
unchanged, by the later stages (manual KV cache, static batching, continuous batching, and
the vLLM/T4 comparison).

This is a learning project: build understanding of LLM inference serving mechanics by
hand-implementing the serving mechanisms rather than relying on a framework's built-in
implementation, while treating hardware constraints (3GB VRAM, Pascal/no-vLLM 1060 vs.
Turing/vLLM-capable T4) as part of the experiment rather than something to engineer around.

## Scope

In scope for this design/implementation pass:
- Project directory structure and package layout
- `uv` package-mode migration (src layout, editable install)
- The shared benchmarking harness (`llm_serving.metrics`) and its output formats
- Supporting modules: `llm_serving.model` (loading), `llm_serving.workload` (synthetic
  prompts), `llm_serving.stages.base` (the stage interface/registry)
- Stage 1: `llm_serving.stages.naive`, `benchmarks/run.py`, `benchmarks/01_naive_generate/`
- Unit tests for the aggregation math in `metrics.py`

Out of scope, deferred to their own future design/implementation passes:
- Stages 2–4 (manual KV cache, static batching, continuous batching) — only the `Stage`
  interface they'll implement is designed now
- The vLLM/T4 leg (stage 5) — only the high-level approach (offline API, not HTTP server) is
  decided now; exact integration lands when that stage is built
- Model selection sweep (which of Qwen2.5-0.5B/1.5B etc. actually shows meaningful
  stage-to-stage differences) — this stays an open question in `CLAUDE.md`

## Hardware / execution model

Three machines, Claude has direct access only to the first:
- **WSL2 laptop** (this session) — CPU-only. Dev environment; runs the scaffold for
  correctness smoke-testing (small input/output lengths), not for real benchmark numbers.
- **GTX 1060 desktop** (Pascal, compute 6.1, 3GB VRAM, no vLLM support) — where stages 1–4
  actually get benchmarked. No remote access; scripts are written here and run there
  manually by the user.
- **Kaggle T4** (Turing, compute 7.5, vLLM-capable) — where stage 5 (vLLM) runs. No remote
  access; same manual hand-off model.

## Directory structure

```
llm-serving-internals/
├── src/
│   └── llm_serving/
│       ├── __init__.py
│       ├── model.py          # load_model_and_tokenizer()
│       ├── workload.py       # synthetic fixed-length prompt generation
│       ├── metrics.py        # RequestMetrics, StageSummary, aggregation, output writers
│       └── stages/
│           ├── __init__.py
│           ├── base.py       # Stage protocol + STAGES registry
│           └── naive.py      # stage 1
├── benchmarks/
│   ├── run.py                 # the one real driver (argparse, dispatch, output)
│   ├── 01_naive_generate/
│   │   ├── run.py             # thin subprocess wrapper, fixed args for this experiment
│   │   ├── notes.md           # write-up template, filled in after real 1060 runs
│   │   └── results/           # requests.csv + summary.json, committed
│   └── 0N_.../                 # one dir per later stage, added as each stage lands
├── tests/
│   └── test_metrics.py        # aggregation math, CPU-only, no model download
├── docs/superpowers/specs/     # this file and future design docs
├── pyproject.toml
├── CLAUDE.md
└── .envrc
```

## Benchmarking convention

**One benchmarking system for every stage**, not one per stage and not one per machine.
This was a deliberate correction from the `cv-inference-triton` project, which ended up with
two incompatible benchmarking paths (`perf_analyzer` CSVs vs. a custom `load_test.py`
client) that didn't share a schema and couldn't be compared directly.

- `benchmarks/run.py` is the single implementation: it loads the model, generates the
  synthetic workload, dispatches to the requested stage's `generate_one()`, collects
  `RequestMetrics`, and writes all three output artifacts.
- Each `benchmarks/0N_experiment_name/run.py` is a thin wrapper that shells out to
  `benchmarks/run.py` (via `subprocess.run([sys.executable, ...], check=True)`) with that
  experiment's fixed CLI args. Numbered directories aren't valid Python module names, so
  this can't be a plain import — subprocess is the equivalent of `cv-inference-triton`'s
  `run_pipeline.sh` calling `load_test.py`, just in Python instead of bash.
- No logic is duplicated between the shared driver and the per-experiment wrappers.

**Benchmarking tool choice**: every stage is measured in-process with no HTTP layer, so
measurement methodology is never a confound between stages. Stages 1–4 call `generate()` or
a hand-rolled decode loop directly, timed by `llm_serving.metrics`. Stage 5 (vLLM) will use
vLLM's offline/in-process API (`vllm.LLM(...).generate(...)`), not its OpenAI-compatible HTTP
server — going through the server plus `benchmark_serving.py` would add network/serialization
overhead the other stages never pay, so a throughput difference would partly reflect the
measurement path rather than the scheduler. vLLM's `RequestOutput` carries native per-request
timing that will get mapped into the same `RequestMetrics`/`StageSummary` schema as every
other stage.

Hardware (1060 vs. T4) remains an intentional, unavoidable confound between the manual
stages and the vLLM leg — the write-up must call this out explicitly rather than letting it
hide inside a single headline "vLLM is N× faster" number. A possible future extension: also
run the hand-rolled continuous-batching stage on the T4, isolating mechanism (hand-rolled vs.
vLLM) from hardware, separate from the 1060's hardware-constrained baseline. Not committed to
for this pass.

## Package layout / tooling

`pyproject.toml` moves from `uv init`'s default app mode to package mode: a `[build-system]`
(hatchling) plus `[tool.hatch.build.targets.wheel] packages = ["src/llm_serving"]`, so
`uv sync` installs `llm_serving` editable into `.venv` and any script — including the
numbered experiment wrappers and `benchmarks/run.py` — can `import llm_serving` regardless of
its own location.

Dependencies: `torch`, `transformers`, `numpy` (percentiles). Dev dependencies (already
present): `ruff`, `pyright`, plus `pytest` added for `tests/test_metrics.py`. No `accelerate`
— everything here is single-process, single-device; no model sharding or distributed setup is
needed at 0.5–1.7B params.

## `llm_serving.model`

```python
def load_model_and_tokenizer(
    model_name: str, device: str | None = None, dtype: torch.dtype | None = None
) -> tuple[PreTrainedModel, PreTrainedTokenizer]
```

- `device`: auto-detects `"cuda"` if available, else `"cpu"`.
- `dtype`: auto-selects fp16 on CUDA (Pascal and Turing both lack native bf16 support — that's
  Ampere+), fp32 on CPU.
- Sets `tokenizer.pad_token = tokenizer.eos_token` when the tokenizer has no pad token
  (true for Qwen2.5).
- Returns the model in `.eval()` mode, moved to the resolved device.

## `llm_serving.workload`

```python
def make_prompt(tokenizer, input_len: int, seed: int) -> torch.Tensor
```

Samples `input_len` random token ids from `range(vocab_size)`, rejecting and resampling any
id that appears in `tokenizer.all_special_ids`, and returns
a `(1, input_len)` tensor — no text round-trip. This gives exact, reproducible control over
prompt length (seeded), matching how vLLM's own benchmark scripts construct synthetic
workloads, and sidesteps chat-template formatting concerns entirely since raw token ids are
fed straight to `generate()`/the decode loop.

## `llm_serving.metrics`

**Per-request:**

```python
@dataclass
class RequestMetrics:
    request_id: int
    prompt_tokens: int
    output_tokens: int
    ttft_s: float
    total_latency_s: float
    inter_token_latencies_s: list[float]   # len == output_tokens - 1
```

**Per-run aggregate:**

```python
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

    request_throughput: float   # req/s over wall_clock_s
    token_throughput: float     # output tok/s over wall_clock_s
    timestamp: str
```

**Functions:**
- `summarize(requests: list[RequestMetrics], *, stage, model, device, dtype, batch_size, input_len, output_len, wall_clock_s) -> StageSummary` — percentiles via `numpy.percentile`.
- `print_table(summary: StageSummary) -> None` — aligned console output.
- `write_requests_csv(requests: list[RequestMetrics], path: Path) -> None` — raw per-request rows.
- `write_summary_json(summary: StageSummary, path: Path) -> None`.

**TTFT/ITL capture**: a small `BaseStreamer` subclass (`TimestampStreamer`) whose `put()`
appends `time.perf_counter()`. `generate()` calls `put()` synchronously as each token is
produced, so no background thread is needed (unlike `TextIteratorStreamer`, which threads
because it's built for iterating text output concurrently — irrelevant here since only
timestamps are needed, not decoded text). `ttft_s` is the first timestamp minus the call
start; `total_latency_s` is the last timestamp minus the call start; inter-token latencies
are the gaps between consecutive timestamps.

Each stage's `generate_one()` runs with `min_new_tokens == max_new_tokens == output_len`
(EOS-based early stopping disabled), so every request in a run decodes exactly the requested
length and is directly comparable.

## `llm_serving.stages`

```python
# base.py
class Stage(Protocol):
    def generate_one(
        self, model, tokenizer, input_ids: torch.Tensor, output_len: int
    ) -> RequestMetrics: ...

STAGES: dict[str, Stage] = {}
```

Each stage module registers itself into `STAGES` under its name (e.g. `STAGES["naive"] = ...`
in `naive.py`) so `benchmarks/run.py` dispatches via lookup rather than growing an
if/elif chain as stages 2–4 are added.

`naive.py` implements stage 1: calls `model.generate()` with the `TimestampStreamer`,
`min_new_tokens`/`max_new_tokens` fixed to `output_len`, and builds the `RequestMetrics` from
the streamer's recorded timestamps.

## `benchmarks/run.py`

CLI:

```
--stage {naive}              # more choices as later stages register themselves
--model Qwen/Qwen2.5-0.5B-Instruct
--input-len 128
--output-len 128
--num-requests 10
--num-warmup 2                # untimed requests first; excludes CUDA kernel/autotune warmup
--seed 0                      # reproducible prompt content across runs
--output-dir <path>
```

Flow: load model + tokenizer once → run `num_warmup` untimed requests (discarded) → time the
full loop wall-clock while collecting one `RequestMetrics` per request → `metrics.summarize()`
→ `print_table()` → `write_requests_csv()` + `write_summary_json()` into `--output-dir`.

`benchmarks/01_naive_generate/run.py` shells out to this with `--stage naive`, the default
model, and `--output-dir benchmarks/01_naive_generate/results`.

## Testing

`tests/test_metrics.py` covers the aggregation math in `metrics.summarize()` — percentile
computation, throughput calculation, edge cases (single request, zero inter-token latencies)
— using synthetic `RequestMetrics` instances directly, no model or tokenizer involved. Fast
and runs on CPU/laptop without any download. Correctness tests comparing generated output
across stages (e.g. stage 2's manual KV cache must produce identical tokens to stage 1) are
deferred to when stage 2 is designed, since they need a real (tiny) model.

## Error handling

None beyond what the libraries already provide. This is a single-user local research
benchmark, not a service — no retries, no partial-failure handling, no input validation
beyond what's needed to catch genuine usage mistakes early (e.g. argparse's own validation of
`--stage` choices).
