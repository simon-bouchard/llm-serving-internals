# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Goal

Build understanding of LLM inference serving mechanics — the LLM-side counterpart to the
CV inference/Triton (geo-classifier) project. Hardware constraints are treated as part of
the experiment, not something to engineer around: the write-up should surface what the
compute capability of each device does and doesn't allow, mirroring the Pascal/TensorRT
finding from the geo-classifier project.

Model candidates (0.5B–1.7B, must fit comfortably in 3GB VRAM): Qwen2.5-0.5B/1.5B,
SmolLM2, TinyLlama. Not finalized — see Open Questions.

## Hardware / execution environments

Three separate machines are in play. Claude only runs in the first one — it has no direct
access to the other two.

- **WSL2 laptop (this session)** — main dev environment. Code, experiment design, and
  analysis happen here. No serious GPU available for the benchmark runs themselves.
- **RTX 1060 desktop (Pascal, compute capability 6.1)** — booted into Ubuntu by the user
  for GPU work. No vLLM support on this card. This is where the hand-rolled KV
  cache / batching / continuous-batching implementation gets benchmarked. Claude has no
  remote access to this machine — write self-contained scripts for the user to run there
  and report results back. (If direct access is ever needed, the user will invoke Claude
  Code from that desktop directly rather than granting SSH access from here.)
  - When writing the benchmark harness for this leg, hand-implement KV caching and
    batching directly on top of a plain `transformers` forward pass — do not reach for a
    serving framework's built-in implementation for the manual stages.
- **Kaggle T4 (compute capability 7.5, vLLM-capable)** — runs the same model through vLLM
  (PagedAttention, continuous batching) as the comparison leg. Claude has no direct access;
  produce a runnable notebook/script for the user to execute on Kaggle and report results
  back.

## Benchmark methodology

Same benchmarking style as the Triton/geo-classifier project: p50/throughput tables,
saturation points. Stages to compare on the 1060:

1. Naive `generate()` baseline
2. Manual KV cache
3. Static batching
4. Continuous / in-flight batching

Then compare against vLLM on the T4 — both on throughput and on what the library handles
that the manual implementation doesn't (this leg should read as a genuine extension of the
methodology, not a bolted-on afterthought — same rigor/format as the manual stages).

## Project structure & benchmarking convention

```
llm-serving-internals/
├── src/
│   └── llm_serving/
│       ├── model.py         # load_model_and_tokenizer(): device/dtype selection, pad-token fixup
│       ├── workload.py      # synthetic fixed-length random-token prompt generation
│       ├── metrics.py       # shared benchmarking "utils": RequestMetrics, StageSummary, percentiles, CSV/JSON/console output
│       └── stages/
│           ├── base.py      # Stage protocol: generate_one(model, tok, input_ids, output_len) -> RequestMetrics
│           └── naive.py     # stage 1, registered under name "naive" (kv_cache/static_batch/continuous_batch land later)
├── benchmarks/
│   ├── run.py                # the one real driver: --stage, --model, --input-len, --output-len, --num-requests, --output-dir
│   ├── 01_naive_generate/
│   │   ├── run.py            # thin wrapper: subprocess-invokes benchmarks/run.py with this experiment's fixed args
│   │   ├── notes.md          # write-up: What changed / Hypothesis / How tests were run / Results / Why / Takeaways / Hardware
│   │   └── results/          # csv + json outputs, committed
│   └── 0N_.../                # one dir per stage, added as each stage lands
└── tests/
    └── test_metrics.py       # percentile/aggregation math, CPU-only, no model download
```

**Everything routes through one benchmarking system** — `benchmarks/run.py` plus
`llm_serving.metrics` — rather than a different tool per stage. This was a deliberate
correction from the cv-inference-triton project, which ended up with two incompatible
benchmarking paths (`perf_analyzer` CSVs vs. the custom `load_test.py` client) that
didn't share a schema.

Numbered experiment dirs (`01_...`, `02_...`) aren't valid Python module names, so their
`run.py` shells out to `benchmarks/run.py` as a subprocess (`sys.executable`) rather than
importing it — same shape as `run_pipeline.sh` calling `load_test.py` in the CV project,
just in Python instead of bash.

**Benchmarking tool choice**: all 5 stages are measured in-process, with no HTTP layer
anywhere, so methodology is never a confound between them. Stages 1–4 call `model.generate()`
or the hand-rolled decode loop directly and are timed by `llm_serving.metrics`. Stage 5
(vLLM on the T4) uses vLLM's **offline/in-process API** (`vllm.LLM(...).generate(...)`), not
its OpenAI-compatible HTTP server — going through the server + `benchmark_serving.py` would
add network/serialization overhead the other stages never pay, making "vLLM is faster"
partly an artifact of the measurement path rather than the scheduler. vLLM's `RequestOutput`
carries native per-request timing (`arrival_time`, `first_token_time`, etc.) that gets mapped
into the same `RequestMetrics`/`StageSummary` schema as every other stage.

Hardware (1060 vs. T4) remains an intentional, unavoidable confound between the manual
stages and the vLLM leg — call it out explicitly in the write-up rather than letting it hide
inside a single headline "vLLM is N× faster" number. A possible future extension: also run
the hand-rolled continuous-batching stage on the T4, so mechanism (hand-rolled vs. vLLM) can
be compared on identical hardware, separate from the 1060's hardware-constrained baseline.

## Environment / tooling

- Python dependency management: `uv` (pyproject.toml + uv.lock), package mode with `src/`
  layout so `uv sync` installs `llm_serving` editable and benchmark scripts can just
  `import llm_serving`.
- Linting/formatting: `ruff`. Type-checking: `pyright`. Config lives in `pyproject.toml`
  (`[tool.ruff]`) and `pyrightconfig.json`.

## Open questions (unresolved — check before assuming an answer)

- Scope of the manual implementation: KV cache + static batching only, or push through to
  a working continuous-batching scheduler?
- Which model size actually shows a meaningful difference between stages, vs. being too
  small to bottleneck on anything interesting.
- Whether to route serving through Triton Inference Server (ties more explicitly to the CV
  project) or stay framework-light (llama.cpp / raw PyTorch) for simplicity.
- How to structure the T4/vLLM comparison so it reads as a genuine extension rather than
  bolted-on.
- Repo name: currently `llm-serving-internals`, not fully locked in.
