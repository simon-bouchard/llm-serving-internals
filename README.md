# LLM Serving Internals

Hand-rolled LLM inference serving mechanics (KV caching, static batching, continuous batching)
built on a plain `transformers` forward pass, benchmarked stage by stage, then compared against
vLLM.

This is the LLM counterpart to [cv-inference-triton](https://github.com/simon-bouchard/cv-inference-triton):
the same benchmark-driven methodology, applied to autoregressive text generation instead of
vision models. Hardware limits are treated as part of the experiment rather than something to
engineer around.

> Work in progress. Stage 1 (baseline) is complete; stage 2 (manual KV cache) is in design.

---

## Stages

Each stage gets its own experiment directory under `benchmarks/` with a write-up covering what
changed, the hypothesis, results, why the numbers look the way they do, and takeaways.

| # | Stage | Hardware | Status |
|---|-------|----------|--------|
| 1 | Naive `generate()` baseline | GTX 1060 | Done |
| 2 | Manual KV cache (hand-written decode loop) | GTX 1060 | In design |
| 3 | Static batching | GTX 1060 | Planned |
| 4 | Continuous / in-flight batching | GTX 1060 | Planned |
| 5 | vLLM (PagedAttention, continuous batching) | Kaggle T4 | Planned |

The GTX 1060 (Pascal, compute capability 6.1) is not supported by vLLM, so the comparison leg
runs on a Kaggle T4. That hardware difference is an intentional, unavoidable confound and is
called out explicitly rather than hidden inside a single "vLLM is N times faster" number.

---

## Results so far

**Stage 1: naive `generate()`** on Qwen2.5-0.5B-Instruct, fp16, batch size 1, 128 input / 128
output tokens.

| Metric | p50 | Mean |
|--------|-----|------|
| End-to-end latency | 3210 ms | 3232 ms |
| Time to first token | 54.8 ms | 55.0 ms |
| Inter-token latency | 24.8 ms | 25.0 ms |

Throughput: **39.6 tok/s** (0.31 req/s).

Latency is almost entirely decode: prefill is about 1.7% of the total. At 25 ms per token, the
model streams roughly 40 GB/s of weights, only about 20% of the 1060's 192 GB/s peak, which
points to kernel-launch overhead and GPU underutilization at batch size 1. That headroom is what
the batching stages are expected to close.

Full write-up: [benchmarks/01_naive_generate/notes.md](benchmarks/01_naive_generate/notes.md)

---

## Methodology

- **One benchmarking path for every stage.** All stages run through `benchmarks/run.py` and
  share the `llm_serving.metrics` schema (per-request latency, TTFT, ITL, throughput), output
  as CSV, JSON and a console summary. This fixes a problem from the Triton project, which ended
  up with two benchmarking tools whose results didn't share a schema.
- **In-process measurement only.** No HTTP layer in any stage. The vLLM leg uses vLLM's offline
  API rather than its OpenAI-compatible server, so network and serialization overhead never
  becomes a confound between stages.
- **Fixed synthetic workload.** Prompts are fixed-length random token sequences generated from
  a seed, with warmup requests excluded from the results.

---

## Project structure

```
src/llm_serving/
  model.py        model/tokenizer loading, device and dtype selection
  workload.py     synthetic fixed-length prompt generation
  metrics.py      RequestMetrics, StageSummary, percentiles, CSV/JSON/console output
  cli.py          benchmark driver arguments
  stages/
    base.py       Stage protocol
    naive.py      stage 1
benchmarks/
  run.py                  the single benchmark driver
  01_naive_generate/      run.py (fixed args), notes.md, results/
tests/                    CPU-only unit tests, no model download
```

---

## Usage

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python benchmarks/01_naive_generate/run.py
```

Run any stage with custom parameters:

```bash
uv run python benchmarks/run.py --stage naive --model Qwen/Qwen2.5-0.5B-Instruct \
  --input-len 128 --output-len 128 --num-requests 10 --output-dir results/
```

Run the tests:

```bash
uv run pytest
```

`torch` is pinned to PyTorch's CUDA 12.1 wheel index, because the 1060's driver (535) caps
out at CUDA 12.2 and the default PyPI wheel fails at import.

---

## Hardware

- **GPU:** NVIDIA GTX 1060 3GB (Pascal, compute capability 6.1), driver 535, CUDA 12.2
- **CPU:** Intel Core i7-6700 (8 threads) @ 4.00 GHz
- **RAM:** 16 GB DDR4 @ 2133 MT/s
- **Comparison leg:** Kaggle T4 (Turing, compute capability 7.5)
