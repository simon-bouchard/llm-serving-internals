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
- GPU: RTX 1060 3GB (Pascal, compute 6.1)
- CPU/RAM: _fill in_
