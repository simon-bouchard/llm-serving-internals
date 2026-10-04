# Experiment 01 — Naive `generate()` Baseline

## What changed
First benchmark stage: calling `model.generate()` as-is (no manual KV cache, no batching),
single request at a time. This is the reference point every later stage (manual KV cache,
static batching, continuous batching) gets compared against.

## Hypothesis
This is the reference point, not a comparison against a prior stage, so the prediction is
about the absolute shape of the numbers rather than "faster/slower than X":

- TTFT should be small relative to total latency: it's one prefill forward pass over 128
  tokens, versus 128 sequential decode steps.
- Inter-token latency should be roughly flat across the run — at `batch_size=1`, every
  decode step does the same amount of work (one token, full KV cache from the growing
  sequence), so there's no batching-related variance yet.
- Decode is likely memory-bandwidth-bound rather than compute-bound at this batch size: a
  0.5B model's weights are small enough that reading them from memory each step probably
  dominates over the actual matmul FLOPs on the 1060. If so, fp16 vs. fp32 shouldn't matter
  much here — that gap should open up once batching adds real compute pressure in later
  stages.

## How tests were run

```bash
python benchmarks/01_naive_generate/run.py
```

Model: Qwen/Qwen2.5-0.5B-Instruct | Input length: 128 | Output length: 128 | Requests: 10 (+2 warmup)

## Results

```
Stage:    naive
Model:    Qwen/Qwen2.5-0.5B-Instruct
Device:   cuda:0 (float16)
Requests: 10  Batch size: 1
Input len: 128  Output len: 128

Latency (ms)         p50       p90       p99      mean       min       max
                 3209.68   3286.37   3391.47   3232.43   3195.99   3403.15

TTFT (ms)            p50       p90      mean
                   54.76     55.04     55.01

ITL (ms)             p50      mean
                   24.79     25.02

Throughput: 0.31 req/s, 39.59 tok/s
```

## Why

All three predictions from the Hypothesis section hold, with one caveat worth flagging:

- **TTFT is small relative to total latency**, as expected: ~55ms out of ~3210ms total
  (~1.7%). One 128-token prefill forward pass is cheap next to 128 sequential decode steps.
- **ITL is essentially flat**: p50 (24.79ms) and mean (25.02ms) are within 1% of each other.
  At `batch_size=1` every decode step really is doing the same fixed amount of work, so
  there's no per-step variance to explain yet — that's expected to change once batching
  stages introduce runs of differently-sized batches.
- **Decode looks memory-bandwidth-bound, but at a fraction of the 1060's peak bandwidth.**
  Back-of-envelope: Qwen2.5-0.5B is ~0.5B params, fp16 → ~1GB of weights to stream from
  memory per forward pass. At 25.02ms/token, that implies ~1GB / 0.025s ≈ 40GB/s of
  effective bandwidth used — against the 1060's ~192GB/s spec, that's roughly **20% of
  peak**. So while the flat ITL confirms *consistent* per-step cost (supporting a
  bandwidth-bound, not compute-bound, picture), the low achieved bandwidth means something
  besides pure weight-streaming is dominating the 25ms — most likely per-layer kernel-launch
  overhead and general GPU underutilization from doing tiny batch=1 matmuls, both classic
  symptoms of unbatched serving rather than a hardware bandwidth ceiling.

## Key takeaways

- Naive single-request `generate()` on the 1060 gets **~40 tok/s** at batch=1 — this is the
  number every later stage should beat, and the gap to the ~192GB/s bandwidth ceiling (using
  only ~20% of it) suggests there's real headroom for batching to close, not just marginal
  gains.
- End-to-end latency here is decode-dominated, not prefill-dominated (55ms TTFT vs. ~3150ms
  of decode) — so for this workload shape, KV-cache/batching work on the decode loop matters
  far more than prefill optimization.
- The GPU is very likely sitting mostly idle between kernel launches at batch=1; static and
  continuous batching (stages 3–4) are the ones expected to actually close the gap toward
  the bandwidth ceiling by keeping the GPU fed with more work per step.

## Hardware
- GPU: GTX 1060 3GB (Pascal, compute 6.1)
- CPU: Intel Core i7-6700 (8 threads) @ 4.00 GHz
- RAM: 15.54 GiB, DDR4 @ 2133 MT/s
- Driver: NVIDIA 535.309.01, CUDA 12.2
