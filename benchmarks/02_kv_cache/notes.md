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
