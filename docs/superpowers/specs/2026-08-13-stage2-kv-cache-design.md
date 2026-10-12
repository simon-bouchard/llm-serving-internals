# Stage 2: Manual KV Cache — design

Started: 2026-08-13. Completed: 2026-10-12.

## Purpose and framing

Stage 2 is primarily an **enabling step** for stages 3–4: it replaces `generate()` with a
hand-written decode loop, which is what batching and continuous batching need (you cannot
interleave differently-progressed sequences through `generate()`).

`generate()` already uses a KV cache by default (`use_cache=True`), so stage 1 vs. stage 2 is
*HF's cached loop vs. our cached loop*, not *no cache vs. cache*. To give experiment 02 a real
finding rather than a parity non-result, it adds a **no-cache control**: the same manual loop
with caching disabled, re-feeding the full sequence every step. Experiment 02 compares three
stages: `naive`, `kv_cache`, `no_cache`.

## Stage 1 fix: pure greedy `generate()`

`do_sample=False` disables temperature/top-k/top-p, but `generate()` still applies the
model's `generation_config.repetition_penalty` (1.1 for Qwen2.5-0.5B-Instruct, 1.05 for the
tiny test model). Stage 1 was therefore greedy *plus* a per-step logits rescoring that the
manual loops do not perform, and outputs diverge as soon as a token repeats. Verified during
design: with `repetition_penalty=1.0`, both manual loops (cached and uncached) match
`generate()` token-for-token on the tiny model; without it, neither does.

Decision: make stage 1 pure greedy, so all three stages in 02 do identical logits work and
produce identical tokens.

- `naive.py` gets `greedy_generate(model, tokenizer, input_ids, output_len, streamer=None)
  -> torch.Tensor`, the single place the `generate()` call lives. It passes
  `repetition_penalty=1.0` explicitly and returns only the newly generated token ids.
  `NaiveStage.generate_one()` calls it; the correctness test calls it too, so the test
  exercises exactly stage 1's code path.
- Experiment 01 is rerun on the 1060 with the fix. Its `results/` are overwritten (and gain
  `itl_by_position.csv`), and `notes.md` is updated: How tests were run notes the penalty is
  disabled, and Results / Why / Takeaways reflect the new numbers.

## KV cache mechanics: manual loop, HF-managed cache tensors

- Stage 2 writes its own step-by-step Python loop (no `generate()` call at all).
- With caching, each step calls `model(next_token, past_key_values=cache, use_cache=True)` and
  feeds the returned cache object back in, passing only the newly generated token after the
  first (prefill) step.
- The K/V tensor storage/concatenation itself is NOT reimplemented; it stays inside
  `model()`'s forward pass. It is architecture-specific (rotary embeddings applied to cached
  positions, per-head layout, etc.), and reimplementing it would mean rebuilding attention
  internals, which is outside this project's goal (serving mechanics: caching/batching/
  scheduling as techniques, not attention math from scratch).

## `llm_serving.stages.kv_cache`

One module, one class, registered twice:

```python
register_stage("kv_cache", ManualDecodeStage(use_cache=True))
register_stage("no_cache", ManualDecodeStage(use_cache=False))
```

### `decode()` — the shared loop

```python
def decode(
    model, input_ids: torch.Tensor, output_len: int, use_cache: bool
) -> tuple[torch.Tensor, float, list[float]]:
    """Greedy-decode output_len tokens; returns (tokens, start, per-token timestamps)."""
```

- Prefill: one forward pass over the prompt, greedy `argmax` on the last position's logits.
- Then `output_len - 1` decode steps, each producing one token:
  - `use_cache=True`: input is the single new token plus `past_key_values` from the previous
    step.
  - `use_cache=False`: input is `torch.cat([input_ids, generated_so_far], dim=-1)` with
    `use_cache=False`, so every step recomputes attention over the full sequence.
- Both branches share the same loop, argmax, and timing code, so the cache is the only
  variable between the two stages.
- Wrapped in `torch.no_grad()`.
- Returns the generated token ids (shape `[1, output_len]`) so the correctness test can call
  it directly. `RequestMetrics` does not carry tokens, and the `Stage` protocol is unchanged.

### Timing: explicit GPU synchronization

CUDA kernel launches are asynchronous: `model(...)` and `argmax` return once work is
*queued*. Without a sync, `perf_counter()` would timestamp enqueue time, not completion,
and stage 2's TTFT/ITL would look artificially better than stage 1's (where `generate()`'s
per-step stopping-criteria check forces a host sync).

A module-level `_sync(device)` helper calls `torch.cuda.synchronize()` when
`device.type == "cuda"` and is a no-op otherwise. It is called immediately before **every**
`perf_counter()` read: the start timestamp, after prefill, and after each decode step.

### `ManualDecodeStage.generate_one()`

Thin wrapper over `decode()`: moves `input_ids` to `model.device`, calls `decode()`, then
derives `RequestMetrics` exactly as `NaiveStage` does: `ttft_s = timestamps[0] - start`,
`total_latency_s = timestamps[-1] - start`, ITLs as consecutive timestamp differences,
`output_tokens = len(timestamps)`. Because the loop is ours, the first timestamp *is* TTFT,
so stage 1's `TimestampStreamer` prompt-echo workaround is not needed.

### Registration

`llm_serving/cli.py` imports `llm_serving.stages.kv_cache` alongside the existing
`llm_serving.stages.naive` import, so both new names appear in `--stage` choices.

## Per-position ITL output (`llm_serving.metrics`)

The existing outputs collapse ITL to a per-request mean (`requests.csv`) and p50/mean
(`summary.json`), which cannot distinguish "ITL grows steadily" from "ITL is uniformly
higher". The no-cache control's whole point is that shape, and stage 4 will need the same
view (ITL spikes when a newly admitted request's prefill stalls the batch).

- `PositionITL` dataclass: `position: int`, `itl_p50_ms: float`, `itl_mean_ms: float`,
  `n: int`.
- `itl_by_position(requests: list[RequestMetrics]) -> list[PositionITL]`:
  - `position` *k* (1-based) is the gap between output token *k* and *k*+1, so a request
    with N output tokens contributes positions 1..N-1.
  - Aggregated across requests per position, vectorised: ITL lists are stacked into a 2D
    numpy array padded with NaN, then `np.nanmedian` / `np.nanmean` along the request axis.
  - `n` is the count of non-NaN entries at that position. Today every request decodes exactly
    `output_len` tokens so `n == num_requests` everywhere; the NaN handling exists so ragged
    lengths in stage 4 cannot silently skew averages.
  - Returns an empty list when no request has any ITLs.
- `write_itl_by_position_csv(rows, path)` writes `results/itl_by_position.csv` with columns
  `position, itl_p50_ms, itl_mean_ms, n`. Called from `run_benchmark()` next to the existing
  two writers.
- `StageSummary` / `summary.json` are **unchanged**, so stage 1's committed results remain
  schema-valid.
- `print_table()` gains a block showing ITL p50 at four positions: `1`, `N // 4`, `N // 2`,
  and `N - 1`, where `N = output_len`. Positions are deduplicated and clamped to what exists,
  and the block is skipped when there are no ITLs. `print_table()` takes the
  `list[PositionITL]` as a new parameter. Example for 128 output tokens:

  ```
  ITL by position (ms)     1      32      64     127
                       24.80   24.75   24.81   24.79
  ```

## Experiment `benchmarks/02_kv_cache/`

### `run.py`

Same thin subprocess-wrapper shape as `01_naive_generate/run.py`, looping over the stages
`naive`, `kv_cache`, `no_cache` in that order. Each invocation uses 01's fixed args
(`Qwen/Qwen2.5-0.5B-Instruct`, `--input-len 128`, `--output-len 128`, `--num-requests 10`,
`--num-warmup 2`, `--seed 0`) and writes to `results/<stage>/`.

`naive` is rerun inside 02 rather than reusing 01's numbers, so all three share one
session (thermal state, driver state) and run-to-run drift does not confound the comparison.

### `notes.md`

Same sections as 01. *What changed / Hypothesis / How tests were run / Hardware* are written
before the run; *Results / Why / Key takeaways* are filled in after the user reports numbers
back from the 1060. Hypotheses committed to up front:

1. `kv_cache` ≈ `naive` within a few percent on throughput and ITL. Same mechanism and, after
   the stage 1 fix, the same logits work; any gap reflects `generate()`'s remaining per-step
   bookkeeping (logits-processor list, stopping-criteria checks, streamer calls) vs. the
   bare loop.
2. TTFT is the same across all three stages: each does one identical prefill pass.
3. `no_cache` ITL climbs with position, starting near the 128-token prefill cost (~55ms)
   rather than ~25ms. The climb is likely sublinear at first, because a 0.5B model at these
   sequence lengths is probably still kernel-launch/overhead-bound rather than
   attention-FLOP-bound; the per-position CSV confirms or refutes this.

## Testing

### Test restructure: split unit vs. integration

- `tests/unit/` — the 6 existing fast, no-download test files moved as-is from `tests/`
  (`test_metrics.py`, `test_model.py`, `test_workload.py`, `test_stages_base.py`,
  `test_stages_naive.py`, `test_cli.py`). Pure file move: all import from `llm_serving.*`.
- `tests/integration/` — slower, model-downloading tests.
- `tests/conftest.py` — a `session`-scoped `tiny_model()` / `tiny_tokenizer()` fixture
  (loads once per pytest run), plus the `FakeModel` / `FakeTokenizer` / `FakeStage`-style
  stubs currently duplicated across `test_stages_base.py` and `test_cli.py`, consolidated.
- The directory split doubles as the CI-selection mechanism later (`pytest tests/unit` vs.
  `pytest tests/integration`) without custom pytest markers.

### Tiny test model

`yujiepan/qwen2-tiny-random`: real Qwen2 architecture (2 layers, hidden size 8, ~2.4M
params), no auth, ~16MB. Verified to load and run `generate()`; its tokenizer exposes
`vocab_size` / `all_special_ids` / `pad_token` as `make_prompt` / `load_model_and_tokenizer`
expect. `from_pretrained()` caches to `~/.cache/huggingface/hub/`, so a fresh machine needs
network once for `pytest tests/integration`, and is offline after that.

### `tests/integration/test_kv_cache_correctness.py`

On the tiny model (CPU, fp32), for a fixed prompt from `make_prompt`:

- `decode(..., use_cache=True)` tokens == `greedy_generate(...)` tokens (stage 1's path).
- `decode(..., use_cache=False)` tokens == the same `greedy_generate()` output.
- The prompt and length are chosen so the greedy output contains a repeated token, which is
  the case that exposed the repetition-penalty mismatch; the test asserts this so it cannot
  silently stop covering it.
- `ManualDecodeStage(use_cache=...).generate_one()` returns `output_tokens == N` and
  `len(inter_token_latencies_s) == N - 1` for both variants.

An earlier hand check of the cached loop reported a match without the penalty override; it
was a false positive (no repeated tokens in that sample). This test makes the check permanent
and covers the repeat case.

### Unit tests (`tests/unit/`)

- `test_metrics.py`: `itl_by_position` p50/mean on hand-built `RequestMetrics` with known
  ITLs; a ragged-length case checking `n` counts and NaN handling; the empty case;
  `write_itl_by_position_csv` header and rows.
- `test_cli.py`: `kv_cache` and `no_cache` appear in `--stage` choices; `run_benchmark()`
  writes `itl_by_position.csv` (using the existing fake-model/stage stubs).

## Documentation updates

- `CLAUDE.md` project-structure tree: add `stages/kv_cache.py`, `tests/unit/`,
  `tests/integration/`, `tests/conftest.py`, `benchmarks/02_kv_cache/`, and
  `itl_by_position.csv` in the results description.
- `README.md`: update if it lists stages or the results file set.

## Out of scope

- KV cache VRAM tracking (`torch.cuda.max_memory_allocated`): deferred to stage 3, where
  memory starts capping batch size on the 3GB card.
- Reimplementing K/V storage or attention.
