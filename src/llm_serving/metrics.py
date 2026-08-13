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

    print(
        f"{'Latency (ms)':<14}{'p50':>10}{'p90':>10}{'p99':>10}{'mean':>10}{'min':>10}{'max':>10}"
    )
    print(
        f"{'':<14}{summary.latency_p50_ms:>10.2f}{summary.latency_p90_ms:>10.2f}"
        f"{summary.latency_p99_ms:>10.2f}{summary.latency_mean_ms:>10.2f}"
        f"{summary.latency_min_ms:>10.2f}{summary.latency_max_ms:>10.2f}"
    )

    print(f"\n{'TTFT (ms)':<14}{'p50':>10}{'p90':>10}{'mean':>10}")
    print(
        f"{'':<14}{summary.ttft_p50_ms:>10.2f}{summary.ttft_p90_ms:>10.2f}"
        f"{summary.ttft_mean_ms:>10.2f}"
    )

    print(f"\n{'ITL (ms)':<14}{'p50':>10}{'mean':>10}")
    print(f"{'':<14}{summary.itl_p50_ms:>10.2f}{summary.itl_mean_ms:>10.2f}")

    print(
        f"\nThroughput: {summary.request_throughput:.2f} req/s, "
        f"{summary.token_throughput:.2f} tok/s"
    )


def write_requests_csv(requests: list[RequestMetrics], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "request_id",
                "prompt_tokens",
                "output_tokens",
                "ttft_s",
                "total_latency_s",
                "mean_itl_s",
            ]
        )
        for r in requests:
            mean_itl = (
                statistics.mean(r.inter_token_latencies_s) if r.inter_token_latencies_s else 0.0
            )
            writer.writerow(
                [
                    r.request_id,
                    r.prompt_tokens,
                    r.output_tokens,
                    r.ttft_s,
                    r.total_latency_s,
                    mean_itl,
                ]
            )


def write_summary_json(summary: StageSummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(asdict(summary), f, indent=2)
