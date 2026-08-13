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
