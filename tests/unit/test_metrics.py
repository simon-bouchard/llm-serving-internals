from pathlib import Path

import pytest

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
