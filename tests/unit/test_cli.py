from pathlib import Path

from llm_serving import cli
from llm_serving.stages.base import STAGES


def test_build_arg_parser_defaults():
    parser = cli.build_arg_parser()
    args = parser.parse_args(["--stage", "naive", "--output-dir", "/tmp/out"])

    assert args.model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert args.input_len == 128
    assert args.output_len == 128
    assert args.num_requests == 10
    assert args.num_warmup == 2
    assert args.seed == 0


def test_run_benchmark_excludes_warmup_requests_from_summary(
    monkeypatch, tmp_path: Path, fake_model, fake_tokenizer, fake_stage
):
    STAGES["fake_cli_stage"] = fake_stage
    monkeypatch.setattr(
        cli, "load_model_and_tokenizer", lambda model_name: (fake_model, fake_tokenizer)
    )

    parser = cli.build_arg_parser()
    args = parser.parse_args(
        [
            "--stage", "fake_cli_stage",
            "--output-dir", str(tmp_path),
            "--num-requests", "3",
            "--num-warmup", "2",
        ]
    )

    summary = cli.run_benchmark(args)

    assert fake_stage.calls == 5  # 2 warmup + 3 timed
    assert summary.num_requests == 3
    assert (tmp_path / "requests.csv").exists()
    assert (tmp_path / "summary.json").exists()


def test_run_benchmark_assigns_sequential_request_ids(
    monkeypatch, tmp_path: Path, fake_model, fake_tokenizer, fake_stage
):
    STAGES["fake_cli_stage_2"] = fake_stage
    monkeypatch.setattr(
        cli, "load_model_and_tokenizer", lambda model_name: (fake_model, fake_tokenizer)
    )

    parser = cli.build_arg_parser()
    args = parser.parse_args(
        [
            "--stage", "fake_cli_stage_2",
            "--output-dir", str(tmp_path),
            "--num-requests", "3",
            "--num-warmup", "0",
        ]
    )

    summary = cli.run_benchmark(args)
    assert summary.num_requests == 3


def test_run_benchmark_writes_itl_by_position_csv(
    monkeypatch, tmp_path: Path, fake_model, fake_tokenizer, fake_stage
):
    STAGES["fake_cli_stage_3"] = fake_stage
    monkeypatch.setattr(
        cli, "load_model_and_tokenizer", lambda model_name: (fake_model, fake_tokenizer)
    )

    parser = cli.build_arg_parser()
    args = parser.parse_args(
        [
            "--stage", "fake_cli_stage_3",
            "--output-dir", str(tmp_path),
            "--output-len", "4",
            "--num-requests", "2",
            "--num-warmup", "0",
        ]
    )

    cli.run_benchmark(args)

    lines = (tmp_path / "itl_by_position.csv").read_text().splitlines()
    assert lines[0] == "position,itl_p50_ms,itl_mean_ms,n"
    assert len(lines) == 4  # header + positions 1..3
