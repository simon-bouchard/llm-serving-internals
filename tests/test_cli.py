from pathlib import Path

import torch

from llm_serving import cli
from llm_serving.metrics import RequestMetrics
from llm_serving.stages.base import STAGES


class FakeModel:
    def parameters(self):
        return iter([torch.zeros(1, dtype=torch.float32, device="cpu")])


class FakeTokenizer:
    pad_token_id = 0
    vocab_size = 100
    all_special_ids = [0]


class FakeStage:
    def __init__(self):
        self.calls = 0

    def generate_one(self, model, tokenizer, input_ids, output_len):
        self.calls += 1
        return RequestMetrics(
            request_id=0,
            prompt_tokens=input_ids.shape[-1],
            output_tokens=output_len,
            ttft_s=0.01,
            total_latency_s=0.02,
            inter_token_latencies_s=[0.01],
        )


def test_build_arg_parser_defaults():
    parser = cli.build_arg_parser()
    args = parser.parse_args(["--stage", "naive", "--output-dir", "/tmp/out"])

    assert args.model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert args.input_len == 128
    assert args.output_len == 128
    assert args.num_requests == 10
    assert args.num_warmup == 2
    assert args.seed == 0


def test_run_benchmark_excludes_warmup_requests_from_summary(monkeypatch, tmp_path: Path):
    fake_stage = FakeStage()
    STAGES["fake_cli_stage"] = fake_stage
    monkeypatch.setattr(
        cli, "load_model_and_tokenizer", lambda model_name: (FakeModel(), FakeTokenizer())
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


def test_run_benchmark_assigns_sequential_request_ids(monkeypatch, tmp_path: Path):
    fake_stage = FakeStage()
    STAGES["fake_cli_stage_2"] = fake_stage
    monkeypatch.setattr(
        cli, "load_model_and_tokenizer", lambda model_name: (FakeModel(), FakeTokenizer())
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
