from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import llm_serving.stages.naive  # noqa: F401 - registers the "naive" stage
from llm_serving.metrics import (
    StageSummary,
    print_table,
    summarize,
    write_requests_csv,
    write_summary_json,
)
from llm_serving.model import load_model_and_tokenizer
from llm_serving.stages.base import STAGES
from llm_serving.workload import make_prompt


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified LLM serving benchmark driver")
    parser.add_argument("--stage", required=True, choices=sorted(STAGES.keys()))
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--num-requests", type=int, default=10)
    parser.add_argument("--num-warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def run_benchmark(args: argparse.Namespace) -> StageSummary:
    stage = STAGES[args.stage]
    model, tokenizer = load_model_and_tokenizer(args.model)
    device = str(next(model.parameters()).device)
    dtype = str(next(model.parameters()).dtype).removeprefix("torch.")

    for i in range(args.num_warmup):
        input_ids = make_prompt(tokenizer, args.input_len, seed=args.seed + i)
        stage.generate_one(model, tokenizer, input_ids, args.output_len)

    requests = []
    start = time.perf_counter()
    for i in range(args.num_requests):
        input_ids = make_prompt(tokenizer, args.input_len, seed=args.seed + args.num_warmup + i)
        result = stage.generate_one(model, tokenizer, input_ids, args.output_len)
        requests.append(replace(result, request_id=i))
    wall_clock_s = time.perf_counter() - start

    summary = summarize(
        requests,
        stage=args.stage,
        model=args.model,
        device=device,
        dtype=dtype,
        batch_size=1,
        input_len=args.input_len,
        output_len=args.output_len,
        wall_clock_s=wall_clock_s,
    )

    print_table(summary)
    write_requests_csv(requests, args.output_dir / "requests.csv")
    write_summary_json(summary, args.output_dir / "summary.json")

    return summary


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_benchmark(args)


if __name__ == "__main__":
    main()
