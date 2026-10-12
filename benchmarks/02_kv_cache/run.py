#!/usr/bin/env python
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
STAGES_TO_RUN = ("naive", "kv_cache", "no_cache")

if __name__ == "__main__":
    for stage in STAGES_TO_RUN:
        subprocess.run(
            [
                sys.executable,
                str(HERE.parent / "run.py"),
                "--stage", stage,
                "--model", "Qwen/Qwen2.5-0.5B-Instruct",
                "--input-len", "128",
                "--output-len", "128",
                "--num-requests", "10",
                "--num-warmup", "2",
                "--seed", "0",
                "--output-dir", str(HERE / "results" / stage),
            ],
            check=True,
        )
