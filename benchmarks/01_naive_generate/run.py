#!/usr/bin/env python
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent

if __name__ == "__main__":
    subprocess.run(
        [
            sys.executable,
            str(HERE.parent / "run.py"),
            "--stage", "naive",
            "--model", "Qwen/Qwen2.5-0.5B-Instruct",
            "--input-len", "128",
            "--output-len", "128",
            "--num-requests", "10",
            "--num-warmup", "2",
            "--seed", "0",
            "--output-dir", str(HERE / "results"),
        ],
        check=True,
    )
