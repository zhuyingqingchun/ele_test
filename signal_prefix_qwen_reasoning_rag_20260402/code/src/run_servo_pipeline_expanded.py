from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from servo_diagnostic import run_pipeline
from servo_diagnostic.config import ServoPlantParams


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the servo simulation pipeline with optional repeated stochastic replicas.")
    parser.add_argument("--base-dir", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--repeat-count", type=int, default=1)
    parser.add_argument("--seed-stride", type=int, default=10000)
    parser.add_argument("--final-time", type=float, default=4.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--skip-per-run-csv", action="store_true")
    args = parser.parse_args()

    params = ServoPlantParams(dt=args.dt, final_time=args.final_time, random_seed=args.random_seed)
    outputs = run_pipeline(
        base_dir=args.base_dir,
        params=params,
        repeat_count=args.repeat_count,
        seed_stride=args.seed_stride,
        save_per_run=not args.skip_per_run_csv,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
