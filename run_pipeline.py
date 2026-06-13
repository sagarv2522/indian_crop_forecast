"""Single entry point for training, serving, and full project runs."""

from __future__ import annotations

import argparse
import subprocess
import sys

from src.mlops.retrain_pipeline import run_pipeline as retrain_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Project orchestration entry point")
    parser.add_argument("--train", action="store_true", help="Run retraining only")
    parser.add_argument("--serve", action="store_true", help="Start the FastAPI service")
    parser.add_argument("--full", action="store_true", help="Run retraining, then start the API")
    parser.add_argument("--crop", default="Tomato")
    parser.add_argument("--state", default="Tamil Nadu")
    parser.add_argument("--freq", default="D")
    args = parser.parse_args()

    if args.train or args.full:
        retrain_pipeline(args.crop, args.state, args.freq)

    if args.serve or args.full:
        subprocess.run([sys.executable, "-m", "uvicorn", "api.main:app", "--reload"], check=True)


if __name__ == "__main__":
    main()
