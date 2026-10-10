"""python -m jmteb_eval run [--sizes 130m 310m] [--limit N]
python -m jmteb_eval summarize
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .constants import DEFAULT_RESULTS_DIRECTORY, DEFAULT_WORK_DIRECTORY, MODEL_SIZES
from .runner import run
from .summary import write_summary

_SMOKE_RESULTS_DIRECTORY_NAME = "smoke_results"


def main() -> None:
    parser = argparse.ArgumentParser(prog="jmteb_eval")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Embed and score every task (resumable).")
    run_parser.add_argument("--sizes", nargs="+", default=list(MODEL_SIZES), choices=MODEL_SIZES)
    run_parser.add_argument("--limit", type=int, help="Use only the first N rows per task (smoke test).")
    run_parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIRECTORY)
    run_parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIRECTORY)

    summary_parser = sub.add_parser("summarize", help="Write SUMMARY.md from finished results.")
    summary_parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIRECTORY)

    args = parser.parse_args()
    if args.command == "run":
        results_directory = (
            args.work_dir / _SMOKE_RESULTS_DIRECTORY_NAME if args.limit is not None else args.results_dir
        )
        run(args.sizes, args.work_dir, results_directory, args.limit)
        print(f"wrote {write_summary(results_directory)}")
    else:
        print(f"wrote {write_summary(args.results_dir)}")


if __name__ == "__main__":
    main()
