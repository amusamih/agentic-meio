"""Summarize the current validation stack across saved artifact lanes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from meio.evaluation.validation_comparison import (
    default_validation_run_dirs,
    summarize_validation_stack,
)
from meio.evaluation.logging_io import jsonable


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize saved validation-lane artifacts across the repo.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Results root used to locate the default validation lanes.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        default=[],
        help="Optional explicit run directories to summarize.",
    )
    args = parser.parse_args()

    run_dirs = tuple(args.run_dir) if args.run_dir else default_validation_run_dirs(args.results_root)
    summary = summarize_validation_stack(run_dirs)
    print(json.dumps(jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
