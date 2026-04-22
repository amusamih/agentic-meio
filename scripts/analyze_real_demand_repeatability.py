"""Analyze repeatability across multiple real-demand runs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from meio.evaluation.real_demand_repeatability import (
    DEFAULT_REAL_DEMAND_RESULTS_ROOT,
    analyze_real_demand_repeatability,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze repeatability across saved real-demand runs."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        default=[],
        help="Repeatable option for explicit run directories to include.",
    )
    parser.add_argument(
        "--latest-n",
        type=int,
        default=None,
        help=(
            "Optional number of latest run directories to include when --run-dir is "
            "not provided."
        ),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_REAL_DEMAND_RESULTS_ROOT,
        help="Results root used when discovering runs automatically.",
    )
    args = parser.parse_args()

    analysis = analyze_real_demand_repeatability(
        tuple(args.run_dir),
        results_root=args.results_root,
        latest_n=args.latest_n,
    )
    print(json.dumps(asdict(analysis), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
