"""Analyze frozen broad-eval Stockpyl batch results without changing policy."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from meio.evaluation.broad_eval_analysis import analyze_frozen_broad_batch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a frozen broad-eval Stockpyl batch directory."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Optional broad-eval batch directory. Defaults to the latest saved broad-eval run.",
    )
    args = parser.parse_args()

    analysis = analyze_frozen_broad_batch(args.run_dir)
    print(json.dumps(asdict(analysis), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
