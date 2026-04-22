"""Analyze held-out Stockpyl paper-candidate results without changing policy."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from meio.evaluation.heldout_analysis import analyze_heldout_batch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a held-out Stockpyl paper-candidate batch directory."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Optional held-out batch directory. Defaults to the latest saved held-out run.",
    )
    args = parser.parse_args()

    analysis = analyze_heldout_batch(args.run_dir)
    print(json.dumps(asdict(analysis), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
