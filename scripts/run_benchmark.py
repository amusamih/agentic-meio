"""Stub entry point for MEIO benchmark runs."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    """Parse arguments and dispatch the benchmark runner."""
    parser = argparse.ArgumentParser(description="Run a MEIO benchmark.")
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to the benchmark configuration file.",
    )
    parser.parse_args()
    raise NotImplementedError("The benchmark runner is not implemented yet.")


if __name__ == "__main__":
    main()
