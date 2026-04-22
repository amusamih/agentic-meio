"""Stub entry point for MEIO experiment runs."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    """Parse arguments and dispatch the experiment runner."""
    parser = argparse.ArgumentParser(description="Run a MEIO experiment.")
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to the experiment configuration file.",
    )
    parser.parse_args()
    raise NotImplementedError("The experiment runner is not implemented yet.")


if __name__ == "__main__":
    main()
