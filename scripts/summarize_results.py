"""Stub entry point for MEIO result summarization."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    """Parse arguments and dispatch result summarization."""
    parser = argparse.ArgumentParser(description="Summarize MEIO results.")
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to the summarization configuration file.",
    )
    parser.parse_args()
    raise NotImplementedError("The result summarizer is not implemented yet.")


if __name__ == "__main__":
    main()
