"""Run a bounded benchmark-qualification pass across candidate benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from meio.data.benchmark_adapters import available_benchmark_adapters
from meio.evaluation.benchmark_selection import (
    BenchmarkCandidate,
    BenchmarkQualificationSummary,
    build_qualification_summary,
    load_qualification_spec,
)


DEFAULT_QUALIFICATION_CONFIGS: tuple[Path, ...] = (
    Path("configs/benchmark/qualification_stockpyl.toml"),
    Path("configs/benchmark/qualification_orgym.toml"),
    Path("configs/benchmark/qualification_mabim.toml"),
)


def run_benchmark_qualification(
    config_paths: Sequence[str | Path] = DEFAULT_QUALIFICATION_CONFIGS,
) -> tuple[BenchmarkQualificationSummary, ...]:
    """Load qualification configs and summarize current benchmark fit."""

    adapters = {
        BenchmarkCandidate(adapter.candidate_id): adapter
        for adapter in available_benchmark_adapters()
    }
    summaries: list[BenchmarkQualificationSummary] = []
    for config_path in config_paths:
        spec = load_qualification_spec(config_path)
        adapter = adapters.get(spec.candidate)
        if adapter is None:
            raise ValueError(f"No benchmark adapter boundary is registered for {spec.candidate.value}.")
        status = adapter.describe()
        summaries.append(
            build_qualification_summary(
                spec=spec,
                topology_style=status.topology_style,
                smoke_testable_now=status.smoke_testable_now,
                available_modules=status.available_modules,
                missing_modules=status.missing_modules,
                integration_work_remaining=status.integration_work_remaining,
                adapter_notes=status.notes,
            )
        )
    return tuple(summaries)


def main() -> None:
    """Run benchmark qualification and print compact structured output."""

    parser = argparse.ArgumentParser(description="Run the MEIO benchmark-qualification pass.")
    parser.add_argument(
        "configs",
        nargs="*",
        type=Path,
        default=list(DEFAULT_QUALIFICATION_CONFIGS),
        help="Qualification TOML files to load.",
    )
    args = parser.parse_args()
    summaries = run_benchmark_qualification(args.configs)
    print(json.dumps([summary.to_record() for summary in summaries], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
