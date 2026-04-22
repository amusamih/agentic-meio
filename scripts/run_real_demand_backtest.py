"""Run the bounded real-demand backtest lane."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tomllib

from meio.backtesting.demand_backtest import (
    write_demand_backtest_artifacts,
    write_demand_backtest_panel_artifacts,
)
from meio.evaluation.logging_io import jsonable


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the bounded real-demand backtest lane.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment/real_demand_backtest.toml"),
        help="Path to the real-demand backtest config.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        help="Run one configured mode or all configured modes.",
    )
    parser.add_argument(
        "--llm-client-mode",
        choices=("config", "fake", "real"),
        default="config",
        help="Override the configured LLM client mode.",
    )
    args = parser.parse_args()

    llm_client_mode_override = None if args.llm_client_mode == "config" else args.llm_client_mode
    config_document = tomllib.loads(args.config.read_text(encoding="utf-8"))
    if "slices" in config_document:
        panel_run = write_demand_backtest_panel_artifacts(
            args.config,
            mode=args.mode,
            llm_client_mode_override=llm_client_mode_override,
        )
        payload = {
            "artifacts_dir": str(panel_run.output_dir),
            "panel_config_hash": panel_run.panel_config_hash,
            "successful_slice_count": sum(
                1 for result in panel_run.slice_results if result.success
            ),
            "failed_slice_count": sum(
                1 for result in panel_run.slice_results if not result.success
            ),
            "slice_results": [
                {
                    "slice_name": result.slice_name,
                    "success": result.success,
                    "output_dir": str(result.output_dir) if result.output_dir is not None else None,
                    "error_message": result.error_message,
                }
                for result in panel_run.slice_results
            ],
            "experiment_metadata": (
                jsonable(panel_run.experiment_metadata)
                if panel_run.experiment_metadata is not None
                else None
            ),
        }
    else:
        metadata, output_dir, written_files = write_demand_backtest_artifacts(
            args.config,
            mode=args.mode,
            llm_client_mode_override=llm_client_mode_override,
        )
        payload = {
            "artifacts_dir": str(output_dir),
            "artifact_files": {key: str(path) for key, path in written_files.items()},
            "experiment_metadata": jsonable(metadata),
        }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
