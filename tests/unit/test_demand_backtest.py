from __future__ import annotations

import json
from pathlib import Path
import shutil
from uuid import uuid4

import pytest

from meio.backtesting.demand_backtest import (
    run_real_demand_backtest_batch,
    write_demand_backtest_panel_artifacts,
)
import scripts.run_real_demand_backtest as run_real_demand_backtest_script


def _write_temp_backtest_inputs(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    demand_lines = ["Date,SKU1,SKU2,SKU3,SKU4,SKU5"]
    leadtime_lines = ["Date,SKU1,SKU2,SKU3,SKU4,SKU5"]
    for day in range(1, 17):
        demand_lines.append(
            f"2020-01-{day:02d},{10 + day % 3},{9 + day % 2},{11},{8 + day % 2},{10}"
        )
        leadtime_lines.append(
            f"2020-01-{day:02d},2,2,3,2,2"
        )
    (data_dir / "demand.csv").write_text("\n".join(demand_lines) + "\n", encoding="utf-8")
    (data_dir / "leadtime.csv").write_text(
        "\n".join(leadtime_lines) + "\n",
        encoding="utf-8",
    )
    (data_dir / "sku_list.csv").write_text(
        "SKU\nSKU1\nSKU2\nSKU3\nSKU4\nSKU5\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "real_demand_backtest.toml"
    config_path.write_text(
        "\n".join(
            (
                "[experiment]",
                'name = "temp_real_demand_backtest"',
                'benchmark_config = "configs/benchmark/serial_3_echelon.toml"',
                'agent_config = "configs/agent/base.toml"',
                'dataset_name = "temp_series"',
                'discovery_module = "TempDataset"',
                f'dataset_root = "{tmp_path.as_posix()}"',
                'demand_csv_path = "data/demand.csv"',
                'leadtime_csv_path = "data/leadtime.csv"',
                'sku_list_path = "data/sku_list.csv"',
                "selected_sku_count = 5",
                'subset_selection = "nearest_benchmark_mean"',
                "training_window_days = 8",
                "history_window_days = 3",
                "forecast_update_window_days = 1",
                "evaluation_horizon_days = 4",
                "roll_forward_stride_days = 1",
                (
                    'mode_set = ["deterministic_baseline", "robust_policy", '
                    '"scenario_rolling_horizon_policy", '
                    '"llm_regret_guarded_risk_sensitive_scenario_planner_orchestrator"]'
                ),
                f'results_dir = "{(tmp_path / "results").as_posix()}"',
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return config_path


def _write_temp_backtest_panel_inputs(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    demand_lines = ["Date,SKU1,SKU2,SKU3,SKU4,SKU5,SKU6"]
    leadtime_lines = ["Date,SKU1,SKU2,SKU3,SKU4,SKU5,SKU6"]
    for day in range(1, 21):
        demand_lines.append(
            f"2020-01-{day:02d},{10 + day % 3},{9 + day % 2},{11},{8 + day % 2},{10},{6 + day % 2}"
        )
        leadtime_lines.append(
            f"2020-01-{day:02d},2,2,3,2,2,4"
        )
    (data_dir / "demand.csv").write_text("\n".join(demand_lines) + "\n", encoding="utf-8")
    (data_dir / "leadtime_store1.csv").write_text(
        "\n".join(leadtime_lines) + "\n",
        encoding="utf-8",
    )
    (data_dir / "leadtime_store2.csv").write_text(
        "\n".join(leadtime_lines) + "\n",
        encoding="utf-8",
    )
    (data_dir / "sku_list.csv").write_text(
        "SKU\nSKU1\nSKU2\nSKU3\nSKU4\nSKU5\nSKU6\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "real_demand_backtest_panel.toml"
    config_path.write_text(
        "\n".join(
            (
                "[experiment]",
                'name = "temp_real_demand_panel"',
                'benchmark_config = "configs/benchmark/serial_3_echelon.toml"',
                'agent_config = "configs/agent/base.toml"',
                'discovery_module = "TempDataset"',
                f'dataset_root = "{tmp_path.as_posix()}"',
                (
                    'mode_set = ["deterministic_baseline", "robust_policy", '
                    '"scenario_rolling_horizon_policy", '
                    '"llm_regret_guarded_risk_sensitive_scenario_planner_orchestrator"]'
                ),
                f'results_dir = "{(tmp_path / "results").as_posix()}"',
                "",
                "[[slices]]",
                'name = "slice_a"',
                'dataset_name = "temp_store1"',
                'demand_csv_path = "data/demand.csv"',
                'leadtime_csv_path = "data/leadtime_store1.csv"',
                'sku_list_path = "data/sku_list.csv"',
                'selected_skus = ["SKU1", "SKU2", "SKU3", "SKU4", "SKU5"]',
                'subset_selection = "explicit_selected_skus"',
                "training_window_days = 8",
                "history_window_days = 3",
                "forecast_update_window_days = 1",
                "evaluation_horizon_days = 4",
                'evaluation_start_date = "2020-01-09"',
                "",
                "[[slices]]",
                'name = "slice_b"',
                'dataset_name = "temp_store2"',
                'demand_csv_path = "data/demand.csv"',
                'leadtime_csv_path = "data/leadtime_store2.csv"',
                'sku_list_path = "data/sku_list.csv"',
                'selected_skus = ["SKU2", "SKU3", "SKU4", "SKU5", "SKU6"]',
                'subset_selection = "explicit_selected_skus"',
                "training_window_days = 8",
                "history_window_days = 3",
                "forecast_update_window_days = 1",
                "evaluation_horizon_days = 4",
                'evaluation_start_date = "2020-01-11"',
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return config_path


def test_run_real_demand_backtest_batch_preserves_optimizer_boundary() -> None:
    tmp_path = Path(".tmp_demand_backtest_tests") / uuid4().hex
    config_path = _write_temp_backtest_inputs(tmp_path)
    try:
        batch = run_real_demand_backtest_batch(config_path)

        assert batch.aggregate_summary.validation_lane == "real_demand_backtest"
        assert {run.mode for run in batch.runs} == {
            "deterministic_baseline",
            "robust_policy",
            "scenario_rolling_horizon_policy",
            "llm_regret_guarded_risk_sensitive_scenario_planner_orchestrator",
        }
        assert all(
            run.episode_summary_record.validation_lane == "real_demand_backtest"
            for run in batch.runs
        )
        assert all(
            run.episode_summary_record.optimizer_order_boundary_preserved
            for run in batch.runs
        )
        assert all(
            run.llm_call_trace_records == ()
            for run in batch.runs
            if run.mode in {"robust_policy", "scenario_rolling_horizon_policy"}
        )
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_run_real_demand_backtest_supports_latest_guarded_agentic_mode() -> None:
    tmp_path = Path(".tmp_demand_backtest_latest_mode_tests") / uuid4().hex
    config_path = _write_temp_backtest_inputs(tmp_path)
    try:
        batch = run_real_demand_backtest_batch(
            config_path,
            mode="llm_regret_guarded_risk_sensitive_scenario_planner_orchestrator",
            llm_client_mode_override="fake",
        )

        assert len(batch.runs) == 1
        run = batch.runs[0]
        assert run.mode == "llm_regret_guarded_risk_sensitive_scenario_planner_orchestrator"
        assert run.summary.optimizer_order_boundary_preserved
        assert run.llm_call_trace_records
        assert {
            "regime_diagnosis_tool",
            "regime_belief_tool",
            "scenario_candidate_generator_tool",
            "risk_sensitive_scenario_evaluator_tool",
            "counterfactual_regret_guard_tool",
        }.issubset({trace.tool_id for trace in run.tool_call_trace_records})
        assert all(
            not tool_result.emits_raw_orders
            for response in run.orchestration_responses
            for tool_result in response.tool_results
        )
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_run_real_demand_backtest_script_emits_artifact_payload(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tmp_path = Path(".tmp_demand_backtest_script_tests") / uuid4().hex
    config_path = _write_temp_backtest_inputs(tmp_path)
    try:
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_real_demand_backtest.py",
                "--config",
                str(config_path),
                "--mode",
                "all",
                "--llm-client-mode",
                "fake",
            ],
        )

        run_real_demand_backtest_script.main()
        payload = json.loads(capsys.readouterr().out)

        assert payload["experiment_metadata"]["validation_lane"] == "real_demand_backtest"
        assert Path(payload["artifacts_dir"]).exists()
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_write_demand_backtest_panel_artifacts_preserves_slice_identity() -> None:
    tmp_path = Path(".tmp_demand_backtest_panel_tests") / uuid4().hex
    config_path = _write_temp_backtest_panel_inputs(tmp_path)
    try:
        panel_run = write_demand_backtest_panel_artifacts(
            config_path,
            llm_client_mode_override="fake",
        )

        assert panel_run.aggregate_summary is not None
        assert len(panel_run.slice_results) == 2
        assert all(result.success for result in panel_run.slice_results)
        assert {result.slice_name for result in panel_run.slice_results} == {"slice_a", "slice_b"}
        assert (panel_run.output_dir / "panel_summary.json").exists()
        assert (panel_run.output_dir / "aggregate_summary.json").exists()
        assert (panel_run.output_dir / "slices" / "slice_a" / "dataset_summary.json").exists()
        assert (panel_run.output_dir / "slices" / "slice_b" / "dataset_summary.json").exists()
        aggregate_payload = json.loads(
            (panel_run.output_dir / "aggregate_summary.json").read_text(encoding="utf-8")
        )
        assert aggregate_payload["validation_lane"] == "real_demand_backtest"
        episode_payload = json.loads(
            (panel_run.output_dir / "episode_summaries.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )
        assert episode_payload["optimizer_order_boundary_preserved"] is True
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_write_demand_backtest_panel_artifacts_supports_single_mode_override() -> None:
    tmp_path = Path(".tmp_demand_backtest_panel_mode_tests") / uuid4().hex
    config_path = _write_temp_backtest_panel_inputs(tmp_path)
    try:
        panel_run = write_demand_backtest_panel_artifacts(
            config_path,
            mode="robust_policy",
            llm_client_mode_override="fake",
        )

        assert panel_run.aggregate_summary is not None
        assert panel_run.aggregate_summary.mode_names == ("robust_policy",)
        assert all(result.success for result in panel_run.slice_results)
        aggregate_payload = json.loads(
            (panel_run.output_dir / "aggregate_summary.json").read_text(encoding="utf-8")
        )
        assert aggregate_payload["mode_names"] == ["robust_policy"]
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_run_real_demand_backtest_script_emits_panel_artifact_payload(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tmp_path = Path(".tmp_demand_backtest_panel_script_tests") / uuid4().hex
    config_path = _write_temp_backtest_panel_inputs(tmp_path)
    try:
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_real_demand_backtest.py",
                "--config",
                str(config_path),
                "--mode",
                "all",
                "--llm-client-mode",
                "fake",
            ],
        )

        run_real_demand_backtest_script.main()
        payload = json.loads(capsys.readouterr().out)

        assert payload["successful_slice_count"] == 2
        assert payload["failed_slice_count"] == 0
        assert payload["panel_config_hash"]
        assert Path(payload["artifacts_dir"]).exists()
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
