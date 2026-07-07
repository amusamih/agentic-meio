from __future__ import annotations

import json
import shutil
from pathlib import Path
from uuid import uuid4

from scripts.run_stockpyl_serial import (
    OFFICIAL_COMPARISON_MODES,
    REGRET_GUARDED_RISK_SENSITIVE_SCENARIO_PLANNER_ORCHESTRATOR_MODE,
    _build_mission,
    _candidate_tool_ids_for_mode,
    run_stockpyl_serial,
    run_stockpyl_serial_batch,
    run_stockpyl_serial_mode_sweep,
    write_stockpyl_serial_artifacts,
)


CONFIG_PATH = Path("configs/experiment/stockpyl_serial_realistic_comparison.toml")
SPREAD_CONFIG_PATH = Path("configs/experiment/stockpyl_serial_spread_sensitivity.toml")


def _batch_by_mode(mode_sweep, mode: str):
    return next(batch for batch in mode_sweep.mode_batches if batch.mode == mode)


def test_run_stockpyl_serial_supports_current_agentic_mode_with_fake_client() -> None:
    benchmark_run = run_stockpyl_serial(
        CONFIG_PATH,
        mode=REGRET_GUARDED_RISK_SENSITIVE_SCENARIO_PLANNER_ORCHESTRATOR_MODE,
        llm_client_mode_override="fake",
    )

    assert benchmark_run.mode == REGRET_GUARDED_RISK_SENSITIVE_SCENARIO_PLANNER_ORCHESTRATOR_MODE
    assert benchmark_run.summary.optimizer_order_boundary_preserved is True
    assert benchmark_run.summary.episode_telemetry is not None
    assert benchmark_run.summary.episode_telemetry.total_tool_call_count >= 0
    assert any(
        trace.tool_id == "counterfactual_regret_guard_tool"
        for trace in benchmark_run.tool_call_trace_records
    )


def test_agentic_tool_ablation_variants_resolve_coherent_sequences() -> None:
    assert _candidate_tool_ids_for_mode("full") == (
        "regime_diagnosis_tool",
        "demand_uncertainty_decomposition_tool",
        "regime_belief_tool",
        "scenario_candidate_generator_tool",
        "risk_sensitive_scenario_evaluator_tool",
        "counterfactual_regret_guard_tool",
    )
    assert _candidate_tool_ids_for_mode("without_demand_decomposition") == (
        "regime_diagnosis_tool",
        "regime_belief_tool",
        "scenario_candidate_generator_tool",
        "risk_sensitive_scenario_evaluator_tool",
        "counterfactual_regret_guard_tool",
    )
    assert _candidate_tool_ids_for_mode("without_risk_sensitive_evaluation") == (
        "regime_diagnosis_tool",
        "demand_uncertainty_decomposition_tool",
        "regime_belief_tool",
        "scenario_candidate_generator_tool",
        "scenario_evaluator_tool",
    )
    assert _candidate_tool_ids_for_mode("without_regret_guard") == (
        "regime_diagnosis_tool",
        "demand_uncertainty_decomposition_tool",
        "regime_belief_tool",
        "scenario_candidate_generator_tool",
        "risk_sensitive_scenario_evaluator_tool",
    )


def test_agentic_tool_ablation_mission_budget_matches_explicit_sequence() -> None:
    mission = _build_mission("without_risk_sensitive_evaluation")

    assert mission.admissible_tool_ids[-1] == "scenario_evaluator_tool"
    assert mission.max_tool_steps == len(mission.admissible_tool_ids)


def test_non_llm_uncertainty_baselines_do_not_emit_llm_call_traces() -> None:
    for mode in ("robust_policy", "scenario_rolling_horizon_policy"):
        benchmark_run = run_stockpyl_serial(CONFIG_PATH, mode=mode)

        assert benchmark_run.mode == mode
        assert benchmark_run.summary.optimizer_order_boundary_preserved is True
        assert benchmark_run.llm_call_trace_records == ()


def test_run_stockpyl_serial_mode_sweep_uses_official_mode_set() -> None:
    mode_sweep = run_stockpyl_serial_mode_sweep(
        CONFIG_PATH,
        llm_client_mode_override="fake",
        max_runs=1,
    )

    assert tuple(batch.mode for batch in mode_sweep.mode_batches) == OFFICIAL_COMPARISON_MODES
    assert mode_sweep.optimizer_order_boundary_preserved is True
    assert _batch_by_mode(mode_sweep, "robust_policy").runs[0].llm_call_trace_records == ()
    assert (
        _batch_by_mode(
            mode_sweep,
            REGRET_GUARDED_RISK_SENSITIVE_SCENARIO_PLANNER_ORCHESTRATOR_MODE,
        ).runs[0].llm_call_trace_records
        != ()
    )


def test_spread_sensitivity_profile_preserves_seed_aggregate_mean() -> None:
    batch = run_stockpyl_serial_batch(
        SPREAD_CONFIG_PATH,
        mode="deterministic_baseline",
        max_runs=3,
    )
    realized_demands = tuple(
        record.realized_demand
        for run in batch.runs
        for record in run.trace.period_records
        if record.realized_demand is not None
    )
    elevated_spread_demands = tuple(
        record.realized_demand
        for run in batch.runs
        for record in run.trace.period_records
        if record.regime_label.value == "demand_regime_shift"
        and record.realized_demand is not None
    )

    assert len(batch.runs) == 3
    assert {run.schedule_name for run in batch.runs} == {"spread_recovery"}
    assert sum(realized_demands) / len(realized_demands) == 10.0
    assert sorted(elevated_spread_demands) == [2.0, 10.0, 18.0]


def test_write_stockpyl_serial_artifacts_writes_current_manifest() -> None:
    output_dir = Path(".tmp_stockpyl_artifacts") / uuid4().hex
    try:
        metadata, artifacts_dir, written_files = write_stockpyl_serial_artifacts(
            CONFIG_PATH,
            mode="all",
            llm_client_mode_override="fake",
            max_runs=1,
            output_dir_override=output_dir,
        )

        assert metadata.mode == "all"
        assert artifacts_dir.parent == output_dir
        manifest = json.loads(written_files["run_manifest"].read_text(encoding="utf-8"))
        assert set(manifest["mode_names"]) == set(OFFICIAL_COMPARISON_MODES)
        assert REGRET_GUARDED_RISK_SENSITIVE_SCENARIO_PLANNER_ORCHESTRATOR_MODE in manifest["mode_names"]
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
