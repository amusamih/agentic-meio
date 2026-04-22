from __future__ import annotations

from importlib.util import find_spec
import json
from pathlib import Path
import shutil
from uuid import uuid4

import pytest

from meio.agents.llm_client import OpenAILLMClient
from meio.agents.prompts import PROMPT_VERSION, prompt_contract_hash
from meio.optimization.contracts import OptimizationResult
import scripts.run_stockpyl_serial as stockpyl_script
from scripts.run_stockpyl_serial import (
    run_stockpyl_serial,
    run_stockpyl_serial_batch,
    run_stockpyl_serial_comparison,
    run_stockpyl_serial_comparison_batch,
    run_stockpyl_serial_mode_sweep,
    write_stockpyl_serial_artifacts,
)


pytestmark = pytest.mark.skipif(find_spec("stockpyl") is None, reason="stockpyl not installed")


def test_run_stockpyl_serial_executes_primary_integration_path() -> None:
    benchmark_run = run_stockpyl_serial(mode="deterministic_orchestrator")

    assert benchmark_run.benchmark_case.adapter_name == "stockpyl_serial"
    assert benchmark_run.summary.benchmark_source == "stockpyl_serial"
    assert benchmark_run.summary.trace_summary.trace_length == 3
    assert benchmark_run.summary.intervention_summary.tool_call_count == 9
    assert benchmark_run.summary.optimizer_order_boundary_preserved is True
    assert benchmark_run.summary.episode_telemetry is not None
    assert benchmark_run.summary.episode_telemetry.provider is None
    assert benchmark_run.summary.decision_quality is not None
    assert benchmark_run.summary.decision_quality.regime_prediction_accuracy is None
    assert benchmark_run.episode_summary_record is not None
    assert benchmark_run.episode_summary_record.validation_lane == "stockpyl_internal"
    assert len(benchmark_run.step_trace_records) == benchmark_run.trace.trace_length
    assert len(benchmark_run.tool_call_trace_records) == 9
    assert benchmark_run.llm_call_trace_records == ()
    assert benchmark_run.summary.fill_rate is not None
    assert any(
        any(order > 0.0 for order in result.replenishment_orders)
        for result in benchmark_run.trace.optimization_results
    )


def test_run_stockpyl_serial_preserves_optimizer_only_order_boundary() -> None:
    benchmark_run = run_stockpyl_serial(mode="deterministic_orchestrator")

    assert all(
        tool_result.emits_raw_orders is False
        for response in benchmark_run.orchestration_responses
        for tool_result in response.tool_results
    )
    assert all(
        not any(isinstance(value, OptimizationResult) for value in tool_result.structured_output.values())
        for response in benchmark_run.orchestration_responses
        for tool_result in response.tool_results
    )


def test_run_stockpyl_serial_supports_deterministic_baseline_mode() -> None:
    benchmark_run = run_stockpyl_serial(mode="deterministic_baseline")

    assert benchmark_run.mode == "deterministic_baseline"
    assert benchmark_run.summary.trace_summary.trace_length == 3
    assert benchmark_run.summary.intervention_summary.tool_call_count == 0
    assert benchmark_run.summary.optimizer_order_boundary_preserved is True
    assert benchmark_run.summary.episode_telemetry is not None
    assert benchmark_run.summary.episode_telemetry.provider is None
    assert benchmark_run.summary.episode_telemetry.average_total_tokens is None
    assert benchmark_run.summary.decision_quality is not None
    assert benchmark_run.summary.decision_quality.no_action_rate > 0.0
    assert benchmark_run.episode_summary_record is not None
    assert benchmark_run.episode_summary_record.validation_lane == "stockpyl_internal"
    assert len(benchmark_run.step_trace_records) == benchmark_run.trace.trace_length
    assert benchmark_run.tool_call_trace_records == ()
    assert benchmark_run.llm_call_trace_records == ()


def test_run_stockpyl_serial_supports_llm_orchestrator_mode_with_fake_client() -> None:
    benchmark_run = run_stockpyl_serial(mode="llm_orchestrator")

    assert benchmark_run.mode == "llm_orchestrator"
    assert benchmark_run.llm_provider == "fake_llm_client"
    assert benchmark_run.summary.trace_summary.trace_length == 3
    assert benchmark_run.summary.optimizer_order_boundary_preserved is True
    assert benchmark_run.successful_response_count == 3
    assert benchmark_run.summary.episode_telemetry is not None
    assert benchmark_run.summary.episode_telemetry.provider == "fake_llm_client"
    assert benchmark_run.summary.episode_telemetry.average_total_tokens == 144.0
    assert benchmark_run.summary.decision_quality is not None
    assert benchmark_run.summary.decision_quality.regime_prediction_accuracy == 1.0
    assert dict(benchmark_run.summary.decision_quality.predicted_regime_counts)[
        "demand_regime_shift"
    ] == 1
    assert benchmark_run.prompt_version == PROMPT_VERSION
    assert benchmark_run.prompt_contract_hash == prompt_contract_hash()
    assert benchmark_run.validation_failure_counts == ()
    assert benchmark_run.episode_summary_record is not None
    assert benchmark_run.episode_summary_record.validation_lane == "stockpyl_internal"
    assert len(benchmark_run.step_trace_records) == benchmark_run.trace.trace_length
    assert len(benchmark_run.llm_call_trace_records) == benchmark_run.trace.trace_length
    assert len(benchmark_run.tool_call_trace_records) == 3
    assert all(
        tool_result.emits_raw_orders is False
        for response in benchmark_run.orchestration_responses
        for tool_result in response.tool_results
    )


def test_run_stockpyl_serial_comparison_now_shows_divergent_outcomes() -> None:
    comparison = run_stockpyl_serial_comparison()

    assert comparison.outcomes_differ is True
    assert comparison.optimizer_order_boundary_preserved is True
    assert (
        comparison.deterministic_orchestrator.summary.average_inventory
        != comparison.deterministic_baseline.summary.average_inventory
    )
    assert (
        comparison.llm_orchestrator.summary.average_inventory
        != comparison.deterministic_baseline.summary.average_inventory
    )


def test_run_stockpyl_serial_batch_aggregates_fake_llm_runs() -> None:
    batch = run_stockpyl_serial_batch(
        "configs/experiment/stockpyl_serial_live_llm.toml",
        mode="llm_orchestrator",
        llm_client_mode_override="fake",
    )

    assert batch.aggregate_summary.run_count == 2
    assert batch.aggregate_summary.llm_diagnostics is not None
    assert batch.aggregate_summary.llm_diagnostics.provider == "fake_llm_client"
    assert batch.aggregate_summary.llm_diagnostics.successful_response_count == 6
    assert batch.aggregate_summary.llm_diagnostics.prompt_version == PROMPT_VERSION
    assert batch.aggregate_summary.llm_diagnostics.prompt_hash == prompt_contract_hash()
    assert batch.aggregate_summary.llm_diagnostics.validation_failure_counts == ()
    assert batch.aggregate_summary.llm_diagnostics.client_error_counts == ()
    assert batch.aggregate_summary.llm_diagnostics.total_retry_count == 0
    assert batch.aggregate_summary.llm_diagnostics.average_total_tokens == 144.0
    assert batch.aggregate_summary.llm_diagnostics.average_llm_latency_ms is not None
    assert batch.aggregate_summary.decision_quality is not None
    assert batch.aggregate_summary.decision_quality.regime_prediction_accuracy == 1.0
    assert batch.aggregate_metrics is not None
    assert batch.aggregate_metrics.validation_lane == "stockpyl_internal"
    assert batch.aggregate_metrics.schedule_names == ("default",)
    assert batch.aggregate_metrics.seed_values == (20260417, 20260418)
    assert all(run.llm_provider == "fake_llm_client" for run in batch.runs)


def test_deterministic_batch_does_not_report_fake_llm_telemetry() -> None:
    batch = run_stockpyl_serial_batch(mode="deterministic_baseline")

    assert batch.aggregate_summary.llm_diagnostics is None
    assert batch.runs[0].summary.episode_telemetry is not None
    assert batch.runs[0].summary.episode_telemetry.provider is None


def test_run_stockpyl_serial_batch_fails_clearly_without_api_key_for_real_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(OpenAILLMClient, "resolve_api_key", lambda self: None)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        run_stockpyl_serial_batch(
            "configs/experiment/stockpyl_serial_live_llm.toml",
            mode="llm_orchestrator",
        )


def test_run_stockpyl_serial_comparison_batch_supports_repeated_runs() -> None:
    comparison = run_stockpyl_serial_comparison_batch(
        "configs/experiment/stockpyl_serial_live_llm.toml",
        llm_client_mode_override="fake",
    )

    assert comparison.outcomes_differ is True
    assert comparison.optimizer_order_boundary_preserved is True
    assert comparison.deterministic_baseline.aggregate_summary.run_count == 2
    assert comparison.deterministic_orchestrator.aggregate_summary.run_count == 2
    assert comparison.llm_orchestrator.aggregate_summary.run_count == 2
    assert comparison.aggregate_results is not None
    assert comparison.aggregate_results.mode_names == (
        "deterministic_baseline",
        "deterministic_orchestrator",
        "llm_orchestrator",
    )


def test_run_stockpyl_serial_batch_supports_max_runs_limit() -> None:
    batch = run_stockpyl_serial_batch(
        "configs/experiment/stockpyl_serial_multi_eval.toml",
        mode="llm_orchestrator",
        llm_client_mode_override="fake",
        max_runs=2,
    )

    assert batch.aggregate_summary.run_count == 2
    assert len(batch.runs) == 2
    assert {(run.schedule_name, run.run_seed) for run in batch.runs} == {
        ("shift_recovery", 20260417),
        ("shift_recovery", 20260418),
    }


def test_run_stockpyl_serial_batch_supports_tool_ablation_variants() -> None:
    batch = run_stockpyl_serial_batch(
        "configs/experiment/stockpyl_serial_tool_ablation.toml",
        mode="llm_orchestrator",
        llm_client_mode_override="fake",
        max_runs=4,
    )

    assert batch.aggregate_metrics is not None
    assert batch.aggregate_metrics.tool_ablation_variants == (
        "full",
        "no_forecast_tool",
        "no_leadtime_tool",
        "no_scenario_tool",
    )
    assert batch.tool_usage_by_ablation
    assert any(
        summary.tool_ablation_variant == "full"
        for summary in batch.tool_usage_by_ablation
    )
    no_forecast_summary = next(
        summary
        for summary in batch.tool_ablation_summaries
        if summary.tool_ablation_variant == "no_forecast_tool"
    )
    assert no_forecast_summary.unavailable_tool_request_count == 0
    assert no_forecast_summary.disabled_tool_fallback_count == 0


def test_run_stockpyl_serial_tool_ablation_disables_requested_tool_safely() -> None:
    batch = run_stockpyl_serial_batch(
        "configs/experiment/stockpyl_serial_tool_ablation.toml",
        mode="llm_orchestrator",
        llm_client_mode_override="fake",
        max_runs=6,
    )

    no_forecast_run = next(
        run for run in batch.runs if run.tool_ablation_variant == "no_forecast_tool"
    )
    assert all(trace.tool_id != "forecast_tool" for trace in no_forecast_run.tool_call_trace_records)
    assert no_forecast_run.summary.optimizer_order_boundary_preserved is True
    assert no_forecast_run.episode_summary_record is not None
    assert no_forecast_run.episode_summary_record.unavailable_tool_request_count == 0
    assert no_forecast_run.episode_summary_record.disabled_tool_fallback_count == 0
    assert no_forecast_run.episode_summary_record.sequencing_blocked_tool_request_count == 0


def test_run_stockpyl_serial_tool_ablation_keeps_scenario_tool_admissible_without_leadtime_tool() -> None:
    batch = run_stockpyl_serial_batch(
        "configs/experiment/stockpyl_serial_tool_ablation.toml",
        mode="llm_orchestrator",
        llm_client_mode_override="fake",
        max_runs=6,
    )

    no_leadtime_run = next(
        run for run in batch.runs if run.tool_ablation_variant == "no_leadtime_tool"
    )

    assert any(
        trace.tool_id == "scenario_tool" for trace in no_leadtime_run.tool_call_trace_records
    )
    assert no_leadtime_run.episode_summary_record is not None
    assert no_leadtime_run.episode_summary_record.sequencing_blocked_tool_request_count == 0


def test_run_stockpyl_serial_comparison_batch_keeps_schedule_seed_pairs_aligned() -> None:
    comparison = run_stockpyl_serial_comparison_batch(
        "configs/experiment/stockpyl_serial_multi_eval.toml",
        llm_client_mode_override="fake",
    )

    baseline_pairs = {
        (run.schedule_name, run.run_seed) for run in comparison.deterministic_baseline.runs
    }
    deterministic_pairs = {
        (run.schedule_name, run.run_seed)
        for run in comparison.deterministic_orchestrator.runs
    }
    llm_pairs = {(run.schedule_name, run.run_seed) for run in comparison.llm_orchestrator.runs}

    assert len(baseline_pairs) == 8
    assert baseline_pairs == deterministic_pairs == llm_pairs
    assert comparison.schedule_names == (
        "shift_recovery",
        "sustained_shift",
        "recovery_false_alarm",
        "long_shift_recovery",
    )
    assert comparison.seed_values == (20260417, 20260418)


def test_paper_candidate_batch_config_keeps_modes_and_full_tool_set_aligned() -> None:
    comparison = run_stockpyl_serial_comparison_batch(
        "configs/experiment/stockpyl_serial_paper_candidate.toml",
        llm_client_mode_override="fake",
    )

    baseline_pairs = {
        (run.schedule_name, run.run_seed, run.tool_ablation_variant)
        for run in comparison.deterministic_baseline.runs
    }
    deterministic_pairs = {
        (run.schedule_name, run.run_seed, run.tool_ablation_variant)
        for run in comparison.deterministic_orchestrator.runs
    }
    llm_pairs = {
        (run.schedule_name, run.run_seed, run.tool_ablation_variant)
        for run in comparison.llm_orchestrator.runs
    }

    assert baseline_pairs == deterministic_pairs == llm_pairs
    assert {variant for _, _, variant in baseline_pairs} == {"full"}
    assert comparison.schedule_names == (
        "shift_recovery",
        "sustained_shift",
        "recovery_false_alarm",
        "long_shift_recovery",
    )
    assert comparison.seed_values == (20260417, 20260418)


def test_heldout_batch_config_keeps_frozen_full_tool_path_aligned() -> None:
    comparison = run_stockpyl_serial_comparison_batch(
        "configs/experiment/stockpyl_serial_heldout_eval.toml",
        llm_client_mode_override="fake",
    )

    baseline_pairs = {
        (run.schedule_name, run.run_seed, run.tool_ablation_variant)
        for run in comparison.deterministic_baseline.runs
    }
    deterministic_pairs = {
        (run.schedule_name, run.run_seed, run.tool_ablation_variant)
        for run in comparison.deterministic_orchestrator.runs
    }
    llm_pairs = {
        (run.schedule_name, run.run_seed, run.tool_ablation_variant)
        for run in comparison.llm_orchestrator.runs
    }

    assert baseline_pairs == deterministic_pairs == llm_pairs
    assert {variant for _, _, variant in baseline_pairs} == {"full"}
    assert len(baseline_pairs) == 10
    assert comparison.schedule_names == (
        "delayed_shift_recovery",
        "delayed_sustained_shift",
        "double_shift_with_gap",
        "recovery_then_relapse",
        "false_alarm_then_real_shift",
    )
    assert comparison.seed_values == (20260417, 20260418)
    assert comparison.llm_orchestrator.aggregate_summary.llm_diagnostics is not None
    assert comparison.llm_orchestrator.aggregate_summary.llm_diagnostics.prompt_version == PROMPT_VERSION


def test_frozen_broad_eval_batch_config_keeps_frozen_full_tool_path_aligned() -> None:
    comparison = run_stockpyl_serial_comparison_batch(
        "configs/experiment/stockpyl_serial_frozen_broad_eval.toml",
        llm_client_mode_override="fake",
        max_runs=6,
    )

    baseline_pairs = {
        (run.schedule_name, run.run_seed, run.tool_ablation_variant)
        for run in comparison.deterministic_baseline.runs
    }
    deterministic_pairs = {
        (run.schedule_name, run.run_seed, run.tool_ablation_variant)
        for run in comparison.deterministic_orchestrator.runs
    }
    llm_pairs = {
        (run.schedule_name, run.run_seed, run.tool_ablation_variant)
        for run in comparison.llm_orchestrator.runs
    }

    assert baseline_pairs == deterministic_pairs == llm_pairs
    assert {variant for _, _, variant in baseline_pairs} == {"full"}
    assert comparison.llm_orchestrator.aggregate_summary.llm_diagnostics is not None
    assert comparison.llm_orchestrator.aggregate_summary.llm_diagnostics.prompt_version == PROMPT_VERSION
    assert comparison.llm_orchestrator.aggregate_summary.llm_diagnostics.prompt_hash == prompt_contract_hash()


def test_paper_candidate_fake_llm_run_records_repeated_stress_moderation_traceability() -> None:
    batch = run_stockpyl_serial_batch(
        "configs/experiment/stockpyl_serial_paper_candidate.toml",
        mode="llm_orchestrator",
        llm_client_mode_override="fake",
    )

    assert any(
        record.repeated_stress_moderation_applied for run in batch.runs for record in run.step_trace_records
    )
    moderated_record = next(
        record
        for run in batch.runs
        for record in run.step_trace_records
        if record.repeated_stress_moderation_applied
    )
    assert moderated_record.proposed_update_requests == (
        "switch_demand_regime",
        "widen_uncertainty",
    )
    assert moderated_record.proposed_update_strength == (
        "switch_demand_regime_plus_widen_uncertainty"
    )
    assert moderated_record.final_update_strength == "reweight_scenarios"
    assert moderated_record.calibration_applied is True
    assert moderated_record.hysteresis_applied is True
    assert moderated_record.update_requests == ("reweight_scenarios",)
    moderated_run = next(
        run
        for run in batch.runs
        if run.episode_summary_record is not None
        and run.episode_summary_record.repeated_stress_moderation_count > 0
    )
    assert moderated_run.episode_summary_record is not None
    assert moderated_run.episode_summary_record.moderated_update_count > 0
    assert moderated_run.episode_summary_record.hysteresis_application_count > 0
    assert moderated_run.episode_summary_record.relapse_moderation_count == 0
    assert any(
        trace.repeated_stress_moderation_applied
        and trace.calibration_applied
        and trace.hysteresis_applied
        and trace.proposed_update_strength
        == "switch_demand_regime_plus_widen_uncertainty"
        and trace.final_update_strength == "reweight_scenarios"
        and trace.final_update_request_types == ("reweight_scenarios",)
        for trace in moderated_run.llm_call_trace_records
    )


def test_heldout_fake_llm_run_records_relapse_moderation_traceability() -> None:
    batch = run_stockpyl_serial_batch(
        "configs/experiment/stockpyl_serial_heldout_eval.toml",
        mode="llm_orchestrator",
        llm_client_mode_override="fake",
        max_runs=8,
    )

    moderated_run = next(
        run
        for run in batch.runs
        if run.schedule_name in {"double_shift_with_gap", "recovery_then_relapse"}
        and run.episode_summary_record is not None
        and run.episode_summary_record.relapse_moderation_count > 0
    )
    assert moderated_run.episode_summary_record is not None
    assert moderated_run.episode_summary_record.unresolved_stress_moderation_count > 0
    moderated_step = next(
        record
        for record in moderated_run.step_trace_records
        if record.relapse_moderation_applied
    )
    assert moderated_step.proposed_update_requests == (
        "switch_demand_regime",
        "widen_uncertainty",
    )
    assert moderated_step.proposed_update_strength == (
        "switch_demand_regime_plus_widen_uncertainty"
    )
    assert moderated_step.final_update_strength == "reweight_scenarios"
    assert moderated_step.calibration_applied is True
    assert moderated_step.hysteresis_applied is True
    assert moderated_step.update_requests == ("reweight_scenarios",)
    assert moderated_step.unresolved_stress_moderation_applied is True
    assert moderated_step.moderation_reason == "relapse_with_unresolved_stress_load"
    assert any(
        trace.relapse_moderation_applied
        and trace.calibration_applied
        and trace.hysteresis_applied
        and trace.proposed_update_strength
        == "switch_demand_regime_plus_widen_uncertainty"
        and trace.final_update_strength == "reweight_scenarios"
        and trace.final_update_request_types == ("reweight_scenarios",)
        for trace in moderated_run.llm_call_trace_records
    )


def test_write_paper_candidate_artifacts_marks_real_modes_as_paper_candidates() -> None:
    output_root = Path(".tmp_stockpyl_artifact_tests") / uuid4().hex
    try:
        metadata, _, written_files = write_stockpyl_serial_artifacts(
            "configs/experiment/stockpyl_serial_paper_candidate.toml",
            mode="deterministic_orchestrator",
            output_dir_override=output_root,
        )

        assert metadata.artifact_use_class.value == "paper_candidate"
        assert metadata.rollout_fidelity_gate_passed is True
        assert metadata.operational_metrics_gate_passed is True
        manifest_payload = json.loads(
            written_files["run_manifest"].read_text(encoding="utf-8")
        )
        assert manifest_payload["artifact_use_class"] == "paper_candidate"
        assert manifest_payload["tool_ablation_variants"] == ["full"]
        assert manifest_payload["mode_names"] == ["deterministic_orchestrator"]

        aggregate_payload = json.loads(
            written_files["aggregate_summary"].read_text(encoding="utf-8")
        )
        assert aggregate_payload["artifact_use_class"] == "paper_candidate"
        assert aggregate_payload["tool_ablation_variants"] == ["full"]
        assert aggregate_payload["mode_summaries"][0]["artifact_use_class"] == "paper_candidate"
    finally:
        if output_root.exists():
            shutil.rmtree(output_root, ignore_errors=True)


def test_write_stockpyl_serial_artifacts_writes_deterministic_files() -> None:
    output_root = Path(".tmp_stockpyl_artifact_tests") / uuid4().hex
    try:
        metadata, output_dir, written_files = write_stockpyl_serial_artifacts(
            mode="deterministic_orchestrator",
            output_dir_override=output_root,
        )

        assert metadata.mode == "deterministic_orchestrator"
        assert metadata.artifact_use_class.value == "paper_candidate"
        assert metadata.validity_gate_passed is True
        assert metadata.rollout_fidelity_gate_passed is True
        assert metadata.operational_metrics_gate_passed is True
        assert output_dir.exists()
        assert set(written_files) == {
            "aggregate_summary",
            "experiment_metadata",
            "episode_summaries",
            "run_manifest",
            "step_traces",
            "llm_call_traces",
            "tool_call_traces",
        }
        assert written_files["experiment_metadata"].exists()
        assert written_files["run_manifest"].exists()
        assert written_files["episode_summaries"].read_text(encoding="utf-8").strip()
        assert written_files["step_traces"].read_text(encoding="utf-8").strip()
        assert written_files["llm_call_traces"].read_text(encoding="utf-8") == ""
        assert written_files["tool_call_traces"].read_text(encoding="utf-8").strip()
        aggregate_payload = json.loads(
            written_files["aggregate_summary"].read_text(encoding="utf-8")
        )
        assert aggregate_payload["benchmark_id"] == "serial_3_echelon"
        assert aggregate_payload["artifact_use_class"] == "paper_candidate"
        assert aggregate_payload["validity_gate_passed"] is True
        assert aggregate_payload["mode_summaries"][0]["validity_summary"][
            "optimizer_order_boundary_preserved"
        ] is True
        assert aggregate_payload["mode_summaries"][0]["artifact_use_class"] == "paper_candidate"
        manifest_payload = json.loads(
            written_files["run_manifest"].read_text(encoding="utf-8")
        )
        assert manifest_payload["artifact_use_class"] == "paper_candidate"
        assert manifest_payload["validity_gate_passed"] is True
        assert manifest_payload["rollout_fidelity_gate_passed"] is True
        assert manifest_payload["operational_metrics_gate_passed"] is True
        assert manifest_payload["artifact_filenames"][-1] == [
            "run_manifest",
            "run_manifest.json",
        ]
    finally:
        if output_root.exists():
            shutil.rmtree(output_root, ignore_errors=True)


def test_write_stockpyl_serial_artifacts_writes_fake_llm_trace_rows() -> None:
    output_root = Path(".tmp_stockpyl_artifact_tests") / uuid4().hex
    try:
        metadata, output_dir, written_files = write_stockpyl_serial_artifacts(
            "configs/experiment/stockpyl_serial_live_llm.toml",
            mode="llm_orchestrator",
            llm_client_mode_override="fake",
            output_dir_override=output_root,
        )

        assert metadata.mode == "llm_orchestrator"
        assert metadata.provider == "fake_llm_client"
        assert metadata.artifact_use_class.value == "internal_only"
        assert "fake_llm_internal_only" in metadata.eligibility_notes
        assert metadata.prompt_version == PROMPT_VERSION
        assert metadata.prompt_hash == prompt_contract_hash()
        assert output_dir.exists()
        llm_trace_lines = written_files["llm_call_traces"].read_text(encoding="utf-8").splitlines()
        assert llm_trace_lines
        first_llm_trace = json.loads(llm_trace_lines[0])
        assert first_llm_trace["prompt_version"] == PROMPT_VERSION
        episode_lines = written_files["episode_summaries"].read_text(encoding="utf-8").splitlines()
        assert episode_lines
        first_episode = json.loads(episode_lines[0])
        assert first_episode["regime_prediction_accuracy"] == 1.0
        assert first_episode["schedule_name"] == "default"
        assert first_episode["run_seed"] == 20260417
        manifest_payload = json.loads(written_files["run_manifest"].read_text(encoding="utf-8"))
        assert manifest_payload["mode_names"] == ["llm_orchestrator"]
        assert manifest_payload["provider"] == "fake_llm_client"
        assert manifest_payload["artifact_use_class"] == "internal_only"
        assert "fake_llm_internal_only" in manifest_payload["eligibility_notes"]
        assert written_files["tool_call_traces"].read_text(encoding="utf-8").strip()
    finally:
        if output_root.exists():
            shutil.rmtree(output_root, ignore_errors=True)


def test_write_stockpyl_serial_artifacts_populates_git_sha_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = Path(".tmp_stockpyl_artifact_tests") / uuid4().hex
    monkeypatch.setattr(stockpyl_script, "_git_commit_sha", lambda: "abc123")
    try:
        metadata, _, _ = write_stockpyl_serial_artifacts(
            "configs/experiment/stockpyl_serial_live_llm.toml",
            mode="llm_orchestrator",
            llm_client_mode_override="fake",
            output_dir_override=output_root,
        )

        assert metadata.git_commit_sha == "abc123"
    finally:
        if output_root.exists():
            shutil.rmtree(output_root, ignore_errors=True)


def test_write_stockpyl_serial_artifacts_writes_multi_schedule_aggregate_summary() -> None:
    output_root = Path(".tmp_stockpyl_artifact_tests") / uuid4().hex
    try:
        _, _, written_files = write_stockpyl_serial_artifacts(
            "configs/experiment/stockpyl_serial_multi_eval.toml",
            mode="all",
            llm_client_mode_override="fake",
            output_dir_override=output_root,
        )

        aggregate_payload = json.loads(
            written_files["aggregate_summary"].read_text(encoding="utf-8")
        )
        assert aggregate_payload["schedule_names"] == [
            "long_shift_recovery",
            "recovery_false_alarm",
            "shift_recovery",
            "sustained_shift",
        ]
        assert aggregate_payload["seed_values"] == [20260417, 20260418]
        assert aggregate_payload["mode_summaries"][0]["robustness_summary"]["schedule_count"] == 4
        manifest_payload = json.loads(written_files["run_manifest"].read_text(encoding="utf-8"))
        assert manifest_payload["mode_names"] == [
            "deterministic_baseline",
            "deterministic_orchestrator",
            "llm_orchestrator",
        ]
    finally:
        if output_root.exists():
            shutil.rmtree(output_root, ignore_errors=True)


def test_write_stockpyl_serial_artifacts_writes_tool_ablation_metadata() -> None:
    output_root = Path(".tmp_stockpyl_artifact_tests") / uuid4().hex
    try:
        metadata, _, written_files = write_stockpyl_serial_artifacts(
            "configs/experiment/stockpyl_serial_tool_ablation.toml",
            mode="llm_orchestrator",
            llm_client_mode_override="fake",
            output_dir_override=output_root,
            max_runs=4,
        )

        assert metadata.tool_ablation_variants == (
            "full",
            "no_forecast_tool",
            "no_leadtime_tool",
            "no_scenario_tool",
        )
        aggregate_payload = json.loads(
            written_files["aggregate_summary"].read_text(encoding="utf-8")
        )
        assert aggregate_payload["tool_ablation_variants"] == [
            "full",
            "no_forecast_tool",
            "no_leadtime_tool",
            "no_scenario_tool",
        ]
        manifest_payload = json.loads(
            written_files["run_manifest"].read_text(encoding="utf-8")
        )
        assert manifest_payload["tool_ablation_variants"] == [
            "full",
            "no_forecast_tool",
            "no_leadtime_tool",
            "no_scenario_tool",
        ]
    finally:
        if output_root.exists():
            shutil.rmtree(output_root, ignore_errors=True)


def test_external_evidence_mode_sweep_preserves_control_and_branch_pairing() -> None:
    sweep = run_stockpyl_serial_mode_sweep(
        "configs/experiment/stockpyl_serial_external_evidence.toml",
        llm_client_mode_override="fake",
        max_runs=2,
    )

    assert tuple(batch.mode for batch in sweep.mode_batches) == (
        "deterministic_baseline",
        "deterministic_orchestrator",
        "llm_orchestrator_internal_only",
        "llm_orchestrator_with_external_evidence",
    )
    run_pairs_by_mode = {
        batch.mode: {(run.schedule_name, run.run_seed) for run in batch.runs}
        for batch in sweep.mode_batches
    }
    expected_pairs = {
        ("shift_recovery", 20260417),
        ("shift_recovery", 20260418),
    }
    assert all(pairs == expected_pairs for pairs in run_pairs_by_mode.values())
    external_batch = next(
        batch for batch in sweep.mode_batches if batch.mode == "llm_orchestrator_with_external_evidence"
    )
    assert external_batch.aggregate_metrics is not None
    assert external_batch.aggregate_metrics.external_evidence_summary is not None
    assert (
        external_batch.aggregate_metrics.external_evidence_summary.external_evidence_period_count
        > 0
    )


def test_write_external_evidence_artifacts_marks_branch_internal_only() -> None:
    output_root = Path(".tmp_stockpyl_artifact_tests") / uuid4().hex
    try:
        metadata, _, written_files = write_stockpyl_serial_artifacts(
            "configs/experiment/stockpyl_serial_external_evidence.toml",
            mode="all",
            llm_client_mode_override="fake",
            output_dir_override=output_root,
            max_runs=2,
        )

        assert metadata.artifact_use_class.value == "internal_only"
        assert "semi_synthetic_external_evidence_branch" in metadata.eligibility_notes
        aggregate_payload = json.loads(
            written_files["aggregate_summary"].read_text(encoding="utf-8")
        )
        mode_names = [item["mode"] for item in aggregate_payload["mode_summaries"]]
        assert mode_names == [
            "deterministic_baseline",
            "deterministic_orchestrator",
            "llm_orchestrator_internal_only",
            "llm_orchestrator_with_external_evidence",
        ]
        external_mode = next(
            item
            for item in aggregate_payload["mode_summaries"]
            if item["mode"] == "llm_orchestrator_with_external_evidence"
        )
        assert external_mode["artifact_use_class"] == "internal_only"
        assert external_mode["external_evidence_summary"]["external_evidence_period_count"] > 0
    finally:
        if output_root.exists():
            shutil.rmtree(output_root, ignore_errors=True)


def test_external_evidence_fake_run_exposes_early_confirmation_gate_fields() -> None:
    batch = run_stockpyl_serial_batch(
        "configs/experiment/stockpyl_serial_external_evidence.toml",
        mode="llm_orchestrator_with_external_evidence",
        llm_client_mode_override="fake",
        max_runs=2,
    )

    assert batch.aggregate_metrics is not None
    assert batch.aggregate_metrics.external_evidence_summary is not None
    assert (
        batch.aggregate_metrics.external_evidence_summary.early_evidence_confirmation_gate_count
        >= 0
    )
    assert (
        batch.aggregate_metrics.external_evidence_summary.early_evidence_family_change_block_count
        >= 0
    )
    assert all(
        record.early_evidence_confirmation_gate_applied in {True, False}
        and record.early_evidence_family_change_blocked in {True, False}
        and isinstance(record.proposed_external_update_requests, tuple)
        and isinstance(record.final_external_update_requests, tuple)
        for run in batch.runs
        for record in run.step_trace_records
    )
