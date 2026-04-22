from __future__ import annotations

import json
import shutil
from pathlib import Path
from uuid import uuid4

from meio.evaluation.logging_io import (
    ensure_output_dir,
    write_episode_summaries_jsonl,
    write_experiment_metadata_json,
    write_llm_call_traces_jsonl,
    write_run_manifest_json,
    write_step_traces_jsonl,
    write_tool_call_traces_jsonl,
)
from meio.agents.prompts import PROMPT_VERSION
from meio.evaluation.logging_schema import (
    ArtifactUseClass,
    CostBreakdownRecord,
    EpisodeSummaryRecord,
    ExperimentMetadata,
    LLMCallTraceRecord,
    RunManifestRecord,
    StepTraceRecord,
    ToolCallTraceRecord,
)


def test_logging_io_writes_json_and_jsonl_artifacts() -> None:
    scratch_dir = Path(".tmp_logging_io_tests") / uuid4().hex
    try:
        output_dir = ensure_output_dir(scratch_dir, "logging_test_group")
        metadata = ExperimentMetadata(
            experiment_id="stockpyl_logging_all",
            run_group_id="logging_test_group",
            timestamp="2026-04-17T10:00:00Z",
            git_commit_sha=None,
            resolved_config={"mode": "all"},
            config_hash="config_hash",
            benchmark_id="serial_3_echelon",
            benchmark_source="stockpyl_serial",
            mode="all",
            provider="mixed",
            model_name="mixed",
            tool_ablation_variants=("full",),
            artifact_use_class=ArtifactUseClass.INTERNAL_ONLY,
            validity_gate_passed=False,
            eligibility_notes=("provisional_stockpyl_rollout",),
        )
        episode = EpisodeSummaryRecord(
            episode_id="episode_1",
            mode="deterministic_baseline",
            benchmark_id="serial_3_echelon",
            topology="serial",
            echelon_count=3,
            tool_ablation_variant="full",
            schedule_name="shift_recovery",
            run_seed=20260417,
            regime_schedule=("normal", "demand_regime_shift", "recovery"),
            total_cost=10.0,
            cost_breakdown=CostBreakdownRecord(
                holding_cost=4.0,
                backlog_cost=3.0,
                ordering_cost=3.0,
            ),
            fill_rate=1.0,
            average_inventory=20.0,
            average_backorder_level=0.0,
            step_count=3,
            replan_count=1,
            intervention_count=1,
            abstain_count=0,
            no_action_count=1,
            tool_call_count=0,
            invalid_output_count=0,
            fallback_count=0,
            total_prompt_tokens=None,
            total_completion_tokens=None,
            total_tokens=None,
            total_llm_latency_ms=None,
            total_orchestration_latency_ms=1.0,
            optimizer_order_boundary_preserved=True,
            client_error_counts=(("network_error", 1),),
            total_retry_count=1,
            failure_before_response_count=1,
            failure_after_response_count=0,
        )
        step = StepTraceRecord(
            episode_id="episode_1",
            mode="deterministic_baseline",
            tool_ablation_variant="full",
            schedule_name="shift_recovery",
            run_seed=20260417,
            period_index=0,
            true_regime_label="normal",
            predicted_regime_label=None,
            confidence=None,
            selected_subgoal="no_action",
            selected_tools=(),
            update_requests=("keep_current",),
            request_replan=False,
            abstain_or_no_action=True,
            demand_outlook=10.0,
            leadtime_outlook=2.0,
            scenario_adjustment_summary={
                "demand_outlook": 10.0,
                "leadtime_outlook": 2.0,
                "safety_buffer_scale": 1.0,
            },
            optimizer_orders=(0.0, 0.0, 0.0),
            inventory_by_echelon=(20.0, 30.0, 40.0),
            pipeline_by_echelon=(0.0, 0.0, 0.0),
            backorders_by_echelon=(0.0, 0.0, 0.0),
            per_period_cost=5.0,
            per_period_fill_rate=1.0,
            decision_changed_optimizer_input=False,
            optimizer_output_changed_state=False,
            intervention_changed_outcome=False,
        )
        llm_call = LLMCallTraceRecord(
            episode_id="episode_2",
            mode="llm_orchestrator",
            tool_ablation_variant="full",
            schedule_name="shift_recovery",
            run_seed=20260417,
            period_index=0,
            call_index=0,
            provider="fake_llm_client",
            model_name="gpt-4o-mini",
            prompt_version=PROMPT_VERSION,
            prompt_text="SYSTEM:\nBounded orchestration only.",
            prompt_hash="prompt_hash",
            raw_output_text='{"request_replan":false}',
            parsed_output={"request_replan": False},
            validation_success=True,
            invalid_output=False,
            fallback_used=False,
            fallback_reason=None,
            requested_tool_ids=("leadtime_tool", "scenario_tool"),
            unavailable_tool_ids=(),
            violated_available_tool_set=False,
            prompt_tokens=96,
            completion_tokens=48,
            total_tokens=144,
            latency_ms=5.0,
            client_error_category="network_error",
            client_error_message="APIConnectionError: Connection error.",
            failure_after_response=False,
        )
        tool_call = ToolCallTraceRecord(
            episode_id="episode_1",
            mode="deterministic_orchestrator",
            tool_ablation_variant="full",
            schedule_name="shift_recovery",
            run_seed=20260417,
            period_index=0,
            call_index=0,
            tool_id="forecast_tool",
            tool_input={"tool_id": "forecast_tool"},
            tool_output={"status": "success"},
            success=True,
            error_type=None,
            latency_ms=1.0,
        )
        manifest = RunManifestRecord(
            experiment_id="stockpyl_logging_all",
            run_group_id="logging_test_group",
            config_hash="config_hash",
            benchmark_id="serial_3_echelon",
            benchmark_source="stockpyl_serial",
            benchmark_config_path="configs/benchmark/serial_3_echelon.toml",
            agent_config_path="configs/agent/live_llm.toml",
            schedule_names=("shift_recovery",),
            seed_values=(20260417,),
            mode_names=("deterministic_baseline",),
            tool_ablation_variants=("full",),
            provider=None,
            model_name=None,
            artifact_use_class=ArtifactUseClass.INTERNAL_ONLY,
            validity_gate_passed=False,
            eligibility_notes=("provisional_stockpyl_rollout",),
            mode_artifact_use_classes=(("deterministic_baseline", "internal_only"),),
            prompt_version=None,
            prompt_hash=None,
            artifact_filenames=(
                ("experiment_metadata", "experiment_metadata.json"),
                ("run_manifest", "run_manifest.json"),
            ),
        )

        metadata_path = write_experiment_metadata_json(output_dir, metadata)
        episode_path = write_episode_summaries_jsonl(output_dir, (episode,))
        step_path = write_step_traces_jsonl(output_dir, (step,))
        llm_path = write_llm_call_traces_jsonl(output_dir, (llm_call,))
        tool_path = write_tool_call_traces_jsonl(output_dir, (tool_call,))
        manifest_path = write_run_manifest_json(output_dir, manifest)

        assert json.loads(metadata_path.read_text(encoding="utf-8"))["benchmark_source"] == (
            "stockpyl_serial"
        )
        assert json.loads(metadata_path.read_text(encoding="utf-8"))["artifact_use_class"] == (
            "internal_only"
        )
        assert len(episode_path.read_text(encoding="utf-8").splitlines()) == 1
        assert len(step_path.read_text(encoding="utf-8").splitlines()) == 1
        assert len(llm_path.read_text(encoding="utf-8").splitlines()) == 1
        assert len(tool_path.read_text(encoding="utf-8").splitlines()) == 1
        assert json.loads(manifest_path.read_text(encoding="utf-8"))["artifact_filenames"][-1] == [
            "run_manifest",
            "run_manifest.json",
        ]
        assert json.loads(manifest_path.read_text(encoding="utf-8"))["mode_artifact_use_classes"] == [
            ["deterministic_baseline", "internal_only"]
        ]
    finally:
        if scratch_dir.exists():
            shutil.rmtree(scratch_dir, ignore_errors=True)
