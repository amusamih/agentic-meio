from __future__ import annotations

from meio.evaluation.logging_schema import (
    ArtifactUseClass,
    CostBreakdownRecord,
    EpisodeSummaryRecord,
    ExperimentMetadata,
    LLMCallTraceRecord,
    RunManifestRecord,
    SCHEMA_VERSION,
    StepTraceRecord,
    ToolCallTraceRecord,
)
from meio.agents.prompts import PROMPT_VERSION


def test_experiment_metadata_construction_accepts_typed_resolved_config() -> None:
    metadata = ExperimentMetadata(
        experiment_id="stockpyl_logging_all",
        run_group_id="stockpyl_logging_all_20260417T100000Z",
        timestamp="2026-04-17T10:00:00Z",
        git_commit_sha="abc123",
        resolved_config={"mode": "all", "episode_count": 2},
        config_hash="deadbeef",
        benchmark_id="serial_3_echelon",
        benchmark_source="stockpyl_serial",
        mode="all",
        provider="mixed",
        model_name="mixed",
        tool_ablation_variants=("full", "no_forecast_tool"),
        artifact_use_class=ArtifactUseClass.INTERNAL_ONLY,
        validity_gate_passed=False,
        eligibility_notes=("provisional_stockpyl_rollout",),
        prompt_version=PROMPT_VERSION,
        prompt_hash="prompt_hash",
    )

    assert metadata.schema_version == SCHEMA_VERSION
    assert metadata.provider == "mixed"
    assert metadata.model_name == "mixed"
    assert metadata.tool_ablation_variants == ("full", "no_forecast_tool")
    assert metadata.artifact_use_class is ArtifactUseClass.INTERNAL_ONLY
    assert metadata.validity_gate_passed is False


def test_run_manifest_record_construction_captures_written_artifacts() -> None:
    manifest = RunManifestRecord(
        experiment_id="stockpyl_serial_multi_eval_all",
        run_group_id="stockpyl_serial_multi_eval_all_20260417T110217Z",
        config_hash="deadbeef",
        benchmark_id="serial_3_echelon",
        benchmark_source="stockpyl_serial",
        benchmark_config_path="configs/benchmark/serial_3_echelon.toml",
        agent_config_path="configs/agent/live_llm.toml",
        schedule_names=("shift_recovery", "sustained_shift"),
        seed_values=(20260417, 20260418),
        mode_names=(
            "deterministic_baseline",
            "deterministic_orchestrator",
            "llm_orchestrator",
        ),
        tool_ablation_variants=("full", "no_scenario_tool"),
        provider="mixed",
        model_name="mixed",
        artifact_use_class=ArtifactUseClass.INTERNAL_ONLY,
        validity_gate_passed=True,
        eligibility_notes=("provisional_operational_metrics",),
        mode_artifact_use_classes=(
            ("deterministic_baseline", "internal_only"),
            ("deterministic_orchestrator", "internal_only"),
        ),
        prompt_version=PROMPT_VERSION,
        prompt_hash="prompt_hash",
        artifact_filenames=(
            ("experiment_metadata", "experiment_metadata.json"),
            ("aggregate_summary", "aggregate_summary.json"),
            ("run_manifest", "run_manifest.json"),
        ),
    )

    assert manifest.schema_version == SCHEMA_VERSION
    assert manifest.seed_values == (20260417, 20260418)
    assert manifest.tool_ablation_variants == ("full", "no_scenario_tool")
    assert manifest.artifact_filenames[-1] == ("run_manifest", "run_manifest.json")
    assert manifest.mode_artifact_use_classes[0] == (
        "deterministic_baseline",
        "internal_only",
    )


def test_step_trace_record_construction_preserves_typed_period_fields() -> None:
    record = StepTraceRecord(
        episode_id="episode_1",
        mode="llm_orchestrator",
        tool_ablation_variant="full",
        schedule_name="shift_recovery",
        run_seed=20260417,
        period_index=1,
        true_regime_label="demand_regime_shift",
        predicted_regime_label="demand_regime_shift",
        confidence=0.84,
        selected_subgoal="query_uncertainty",
        selected_tools=("forecast_tool", "leadtime_tool", "scenario_tool"),
        proposed_update_requests=("switch_demand_regime", "widen_uncertainty"),
        proposed_update_strength="switch_demand_regime_plus_widen_uncertainty",
        final_update_strength="reweight_scenarios",
        calibration_applied=True,
        hysteresis_applied=True,
        update_requests=("reweight_scenarios",),
        request_replan=True,
        abstain_or_no_action=False,
        demand_outlook=15.0,
        leadtime_outlook=2.5,
        scenario_adjustment_summary={
            "demand_outlook": 15.0,
            "leadtime_outlook": 2.5,
            "safety_buffer_scale": 1.1,
        },
        optimizer_orders=(12.0, 5.0, 0.0),
        inventory_by_echelon=(20.0, 25.0, 30.0),
        pipeline_by_echelon=(2.0, 1.0, 0.0),
        backorders_by_echelon=(0.0, 0.0, 0.0),
        per_period_cost=42.5,
        per_period_fill_rate=1.0,
        decision_changed_optimizer_input=True,
        optimizer_output_changed_state=True,
        intervention_changed_outcome=True,
        calibration_reason="repeated_stress_not_materially_worsening",
        external_evidence_fusion_cap_applied=True,
        external_evidence_role_labels=("leading_demand_shift",),
        external_evidence_corroboration_count=2,
        corroboration_gate_applied=True,
        corroborated_family_change_allowed=False,
        proposed_external_strengthening={
            "demand_outlook": 15.45,
            "leadtime_outlook": 2.5,
            "safety_buffer_scale": 1.133,
        },
        final_external_strengthening={
            "demand_outlook": 15.0,
            "leadtime_outlook": 2.5,
            "safety_buffer_scale": 1.1,
        },
        external_evidence_fusion_cap_reason=(
            "confirming_external_evidence_with_sufficient_internal_signal"
        ),
        corroboration_gate_reason="uncorroborated_leading_external_evidence",
        early_evidence_confirmation_gate_applied=True,
        early_evidence_family_change_blocked=True,
        proposed_external_update_requests=(
            "switch_demand_regime",
            "widen_uncertainty",
        ),
        final_external_update_requests=("keep_current",),
        early_evidence_confirmation_gate_reason=(
            "leading_external_evidence_before_internal_stress"
        ),
    )

    assert record.selected_tools[-1] == "scenario_tool"
    assert record.scenario_adjustment_summary is not None
    assert record.scenario_adjustment_summary["safety_buffer_scale"] == 1.1
    assert record.proposed_update_strength == "switch_demand_regime_plus_widen_uncertainty"
    assert record.final_update_strength == "reweight_scenarios"
    assert record.calibration_applied is True
    assert record.hysteresis_applied is True
    assert record.unavailable_tool_request is False
    assert record.sequencing_blocked_tool_request_count == 0
    assert record.clean_intervention is False
    assert record.external_evidence_fusion_cap_applied is True
    assert record.external_evidence_role_labels == ("leading_demand_shift",)
    assert record.external_evidence_corroboration_count == 2
    assert record.corroboration_gate_applied is True
    assert record.corroborated_family_change_allowed is False
    assert record.proposed_external_strengthening is not None
    assert record.final_external_strengthening is not None
    assert record.early_evidence_confirmation_gate_applied is True
    assert record.early_evidence_family_change_blocked is True
    assert record.proposed_external_update_requests == (
        "switch_demand_regime",
        "widen_uncertainty",
    )
    assert record.final_external_update_requests == ("keep_current",)


def test_episode_summary_and_trace_records_capture_logging_fields() -> None:
    episode = EpisodeSummaryRecord(
        episode_id="episode_1",
        mode="deterministic_orchestrator",
        benchmark_id="serial_3_echelon",
        topology="serial",
        echelon_count=3,
        tool_ablation_variant="full",
        schedule_name="shift_recovery",
        run_seed=20260417,
        regime_schedule=("normal", "demand_regime_shift", "recovery"),
        total_cost=120.0,
        cost_breakdown=CostBreakdownRecord(
            holding_cost=60.0,
            backlog_cost=20.0,
            ordering_cost=40.0,
        ),
        fill_rate=1.0,
        average_inventory=25.0,
        average_backorder_level=0.0,
        step_count=3,
        replan_count=1,
        intervention_count=1,
        abstain_count=0,
        no_action_count=0,
        tool_call_count=3,
        invalid_output_count=0,
        fallback_count=0,
        total_prompt_tokens=None,
        total_completion_tokens=None,
        total_tokens=None,
        total_llm_latency_ms=None,
        total_orchestration_latency_ms=1.5,
        optimizer_order_boundary_preserved=True,
        client_error_counts=(("network_error", 1),),
        total_retry_count=1,
        failure_before_response_count=1,
        failure_after_response_count=0,
        regime_prediction_accuracy=0.5,
        predicted_regime_counts=(("normal", 2),),
        confusion_counts=(("normal", "normal", 1), ("recovery", "normal", 1)),
        missed_intervention_count=1,
        unnecessary_intervention_count=0,
        average_confidence=0.9,
        unavailable_tool_request_count=0,
        disabled_tool_fallback_count=0,
        sequencing_blocked_tool_request_count=1,
        clean_intervention_count=1,
        clean_optimizer_input_change_count=1,
        repeated_stress_moderation_count=1,
        relapse_moderation_count=2,
        unresolved_stress_moderation_count=2,
        moderated_update_count=3,
        hysteresis_application_count=3,
        evidence_fusion_cap_count=2,
        capped_external_strengthening_count=2,
        early_evidence_confirmation_gate_count=1,
        early_evidence_family_change_block_count=1,
        corroboration_gate_count=1,
        corroborated_family_change_count=2,
        role_level_usage_counts=(("leading_demand_shift", 2),),
    )
    llm_trace = LLMCallTraceRecord(
        episode_id="episode_1",
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
        requested_tool_ids=("forecast_tool", "scenario_tool"),
        unavailable_tool_ids=("forecast_tool",),
        violated_available_tool_set=True,
        proposed_update_strength="switch_demand_regime_plus_widen_uncertainty",
        final_update_strength="reweight_scenarios",
        calibration_applied=True,
        hysteresis_applied=True,
        final_update_request_types=("reweight_scenarios",),
        repeated_stress_moderation_applied=False,
        relapse_moderation_applied=True,
        unresolved_stress_moderation_applied=True,
        calibration_reason="relapse_with_unresolved_stress_load",
        moderation_reason="relapse_with_unresolved_stress_load",
        prompt_tokens=96,
        completion_tokens=48,
        total_tokens=144,
        latency_ms=5.0,
        client_error_category="network_error",
        client_error_message="APIConnectionError: Connection error.",
        failure_after_response=False,
    )
    tool_trace = ToolCallTraceRecord(
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

    assert episode.cost_breakdown.ordering_cost == 40.0
    assert episode.client_error_counts == (("network_error", 1),)
    assert episode.clean_intervention_count == 1
    assert episode.sequencing_blocked_tool_request_count == 1
    assert episode.relapse_moderation_count == 2
    assert episode.unresolved_stress_moderation_count == 2
    assert episode.hysteresis_application_count == 3
    assert episode.evidence_fusion_cap_count == 2
    assert episode.capped_external_strengthening_count == 2
    assert episode.early_evidence_confirmation_gate_count == 1
    assert episode.early_evidence_family_change_block_count == 1
    assert episode.corroboration_gate_count == 1
    assert episode.corroborated_family_change_count == 2
    assert episode.role_level_usage_counts == (("leading_demand_shift", 2),)
    assert llm_trace.total_tokens == 144
    assert llm_trace.client_error_category == "network_error"
    assert llm_trace.violated_available_tool_set is True
    assert llm_trace.proposed_update_strength == "switch_demand_regime_plus_widen_uncertainty"
    assert llm_trace.final_update_strength == "reweight_scenarios"
    assert llm_trace.calibration_applied is True
    assert llm_trace.hysteresis_applied is True
    assert llm_trace.relapse_moderation_applied is True
    assert tool_trace.tool_output == {"status": "success"}
