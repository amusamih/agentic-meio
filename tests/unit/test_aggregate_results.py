from __future__ import annotations

from meio.evaluation.aggregate_results import (
    aggregate_batch_episode_summaries,
    aggregate_mode_episode_summaries,
)
from meio.evaluation.logging_schema import (
    CostBreakdownRecord,
    EpisodeSummaryRecord,
    StepTraceRecord,
    ToolCallTraceRecord,
)


def _episode_record(
    *,
    episode_id: str,
    mode: str,
    schedule_name: str,
    run_seed: int,
    total_cost: float,
    fill_rate: float,
    no_action_count: int,
    replan_count: int,
    intervention_count: int,
    tool_call_count: int,
    total_tokens: int | None,
    total_llm_latency_ms: float | None,
    tool_ablation_variant: str = "full",
    invalid_output_count: int = 0,
    fallback_count: int = 0,
    client_error_counts: tuple[tuple[str, int], ...] = (),
    total_retry_count: int = 0,
    unavailable_tool_request_count: int = 0,
    disabled_tool_fallback_count: int = 0,
    sequencing_blocked_tool_request_count: int = 0,
    clean_intervention_count: int = 0,
    clean_optimizer_input_change_count: int = 0,
    repeated_stress_moderation_count: int = 0,
    relapse_moderation_count: int = 0,
    unresolved_stress_moderation_count: int = 0,
    moderated_update_count: int = 0,
    hysteresis_application_count: int = 0,
    evidence_fusion_cap_count: int = 0,
    capped_external_strengthening_count: int = 0,
    early_evidence_confirmation_gate_count: int = 0,
    early_evidence_family_change_block_count: int = 0,
    corroboration_gate_count: int = 0,
    corroborated_family_change_count: int = 0,
    role_level_usage_counts: tuple[tuple[str, int], ...] = (),
) -> EpisodeSummaryRecord:
    return EpisodeSummaryRecord(
        episode_id=episode_id,
        mode=mode,
        benchmark_id="serial_3_echelon",
        topology="serial",
        echelon_count=3,
        tool_ablation_variant=tool_ablation_variant,
        schedule_name=schedule_name,
        run_seed=run_seed,
        regime_schedule=("normal", "demand_regime_shift", "recovery"),
        total_cost=total_cost,
        cost_breakdown=CostBreakdownRecord(
            holding_cost=60.0,
            backlog_cost=10.0,
            ordering_cost=20.0,
        ),
        fill_rate=fill_rate,
        average_inventory=25.0,
        average_backorder_level=0.0,
        step_count=3,
        replan_count=replan_count,
        intervention_count=intervention_count,
        abstain_count=0,
        no_action_count=no_action_count,
        tool_call_count=tool_call_count,
        invalid_output_count=invalid_output_count,
        fallback_count=fallback_count,
        total_prompt_tokens=100 if total_tokens is not None else None,
        total_completion_tokens=50 if total_tokens is not None else None,
        total_tokens=total_tokens,
        total_llm_latency_ms=total_llm_latency_ms,
        total_orchestration_latency_ms=10.0,
        optimizer_order_boundary_preserved=True,
        client_error_counts=client_error_counts,
        total_retry_count=total_retry_count,
        failure_before_response_count=sum(count for _, count in client_error_counts),
        failure_after_response_count=0,
        regime_prediction_accuracy=1.0,
        predicted_regime_counts=(("normal", 1), ("demand_regime_shift", 1), ("recovery", 1)),
        confusion_counts=(
            ("normal", "normal", 1),
            ("demand_regime_shift", "demand_regime_shift", 1),
            ("recovery", "recovery", 1),
        ),
        missed_intervention_count=0,
        unnecessary_intervention_count=0,
        average_confidence=0.9,
        unavailable_tool_request_count=unavailable_tool_request_count,
        disabled_tool_fallback_count=disabled_tool_fallback_count,
        sequencing_blocked_tool_request_count=sequencing_blocked_tool_request_count,
        clean_intervention_count=clean_intervention_count,
        clean_optimizer_input_change_count=clean_optimizer_input_change_count,
        repeated_stress_moderation_count=repeated_stress_moderation_count,
        relapse_moderation_count=relapse_moderation_count,
        unresolved_stress_moderation_count=unresolved_stress_moderation_count,
        moderated_update_count=moderated_update_count,
        hysteresis_application_count=hysteresis_application_count,
        evidence_fusion_cap_count=evidence_fusion_cap_count,
        capped_external_strengthening_count=capped_external_strengthening_count,
        early_evidence_confirmation_gate_count=early_evidence_confirmation_gate_count,
        early_evidence_family_change_block_count=early_evidence_family_change_block_count,
        corroboration_gate_count=corroboration_gate_count,
        corroborated_family_change_count=corroborated_family_change_count,
        role_level_usage_counts=role_level_usage_counts,
    )


def test_aggregate_mode_episode_summaries_computes_schedule_and_seed_aware_means() -> None:
    records = (
        _episode_record(
            episode_id="run_0",
            mode="llm_orchestrator",
            schedule_name="shift_recovery",
            run_seed=20260417,
            total_cost=300.0,
            fill_rate=1.0,
            no_action_count=2,
            replan_count=1,
            intervention_count=1,
            tool_call_count=3,
            total_tokens=4500,
            total_llm_latency_ms=8000.0,
        ),
        _episode_record(
            episode_id="run_1",
            mode="llm_orchestrator",
            schedule_name="sustained_shift",
            run_seed=20260418,
            total_cost=360.0,
            fill_rate=0.95,
            no_action_count=1,
            replan_count=2,
            intervention_count=2,
            tool_call_count=6,
            total_tokens=4800,
            total_llm_latency_ms=9000.0,
            client_error_counts=(("network_error", 1),),
            total_retry_count=1,
            unavailable_tool_request_count=1,
            disabled_tool_fallback_count=1,
            sequencing_blocked_tool_request_count=2,
            clean_intervention_count=1,
            clean_optimizer_input_change_count=1,
            relapse_moderation_count=1,
            unresolved_stress_moderation_count=1,
            moderated_update_count=1,
            hysteresis_application_count=1,
        ),
    )

    summary = aggregate_mode_episode_summaries("llm_orchestrator", records)

    assert summary.run_count == 2
    assert summary.schedule_names == ("shift_recovery", "sustained_shift")
    assert summary.seed_values == (20260417, 20260418)
    assert summary.benchmark_metrics.average_total_cost == 330.0
    assert summary.decision_quality.no_action_rate == 0.5
    assert summary.decision_quality.replan_rate == 0.5
    assert summary.telemetry_metrics.average_total_tokens == 4650.0
    assert summary.telemetry_metrics.client_error_counts == (("network_error", 1),)
    assert summary.telemetry_metrics.total_retry_count == 1
    assert summary.validity_summary.optimizer_order_boundary_preserved is True
    assert summary.performance_summary.average_fill_rate == 0.975
    assert summary.tool_use_summary.average_tool_call_count == 4.5
    assert summary.tool_use_summary.unavailable_tool_request_count == 1
    assert summary.tool_use_summary.disabled_tool_fallback_count == 1
    assert summary.tool_use_summary.sequencing_blocked_tool_request_count == 2
    assert summary.tool_use_summary.clean_intervention_count == 1
    assert summary.tool_use_summary.relapse_moderation_count == 1
    assert summary.tool_use_summary.unresolved_stress_moderation_count == 1
    assert summary.tool_use_summary.moderated_update_count == 1
    assert summary.tool_use_summary.hysteresis_application_count == 1
    assert summary.robustness_summary.schedule_count == 2
    assert summary.robustness_summary.seed_count == 2
    assert summary.artifact_use_class.value == "internal_only"
    assert summary.validity_gate_passed is False
    assert summary.tool_ablation_variants == ("full",)
    assert len(summary.schedule_breakdown) == 2


def test_aggregate_batch_episode_summaries_groups_records_by_mode() -> None:
    baseline_record = _episode_record(
        episode_id="baseline_0",
        mode="deterministic_baseline",
        schedule_name="shift_recovery",
        run_seed=20260417,
        total_cost=280.0,
        fill_rate=1.0,
        no_action_count=2,
        replan_count=1,
        intervention_count=1,
        tool_call_count=0,
        total_tokens=None,
        total_llm_latency_ms=None,
    )
    llm_record = _episode_record(
        episode_id="llm_0",
        mode="llm_orchestrator",
        schedule_name="shift_recovery",
        run_seed=20260417,
        total_cost=330.0,
        fill_rate=1.0,
        no_action_count=2,
        replan_count=1,
        intervention_count=1,
        tool_call_count=3,
        total_tokens=4623,
        total_llm_latency_ms=7800.0,
    )

    batch_summary = aggregate_batch_episode_summaries(
        benchmark_id="serial_3_echelon",
        records_by_mode={
            "deterministic_baseline": (baseline_record,),
            "llm_orchestrator": (llm_record,),
        },
    )

    assert batch_summary.benchmark_id == "serial_3_echelon"
    assert batch_summary.mode_names == ("deterministic_baseline", "llm_orchestrator")
    assert batch_summary.schedule_names == ("shift_recovery",)
    assert batch_summary.seed_values == (20260417,)
    assert batch_summary.tool_ablation_variants == ("full",)
    assert batch_summary.artifact_use_class.value == "internal_only"
    assert batch_summary.validity_gate_passed is False
    assert batch_summary.schema_version.startswith("meio.experiment_log.")


def test_aggregate_mode_episode_summaries_include_external_evidence_cap_counts() -> None:
    records = (
        _episode_record(
            episode_id="external_0",
            mode="llm_orchestrator_with_external_evidence",
            schedule_name="double_shift_with_gap",
            run_seed=20260417,
            total_cost=320.0,
            fill_rate=1.0,
            no_action_count=1,
            replan_count=2,
            intervention_count=2,
            tool_call_count=4,
            total_tokens=4200,
            total_llm_latency_ms=7000.0,
            evidence_fusion_cap_count=2,
            capped_external_strengthening_count=2,
            early_evidence_confirmation_gate_count=1,
            early_evidence_family_change_block_count=1,
            corroboration_gate_count=1,
            corroborated_family_change_count=2,
            role_level_usage_counts=(("leading_demand_shift", 2),),
        ),
    )

    summary = aggregate_mode_episode_summaries(
        "llm_orchestrator_with_external_evidence",
        records,
    )

    assert summary.external_evidence_summary is not None
    assert summary.external_evidence_summary.evidence_fusion_cap_count == 2
    assert summary.external_evidence_summary.capped_external_strengthening_count == 2
    assert summary.external_evidence_summary.early_evidence_confirmation_gate_count == 1
    assert summary.external_evidence_summary.early_evidence_family_change_block_count == 1
    assert summary.external_evidence_summary.corroboration_gate_count == 1
    assert summary.external_evidence_summary.corroborated_family_change_count == 2
    assert summary.external_evidence_summary.role_level_usage_counts == (
        ("leading_demand_shift", 2),
    )


def test_aggregate_mode_episode_summaries_include_tool_attribution_and_ablation_breakdown() -> None:
    records = (
        _episode_record(
            episode_id="full_episode",
            mode="llm_orchestrator",
            schedule_name="shift_recovery",
            run_seed=20260417,
            total_cost=300.0,
            fill_rate=1.0,
            no_action_count=2,
            replan_count=1,
            intervention_count=1,
            tool_call_count=3,
            total_tokens=4500,
            total_llm_latency_ms=8000.0,
            tool_ablation_variant="full",
        ),
        _episode_record(
            episode_id="no_forecast_episode",
            mode="llm_orchestrator",
            schedule_name="shift_recovery",
            run_seed=20260417,
            total_cost=320.0,
            fill_rate=0.95,
            no_action_count=3,
            replan_count=0,
            intervention_count=0,
            tool_call_count=2,
            total_tokens=4300,
            total_llm_latency_ms=7900.0,
            tool_ablation_variant="no_forecast_tool",
            unavailable_tool_request_count=1,
            disabled_tool_fallback_count=1,
            sequencing_blocked_tool_request_count=2,
        ),
    )
    step_trace_records = (
        StepTraceRecord(
            episode_id="full_episode",
            mode="llm_orchestrator",
            tool_ablation_variant="full",
            schedule_name="shift_recovery",
            run_seed=20260417,
            period_index=1,
            true_regime_label="demand_regime_shift",
            predicted_regime_label="demand_regime_shift",
            confidence=0.9,
            selected_subgoal="query_uncertainty",
            selected_tools=("forecast_tool", "leadtime_tool", "scenario_tool"),
            update_requests=("switch_demand_regime",),
            request_replan=True,
            abstain_or_no_action=False,
            demand_outlook=15.0,
            leadtime_outlook=2.0,
            scenario_adjustment_summary={
                "demand_outlook": 15.0,
                "leadtime_outlook": 2.0,
                "safety_buffer_scale": 1.2,
            },
            optimizer_orders=(24.0, 22.0, 18.0),
            inventory_by_echelon=(20.0, 25.0, 30.0),
            pipeline_by_echelon=(0.0, 0.0, 0.0),
            backorders_by_echelon=(0.0, 0.0, 0.0),
            per_period_cost=100.0,
            per_period_fill_rate=1.0,
            decision_changed_optimizer_input=True,
            optimizer_output_changed_state=True,
            intervention_changed_outcome=True,
        ),
        StepTraceRecord(
            episode_id="no_forecast_episode",
            mode="llm_orchestrator",
            tool_ablation_variant="no_forecast_tool",
            schedule_name="shift_recovery",
            run_seed=20260417,
            period_index=1,
            true_regime_label="demand_regime_shift",
            predicted_regime_label="demand_regime_shift",
            confidence=0.9,
            selected_subgoal="no_action",
            selected_tools=("leadtime_tool", "scenario_tool"),
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
            optimizer_orders=(18.0, 16.0, 12.0),
            inventory_by_echelon=(20.0, 25.0, 30.0),
            pipeline_by_echelon=(0.0, 0.0, 0.0),
            backorders_by_echelon=(0.0, 0.0, 0.0),
            per_period_cost=110.0,
            per_period_fill_rate=1.0,
            decision_changed_optimizer_input=False,
            optimizer_output_changed_state=True,
            intervention_changed_outcome=False,
        ),
    )
    tool_call_records = (
        ToolCallTraceRecord(
            episode_id="full_episode",
            mode="llm_orchestrator",
            tool_ablation_variant="full",
            schedule_name="shift_recovery",
            run_seed=20260417,
            period_index=1,
            call_index=0,
            tool_id="forecast_tool",
            tool_input={"tool_id": "forecast_tool"},
            tool_output={"status": "success"},
            success=True,
            error_type=None,
            latency_ms=1.0,
            pre_tool_decision={
                "regime_label": "normal",
                "update_request_types": ["keep_current"],
                "request_replan": False,
            },
            post_tool_decision={
                "regime_label": "demand_regime_shift",
                "update_request_types": ["switch_demand_regime"],
                "request_replan": False,
            },
            pre_tool_optimizer_input={
                "demand_outlook": 10.0,
                "leadtime_outlook": 2.0,
                "safety_buffer_scale": 1.0,
            },
            post_tool_optimizer_input={
                "demand_outlook": 15.0,
                "leadtime_outlook": 2.0,
                "safety_buffer_scale": 1.0,
            },
            decision_changed=True,
            optimizer_input_changed=True,
        ),
    )

    summary = aggregate_mode_episode_summaries(
        "llm_orchestrator",
        records,
        step_trace_records=step_trace_records,
        tool_call_records=tool_call_records,
    )

    assert summary.tool_ablation_variants == ("full", "no_forecast_tool")
    assert len(summary.tool_attribution) == 2
    assert len(summary.ablation_breakdown) == 2
    no_forecast_summary = next(
        item
        for item in summary.ablation_breakdown
        if item.tool_ablation_variant == "no_forecast_tool"
    )
    assert no_forecast_summary.tool_ablation_summary is not None
    assert no_forecast_summary.tool_ablation_summary.removed_tool_id == "forecast_tool"
    assert no_forecast_summary.tool_ablation_summary.decision_change_count == 1
    assert no_forecast_summary.tool_ablation_summary.unavailable_tool_request_count == 1
    assert no_forecast_summary.tool_ablation_summary.disabled_tool_fallback_count == 1
    assert (
        no_forecast_summary.tool_ablation_summary.sequencing_blocked_tool_request_count
        == 2
    )
