from __future__ import annotations

import pytest

from meio.evaluation.logging_schema import (
    CostBreakdownRecord,
    EpisodeSummaryRecord,
    StepTraceRecord,
    ToolCallTraceRecord,
)
from meio.evaluation.tool_attribution import (
    REMOVED_TOOL_BY_ABLATION,
    summarize_tool_ablation,
    summarize_tool_usage,
)


def _episode_record(
    *,
    episode_id: str,
    tool_ablation_variant: str,
    total_cost: float,
    fill_rate: float,
    no_action_count: int,
    replan_count: int,
    intervention_count: int,
    tool_call_count: int,
) -> EpisodeSummaryRecord:
    return EpisodeSummaryRecord(
        episode_id=episode_id,
        mode="llm_orchestrator",
        benchmark_id="serial_3_echelon",
        topology="serial",
        echelon_count=3,
        tool_ablation_variant=tool_ablation_variant,
        schedule_name="shift_recovery",
        run_seed=20260417,
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
        invalid_output_count=0,
        fallback_count=0,
        total_prompt_tokens=100,
        total_completion_tokens=50,
        total_tokens=150,
        total_llm_latency_ms=10.0,
        total_orchestration_latency_ms=12.0,
        optimizer_order_boundary_preserved=True,
        regime_prediction_accuracy=1.0,
        predicted_regime_counts=(("normal", 1), ("demand_regime_shift", 1), ("recovery", 1)),
        confusion_counts=(
            ("normal", "normal", 1),
            ("demand_regime_shift", "demand_regime_shift", 1),
            ("recovery", "recovery", 1),
        ),
        missed_intervention_count=0,
        unnecessary_intervention_count=0,
        average_confidence=0.85,
    )


def _step_record(
    *,
    period_index: int,
    tool_ablation_variant: str,
    selected_tools: tuple[str, ...],
    request_replan: bool,
    demand_outlook: float,
    optimizer_orders: tuple[float, ...],
    per_period_cost: float,
) -> StepTraceRecord:
    return StepTraceRecord(
        episode_id=f"{tool_ablation_variant}_episode",
        mode="llm_orchestrator",
        tool_ablation_variant=tool_ablation_variant,
        schedule_name="shift_recovery",
        run_seed=20260417,
        period_index=period_index,
        true_regime_label=("normal" if period_index == 0 else "demand_regime_shift"),
        predicted_regime_label=("normal" if period_index == 0 else "demand_regime_shift"),
        confidence=0.85,
        selected_subgoal=("no_action" if not request_replan else "query_uncertainty"),
        selected_tools=selected_tools,
        update_requests=(("keep_current",) if not request_replan else ("switch_demand_regime",)),
        request_replan=request_replan,
        abstain_or_no_action=not request_replan,
        demand_outlook=demand_outlook,
        leadtime_outlook=2.0,
        scenario_adjustment_summary={
            "demand_outlook": demand_outlook,
            "leadtime_outlook": 2.0,
            "safety_buffer_scale": 1.0 if not request_replan else 1.2,
        },
        optimizer_orders=optimizer_orders,
        inventory_by_echelon=(20.0, 25.0, 30.0),
        pipeline_by_echelon=(0.0, 0.0, 0.0),
        backorders_by_echelon=(0.0, 0.0, 0.0),
        per_period_cost=per_period_cost,
        per_period_fill_rate=1.0,
        decision_changed_optimizer_input=request_replan,
        optimizer_output_changed_state=True,
        intervention_changed_outcome=request_replan,
    )


def test_summarize_tool_usage_counts_direct_decision_and_optimizer_input_changes() -> None:
    records = (
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
            pre_tool_optimizer_input={"demand_outlook": 10.0, "leadtime_outlook": 2.0, "safety_buffer_scale": 1.0},
            post_tool_optimizer_input={"demand_outlook": 15.0, "leadtime_outlook": 2.0, "safety_buffer_scale": 1.0},
            decision_changed=True,
            optimizer_input_changed=True,
        ),
        ToolCallTraceRecord(
            episode_id="full_episode",
            mode="llm_orchestrator",
            tool_ablation_variant="full",
            schedule_name="shift_recovery",
            run_seed=20260417,
            period_index=1,
            call_index=1,
            tool_id="forecast_tool",
            tool_input={"tool_id": "forecast_tool"},
            tool_output={"status": "success"},
            success=True,
            error_type=None,
            latency_ms=1.0,
            pre_tool_decision={
                "regime_label": "demand_regime_shift",
                "update_request_types": ["switch_demand_regime"],
                "request_replan": False,
            },
            post_tool_decision={
                "regime_label": "demand_regime_shift",
                "update_request_types": ["switch_demand_regime"],
                "request_replan": True,
            },
            pre_tool_optimizer_input={"demand_outlook": 15.0, "leadtime_outlook": 2.0, "safety_buffer_scale": 1.0},
            post_tool_optimizer_input={"demand_outlook": 15.0, "leadtime_outlook": 2.0, "safety_buffer_scale": 1.0},
            decision_changed=True,
            optimizer_input_changed=False,
        ),
    )

    summary = summarize_tool_usage(
        mode="llm_orchestrator",
        tool_ablation_variant="full",
        tool_call_records=records,
    )

    assert len(summary.tool_records) == 1
    record = summary.tool_records[0]
    assert record.tool_id == "forecast_tool"
    assert record.call_count == 2
    assert record.changed_regime_label_count == 1
    assert record.changed_update_request_count == 1
    assert record.changed_replan_flag_count == 1
    assert record.changed_optimizer_input_count == 1
    assert record.changed_optimizer_orders_count is None


def test_summarize_tool_ablation_compares_against_full_reference() -> None:
    full_episodes = (
        _episode_record(
            episode_id="full_episode",
            tool_ablation_variant="full",
            total_cost=300.0,
            fill_rate=1.0,
            no_action_count=2,
            replan_count=1,
            intervention_count=1,
            tool_call_count=3,
        ),
    )
    ablated_episodes = (
        _episode_record(
            episode_id="no_forecast_episode",
            tool_ablation_variant="no_forecast_tool",
            total_cost=320.0,
            fill_rate=0.95,
            no_action_count=3,
            replan_count=0,
            intervention_count=0,
            tool_call_count=2,
        ),
    )
    full_steps = (
        _step_record(
            period_index=0,
            tool_ablation_variant="full",
            selected_tools=(),
            request_replan=False,
            demand_outlook=10.0,
            optimizer_orders=(0.0, 0.0, 0.0),
            per_period_cost=90.0,
        ),
        _step_record(
            period_index=1,
            tool_ablation_variant="full",
            selected_tools=("forecast_tool", "leadtime_tool", "scenario_tool"),
            request_replan=True,
            demand_outlook=15.0,
            optimizer_orders=(24.0, 22.0, 18.0),
            per_period_cost=105.0,
        ),
    )
    ablated_steps = (
        _step_record(
            period_index=0,
            tool_ablation_variant="no_forecast_tool",
            selected_tools=(),
            request_replan=False,
            demand_outlook=10.0,
            optimizer_orders=(0.0, 0.0, 0.0),
            per_period_cost=90.0,
        ),
        _step_record(
            period_index=1,
            tool_ablation_variant="no_forecast_tool",
            selected_tools=("leadtime_tool", "scenario_tool"),
            request_replan=False,
            demand_outlook=10.0,
            optimizer_orders=(18.0, 16.0, 12.0),
            per_period_cost=115.0,
        ),
    )

    summary = summarize_tool_ablation(
        mode="llm_orchestrator",
        tool_ablation_variant="no_forecast_tool",
        full_episode_records=full_episodes,
        ablated_episode_records=ablated_episodes,
        full_step_records=full_steps,
        ablated_step_records=ablated_steps,
    )

    assert REMOVED_TOOL_BY_ABLATION["no_forecast_tool"] == "forecast_tool"
    assert summary.removed_tool_id == "forecast_tool"
    assert summary.run_count == 1
    assert summary.no_action_rate_delta_vs_full > 0.0
    assert summary.replan_rate_delta_vs_full < 0.0
    assert summary.total_cost_delta_vs_full == 20.0
    assert summary.fill_rate_delta_vs_full == pytest.approx(-0.05)
    assert summary.tool_call_count_delta_vs_full == -1.0
    assert summary.decision_change_count == 1
    assert summary.optimizer_input_change_count == 1
    assert summary.optimizer_orders_change_count == 1
    assert summary.outcome_proxy_change_count == 1
    assert summary.unavailable_tool_request_count == 0
    assert summary.disabled_tool_fallback_count == 0
    assert summary.sequencing_blocked_tool_request_count == 0
