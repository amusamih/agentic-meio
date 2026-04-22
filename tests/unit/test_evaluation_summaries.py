from __future__ import annotations

from meio.contracts import AgentSignal, OperationalSubgoal
from meio.agents.telemetry import EpisodeTelemetrySummary
from meio.evaluation.decision_quality import DecisionQualitySummary
from meio.evaluation.summaries import (
    BenchmarkRunSummary,
    InterventionSummary,
    TraceSummary,
    build_episode_summary_record,
    build_benchmark_run_summary,
)
from meio.evaluation.logging_schema import CostBreakdownRecord
from meio.optimization.contracts import OptimizationResult
from meio.simulation.evidence import DemandEvidence, LeadTimeEvidence
from meio.simulation.state import EpisodeTrace, Observation, SimulationState


def test_benchmark_run_summary_construction_from_trace() -> None:
    trace = EpisodeTrace(
        run_id="summary_test_episode_1",
        benchmark_id="serial_3_echelon",
        states=(
            SimulationState(
                benchmark_id="serial_3_echelon",
                time_index=0,
                inventory_level=(20.0, 30.0, 40.0),
                stage_names=("retailer", "regional_dc", "plant"),
                pipeline_inventory=(0.0, 0.0, 0.0),
                backorder_level=(0.0, 0.0, 0.0),
            ),
        ),
        observations=(
            Observation(
                time_index=0,
                demand_evidence=DemandEvidence(history=(10.0, 10.0, 10.0), latest_realization=(10.0,)),
                leadtime_evidence=LeadTimeEvidence(
                    history=(2.0, 2.0, 2.0),
                    latest_realization=(2.0,),
                    upstream_stage_index=2,
                    downstream_stage_index=1,
                ),
            ),
        ),
        agent_signals=(
            AgentSignal(
                selected_subgoal=OperationalSubgoal.REQUEST_REPLAN,
                selected_tool_id="forecast_tool",
                tool_sequence=("forecast_tool", "leadtime_tool", "scenario_tool"),
                request_replan=True,
                rationale="typed_summary_test",
            ),
        ),
        optimization_results=(
            OptimizationResult(
                replenishment_orders=(0.0, 0.0, 0.0),
                planning_horizon=1,
                objective_value=0.0,
                provenance="trusted_optimizer_smoke_adapter",
            ),
        ),
    )

    summary = build_benchmark_run_summary(
        run_id="summary_test",
        benchmark_id="serial_3_echelon",
        benchmark_source="stockpyl_serial",
        topology="serial",
        echelon_count=3,
        traces=(trace,),
        optimizer_order_boundary_preserved=True,
        notes=("summary_test",),
    )

    assert isinstance(summary, BenchmarkRunSummary)
    assert summary.intervention_summary.tool_call_count == 3
    assert summary.intervention_summary.replan_count == 1
    assert summary.trace_summary.trace_length == 1
    assert summary.average_inventory == 30.0
    assert summary.optimizer_order_boundary_preserved is True


def test_build_episode_summary_record_scales_episode_telemetry_totals() -> None:
    summary = BenchmarkRunSummary(
        run_id="summary_test_episode",
        benchmark_id="serial_3_echelon",
        benchmark_source="stockpyl_serial",
        topology="serial",
        echelon_count=3,
        intervention_summary=InterventionSummary(
            tool_call_count=3,
            replan_count=1,
            abstain_count=0,
            no_action_count=0,
        ),
        trace_summary=TraceSummary(
            episode_count=1,
            trace_length=3,
            state_count=3,
            observation_count=3,
            agent_signal_count=3,
            optimization_call_count=3,
        ),
        total_cost=120.0,
        fill_rate=1.0,
        average_inventory=25.0,
        average_backorder_level=0.0,
        optimizer_order_boundary_preserved=True,
        episode_telemetry=EpisodeTelemetrySummary(
            provider="fake_llm_client",
            model_name="gpt-4o-mini",
            step_count=3,
            llm_call_count=2,
            average_prompt_tokens=96.0,
            average_completion_tokens=48.0,
            average_total_tokens=144.0,
            average_llm_latency_ms=5.0,
            average_orchestration_latency_ms=6.0,
            client_error_counts=(("network_error", 1),),
            total_retry_count=1,
            failure_before_response_count=1,
            failure_after_response_count=0,
        ),
        decision_quality=DecisionQualitySummary(
            step_count=3,
            regime_prediction_accuracy=1.0,
            predicted_regime_counts=(("normal", 1), ("demand_regime_shift", 1), ("recovery", 1)),
            confusion_counts=(
                ("normal", "normal", 1),
                ("demand_regime_shift", "demand_regime_shift", 1),
                ("recovery", "recovery", 1),
            ),
            no_action_rate=2 / 3,
            replan_rate=1 / 3,
            intervention_rate=1 / 3,
            missed_intervention_count=0,
            unnecessary_intervention_count=0,
            average_confidence=0.88,
        ),
    )

    record = build_episode_summary_record(
        summary,
        mode="llm_orchestrator",
        tool_ablation_variant="full",
        schedule_name="shift_recovery",
        run_seed=20260417,
        regime_schedule=("normal", "demand_regime_shift", "recovery"),
        cost_breakdown=CostBreakdownRecord(
            holding_cost=60.0,
            backlog_cost=20.0,
            ordering_cost=40.0,
        ),
        intervention_count=2,
        invalid_output_count=1,
        fallback_count=1,
    )

    assert record.total_prompt_tokens == 192
    assert record.total_completion_tokens == 96
    assert record.total_tokens == 288
    assert record.total_llm_latency_ms == 10.0
    assert record.total_orchestration_latency_ms == 18.0
    assert record.client_error_counts == (("network_error", 1),)
    assert record.total_retry_count == 1
    assert record.tool_ablation_variant == "full"
    assert record.schedule_name == "shift_recovery"
    assert record.run_seed == 20260417
    assert record.step_count == 3
    assert record.no_action_count == 0
    assert record.regime_prediction_accuracy == 1.0
    assert record.missed_intervention_count == 0
