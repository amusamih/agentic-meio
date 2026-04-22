from __future__ import annotations

from meio.config.schemas import CostConfig
from meio.contracts import AgentSignal, OperationalSubgoal, RegimeLabel
from meio.evaluation.rollout_metrics import (
    compute_period_cost_breakdown,
    compute_period_fill_rate,
    compute_period_total_cost,
    compute_rollout_metrics,
)
from meio.optimization.contracts import OptimizationResult
from meio.simulation.evidence import DemandEvidence, LeadTimeEvidence
from meio.simulation.state import EpisodeTrace, Observation, PeriodTraceRecord, SimulationState


def test_compute_rollout_metrics_returns_provisional_summary() -> None:
    trace = EpisodeTrace(run_id="run_metrics", benchmark_id="serial_3_echelon")
    for time_index, demand_value in enumerate((10.0, 14.0)):
        state = SimulationState(
            benchmark_id="serial_3_echelon",
            time_index=time_index,
            inventory_level=(20.0 - time_index, 30.0, 40.0),
            backorder_level=(float(time_index), 0.0, 0.0),
            regime_label=RegimeLabel.NORMAL if time_index == 0 else RegimeLabel.DEMAND_REGIME_SHIFT,
        )
        observation = Observation(
            time_index=time_index,
            demand_evidence=DemandEvidence(history=(demand_value,), latest_realization=(demand_value,)),
            leadtime_evidence=LeadTimeEvidence(history=(2.0,), latest_realization=(2.0,)),
            regime_label=state.regime_label,
        )
        signal = AgentSignal(
            selected_subgoal=OperationalSubgoal.REQUEST_REPLAN,
            tool_sequence=("forecast_tool", "leadtime_tool", "scenario_tool"),
            request_replan=True,
            rationale="Controlled rollout signal.",
        )
        optimization_result = OptimizationResult(
            replenishment_orders=(0.0, 0.0, 0.0),
            planning_horizon=1,
        )
        trace = trace.append_period(
            PeriodTraceRecord(
                time_index=time_index,
                regime_label=state.regime_label,
                state=state,
                observation=observation,
                agent_signal=signal,
                optimization_result=optimization_result,
                realized_demand=demand_value,
                demand_load=demand_value + float(time_index),
                served_demand=demand_value,
                unmet_demand=float(time_index),
            )
        )

    metrics = compute_rollout_metrics(
        trace,
        CostConfig(holding_cost=1.0, backorder_cost=5.0, ordering_cost=0.5),
    )

    assert metrics.total_replan_count == 2
    assert metrics.total_tool_call_count == 6
    assert metrics.fill_rate is not None
    assert metrics.total_cost is not None
    assert metrics.holding_cost > 0.0
    assert metrics.backlog_cost > 0.0
    assert metrics.ordering_cost == 0.0


def test_period_metric_helpers_return_provisional_cost_and_service_values() -> None:
    state = SimulationState(
        benchmark_id="serial_3_echelon",
        time_index=0,
        inventory_level=(20.0, 30.0, 40.0),
        backorder_level=(1.0, 0.0, 0.0),
        regime_label=RegimeLabel.DEMAND_REGIME_SHIFT,
    )
    observation = Observation(
        time_index=0,
        demand_evidence=DemandEvidence(history=(10.0,), latest_realization=(10.0,)),
        leadtime_evidence=LeadTimeEvidence(history=(2.0,), latest_realization=(2.0,)),
        regime_label=RegimeLabel.DEMAND_REGIME_SHIFT,
    )
    signal = AgentSignal(
        selected_subgoal=OperationalSubgoal.REQUEST_REPLAN,
        tool_sequence=("forecast_tool",),
        request_replan=True,
        rationale="Controlled rollout signal.",
    )
    result = OptimizationResult(replenishment_orders=(5.0, 0.0, 0.0), planning_horizon=1)
    record = PeriodTraceRecord(
        time_index=0,
        regime_label=RegimeLabel.DEMAND_REGIME_SHIFT,
        state=state,
        observation=observation,
        agent_signal=signal,
        optimization_result=result,
        realized_demand=10.0,
        demand_load=11.0,
        served_demand=10.0,
        unmet_demand=1.0,
    )

    breakdown = compute_period_cost_breakdown(
        record,
        CostConfig(holding_cost=1.0, backorder_cost=5.0, ordering_cost=0.5),
    )

    assert breakdown == (90.0, 5.0, 2.5, 0.0)
    assert compute_period_total_cost(
        record,
        CostConfig(holding_cost=1.0, backorder_cost=5.0, ordering_cost=0.5),
    ) == 97.5
    assert compute_period_fill_rate(record) == 10.0 / 11.0
