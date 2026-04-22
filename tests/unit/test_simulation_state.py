from __future__ import annotations

from meio.agents.telemetry import OrchestrationStepTelemetry
from meio.contracts import AgentSignal, OperationalSubgoal, RegimeLabel
from meio.optimization.contracts import OptimizationResult
from meio.simulation.evidence import DemandEvidence, LeadTimeEvidence
from meio.simulation.state import EpisodeTrace, Observation, PeriodTraceRecord, SimulationState


def test_simulation_state_accepts_serial_boundary_values() -> None:
    state = SimulationState(
        benchmark_id="serial_3_echelon",
        time_index=0,
        inventory_level=(20.0, 30.0, 40.0),
        stage_names=("retailer", "regional_dc", "plant"),
        pipeline_inventory=(5.0, 3.0, 1.0),
        backorder_level=(0.0, 0.0, 0.0),
        regime_label=RegimeLabel.NORMAL,
    )

    assert state.inventory_level == (20.0, 30.0, 40.0)
    assert state.stage_names == ("retailer", "regional_dc", "plant")
    assert state.regime_label is RegimeLabel.NORMAL


def test_episode_trace_accepts_typed_entries() -> None:
    trace = EpisodeTrace(
        run_id="run_001",
        benchmark_id="serial_3_echelon",
        states=(
            SimulationState(
                benchmark_id="serial_3_echelon",
                time_index=0,
                inventory_level=(20.0, 30.0, 40.0),
            ),
        ),
        observations=(
            Observation(
                time_index=0,
                demand_evidence=DemandEvidence(
                    history=(8.0, 8.0, 9.0),
                    latest_realization=(9.0,),
                    stage_index=1,
                ),
                leadtime_evidence=LeadTimeEvidence(
                    history=(2.0, 2.0, 2.0),
                    latest_realization=(2.0,),
                    upstream_stage_index=3,
                    downstream_stage_index=2,
                ),
            ),
        ),
        agent_signals=(
            AgentSignal(
                selected_subgoal=OperationalSubgoal.NO_ACTION,
                no_action=True,
                rationale="No bounded intervention requested.",
            ),
        ),
        optimization_results=(
            OptimizationResult(
                replenishment_orders=(5.0, 4.0, 3.0),
                planning_horizon=3,
            ),
        ),
    )

    assert trace.benchmark_id == "serial_3_echelon"
    assert trace.observations[0].demand_realization == (9.0,)
    assert trace.optimization_results[0].replenishment_orders == (5.0, 4.0, 3.0)
    assert trace.trace_length == 1


def test_episode_trace_append_period_supports_multi_period_growth() -> None:
    state = SimulationState(
        benchmark_id="serial_3_echelon",
        time_index=0,
        inventory_level=(20.0, 30.0, 40.0),
        backorder_level=(0.0, 0.0, 0.0),
    )
    observation = Observation(
        time_index=0,
        demand_evidence=DemandEvidence(history=(8.0, 9.0, 10.0), latest_realization=(10.0,)),
        leadtime_evidence=LeadTimeEvidence(history=(2.0, 2.0, 2.0), latest_realization=(2.0,)),
        regime_label=RegimeLabel.NORMAL,
    )
    signal = AgentSignal(
        selected_subgoal=OperationalSubgoal.REQUEST_REPLAN,
        tool_sequence=("forecast_tool", "leadtime_tool", "scenario_tool"),
        request_replan=True,
        rationale="Typed runtime requested replanning.",
    )
    result = OptimizationResult(replenishment_orders=(0.0, 0.0, 0.0), planning_horizon=1)
    record = PeriodTraceRecord(
        time_index=0,
        regime_label=RegimeLabel.NORMAL,
        state=state,
        observation=observation,
        agent_signal=signal,
        optimization_result=result,
        realized_demand=10.0,
        demand_load=10.0,
        served_demand=10.0,
        unmet_demand=0.0,
        step_telemetry=OrchestrationStepTelemetry(
            orchestration_latency_ms=3.0,
            tool_call_count=3,
            selected_tools=("forecast_tool", "leadtime_tool", "scenario_tool"),
            request_replan=True,
        ),
        notes=("period_0",),
    )

    trace = EpisodeTrace(run_id="run_002", benchmark_id="serial_3_echelon").append_period(record)

    assert trace.trace_length == 1
    assert trace.period_records[0].served_demand == 10.0
    assert trace.states[0].time_index == 0
    assert trace.step_telemetry[0].tool_call_count == 3
