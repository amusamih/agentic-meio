from __future__ import annotations

from meio.agents.baselines import DeterministicBaselinePolicy
from meio.contracts import OperationalSubgoal, RegimeLabel, UpdateRequestType
from meio.simulation.serial_benchmark import (
    build_initial_simulation_state,
    build_period_observation,
    build_runtime_evidence,
    build_serial_benchmark_case,
)


def test_deterministic_baseline_requests_replan_for_demand_regime_shift() -> None:
    case = build_serial_benchmark_case()
    state = build_initial_simulation_state(case, regime_label=RegimeLabel.DEMAND_REGIME_SHIFT)
    observation = build_period_observation(case, state, RegimeLabel.DEMAND_REGIME_SHIFT)
    evidence = build_runtime_evidence(case, observation)

    decision = DeterministicBaselinePolicy().decide(state, observation, evidence)

    assert decision.signal.selected_subgoal is OperationalSubgoal.REQUEST_REPLAN
    assert decision.signal.tool_sequence == ()
    assert decision.scenario_update_result.request_replan is True
    assert decision.scenario_update_result.applied_update_types == (
        UpdateRequestType.REWEIGHT_SCENARIOS,
    )
    assert decision.scenario_update_result.adjustment.demand_outlook > 14.0


def test_deterministic_baseline_keeps_current_for_normal_regime() -> None:
    case = build_serial_benchmark_case()
    state = build_initial_simulation_state(case, regime_label=RegimeLabel.NORMAL)
    observation = build_period_observation(case, state, RegimeLabel.NORMAL)
    evidence = build_runtime_evidence(case, observation)

    decision = DeterministicBaselinePolicy().decide(state, observation, evidence)

    assert decision.signal.no_action is True
    assert decision.scenario_update_result.applied_update_types == (
        UpdateRequestType.KEEP_CURRENT,
    )
    assert decision.scenario_update_result.adjustment.safety_buffer_scale == 1.0
