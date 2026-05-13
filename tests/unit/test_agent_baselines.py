from __future__ import annotations

from meio.agents.baselines import DeterministicBaselinePolicy
from meio.agents.uncertainty_baselines import (
    RobustUncertaintyPolicy,
    ScenarioRollingHorizonPolicy,
)
from meio.config.schemas import RobustPolicyConfig, ScenarioRollingHorizonPolicyConfig
from meio.contracts import (
    OperationalSubgoal,
    RegimeLabel,
    UpdateRequestType,
)
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


def test_robust_policy_uses_empirical_quantile_without_llm_or_tools() -> None:
    case = build_serial_benchmark_case()
    state = build_initial_simulation_state(case, regime_label=RegimeLabel.DEMAND_REGIME_SHIFT)
    observation = build_period_observation(case, state, RegimeLabel.DEMAND_REGIME_SHIFT)
    evidence = build_runtime_evidence(case, observation)

    decision = RobustUncertaintyPolicy(
        RobustPolicyConfig(window_length=3, quantile=0.80, safety_buffer_scale=1.05)
    ).decide(state, observation, evidence, case)

    assert decision.signal.tool_sequence == ()
    assert decision.signal.selected_tool_id is None
    assert decision.scenario_update_result.adjustment.safety_buffer_scale == 1.05
    assert decision.scenario_update_result.provenance == "robust_uncertainty_protected_policy"


def test_rolling_horizon_policy_uses_configured_scenario_search_without_llm() -> None:
    case = build_serial_benchmark_case()
    state = build_initial_simulation_state(case, regime_label=RegimeLabel.DEMAND_REGIME_SHIFT)
    observation = build_period_observation(case, state, RegimeLabel.DEMAND_REGIME_SHIFT)
    evidence = build_runtime_evidence(case, observation)

    decision = ScenarioRollingHorizonPolicy(
        ScenarioRollingHorizonPolicyConfig(
            horizon_length=3,
            scenario_count=4,
            random_seed=7,
        )
    ).decide(state, observation, evidence, case)

    assert decision.signal.tool_sequence == ()
    assert decision.signal.selected_tool_id is None
    assert decision.scenario_update_result.provenance == "scenario_rolling_horizon_policy"
    assert decision.scenario_update_result.adjustment.demand_outlook > 0.0
