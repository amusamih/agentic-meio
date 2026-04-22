from __future__ import annotations

from meio.contracts import OperationalSubgoal, RegimeLabel, ToolClass, ToolInvocation
from meio.agents.baselines import DeterministicBaselinePolicy
from meio.forecasting.adapters import DeterministicForecastTool
from meio.forecasting.contracts import ForecastResult
from meio.leadtime.adapters import DeterministicLeadTimeTool
from meio.leadtime.contracts import LeadTimeResult
from meio.optimization.adapters import TrustedOptimizerAdapter, build_optimization_request
from meio.optimization.contracts import OptimizationResult
from meio.scenarios.adapters import DeterministicScenarioTool
from meio.scenarios.contracts import ScenarioUpdateResult
from meio.simulation.serial_benchmark import (
    build_initial_observation,
    build_initial_simulation_state,
    build_period_observation,
    build_runtime_evidence,
    build_serial_benchmark_case,
)
from meio.simulation.state import SimulationState


def test_deterministic_forecast_tool_returns_typed_result() -> None:
    case = build_serial_benchmark_case()
    state = build_initial_simulation_state(case)
    observation = build_initial_observation(case, state)
    evidence = build_runtime_evidence(case, observation)
    result = DeterministicForecastTool().invoke(
        ToolInvocation(
            tool_id="forecast_tool",
            tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
            subgoal=OperationalSubgoal.QUERY_UNCERTAINTY,
            system_state=state,
            observation=observation,
            evidence=evidence,
        )
    )

    assert isinstance(result.structured_output["forecast_result"], ForecastResult)


def test_deterministic_leadtime_tool_returns_typed_result() -> None:
    case = build_serial_benchmark_case()
    state = build_initial_simulation_state(case)
    observation = build_initial_observation(case, state)
    evidence = build_runtime_evidence(case, observation)
    result = DeterministicLeadTimeTool().invoke(
        ToolInvocation(
            tool_id="leadtime_tool",
            tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
            subgoal=OperationalSubgoal.QUERY_UNCERTAINTY,
            system_state=state,
            observation=observation,
            evidence=evidence,
        )
    )

    assert isinstance(result.structured_output["leadtime_result"], LeadTimeResult)


def test_deterministic_scenario_tool_and_optimizer_preserve_order_boundary() -> None:
    case = build_serial_benchmark_case()
    state = build_initial_simulation_state(case)
    observation = build_initial_observation(case, state)
    evidence = build_runtime_evidence(case, observation)
    forecast_result = DeterministicForecastTool().invoke(
        ToolInvocation(
            tool_id="forecast_tool",
            tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
            subgoal=OperationalSubgoal.QUERY_UNCERTAINTY,
            system_state=state,
            observation=observation,
            evidence=evidence,
        )
    )
    leadtime_result = DeterministicLeadTimeTool().invoke(
        ToolInvocation(
            tool_id="leadtime_tool",
            tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
            subgoal=OperationalSubgoal.QUERY_UNCERTAINTY,
            system_state=state,
            observation=observation,
            evidence=evidence,
            prior_results=(forecast_result,),
        )
    )
    scenario_result = DeterministicScenarioTool().invoke(
        ToolInvocation(
            tool_id="scenario_tool",
            tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
            subgoal=OperationalSubgoal.UPDATE_UNCERTAINTY,
            system_state=state,
            observation=observation,
            evidence=evidence,
            prior_results=(forecast_result, leadtime_result),
        )
    )

    typed_scenario_result = scenario_result.structured_output["scenario_update_result"]
    assert isinstance(typed_scenario_result, ScenarioUpdateResult)
    assert scenario_result.emits_raw_orders is False

    optimization_request = build_optimization_request(
        state,
        typed_scenario_result,
        base_stock_levels=case.base_stock_levels,
    )
    optimization_result = TrustedOptimizerAdapter().solve(optimization_request)

    assert isinstance(optimization_result, OptimizationResult)
    assert len(optimization_result.replenishment_orders) == 3
    assert all(order >= 0.0 for order in optimization_result.replenishment_orders)


def test_trusted_optimizer_produces_nonzero_orders_when_state_and_updates_warrant_it() -> None:
    case = build_serial_benchmark_case()
    state = build_initial_simulation_state(case, regime_label=RegimeLabel.DEMAND_REGIME_SHIFT)
    depleted_state = SimulationState(
        benchmark_id=state.benchmark_id,
        time_index=1,
        inventory_level=(10.0, 25.0, 35.0),
        stage_names=state.stage_names,
        pipeline_inventory=state.pipeline_inventory,
        backorder_level=(2.0, 0.0, 0.0),
        regime_label=RegimeLabel.DEMAND_REGIME_SHIFT,
    )
    observation = build_period_observation(
        case,
        depleted_state,
        RegimeLabel.DEMAND_REGIME_SHIFT,
    )
    evidence = build_runtime_evidence(case, observation)
    forecast_result = DeterministicForecastTool().invoke(
        ToolInvocation(
            tool_id="forecast_tool",
            tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
            subgoal=OperationalSubgoal.QUERY_UNCERTAINTY,
            system_state=depleted_state,
            observation=observation,
            evidence=evidence,
        )
    )
    leadtime_result = DeterministicLeadTimeTool().invoke(
        ToolInvocation(
            tool_id="leadtime_tool",
            tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
            subgoal=OperationalSubgoal.QUERY_UNCERTAINTY,
            system_state=depleted_state,
            observation=observation,
            evidence=evidence,
            prior_results=(forecast_result,),
        )
    )
    scenario_result = DeterministicScenarioTool().invoke(
        ToolInvocation(
            tool_id="scenario_tool",
            tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
            subgoal=OperationalSubgoal.UPDATE_UNCERTAINTY,
            system_state=depleted_state,
            observation=observation,
            evidence=evidence,
            prior_results=(forecast_result, leadtime_result),
        )
    )

    typed_scenario_result = scenario_result.structured_output["scenario_update_result"]
    optimization_request = build_optimization_request(
        depleted_state,
        typed_scenario_result,
        base_stock_levels=case.base_stock_levels,
    )
    optimization_result = TrustedOptimizerAdapter().solve(optimization_request)

    assert optimization_request.scenario_adjustment.demand_outlook > observation.demand_realization[-1]
    assert any(order > 0.0 for order in optimization_result.replenishment_orders)


def test_update_logic_changes_optimizer_inputs_for_the_same_state() -> None:
    case = build_serial_benchmark_case()
    state = SimulationState(
        benchmark_id=case.benchmark_id,
        time_index=1,
        inventory_level=(10.0, 25.0, 35.0),
        stage_names=case.stage_names,
        pipeline_inventory=(0.0, 0.0, 0.0),
        backorder_level=(2.0, 0.0, 0.0),
        regime_label=RegimeLabel.DEMAND_REGIME_SHIFT,
    )
    observation = build_period_observation(case, state, RegimeLabel.DEMAND_REGIME_SHIFT)
    evidence = build_runtime_evidence(case, observation)

    baseline_decision = DeterministicBaselinePolicy().decide(state, observation, evidence)
    baseline_request = build_optimization_request(
        state,
        baseline_decision.scenario_update_result,
        base_stock_levels=case.base_stock_levels,
    )

    forecast_result = DeterministicForecastTool().invoke(
        ToolInvocation(
            tool_id="forecast_tool",
            tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
            subgoal=OperationalSubgoal.QUERY_UNCERTAINTY,
            system_state=state,
            observation=observation,
            evidence=evidence,
        )
    )
    leadtime_result = DeterministicLeadTimeTool().invoke(
        ToolInvocation(
            tool_id="leadtime_tool",
            tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
            subgoal=OperationalSubgoal.QUERY_UNCERTAINTY,
            system_state=state,
            observation=observation,
            evidence=evidence,
            prior_results=(forecast_result,),
        )
    )
    scenario_result = DeterministicScenarioTool().invoke(
        ToolInvocation(
            tool_id="scenario_tool",
            tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
            subgoal=OperationalSubgoal.UPDATE_UNCERTAINTY,
            system_state=state,
            observation=observation,
            evidence=evidence,
            prior_results=(forecast_result, leadtime_result),
        )
    )
    orchestration_request = build_optimization_request(
        state,
        scenario_result.structured_output["scenario_update_result"],
        base_stock_levels=case.base_stock_levels,
    )

    assert (
        orchestration_request.scenario_adjustment.demand_outlook
        > baseline_request.scenario_adjustment.demand_outlook
    )
    assert (
        orchestration_request.scenario_adjustment.safety_buffer_scale
        > baseline_request.scenario_adjustment.safety_buffer_scale
    )
