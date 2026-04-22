from __future__ import annotations

from importlib.util import find_spec

import pytest

from meio.config.schemas import BenchmarkConfig, CostConfig, SerialStageConfig, SerialSystemConfig
from meio.contracts import BackorderPolicy, BenchmarkFamily, MissionSpec, OperationalSubgoal, RegimeLabel
from meio.optimization.contracts import OptimizationResult
from meio.simulation.serial_benchmark import (
    DEFAULT_SERIAL_ECHELON_COUNT,
    advance_serial_state,
    build_initial_observation,
    build_initial_simulation_state,
    build_period_observation,
    build_serial_benchmark_case,
    build_serial_orchestration_request,
    build_runtime_evidence,
    regime_for_period,
    validate_serial_benchmark_config,
)
from meio.simulation.state import SimulationState

pytestmark = pytest.mark.skipif(find_spec("stockpyl") is None, reason="stockpyl not installed")


def test_validate_serial_benchmark_config_accepts_first_milestone_case() -> None:
    config = BenchmarkConfig(
        benchmark_family=BenchmarkFamily.SERIAL,
        system=SerialSystemConfig(
            topology="serial",
            echelon_count=3,
            stages=(
                SerialStageConfig(stage_index=1, stage_name="retailer", initial_inventory=20),
                SerialStageConfig(stage_index=2, stage_name="regional_dc", initial_inventory=30),
                SerialStageConfig(stage_index=3, stage_name="plant", initial_inventory=40),
            ),
        ),
        costs=CostConfig(holding_cost=1.0, backorder_cost=5.0, ordering_cost=0.5),
        service_model=BackorderPolicy.BACKORDERS,
        scenario_families=("normal", "demand_regime_shift"),
        random_seed=7,
    )

    validated = validate_serial_benchmark_config(config)

    assert validated is config


def test_build_serial_benchmark_case_uses_default_three_echelon_boundary() -> None:
    case = build_serial_benchmark_case()

    assert case.benchmark_config.echelon_count == DEFAULT_SERIAL_ECHELON_COUNT
    assert case.stage_names == ("stage_1", "stage_2", "stage_3")
    assert case.adapter_name == "stockpyl_serial"
    assert "Stockpyl-backed" in case.milestone_notes


def test_build_serial_orchestration_request_exposes_stockpyl_backed_state_and_evidence() -> None:
    case = build_serial_benchmark_case()
    mission = MissionSpec(
        mission_id="serial_runtime",
        objective="Inspect evidence and remain inside optimizer guardrails.",
    )

    request = build_serial_orchestration_request(
        case=case,
        mission=mission,
        requested_subgoal=OperationalSubgoal.INSPECT_EVIDENCE,
    )

    assert request.system_state is not None
    assert request.system_state.benchmark_id == "serial_3_echelon"
    assert request.evidence is not None
    assert request.evidence.scenario_families == (RegimeLabel.NORMAL,)
    assert request.requested_subgoal is OperationalSubgoal.INSPECT_EVIDENCE


def test_serial_benchmark_builds_typed_state_observation_and_evidence() -> None:
    case = build_serial_benchmark_case()
    state = build_initial_simulation_state(case)
    observation = build_initial_observation(case, state)
    evidence = build_runtime_evidence(case, observation)

    assert state.stage_names == ("stage_1", "stage_2", "stage_3")
    assert observation.demand_realization == (10.0,)
    assert observation.leadtime_realization == (2.0,)
    assert evidence.scenario_families == (RegimeLabel.NORMAL,)
    assert evidence.demand_baseline_value == 10.0
    assert evidence.leadtime_baseline_value == 2.0


def test_serial_benchmark_default_case_carries_stockpyl_instance_metadata() -> None:
    case = build_serial_benchmark_case()

    assert case.stockpyl_instance.stage_names == ("stage_1", "stage_2", "stage_3")
    assert case.stockpyl_instance.primary_inbound_stage_index == 2


def test_regime_schedule_application_changes_period_observation() -> None:
    case = build_serial_benchmark_case()
    schedule = (
        RegimeLabel.NORMAL,
        RegimeLabel.DEMAND_REGIME_SHIFT,
        RegimeLabel.RECOVERY,
    )
    initial_state = build_initial_simulation_state(case, regime_label=regime_for_period(schedule, 0))
    shifted_state = build_initial_simulation_state(case, time_index=1, regime_label=regime_for_period(schedule, 1))

    normal_observation = build_period_observation(case, initial_state, regime_for_period(schedule, 0))
    shifted_observation = build_period_observation(case, shifted_state, regime_for_period(schedule, 1))

    assert normal_observation.demand_realization == (10.0,)
    assert shifted_observation.demand_realization == (14.0,)
    assert shifted_observation.demand_evidence.history[-2] == 10.0
    assert shifted_observation.demand_evidence.history[-1] == 14.0


def test_recovery_observation_history_shows_return_toward_baseline() -> None:
    case = build_serial_benchmark_case()
    recovery_state = build_initial_simulation_state(case, time_index=2, regime_label=RegimeLabel.RECOVERY)

    recovery_observation = build_period_observation(case, recovery_state, RegimeLabel.RECOVERY)

    assert recovery_observation.demand_realization == (11.0,)
    assert recovery_observation.demand_evidence.history == (14.0, 12.0, 11.0)


def test_advance_serial_state_applies_simple_backorder_progression() -> None:
    case = build_serial_benchmark_case()
    state = build_initial_simulation_state(case, regime_label=RegimeLabel.DEMAND_REGIME_SHIFT)
    observation = build_period_observation(case, state, RegimeLabel.DEMAND_REGIME_SHIFT)
    transition = advance_serial_state(
        case,
        current_state=state,
        observation=observation,
        optimization_result=OptimizationResult(
            replenishment_orders=(0.0, 0.0, 0.0),
            planning_horizon=1,
        ),
        next_regime=RegimeLabel.RECOVERY,
    )

    assert transition.realized_demand == 14.0
    assert transition.served_demand == 0.0
    assert transition.unmet_demand == 14.0
    assert transition.next_state.time_index == 1
    assert transition.next_state.regime_label is RegimeLabel.RECOVERY


def test_advance_serial_state_responds_to_nonzero_optimizer_orders() -> None:
    case = build_serial_benchmark_case()
    state = SimulationState(
        benchmark_id=case.benchmark_id,
        time_index=0,
        inventory_level=(0.0, 20.0, 30.0),
        stage_names=case.stage_names,
        pipeline_inventory=(0.0, 0.0, 0.0),
        backorder_level=(0.0, 0.0, 0.0),
        regime_label=RegimeLabel.DEMAND_REGIME_SHIFT,
    )
    observation = build_period_observation(case, state, RegimeLabel.DEMAND_REGIME_SHIFT)
    transition = advance_serial_state(
        case,
        current_state=state,
        observation=observation,
        optimization_result=OptimizationResult(
            replenishment_orders=(18.0, 10.0, 5.0),
            planning_horizon=1,
        ),
        next_regime=RegimeLabel.RECOVERY,
    )

    assert transition.served_demand == 0.0
    assert transition.unmet_demand == 14.0
    assert transition.outbound_shipments == (0.0, 18.0, 10.0)
    assert transition.next_state.inventory_level == (0.0, 2.0, 20.0)
    assert transition.next_state.pipeline_inventory == (18.0, 10.0, 5.0)
    assert transition.next_state.in_transit_inventory == ((0.0, 18.0), (0.0, 10.0), (0.0, 5.0))
