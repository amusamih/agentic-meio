"""Boundary skeleton for the first canonical serial benchmark milestone."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from meio.agents.runtime import OrchestrationRequest
from meio.config.schemas import BenchmarkConfig, CostConfig, SerialStageConfig, SerialSystemConfig
from meio.contracts import (
    BackorderPolicy,
    BenchmarkFamily,
    MissionSpec,
    OperationalSubgoal,
    RegimeLabel,
    UpdateRequestType,
)
from meio.data.stockpyl_adapter import StockpylSerialAdapter, StockpylSerialInstance
from meio.simulation.evidence import DemandEvidence, LeadTimeEvidence, RuntimeEvidence
from meio.simulation.state import Observation, SimulationState

if TYPE_CHECKING:
    from meio.data.external_evidence import ExternalEvidenceBatch

DEFAULT_SERIAL_ECHELON_COUNT = 3
FUTURE_READY_SERIAL_ECHELON_COUNTS = frozenset({3, 5})
REGIME_DEMAND_MULTIPLIERS = {
    RegimeLabel.NORMAL: 1.0,
    RegimeLabel.DEMAND_REGIME_SHIFT: 1.4,
    RegimeLabel.SUPPLY_DISRUPTION: 1.0,
    RegimeLabel.JOINT_DISRUPTION: 1.35,
    RegimeLabel.RECOVERY: 1.1,
}
REGIME_LEADTIME_MULTIPLIERS = {
    RegimeLabel.NORMAL: 1.0,
    RegimeLabel.DEMAND_REGIME_SHIFT: 1.0,
    RegimeLabel.SUPPLY_DISRUPTION: 1.5,
    RegimeLabel.JOINT_DISRUPTION: 1.5,
    RegimeLabel.RECOVERY: 1.0,
}


def _build_demand_history(
    baseline_value: float,
    regime_label: RegimeLabel,
    history_window: int,
    previous_regime_label: RegimeLabel | None = None,
) -> tuple[float, ...]:
    current_value = baseline_value * REGIME_DEMAND_MULTIPLIERS[regime_label]
    if history_window <= 1:
        return (current_value,)
    if (
        regime_label is RegimeLabel.DEMAND_REGIME_SHIFT
        and previous_regime_label is RegimeLabel.DEMAND_REGIME_SHIFT
    ):
        previous_value = (
            baseline_value * REGIME_DEMAND_MULTIPLIERS[previous_regime_label]
        )
        template = (baseline_value, previous_value, current_value)
    elif regime_label is RegimeLabel.RECOVERY:
        template = (baseline_value * 1.4, baseline_value * 1.2, current_value)
    else:
        template = (baseline_value, baseline_value, current_value)
    return template[-history_window:]


def _build_leadtime_history(
    baseline_value: float,
    regime_label: RegimeLabel,
    history_window: int,
    previous_regime_label: RegimeLabel | None = None,
) -> tuple[float, ...]:
    current_value = baseline_value * REGIME_LEADTIME_MULTIPLIERS[regime_label]
    if history_window <= 1:
        return (current_value,)
    if regime_label is RegimeLabel.SUPPLY_DISRUPTION:
        template = (baseline_value, baseline_value * 1.25, current_value)
    elif regime_label is RegimeLabel.JOINT_DISRUPTION:
        template = (baseline_value, baseline_value * 1.25, current_value)
    elif regime_label is RegimeLabel.RECOVERY:
        template = (baseline_value * 1.5, baseline_value * 1.25, current_value)
    else:
        template = (baseline_value, baseline_value, current_value)
    return template[-history_window:]


@dataclass(frozen=True, slots=True)
class SerialBenchmarkCase:
    """A structured serial benchmark boundary backed by the active adapter path."""

    benchmark_id: str
    benchmark_config: BenchmarkConfig
    adapter_name: str
    topology: str
    echelon_count: int
    stage_names: tuple[str, ...]
    initial_inventory: tuple[int, ...]
    base_stock_levels: tuple[float, ...]
    shipment_lead_times: tuple[int, ...]
    holding_costs: tuple[float, ...]
    stockout_costs: tuple[float, ...]
    stockpyl_instance: StockpylSerialInstance
    milestone_notes: str


@dataclass(frozen=True, slots=True)
class SerialPeriodTransition:
    """Typed state transition for the controlled serial rollout."""

    next_state: SimulationState
    realized_demand: float
    demand_load: float
    served_demand: float
    unmet_demand: float
    outbound_shipments: tuple[float, ...] = ()
    scheduled_arrivals: tuple[float, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.next_state, SimulationState):
            raise TypeError("next_state must be a SimulationState.")
        for field_name in (
            "realized_demand",
            "demand_load",
            "served_demand",
            "unmet_demand",
        ):
            value = getattr(self, field_name)
            if value < 0.0:
                raise ValueError(f"{field_name} must be non-negative.")
        object.__setattr__(self, "outbound_shipments", tuple(self.outbound_shipments))
        object.__setattr__(self, "scheduled_arrivals", tuple(self.scheduled_arrivals))
        object.__setattr__(self, "notes", tuple(self.notes))
        for field_name in ("outbound_shipments", "scheduled_arrivals"):
            for value in getattr(self, field_name):
                if value < 0.0:
                    raise ValueError(f"{field_name} must contain non-negative values.")
        for note in self.notes:
            if not note.strip():
                raise ValueError("notes must contain non-empty strings.")


def validate_serial_benchmark_config(config: BenchmarkConfig) -> BenchmarkConfig:
    """Validate the boundary assumptions for the first serial benchmark."""

    if config.benchmark_family is not BenchmarkFamily.SERIAL:
        raise ValueError("The first benchmark boundary only supports the serial family.")
    if config.topology != "serial":
        raise ValueError("The first benchmark boundary only supports serial topology.")
    if config.service_model is not BackorderPolicy.BACKORDERS:
        raise ValueError("The first benchmark boundary only supports backorders.")
    if config.echelon_count not in FUTURE_READY_SERIAL_ECHELON_COUNTS:
        raise ValueError("The serial benchmark boundary currently supports 3 or 5 echelons.")
    return config


def build_serial_benchmark_case(
    config: BenchmarkConfig | None = None,
    adapter: StockpylSerialAdapter | None = None,
) -> SerialBenchmarkCase:
    """Build a typed serial benchmark case through the active Stockpyl adapter."""

    resolved_config = validate_serial_benchmark_config(config or _default_benchmark_config())
    resolved_adapter = adapter or StockpylSerialAdapter()
    stockpyl_instance = resolved_adapter.build_instance(resolved_config)
    return SerialBenchmarkCase(
        benchmark_id=stockpyl_instance.benchmark_id,
        benchmark_config=resolved_config,
        adapter_name=stockpyl_instance.adapter_name,
        topology=stockpyl_instance.topology,
        echelon_count=stockpyl_instance.echelon_count,
        stage_names=stockpyl_instance.stage_names,
        initial_inventory=stockpyl_instance.initial_inventory,
        base_stock_levels=stockpyl_instance.base_stock_levels,
        shipment_lead_times=stockpyl_instance.shipment_lead_times,
        holding_costs=stockpyl_instance.holding_costs,
        stockout_costs=stockpyl_instance.stockout_costs,
        stockpyl_instance=stockpyl_instance,
        milestone_notes=(
            "Stockpyl-backed serial benchmark boundary with explicit in-transit queues, "
            "internal order backlogs, and stage-weighted operating costs."
        ),
    )


def build_serial_orchestration_request(
    case: SerialBenchmarkCase,
    mission: MissionSpec,
    system_state: SimulationState | None = None,
    observation: Observation | None = None,
    evidence: RuntimeEvidence | None = None,
    requested_subgoal: OperationalSubgoal = OperationalSubgoal.INSPECT_EVIDENCE,
    candidate_tool_ids: tuple[str, ...] = (),
    recent_regime_history: tuple[RegimeLabel, ...] = (),
    recent_stress_reference_demand_value: float | None = None,
    recent_update_request_history: tuple[tuple[UpdateRequestType, ...], ...] = (),
) -> OrchestrationRequest:
    """Build a typed orchestration request for the serial benchmark."""

    resolved_state = system_state or build_initial_simulation_state(case)
    resolved_observation = observation or build_initial_observation(case, resolved_state)
    resolved_evidence = evidence or build_runtime_evidence(case, resolved_observation)

    return OrchestrationRequest(
        mission=mission,
        system_state=resolved_state,
        observation=resolved_observation,
        evidence=resolved_evidence,
        requested_subgoal=requested_subgoal,
        candidate_tool_ids=candidate_tool_ids,
        recent_regime_history=recent_regime_history,
        recent_stress_reference_demand_value=recent_stress_reference_demand_value,
        recent_update_request_history=recent_update_request_history,
    )


def build_initial_simulation_state(
    case: SerialBenchmarkCase,
    time_index: int = 0,
    regime_label: RegimeLabel = RegimeLabel.NORMAL,
) -> SimulationState:
    """Build the initial typed simulation state for the serial smoke path."""

    return SimulationState(
        benchmark_id=case.benchmark_id,
        time_index=time_index,
        inventory_level=tuple(float(value) for value in case.initial_inventory),
        stage_names=case.stage_names,
        pipeline_inventory=tuple(0.0 for _ in case.stage_names),
        in_transit_inventory=tuple(
            tuple(0.0 for _ in range(max(0, lead_time)))
            for lead_time in case.shipment_lead_times
        ),
        backorder_level=tuple(0.0 for _ in case.stage_names),
        regime_label=regime_label,
    )


def build_initial_observation(
    case: SerialBenchmarkCase,
    system_state: SimulationState | None = None,
    time_index: int = 0,
) -> Observation:
    """Build the initial typed observation for the Stockpyl serial path."""

    resolved_state = system_state or build_initial_simulation_state(case, time_index=time_index)
    return build_period_observation(case, resolved_state, resolved_state.regime_label)


def build_period_observation(
    case: SerialBenchmarkCase,
    system_state: SimulationState,
    regime_label: RegimeLabel,
    previous_regime_label: RegimeLabel | None = None,
) -> Observation:
    """Build a typed observation for one scheduled rollout period."""

    stockpyl_instance = case.stockpyl_instance
    baseline_demand = float(stockpyl_instance.demand_mean)
    baseline_leadtime = float(stockpyl_instance.primary_inbound_lead_time)
    demand_level = baseline_demand * REGIME_DEMAND_MULTIPLIERS[regime_label]
    leadtime_value = baseline_leadtime * REGIME_LEADTIME_MULTIPLIERS[regime_label]
    upstream_stage_index = stockpyl_instance.primary_inbound_stage_index
    downstream_stage_index = stockpyl_instance.downstream_stage.stage_index
    demand_evidence = DemandEvidence(
        history=_build_demand_history(
            baseline_demand,
            regime_label,
            stockpyl_instance.evidence_history_window,
            previous_regime_label=previous_regime_label,
        ),
        latest_realization=(demand_level,),
        stage_index=downstream_stage_index,
    )
    leadtime_evidence = LeadTimeEvidence(
        history=_build_leadtime_history(
            baseline_leadtime,
            regime_label,
            stockpyl_instance.evidence_history_window,
            previous_regime_label=previous_regime_label,
        ),
        latest_realization=(leadtime_value,),
        upstream_stage_index=upstream_stage_index,
        downstream_stage_index=downstream_stage_index,
    )
    return Observation(
        time_index=system_state.time_index,
        demand_evidence=demand_evidence,
        leadtime_evidence=leadtime_evidence,
        regime_label=regime_label,
        notes=("stockpyl_serial_observation",),
    )


def build_runtime_evidence(
    case: SerialBenchmarkCase,
    observation: Observation,
    *,
    external_evidence_batch: "ExternalEvidenceBatch | None" = None,
) -> RuntimeEvidence:
    """Build the typed combined runtime evidence envelope for the serial path."""

    return RuntimeEvidence(
        time_index=observation.time_index,
        demand=observation.demand_evidence,
        leadtime=observation.leadtime_evidence,
        scenario_families=case.stockpyl_instance.scenario_families,
        demand_baseline_value=float(case.stockpyl_instance.demand_mean),
        leadtime_baseline_value=float(case.stockpyl_instance.primary_inbound_lead_time),
        external_evidence_batch=external_evidence_batch,
        notes=(case.milestone_notes, case.adapter_name),
    )


def regime_for_period(
    regime_schedule: tuple[RegimeLabel, ...],
    time_index: int,
) -> RegimeLabel:
    """Return the scheduled regime label for one controlled rollout period."""

    if not regime_schedule:
        raise ValueError("regime_schedule must not be empty.")
    if time_index < 0:
        raise ValueError("time_index must be non-negative.")
    if time_index >= len(regime_schedule):
        raise ValueError("time_index exceeds the configured regime schedule.")
    return regime_schedule[time_index]


def advance_serial_state(
    case: SerialBenchmarkCase,
    current_state: SimulationState,
    observation: Observation,
    optimization_result: object,
    next_regime: RegimeLabel | None = None,
) -> SerialPeriodTransition:
    """Advance the serial benchmark state with an explicit queued shipment rule.

    Orders are treated as stage-level replenishment requests. Internal stages
    ship against downstream requests using on-hand inventory, scheduled
    in-transit queues capture shipment lead times, and the upstream-most stage
    is replenished by an exogenous supplier through the same queue logic.
    """

    from meio.optimization.contracts import OptimizationResult

    if not isinstance(optimization_result, OptimizationResult):
        raise TypeError("optimization_result must be an OptimizationResult.")
    if len(optimization_result.replenishment_orders) != len(current_state.inventory_level):
        raise ValueError(
            "optimization_result.replenishment_orders must match the inventory dimension."
        )
    if len(case.shipment_lead_times) != len(current_state.inventory_level):
        raise ValueError("case.shipment_lead_times must match the inventory dimension.")
    if current_state.in_transit_inventory:
        current_queues = tuple(current_state.in_transit_inventory)
    else:
        current_queues = tuple(
            tuple(0.0 for _ in range(max(0, lead_time)))
            for lead_time in case.shipment_lead_times
        )
    realized_demand = observation.demand_realization[-1]
    current_backorders = tuple(current_state.backorder_level) or tuple(
        0.0 for _ in range(len(current_state.inventory_level))
    )
    demand_load = current_backorders[0] + realized_demand
    replenishment_orders = optimization_result.replenishment_orders
    stage_count = len(replenishment_orders)
    on_hand_inventory = list(current_state.inventory_level)
    next_backorder = [0.0 for _ in range(stage_count)]
    outbound_shipments = [0.0 for _ in range(stage_count)]
    scheduled_arrivals = [
        float(queue[0]) if queue else 0.0
        for queue in current_queues
    ]
    next_queues: list[list[float]] = [
        (list(queue[1:]) + [0.0]) if queue else []
        for queue in current_queues
    ]

    served_demand = min(on_hand_inventory[0], demand_load)
    on_hand_inventory[0] -= served_demand
    unmet_demand = demand_load - served_demand
    next_backorder[0] = unmet_demand
    outbound_shipments[0] = served_demand

    for stage_index in range(1, stage_count):
        requested_by_downstream = (
            current_backorders[stage_index] + replenishment_orders[stage_index - 1]
        )
        shipped_to_downstream = min(on_hand_inventory[stage_index], requested_by_downstream)
        on_hand_inventory[stage_index] -= shipped_to_downstream
        next_backorder[stage_index] = requested_by_downstream - shipped_to_downstream
        outbound_shipments[stage_index] = shipped_to_downstream
        downstream_queue = next_queues[stage_index - 1]
        if downstream_queue:
            downstream_queue[-1] += shipped_to_downstream
        else:
            scheduled_arrivals[stage_index - 1] += shipped_to_downstream

    upstream_replenishment = replenishment_orders[-1]
    upstream_queue = next_queues[-1]
    if upstream_queue:
        upstream_queue[-1] += upstream_replenishment
    else:
        scheduled_arrivals[-1] += upstream_replenishment

    next_inventory = tuple(
        on_hand_inventory[index] + scheduled_arrivals[index]
        for index in range(stage_count)
    )
    next_in_transit_inventory = tuple(tuple(queue) for queue in next_queues)
    next_pipeline = tuple(
        float(sum(queue))
        for queue in next_in_transit_inventory
    )
    next_state = SimulationState(
        benchmark_id=case.benchmark_id,
        time_index=current_state.time_index + 1,
        inventory_level=next_inventory,
        stage_names=current_state.stage_names,
        pipeline_inventory=next_pipeline,
        in_transit_inventory=next_in_transit_inventory,
        backorder_level=tuple(next_backorder),
        regime_label=next_regime or current_state.regime_label,
    )
    return SerialPeriodTransition(
        next_state=next_state,
        realized_demand=realized_demand,
        demand_load=demand_load,
        served_demand=served_demand,
        unmet_demand=unmet_demand,
        outbound_shipments=tuple(outbound_shipments),
        scheduled_arrivals=tuple(scheduled_arrivals),
        notes=(
            "leadtime_queue_serial_rollout",
            "internal_backorders_explicit",
            "stage_weighted_costs_supported",
        ),
    )


def _default_benchmark_config() -> BenchmarkConfig:
    stages = tuple(
        SerialStageConfig(
            stage_index=index,
            stage_name=f"stage_{index}",
            initial_inventory=0,
            shipment_lead_time=2,
            base_stock_level=0,
        )
        for index in range(1, DEFAULT_SERIAL_ECHELON_COUNT + 1)
    )
    return BenchmarkConfig(
        benchmark_family=BenchmarkFamily.SERIAL,
        system=SerialSystemConfig(
            topology="serial",
            echelon_count=DEFAULT_SERIAL_ECHELON_COUNT,
            stages=stages,
        ),
        costs=CostConfig(holding_cost=1.0, backorder_cost=5.0, ordering_cost=0.5),
        service_model=BackorderPolicy.BACKORDERS,
        scenario_families=(RegimeLabel.NORMAL.value,),
        random_seed=0,
        demand_mean=10.0,
    )


__all__ = [
    "DEFAULT_SERIAL_ECHELON_COUNT",
    "REGIME_DEMAND_MULTIPLIERS",
    "REGIME_LEADTIME_MULTIPLIERS",
    "SerialBenchmarkCase",
    "SerialPeriodTransition",
    "advance_serial_state",
    "build_initial_observation",
    "build_initial_simulation_state",
    "build_period_observation",
    "build_serial_benchmark_case",
    "build_serial_orchestration_request",
    "build_runtime_evidence",
    "regime_for_period",
    "validate_serial_benchmark_config",
]
