"""Typed simulation state containers for the first serial benchmark path."""

from __future__ import annotations

from dataclasses import dataclass, field

from meio.agents.telemetry import OrchestrationStepTelemetry
from meio.contracts import AgentSignal, RegimeLabel
from meio.optimization.contracts import OptimizationResult
from meio.simulation.evidence import DemandEvidence, LeadTimeEvidence


def _coerce_series(
    values: tuple[float, ...],
    field_name: str,
    *,
    allow_empty: bool,
) -> tuple[float, ...]:
    if not values:
        if allow_empty:
            return ()
        raise ValueError(f"{field_name} must not be empty.")
    result: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{field_name} must contain numeric values.")
        if float(value) < 0.0:
            raise ValueError(f"{field_name} must contain non-negative values.")
        result.append(float(value))
    return tuple(result)


def _coerce_nested_series(
    values: tuple[tuple[float, ...], ...],
    field_name: str,
    *,
    allow_empty: bool,
) -> tuple[tuple[float, ...], ...]:
    if not values:
        if allow_empty:
            return ()
        raise ValueError(f"{field_name} must not be empty.")
    return tuple(
        _coerce_series(tuple(series), field_name, allow_empty=True)
        for series in values
    )


@dataclass(frozen=True, slots=True)
class SimulationState:
    """Compact simulation state for the first serial benchmark path."""

    benchmark_id: str
    time_index: int
    inventory_level: tuple[float, ...]
    stage_names: tuple[str, ...] = field(default_factory=tuple)
    pipeline_inventory: tuple[float, ...] = field(default_factory=tuple)
    in_transit_inventory: tuple[tuple[float, ...], ...] = field(default_factory=tuple)
    backorder_level: tuple[float, ...] = field(default_factory=tuple)
    regime_label: RegimeLabel = RegimeLabel.NORMAL

    def __post_init__(self) -> None:
        if not self.benchmark_id.strip():
            raise ValueError("benchmark_id must be a non-empty string.")
        if self.time_index < 0:
            raise ValueError("time_index must be non-negative.")
        inventory_level = _coerce_series(
            self.inventory_level,
            "inventory_level",
            allow_empty=False,
        )
        pipeline_inventory = _coerce_series(
            self.pipeline_inventory,
            "pipeline_inventory",
            allow_empty=True,
        )
        in_transit_inventory = _coerce_nested_series(
            self.in_transit_inventory,
            "in_transit_inventory",
            allow_empty=True,
        )
        backorder_level = _coerce_series(
            self.backorder_level,
            "backorder_level",
            allow_empty=True,
        )
        if pipeline_inventory and len(pipeline_inventory) != len(inventory_level):
            raise ValueError("pipeline_inventory must match inventory_level length when provided.")
        if backorder_level and len(backorder_level) != len(inventory_level):
            raise ValueError("backorder_level must match inventory_level length when provided.")
        if in_transit_inventory and len(in_transit_inventory) != len(inventory_level):
            raise ValueError(
                "in_transit_inventory must match inventory_level length when provided."
            )
        if in_transit_inventory:
            derived_pipeline_inventory = tuple(
                float(sum(stage_pipeline))
                for stage_pipeline in in_transit_inventory
            )
            if pipeline_inventory:
                if pipeline_inventory != derived_pipeline_inventory:
                    raise ValueError(
                        "pipeline_inventory must match the sum of in_transit_inventory "
                        "when both are provided."
                    )
            else:
                pipeline_inventory = derived_pipeline_inventory
        if not isinstance(self.regime_label, RegimeLabel):
            raise TypeError("regime_label must be a RegimeLabel.")
        object.__setattr__(self, "stage_names", tuple(self.stage_names))
        if self.stage_names and len(self.stage_names) != len(inventory_level):
            raise ValueError("stage_names must match inventory_level length when provided.")
        for stage_name in self.stage_names:
            if not stage_name.strip():
                raise ValueError("stage_names must contain non-empty strings.")
        object.__setattr__(self, "inventory_level", inventory_level)
        object.__setattr__(self, "pipeline_inventory", pipeline_inventory)
        object.__setattr__(self, "in_transit_inventory", in_transit_inventory)
        object.__setattr__(self, "backorder_level", backorder_level)


@dataclass(frozen=True, slots=True)
class Observation:
    """Observable simulation outputs available to bounded tools and agents."""

    time_index: int
    demand_evidence: DemandEvidence
    leadtime_evidence: LeadTimeEvidence
    regime_label: RegimeLabel | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.time_index < 0:
            raise ValueError("time_index must be non-negative.")
        if not isinstance(self.demand_evidence, DemandEvidence):
            raise TypeError("demand_evidence must be a DemandEvidence.")
        if not isinstance(self.leadtime_evidence, LeadTimeEvidence):
            raise TypeError("leadtime_evidence must be a LeadTimeEvidence.")
        if self.regime_label is not None and not isinstance(self.regime_label, RegimeLabel):
            raise TypeError("regime_label must be a RegimeLabel when provided.")
        object.__setattr__(self, "notes", tuple(self.notes))
        for note in self.notes:
            if not note.strip():
                raise ValueError("notes must contain non-empty strings.")

    @property
    def demand_realization(self) -> tuple[float, ...]:
        """Return the most recent observed demand realization."""

        if self.demand_evidence.latest_realization:
            return self.demand_evidence.latest_realization
        return (self.demand_evidence.latest_value,)

    @property
    def leadtime_realization(self) -> tuple[float, ...]:
        """Return the most recent observed lead-time realization."""

        if self.leadtime_evidence.latest_realization:
            return self.leadtime_evidence.latest_realization
        return (self.leadtime_evidence.latest_value,)


@dataclass(frozen=True, slots=True)
class PeriodTraceRecord:
    """Typed per-period trace record for the provisional controlled rollout."""

    time_index: int
    regime_label: RegimeLabel
    state: SimulationState
    observation: Observation
    agent_signal: AgentSignal
    optimization_result: OptimizationResult
    next_state: SimulationState | None = None
    realized_demand: float | None = None
    demand_load: float | None = None
    served_demand: float | None = None
    unmet_demand: float | None = None
    step_telemetry: OrchestrationStepTelemetry | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.time_index < 0:
            raise ValueError("time_index must be non-negative.")
        if not isinstance(self.regime_label, RegimeLabel):
            raise TypeError("regime_label must be a RegimeLabel.")
        if not isinstance(self.state, SimulationState):
            raise TypeError("state must be a SimulationState.")
        if not isinstance(self.observation, Observation):
            raise TypeError("observation must be an Observation.")
        if not isinstance(self.agent_signal, AgentSignal):
            raise TypeError("agent_signal must be an AgentSignal.")
        if not isinstance(self.optimization_result, OptimizationResult):
            raise TypeError("optimization_result must be an OptimizationResult.")
        if self.next_state is not None and not isinstance(self.next_state, SimulationState):
            raise TypeError("next_state must be a SimulationState when provided.")
        if self.step_telemetry is not None and not isinstance(
            self.step_telemetry,
            OrchestrationStepTelemetry,
        ):
            raise TypeError("step_telemetry must be an OrchestrationStepTelemetry when provided.")
        if self.state.time_index != self.time_index:
            raise ValueError("state.time_index must match time_index.")
        if self.observation.time_index != self.time_index:
            raise ValueError("observation.time_index must match time_index.")
        if self.next_state is not None:
            if self.next_state.benchmark_id != self.state.benchmark_id:
                raise ValueError("next_state benchmark_id must match state benchmark_id.")
            if self.next_state.time_index != self.time_index + 1:
                raise ValueError("next_state.time_index must equal time_index + 1.")
        object.__setattr__(self, "notes", tuple(self.notes))
        for field_name in (
            "realized_demand",
            "demand_load",
            "served_demand",
            "unmet_demand",
        ):
            value = getattr(self, field_name)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise TypeError(f"{field_name} must be numeric when provided.")
                if float(value) < 0.0:
                    raise ValueError(f"{field_name} must be non-negative when provided.")
        for note in self.notes:
            if not note.strip():
                raise ValueError("notes must contain non-empty strings.")

    @property
    def ending_state(self) -> SimulationState:
        """Return the end-of-period state for this record."""

        return self.next_state if self.next_state is not None else self.state


@dataclass(frozen=True, slots=True)
class EpisodeTrace:
    """Structured trace container for one benchmark run or episode."""

    run_id: str
    benchmark_id: str
    period_records: tuple[PeriodTraceRecord, ...] = field(default_factory=tuple)
    states: tuple[SimulationState, ...] = field(default_factory=tuple)
    observations: tuple[Observation, ...] = field(default_factory=tuple)
    agent_signals: tuple[AgentSignal, ...] = field(default_factory=tuple)
    optimization_results: tuple[OptimizationResult, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id must be a non-empty string.")
        if not self.benchmark_id.strip():
            raise ValueError("benchmark_id must be a non-empty string.")
        object.__setattr__(self, "period_records", tuple(self.period_records))
        object.__setattr__(self, "states", tuple(self.states))
        object.__setattr__(self, "observations", tuple(self.observations))
        object.__setattr__(self, "agent_signals", tuple(self.agent_signals))
        object.__setattr__(self, "optimization_results", tuple(self.optimization_results))
        for record in self.period_records:
            if not isinstance(record, PeriodTraceRecord):
                raise TypeError("period_records must contain PeriodTraceRecord values.")
        for state in self.states:
            if not isinstance(state, SimulationState):
                raise TypeError("states must contain SimulationState values.")
        for observation in self.observations:
            if not isinstance(observation, Observation):
                raise TypeError("observations must contain Observation values.")
        for signal in self.agent_signals:
            if not isinstance(signal, AgentSignal):
                raise TypeError("agent_signals must contain AgentSignal values.")
        for result in self.optimization_results:
            if not isinstance(result, OptimizationResult):
                raise TypeError("optimization_results must contain OptimizationResult values.")
        if self.period_records:
            if not self.states:
                object.__setattr__(
                    self,
                    "states",
                    tuple(record.state for record in self.period_records),
                )
            if not self.observations:
                object.__setattr__(
                    self,
                    "observations",
                    tuple(record.observation for record in self.period_records),
                )
            if not self.agent_signals:
                object.__setattr__(
                    self,
                    "agent_signals",
                    tuple(record.agent_signal for record in self.period_records),
                )
            if not self.optimization_results:
                object.__setattr__(
                    self,
                    "optimization_results",
                    tuple(record.optimization_result for record in self.period_records),
                )
        elif self.states or self.observations or self.agent_signals or self.optimization_results:
            lengths = {
                len(self.states),
                len(self.observations),
                len(self.agent_signals),
                len(self.optimization_results),
            }
            if len(lengths) != 1:
                raise ValueError(
                    "states, observations, agent_signals, and optimization_results must "
                    "have matching lengths when period_records are not provided."
                )
            period_records = tuple(
                PeriodTraceRecord(
                    time_index=state.time_index,
                    regime_label=state.regime_label,
                    state=state,
                    observation=observation,
                    agent_signal=signal,
                    optimization_result=result,
                )
                for state, observation, signal, result in zip(
                    self.states,
                    self.observations,
                    self.agent_signals,
                    self.optimization_results,
                    strict=True,
                )
            )
            object.__setattr__(self, "period_records", period_records)

    @property
    def trace_length(self) -> int:
        """Return the number of typed per-period records in the trace."""

        return len(self.period_records)

    @property
    def step_telemetry(self) -> tuple[OrchestrationStepTelemetry, ...]:
        """Return all available per-period step telemetry records."""

        return tuple(
            record.step_telemetry
            for record in self.period_records
            if record.step_telemetry is not None
        )

    @property
    def ending_states(self) -> tuple[SimulationState, ...]:
        """Return end-of-period states for all typed per-period records."""

        return tuple(record.ending_state for record in self.period_records)

    def append_period(self, record: PeriodTraceRecord) -> EpisodeTrace:
        """Return a new trace with an appended period record."""

        if not isinstance(record, PeriodTraceRecord):
            raise TypeError("record must be a PeriodTraceRecord.")
        if record.state.benchmark_id != self.benchmark_id:
            raise ValueError("record benchmark_id must match the trace benchmark_id.")
        return EpisodeTrace(
            run_id=self.run_id,
            benchmark_id=self.benchmark_id,
            period_records=self.period_records + (record,),
        )


RunTrace = EpisodeTrace


__all__ = [
    "EpisodeTrace",
    "Observation",
    "PeriodTraceRecord",
    "RunTrace",
    "SimulationState",
]
