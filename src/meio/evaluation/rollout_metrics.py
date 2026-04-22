"""Typed rollout metrics for the controlled Stockpyl path."""

from __future__ import annotations

from dataclasses import dataclass, field

from meio.config.schemas import CostConfig
from meio.simulation.state import EpisodeTrace


@dataclass(frozen=True, slots=True)
class RolloutMetrics:
    """Compact operational metrics for a controlled multi-period rollout."""

    average_inventory: float | None
    average_backorder_level: float | None
    total_replan_count: int
    total_tool_call_count: int
    abstain_count: int
    holding_cost: float = 0.0
    backlog_cost: float = 0.0
    ordering_cost: float = 0.0
    other_cost: float = 0.0
    total_cost: float | None = None
    fill_rate: float | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "notes", tuple(self.notes))
        for field_name in (
            "total_replan_count",
            "total_tool_call_count",
            "abstain_count",
        ):
            value = getattr(self, field_name)
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative.")
        for field_name in (
            "average_inventory",
            "average_backorder_level",
            "holding_cost",
            "backlog_cost",
            "ordering_cost",
            "other_cost",
            "total_cost",
            "fill_rate",
        ):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be numeric when provided.")
        for note in self.notes:
            if not note.strip():
                raise ValueError("notes must contain non-empty strings.")


def compute_period_fill_rate(record) -> float | None:
    """Return the per-period downstream fill rate."""

    if record.demand_load is None or record.demand_load <= 0.0:
        return None
    served_demand = record.served_demand or 0.0
    return served_demand / record.demand_load


def compute_period_cost_breakdown(
    record,
    cost_config: CostConfig,
    *,
    holding_cost_by_stage: tuple[float, ...] | None = None,
    stockout_cost_by_stage: tuple[float, ...] | None = None,
) -> tuple[float, float, float, float]:
    """Return explicit per-period cost components.

    Holding and backlog costs are charged on the end-of-period state when the
    next state is available. Backlog cost defaults to the configured global
    value, but stage-specific stockout costs may be supplied by the benchmark
    adapter for the active Stockpyl path.
    """

    ending_state = record.ending_state
    inventory_level = ending_state.inventory_level
    backorder_level = ending_state.backorder_level
    if holding_cost_by_stage is None:
        holding_cost_by_stage = tuple(
            cost_config.holding_cost for _ in inventory_level
        )
    if stockout_cost_by_stage is None:
        stockout_cost_by_stage = tuple(
            cost_config.backorder_cost for _ in backorder_level
        )
    holding_cost = sum(
        unit_cost * inventory
        for unit_cost, inventory in zip(
            holding_cost_by_stage,
            inventory_level,
            strict=True,
        )
    )
    backlog_cost = sum(
        unit_cost * backlog
        for unit_cost, backlog in zip(
            stockout_cost_by_stage,
            backorder_level,
            strict=True,
        )
    )
    ordering_cost = cost_config.ordering_cost * sum(record.optimization_result.replenishment_orders)
    other_cost = 0.0
    return holding_cost, backlog_cost, ordering_cost, other_cost


def compute_period_total_cost(
    record,
    cost_config: CostConfig,
    *,
    holding_cost_by_stage: tuple[float, ...] | None = None,
    stockout_cost_by_stage: tuple[float, ...] | None = None,
) -> float:
    """Return the total cost for one controlled rollout period."""

    holding_cost, backlog_cost, ordering_cost, other_cost = compute_period_cost_breakdown(
        record,
        cost_config,
        holding_cost_by_stage=holding_cost_by_stage,
        stockout_cost_by_stage=stockout_cost_by_stage,
    )
    return holding_cost + backlog_cost + ordering_cost + other_cost


def compute_rollout_metrics(
    trace: EpisodeTrace,
    cost_config: CostConfig,
    *,
    holding_cost_by_stage: tuple[float, ...] | None = None,
    stockout_cost_by_stage: tuple[float, ...] | None = None,
) -> RolloutMetrics:
    """Compute compact metrics from the current rollout trace.

    The active Stockpyl path uses explicit end-of-period inventory, internal
    backlog, and in-transit queues, so these metrics are derived from the
    ending state of each recorded period rather than the pre-decision state.
    """

    ending_states = trace.ending_states
    inventory_points = tuple(
        value for state in ending_states for value in state.inventory_level
    )
    backorder_points = tuple(
        value for state in ending_states for value in state.backorder_level
    )
    average_inventory = (
        sum(inventory_points) / len(inventory_points) if inventory_points else None
    )
    average_backorder_level = (
        sum(backorder_points) / len(backorder_points) if backorder_points else None
    )
    total_demand_load = sum(record.demand_load or 0.0 for record in trace.period_records)
    total_served_demand = sum(record.served_demand or 0.0 for record in trace.period_records)
    fill_rate = None
    if total_demand_load > 0.0:
        fill_rate = total_served_demand / total_demand_load
    holding_cost = 0.0
    backlog_cost = 0.0
    ordering_cost = 0.0
    other_cost = 0.0
    for record in trace.period_records:
        period_holding_cost, period_backlog_cost, period_ordering_cost, period_other_cost = (
            compute_period_cost_breakdown(
                record,
                cost_config,
                holding_cost_by_stage=holding_cost_by_stage,
                stockout_cost_by_stage=stockout_cost_by_stage,
            )
        )
        holding_cost += period_holding_cost
        backlog_cost += period_backlog_cost
        ordering_cost += period_ordering_cost
        other_cost += period_other_cost
    return RolloutMetrics(
        average_inventory=average_inventory,
        average_backorder_level=average_backorder_level,
        total_replan_count=sum(1 for signal in trace.agent_signals if signal.request_replan),
        total_tool_call_count=sum(len(signal.tool_sequence) for signal in trace.agent_signals),
        abstain_count=sum(1 for signal in trace.agent_signals if signal.abstained),
        holding_cost=holding_cost,
        backlog_cost=backlog_cost,
        ordering_cost=ordering_cost,
        other_cost=other_cost,
        total_cost=holding_cost + backlog_cost + ordering_cost + other_cost,
        fill_rate=fill_rate,
        notes=(
            "explicit_end_of_period_inventory_costs",
            "downstream_fill_rate_from_realized_service",
        ),
    )


__all__ = [
    "RolloutMetrics",
    "compute_period_cost_breakdown",
    "compute_period_fill_rate",
    "compute_period_total_cost",
    "compute_rollout_metrics",
]
