"""Typed optimization contracts for the first MEIO milestone."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from meio.contracts import BackorderPolicy
from meio.scenarios.contracts import ScenarioAdjustmentSummary, ScenarioSummary


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


class OptimizationStatus(StrEnum):
    """Minimal status values for the trusted optimizer boundary."""

    SUCCESS = "success"
    INFEASIBLE = "infeasible"
    NOT_RUN = "not_run"


@dataclass(frozen=True, slots=True)
class OptimizationRequest:
    """Typed optimizer input.

    This boundary is the only place in the first milestone architecture that is
    allowed to request replenishment decisions from a trusted solver or policy.
    """

    inventory_level: tuple[float, ...]
    planning_horizon: int
    base_stock_levels: tuple[float, ...]
    scenario_adjustment: ScenarioAdjustmentSummary
    pipeline_inventory: tuple[float, ...] = field(default_factory=tuple)
    backorder_level: tuple[float, ...] = field(default_factory=tuple)
    scenario_summaries: tuple[ScenarioSummary, ...] = field(default_factory=tuple)
    service_model: BackorderPolicy = BackorderPolicy.BACKORDERS
    time_index: int = 0

    def __post_init__(self) -> None:
        inventory_level = _coerce_series(
            self.inventory_level,
            "inventory_level",
            allow_empty=False,
        )
        object.__setattr__(self, "inventory_level", inventory_level)
        base_stock_levels = _coerce_series(
            self.base_stock_levels,
            "base_stock_levels",
            allow_empty=False,
        )
        if len(base_stock_levels) != len(inventory_level):
            raise ValueError("base_stock_levels must match inventory_level length.")
        object.__setattr__(self, "base_stock_levels", base_stock_levels)
        pipeline_inventory = _coerce_series(
            self.pipeline_inventory,
            "pipeline_inventory",
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
        object.__setattr__(self, "pipeline_inventory", pipeline_inventory)
        object.__setattr__(self, "backorder_level", backorder_level)
        object.__setattr__(self, "scenario_summaries", tuple(self.scenario_summaries))
        if self.planning_horizon <= 0:
            raise ValueError("planning_horizon must be positive.")
        if self.time_index < 0:
            raise ValueError("time_index must be non-negative.")
        if not isinstance(self.service_model, BackorderPolicy):
            raise TypeError("service_model must be a BackorderPolicy.")
        if not isinstance(self.scenario_adjustment, ScenarioAdjustmentSummary):
            raise TypeError("scenario_adjustment must be a ScenarioAdjustmentSummary.")
        for scenario in self.scenario_summaries:
            if not isinstance(scenario, ScenarioSummary):
                raise TypeError("scenario_summaries must contain ScenarioSummary values.")


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """Typed optimizer output carrying replenishment decisions.

    This is the only bounded interface in the first milestone allowed to carry
    raw replenishment-order decisions.
    """

    replenishment_orders: tuple[float, ...]
    planning_horizon: int
    status: OptimizationStatus = OptimizationStatus.SUCCESS
    objective_value: float | None = None
    provenance: str = ""
    metadata: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        replenishment_orders = _coerce_series(
            self.replenishment_orders,
            "replenishment_orders",
            allow_empty=False,
        )
        object.__setattr__(self, "replenishment_orders", replenishment_orders)
        object.__setattr__(self, "metadata", tuple(self.metadata))
        if self.planning_horizon <= 0:
            raise ValueError("planning_horizon must be positive.")
        if not isinstance(self.status, OptimizationStatus):
            raise TypeError("status must be an OptimizationStatus.")
        if self.objective_value is not None and not isinstance(self.objective_value, (int, float)):
            raise TypeError("objective_value must be numeric when provided.")
        for item in self.metadata:
            if not item.strip():
                raise ValueError("metadata must contain non-empty strings.")


__all__ = [
    "OptimizationRequest",
    "OptimizationResult",
    "OptimizationStatus",
]
