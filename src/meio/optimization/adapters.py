"""Trusted optimization adapters for the first smoke path."""

from __future__ import annotations

from dataclasses import dataclass

from meio.optimization.contracts import (
    OptimizationRequest,
    OptimizationResult,
    OptimizationStatus,
)
from meio.scenarios.contracts import ScenarioUpdateResult
from meio.simulation.state import SimulationState


def build_optimization_request(
    system_state: SimulationState,
    scenario_update_result: ScenarioUpdateResult,
    base_stock_levels: tuple[float, ...],
    planning_horizon: int = 1,
) -> OptimizationRequest:
    """Build a typed optimization request from the smoke-path state and scenarios."""

    return OptimizationRequest(
        inventory_level=system_state.inventory_level,
        base_stock_levels=base_stock_levels,
        scenario_adjustment=scenario_update_result.adjustment,
        pipeline_inventory=system_state.pipeline_inventory,
        backorder_level=system_state.backorder_level,
        scenario_summaries=scenario_update_result.scenarios,
        planning_horizon=planning_horizon,
        time_index=system_state.time_index,
    )


@dataclass(frozen=True, slots=True)
class TrustedOptimizerAdapter:
    """Trusted provisional optimizer adapter.

    This is intentionally the only smoke-path component allowed to return raw
    replenishment-order decisions. The current logic is a transparent,
    deterministic order-up-to heuristic, not the final trusted optimizer.
    """

    provenance: str = "trusted_optimizer_provisional_base_stock_adapter"
    max_order_multiplier: float = 1.5

    def solve(self, request: OptimizationRequest) -> OptimizationResult:
        replenishment_orders = self._compute_replenishment_orders(request)
        return OptimizationResult(
            replenishment_orders=replenishment_orders,
            planning_horizon=request.planning_horizon,
            status=OptimizationStatus.SUCCESS,
            objective_value=sum(replenishment_orders),
            provenance=self.provenance,
            metadata=(
                "provisional_base_stock_style_heuristic",
                "not_final_trusted_optimizer",
            ),
        )

    def _compute_replenishment_orders(
        self,
        request: OptimizationRequest,
    ) -> tuple[float, ...]:
        effective_demand = (
            request.scenario_adjustment.demand_outlook
            * request.scenario_adjustment.safety_buffer_scale
        )
        effective_leadtime = max(1.0, request.scenario_adjustment.leadtime_outlook)
        pipeline_inventory = request.pipeline_inventory or tuple(
            0.0 for _ in request.inventory_level
        )
        backorder_level = request.backorder_level or tuple(
            0.0 for _ in request.inventory_level
        )
        replenishment_orders: list[float] = []
        for index, (
            base_stock_level,
            inventory_level,
            pipeline_value,
            backorder_value,
        ) in enumerate(
            zip(
                request.base_stock_levels,
                request.inventory_level,
                pipeline_inventory,
                backorder_level,
                strict=True,
            )
        ):
            stage_cover_multiplier = 1.0 + 0.5 * index
            target_inventory_position = max(
                base_stock_level,
                effective_demand * effective_leadtime * stage_cover_multiplier,
            )
            if index == 0:
                target_inventory_position += backorder_value
            current_inventory_position = inventory_level + pipeline_value - backorder_value
            uncapped_order = max(0.0, target_inventory_position - current_inventory_position)
            order_cap = max(base_stock_level, 1.0) * self.max_order_multiplier
            replenishment_orders.append(min(uncapped_order, order_cap))
        return tuple(replenishment_orders)


__all__ = ["TrustedOptimizerAdapter", "build_optimization_request"]
