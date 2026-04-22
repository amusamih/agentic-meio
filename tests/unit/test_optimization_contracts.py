from __future__ import annotations

import pytest

from meio.contracts import BackorderPolicy, RegimeLabel
from meio.optimization.contracts import (
    OptimizationRequest,
    OptimizationResult,
    OptimizationStatus,
)
from meio.scenarios.contracts import ScenarioAdjustmentSummary, ScenarioSummary


def test_optimization_result_carries_replenishment_orders() -> None:
    result = OptimizationResult(
        replenishment_orders=(5.0, 4.0, 3.0),
        planning_horizon=3,
        status=OptimizationStatus.SUCCESS,
        objective_value=12.5,
        provenance="trusted_optimizer",
    )

    assert result.replenishment_orders == (5.0, 4.0, 3.0)
    assert result.status is OptimizationStatus.SUCCESS


def test_optimization_request_rejects_mismatched_pipeline_length() -> None:
    with pytest.raises(ValueError, match="pipeline_inventory"):
        OptimizationRequest(
            inventory_level=(10.0, 8.0, 6.0),
            base_stock_levels=(12.0, 10.0, 8.0),
            scenario_adjustment=ScenarioAdjustmentSummary(
                demand_outlook=10.0,
                leadtime_outlook=2.0,
            ),
            pipeline_inventory=(1.0, 2.0),
            backorder_level=(0.0, 0.0, 1.0),
            scenario_summaries=(
                ScenarioSummary(
                    scenario_id="baseline",
                    regime_label=RegimeLabel.NORMAL,
                ),
            ),
            planning_horizon=2,
            service_model=BackorderPolicy.BACKORDERS,
        )
