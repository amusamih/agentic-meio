from __future__ import annotations

import pytest

from meio.contracts import RegimeLabel, UpdateRequest, UpdateRequestType
from meio.forecasting.contracts import ForecastResult
from meio.leadtime.contracts import LeadTimeResult
from meio.scenarios.contracts import (
    ScenarioAdjustmentSummary,
    ScenarioSummary,
    ScenarioUpdateRequest,
    ScenarioUpdateResult,
)


def test_scenario_update_request_accepts_typed_inputs() -> None:
    request = ScenarioUpdateRequest(
        update_requests=(
            UpdateRequest(
                request_type=UpdateRequestType.REWEIGHT_SCENARIOS,
                target="serial_primary",
            ),
        ),
        current_regime=RegimeLabel.DEMAND_REGIME_SHIFT,
        current_scenarios=(
            ScenarioSummary(
                scenario_id="baseline",
                regime_label=RegimeLabel.NORMAL,
                weight=0.5,
            ),
        ),
        forecast_result=ForecastResult(
            horizon=2,
            point_forecast=(10.0, 11.0),
            uncertainty_scale=(1.0, 1.2),
            provenance="forecast_tool",
        ),
        leadtime_result=LeadTimeResult(
            horizon=2,
            expected_lead_time=(2.0, 2.5),
            uncertainty_scale=(0.2, 0.3),
            provenance="leadtime_tool",
        ),
    )

    assert request.current_regime is RegimeLabel.DEMAND_REGIME_SHIFT
    assert request.current_scenarios[0].scenario_id == "baseline"


def test_scenario_update_result_rejects_keep_current_mixed_with_other_types() -> None:
    with pytest.raises(ValueError, match="keep_current"):
        ScenarioUpdateResult(
            scenarios=(
                ScenarioSummary(
                    scenario_id="baseline",
                    regime_label=RegimeLabel.NORMAL,
                ),
            ),
            applied_update_types=(
                UpdateRequestType.KEEP_CURRENT,
                UpdateRequestType.WIDEN_UNCERTAINTY,
            ),
            adjustment=ScenarioAdjustmentSummary(
                demand_outlook=10.0,
                leadtime_outlook=2.0,
            ),
        )
