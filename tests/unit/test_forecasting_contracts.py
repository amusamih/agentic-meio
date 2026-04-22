from __future__ import annotations

import pytest

from meio.forecasting.contracts import (
    ForecastRequest,
    ForecastResult,
    ForecastUncertaintyFormat,
)


def test_forecast_request_accepts_basic_numeric_inputs() -> None:
    request = ForecastRequest(
        demand_history=(10.0, 12.0, 14.0),
        horizon=2,
        time_index=3,
        stage_index=1,
    )

    assert request.demand_history == (10.0, 12.0, 14.0)
    assert request.horizon == 2
    assert request.stage_index == 1


def test_forecast_result_rejects_uncertainty_length_mismatch() -> None:
    with pytest.raises(ValueError, match="uncertainty_scale"):
        ForecastResult(
            horizon=3,
            point_forecast=(11.0, 12.0, 13.0),
            uncertainty_scale=(1.0, 1.5),
            uncertainty_format=ForecastUncertaintyFormat.LOCATION_SCALE,
        )
