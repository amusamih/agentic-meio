from __future__ import annotations

import pytest

from meio.leadtime.contracts import (
    LeadTimeRequest,
    LeadTimeResult,
    LeadTimeUncertaintyFormat,
)


def test_leadtime_request_accepts_basic_numeric_inputs() -> None:
    request = LeadTimeRequest(
        observed_lead_times=(2.0, 3.0, 2.5),
        horizon=2,
        time_index=4,
        upstream_stage_index=3,
        downstream_stage_index=2,
    )

    assert request.observed_lead_times == (2.0, 3.0, 2.5)
    assert request.horizon == 2
    assert request.upstream_stage_index == 3


def test_leadtime_result_rejects_negative_expected_values() -> None:
    with pytest.raises(ValueError, match="expected_lead_time"):
        LeadTimeResult(
            horizon=2,
            expected_lead_time=(2.0, -1.0),
            uncertainty_format=LeadTimeUncertaintyFormat.POINT_ONLY,
        )
