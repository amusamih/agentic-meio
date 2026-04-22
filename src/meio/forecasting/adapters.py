"""Deterministic forecasting adapters for the first smoke path."""

from __future__ import annotations

from dataclasses import dataclass

from meio.contracts import (
    BoundedTool,
    OperationalSubgoal,
    ToolClass,
    ToolInvocation,
    ToolResult,
    ToolSpec,
    ToolStatus,
)
from meio.forecasting.contracts import ForecastRequest, ForecastResult


def build_forecast_request(
    invocation: ToolInvocation,
    horizon: int,
) -> ForecastRequest:
    """Build a typed forecast request from the bounded runtime envelope."""

    if invocation.evidence is None:
        raise ValueError("Forecast adapter requires typed runtime evidence.")
    time_index = invocation.system_state.time_index if invocation.system_state else invocation.evidence.time_index
    return ForecastRequest(
        demand_history=invocation.evidence.demand.history,
        horizon=horizon,
        time_index=time_index,
        stage_index=invocation.evidence.demand.stage_index,
    )


@dataclass(frozen=True, slots=True)
class DeterministicForecastTool(BoundedTool):
    """Deterministic placeholder forecasting tool for smoke runs."""

    tool_id: str = "forecast_tool"
    horizon: int = 2
    next_tool_id: str | None = "leadtime_tool"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            tool_id=self.tool_id,
            tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
            supported_subgoals=(
                OperationalSubgoal.INSPECT_EVIDENCE,
                OperationalSubgoal.QUERY_UNCERTAINTY,
            ),
            description="Deterministic placeholder forecast adapter for smoke runs.",
        )

    def invoke(self, invocation: ToolInvocation) -> ToolResult:
        request = build_forecast_request(invocation, self.horizon)
        baseline_value = request.demand_history[-1]
        forecast_result = ForecastResult(
            horizon=request.horizon,
            point_forecast=tuple(baseline_value for _ in range(request.horizon)),
            uncertainty_scale=tuple(max(1.0, baseline_value * 0.1) for _ in range(request.horizon)),
            provenance="deterministic_forecast_smoke_adapter",
            metadata=("placeholder_adapter",),
        )
        return ToolResult(
            tool_id=invocation.tool_id,
            tool_class=invocation.tool_class,
            subgoal=invocation.subgoal,
            status=ToolStatus.SUCCESS,
            structured_output={"forecast_result": forecast_result},
            provenance=forecast_result.provenance,
            next_tool_id=self.next_tool_id,
            next_subgoal=OperationalSubgoal.QUERY_UNCERTAINTY,
        )


__all__ = ["DeterministicForecastTool", "build_forecast_request"]
