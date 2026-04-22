"""Deterministic lead-time adapters for the first smoke path."""

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
from meio.leadtime.contracts import LeadTimeRequest, LeadTimeResult


def build_leadtime_request(
    invocation: ToolInvocation,
    horizon: int,
) -> LeadTimeRequest:
    """Build a typed lead-time request from the bounded runtime envelope."""

    if invocation.evidence is None:
        raise ValueError("Lead-time adapter requires typed runtime evidence.")
    time_index = invocation.system_state.time_index if invocation.system_state else invocation.evidence.time_index
    return LeadTimeRequest(
        observed_lead_times=invocation.evidence.leadtime.history,
        horizon=horizon,
        time_index=time_index,
        upstream_stage_index=invocation.evidence.leadtime.upstream_stage_index,
        downstream_stage_index=invocation.evidence.leadtime.downstream_stage_index,
    )


@dataclass(frozen=True, slots=True)
class DeterministicLeadTimeTool(BoundedTool):
    """Deterministic placeholder lead-time tool for smoke runs."""

    tool_id: str = "leadtime_tool"
    horizon: int = 2
    next_tool_id: str | None = "scenario_tool"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            tool_id=self.tool_id,
            tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
            supported_subgoals=(OperationalSubgoal.QUERY_UNCERTAINTY,),
            description="Deterministic placeholder lead-time adapter for smoke runs.",
        )

    def invoke(self, invocation: ToolInvocation) -> ToolResult:
        request = build_leadtime_request(invocation, self.horizon)
        baseline_value = request.observed_lead_times[-1]
        leadtime_result = LeadTimeResult(
            horizon=request.horizon,
            expected_lead_time=tuple(baseline_value for _ in range(request.horizon)),
            uncertainty_scale=tuple(0.25 for _ in range(request.horizon)),
            provenance="deterministic_leadtime_smoke_adapter",
            metadata=("placeholder_adapter",),
        )
        return ToolResult(
            tool_id=invocation.tool_id,
            tool_class=invocation.tool_class,
            subgoal=invocation.subgoal,
            status=ToolStatus.SUCCESS,
            structured_output={"leadtime_result": leadtime_result},
            provenance=leadtime_result.provenance,
            next_tool_id=self.next_tool_id,
            next_subgoal=OperationalSubgoal.UPDATE_UNCERTAINTY,
        )


__all__ = ["DeterministicLeadTimeTool", "build_leadtime_request"]
