"""Deterministic scenario-update adapters for the active serial path."""

from __future__ import annotations

from dataclasses import dataclass

from meio.contracts import (
    AgentAssessment,
    BoundedTool,
    OperationalSubgoal,
    RegimeLabel,
    ToolClass,
    ToolInvocation,
    ToolResult,
    ToolSpec,
    ToolStatus,
    UpdateRequest,
    UpdateRequestType,
)
from meio.forecasting.contracts import ForecastResult
from meio.leadtime.contracts import LeadTimeResult
from meio.scenarios.contracts import (
    ScenarioAdjustmentSummary,
    ScenarioSummary,
    ScenarioUpdateRequest,
    ScenarioUpdateResult,
)

_DEFAULT_UPDATE_TYPES = {
    RegimeLabel.NORMAL: (UpdateRequestType.KEEP_CURRENT,),
    RegimeLabel.DEMAND_REGIME_SHIFT: (
        UpdateRequestType.REWEIGHT_SCENARIOS,
        UpdateRequestType.WIDEN_UNCERTAINTY,
    ),
    RegimeLabel.SUPPLY_DISRUPTION: (
        UpdateRequestType.SWITCH_LEADTIME_REGIME,
        UpdateRequestType.WIDEN_UNCERTAINTY,
    ),
    RegimeLabel.JOINT_DISRUPTION: (
        UpdateRequestType.REWEIGHT_SCENARIOS,
        UpdateRequestType.WIDEN_UNCERTAINTY,
    ),
    RegimeLabel.RECOVERY: (UpdateRequestType.REWEIGHT_SCENARIOS,),
}
_REGIME_BASE_SCALES = {
    RegimeLabel.NORMAL: (1.0, 1.0, 1.0),
    RegimeLabel.DEMAND_REGIME_SHIFT: (1.1, 1.0, 1.05),
    RegimeLabel.SUPPLY_DISRUPTION: (1.0, 1.15, 1.05),
    RegimeLabel.JOINT_DISRUPTION: (1.1, 1.15, 1.08),
    RegimeLabel.RECOVERY: (1.02, 1.0, 1.02),
}
_UPDATE_TYPE_EFFECTS = {
    UpdateRequestType.KEEP_CURRENT: (1.0, 1.0, 1.0),
    UpdateRequestType.REWEIGHT_SCENARIOS: (1.03, 1.0, 1.02),
    UpdateRequestType.SWITCH_DEMAND_REGIME: (1.12, 1.0, 1.05),
    UpdateRequestType.SWITCH_LEADTIME_REGIME: (1.0, 1.15, 1.05),
    UpdateRequestType.WIDEN_UNCERTAINTY: (1.0, 1.0, 1.10),
}


def _extract_prior_output(
    invocation: ToolInvocation,
    key: str,
    expected_type: type[object],
) -> object:
    for result in reversed(invocation.prior_results):
        value = result.structured_output.get(key)
        if isinstance(value, expected_type):
            return value
    raise ValueError(f"Scenario adapter requires prior output {key!r}.")


def _fallback_forecast_from_observation(invocation: ToolInvocation) -> ForecastResult:
    if invocation.observation is None:
        raise ValueError("Scenario adapter requires an observation when forecast output is absent.")
    demand_value = invocation.observation.demand_realization[-1]
    return ForecastResult(
        horizon=1,
        point_forecast=(demand_value,),
        uncertainty_scale=(max(1.0, demand_value * 0.1),),
        provenance="scenario_tool_observation_fallback",
        metadata=("forecast_tool_ablated_or_unavailable",),
    )


def _fallback_leadtime_from_observation(invocation: ToolInvocation) -> LeadTimeResult:
    if invocation.observation is None:
        raise ValueError("Scenario adapter requires an observation when lead-time output is absent.")
    leadtime_value = invocation.observation.leadtime_realization[-1]
    return LeadTimeResult(
        horizon=1,
        expected_lead_time=(leadtime_value,),
        uncertainty_scale=(0.25,),
        provenance="scenario_tool_observation_fallback",
        metadata=("leadtime_tool_ablated_or_unavailable",),
    )


def build_scenario_update_request(invocation: ToolInvocation) -> ScenarioUpdateRequest:
    """Build a typed scenario-update request from prior bounded tool outputs."""

    try:
        forecast_result = _extract_prior_output(
            invocation,
            "forecast_result",
            ForecastResult,
        )
    except ValueError:
        forecast_result = _fallback_forecast_from_observation(invocation)
    try:
        leadtime_result = _extract_prior_output(
            invocation,
            "leadtime_result",
            LeadTimeResult,
        )
    except ValueError:
        leadtime_result = _fallback_leadtime_from_observation(invocation)

    current_regime = RegimeLabel.NORMAL
    if invocation.system_state is not None:
        current_regime = invocation.system_state.regime_label
    if (
        invocation.observation is not None
        and invocation.observation.regime_label is not None
    ):
        current_regime = invocation.observation.regime_label
    if invocation.agent_assessment is not None:
        current_regime = invocation.agent_assessment.regime_label
        request_types = tuple(
            update_request.request_type
            for update_request in invocation.agent_assessment.update_requests
        ) or (UpdateRequestType.KEEP_CURRENT,)
    else:
        request_types = _DEFAULT_UPDATE_TYPES[current_regime]

    return ScenarioUpdateRequest(
        update_requests=tuple(
            UpdateRequest(
                request_type=request_type,
                target="serial_smoke_scenarios",
                notes="Deterministic placeholder scenario update.",
            )
            for request_type in request_types
        ),
        current_regime=current_regime,
        current_scenarios=(
            ScenarioSummary(
                scenario_id=f"{current_regime.value}_baseline",
                regime_label=current_regime,
            ),
        ),
        forecast_result=forecast_result,  # type: ignore[arg-type]
        leadtime_result=leadtime_result,  # type: ignore[arg-type]
    )


def _resolve_adjustment(
    request: ScenarioUpdateRequest,
    assessment: AgentAssessment | None,
) -> tuple[float, float, float, bool, tuple[UpdateRequestType, ...]]:
    demand_scale, leadtime_scale, buffer_scale = _REGIME_BASE_SCALES[
        request.current_regime
    ]
    applied_update_types = tuple(
        update_request.request_type for update_request in request.update_requests
    )
    for update_type in applied_update_types:
        demand_factor, leadtime_factor, buffer_factor = _UPDATE_TYPE_EFFECTS[update_type]
        demand_scale *= demand_factor
        leadtime_scale *= leadtime_factor
        buffer_scale *= buffer_factor
    if assessment is not None:
        request_replan = assessment.request_replan
    else:
        request_replan = request.current_regime is not RegimeLabel.NORMAL
    return (
        demand_scale,
        leadtime_scale,
        buffer_scale,
        request_replan,
        applied_update_types,
    )


@dataclass(frozen=True, slots=True)
class DeterministicScenarioTool(BoundedTool):
    """Deterministic placeholder scenario tool for smoke runs."""

    tool_id: str = "scenario_tool"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            tool_id=self.tool_id,
            tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
            supported_subgoals=(
                OperationalSubgoal.QUERY_UNCERTAINTY,
                OperationalSubgoal.UPDATE_UNCERTAINTY,
            ),
            description=(
                "Deterministic placeholder scenario adapter for bounded "
                "uncertainty inspection and update routing."
            ),
        )

    def invoke(self, invocation: ToolInvocation) -> ToolResult:
        request = build_scenario_update_request(invocation)
        (
            demand_scale,
            leadtime_scale,
            buffer_scale,
            request_replan,
            applied_update_types,
        ) = _resolve_adjustment(request, invocation.agent_assessment)
        demand_outlook = request.forecast_result.point_forecast[0] * demand_scale
        leadtime_outlook = request.leadtime_result.expected_lead_time[0] * leadtime_scale
        scenario_update_result = ScenarioUpdateResult(
            scenarios=(
                ScenarioSummary(
                    scenario_id="serial_smoke",
                    regime_label=request.current_regime,
                    weight=1.0,
                    demand_scale=demand_scale,
                    leadtime_scale=leadtime_scale,
                ),
            ),
            applied_update_types=applied_update_types,
            adjustment=ScenarioAdjustmentSummary(
                demand_outlook=demand_outlook,
                leadtime_outlook=leadtime_outlook,
                safety_buffer_scale=buffer_scale,
            ),
            request_replan=request_replan,
            provenance="deterministic_scenario_smoke_adapter",
        )
        return ToolResult(
            tool_id=invocation.tool_id,
            tool_class=invocation.tool_class,
            subgoal=invocation.subgoal,
            status=ToolStatus.SUCCESS,
            structured_output={"scenario_update_result": scenario_update_result},
            provenance=scenario_update_result.provenance,
            next_subgoal=(
                OperationalSubgoal.REQUEST_REPLAN
                if scenario_update_result.request_replan
                else OperationalSubgoal.NO_ACTION
            ),
            request_replan=scenario_update_result.request_replan,
        )


__all__ = ["DeterministicScenarioTool", "build_scenario_update_request"]
