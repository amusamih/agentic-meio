from __future__ import annotations

import pytest

from meio.contracts import (
    AgentAssessment,
    MissionSpec,
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


def test_agent_assessment_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        AgentAssessment(
            regime_label=RegimeLabel.NORMAL,
            confidence=1.2,
            rationale="Confidence must stay bounded.",
        )


def test_agent_assessment_rejects_non_update_request_values() -> None:
    with pytest.raises(TypeError, match="update_requests"):
        AgentAssessment(
            regime_label=RegimeLabel.DEMAND_REGIME_SHIFT,
            confidence=0.8,
            rationale="Demand regime shifted materially.",
            update_requests=("reweight_scenarios",),
        )


def test_agent_assessment_construction_preserves_typed_requests() -> None:
    request = UpdateRequest(
        request_type=UpdateRequestType.REWEIGHT_SCENARIOS,
        target="demand_model",
        notes="Increase tail weight for stressed demand scenarios.",
    )
    assessment = AgentAssessment(
        regime_label=RegimeLabel.DEMAND_REGIME_SHIFT,
        confidence=0.75,
        rationale="Observed demand exceeded the reference envelope.",
        update_requests=[request],
        request_replan=True,
    )

    assert assessment.regime_label is RegimeLabel.DEMAND_REGIME_SHIFT
    assert assessment.update_requests == (request,)
    assert assessment.request_replan is True


def test_tool_result_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        ToolResult(
            tool_id="evidence_tool",
            tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
            subgoal=OperationalSubgoal.INSPECT_EVIDENCE,
            confidence=1.5,
        )


def test_mission_spec_rejects_nonpositive_max_tool_steps() -> None:
    with pytest.raises(ValueError, match="max_tool_steps"):
        MissionSpec(
            mission_id="serial_mission",
            objective="Preserve bounded orchestration behavior.",
            max_tool_steps=0,
        )


def test_tool_invocation_preserves_typed_prior_results() -> None:
    prior_result = ToolResult(
        tool_id="evidence_tool",
        tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
        subgoal=OperationalSubgoal.INSPECT_EVIDENCE,
        status=ToolStatus.SUCCESS,
        structured_output={"evidence_count": 3},
    )
    invocation = ToolInvocation(
        tool_id="follow_on_tool",
        tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
        subgoal=OperationalSubgoal.CLASSIFY_REGIME,
        prior_results=[prior_result],
    )

    assert invocation.prior_results == (prior_result,)


def test_tool_spec_rejects_empty_subgoal_set() -> None:
    with pytest.raises(ValueError, match="supported_subgoals"):
        ToolSpec(
            tool_id="invalid_tool",
            tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
            supported_subgoals=(),
        )
