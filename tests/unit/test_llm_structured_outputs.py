from __future__ import annotations

import pytest

from meio.agents.structured_outputs import (
    LLMOutputValidationError,
    parse_llm_output,
    parse_llm_output_detailed,
)
from meio.contracts import OperationalSubgoal, UpdateRequestType


def test_parse_llm_output_accepts_valid_payload() -> None:
    decision = parse_llm_output(
        '{"selected_subgoal":"query_uncertainty","candidate_tool_ids":["forecast_tool","leadtime_tool"],'
        '"regime_label":"demand_regime_shift","confidence":0.8,'
        '"update_request_types":["switch_demand_regime"],"request_replan":true,'
        '"rationale":"Demand evidence warrants a bounded uncertainty query."}',
        allowed_tool_ids=("forecast_tool", "leadtime_tool", "scenario_tool"),
        allowed_update_types=(UpdateRequestType.SWITCH_DEMAND_REGIME,),
    )

    assert decision.selected_subgoal is OperationalSubgoal.QUERY_UNCERTAINTY
    assert decision.update_requests[0].request_type is UpdateRequestType.SWITCH_DEMAND_REGIME


def test_parse_llm_output_rejects_disallowed_fields() -> None:
    with pytest.raises(LLMOutputValidationError, match="disallowed fields") as exc_info:
        parse_llm_output(
            '{"selected_subgoal":"no_action","candidate_tool_ids":[],"regime_label":"normal",'
            '"confidence":0.9,"update_request_types":["keep_current"],"request_replan":false,'
            '"rationale":"No action.","raw_orders":[1,2,3]}',
            allowed_tool_ids=("forecast_tool",),
            allowed_update_types=(UpdateRequestType.KEEP_CURRENT,),
        )
    assert exc_info.value.reason_code == "disallowed_fields"


def test_parse_llm_output_rejects_invalid_confidence() -> None:
    with pytest.raises(LLMOutputValidationError, match="confidence") as exc_info:
        parse_llm_output(
            '{"selected_subgoal":"no_action","candidate_tool_ids":[],"regime_label":"normal",'
            '"confidence":1.5,"update_request_types":["keep_current"],"request_replan":false,'
            '"rationale":"No action."}',
            allowed_tool_ids=("forecast_tool",),
            allowed_update_types=(UpdateRequestType.KEEP_CURRENT,),
        )
    assert exc_info.value.reason_code == "invalid_confidence"


def test_parse_llm_output_accepts_observed_live_fence_and_confidence_alias() -> None:
    parsed_output = parse_llm_output_detailed(
        """```json
        {
          "selected_subgoal": "no_action",
          "candidate_tool_ids": [],
          "regime_label": "normal",
          "confidence": "high",
          "update_request_types": [],
          "request_replan": false,
          "rationale": "No intervention is justified."
        }
        ```""",
        allowed_tool_ids=("forecast_tool", "leadtime_tool", "scenario_tool"),
        allowed_update_types=(
            UpdateRequestType.KEEP_CURRENT,
            UpdateRequestType.WIDEN_UNCERTAINTY,
        ),
    )

    assert parsed_output.decision.selected_subgoal is OperationalSubgoal.NO_ACTION
    assert parsed_output.decision.confidence == 0.85
    assert "trimmed_code_fence" in parsed_output.normalization_notes
    assert "normalized_confidence_alias:high" in parsed_output.normalization_notes


def test_parse_llm_output_classifies_non_json_output() -> None:
    with pytest.raises(LLMOutputValidationError, match="valid JSON") as exc_info:
        parse_llm_output(
            "I think no action is best.",
            allowed_tool_ids=("forecast_tool",),
            allowed_update_types=(UpdateRequestType.KEEP_CURRENT,),
        )

    assert exc_info.value.reason_code == "malformed_json"


def test_parse_llm_output_classifies_requested_missing_tool() -> None:
    with pytest.raises(LLMOutputValidationError) as exc_info:
        parse_llm_output_detailed(
            '{"selected_subgoal":"query_uncertainty","candidate_tool_ids":["forecast_tool","scenario_tool"],'
            '"regime_label":"demand_regime_shift","confidence":0.8,'
            '"update_request_types":["switch_demand_regime"],"request_replan":true,'
            '"rationale":"Demand evidence warrants a bounded uncertainty query."}',
            allowed_tool_ids=("leadtime_tool", "scenario_tool"),
            allowed_update_types=(UpdateRequestType.SWITCH_DEMAND_REGIME,),
        )

    assert exc_info.value.reason_code == "requested_missing_tool"
    assert exc_info.value.details["requested_tool_ids"] == (
        "forecast_tool",
        "scenario_tool",
    )
    assert exc_info.value.details["unavailable_tool_ids"] == ("forecast_tool",)
