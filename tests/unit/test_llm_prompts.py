from __future__ import annotations

from meio.agents.llm_client import LLMClientContext
from meio.agents.prompts import (
    PROMPT_VERSION,
    build_prompt_messages,
    build_system_prompt,
    prompt_contract_hash,
)
from meio.contracts import OperationalSubgoal, RegimeLabel, ToolClass, ToolSpec


def test_build_system_prompt_keeps_optimizer_boundary_explicit() -> None:
    prompt = build_system_prompt()

    assert "optimizer is the sole order-producing boundary" in prompt.lower()
    assert "never emit replenishment orders" in prompt.lower()


def test_build_prompt_messages_include_typed_context_and_tools() -> None:
    context = LLMClientContext(
        benchmark_id="serial_3_echelon",
        mission_id="test_mission",
        time_index=1,
        regime_label=RegimeLabel.DEMAND_REGIME_SHIFT,
        demand_value=14.0,
        leadtime_value=2.0,
        inventory_level=(20.0, 30.0, 40.0),
        backorder_level=(1.0, 0.0, 0.0),
        available_tool_ids=("forecast_tool", "leadtime_tool", "scenario_tool"),
        max_tool_steps=3,
        demand_baseline_value=10.0,
        demand_change_from_baseline=4.0,
        demand_ratio_to_baseline=1.4,
        previous_demand_value=10.0,
        demand_change_from_previous=4.0,
        leadtime_baseline_value=2.0,
        leadtime_change_from_baseline=0.0,
        leadtime_ratio_to_baseline=1.0,
        previous_leadtime_value=2.0,
        leadtime_change_from_previous=0.0,
        downstream_inventory_value=20.0,
        total_inventory_value=90.0,
        total_backorder_value=1.0,
        inventory_gap_to_demand=6.0,
    )
    tool_specs = (
        ToolSpec(
            tool_id="forecast_tool",
            tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
            supported_subgoals=(OperationalSubgoal.QUERY_UNCERTAINTY,),
            description="Forecast demand.",
        ),
        ToolSpec(
            tool_id="scenario_tool",
            tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
            supported_subgoals=(OperationalSubgoal.UPDATE_UNCERTAINTY,),
            description="Update scenarios.",
        ),
    )

    messages = build_prompt_messages(context, tool_specs)

    assert len(messages) == 2
    assert "forecast_tool" in messages[1].content
    assert "demand_regime_shift" in messages[1].content
    assert "- demand_ratio_to_baseline: 1.4" in messages[1].content
    assert '- available_tool_ids: ["forecast_tool", "leadtime_tool", "scenario_tool"]' in messages[1].content
    assert "Decision guidance:" in messages[1].content
    assert "Examples:" in messages[1].content
    assert "- regime_label:" not in messages[1].content
    assert "- demand_material_departure_from_baseline: true" in messages[1].content
    assert "- pipeline_ratio_to_baseline:" not in messages[1].content
    assert '"selected_subgoal":"query_uncertainty"' in messages[1].content
    assert "Return exactly one JSON object with exactly these keys and no extras" in messages[1].content
    assert "Do not wrap the JSON in code fences" in messages[1].content
    assert "confidence` must be a JSON number in [0.0, 1.0]" in messages[1].content


def test_prompt_examples_respect_available_tool_set() -> None:
    context = LLMClientContext(
        benchmark_id="serial_3_echelon",
        mission_id="test_mission",
        time_index=1,
        regime_label=RegimeLabel.DEMAND_REGIME_SHIFT,
        demand_value=14.0,
        leadtime_value=2.0,
        inventory_level=(20.0, 30.0, 40.0),
        backorder_level=(1.0, 0.0, 0.0),
        available_tool_ids=("leadtime_tool", "scenario_tool"),
        max_tool_steps=3,
    )
    tool_specs = (
        ToolSpec(
            tool_id="forecast_tool",
            tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
            supported_subgoals=(OperationalSubgoal.QUERY_UNCERTAINTY,),
            description="Forecast demand.",
        ),
        ToolSpec(
            tool_id="leadtime_tool",
            tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
            supported_subgoals=(OperationalSubgoal.QUERY_UNCERTAINTY,),
            description="Assess lead time.",
        ),
        ToolSpec(
            tool_id="scenario_tool",
            tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
            supported_subgoals=(OperationalSubgoal.UPDATE_UNCERTAINTY,),
            description="Update scenarios.",
        ),
    )

    messages = build_prompt_messages(context, tool_specs)

    assert '"candidate_tool_ids":["leadtime_tool","scenario_tool"]' in messages[1].content
    assert '"forecast_tool"' not in messages[1].content.split("2. Initial demand regime shift:\n", 1)[1]
    assert "scenario_tool` is allowed in the same bounded query path" in messages[1].content
    assert "exact order you provide" in messages[1].content


def test_prompt_includes_calibration_cues_and_guidance() -> None:
    context = LLMClientContext(
        benchmark_id="serial_3_echelon",
        mission_id="test_mission",
        time_index=3,
        regime_label=RegimeLabel.RECOVERY,
        demand_value=11.0,
        leadtime_value=2.0,
        inventory_level=(0.0, 0.0, 0.0),
        backorder_level=(18.0, 19.8, 22.7),
        available_tool_ids=("forecast_tool", "leadtime_tool", "scenario_tool"),
        max_tool_steps=3,
        demand_baseline_value=10.0,
        demand_change_from_baseline=1.0,
        demand_ratio_to_baseline=1.1,
        previous_demand_value=14.0,
        demand_change_from_previous=-3.0,
        leadtime_baseline_value=2.0,
        leadtime_change_from_baseline=0.0,
        leadtime_ratio_to_baseline=1.0,
        previous_leadtime_value=2.0,
        leadtime_change_from_previous=0.0,
        downstream_inventory_value=0.0,
        total_inventory_value=0.0,
        pipeline_total_value=146.4,
        total_backorder_value=60.5,
        inventory_gap_to_demand=-11.0,
        pipeline_ratio_to_baseline=14.64,
        backorder_ratio_to_baseline=6.05,
        repeated_stress_detected=False,
        pipeline_heavy_vs_baseline=True,
        backlog_heavy_vs_baseline=True,
        recovery_with_high_pipeline_load=True,
        recovery_with_high_backorder_load=True,
    )
    tool_specs = (
        ToolSpec(
            tool_id="forecast_tool",
            tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
            supported_subgoals=(OperationalSubgoal.QUERY_UNCERTAINTY,),
            description="Forecast demand.",
        ),
        ToolSpec(
            tool_id="leadtime_tool",
            tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
            supported_subgoals=(OperationalSubgoal.QUERY_UNCERTAINTY,),
            description="Assess lead time.",
        ),
        ToolSpec(
            tool_id="scenario_tool",
            tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
            supported_subgoals=(OperationalSubgoal.UPDATE_UNCERTAINTY,),
            description="Update scenarios.",
        ),
    )

    messages = build_prompt_messages(context, tool_specs)

    assert "- total_pipeline: 146.4" in messages[1].content
    assert "- backorder_ratio_to_baseline: 6.05" in messages[1].content
    assert "- recovery_with_high_backorder_load: true" in messages[1].content
    assert "Repeated demand stress does not automatically require stronger escalation" in messages[1].content
    assert '"update_request_types":["reweight_scenarios"]' in messages[1].content
    assert "avoid another recovery replan right now" in messages[1].content


def test_prompt_contract_metadata_is_stable_and_versioned() -> None:
    first_hash = prompt_contract_hash()
    second_hash = prompt_contract_hash()

    assert PROMPT_VERSION == "meio.llm_orchestrator.v8"
    assert first_hash == second_hash
    assert len(first_hash) == 64
