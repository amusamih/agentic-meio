from __future__ import annotations

from meio.agents.llm_client import LLMClientContext
from meio.agents.prompts import (
    PROMPT_VERSION,
    REGRET_GUARDED_TOOL_SEQUENCE,
    build_prompt_messages,
    build_system_prompt,
    prompt_contract_hash,
)
from meio.contracts import OperationalSubgoal, RegimeLabel, ToolClass, ToolSpec


def _tool_specs() -> tuple[ToolSpec, ...]:
    return tuple(
        ToolSpec(
            tool_id=tool_id,
            tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
            supported_subgoals=(OperationalSubgoal.QUERY_UNCERTAINTY,),
            description=f"Current bounded tool {tool_id}.",
        )
        for tool_id in REGRET_GUARDED_TOOL_SEQUENCE
    )


def test_build_system_prompt_keeps_optimizer_boundary_explicit() -> None:
    prompt = build_system_prompt()

    assert "optimizer is the sole order-producing boundary" in prompt.lower()
    assert "never emit replenishment orders" in prompt.lower()
    assert "regret-guarded scenario-planning tools" in prompt


def test_build_prompt_messages_include_current_context_and_tool_sequence() -> None:
    context = LLMClientContext(
        benchmark_id="serial_3_echelon",
        mission_id="test_mission",
        time_index=1,
        regime_label=RegimeLabel.DEMAND_REGIME_SHIFT,
        demand_value=14.0,
        leadtime_value=2.0,
        inventory_level=(20.0, 30.0, 40.0),
        backorder_level=(1.0, 0.0, 0.0),
        available_tool_ids=REGRET_GUARDED_TOOL_SEQUENCE,
        max_tool_steps=5,
        demand_baseline_value=10.0,
        demand_change_from_baseline=4.0,
        demand_ratio_to_baseline=1.4,
        previous_demand_value=10.0,
        demand_change_from_previous=4.0,
        downstream_inventory_value=20.0,
        total_inventory_value=90.0,
        total_backorder_value=1.0,
        inventory_gap_to_demand=6.0,
    )

    messages = build_prompt_messages(context, _tool_specs())

    assert len(messages) == 2
    assert "demand_regime_shift" in messages[1].content
    assert "- demand_ratio_to_baseline: 1.4" in messages[1].content
    assert "- demand_material_departure_from_baseline: true" in messages[1].content
    for tool_id in REGRET_GUARDED_TOOL_SEQUENCE:
        assert tool_id in messages[1].content
    assert "Return exactly one JSON object with exactly these keys and no extras" in messages[1].content
    assert "Do not wrap the JSON in code fences" in messages[1].content
    assert "confidence` must be a JSON number in [0.0, 1.0]" in messages[1].content


def test_prompt_includes_recovery_regret_guard_cues() -> None:
    context = LLMClientContext(
        benchmark_id="serial_3_echelon",
        mission_id="test_mission",
        time_index=3,
        regime_label=RegimeLabel.RECOVERY,
        demand_value=11.0,
        leadtime_value=2.0,
        inventory_level=(0.0, 0.0, 0.0),
        backorder_level=(18.0, 19.8, 22.7),
        available_tool_ids=REGRET_GUARDED_TOOL_SEQUENCE,
        max_tool_steps=5,
        demand_baseline_value=10.0,
        demand_ratio_to_baseline=1.1,
        previous_demand_value=14.0,
        downstream_inventory_value=0.0,
        total_inventory_value=0.0,
        pipeline_total_value=146.4,
        total_backorder_value=60.5,
        inventory_gap_to_demand=-11.0,
        pipeline_ratio_to_baseline=14.64,
        backorder_ratio_to_baseline=6.05,
        pipeline_heavy_vs_baseline=True,
        backlog_heavy_vs_baseline=True,
        recovery_with_high_pipeline_load=True,
        recovery_with_high_backorder_load=True,
    )

    messages = build_prompt_messages(context, _tool_specs())

    assert "- total_pipeline: 146.4" in messages[1].content
    assert "- backorder_ratio_to_baseline: 6.05" in messages[1].content
    assert "- recovery_with_high_backorder_load: true" in messages[1].content
    assert "regret guard should prevent over-correction" in messages[1].content
    assert '"update_request_types":["reweight_scenarios"]' in messages[1].content


def test_prompt_contract_metadata_is_stable_and_versioned() -> None:
    first_hash = prompt_contract_hash()
    second_hash = prompt_contract_hash()

    assert PROMPT_VERSION == "meio.llm_orchestrator.v14"
    assert first_hash == second_hash
    assert len(first_hash) == 64
