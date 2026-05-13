"""Prompt builders for the bounded regret-guarded MEIO orchestration agent."""

from __future__ import annotations

import hashlib
import json

from meio.agents.llm_client import LLMClientContext, LLMMessage
from meio.contracts import OperationalSubgoal, RegimeLabel, ToolSpec, UpdateRequestType

PROMPT_VERSION = "meio.llm_orchestrator.v14"

REGRET_GUARDED_TOOL_SEQUENCE = (
    "regime_diagnosis_tool",
    "regime_belief_tool",
    "scenario_candidate_generator_tool",
    "risk_sensitive_scenario_evaluator_tool",
    "counterfactual_regret_guard_tool",
)


def build_system_prompt() -> str:
    """Return the bounded system prompt for the current orchestration agent."""

    return (
        "You are the bounded MEIO orchestration agent. "
        "Your job is to inspect typed inventory evidence, diagnose uncertainty, "
        "request the governed scenario-planning tool path when intervention is justified, "
        "and preserve the optimizer boundary. "
        "Never emit replenishment orders, quantities, or raw control actions. "
        "The optimizer is the sole order-producing boundary. "
        "Use the least strong uncertainty update that is supported by the evidence. "
        "When the regret-guarded scenario-planning tools are available, request them in the "
        "provided order so the deterministic tools perform diagnosis, candidate generation, "
        "risk-sensitive scoring, and counterfactual regret guarding before downstream handoff. "
        "Return exactly one JSON object and nothing else. "
        "Do not use markdown, code fences, headings, or prose outside the JSON object."
    )


def build_user_prompt(
    context: LLMClientContext,
    tool_specs: tuple[ToolSpec, ...],
) -> str:
    """Build a compact user prompt from typed runtime context."""

    tool_lines = "\n".join(
        f"- {tool_spec.tool_id}: {tool_spec.description or 'bounded tool'}"
        for tool_spec in tool_specs
        if tool_spec.tool_id in context.available_tool_ids
    )
    available_tool_ids_text = json.dumps(list(context.available_tool_ids))
    return (
        "Decision context:\n"
        f"- benchmark_id: {context.benchmark_id}\n"
        f"- mission_id: {context.mission_id}\n"
        f"- time_index: {context.time_index}\n"
        f"{_build_decision_signals(context)}\n"
        f"- max_tool_steps: {context.max_tool_steps}\n"
        f"- available_tool_ids: {available_tool_ids_text}\n"
        "Available bounded tools:\n"
        f"{tool_lines}\n"
        f"{_build_decision_guidance(context.available_tool_ids)}\n"
        f"{_build_few_shot_examples(context.available_tool_ids)}\n"
        f"{_build_response_contract()}\n"
        "If intervention is not justified, prefer `no_action` or `abstain` rather than "
        "inventing an intervention. If demand or lead-time signals materially depart from "
        "baseline and planning inputs should change, do not hide that by returning normal "
        "plus no_action. If recovery is already carrying heavy prior pipeline or backlog, "
        "the regret guard should be used to avoid over-correcting."
    )


def build_prompt_messages(
    context: LLMClientContext,
    tool_specs: tuple[ToolSpec, ...],
) -> tuple[LLMMessage, ...]:
    """Return the full prompt message sequence for bounded LLM orchestration."""

    return (
        LLMMessage(role="system", content=build_system_prompt()),
        LLMMessage(role="user", content=build_user_prompt(context, tool_specs)),
    )


def prompt_contract_hash() -> str:
    """Return a stable hash for the current prompt contract version."""

    payload = "\n\n".join(
        (
            PROMPT_VERSION,
            build_system_prompt(),
            _build_decision_guidance(REGRET_GUARDED_TOOL_SEQUENCE),
            _build_few_shot_examples(REGRET_GUARDED_TOOL_SEQUENCE),
            _build_response_contract(),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_response_contract() -> str:
    allowed_subgoals = ", ".join(subgoal.value for subgoal in OperationalSubgoal)
    allowed_regimes = ", ".join(label.value for label in RegimeLabel)
    allowed_update_types = ", ".join(update_type.value for update_type in UpdateRequestType)
    return (
        "Return exactly one JSON object with exactly these keys and no extras:\n"
        "{\n"
        '  "selected_subgoal": "no_action",\n'
        '  "candidate_tool_ids": [],\n'
        '  "regime_label": "normal",\n'
        '  "confidence": 0.95,\n'
        '  "update_request_types": ["keep_current"],\n'
        '  "request_replan": false,\n'
        '  "rationale": "Short bounded rationale."\n'
        "}\n"
        f"Allowed selected_subgoal values: {allowed_subgoals}\n"
        f"Allowed regime_label values: {allowed_regimes}\n"
        f"Allowed update_request_types values: {allowed_update_types}\n"
        "Rules:\n"
        "- `confidence` must be a JSON number in [0.0, 1.0], not words like high or low.\n"
        "- `candidate_tool_ids` must be a subset of `available_tool_ids`.\n"
        "- For `no_action` or `abstain`, use `candidate_tool_ids: []` and `request_replan: false`.\n"
        "- If `candidate_tool_ids` is non-empty, use `query_uncertainty` or `update_uncertainty` as the selected_subgoal.\n"
        "- Use `[]` for no update requests. Do not use null.\n"
        "- Do not copy the example values blindly; infer regime, subgoal, and updates from the signals.\n"
        "- Do not wrap the JSON in code fences.\n"
        "- Do not add explanations before or after the JSON."
    )


def _build_decision_signals(context: LLMClientContext) -> str:
    lines = [
        f"- demand_observed: {context.demand_value}",
        f"- leadtime_observed: {context.leadtime_value}",
        f"- downstream_inventory: {context.downstream_inventory_value}",
        f"- total_inventory: {context.total_inventory_value}",
        f"- total_pipeline: {context.pipeline_total_value}",
        f"- total_backorder: {context.total_backorder_value}",
        f"- inventory_gap_to_demand: {context.inventory_gap_to_demand}",
    ]
    optional_pairs = (
        ("demand_baseline", context.demand_baseline_value),
        ("demand_change_from_baseline", context.demand_change_from_baseline),
        ("demand_ratio_to_baseline", context.demand_ratio_to_baseline),
        ("previous_demand_observed", context.previous_demand_value),
        ("demand_change_from_previous", context.demand_change_from_previous),
        ("leadtime_baseline", context.leadtime_baseline_value),
        ("leadtime_change_from_baseline", context.leadtime_change_from_baseline),
        ("leadtime_ratio_to_baseline", context.leadtime_ratio_to_baseline),
        ("previous_leadtime_observed", context.previous_leadtime_value),
        ("leadtime_change_from_previous", context.leadtime_change_from_previous),
        ("pipeline_ratio_to_baseline", context.pipeline_ratio_to_baseline),
        ("backorder_ratio_to_baseline", context.backorder_ratio_to_baseline),
    )
    lines.extend(f"- {name}: {value}" for name, value in optional_pairs if value is not None)

    demand_material_departure = _material_departure_from_baseline(
        context.demand_ratio_to_baseline,
    )
    if demand_material_departure is not None:
        lines.append(
            f"- demand_material_departure_from_baseline: {str(demand_material_departure).lower()}"
        )
    leadtime_material_departure = _material_departure_from_baseline(
        context.leadtime_ratio_to_baseline,
    )
    if leadtime_material_departure is not None:
        lines.append(
            "- leadtime_material_departure_from_baseline: "
            f"{str(leadtime_material_departure).lower()}"
        )

    demand_returning = _returning_toward_baseline(
        current_value=context.demand_value,
        previous_value=context.previous_demand_value,
        baseline_value=context.demand_baseline_value,
    )
    if demand_returning is not None:
        lines.append(f"- demand_returning_toward_baseline: {str(demand_returning).lower()}")
    leadtime_returning = _returning_toward_baseline(
        current_value=context.leadtime_value,
        previous_value=context.previous_leadtime_value,
        baseline_value=context.leadtime_baseline_value,
    )
    if leadtime_returning is not None:
        lines.append(f"- leadtime_returning_toward_baseline: {str(leadtime_returning).lower()}")

    boolean_pairs = (
        ("repeated_stress_detected", context.repeated_stress_detected),
        ("pipeline_heavy_vs_baseline", context.pipeline_heavy_vs_baseline),
        ("backlog_heavy_vs_baseline", context.backlog_heavy_vs_baseline),
        ("recovery_with_high_pipeline_load", context.recovery_with_high_pipeline_load),
        ("recovery_with_high_backorder_load", context.recovery_with_high_backorder_load),
    )
    lines.extend(
        f"- {name}: {str(value).lower()}"
        for name, value in boolean_pairs
        if value is not None
    )
    return "\n".join(lines)


def _build_decision_guidance(available_tool_ids: tuple[str, ...]) -> str:
    available_tools_text = json.dumps(list(available_tool_ids))
    sequence_available = all(tool_id in available_tool_ids for tool_id in REGRET_GUARDED_TOOL_SEQUENCE)
    sequence_guidance = ""
    if sequence_available:
        sequence_guidance = (
            "\n"
            "- If intervention is justified, request the full regret-guarded sequence: "
            "`regime_diagnosis_tool`, `regime_belief_tool`, "
            "`scenario_candidate_generator_tool`, "
            "`risk_sensitive_scenario_evaluator_tool`, and "
            "`counterfactual_regret_guard_tool`.\n"
            "- The first two tools diagnose the current regime and uncertainty belief; "
            "the generator constructs candidate scenario inputs; the evaluator scores "
            "expected cost plus uncertainty quality; the regret guard blocks candidates "
            "that are not justified against the protected incumbent.\n"
            "- Your role is to choose whether the evidence warrants this bounded tool "
            "path and to report regime, confidence, and update intent. The tools and "
            "trusted downstream decision component handle scenario resolution and orders."
        )
    return (
        "Decision guidance:\n"
        f"- You may only request tools from available_tool_ids={available_tools_text}.\n"
        "- Do not request tools for routine normal periods unless evidence shows material demand, lead-time, backlog, or pipeline stress.\n"
        "- Repeated stress does not automatically require stronger escalation; prefer a bounded update and let the regret guard check whether it is worth applying.\n"
        "- Recovery with high carried pipeline or backlog is risky; use the regret guard to avoid over-ordering from stale stress.\n"
        "- The agent may request scenario updates, but must never generate raw replenishment requests."
        f"{sequence_guidance}"
    )


def _build_few_shot_examples(available_tool_ids: tuple[str, ...] = REGRET_GUARDED_TOOL_SEQUENCE) -> str:
    selected_tool_ids = tuple(
        tool_id for tool_id in REGRET_GUARDED_TOOL_SEQUENCE if tool_id in available_tool_ids
    )
    tool_json = json.dumps(list(selected_tool_ids), separators=(",", ":"))
    return (
        "Examples:\n"
        "1. Normal stable period:\n"
        '{"selected_subgoal":"no_action","candidate_tool_ids":[],"regime_label":"normal",'
        '"confidence":0.84,"update_request_types":["keep_current"],"request_replan":false,'
        '"rationale":"Evidence is close to baseline, so no uncertainty intervention is justified."}\n'
        "2. Material demand or lead-time shift:\n"
        '{"selected_subgoal":"query_uncertainty","candidate_tool_ids":'
        f"{tool_json}"
        ',"regime_label":"demand_regime_shift","confidence":0.88,'
        '"update_request_types":["switch_demand_regime","widen_uncertainty"],'
        '"request_replan":true,'
        '"rationale":"Material departure from baseline warrants bounded scenario planning and regret guarding."}\n'
        "3. Recovery with high carried load:\n"
        '{"selected_subgoal":"query_uncertainty","candidate_tool_ids":'
        f"{tool_json}"
        ',"regime_label":"recovery","confidence":0.78,'
        '"update_request_types":["reweight_scenarios"],"request_replan":true,'
        '"rationale":"Recovery is visible but carried pipeline or backlog means the regret guard should prevent over-correction."}'
    )


def _material_departure_from_baseline(ratio: float | None) -> bool | None:
    if ratio is None:
        return None
    return ratio <= 0.75 or ratio >= 1.25


def _returning_toward_baseline(
    *,
    current_value: float | None,
    previous_value: float | None,
    baseline_value: float | None,
) -> bool | None:
    if current_value is None or previous_value is None or baseline_value in (None, 0.0):
        return None
    return abs(current_value - baseline_value) < abs(previous_value - baseline_value)


__all__ = [
    "PROMPT_VERSION",
    "REGRET_GUARDED_TOOL_SEQUENCE",
    "build_prompt_messages",
    "build_system_prompt",
    "build_user_prompt",
    "prompt_contract_hash",
]
