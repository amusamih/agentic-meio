"""Prompt builders for bounded LLM orchestration."""

from __future__ import annotations

import json
import hashlib

from meio.agents.llm_client import LLMClientContext, LLMMessage
from meio.contracts import OperationalSubgoal, RegimeLabel, ToolSpec, UpdateRequestType

PROMPT_VERSION = "meio.llm_orchestrator.v8"


def build_system_prompt() -> str:
    """Return the bounded system prompt for the LLM orchestrator."""

    return (
        "You are the bounded MEIO orchestration agent. "
        "Inspect typed benchmark context, choose bounded tool use, and request bounded "
        "uncertainty updates only when justified. "
        "Never emit replenishment orders, quantities, or any raw control action. "
        "The optimizer is the sole order-producing boundary. "
        "Use the provided baseline-relative evidence summaries plus current load cues to infer regime and intervention need. "
        "When bounded tools are needed, express that through query_uncertainty or update_uncertainty. "
        "Scenario updates are still bounded uncertainty actions, not raw control actions. "
        "Do not default to normal or no_action when the evidence materially departs from baseline, "
        "but do not assume each repeated stressed period requires a stronger intervention than the last. "
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
    response_contract = _build_response_contract()
    decision_signals = _build_decision_signals(context)
    decision_guidance = _build_decision_guidance(context.available_tool_ids)
    few_shot_examples = _build_few_shot_examples(context.available_tool_ids)
    external_evidence_guidance = ""
    if "external_evidence_tool" in context.available_tool_ids:
        external_evidence_guidance = (
            " If semi-synthetic external evidence is present, treat it as extra uncertainty "
            "context and use the bounded external evidence tool when that context should "
            "influence planning."
        )
    return (
        "Decision context:\n"
        f"- benchmark_id: {context.benchmark_id}\n"
        f"- mission_id: {context.mission_id}\n"
        f"- time_index: {context.time_index}\n"
        f"{decision_signals}\n"
        f"- max_tool_steps: {context.max_tool_steps}\n"
        f"- available_tool_ids: {available_tool_ids_text}\n"
        "Available bounded tools:\n"
        f"{tool_lines}\n"
        f"{decision_guidance}\n"
        f"{few_shot_examples}\n"
        f"{response_contract}\n"
        "If intervention is not justified, prefer `no_action` or `abstain` rather than "
        "inventing an intervention. "
        "If demand or lead-time signals materially depart from baseline and planning inputs "
        f"should change, do not hide that by returning normal plus no_action.{external_evidence_guidance} "
        "If stress is repeated without worsening, or recovery is already carrying heavy prior "
        "load, do not escalate just because the regime label is non-normal."
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
            _build_decision_guidance(
                ("forecast_tool", "leadtime_tool", "scenario_tool")
            ),
            _build_few_shot_examples(),
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
    lines.extend(
        f"- {name}: {value}"
        for name, value in optional_pairs
        if value is not None
    )
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
        lines.append(
            f"- leadtime_returning_toward_baseline: {str(leadtime_returning).lower()}"
        )
    boolean_pairs = (
        ("repeated_stress_detected", context.repeated_stress_detected),
        ("pipeline_heavy_vs_baseline", context.pipeline_heavy_vs_baseline),
        ("backlog_heavy_vs_baseline", context.backlog_heavy_vs_baseline),
        (
            "recovery_with_high_pipeline_load",
            context.recovery_with_high_pipeline_load,
        ),
        (
            "recovery_with_high_backorder_load",
            context.recovery_with_high_backorder_load,
        ),
    )
    lines.extend(
        f"- {name}: {str(value).lower()}"
        for name, value in boolean_pairs
        if value is not None
    )
    if context.external_evidence_present is not None:
        lines.append(
            f"- external_evidence_present: {str(context.external_evidence_present).lower()}"
        )
    if context.external_evidence_source_count is not None:
        lines.append(
            f"- external_evidence_source_count: {context.external_evidence_source_count}"
        )
    if context.external_evidence_false_alarm_present is not None:
        lines.append(
            "- external_evidence_false_alarm_present: "
            f"{str(context.external_evidence_false_alarm_present).lower()}"
        )
    lines.extend(
        f"- external_evidence_summary: {summary}"
        for summary in context.external_evidence_summaries
    )
    return "\n".join(lines)


def _build_decision_guidance(available_tool_ids: tuple[str, ...]) -> str:
    available_tools_text = json.dumps(list(available_tool_ids))
    guidance = (
        "Decision guidance:\n"
        f"- Only choose tool ids from available_tool_ids: {available_tools_text}.\n"
        "- If a normally useful tool is unavailable, do not request it. Use the remaining available tools or take a bounded no-tool action instead.\n"
        "- Use `normal` only when demand and lead-time signals stay close to baseline and no planning change is justified.\n"
        "- Use `demand_regime_shift` when demand materially departs from baseline, especially if downstream inventory is tight or demand recently jumped.\n"
        "- Use `supply_disruption` when lead-time materially departs from baseline in a way that should change planning.\n"
        "- Use `joint_disruption` when both demand and lead-time materially depart from baseline.\n"
        "- Use `recovery` when current signals move back toward baseline after a prior stressed value, especially when the latest value is closer to baseline than the previous stressed value.\n"
        "- Repeated demand stress does not automatically require stronger escalation on every stressed step.\n"
        "- When `repeated_stress_detected` is true and demand is not materially worsening versus the previous stressed period, prefer a milder bounded update such as `reweight_scenarios` over `switch_demand_regime`, and do not add `widen_uncertainty` unless ambiguity is actually increasing.\n"
        "- In recovery, use current system load as evidence about whether prior stress-driven orders are already in flight.\n"
        "- When `recovery_with_high_backorder_load` is true, prefer `keep_current` with `request_replan: false` unless demand or lead-time still materially depart from baseline or are worsening again.\n"
        "- When `recovery_with_high_pipeline_load` is true but backlog is not heavy, a milder recovery action can still be justified, but avoid reflexive replanning.\n"
        "- When bounded tools are needed to inspect or propagate uncertainty changes, choose `query_uncertainty` and list the relevant tools in execution order.\n"
        "- `scenario_tool` is allowed in the same bounded query path as a sequenced uncertainty-update step after evidence inspection, and it may still be used when forecast or leadtime outputs are partially missing.\n"
        "- The runtime will execute `candidate_tool_ids` in the exact order you provide. Do not rely on implicit extra tool chaining.\n"
        "- Use `update_uncertainty` only when the next step is directly to apply bounded uncertainty updates after evidence gathering.\n"
        "- Set `request_replan` to true when the inferred regime should change planning inputs now.\n"
        "- Do not use `request_replan` as the selected_subgoal when you are also asking for tool calls.\n"
        "- Use bounded update requests to match the signal: demand shift for demand changes, lead-time shift for lead-time changes, widen uncertainty when ambiguity remains.\n"
        "- If no suitable bounded tool is available in this ablation condition, prefer `request_replan` with `candidate_tool_ids: []` or a justified `no_action`, rather than naming a missing tool.\n"
        "- Use `keep_current` only when no planning change is justified."
    )
    if "external_evidence_tool" in available_tool_ids:
        guidance += (
            "\n- If external evidence is present and could materially clarify early, delayed, "
            "false-alarm, or relapse risk, call `external_evidence_tool` before translating that "
            "evidence into bounded update pressure.\n"
            "- Treat external evidence as exogenous uncertainty context only. It never authorizes "
            "raw control actions and it should not replace the optimizer.\n"
            "- False-alarm external evidence can justify caution, inspection, or keep_current, "
            "rather than reflexive escalation."
        )
    return guidance


def _build_few_shot_examples(
    available_tool_ids: tuple[str, ...] | None = None,
) -> str:
    preferred_tool_ids = ("forecast_tool", "leadtime_tool", "scenario_tool")
    selected_tool_ids = tuple(
        tool_id
        for tool_id in preferred_tool_ids
        if available_tool_ids is None or tool_id in available_tool_ids
    )
    if selected_tool_ids:
        demand_shift_example = json.dumps(
            {
                "selected_subgoal": "query_uncertainty",
                "candidate_tool_ids": list(selected_tool_ids),
                "regime_label": "demand_regime_shift",
                "confidence": 0.86,
                "update_request_types": ["switch_demand_regime", "widen_uncertainty"],
                "request_replan": True,
                "rationale": (
                    "Demand materially exceeds baseline and the available bounded tools "
                    "should be used in sequence before replanning."
                ),
            },
            separators=(",", ":"),
        )
    else:
        demand_shift_example = json.dumps(
            {
                "selected_subgoal": "request_replan",
                "candidate_tool_ids": [],
                "regime_label": "demand_regime_shift",
                "confidence": 0.82,
                "update_request_types": ["switch_demand_regime"],
                "request_replan": True,
                "rationale": (
                    "Demand materially exceeds baseline but no bounded uncertainty tool "
                    "is available in this ablation condition."
                ),
            },
            separators=(",", ":"),
        )
    return (
        "Examples:\n"
        "1. Normal baseline:\n"
        "Signals: demand_observed=10.0, demand_ratio_to_baseline=1.0, leadtime_ratio_to_baseline=1.0, downstream_inventory=20.0, total_backorder=0.0\n"
        '{"selected_subgoal":"no_action","candidate_tool_ids":[],"regime_label":"normal","confidence":0.95,"update_request_types":["keep_current"],"request_replan":false,"rationale":"Signals are close to baseline and no planning change is justified."}\n'
        "2. Initial demand regime shift:\n"
        "Signals: demand_observed=14.0, demand_ratio_to_baseline=1.4, demand_change_from_previous=4.0, downstream_inventory=10.0, total_backorder=0.0\n"
        f"{demand_shift_example}\n"
        "3. Repeated demand stress without worsening:\n"
        "Signals: demand_observed=14.0, previous_demand_observed=13.8, demand_ratio_to_baseline=1.4, demand_change_from_previous=0.2, repeated_stress_detected=true, downstream_inventory=0.0, total_backorder=4.0\n"
        f"{_build_repeated_stress_example(selected_tool_ids)}\n"
        "4. Recovery under heavy carried load:\n"
        "Signals: demand_observed=11.0, previous_demand_observed=14.0, demand_ratio_to_baseline=1.1, demand_change_from_previous=-3.0, demand_returning_toward_baseline=true, recovery_with_high_pipeline_load=true, recovery_with_high_backorder_load=true, total_backorder=55.0\n"
        '{"selected_subgoal":"no_action","candidate_tool_ids":[],"regime_label":"recovery","confidence":0.81,"update_request_types":["keep_current"],"request_replan":false,"rationale":"Signals are recovering and prior stress orders are already reflected in heavy carried load, so avoid another recovery replan right now."}'
    )


def _build_repeated_stress_example(selected_tool_ids: tuple[str, ...]) -> str:
    if selected_tool_ids:
        payload = {
            "selected_subgoal": "query_uncertainty",
            "candidate_tool_ids": list(selected_tool_ids),
            "regime_label": "demand_regime_shift",
            "confidence": 0.82,
            "update_request_types": ["reweight_scenarios"],
            "request_replan": True,
            "rationale": (
                "Stress persists, but the signal is not materially worse than the prior "
                "stressed period, so use a milder bounded update instead of stronger escalation."
            ),
        }
    else:
        payload = {
            "selected_subgoal": "request_replan",
            "candidate_tool_ids": [],
            "regime_label": "demand_regime_shift",
            "confidence": 0.79,
            "update_request_types": ["reweight_scenarios"],
            "request_replan": True,
            "rationale": (
                "Stress persists without clear worsening, so request a milder planning update "
                "rather than stronger escalation."
            ),
        }
    return json.dumps(payload, separators=(",", ":"))


def _material_departure_from_baseline(
    ratio_to_baseline: float | None,
) -> bool | None:
    if ratio_to_baseline is None:
        return None
    return ratio_to_baseline >= 1.15 or ratio_to_baseline <= 0.85


def _returning_toward_baseline(
    *,
    current_value: float,
    previous_value: float | None,
    baseline_value: float | None,
) -> bool | None:
    if previous_value is None or baseline_value is None:
        return None
    current_distance = abs(current_value - baseline_value)
    previous_distance = abs(previous_value - baseline_value)
    return current_distance < previous_distance


__all__ = [
    "PROMPT_VERSION",
    "build_prompt_messages",
    "build_system_prompt",
    "build_user_prompt",
    "prompt_contract_hash",
]
