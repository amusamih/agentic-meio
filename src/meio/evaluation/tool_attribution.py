"""Tool attribution and ablation summaries for the active Stockpyl path."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from meio.evaluation.logging_schema import (
    EpisodeSummaryRecord,
    StepTraceRecord,
    ToolCallTraceRecord,
)

REMOVED_TOOL_BY_ABLATION = {
    "full": None,
    "no_forecast_tool": "forecast_tool",
    "no_leadtime_tool": "leadtime_tool",
    "no_scenario_tool": "scenario_tool",
}


def _mean_or_none(values: tuple[float | None, ...]) -> float | None:
    numeric = tuple(value for value in values if value is not None)
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


@dataclass(frozen=True, slots=True)
class ToolAttributionRecord:
    """Direct tool-use attribution summary from bounded tool-call traces."""

    tool_id: str
    mode: str
    tool_ablation_variant: str
    call_count: int
    periods_called: tuple[int, ...]
    schedules_called_in: tuple[str, ...]
    changed_regime_label_count: int | None
    changed_update_request_count: int | None
    changed_replan_flag_count: int | None
    changed_optimizer_input_count: int | None
    changed_optimizer_orders_count: int | None
    changed_outcome_proxy_count: int | None

    def __post_init__(self) -> None:
        if not self.tool_id.strip():
            raise ValueError("tool_id must be non-empty.")
        if not self.mode.strip():
            raise ValueError("mode must be non-empty.")
        if not self.tool_ablation_variant.strip():
            raise ValueError("tool_ablation_variant must be non-empty.")
        if self.call_count < 0:
            raise ValueError("call_count must be non-negative.")
        object.__setattr__(self, "periods_called", tuple(self.periods_called))
        object.__setattr__(self, "schedules_called_in", tuple(self.schedules_called_in))


@dataclass(frozen=True, slots=True)
class ToolUsageSummary:
    """Per-mode and per-ablation tool-use summary."""

    mode: str
    tool_ablation_variant: str
    tool_records: tuple[ToolAttributionRecord, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.mode.strip():
            raise ValueError("mode must be non-empty.")
        if not self.tool_ablation_variant.strip():
            raise ValueError("tool_ablation_variant must be non-empty.")
        object.__setattr__(self, "tool_records", tuple(self.tool_records))


@dataclass(frozen=True, slots=True)
class ToolAblationSummary:
    """Matched comparison between one ablation variant and the full tool set."""

    mode: str
    tool_ablation_variant: str
    removed_tool_id: str | None
    run_count: int
    regime_accuracy_delta_vs_full: float | None
    no_action_rate_delta_vs_full: float | None
    replan_rate_delta_vs_full: float | None
    intervention_rate_delta_vs_full: float | None
    missed_intervention_delta_vs_full: int | None
    unnecessary_intervention_delta_vs_full: int | None
    total_cost_delta_vs_full: float | None
    fill_rate_delta_vs_full: float | None
    tool_call_count_delta_vs_full: float | None
    decision_change_count: int | None
    optimizer_input_change_count: int | None
    optimizer_orders_change_count: int | None
    outcome_proxy_change_count: int | None
    unavailable_tool_request_count: int = 0
    disabled_tool_fallback_count: int = 0
    sequencing_blocked_tool_request_count: int = 0
    clean_intervention_count: int = 0
    clean_optimizer_input_change_count: int = 0

    def __post_init__(self) -> None:
        if not self.mode.strip():
            raise ValueError("mode must be non-empty.")
        if not self.tool_ablation_variant.strip():
            raise ValueError("tool_ablation_variant must be non-empty.")
        for field_name in (
            "run_count",
            "unavailable_tool_request_count",
            "disabled_tool_fallback_count",
            "sequencing_blocked_tool_request_count",
            "clean_intervention_count",
            "clean_optimizer_input_change_count",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative.")


def summarize_tool_usage(
    *,
    mode: str,
    tool_ablation_variant: str,
    tool_call_records: tuple[ToolCallTraceRecord, ...],
) -> ToolUsageSummary:
    """Aggregate direct bounded tool-use traces for one mode and ablation."""

    records_by_tool: dict[str, list[ToolCallTraceRecord]] = defaultdict(list)
    for record in tool_call_records:
        records_by_tool[record.tool_id].append(record)
    tool_records = tuple(
        _build_tool_attribution_record(
            tool_id=tool_id,
            mode=mode,
            tool_ablation_variant=tool_ablation_variant,
            records=tuple(records),
        )
        for tool_id, records in sorted(records_by_tool.items())
    )
    return ToolUsageSummary(
        mode=mode,
        tool_ablation_variant=tool_ablation_variant,
        tool_records=tool_records,
    )


def summarize_tool_ablation(
    *,
    mode: str,
    tool_ablation_variant: str,
    full_episode_records: tuple[EpisodeSummaryRecord, ...],
    ablated_episode_records: tuple[EpisodeSummaryRecord, ...],
    full_step_records: tuple[StepTraceRecord, ...],
    ablated_step_records: tuple[StepTraceRecord, ...],
) -> ToolAblationSummary:
    """Compare one tool-ablation variant against the full tool set."""

    removed_tool_id = REMOVED_TOOL_BY_ABLATION[tool_ablation_variant]
    if tool_ablation_variant == "full":
        return ToolAblationSummary(
            mode=mode,
            tool_ablation_variant=tool_ablation_variant,
            removed_tool_id=removed_tool_id,
            run_count=len(ablated_episode_records),
            regime_accuracy_delta_vs_full=0.0,
            no_action_rate_delta_vs_full=0.0,
            replan_rate_delta_vs_full=0.0,
            intervention_rate_delta_vs_full=0.0,
            missed_intervention_delta_vs_full=0,
            unnecessary_intervention_delta_vs_full=0,
            total_cost_delta_vs_full=0.0,
            fill_rate_delta_vs_full=0.0,
            tool_call_count_delta_vs_full=0.0,
            decision_change_count=0,
            optimizer_input_change_count=0,
            optimizer_orders_change_count=0,
            outcome_proxy_change_count=0,
            unavailable_tool_request_count=0,
            disabled_tool_fallback_count=0,
            sequencing_blocked_tool_request_count=0,
            clean_intervention_count=sum(
                record.clean_intervention_count for record in ablated_episode_records
            ),
            clean_optimizer_input_change_count=sum(
                record.clean_optimizer_input_change_count
                for record in ablated_episode_records
            ),
        )
    matched_full_episodes = _match_episode_records(full_episode_records, ablated_episode_records)
    matched_full_steps, matched_ablated_steps = _match_step_records(
        full_step_records,
        ablated_step_records,
    )
    return ToolAblationSummary(
        mode=mode,
        tool_ablation_variant=tool_ablation_variant,
        removed_tool_id=removed_tool_id,
        run_count=len(matched_full_episodes),
        regime_accuracy_delta_vs_full=_episode_delta(
            matched_full_episodes,
            ablated_episode_records,
            "regime_prediction_accuracy",
        ),
        no_action_rate_delta_vs_full=_rate_delta(
            matched_full_episodes,
            ablated_episode_records,
            numerator_field="no_action_count",
        ),
        replan_rate_delta_vs_full=_rate_delta(
            matched_full_episodes,
            ablated_episode_records,
            numerator_field="replan_count",
        ),
        intervention_rate_delta_vs_full=_rate_delta(
            matched_full_episodes,
            ablated_episode_records,
            numerator_field="intervention_count",
        ),
        missed_intervention_delta_vs_full=_int_episode_delta(
            matched_full_episodes,
            ablated_episode_records,
            "missed_intervention_count",
        ),
        unnecessary_intervention_delta_vs_full=_int_episode_delta(
            matched_full_episodes,
            ablated_episode_records,
            "unnecessary_intervention_count",
        ),
        total_cost_delta_vs_full=_episode_delta(
            matched_full_episodes,
            ablated_episode_records,
            "total_cost",
        ),
        fill_rate_delta_vs_full=_episode_delta(
            matched_full_episodes,
            ablated_episode_records,
            "fill_rate",
        ),
        tool_call_count_delta_vs_full=_episode_delta(
            matched_full_episodes,
            ablated_episode_records,
            "tool_call_count",
        ),
        decision_change_count=sum(
            1
            for full_record, ablated_record in zip(
                matched_full_steps,
                matched_ablated_steps,
                strict=True,
            )
            if _decision_state(full_record) != _decision_state(ablated_record)
        ),
        optimizer_input_change_count=sum(
            1
            for full_record, ablated_record in zip(
                matched_full_steps,
                matched_ablated_steps,
                strict=True,
            )
            if _optimizer_input_state(full_record) != _optimizer_input_state(ablated_record)
        ),
        optimizer_orders_change_count=sum(
            1
            for full_record, ablated_record in zip(
                matched_full_steps,
                matched_ablated_steps,
                strict=True,
            )
            if full_record.optimizer_orders != ablated_record.optimizer_orders
        ),
        outcome_proxy_change_count=sum(
            1
            for full_record, ablated_record in zip(
                matched_full_steps,
                matched_ablated_steps,
                strict=True,
            )
            if _outcome_proxy_state(full_record) != _outcome_proxy_state(ablated_record)
        ),
        unavailable_tool_request_count=sum(
            record.unavailable_tool_request_count for record in ablated_episode_records
        ),
        disabled_tool_fallback_count=sum(
            record.disabled_tool_fallback_count for record in ablated_episode_records
        ),
        sequencing_blocked_tool_request_count=sum(
            record.sequencing_blocked_tool_request_count
            for record in ablated_episode_records
        ),
        clean_intervention_count=sum(
            record.clean_intervention_count for record in ablated_episode_records
        ),
        clean_optimizer_input_change_count=sum(
            record.clean_optimizer_input_change_count for record in ablated_episode_records
        ),
    )


def _build_tool_attribution_record(
    *,
    tool_id: str,
    mode: str,
    tool_ablation_variant: str,
    records: tuple[ToolCallTraceRecord, ...],
) -> ToolAttributionRecord:
    return ToolAttributionRecord(
        tool_id=tool_id,
        mode=mode,
        tool_ablation_variant=tool_ablation_variant,
        call_count=len(records),
        periods_called=tuple(sorted({record.period_index for record in records})),
        schedules_called_in=tuple(
            sorted({record.schedule_name for record in records if record.schedule_name is not None})
        ),
        changed_regime_label_count=sum(
            1
            for record in records
            if _changed_decision_field(record, "regime_label")
        ),
        changed_update_request_count=sum(
            1
            for record in records
            if _changed_decision_field(record, "update_request_types")
        ),
        changed_replan_flag_count=sum(
            1
            for record in records
            if _changed_decision_field(record, "request_replan")
        ),
        changed_optimizer_input_count=sum(
            1
            for record in records
            if record.optimizer_input_changed is True
        ),
        changed_optimizer_orders_count=None,
        changed_outcome_proxy_count=None,
    )


def _changed_decision_field(record: ToolCallTraceRecord, field_name: str) -> bool:
    if record.pre_tool_decision is None or record.post_tool_decision is None:
        return False
    return record.pre_tool_decision.get(field_name) != record.post_tool_decision.get(field_name)


def _match_episode_records(
    full_records: tuple[EpisodeSummaryRecord, ...],
    ablated_records: tuple[EpisodeSummaryRecord, ...],
) -> tuple[EpisodeSummaryRecord, ...]:
    ablated_keys = {
        (record.schedule_name, record.run_seed)
        for record in ablated_records
    }
    return tuple(
        record
        for record in full_records
        if (record.schedule_name, record.run_seed) in ablated_keys
    )


def _match_step_records(
    full_records: tuple[StepTraceRecord, ...],
    ablated_records: tuple[StepTraceRecord, ...],
) -> tuple[tuple[StepTraceRecord, ...], tuple[StepTraceRecord, ...]]:
    full_by_key = {
        (record.schedule_name, record.run_seed, record.period_index): record
        for record in full_records
    }
    matched_full: list[StepTraceRecord] = []
    matched_ablated: list[StepTraceRecord] = []
    for record in ablated_records:
        key = (record.schedule_name, record.run_seed, record.period_index)
        full_record = full_by_key.get(key)
        if full_record is None:
            continue
        matched_full.append(full_record)
        matched_ablated.append(record)
    return tuple(matched_full), tuple(matched_ablated)


def _episode_delta(
    full_records: tuple[EpisodeSummaryRecord, ...],
    ablated_records: tuple[EpisodeSummaryRecord, ...],
    field_name: str,
) -> float | None:
    if not full_records or not ablated_records:
        return None
    full_mean = _mean_or_none(tuple(getattr(record, field_name) for record in full_records))
    ablated_mean = _mean_or_none(
        tuple(getattr(record, field_name) for record in ablated_records)
    )
    if full_mean is None or ablated_mean is None:
        return None
    return ablated_mean - full_mean


def _int_episode_delta(
    full_records: tuple[EpisodeSummaryRecord, ...],
    ablated_records: tuple[EpisodeSummaryRecord, ...],
    field_name: str,
) -> int | None:
    if not full_records or not ablated_records:
        return None
    return sum(getattr(record, field_name) for record in ablated_records) - sum(
        getattr(record, field_name) for record in full_records
    )


def _rate_delta(
    full_records: tuple[EpisodeSummaryRecord, ...],
    ablated_records: tuple[EpisodeSummaryRecord, ...],
    *,
    numerator_field: str,
) -> float | None:
    full_steps = sum(record.step_count for record in full_records)
    ablated_steps = sum(record.step_count for record in ablated_records)
    if full_steps == 0 or ablated_steps == 0:
        return None
    full_rate = sum(getattr(record, numerator_field) for record in full_records) / full_steps
    ablated_rate = sum(getattr(record, numerator_field) for record in ablated_records) / ablated_steps
    return ablated_rate - full_rate


def _decision_state(record: StepTraceRecord) -> tuple[object, ...]:
    return (
        record.predicted_regime_label,
        record.selected_subgoal,
        record.update_requests,
        record.request_replan,
        record.abstain_or_no_action,
    )


def _optimizer_input_state(record: StepTraceRecord) -> tuple[object, ...]:
    return (
        record.demand_outlook,
        record.leadtime_outlook,
        record.scenario_adjustment_summary,
    )


def _outcome_proxy_state(record: StepTraceRecord) -> tuple[object, ...]:
    return (
        record.inventory_by_echelon,
        record.backorders_by_echelon,
        record.per_period_cost,
        record.per_period_fill_rate,
    )


__all__ = [
    "REMOVED_TOOL_BY_ABLATION",
    "ToolAblationSummary",
    "ToolAttributionRecord",
    "ToolUsageSummary",
    "summarize_tool_ablation",
    "summarize_tool_usage",
]
