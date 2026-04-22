"""Aggregate summaries for controlled multi-schedule benchmark evaluation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from meio.evaluation.logging_schema import (
    ArtifactUseClass,
    EpisodeSummaryRecord,
    StepTraceRecord,
    ToolCallTraceRecord,
    SCHEMA_VERSION,
)
from meio.evaluation.tool_attribution import (
    ToolAblationSummary,
    ToolUsageSummary,
    summarize_tool_ablation,
    summarize_tool_usage,
)


def _mean_or_none(values: tuple[float | int | None, ...]) -> float | None:
    numeric = tuple(float(value) for value in values if value is not None)
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def _single_text_or_mixed(values: tuple[str | None, ...]) -> str | None:
    normalized = tuple(value for value in values if value is not None)
    if not normalized:
        return None
    if len(set(normalized)) == 1:
        return normalized[0]
    return "mixed"


@dataclass(frozen=True, slots=True)
class AggregateBenchmarkMetrics:
    """Aggregate benchmark-level metrics over repeated controlled runs."""

    average_total_cost: float | None
    average_fill_rate: float | None
    average_inventory: float | None
    average_backorder_level: float | None
    average_tool_call_count: float

    def __post_init__(self) -> None:
        if self.average_tool_call_count < 0.0:
            raise ValueError("average_tool_call_count must be non-negative.")


@dataclass(frozen=True, slots=True)
class AggregateValiditySummary:
    """Validity and safety summary aligned with the evaluation rubric."""

    optimizer_order_boundary_preserved: bool
    invalid_output_count: int
    fallback_count: int
    failure_before_response_count: int
    failure_after_response_count: int
    rollout_fidelity_gate_passed: bool = False
    operational_metrics_gate_passed: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "invalid_output_count",
            "fallback_count",
            "failure_before_response_count",
            "failure_after_response_count",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative.")


@dataclass(frozen=True, slots=True)
class AggregatePerformanceSummary:
    """Operational performance summary aligned with the evaluation rubric."""

    average_total_cost: float | None
    average_fill_rate: float | None
    average_inventory: float | None
    average_backorder_level: float | None

    def __post_init__(self) -> None:
        for field_name in (
            "average_total_cost",
            "average_fill_rate",
            "average_inventory",
            "average_backorder_level",
        ):
            value = getattr(self, field_name)
            if value is not None and value < 0.0:
                raise ValueError(f"{field_name} must be non-negative when provided.")


@dataclass(frozen=True, slots=True)
class AggregateToolUseSummary:
    """Compact tool-use summary for comparing bounded outer-loop value."""

    average_tool_call_count: float
    no_action_rate: float
    replan_rate: float
    intervention_rate: float
    unavailable_tool_request_count: int = 0
    disabled_tool_fallback_count: int = 0
    sequencing_blocked_tool_request_count: int = 0
    clean_intervention_count: int = 0
    clean_optimizer_input_change_count: int = 0
    repeated_stress_moderation_count: int = 0
    relapse_moderation_count: int = 0
    unresolved_stress_moderation_count: int = 0
    moderated_update_count: int = 0
    hysteresis_application_count: int = 0

    def __post_init__(self) -> None:
        if self.average_tool_call_count < 0.0:
            raise ValueError("average_tool_call_count must be non-negative.")
        for field_name in ("no_action_rate", "replan_rate", "intervention_rate"):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be within [0.0, 1.0].")
        for field_name in (
            "unavailable_tool_request_count",
            "disabled_tool_fallback_count",
            "sequencing_blocked_tool_request_count",
            "clean_intervention_count",
            "clean_optimizer_input_change_count",
            "repeated_stress_moderation_count",
            "relapse_moderation_count",
            "unresolved_stress_moderation_count",
            "moderated_update_count",
            "hysteresis_application_count",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative.")


@dataclass(frozen=True, slots=True)
class AggregateExternalEvidenceSummary:
    """Compact summary for semi-synthetic external evidence use."""

    external_evidence_source: str | None
    external_evidence_period_count: int
    external_evidence_tool_call_count: int
    false_alarm_evidence_count: int
    evidence_supported_intervention_count: int
    external_evidence_changed_optimizer_input_count: int
    evidence_fusion_cap_count: int = 0
    capped_external_strengthening_count: int = 0
    early_evidence_confirmation_gate_count: int = 0
    early_evidence_family_change_block_count: int = 0
    corroboration_gate_count: int = 0
    corroborated_family_change_count: int = 0
    role_level_usage_counts: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        if self.external_evidence_source is not None and not self.external_evidence_source.strip():
            raise ValueError("external_evidence_source must be non-empty when provided.")
        for field_name in (
            "external_evidence_period_count",
            "external_evidence_tool_call_count",
            "false_alarm_evidence_count",
            "evidence_supported_intervention_count",
            "external_evidence_changed_optimizer_input_count",
            "evidence_fusion_cap_count",
            "capped_external_strengthening_count",
            "early_evidence_confirmation_gate_count",
            "early_evidence_family_change_block_count",
            "corroboration_gate_count",
            "corroborated_family_change_count",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative.")
        object.__setattr__(self, "role_level_usage_counts", tuple(self.role_level_usage_counts))
        for role_label, count in self.role_level_usage_counts:
            if not role_label.strip():
                raise ValueError("role_level_usage_counts must use non-empty role labels.")
            if count < 0:
                raise ValueError("role_level_usage_counts must use non-negative counts.")


@dataclass(frozen=True, slots=True)
class AggregateRobustnessSummary:
    """Compact robustness-screen summary across schedules and seeds."""

    run_count: int
    schedule_count: int
    seed_count: int
    schedule_names: tuple[str, ...]
    seed_values: tuple[int, ...]
    schedule_run_counts: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("run_count", "schedule_count", "seed_count"):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative.")
        object.__setattr__(self, "schedule_names", tuple(self.schedule_names))
        object.__setattr__(self, "seed_values", tuple(self.seed_values))
        object.__setattr__(self, "schedule_run_counts", tuple(self.schedule_run_counts))
        for schedule_name in self.schedule_names:
            if not schedule_name.strip():
                raise ValueError("schedule_names must contain non-empty names.")
        for schedule_name, count in self.schedule_run_counts:
            if not schedule_name.strip():
                raise ValueError("schedule_run_counts must use non-empty schedule names.")
            if count < 0:
                raise ValueError("schedule_run_counts must use non-negative counts.")
        for seed_value in self.seed_values:
            if seed_value < 0:
                raise ValueError("seed_values must use non-negative integers.")


@dataclass(frozen=True, slots=True)
class AggregateDecisionQualityMetrics:
    """Aggregate decision-quality metrics across episode summaries."""

    regime_prediction_accuracy: float | None
    no_action_rate: float
    replan_rate: float
    intervention_rate: float
    missed_intervention_count: int
    unnecessary_intervention_count: int
    average_confidence: float | None
    predicted_regime_counts: tuple[tuple[str, int], ...] = ()
    confusion_counts: tuple[tuple[str, str, int], ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("no_action_rate", "replan_rate", "intervention_rate"):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be within [0.0, 1.0].")
        for field_name in ("missed_intervention_count", "unnecessary_intervention_count"):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative.")


@dataclass(frozen=True, slots=True)
class AggregateTelemetryMetrics:
    """Aggregate runtime telemetry over repeated controlled runs."""

    average_prompt_tokens: float | None
    average_completion_tokens: float | None
    average_total_tokens: float | None
    average_llm_latency_ms: float | None
    average_orchestration_latency_ms: float | None
    invalid_output_count: int
    fallback_count: int
    client_error_counts: tuple[tuple[str, int], ...] = ()
    total_retry_count: int = 0
    failure_before_response_count: int = 0
    failure_after_response_count: int = 0

    def __post_init__(self) -> None:
        for field_name in (
            "invalid_output_count",
            "fallback_count",
            "total_retry_count",
            "failure_before_response_count",
            "failure_after_response_count",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative.")


@dataclass(frozen=True, slots=True)
class ScheduleAggregateSummary:
    """Aggregate summary for one named schedule within one mode."""

    schedule_name: str
    run_count: int
    seed_values: tuple[int, ...]
    benchmark_metrics: AggregateBenchmarkMetrics
    decision_quality: AggregateDecisionQualityMetrics
    telemetry_metrics: AggregateTelemetryMetrics
    validity_summary: AggregateValiditySummary
    performance_summary: AggregatePerformanceSummary
    tool_use_summary: AggregateToolUseSummary
    external_evidence_summary: AggregateExternalEvidenceSummary | None
    robustness_summary: AggregateRobustnessSummary

    def __post_init__(self) -> None:
        if not self.schedule_name.strip():
            raise ValueError("schedule_name must be non-empty.")
        if self.run_count <= 0:
            raise ValueError("run_count must be positive.")
        object.__setattr__(self, "seed_values", tuple(self.seed_values))


@dataclass(frozen=True, slots=True)
class ModeAggregateSummary:
    """Aggregate summary for one evaluation mode across schedules and seeds."""

    mode: str
    run_count: int
    schedule_names: tuple[str, ...]
    seed_values: tuple[int, ...]
    benchmark_metrics: AggregateBenchmarkMetrics
    decision_quality: AggregateDecisionQualityMetrics
    telemetry_metrics: AggregateTelemetryMetrics
    validity_summary: AggregateValiditySummary
    performance_summary: AggregatePerformanceSummary
    tool_use_summary: AggregateToolUseSummary
    external_evidence_summary: AggregateExternalEvidenceSummary | None
    robustness_summary: AggregateRobustnessSummary
    optimizer_order_boundary_preserved: bool
    validation_lane: str | None = None
    artifact_use_class: ArtifactUseClass = ArtifactUseClass.INTERNAL_ONLY
    validity_gate_passed: bool = False
    eligibility_notes: tuple[str, ...] = field(default_factory=tuple)
    tool_ablation_variants: tuple[str, ...] = field(default_factory=tuple)
    tool_attribution: tuple[ToolUsageSummary, ...] = field(default_factory=tuple)
    ablation_breakdown: tuple["AblationAggregateSummary", ...] = field(default_factory=tuple)
    schedule_breakdown: tuple[ScheduleAggregateSummary, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.mode.strip():
            raise ValueError("mode must be non-empty.")
        if self.validation_lane is not None and not self.validation_lane.strip():
            raise ValueError("validation_lane must be non-empty when provided.")
        if self.run_count <= 0:
            raise ValueError("run_count must be positive.")
        object.__setattr__(self, "schedule_names", tuple(self.schedule_names))
        object.__setattr__(self, "seed_values", tuple(self.seed_values))
        object.__setattr__(self, "eligibility_notes", tuple(self.eligibility_notes))
        object.__setattr__(self, "tool_ablation_variants", tuple(self.tool_ablation_variants))
        object.__setattr__(self, "tool_attribution", tuple(self.tool_attribution))
        object.__setattr__(self, "ablation_breakdown", tuple(self.ablation_breakdown))
        object.__setattr__(self, "schedule_breakdown", tuple(self.schedule_breakdown))


@dataclass(frozen=True, slots=True)
class BatchAggregateSummary:
    """Top-level aggregate comparison summary across supported modes."""

    benchmark_id: str
    mode_names: tuple[str, ...]
    schedule_names: tuple[str, ...]
    seed_values: tuple[int, ...]
    tool_ablation_variants: tuple[str, ...]
    mode_summaries: tuple[ModeAggregateSummary, ...]
    validation_lane: str | None = None
    artifact_use_class: ArtifactUseClass = ArtifactUseClass.INTERNAL_ONLY
    validity_gate_passed: bool = False
    eligibility_notes: tuple[str, ...] = field(default_factory=tuple)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.benchmark_id.strip():
            raise ValueError("benchmark_id must be non-empty.")
        if self.validation_lane is not None and not self.validation_lane.strip():
            raise ValueError("validation_lane must be non-empty when provided.")
        object.__setattr__(self, "mode_names", tuple(self.mode_names))
        object.__setattr__(self, "schedule_names", tuple(self.schedule_names))
        object.__setattr__(self, "seed_values", tuple(self.seed_values))
        object.__setattr__(self, "tool_ablation_variants", tuple(self.tool_ablation_variants))
        object.__setattr__(self, "mode_summaries", tuple(self.mode_summaries))
        object.__setattr__(self, "eligibility_notes", tuple(self.eligibility_notes))


@dataclass(frozen=True, slots=True)
class AblationAggregateSummary:
    """Aggregate summary for one tool-ablation variant within one mode."""

    tool_ablation_variant: str
    run_count: int
    schedule_names: tuple[str, ...]
    seed_values: tuple[int, ...]
    benchmark_metrics: AggregateBenchmarkMetrics
    decision_quality: AggregateDecisionQualityMetrics
    telemetry_metrics: AggregateTelemetryMetrics
    validity_summary: AggregateValiditySummary
    performance_summary: AggregatePerformanceSummary
    tool_use_summary: AggregateToolUseSummary
    external_evidence_summary: AggregateExternalEvidenceSummary | None
    robustness_summary: AggregateRobustnessSummary
    optimizer_order_boundary_preserved: bool
    tool_usage_summary: ToolUsageSummary | None = None
    tool_ablation_summary: ToolAblationSummary | None = None

    def __post_init__(self) -> None:
        if not self.tool_ablation_variant.strip():
            raise ValueError("tool_ablation_variant must be non-empty.")
        if self.run_count <= 0:
            raise ValueError("run_count must be positive.")
        object.__setattr__(self, "schedule_names", tuple(self.schedule_names))
        object.__setattr__(self, "seed_values", tuple(self.seed_values))


def aggregate_mode_episode_summaries(
    mode: str,
    records: tuple[EpisodeSummaryRecord, ...],
    *,
    validation_lane: str | None = None,
    step_trace_records: tuple[StepTraceRecord, ...] = (),
    tool_call_records: tuple[ToolCallTraceRecord, ...] = (),
) -> ModeAggregateSummary:
    """Aggregate episode summaries for one mode across schedules and seeds."""

    if not records:
        raise ValueError("records must not be empty.")
    schedule_names = tuple(
        sorted(
            {
                record.schedule_name
                for record in records
                if record.schedule_name is not None
            }
        )
    )
    seed_values = tuple(
        sorted(
            {
                record.run_seed
                for record in records
                if record.run_seed is not None
            }
        )
    )
    tool_ablation_variants = tuple(
        sorted({record.tool_ablation_variant for record in records})
    )
    schedule_breakdown = tuple(
        _aggregate_schedule_records(schedule_name, records)
        for schedule_name in schedule_names
    )
    tool_attribution = tuple(
        summarize_tool_usage(
            mode=mode,
            tool_ablation_variant=tool_ablation_variant,
            tool_call_records=tuple(
                record
                for record in tool_call_records
                if record.tool_ablation_variant == tool_ablation_variant
            ),
        )
        for tool_ablation_variant in tool_ablation_variants
    )
    full_records = tuple(
        record for record in records if record.tool_ablation_variant == "full"
    )
    full_step_records = tuple(
        record for record in step_trace_records if record.tool_ablation_variant == "full"
    )
    ablation_breakdown = tuple(
        _aggregate_ablation_records(
            mode=mode,
            tool_ablation_variant=tool_ablation_variant,
            records=records,
            step_trace_records=step_trace_records,
            tool_call_records=tool_call_records,
            full_records=full_records,
            full_step_records=full_step_records,
        )
        for tool_ablation_variant in tool_ablation_variants
    )
    return ModeAggregateSummary(
        mode=mode,
        validation_lane=validation_lane,
        run_count=len(records),
        schedule_names=schedule_names,
        seed_values=seed_values,
        benchmark_metrics=_aggregate_benchmark_metrics(records),
        decision_quality=_aggregate_decision_quality(records),
        telemetry_metrics=_aggregate_telemetry_metrics(records),
        validity_summary=_aggregate_validity_summary(records),
        performance_summary=_aggregate_performance_summary(records),
        tool_use_summary=_aggregate_tool_use_summary(records),
        external_evidence_summary=_aggregate_external_evidence_summary(records),
        robustness_summary=_aggregate_robustness_summary(
            records=records,
            schedule_names=schedule_names,
            seed_values=seed_values,
        ),
        optimizer_order_boundary_preserved=all(
            record.optimizer_order_boundary_preserved for record in records
        ),
        tool_ablation_variants=tool_ablation_variants,
        tool_attribution=tool_attribution,
        ablation_breakdown=ablation_breakdown,
        schedule_breakdown=schedule_breakdown,
    )


def aggregate_batch_episode_summaries(
    *,
    benchmark_id: str,
    records_by_mode: dict[str, tuple[EpisodeSummaryRecord, ...]],
    validation_lane: str | None = None,
    step_trace_records_by_mode: dict[str, tuple[StepTraceRecord, ...]] | None = None,
    tool_call_records_by_mode: dict[str, tuple[ToolCallTraceRecord, ...]] | None = None,
) -> BatchAggregateSummary:
    """Aggregate episode summaries across modes for one benchmark batch."""

    if not records_by_mode:
        raise ValueError("records_by_mode must not be empty.")
    step_trace_records_by_mode = step_trace_records_by_mode or {}
    tool_call_records_by_mode = tool_call_records_by_mode or {}
    mode_summaries = tuple(
        aggregate_mode_episode_summaries(
            mode,
            records,
            validation_lane=validation_lane,
            step_trace_records=step_trace_records_by_mode.get(mode, ()),
            tool_call_records=tool_call_records_by_mode.get(mode, ()),
        )
        for mode, records in sorted(records_by_mode.items())
    )
    schedule_names = tuple(
        sorted(
            {
                schedule_name
                for summary in mode_summaries
                for schedule_name in summary.schedule_names
            }
        )
    )
    seed_values = tuple(
        sorted(
            {
                seed_value
                for summary in mode_summaries
                for seed_value in summary.seed_values
            }
        )
    )
    tool_ablation_variants = tuple(
        sorted(
            {
                tool_ablation_variant
                for summary in mode_summaries
                for tool_ablation_variant in summary.tool_ablation_variants
            }
        )
    )
    return BatchAggregateSummary(
        benchmark_id=benchmark_id,
        validation_lane=validation_lane,
        mode_names=tuple(summary.mode for summary in mode_summaries),
        schedule_names=schedule_names,
        seed_values=seed_values,
        tool_ablation_variants=tool_ablation_variants,
        mode_summaries=mode_summaries,
    )


def _aggregate_schedule_records(
    schedule_name: str,
    records: tuple[EpisodeSummaryRecord, ...],
) -> ScheduleAggregateSummary:
    selected = tuple(record for record in records if record.schedule_name == schedule_name)
    return ScheduleAggregateSummary(
        schedule_name=schedule_name,
        run_count=len(selected),
        seed_values=tuple(
            sorted(
                {
                    record.run_seed
                    for record in selected
                    if record.run_seed is not None
                }
            )
        ),
        benchmark_metrics=_aggregate_benchmark_metrics(selected),
        decision_quality=_aggregate_decision_quality(selected),
        telemetry_metrics=_aggregate_telemetry_metrics(selected),
        validity_summary=_aggregate_validity_summary(selected),
        performance_summary=_aggregate_performance_summary(selected),
        tool_use_summary=_aggregate_tool_use_summary(selected),
        external_evidence_summary=_aggregate_external_evidence_summary(selected),
        robustness_summary=_aggregate_robustness_summary(
            records=selected,
            schedule_names=(schedule_name,),
            seed_values=tuple(
                sorted(
                    {
                        record.run_seed
                        for record in selected
                        if record.run_seed is not None
                    }
                )
            ),
        ),
    )


def _aggregate_ablation_records(
    *,
    mode: str,
    tool_ablation_variant: str,
    records: tuple[EpisodeSummaryRecord, ...],
    step_trace_records: tuple[StepTraceRecord, ...],
    tool_call_records: tuple[ToolCallTraceRecord, ...],
    full_records: tuple[EpisodeSummaryRecord, ...],
    full_step_records: tuple[StepTraceRecord, ...],
) -> AblationAggregateSummary:
    selected_records = tuple(
        record
        for record in records
        if record.tool_ablation_variant == tool_ablation_variant
    )
    selected_step_records = tuple(
        record
        for record in step_trace_records
        if record.tool_ablation_variant == tool_ablation_variant
    )
    selected_tool_call_records = tuple(
        record
        for record in tool_call_records
        if record.tool_ablation_variant == tool_ablation_variant
    )
    schedule_names = tuple(
        sorted(
            {
                record.schedule_name
                for record in selected_records
                if record.schedule_name is not None
            }
        )
    )
    seed_values = tuple(
        sorted(
            {
                record.run_seed
                for record in selected_records
                if record.run_seed is not None
            }
        )
    )
    return AblationAggregateSummary(
        tool_ablation_variant=tool_ablation_variant,
        run_count=len(selected_records),
        schedule_names=schedule_names,
        seed_values=seed_values,
        benchmark_metrics=_aggregate_benchmark_metrics(selected_records),
        decision_quality=_aggregate_decision_quality(selected_records),
        telemetry_metrics=_aggregate_telemetry_metrics(selected_records),
        validity_summary=_aggregate_validity_summary(selected_records),
        performance_summary=_aggregate_performance_summary(selected_records),
        tool_use_summary=_aggregate_tool_use_summary(selected_records),
        external_evidence_summary=_aggregate_external_evidence_summary(selected_records),
        robustness_summary=_aggregate_robustness_summary(
            records=selected_records,
            schedule_names=schedule_names,
            seed_values=seed_values,
        ),
        optimizer_order_boundary_preserved=all(
            record.optimizer_order_boundary_preserved for record in selected_records
        ),
        tool_usage_summary=summarize_tool_usage(
            mode=mode,
            tool_ablation_variant=tool_ablation_variant,
            tool_call_records=selected_tool_call_records,
        ),
        tool_ablation_summary=summarize_tool_ablation(
            mode=mode,
            tool_ablation_variant=tool_ablation_variant,
            full_episode_records=full_records,
            ablated_episode_records=selected_records,
            full_step_records=full_step_records,
            ablated_step_records=selected_step_records,
        ),
    )


def _aggregate_benchmark_metrics(
    records: tuple[EpisodeSummaryRecord, ...],
) -> AggregateBenchmarkMetrics:
    return AggregateBenchmarkMetrics(
        average_total_cost=_mean_or_none(tuple(record.total_cost for record in records)),
        average_fill_rate=_mean_or_none(tuple(record.fill_rate for record in records)),
        average_inventory=_mean_or_none(tuple(record.average_inventory for record in records)),
        average_backorder_level=_mean_or_none(
            tuple(record.average_backorder_level for record in records)
        ),
        average_tool_call_count=sum(record.tool_call_count for record in records) / len(records),
    )


def _aggregate_validity_summary(
    records: tuple[EpisodeSummaryRecord, ...],
) -> AggregateValiditySummary:
    return AggregateValiditySummary(
        optimizer_order_boundary_preserved=all(
            record.optimizer_order_boundary_preserved for record in records
        ),
        invalid_output_count=sum(record.invalid_output_count for record in records),
        fallback_count=sum(record.fallback_count for record in records),
        failure_before_response_count=sum(
            record.failure_before_response_count for record in records
        ),
        failure_after_response_count=sum(
            record.failure_after_response_count for record in records
        ),
        rollout_fidelity_gate_passed=all(
            record.rollout_fidelity_gate_passed for record in records
        ),
        operational_metrics_gate_passed=all(
            record.operational_metrics_gate_passed for record in records
        ),
    )


def _aggregate_performance_summary(
    records: tuple[EpisodeSummaryRecord, ...],
) -> AggregatePerformanceSummary:
    return AggregatePerformanceSummary(
        average_total_cost=_mean_or_none(tuple(record.total_cost for record in records)),
        average_fill_rate=_mean_or_none(tuple(record.fill_rate for record in records)),
        average_inventory=_mean_or_none(tuple(record.average_inventory for record in records)),
        average_backorder_level=_mean_or_none(
            tuple(record.average_backorder_level for record in records)
        ),
    )


def _aggregate_decision_quality(
    records: tuple[EpisodeSummaryRecord, ...],
) -> AggregateDecisionQualityMetrics:
    total_steps = sum(record.step_count for record in records)
    predicted_counter: Counter[str] = Counter()
    confusion_counter: Counter[tuple[str, str]] = Counter()
    weighted_accuracy_numerator = 0.0
    weighted_accuracy_denominator = 0
    confidence_weighted_sum = 0.0
    confidence_weight_count = 0
    for record in records:
        predicted_counter.update(dict(record.predicted_regime_counts))
        confusion_counter.update(
            {(true_label, predicted_label): count for true_label, predicted_label, count in record.confusion_counts}
        )
        if record.regime_prediction_accuracy is not None:
            weighted_accuracy_numerator += record.regime_prediction_accuracy * record.step_count
            weighted_accuracy_denominator += record.step_count
        if record.average_confidence is not None:
            confidence_weighted_sum += record.average_confidence * record.step_count
            confidence_weight_count += record.step_count
    regime_prediction_accuracy = None
    if weighted_accuracy_denominator > 0:
        regime_prediction_accuracy = weighted_accuracy_numerator / weighted_accuracy_denominator
    average_confidence = None
    if confidence_weight_count > 0:
        average_confidence = confidence_weighted_sum / confidence_weight_count
    return AggregateDecisionQualityMetrics(
        regime_prediction_accuracy=regime_prediction_accuracy,
        no_action_rate=(
            sum(record.no_action_count for record in records) / total_steps if total_steps else 0.0
        ),
        replan_rate=(
            sum(record.replan_count for record in records) / total_steps if total_steps else 0.0
        ),
        intervention_rate=(
            sum(record.intervention_count for record in records) / total_steps
            if total_steps
            else 0.0
        ),
        missed_intervention_count=sum(record.missed_intervention_count for record in records),
        unnecessary_intervention_count=sum(
            record.unnecessary_intervention_count for record in records
        ),
        average_confidence=average_confidence,
        predicted_regime_counts=tuple(sorted(predicted_counter.items())),
        confusion_counts=tuple(
            sorted(
                (
                    true_label,
                    predicted_label,
                    count,
                )
                for (true_label, predicted_label), count in confusion_counter.items()
            )
        ),
    )


def _aggregate_tool_use_summary(
    records: tuple[EpisodeSummaryRecord, ...],
) -> AggregateToolUseSummary:
    total_steps = sum(record.step_count for record in records)
    return AggregateToolUseSummary(
        average_tool_call_count=sum(record.tool_call_count for record in records) / len(records),
        no_action_rate=(
            sum(record.no_action_count for record in records) / total_steps if total_steps else 0.0
        ),
        replan_rate=(
            sum(record.replan_count for record in records) / total_steps if total_steps else 0.0
        ),
        intervention_rate=(
            sum(record.intervention_count for record in records) / total_steps
            if total_steps
            else 0.0
        ),
        unavailable_tool_request_count=sum(
            record.unavailable_tool_request_count for record in records
        ),
        disabled_tool_fallback_count=sum(
            record.disabled_tool_fallback_count for record in records
        ),
        sequencing_blocked_tool_request_count=sum(
            record.sequencing_blocked_tool_request_count for record in records
        ),
        clean_intervention_count=sum(
            record.clean_intervention_count for record in records
        ),
        clean_optimizer_input_change_count=sum(
            record.clean_optimizer_input_change_count for record in records
        ),
        repeated_stress_moderation_count=sum(
            record.repeated_stress_moderation_count for record in records
        ),
        relapse_moderation_count=sum(
            record.relapse_moderation_count for record in records
        ),
        unresolved_stress_moderation_count=sum(
            record.unresolved_stress_moderation_count for record in records
        ),
        moderated_update_count=sum(
            record.moderated_update_count for record in records
        ),
        hysteresis_application_count=sum(
            record.hysteresis_application_count for record in records
        ),
    )


def _aggregate_telemetry_metrics(
    records: tuple[EpisodeSummaryRecord, ...],
) -> AggregateTelemetryMetrics:
    client_error_counter: Counter[str] = Counter()
    for record in records:
        client_error_counter.update(dict(record.client_error_counts))
    return AggregateTelemetryMetrics(
        average_prompt_tokens=_mean_or_none(tuple(record.total_prompt_tokens for record in records)),
        average_completion_tokens=_mean_or_none(
            tuple(record.total_completion_tokens for record in records)
        ),
        average_total_tokens=_mean_or_none(tuple(record.total_tokens for record in records)),
        average_llm_latency_ms=_mean_or_none(
            tuple(record.total_llm_latency_ms for record in records)
        ),
        average_orchestration_latency_ms=_mean_or_none(
            tuple(record.total_orchestration_latency_ms for record in records)
        ),
        invalid_output_count=sum(record.invalid_output_count for record in records),
        fallback_count=sum(record.fallback_count for record in records),
        client_error_counts=tuple(sorted(client_error_counter.items())),
        total_retry_count=sum(record.total_retry_count for record in records),
        failure_before_response_count=sum(
            record.failure_before_response_count for record in records
        ),
        failure_after_response_count=sum(
            record.failure_after_response_count for record in records
        ),
    )


def _aggregate_external_evidence_summary(
    records: tuple[EpisodeSummaryRecord, ...],
) -> AggregateExternalEvidenceSummary | None:
    external_evidence_source = _single_text_or_mixed(
        tuple(record.external_evidence_source for record in records)
    )
    period_count = sum(record.external_evidence_period_count for record in records)
    tool_call_count = sum(record.external_evidence_tool_call_count for record in records)
    false_alarm_count = sum(record.false_alarm_evidence_count for record in records)
    intervention_count = sum(
        record.evidence_supported_intervention_count for record in records
    )
    optimizer_input_change_count = sum(
        record.external_evidence_changed_optimizer_input_count for record in records
    )
    evidence_fusion_cap_count = sum(record.evidence_fusion_cap_count for record in records)
    capped_external_strengthening_count = sum(
        record.capped_external_strengthening_count for record in records
    )
    early_evidence_confirmation_gate_count = sum(
        record.early_evidence_confirmation_gate_count for record in records
    )
    early_evidence_family_change_block_count = sum(
        record.early_evidence_family_change_block_count for record in records
    )
    corroboration_gate_count = sum(
        record.corroboration_gate_count for record in records
    )
    corroborated_family_change_count = sum(
        record.corroborated_family_change_count for record in records
    )
    role_level_counter: Counter[str] = Counter()
    for record in records:
        role_level_counter.update(dict(record.role_level_usage_counts))
    if (
        period_count == 0
        and tool_call_count == 0
        and false_alarm_count == 0
        and intervention_count == 0
        and optimizer_input_change_count == 0
        and evidence_fusion_cap_count == 0
        and capped_external_strengthening_count == 0
        and early_evidence_confirmation_gate_count == 0
        and early_evidence_family_change_block_count == 0
        and corroboration_gate_count == 0
        and corroborated_family_change_count == 0
        and not role_level_counter
    ):
        return None
    return AggregateExternalEvidenceSummary(
        external_evidence_source=external_evidence_source,
        external_evidence_period_count=period_count,
        external_evidence_tool_call_count=tool_call_count,
        false_alarm_evidence_count=false_alarm_count,
        evidence_supported_intervention_count=intervention_count,
        external_evidence_changed_optimizer_input_count=optimizer_input_change_count,
        evidence_fusion_cap_count=evidence_fusion_cap_count,
        capped_external_strengthening_count=capped_external_strengthening_count,
        early_evidence_confirmation_gate_count=early_evidence_confirmation_gate_count,
        early_evidence_family_change_block_count=early_evidence_family_change_block_count,
        corroboration_gate_count=corroboration_gate_count,
        corroborated_family_change_count=corroborated_family_change_count,
        role_level_usage_counts=tuple(sorted(role_level_counter.items())),
    )


def _aggregate_robustness_summary(
    *,
    records: tuple[EpisodeSummaryRecord, ...],
    schedule_names: tuple[str, ...],
    seed_values: tuple[int, ...],
) -> AggregateRobustnessSummary:
    schedule_run_counts = tuple(
        (
            schedule_name,
            sum(1 for record in records if record.schedule_name == schedule_name),
        )
        for schedule_name in schedule_names
    )
    return AggregateRobustnessSummary(
        run_count=len(records),
        schedule_count=len(schedule_names),
        seed_count=len(seed_values),
        schedule_names=schedule_names,
        seed_values=seed_values,
        schedule_run_counts=schedule_run_counts,
    )


__all__ = [
    "AggregateBenchmarkMetrics",
    "AggregateDecisionQualityMetrics",
    "AggregateExternalEvidenceSummary",
    "AggregatePerformanceSummary",
    "AggregateTelemetryMetrics",
    "AggregateToolUseSummary",
    "AggregateRobustnessSummary",
    "AggregateValiditySummary",
    "AblationAggregateSummary",
    "BatchAggregateSummary",
    "ModeAggregateSummary",
    "ScheduleAggregateSummary",
    "aggregate_batch_episode_summaries",
    "aggregate_mode_episode_summaries",
]
