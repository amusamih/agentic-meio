"""Typed first-pass evaluation summaries for benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass, field

from meio.agents.telemetry import EpisodeTelemetrySummary
from meio.evaluation.decision_quality import DecisionQualitySummary
from meio.evaluation.logging_schema import CostBreakdownRecord, EpisodeSummaryRecord
from meio.simulation.state import EpisodeTrace


def _validate_non_negative_int(value: int, field_name: str) -> None:
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative.")


@dataclass(frozen=True, slots=True)
class InterventionSummary:
    """Compact intervention counts extracted from structured traces."""

    tool_call_count: int = 0
    replan_count: int = 0
    abstain_count: int = 0
    no_action_count: int = 0

    def __post_init__(self) -> None:
        _validate_non_negative_int(self.tool_call_count, "tool_call_count")
        _validate_non_negative_int(self.replan_count, "replan_count")
        _validate_non_negative_int(self.abstain_count, "abstain_count")
        _validate_non_negative_int(self.no_action_count, "no_action_count")


@dataclass(frozen=True, slots=True)
class TraceSummary:
    """Compact trace-shape summary for a benchmark run."""

    episode_count: int
    trace_length: int
    state_count: int
    observation_count: int
    agent_signal_count: int
    optimization_call_count: int

    def __post_init__(self) -> None:
        _validate_non_negative_int(self.episode_count, "episode_count")
        _validate_non_negative_int(self.trace_length, "trace_length")
        _validate_non_negative_int(self.state_count, "state_count")
        _validate_non_negative_int(self.observation_count, "observation_count")
        _validate_non_negative_int(self.agent_signal_count, "agent_signal_count")
        _validate_non_negative_int(self.optimization_call_count, "optimization_call_count")


@dataclass(frozen=True, slots=True)
class BenchmarkRunSummary:
    """First-pass typed benchmark run summary."""

    run_id: str
    benchmark_id: str
    benchmark_source: str
    topology: str
    echelon_count: int
    intervention_summary: InterventionSummary
    trace_summary: TraceSummary
    total_cost: float | None = None
    fill_rate: float | None = None
    average_inventory: float | None = None
    average_backorder_level: float | None = None
    optimizer_order_boundary_preserved: bool = True
    episode_telemetry: EpisodeTelemetrySummary | None = None
    decision_quality: DecisionQualitySummary | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for field_name, value in (
            ("run_id", self.run_id),
            ("benchmark_id", self.benchmark_id),
            ("benchmark_source", self.benchmark_source),
            ("topology", self.topology),
        ):
            if not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string.")
        if self.echelon_count <= 0:
            raise ValueError("echelon_count must be positive.")
        if not isinstance(self.intervention_summary, InterventionSummary):
            raise TypeError("intervention_summary must be an InterventionSummary.")
        if not isinstance(self.trace_summary, TraceSummary):
            raise TypeError("trace_summary must be a TraceSummary.")
        if self.episode_telemetry is not None and not isinstance(
            self.episode_telemetry,
            EpisodeTelemetrySummary,
        ):
            raise TypeError("episode_telemetry must be an EpisodeTelemetrySummary when provided.")
        if self.decision_quality is not None and not isinstance(
            self.decision_quality,
            DecisionQualitySummary,
        ):
            raise TypeError("decision_quality must be a DecisionQualitySummary when provided.")
        object.__setattr__(self, "notes", tuple(self.notes))
        for value in (
            self.total_cost,
            self.fill_rate,
            self.average_inventory,
            self.average_backorder_level,
        ):
            if value is not None and not isinstance(value, (int, float)):
                raise TypeError("numeric summary fields must be numeric when provided.")
        for note in self.notes:
            if not note.strip():
                raise ValueError("notes must contain non-empty strings.")

    def to_record(self) -> dict[str, object]:
        """Return a compact JSON-serializable summary."""

        return {
            "run_id": self.run_id,
            "benchmark_id": self.benchmark_id,
            "benchmark_source": self.benchmark_source,
            "topology": self.topology,
            "echelon_count": self.echelon_count,
            "tool_call_count": self.intervention_summary.tool_call_count,
            "replan_count": self.intervention_summary.replan_count,
            "abstain_count": self.intervention_summary.abstain_count,
            "no_action_count": self.intervention_summary.no_action_count,
            "trace_length": self.trace_summary.trace_length,
            "episode_count": self.trace_summary.episode_count,
            "optimization_call_count": self.trace_summary.optimization_call_count,
            "total_cost": self.total_cost,
            "fill_rate": self.fill_rate,
            "average_inventory": self.average_inventory,
            "average_backorder_level": self.average_backorder_level,
            "optimizer_order_boundary_preserved": self.optimizer_order_boundary_preserved,
            "episode_telemetry": (
                self.episode_telemetry.to_record()
                if self.episode_telemetry is not None
                else None
            ),
            "decision_quality": (
                self.decision_quality.to_record()
                if self.decision_quality is not None
                else None
            ),
            "notes": list(self.notes),
        }


def build_episode_summary_record(
    summary: BenchmarkRunSummary,
    *,
    mode: str,
    validation_lane: str | None = None,
    tool_ablation_variant: str,
    schedule_name: str | None,
    run_seed: int | None,
    regime_schedule: tuple[str, ...],
    cost_breakdown: CostBreakdownRecord,
    intervention_count: int,
    invalid_output_count: int,
    fallback_count: int,
    unavailable_tool_request_count: int = 0,
    disabled_tool_fallback_count: int = 0,
    sequencing_blocked_tool_request_count: int = 0,
    clean_intervention_count: int = 0,
    clean_optimizer_input_change_count: int = 0,
    repeated_stress_moderation_count: int = 0,
    relapse_moderation_count: int = 0,
    unresolved_stress_moderation_count: int = 0,
    moderated_update_count: int = 0,
    hysteresis_application_count: int = 0,
    external_evidence_source: str | None = None,
    external_evidence_period_count: int = 0,
    external_evidence_tool_call_count: int = 0,
    false_alarm_evidence_count: int = 0,
    evidence_supported_intervention_count: int = 0,
    external_evidence_changed_optimizer_input_count: int = 0,
    evidence_fusion_cap_count: int = 0,
    capped_external_strengthening_count: int = 0,
    early_evidence_confirmation_gate_count: int = 0,
    early_evidence_family_change_block_count: int = 0,
    corroboration_gate_count: int = 0,
    corroborated_family_change_count: int = 0,
    role_level_usage_counts: tuple[tuple[str, int], ...] = (),
    rollout_fidelity_gate_passed: bool = False,
    operational_metrics_gate_passed: bool = False,
) -> EpisodeSummaryRecord:
    """Build one episode-level logging record from the current summary object."""

    if summary.episode_telemetry is None:
        total_prompt_tokens = None
        total_completion_tokens = None
        total_tokens = None
        total_llm_latency_ms = None
        total_orchestration_latency_ms = None
        client_error_counts: tuple[tuple[str, int], ...] = ()
        total_retry_count = 0
        failure_before_response_count = 0
        failure_after_response_count = 0
    else:
        llm_call_count = summary.episode_telemetry.llm_call_count
        step_count = summary.episode_telemetry.step_count
        total_prompt_tokens = _scaled_total_from_average(
            summary.episode_telemetry.average_prompt_tokens,
            llm_call_count,
        )
        total_completion_tokens = _scaled_total_from_average(
            summary.episode_telemetry.average_completion_tokens,
            llm_call_count,
        )
        total_tokens = _scaled_total_from_average(
            summary.episode_telemetry.average_total_tokens,
            llm_call_count,
        )
        total_llm_latency_ms = _scaled_total_from_average(
            summary.episode_telemetry.average_llm_latency_ms,
            llm_call_count,
            cast_to_int=False,
        )
        total_orchestration_latency_ms = _scaled_total_from_average(
            summary.episode_telemetry.average_orchestration_latency_ms,
            step_count,
            cast_to_int=False,
        )
        client_error_counts = summary.episode_telemetry.client_error_counts
        total_retry_count = summary.episode_telemetry.total_retry_count
        failure_before_response_count = (
            summary.episode_telemetry.failure_before_response_count
        )
        failure_after_response_count = summary.episode_telemetry.failure_after_response_count
    decision_quality = summary.decision_quality
    return EpisodeSummaryRecord(
        episode_id=summary.run_id,
        mode=mode,
        benchmark_id=summary.benchmark_id,
        validation_lane=validation_lane,
        topology=summary.topology,
        echelon_count=summary.echelon_count,
        tool_ablation_variant=tool_ablation_variant,
        schedule_name=schedule_name,
        run_seed=run_seed,
        regime_schedule=regime_schedule,
        total_cost=summary.total_cost,
        cost_breakdown=cost_breakdown,
        fill_rate=summary.fill_rate,
        average_inventory=summary.average_inventory,
        average_backorder_level=summary.average_backorder_level,
        step_count=summary.trace_summary.trace_length,
        replan_count=summary.intervention_summary.replan_count,
        intervention_count=intervention_count,
        abstain_count=summary.intervention_summary.abstain_count,
        no_action_count=summary.intervention_summary.no_action_count,
        tool_call_count=summary.intervention_summary.tool_call_count,
        invalid_output_count=invalid_output_count,
        fallback_count=fallback_count,
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        total_tokens=total_tokens,
        total_llm_latency_ms=total_llm_latency_ms,
        total_orchestration_latency_ms=total_orchestration_latency_ms,
        optimizer_order_boundary_preserved=summary.optimizer_order_boundary_preserved,
        rollout_fidelity_gate_passed=rollout_fidelity_gate_passed,
        operational_metrics_gate_passed=operational_metrics_gate_passed,
        client_error_counts=client_error_counts,
        total_retry_count=total_retry_count,
        failure_before_response_count=failure_before_response_count,
        failure_after_response_count=failure_after_response_count,
        regime_prediction_accuracy=(
            decision_quality.regime_prediction_accuracy
            if decision_quality is not None
            else None
        ),
        predicted_regime_counts=(
            decision_quality.predicted_regime_counts
            if decision_quality is not None
            else ()
        ),
        confusion_counts=(
            decision_quality.confusion_counts
            if decision_quality is not None
            else ()
        ),
        missed_intervention_count=(
            decision_quality.missed_intervention_count
            if decision_quality is not None
            else 0
        ),
        unnecessary_intervention_count=(
            decision_quality.unnecessary_intervention_count
            if decision_quality is not None
            else 0
        ),
        average_confidence=(
            decision_quality.average_confidence
            if decision_quality is not None
            else None
        ),
        unavailable_tool_request_count=unavailable_tool_request_count,
        disabled_tool_fallback_count=disabled_tool_fallback_count,
        sequencing_blocked_tool_request_count=sequencing_blocked_tool_request_count,
        clean_intervention_count=clean_intervention_count,
        clean_optimizer_input_change_count=clean_optimizer_input_change_count,
        repeated_stress_moderation_count=repeated_stress_moderation_count,
        relapse_moderation_count=relapse_moderation_count,
        unresolved_stress_moderation_count=unresolved_stress_moderation_count,
        moderated_update_count=moderated_update_count,
        hysteresis_application_count=hysteresis_application_count,
        external_evidence_source=external_evidence_source,
        external_evidence_period_count=external_evidence_period_count,
        external_evidence_tool_call_count=external_evidence_tool_call_count,
        false_alarm_evidence_count=false_alarm_evidence_count,
        evidence_supported_intervention_count=evidence_supported_intervention_count,
        external_evidence_changed_optimizer_input_count=(
            external_evidence_changed_optimizer_input_count
        ),
        evidence_fusion_cap_count=evidence_fusion_cap_count,
        capped_external_strengthening_count=capped_external_strengthening_count,
        early_evidence_confirmation_gate_count=early_evidence_confirmation_gate_count,
        early_evidence_family_change_block_count=early_evidence_family_change_block_count,
        corroboration_gate_count=corroboration_gate_count,
        corroborated_family_change_count=corroborated_family_change_count,
        role_level_usage_counts=role_level_usage_counts,
    )


def summarize_interventions(traces: tuple[EpisodeTrace, ...]) -> InterventionSummary:
    """Aggregate intervention counts from typed traces."""

    signals = tuple(signal for trace in traces for signal in trace.agent_signals)
    return InterventionSummary(
        tool_call_count=sum(len(signal.tool_sequence) for signal in signals),
        replan_count=sum(1 for signal in signals if signal.request_replan),
        abstain_count=sum(1 for signal in signals if signal.abstained),
        no_action_count=sum(1 for signal in signals if signal.no_action),
    )


def summarize_traces(traces: tuple[EpisodeTrace, ...]) -> TraceSummary:
    """Aggregate trace-shape counts from typed traces."""

    return TraceSummary(
        episode_count=len(traces),
        trace_length=sum(len(trace.states) for trace in traces),
        state_count=sum(len(trace.states) for trace in traces),
        observation_count=sum(len(trace.observations) for trace in traces),
        agent_signal_count=sum(len(trace.agent_signals) for trace in traces),
        optimization_call_count=sum(len(trace.optimization_results) for trace in traces),
    )


def build_benchmark_run_summary(
    run_id: str,
    benchmark_id: str,
    benchmark_source: str,
    topology: str,
    echelon_count: int,
    traces: tuple[EpisodeTrace, ...],
    optimizer_order_boundary_preserved: bool,
    notes: tuple[str, ...] = (),
) -> BenchmarkRunSummary:
    """Build the first-pass benchmark run summary from typed traces."""

    inventory_points = tuple(
        value
        for trace in traces
        for state in trace.ending_states
        for value in state.inventory_level
    )
    backorder_points = tuple(
        value
        for trace in traces
        for state in trace.ending_states
        for value in state.backorder_level
    )
    average_inventory = (
        sum(inventory_points) / len(inventory_points) if inventory_points else None
    )
    average_backorder_level = (
        sum(backorder_points) / len(backorder_points) if backorder_points else None
    )
    return BenchmarkRunSummary(
        run_id=run_id,
        benchmark_id=benchmark_id,
        benchmark_source=benchmark_source,
        topology=topology,
        echelon_count=echelon_count,
        intervention_summary=summarize_interventions(traces),
        trace_summary=summarize_traces(traces),
        total_cost=None,
        fill_rate=None,
        average_inventory=average_inventory,
        average_backorder_level=average_backorder_level,
        optimizer_order_boundary_preserved=optimizer_order_boundary_preserved,
        notes=notes,
    )


def _scaled_total_from_average(
    average: float | None,
    count: int,
    *,
    cast_to_int: bool = True,
) -> int | float | None:
    if average is None:
        return None
    total = average * count
    if cast_to_int:
        return int(round(total))
    return total


__all__ = [
    "BenchmarkRunSummary",
    "build_episode_summary_record",
    "InterventionSummary",
    "TraceSummary",
    "build_benchmark_run_summary",
    "summarize_interventions",
    "summarize_traces",
]
