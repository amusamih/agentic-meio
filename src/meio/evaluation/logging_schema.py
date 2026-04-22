"""Typed experiment logging schema for MEIO benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

SCHEMA_VERSION = "meio.experiment_log.v16"


def _validate_non_empty_text(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")


def _validate_non_negative_int(value: int, field_name: str) -> None:
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative.")


def _validate_optional_non_negative_int(value: int | None, field_name: str) -> None:
    if value is not None and value < 0:
        raise ValueError(f"{field_name} must be non-negative when provided.")


def _validate_optional_non_negative_float(value: float | None, field_name: str) -> None:
    if value is not None and value < 0.0:
        raise ValueError(f"{field_name} must be non-negative when provided.")


def _validate_string_tuple(values: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    normalized = tuple(values)
    for value in normalized:
        _validate_non_empty_text(value, field_name)
    return normalized


def _validate_string_pair_tuple(
    values: tuple[tuple[str, str], ...],
    field_name: str,
) -> tuple[tuple[str, str], ...]:
    normalized = tuple(values)
    for key, value in normalized:
        _validate_non_empty_text(key, field_name)
        _validate_non_empty_text(value, field_name)
    return normalized


class ArtifactUseClass(str, Enum):
    """Coarse governance class for saved experiment artifacts."""

    INTERNAL_ONLY = "internal_only"
    PAPER_CANDIDATE = "paper_candidate"


@dataclass(frozen=True, slots=True)
class CostBreakdownRecord:
    """Compact episode- or period-level cost breakdown."""

    holding_cost: float
    backlog_cost: float
    ordering_cost: float
    other_cost: float = 0.0

    def __post_init__(self) -> None:
        for field_name in (
            "holding_cost",
            "backlog_cost",
            "ordering_cost",
            "other_cost",
        ):
            _validate_optional_non_negative_float(getattr(self, field_name), field_name)


@dataclass(frozen=True, slots=True)
class ExperimentMetadata:
    """Experiment-level metadata for one logged run group."""

    experiment_id: str
    run_group_id: str
    timestamp: str
    git_commit_sha: str | None
    resolved_config: dict[str, object]
    config_hash: str
    benchmark_id: str
    benchmark_source: str
    mode: str
    provider: str | None
    model_name: str | None
    validation_lane: str | None = None
    external_evidence_source: str | None = None
    tool_ablation_variants: tuple[str, ...] = ()
    artifact_use_class: ArtifactUseClass = ArtifactUseClass.INTERNAL_ONLY
    validity_gate_passed: bool = False
    rollout_fidelity_gate_passed: bool = False
    operational_metrics_gate_passed: bool = False
    eligibility_notes: tuple[str, ...] = ()
    prompt_version: str | None = None
    prompt_hash: str | None = None
    benchmark_random_seed: int | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "experiment_id",
            "run_group_id",
            "timestamp",
            "config_hash",
            "benchmark_id",
            "benchmark_source",
            "mode",
            "schema_version",
        ):
            _validate_non_empty_text(getattr(self, field_name), field_name)
        if self.validation_lane is not None:
            _validate_non_empty_text(self.validation_lane, "validation_lane")
        if self.git_commit_sha is not None:
            _validate_non_empty_text(self.git_commit_sha, "git_commit_sha")
        if self.provider is not None:
            _validate_non_empty_text(self.provider, "provider")
        if self.model_name is not None:
            _validate_non_empty_text(self.model_name, "model_name")
        if self.external_evidence_source is not None:
            _validate_non_empty_text(
                self.external_evidence_source,
                "external_evidence_source",
            )
        if not isinstance(self.artifact_use_class, ArtifactUseClass):
            raise TypeError("artifact_use_class must be an ArtifactUseClass.")
        object.__setattr__(
            self,
            "tool_ablation_variants",
            _validate_string_tuple(
                self.tool_ablation_variants,
                "tool_ablation_variants",
            ),
        )
        object.__setattr__(
            self,
            "eligibility_notes",
            _validate_string_tuple(self.eligibility_notes, "eligibility_notes"),
        )
        if self.prompt_version is not None:
            _validate_non_empty_text(self.prompt_version, "prompt_version")
        if self.prompt_hash is not None:
            _validate_non_empty_text(self.prompt_hash, "prompt_hash")
        if not isinstance(self.resolved_config, dict):
            raise TypeError("resolved_config must be a dictionary.")
        _validate_optional_non_negative_int(
            self.benchmark_random_seed,
            "benchmark_random_seed",
        )


@dataclass(frozen=True, slots=True)
class RunManifestRecord:
    """Compact manifest for one written experiment artifact directory."""

    experiment_id: str
    run_group_id: str
    config_hash: str
    benchmark_id: str
    benchmark_source: str
    benchmark_config_path: str | None
    agent_config_path: str | None
    schedule_names: tuple[str, ...]
    seed_values: tuple[int, ...]
    mode_names: tuple[str, ...]
    tool_ablation_variants: tuple[str, ...]
    provider: str | None
    model_name: str | None
    validation_lane: str | None = None
    external_evidence_source: str | None = None
    artifact_use_class: ArtifactUseClass = ArtifactUseClass.INTERNAL_ONLY
    validity_gate_passed: bool = False
    rollout_fidelity_gate_passed: bool = False
    operational_metrics_gate_passed: bool = False
    eligibility_notes: tuple[str, ...] = ()
    mode_artifact_use_classes: tuple[tuple[str, str], ...] = ()
    prompt_version: str | None = None
    prompt_hash: str | None = None
    artifact_filenames: tuple[tuple[str, str], ...] = ()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "experiment_id",
            "run_group_id",
            "config_hash",
            "benchmark_id",
            "benchmark_source",
            "schema_version",
        ):
            _validate_non_empty_text(getattr(self, field_name), field_name)
        if self.benchmark_config_path is not None:
            _validate_non_empty_text(self.benchmark_config_path, "benchmark_config_path")
        if self.validation_lane is not None:
            _validate_non_empty_text(self.validation_lane, "validation_lane")
        if self.agent_config_path is not None:
            _validate_non_empty_text(self.agent_config_path, "agent_config_path")
        if self.provider is not None:
            _validate_non_empty_text(self.provider, "provider")
        if self.model_name is not None:
            _validate_non_empty_text(self.model_name, "model_name")
        if self.external_evidence_source is not None:
            _validate_non_empty_text(
                self.external_evidence_source,
                "external_evidence_source",
            )
        if not isinstance(self.artifact_use_class, ArtifactUseClass):
            raise TypeError("artifact_use_class must be an ArtifactUseClass.")
        object.__setattr__(
            self,
            "eligibility_notes",
            _validate_string_tuple(self.eligibility_notes, "eligibility_notes"),
        )
        if self.prompt_version is not None:
            _validate_non_empty_text(self.prompt_version, "prompt_version")
        if self.prompt_hash is not None:
            _validate_non_empty_text(self.prompt_hash, "prompt_hash")
        object.__setattr__(
            self,
            "schedule_names",
            _validate_string_tuple(self.schedule_names, "schedule_names"),
        )
        object.__setattr__(self, "seed_values", tuple(self.seed_values))
        for value in self.seed_values:
            _validate_non_negative_int(value, "seed_values")
        object.__setattr__(
            self,
            "mode_names",
            _validate_string_tuple(self.mode_names, "mode_names"),
        )
        object.__setattr__(
            self,
            "tool_ablation_variants",
            _validate_string_tuple(
                self.tool_ablation_variants,
                "tool_ablation_variants",
            ),
        )
        object.__setattr__(
            self,
            "mode_artifact_use_classes",
            _validate_string_pair_tuple(
                self.mode_artifact_use_classes,
                "mode_artifact_use_classes",
            ),
        )
        object.__setattr__(
            self,
            "artifact_filenames",
            _validate_string_pair_tuple(self.artifact_filenames, "artifact_filenames"),
        )


@dataclass(frozen=True, slots=True)
class EpisodeSummaryRecord:
    """Episode-level aggregate record for later analysis."""

    episode_id: str
    mode: str
    benchmark_id: str
    topology: str
    echelon_count: int
    tool_ablation_variant: str
    schedule_name: str | None
    run_seed: int | None
    regime_schedule: tuple[str, ...]
    total_cost: float | None
    cost_breakdown: CostBreakdownRecord
    fill_rate: float | None
    average_inventory: float | None
    average_backorder_level: float | None
    step_count: int
    replan_count: int
    intervention_count: int
    abstain_count: int
    no_action_count: int
    tool_call_count: int
    invalid_output_count: int
    fallback_count: int
    total_prompt_tokens: int | None
    total_completion_tokens: int | None
    total_tokens: int | None
    total_llm_latency_ms: float | None
    total_orchestration_latency_ms: float | None
    optimizer_order_boundary_preserved: bool
    rollout_fidelity_gate_passed: bool = False
    operational_metrics_gate_passed: bool = False
    client_error_counts: tuple[tuple[str, int], ...] = ()
    total_retry_count: int = 0
    failure_before_response_count: int = 0
    failure_after_response_count: int = 0
    regime_prediction_accuracy: float | None = None
    predicted_regime_counts: tuple[tuple[str, int], ...] = ()
    confusion_counts: tuple[tuple[str, str, int], ...] = ()
    missed_intervention_count: int = 0
    unnecessary_intervention_count: int = 0
    average_confidence: float | None = None
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
    external_evidence_source: str | None = None
    external_evidence_period_count: int = 0
    external_evidence_tool_call_count: int = 0
    false_alarm_evidence_count: int = 0
    evidence_supported_intervention_count: int = 0
    external_evidence_changed_optimizer_input_count: int = 0
    evidence_fusion_cap_count: int = 0
    capped_external_strengthening_count: int = 0
    early_evidence_confirmation_gate_count: int = 0
    early_evidence_family_change_block_count: int = 0
    corroboration_gate_count: int = 0
    corroborated_family_change_count: int = 0
    role_level_usage_counts: tuple[tuple[str, int], ...] = ()
    validation_lane: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "episode_id",
            "mode",
            "benchmark_id",
            "topology",
            "tool_ablation_variant",
        ):
            _validate_non_empty_text(getattr(self, field_name), field_name)
        if self.schedule_name is not None:
            _validate_non_empty_text(self.schedule_name, "schedule_name")
        if self.validation_lane is not None:
            _validate_non_empty_text(self.validation_lane, "validation_lane")
        if self.external_evidence_source is not None:
            _validate_non_empty_text(
                self.external_evidence_source,
                "external_evidence_source",
            )
        _validate_optional_non_negative_int(self.run_seed, "run_seed")
        _validate_non_negative_int(self.echelon_count, "echelon_count")
        if self.echelon_count == 0:
            raise ValueError("echelon_count must be positive.")
        object.__setattr__(
            self,
            "regime_schedule",
            _validate_string_tuple(self.regime_schedule, "regime_schedule"),
        )
        if not isinstance(self.cost_breakdown, CostBreakdownRecord):
            raise TypeError("cost_breakdown must be a CostBreakdownRecord.")
        for field_name in (
            "step_count",
            "replan_count",
            "intervention_count",
            "abstain_count",
            "no_action_count",
            "tool_call_count",
            "invalid_output_count",
            "fallback_count",
            "total_retry_count",
            "failure_before_response_count",
            "failure_after_response_count",
            "missed_intervention_count",
            "unnecessary_intervention_count",
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
            _validate_non_negative_int(getattr(self, field_name), field_name)
        object.__setattr__(self, "role_level_usage_counts", tuple(self.role_level_usage_counts))
        for role_label, count in self.role_level_usage_counts:
            _validate_non_empty_text(role_label, "role_level_usage_counts")
            _validate_non_negative_int(count, "role_level_usage_counts")
        for field_name in (
            "total_cost",
            "fill_rate",
            "average_inventory",
            "average_backorder_level",
            "total_llm_latency_ms",
            "total_orchestration_latency_ms",
        ):
            _validate_optional_non_negative_float(getattr(self, field_name), field_name)
        for field_name in ("regime_prediction_accuracy", "average_confidence"):
            value = getattr(self, field_name)
            _validate_optional_non_negative_float(value, field_name)
            if value is not None and value > 1.0:
                raise ValueError(f"{field_name} must be within [0.0, 1.0] when provided.")
        for field_name in (
            "total_prompt_tokens",
            "total_completion_tokens",
            "total_tokens",
        ):
            _validate_optional_non_negative_int(getattr(self, field_name), field_name)
        object.__setattr__(
            self,
            "predicted_regime_counts",
            tuple(self.predicted_regime_counts),
        )
        object.__setattr__(self, "confusion_counts", tuple(self.confusion_counts))
        object.__setattr__(self, "client_error_counts", tuple(self.client_error_counts))
        for category, count in self.client_error_counts:
            _validate_non_empty_text(category, "client_error_counts")
            _validate_non_negative_int(count, "client_error_counts")
        for label, count in self.predicted_regime_counts:
            _validate_non_empty_text(label, "predicted_regime_counts")
            _validate_non_negative_int(count, "predicted_regime_counts")
        for true_label, predicted_label, count in self.confusion_counts:
            _validate_non_empty_text(true_label, "confusion_counts")
            _validate_non_empty_text(predicted_label, "confusion_counts")
            _validate_non_negative_int(count, "confusion_counts")


@dataclass(frozen=True, slots=True)
class StepTraceRecord:
    """Period-level trace row for benchmark rollout analysis."""

    episode_id: str
    mode: str
    tool_ablation_variant: str
    schedule_name: str | None
    run_seed: int | None
    period_index: int
    true_regime_label: str
    predicted_regime_label: str | None
    confidence: float | None
    selected_subgoal: str
    selected_tools: tuple[str, ...]
    update_requests: tuple[str, ...]
    request_replan: bool
    abstain_or_no_action: bool
    demand_outlook: float | None
    leadtime_outlook: float | None
    scenario_adjustment_summary: dict[str, float] | None
    optimizer_orders: tuple[float, ...]
    inventory_by_echelon: tuple[float, ...]
    pipeline_by_echelon: tuple[float, ...]
    backorders_by_echelon: tuple[float, ...]
    per_period_cost: float | None
    per_period_fill_rate: float | None
    decision_changed_optimizer_input: bool
    optimizer_output_changed_state: bool
    intervention_changed_outcome: bool
    unavailable_tool_request: bool = False
    disabled_tool_fallback: bool = False
    sequencing_blocked_tool_request_count: int = 0
    clean_intervention: bool = False
    proposed_update_requests: tuple[str, ...] = ()
    proposed_update_strength: str | None = None
    final_update_strength: str | None = None
    calibration_applied: bool = False
    hysteresis_applied: bool = False
    repeated_stress_moderation_applied: bool = False
    relapse_moderation_applied: bool = False
    unresolved_stress_moderation_applied: bool = False
    calibration_reason: str | None = None
    moderation_reason: str | None = None
    external_evidence_source: str | None = None
    external_evidence_present: bool = False
    external_evidence_false_alarm: bool = False
    external_evidence_tool_used: bool = False
    external_evidence_supported_intervention: bool = False
    external_evidence_role_labels: tuple[str, ...] = ()
    external_evidence_corroboration_count: int = 0
    external_evidence_changed_optimizer_input: bool | None = None
    external_evidence_fusion_cap_applied: bool = False
    corroboration_gate_applied: bool = False
    corroborated_family_change_allowed: bool = False
    proposed_external_strengthening: dict[str, float] | None = None
    final_external_strengthening: dict[str, float] | None = None
    external_evidence_fusion_cap_reason: str | None = None
    corroboration_gate_reason: str | None = None
    early_evidence_confirmation_gate_applied: bool = False
    early_evidence_family_change_blocked: bool = False
    proposed_external_update_requests: tuple[str, ...] = ()
    final_external_update_requests: tuple[str, ...] = ()
    early_evidence_confirmation_gate_reason: str | None = None
    validation_lane: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "episode_id",
            "mode",
            "tool_ablation_variant",
            "true_regime_label",
            "selected_subgoal",
        ):
            _validate_non_empty_text(getattr(self, field_name), field_name)
        if self.schedule_name is not None:
            _validate_non_empty_text(self.schedule_name, "schedule_name")
        if self.validation_lane is not None:
            _validate_non_empty_text(self.validation_lane, "validation_lane")
        _validate_optional_non_negative_int(self.run_seed, "run_seed")
        if self.predicted_regime_label is not None:
            _validate_non_empty_text(self.predicted_regime_label, "predicted_regime_label")
        if self.external_evidence_source is not None:
            _validate_non_empty_text(
                self.external_evidence_source,
                "external_evidence_source",
            )
        _validate_non_negative_int(self.period_index, "period_index")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be within [0.0, 1.0] when provided.")
        object.__setattr__(
            self,
            "selected_tools",
            _validate_string_tuple(self.selected_tools, "selected_tools"),
        )
        object.__setattr__(
            self,
            "proposed_update_requests",
            _validate_string_tuple(
                self.proposed_update_requests,
                "proposed_update_requests",
            ),
        )
        object.__setattr__(
            self,
            "proposed_external_update_requests",
            _validate_string_tuple(
                self.proposed_external_update_requests,
                "proposed_external_update_requests",
            ),
        )
        object.__setattr__(
            self,
            "update_requests",
            _validate_string_tuple(self.update_requests, "update_requests"),
        )
        object.__setattr__(
            self,
            "final_external_update_requests",
            _validate_string_tuple(
                self.final_external_update_requests,
                "final_external_update_requests",
            ),
        )
        object.__setattr__(
            self,
            "external_evidence_role_labels",
            _validate_string_tuple(
                self.external_evidence_role_labels,
                "external_evidence_role_labels",
            ),
        )
        if self.moderation_reason is not None:
            _validate_non_empty_text(self.moderation_reason, "moderation_reason")
        if self.calibration_reason is not None:
            _validate_non_empty_text(self.calibration_reason, "calibration_reason")
        if self.external_evidence_fusion_cap_reason is not None:
            _validate_non_empty_text(
                self.external_evidence_fusion_cap_reason,
                "external_evidence_fusion_cap_reason",
            )
        if self.corroboration_gate_reason is not None:
            _validate_non_empty_text(
                self.corroboration_gate_reason,
                "corroboration_gate_reason",
            )
        if self.early_evidence_confirmation_gate_reason is not None:
            _validate_non_empty_text(
                self.early_evidence_confirmation_gate_reason,
                "early_evidence_confirmation_gate_reason",
            )
        if self.proposed_update_strength is not None:
            _validate_non_empty_text(
                self.proposed_update_strength,
                "proposed_update_strength",
            )
        if self.final_update_strength is not None:
            _validate_non_empty_text(
                self.final_update_strength,
                "final_update_strength",
            )
        for field_name in (
            "demand_outlook",
            "leadtime_outlook",
            "per_period_cost",
            "per_period_fill_rate",
        ):
            _validate_optional_non_negative_float(getattr(self, field_name), field_name)
        _validate_non_negative_int(
            self.sequencing_blocked_tool_request_count,
            "sequencing_blocked_tool_request_count",
        )
        _validate_non_negative_int(
            self.external_evidence_corroboration_count,
            "external_evidence_corroboration_count",
        )
        for field_name in (
            "optimizer_orders",
            "inventory_by_echelon",
            "pipeline_by_echelon",
            "backorders_by_echelon",
        ):
            values = tuple(getattr(self, field_name))
            object.__setattr__(self, field_name, values)
            for value in values:
                _validate_optional_non_negative_float(float(value), field_name)
        if self.scenario_adjustment_summary is not None:
            object.__setattr__(
                self,
                "scenario_adjustment_summary",
                dict(self.scenario_adjustment_summary),
            )
        for field_name in (
            "proposed_external_strengthening",
            "final_external_strengthening",
        ):
            payload = getattr(self, field_name)
            if payload is not None:
                normalized_payload = dict(payload)
                object.__setattr__(self, field_name, normalized_payload)
                for key, value in normalized_payload.items():
                    _validate_non_empty_text(str(key), field_name)
                    _validate_optional_non_negative_float(float(value), field_name)
        if self.external_evidence_changed_optimizer_input is not None and not isinstance(
            self.external_evidence_changed_optimizer_input,
            bool,
        ):
            raise TypeError(
                "external_evidence_changed_optimizer_input must be a bool when provided."
            )


@dataclass(frozen=True, slots=True)
class LLMCallTraceRecord:
    """Per-call LLM trace row."""

    episode_id: str
    mode: str
    tool_ablation_variant: str
    schedule_name: str | None
    run_seed: int | None
    period_index: int
    call_index: int
    provider: str
    model_name: str
    prompt_version: str | None
    prompt_text: str
    prompt_hash: str
    raw_output_text: str | None
    parsed_output: dict[str, object] | None
    validation_success: bool
    invalid_output: bool
    fallback_used: bool
    fallback_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    latency_ms: float | None
    requested_tool_ids: tuple[str, ...] = ()
    unavailable_tool_ids: tuple[str, ...] = ()
    violated_available_tool_set: bool = False
    proposed_update_strength: str | None = None
    final_update_strength: str | None = None
    calibration_applied: bool = False
    hysteresis_applied: bool = False
    final_update_request_types: tuple[str, ...] = ()
    repeated_stress_moderation_applied: bool = False
    relapse_moderation_applied: bool = False
    unresolved_stress_moderation_applied: bool = False
    calibration_reason: str | None = None
    moderation_reason: str | None = None
    retry_count: int = 0
    client_error_category: str | None = None
    client_error_message: str | None = None
    failure_after_response: bool | None = None
    external_evidence_present: bool = False
    external_evidence_tool_available: bool = False
    validation_lane: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "episode_id",
            "mode",
            "provider",
            "model_name",
            "prompt_text",
            "prompt_hash",
            "tool_ablation_variant",
        ):
            _validate_non_empty_text(getattr(self, field_name), field_name)
        if self.schedule_name is not None:
            _validate_non_empty_text(self.schedule_name, "schedule_name")
        if self.validation_lane is not None:
            _validate_non_empty_text(self.validation_lane, "validation_lane")
        _validate_optional_non_negative_int(self.run_seed, "run_seed")
        if self.prompt_version is not None:
            _validate_non_empty_text(self.prompt_version, "prompt_version")
        _validate_non_negative_int(self.period_index, "period_index")
        _validate_non_negative_int(self.call_index, "call_index")
        if self.raw_output_text is not None:
            _validate_non_empty_text(self.raw_output_text, "raw_output_text")
        if self.fallback_reason is not None:
            _validate_non_empty_text(self.fallback_reason, "fallback_reason")
        object.__setattr__(
            self,
            "requested_tool_ids",
            _validate_string_tuple(self.requested_tool_ids, "requested_tool_ids"),
        )
        object.__setattr__(
            self,
            "unavailable_tool_ids",
            _validate_string_tuple(self.unavailable_tool_ids, "unavailable_tool_ids"),
        )
        object.__setattr__(
            self,
            "final_update_request_types",
            _validate_string_tuple(
                self.final_update_request_types,
                "final_update_request_types",
            ),
        )
        if self.proposed_update_strength is not None:
            _validate_non_empty_text(
                self.proposed_update_strength,
                "proposed_update_strength",
            )
        if self.final_update_strength is not None:
            _validate_non_empty_text(
                self.final_update_strength,
                "final_update_strength",
            )
        if self.calibration_reason is not None:
            _validate_non_empty_text(self.calibration_reason, "calibration_reason")
        if self.moderation_reason is not None:
            _validate_non_empty_text(self.moderation_reason, "moderation_reason")
        if self.client_error_category is not None:
            _validate_non_empty_text(self.client_error_category, "client_error_category")
        if self.client_error_message is not None:
            _validate_non_empty_text(self.client_error_message, "client_error_message")
        for field_name in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        ):
            _validate_optional_non_negative_int(getattr(self, field_name), field_name)
        _validate_optional_non_negative_float(self.latency_ms, "latency_ms")
        _validate_non_negative_int(self.retry_count, "retry_count")
        if self.failure_after_response is not None and not isinstance(
            self.failure_after_response,
            bool,
        ):
            raise TypeError("failure_after_response must be a bool when provided.")
        if self.parsed_output is not None:
            object.__setattr__(self, "parsed_output", dict(self.parsed_output))


@dataclass(frozen=True, slots=True)
class ToolCallTraceRecord:
    """Per-call bounded tool trace row."""

    episode_id: str
    mode: str
    tool_ablation_variant: str
    schedule_name: str | None
    run_seed: int | None
    period_index: int
    call_index: int
    tool_id: str
    tool_input: dict[str, object]
    tool_output: dict[str, object] | None
    success: bool
    error_type: str | None
    latency_ms: float | None
    pre_tool_decision: dict[str, object] | None = None
    post_tool_decision: dict[str, object] | None = None
    pre_tool_optimizer_input: dict[str, float] | None = None
    post_tool_optimizer_input: dict[str, float] | None = None
    decision_changed: bool | None = None
    optimizer_input_changed: bool | None = None
    validation_lane: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("episode_id", "mode", "tool_ablation_variant", "tool_id"):
            _validate_non_empty_text(getattr(self, field_name), field_name)
        if self.schedule_name is not None:
            _validate_non_empty_text(self.schedule_name, "schedule_name")
        if self.validation_lane is not None:
            _validate_non_empty_text(self.validation_lane, "validation_lane")
        _validate_optional_non_negative_int(self.run_seed, "run_seed")
        _validate_non_negative_int(self.period_index, "period_index")
        _validate_non_negative_int(self.call_index, "call_index")
        object.__setattr__(self, "tool_input", dict(self.tool_input))
        if self.tool_output is not None:
            object.__setattr__(self, "tool_output", dict(self.tool_output))
        if self.pre_tool_decision is not None:
            object.__setattr__(self, "pre_tool_decision", dict(self.pre_tool_decision))
        if self.post_tool_decision is not None:
            object.__setattr__(self, "post_tool_decision", dict(self.post_tool_decision))
        if self.pre_tool_optimizer_input is not None:
            object.__setattr__(
                self,
                "pre_tool_optimizer_input",
                dict(self.pre_tool_optimizer_input),
            )
        if self.post_tool_optimizer_input is not None:
            object.__setattr__(
                self,
                "post_tool_optimizer_input",
                dict(self.post_tool_optimizer_input),
            )
        if self.error_type is not None:
            _validate_non_empty_text(self.error_type, "error_type")
        _validate_optional_non_negative_float(self.latency_ms, "latency_ms")


__all__ = [
    "ArtifactUseClass",
    "CostBreakdownRecord",
    "EpisodeSummaryRecord",
    "ExperimentMetadata",
    "LLMCallTraceRecord",
    "RunManifestRecord",
    "SCHEMA_VERSION",
    "StepTraceRecord",
    "ToolCallTraceRecord",
]
