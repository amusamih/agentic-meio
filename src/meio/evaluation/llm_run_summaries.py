"""Typed aggregate summaries for bounded LLM comparison runs."""

from __future__ import annotations

from dataclasses import dataclass

from meio.evaluation.decision_quality import DecisionQualitySummary


@dataclass(frozen=True, slots=True)
class LLMRunDiagnostics:
    """Aggregate LLM validity and fallback diagnostics across repeated runs."""

    model_name: str
    provider: str
    run_count: int
    invalid_output_count: int
    fallback_count: int
    successful_response_count: int
    optimizer_order_boundary_preserved: bool
    prompt_version: str | None = None
    prompt_hash: str | None = None
    validation_failure_counts: tuple[tuple[str, int], ...] = ()
    client_error_counts: tuple[tuple[str, int], ...] = ()
    average_prompt_tokens: float | None = None
    average_completion_tokens: float | None = None
    average_total_tokens: float | None = None
    average_llm_latency_ms: float | None = None
    average_orchestration_latency_ms: float | None = None
    total_retry_count: int = 0
    failure_before_response_count: int = 0
    failure_after_response_count: int = 0

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name must be a non-empty string.")
        if not self.provider.strip():
            raise ValueError("provider must be a non-empty string.")
        if self.prompt_version is not None and not self.prompt_version.strip():
            raise ValueError("prompt_version must be non-empty when provided.")
        if self.prompt_hash is not None and not self.prompt_hash.strip():
            raise ValueError("prompt_hash must be non-empty when provided.")
        for field_name in (
            "run_count",
            "invalid_output_count",
            "fallback_count",
            "successful_response_count",
            "total_retry_count",
            "failure_before_response_count",
            "failure_after_response_count",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative.")
        for field_name in (
            "average_prompt_tokens",
            "average_completion_tokens",
            "average_total_tokens",
            "average_llm_latency_ms",
            "average_orchestration_latency_ms",
        ):
            value = getattr(self, field_name)
            if value is not None and value < 0.0:
                raise ValueError(f"{field_name} must be non-negative when provided.")
        for reason, count in self.validation_failure_counts:
            if not reason.strip():
                raise ValueError("validation_failure_counts must use non-empty reasons.")
            if count < 0:
                raise ValueError("validation_failure_counts must use non-negative counts.")
        for reason, count in self.client_error_counts:
            if not reason.strip():
                raise ValueError("client_error_counts must use non-empty reasons.")
            if count < 0:
                raise ValueError("client_error_counts must use non-negative counts.")


@dataclass(frozen=True, slots=True)
class ComparisonRunSummary:
    """Aggregate summary for repeated runs in one comparison mode."""

    mode: str
    run_count: int
    average_tool_call_count: float
    average_replan_count: float
    average_abstain_count: float
    average_no_action_count: float
    average_total_cost: float | None
    average_fill_rate: float | None
    optimizer_order_boundary_preserved: bool
    llm_diagnostics: LLMRunDiagnostics | None = None
    decision_quality: DecisionQualitySummary | None = None

    def __post_init__(self) -> None:
        if not self.mode.strip():
            raise ValueError("mode must be a non-empty string.")
        if self.run_count <= 0:
            raise ValueError("run_count must be positive.")
        for field_name in (
            "average_tool_call_count",
            "average_replan_count",
            "average_abstain_count",
            "average_no_action_count",
        ):
            if getattr(self, field_name) < 0.0:
                raise ValueError(f"{field_name} must be non-negative.")
        for field_name in ("average_total_cost", "average_fill_rate"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be numeric when provided.")
        if self.decision_quality is not None and not isinstance(
            self.decision_quality,
            DecisionQualitySummary,
        ):
            raise TypeError("decision_quality must be a DecisionQualitySummary when provided.")
