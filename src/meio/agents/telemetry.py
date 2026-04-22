"""Typed telemetry helpers for bounded orchestration runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


def _validate_optional_non_negative_int(value: int | None, field_name: str) -> None:
    if value is not None and value < 0:
        raise ValueError(f"{field_name} must be non-negative when provided.")


def _validate_non_negative_int(value: int, field_name: str) -> None:
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative.")


def _validate_optional_non_negative_float(value: float | None, field_name: str) -> None:
    if value is not None and value < 0.0:
        raise ValueError(f"{field_name} must be non-negative when provided.")


def _mean_or_none(values: tuple[float | int | None, ...]) -> float | None:
    numeric = tuple(float(value) for value in values if value is not None)
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def _sum_or_none(values: tuple[float | None, ...]) -> float | None:
    numeric = tuple(value for value in values if value is not None)
    if not numeric:
        return None
    return float(sum(numeric))


def _validate_optional_text(value: str | None, field_name: str) -> None:
    if value is not None and not value.strip():
        raise ValueError(f"{field_name} must be non-empty when provided.")


class ClientErrorCategory(StrEnum):
    """Bounded categories for runtime client failures."""

    TIMEOUT_ERROR = "timeout_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    NETWORK_ERROR = "network_error"
    API_ERROR = "api_error"
    RESPONSE_PARSE_ERROR = "response_parse_error"
    UNKNOWN_CLIENT_ERROR = "unknown_client_error"


_CLIENT_ERROR_CATEGORY_VALUES = {item.value for item in ClientErrorCategory}


@dataclass(frozen=True, slots=True)
class LLMCallTelemetry:
    """Compact telemetry for one bounded LLM completion call."""

    provider: str
    model_name: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    llm_latency_ms: float | None = None
    estimated_cost: float | None = None
    client_error_category: ClientErrorCategory | None = None
    client_error_message: str | None = None
    retry_count: int = 0
    failure_after_response: bool | None = None

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("provider must be a non-empty string.")
        if not self.model_name.strip():
            raise ValueError("model_name must be a non-empty string.")
        _validate_optional_non_negative_int(self.prompt_tokens, "prompt_tokens")
        _validate_optional_non_negative_int(self.completion_tokens, "completion_tokens")
        _validate_optional_non_negative_int(self.total_tokens, "total_tokens")
        _validate_optional_non_negative_float(self.llm_latency_ms, "llm_latency_ms")
        _validate_optional_non_negative_float(self.estimated_cost, "estimated_cost")
        if self.client_error_category is not None and not isinstance(
            self.client_error_category,
            ClientErrorCategory,
        ):
            raise TypeError(
                "client_error_category must be a ClientErrorCategory when provided."
            )
        _validate_optional_text(self.client_error_message, "client_error_message")
        if self.retry_count < 0:
            raise ValueError("retry_count must be non-negative.")
        if self.failure_after_response is not None and not isinstance(
            self.failure_after_response,
            bool,
        ):
            raise TypeError("failure_after_response must be a bool when provided.")


@dataclass(frozen=True, slots=True)
class LLMCallTrace:
    """Detailed bounded trace information for one LLM call."""

    provider: str
    model_name: str
    prompt_text: str
    prompt_hash: str
    prompt_version: str | None = None
    raw_output_text: str | None = None
    parsed_output: dict[str, object] | None = None
    validation_success: bool = False
    invalid_output: bool = False
    fallback_used: bool = False
    fallback_reason: str | None = None
    requested_tool_ids: tuple[str, ...] = field(default_factory=tuple)
    unavailable_tool_ids: tuple[str, ...] = field(default_factory=tuple)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: float | None = None
    retry_count: int = 0
    client_error_category: ClientErrorCategory | None = None
    client_error_message: str | None = None
    failure_after_response: bool | None = None

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("provider must be a non-empty string.")
        if not self.model_name.strip():
            raise ValueError("model_name must be a non-empty string.")
        _validate_optional_text(self.prompt_version, "prompt_version")
        if not self.prompt_text.strip():
            raise ValueError("prompt_text must be a non-empty string.")
        if not self.prompt_hash.strip():
            raise ValueError("prompt_hash must be a non-empty string.")
        _validate_optional_text(self.raw_output_text, "raw_output_text")
        _validate_optional_text(self.fallback_reason, "fallback_reason")
        object.__setattr__(self, "requested_tool_ids", tuple(self.requested_tool_ids))
        object.__setattr__(self, "unavailable_tool_ids", tuple(self.unavailable_tool_ids))
        for tool_id in self.requested_tool_ids:
            _validate_optional_text(tool_id, "requested_tool_ids")
        for tool_id in self.unavailable_tool_ids:
            _validate_optional_text(tool_id, "unavailable_tool_ids")
        _validate_optional_non_negative_int(self.prompt_tokens, "prompt_tokens")
        _validate_optional_non_negative_int(self.completion_tokens, "completion_tokens")
        _validate_optional_non_negative_int(self.total_tokens, "total_tokens")
        _validate_optional_non_negative_float(self.latency_ms, "latency_ms")
        if self.retry_count < 0:
            raise ValueError("retry_count must be non-negative.")
        if self.client_error_category is not None and not isinstance(
            self.client_error_category,
            ClientErrorCategory,
        ):
            raise TypeError(
                "client_error_category must be a ClientErrorCategory when provided."
            )
        _validate_optional_text(self.client_error_message, "client_error_message")
        if self.failure_after_response is not None and not isinstance(
            self.failure_after_response,
            bool,
        ):
            raise TypeError("failure_after_response must be a bool when provided.")
        if self.parsed_output is not None:
            object.__setattr__(self, "parsed_output", dict(self.parsed_output))


@dataclass(frozen=True, slots=True)
class ToolCallTrace:
    """Detailed bounded trace information for one tool invocation."""

    tool_id: str
    tool_input: dict[str, object]
    tool_output: dict[str, object] | None
    success: bool
    pre_tool_decision: dict[str, object] | None = None
    post_tool_decision: dict[str, object] | None = None
    pre_tool_optimizer_input: dict[str, float] | None = None
    post_tool_optimizer_input: dict[str, float] | None = None
    decision_changed: bool | None = None
    optimizer_input_changed: bool | None = None
    error_type: str | None = None
    latency_ms: float | None = None

    def __post_init__(self) -> None:
        if not self.tool_id.strip():
            raise ValueError("tool_id must be a non-empty string.")
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
        _validate_optional_text(self.error_type, "error_type")
        _validate_optional_non_negative_float(self.latency_ms, "latency_ms")


@dataclass(frozen=True, slots=True)
class OrchestrationStepTelemetry:
    """Compact telemetry for one bounded orchestration step."""

    provider: str | None = None
    model_name: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    llm_latency_ms: float | None = None
    orchestration_latency_ms: float | None = None
    invalid_output: bool = False
    fallback_used: bool = False
    tool_call_count: int = 0
    selected_tools: tuple[str, ...] = field(default_factory=tuple)
    request_replan: bool = False
    abstain_or_no_action: bool = False
    estimated_cost: float | None = None
    client_error_category: ClientErrorCategory | None = None
    client_error_message: str | None = None
    retry_count: int = 0
    failure_after_response: bool | None = None

    def __post_init__(self) -> None:
        if self.provider is not None and not self.provider.strip():
            raise ValueError("provider must be non-empty when provided.")
        if self.model_name is not None and not self.model_name.strip():
            raise ValueError("model_name must be non-empty when provided.")
        _validate_optional_non_negative_int(self.prompt_tokens, "prompt_tokens")
        _validate_optional_non_negative_int(self.completion_tokens, "completion_tokens")
        _validate_optional_non_negative_int(self.total_tokens, "total_tokens")
        _validate_optional_non_negative_float(self.llm_latency_ms, "llm_latency_ms")
        _validate_optional_non_negative_float(
            self.orchestration_latency_ms,
            "orchestration_latency_ms",
        )
        _validate_optional_non_negative_float(self.estimated_cost, "estimated_cost")
        if self.tool_call_count < 0:
            raise ValueError("tool_call_count must be non-negative.")
        if self.client_error_category is not None and not isinstance(
            self.client_error_category,
            ClientErrorCategory,
        ):
            raise TypeError(
                "client_error_category must be a ClientErrorCategory when provided."
            )
        _validate_optional_text(self.client_error_message, "client_error_message")
        if self.retry_count < 0:
            raise ValueError("retry_count must be non-negative.")
        if self.failure_after_response is not None and not isinstance(
            self.failure_after_response,
            bool,
        ):
            raise TypeError("failure_after_response must be a bool when provided.")
        object.__setattr__(self, "selected_tools", tuple(self.selected_tools))
        for tool_id in self.selected_tools:
            if not tool_id.strip():
                raise ValueError("selected_tools must contain non-empty tool identifiers.")


@dataclass(frozen=True, slots=True)
class EpisodeTelemetrySummary:
    """Aggregate telemetry across one benchmark run or episode."""

    provider: str | None = None
    model_name: str | None = None
    step_count: int = 0
    llm_call_count: int = 0
    average_prompt_tokens: float | None = None
    average_completion_tokens: float | None = None
    average_total_tokens: float | None = None
    average_llm_latency_ms: float | None = None
    average_orchestration_latency_ms: float | None = None
    invalid_output_count: int = 0
    fallback_count: int = 0
    successful_response_count: int = 0
    total_tool_call_count: int = 0
    total_replan_count: int = 0
    abstain_or_no_action_count: int = 0
    total_estimated_cost: float | None = None
    client_error_counts: tuple[tuple[str, int], ...] = ()
    total_retry_count: int = 0
    failure_before_response_count: int = 0
    failure_after_response_count: int = 0

    def __post_init__(self) -> None:
        if self.provider is not None and not self.provider.strip():
            raise ValueError("provider must be non-empty when provided.")
        if self.model_name is not None and not self.model_name.strip():
            raise ValueError("model_name must be non-empty when provided.")
        for field_name in (
            "step_count",
            "llm_call_count",
            "invalid_output_count",
            "fallback_count",
            "successful_response_count",
            "total_tool_call_count",
            "total_replan_count",
            "abstain_or_no_action_count",
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
            "total_estimated_cost",
        ):
            _validate_optional_non_negative_float(getattr(self, field_name), field_name)
        object.__setattr__(self, "client_error_counts", tuple(self.client_error_counts))
        for category, count in self.client_error_counts:
            _validate_non_negative_int(count, "client_error_counts")
            if category not in {item.value for item in ClientErrorCategory}:
                raise ValueError("client_error_counts must use known client error categories.")

    def to_record(self) -> dict[str, object]:
        """Return a compact JSON-serializable telemetry summary."""

        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "step_count": self.step_count,
            "llm_call_count": self.llm_call_count,
            "average_prompt_tokens": self.average_prompt_tokens,
            "average_completion_tokens": self.average_completion_tokens,
            "average_total_tokens": self.average_total_tokens,
            "average_llm_latency_ms": self.average_llm_latency_ms,
            "average_orchestration_latency_ms": self.average_orchestration_latency_ms,
            "invalid_output_count": self.invalid_output_count,
            "fallback_count": self.fallback_count,
            "successful_response_count": self.successful_response_count,
            "total_tool_call_count": self.total_tool_call_count,
            "total_replan_count": self.total_replan_count,
            "abstain_or_no_action_count": self.abstain_or_no_action_count,
            "total_estimated_cost": self.total_estimated_cost,
            "client_error_counts": [list(item) for item in self.client_error_counts],
            "total_retry_count": self.total_retry_count,
            "failure_before_response_count": self.failure_before_response_count,
            "failure_after_response_count": self.failure_after_response_count,
        }


def summarize_episode_telemetry(
    step_telemetry: tuple[OrchestrationStepTelemetry, ...],
) -> EpisodeTelemetrySummary:
    """Aggregate per-step telemetry into one compact episode summary."""

    llm_steps = tuple(
        step
        for step in step_telemetry
        if step.provider is not None and step.model_name is not None
    )
    provider = llm_steps[0].provider if llm_steps else None
    model_name = llm_steps[0].model_name if llm_steps else None
    client_error_counter: dict[str, int] = {}
    for step in step_telemetry:
        if step.client_error_category is None:
            continue
        client_error_counter[step.client_error_category.value] = (
            client_error_counter.get(step.client_error_category.value, 0) + 1
        )
    return EpisodeTelemetrySummary(
        provider=provider,
        model_name=model_name,
        step_count=len(step_telemetry),
        llm_call_count=len(llm_steps),
        average_prompt_tokens=_mean_or_none(tuple(step.prompt_tokens for step in llm_steps)),
        average_completion_tokens=_mean_or_none(
            tuple(step.completion_tokens for step in llm_steps)
        ),
        average_total_tokens=_mean_or_none(tuple(step.total_tokens for step in llm_steps)),
        average_llm_latency_ms=_mean_or_none(tuple(step.llm_latency_ms for step in llm_steps)),
        average_orchestration_latency_ms=_mean_or_none(
            tuple(step.orchestration_latency_ms for step in step_telemetry)
        ),
        invalid_output_count=sum(1 for step in step_telemetry if step.invalid_output),
        fallback_count=sum(1 for step in step_telemetry if step.fallback_used),
        successful_response_count=sum(
            1
            for step in llm_steps
            if not step.invalid_output and not step.fallback_used
        ),
        total_tool_call_count=sum(step.tool_call_count for step in step_telemetry),
        total_replan_count=sum(1 for step in step_telemetry if step.request_replan),
        abstain_or_no_action_count=sum(
            1 for step in step_telemetry if step.abstain_or_no_action
        ),
        total_estimated_cost=_sum_or_none(
            tuple(step.estimated_cost for step in step_telemetry)
        ),
        client_error_counts=tuple(sorted(client_error_counter.items())),
        total_retry_count=sum(step.retry_count for step in step_telemetry),
        failure_before_response_count=sum(
            1
            for step in step_telemetry
            if step.client_error_category is not None and step.failure_after_response is False
        ),
        failure_after_response_count=sum(
            1
            for step in step_telemetry
            if step.client_error_category is not None and step.failure_after_response is True
        ),
    )


__all__ = [
    "ClientErrorCategory",
    "EpisodeTelemetrySummary",
    "LLMCallTrace",
    "LLMCallTelemetry",
    "OrchestrationStepTelemetry",
    "ToolCallTrace",
    "summarize_episode_telemetry",
]
