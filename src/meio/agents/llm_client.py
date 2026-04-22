"""Minimal client abstractions for bounded LLM orchestration."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from time import sleep
from typing import Protocol, runtime_checkable
from urllib.parse import urlparse

from meio.agents.telemetry import ClientErrorCategory, LLMCallTelemetry
from meio.contracts import RegimeLabel
from meio.utils.env import DEFAULT_ENV_FILE, load_env_value


def _validate_optional_non_negative_int(value: int | None, field_name: str) -> None:
    if value is not None and value < 0:
        raise ValueError(f"{field_name} must be non-negative when provided.")


@dataclass(frozen=True, slots=True)
class LLMMessage:
    """Structured message sent to an LLM client."""

    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant"}:
            raise ValueError("role must be one of system, user, or assistant.")
        if not self.content.strip():
            raise ValueError("content must be a non-empty string.")


@dataclass(frozen=True, slots=True)
class LLMClientContext:
    """Compact typed context for prompt building and fake-client control."""

    benchmark_id: str
    mission_id: str
    time_index: int
    regime_label: RegimeLabel
    demand_value: float
    leadtime_value: float
    inventory_level: tuple[float, ...]
    backorder_level: tuple[float, ...]
    available_tool_ids: tuple[str, ...]
    max_tool_steps: int
    demand_baseline_value: float | None = None
    demand_change_from_baseline: float | None = None
    demand_ratio_to_baseline: float | None = None
    previous_demand_value: float | None = None
    demand_change_from_previous: float | None = None
    leadtime_baseline_value: float | None = None
    leadtime_change_from_baseline: float | None = None
    leadtime_ratio_to_baseline: float | None = None
    previous_leadtime_value: float | None = None
    leadtime_change_from_previous: float | None = None
    downstream_inventory_value: float | None = None
    total_inventory_value: float | None = None
    pipeline_total_value: float | None = None
    total_backorder_value: float | None = None
    inventory_gap_to_demand: float | None = None
    pipeline_ratio_to_baseline: float | None = None
    backorder_ratio_to_baseline: float | None = None
    repeated_stress_detected: bool | None = None
    pipeline_heavy_vs_baseline: bool | None = None
    backlog_heavy_vs_baseline: bool | None = None
    recovery_with_high_pipeline_load: bool | None = None
    recovery_with_high_backorder_load: bool | None = None
    external_evidence_present: bool | None = None
    external_evidence_source_count: int | None = None
    external_evidence_false_alarm_present: bool | None = None
    external_evidence_summaries: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.benchmark_id.strip():
            raise ValueError("benchmark_id must be a non-empty string.")
        if not self.mission_id.strip():
            raise ValueError("mission_id must be a non-empty string.")
        if self.time_index < 0:
            raise ValueError("time_index must be non-negative.")
        if not isinstance(self.regime_label, RegimeLabel):
            raise TypeError("regime_label must be a RegimeLabel.")
        if self.demand_value < 0.0:
            raise ValueError("demand_value must be non-negative.")
        if self.leadtime_value <= 0.0:
            raise ValueError("leadtime_value must be positive.")
        object.__setattr__(self, "inventory_level", tuple(self.inventory_level))
        object.__setattr__(self, "backorder_level", tuple(self.backorder_level))
        object.__setattr__(self, "available_tool_ids", tuple(self.available_tool_ids))
        if not self.available_tool_ids:
            raise ValueError("available_tool_ids must not be empty.")
        if self.max_tool_steps <= 0:
            raise ValueError("max_tool_steps must be positive.")
        for field_name in (
            "demand_baseline_value",
            "demand_change_from_baseline",
            "demand_ratio_to_baseline",
            "previous_demand_value",
            "demand_change_from_previous",
            "leadtime_baseline_value",
            "leadtime_change_from_baseline",
            "leadtime_ratio_to_baseline",
            "previous_leadtime_value",
            "leadtime_change_from_previous",
            "downstream_inventory_value",
            "total_inventory_value",
            "pipeline_total_value",
            "total_backorder_value",
            "inventory_gap_to_demand",
            "pipeline_ratio_to_baseline",
            "backorder_ratio_to_baseline",
        ):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
            ):
                raise TypeError(f"{field_name} must be numeric when provided.")
        for field_name in (
            "repeated_stress_detected",
            "pipeline_heavy_vs_baseline",
            "backlog_heavy_vs_baseline",
            "recovery_with_high_pipeline_load",
            "recovery_with_high_backorder_load",
            "external_evidence_present",
            "external_evidence_false_alarm_present",
        ):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"{field_name} must be a bool when provided.")
        _validate_optional_non_negative_int(
            self.external_evidence_source_count,
            "external_evidence_source_count",
        )
        object.__setattr__(
            self,
            "external_evidence_summaries",
            tuple(self.external_evidence_summaries),
        )
        for summary in self.external_evidence_summaries:
            if not summary.strip():
                raise ValueError(
                    "external_evidence_summaries must contain non-empty strings."
                )


@dataclass(frozen=True, slots=True)
class LLMCompletionRequest:
    """Bounded request payload for an LLM client."""

    model: str
    messages: tuple[LLMMessage, ...]
    context: LLMClientContext
    temperature: float | None = None
    prompt_version: str | None = None
    prompt_contract_hash: str | None = None

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must be a non-empty string.")
        object.__setattr__(self, "messages", tuple(self.messages))
        if not self.messages:
            raise ValueError("messages must not be empty.")
        if not isinstance(self.context, LLMClientContext):
            raise TypeError("context must be an LLMClientContext.")
        if self.temperature is not None and not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be within [0.0, 2.0] when provided.")
        if self.prompt_version is not None and not self.prompt_version.strip():
            raise ValueError("prompt_version must be non-empty when provided.")
        if self.prompt_contract_hash is not None and not self.prompt_contract_hash.strip():
            raise ValueError("prompt_contract_hash must be non-empty when provided.")
        for message in self.messages:
            if not isinstance(message, LLMMessage):
                raise TypeError("messages must contain LLMMessage values.")


@dataclass(frozen=True, slots=True)
class LLMCompletionResponse:
    """Structured response returned by an LLM client."""

    model: str
    provider: str
    content: str
    telemetry: LLMCallTelemetry | None = None
    metadata: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must be a non-empty string.")
        if not self.provider.strip():
            raise ValueError("provider must be a non-empty string.")
        if not self.content.strip():
            raise ValueError("content must be a non-empty string.")
        if self.telemetry is not None and not isinstance(self.telemetry, LLMCallTelemetry):
            raise TypeError("telemetry must be an LLMCallTelemetry when provided.")
        object.__setattr__(self, "metadata", tuple(self.metadata))


class LLMClientRuntimeError(RuntimeError):
    """Typed runtime failure raised by the bounded LLM client layer."""

    def __init__(
        self,
        *,
        category: ClientErrorCategory,
        message_summary: str,
        retry_count: int = 0,
        failure_after_response: bool = False,
    ) -> None:
        if not message_summary.strip():
            raise ValueError("message_summary must be non-empty.")
        if retry_count < 0:
            raise ValueError("retry_count must be non-negative.")
        super().__init__(message_summary)
        self.category = category
        self.message_summary = message_summary
        self.retry_count = retry_count
        self.failure_after_response = failure_after_response


@runtime_checkable
class LLMClient(Protocol):
    """Minimal protocol for bounded LLM completion clients."""

    def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        """Return one bounded completion response."""


_FAKE_RESPONSES = {
    RegimeLabel.NORMAL: {
        "selected_subgoal": "no_action",
        "candidate_tool_ids": [],
        "regime_label": "normal",
        "confidence": 0.95,
        "update_request_types": ["keep_current"],
        "request_replan": False,
        "rationale": "No intervention is justified under the current baseline regime.",
    },
    RegimeLabel.DEMAND_REGIME_SHIFT: {
        "selected_subgoal": "query_uncertainty",
        "candidate_tool_ids": ["forecast_tool", "leadtime_tool", "scenario_tool"],
        "regime_label": "demand_regime_shift",
        "confidence": 0.86,
        "update_request_types": ["switch_demand_regime", "widen_uncertainty"],
        "request_replan": True,
        "rationale": "Demand evidence suggests a regime shift and warrants bounded replanning.",
    },
    RegimeLabel.SUPPLY_DISRUPTION: {
        "selected_subgoal": "query_uncertainty",
        "candidate_tool_ids": ["forecast_tool", "leadtime_tool", "scenario_tool"],
        "regime_label": "supply_disruption",
        "confidence": 0.84,
        "update_request_types": ["switch_leadtime_regime", "widen_uncertainty"],
        "request_replan": True,
        "rationale": "Lead-time evidence suggests a supply disruption and warrants bounded replanning.",
    },
    RegimeLabel.JOINT_DISRUPTION: {
        "selected_subgoal": "query_uncertainty",
        "candidate_tool_ids": ["forecast_tool", "leadtime_tool", "scenario_tool"],
        "regime_label": "joint_disruption",
        "confidence": 0.83,
        "update_request_types": ["switch_demand_regime", "switch_leadtime_regime", "widen_uncertainty"],
        "request_replan": True,
        "rationale": "Joint disruption evidence warrants a guarded multi-tool uncertainty update.",
    },
    RegimeLabel.RECOVERY: {
        "selected_subgoal": "no_action",
        "candidate_tool_ids": [],
        "regime_label": "recovery",
        "confidence": 0.82,
        "update_request_types": ["keep_current"],
        "request_replan": False,
        "rationale": "Recovery evidence does not justify another intervention right now.",
    },
}

_PREFERRED_TOOL_IDS_BY_REGIME = {
    RegimeLabel.NORMAL: (),
    RegimeLabel.DEMAND_REGIME_SHIFT: ("forecast_tool", "leadtime_tool", "scenario_tool"),
    RegimeLabel.SUPPLY_DISRUPTION: ("forecast_tool", "leadtime_tool", "scenario_tool"),
    RegimeLabel.JOINT_DISRUPTION: ("forecast_tool", "leadtime_tool", "scenario_tool"),
    RegimeLabel.RECOVERY: (),
}


@dataclass(frozen=True, slots=True)
class FakeLLMClient:
    """Deterministic fake client for bounded orchestration tests and smoke runs."""

    provider: str = "fake_llm_client"

    def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        payload = _fake_payload_for_context(request.context)
        return LLMCompletionResponse(
            model=request.model,
            provider=self.provider,
            content=json.dumps(payload, sort_keys=True),
            telemetry=LLMCallTelemetry(
                provider=self.provider,
                model_name=request.model,
                prompt_tokens=96,
                completion_tokens=48,
                total_tokens=144,
            ),
            metadata=("deterministic_fake_response",),
        )


def _fake_payload_for_context(context: LLMClientContext) -> dict[str, object]:
    payload = dict(_FAKE_RESPONSES[context.regime_label])
    preferred_tool_ids = _PREFERRED_TOOL_IDS_BY_REGIME[context.regime_label]
    selected_tool_ids = tuple(
        tool_id
        for tool_id in preferred_tool_ids
        if tool_id in context.available_tool_ids
    )
    if (
        context.external_evidence_present
        and "external_evidence_tool" in context.available_tool_ids
    ):
        selected_tool_ids = ("external_evidence_tool",) + selected_tool_ids
    deduped_tool_ids: list[str] = []
    for tool_id in selected_tool_ids:
        if tool_id not in deduped_tool_ids:
            deduped_tool_ids.append(tool_id)
    if deduped_tool_ids:
        payload["candidate_tool_ids"] = deduped_tool_ids
        payload["selected_subgoal"] = "query_uncertainty"
        if context.regime_label is RegimeLabel.NORMAL:
            payload["request_replan"] = False
            payload["update_request_types"] = ["keep_current"]
            payload["rationale"] = (
                "External evidence is present, so inspect it first before deciding whether "
                "any bounded planning change is justified."
            )
        else:
            payload["request_replan"] = True
            payload["rationale"] = (
                "The available bounded tools should be used before guarded replanning."
            )
        return payload
    payload["candidate_tool_ids"] = []
    if context.regime_label in {
        RegimeLabel.DEMAND_REGIME_SHIFT,
        RegimeLabel.SUPPLY_DISRUPTION,
        RegimeLabel.JOINT_DISRUPTION,
    }:
        payload["selected_subgoal"] = "request_replan"
        payload["request_replan"] = True
        payload["rationale"] = (
            "Evidence suggests planning inputs should change, but no bounded uncertainty "
            "tool is available in this ablation condition."
        )
    return payload


@dataclass(frozen=True, slots=True)
class OpenAILLMClient:
    """Optional OpenAI-backed client for bounded orchestration."""

    provider: str = "openai"
    api_key_env: str = "OPENAI_API_KEY"
    env_file_path: Path = DEFAULT_ENV_FILE
    request_timeout_s: float = 20.0
    max_retries: int = 1

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("provider must be a non-empty string.")
        if self.request_timeout_s <= 0.0:
            raise ValueError("request_timeout_s must be positive.")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative.")

    def resolve_api_key(self) -> str | None:
        """Resolve the API key from process env first, then the repo .env file."""

        return load_env_value(self.api_key_env, env_file_path=self.env_file_path)

    def ensure_available(self) -> None:
        """Fail clearly when the real client is requested without prerequisites."""

        api_key = self.resolve_api_key()
        if not api_key:
            raise RuntimeError(
                f"{self.api_key_env} is not set in the environment or .env file; "
                "cannot use the OpenAI LLM client."
            )
        try:
            import openai  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("openai package is not available for the OpenAI client.") from exc

    def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.ensure_available()

        api_key = self.resolve_api_key()
        assert api_key is not None
        client = self._build_openai_client(api_key=api_key)
        try:
            request_payload: dict[str, object] = {
                "model": request.model,
                "input": [
                    {"role": message.role, "content": message.content}
                    for message in request.messages
                ],
            }
            if request.temperature is not None:
                request_payload["temperature"] = request.temperature
            metadata: list[str] = ["real_api_response"]
            if _should_bypass_proxy_env():
                metadata.append("proxy_env_bypassed")
            response = None
            retries_used = 0
            attempt_count = self.max_retries + 1
            for attempt_index in range(attempt_count):
                try:
                    response = client.responses.create(**request_payload)
                    retries_used = attempt_index
                    break
                except Exception as exc:  # pragma: no cover - exercised in live environments
                    category, message_summary = _classify_client_exception(exc)
                    if (
                        attempt_index < self.max_retries
                        and category in _RETRYABLE_ERROR_CATEGORIES
                    ):
                        retries_used = attempt_index + 1
                        sleep(_retry_delay_s(attempt_index + 1))
                        continue
                    raise LLMClientRuntimeError(
                        category=category,
                        message_summary=message_summary,
                        retry_count=attempt_index,
                        failure_after_response=False,
                    ) from exc
            if response is None:
                raise LLMClientRuntimeError(
                    category=ClientErrorCategory.UNKNOWN_CLIENT_ERROR,
                    message_summary="OpenAI client did not produce a response.",
                    retry_count=retries_used,
                    failure_after_response=False,
                )
            try:
                content = _extract_response_content(response)
            except LLMClientRuntimeError as exc:
                raise LLMClientRuntimeError(
                    category=exc.category,
                    message_summary=exc.message_summary,
                    retry_count=retries_used,
                    failure_after_response=exc.failure_after_response,
                ) from exc
            prompt_tokens = None
            completion_tokens = None
            total_tokens = None
            usage = getattr(response, "usage", None)
            try:
                if usage is not None:
                    prompt_tokens = _coerce_optional_usage_int(
                        getattr(usage, "input_tokens", None)
                    )
                    completion_tokens = _coerce_optional_usage_int(
                        getattr(usage, "output_tokens", None)
                    )
                    total_tokens = _coerce_optional_usage_int(
                        getattr(usage, "total_tokens", None)
                    )
            except (TypeError, ValueError) as exc:
                raise LLMClientRuntimeError(
                    category=ClientErrorCategory.RESPONSE_PARSE_ERROR,
                    message_summary=_summarize_exception(exc),
                    retry_count=retries_used,
                    failure_after_response=True,
                ) from exc
            if prompt_tokens is None and completion_tokens is None and total_tokens is None:
                metadata.append("usage_unavailable")
            if retries_used > 0:
                metadata.append("retried_request")
            return LLMCompletionResponse(
                model=request.model,
                provider=self.provider,
                content=content,
                telemetry=LLMCallTelemetry(
                    provider=self.provider,
                    model_name=request.model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    retry_count=retries_used,
                ),
                metadata=tuple(metadata),
            )
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()

    def _build_openai_client(self, *, api_key: str):
        from openai import OpenAI

        client_kwargs: dict[str, object] = {
            "api_key": api_key,
            "timeout": self.request_timeout_s,
            "max_retries": 0,
        }
        if _should_bypass_proxy_env():
            import httpx

            client_kwargs["http_client"] = httpx.Client(
                timeout=self.request_timeout_s,
                trust_env=False,
            )
        return OpenAI(**client_kwargs)


def _coerce_optional_usage_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("usage metadata values must be integers when provided.")
    if value < 0:
        raise ValueError("usage metadata values must be non-negative.")
    return value


_RETRYABLE_ERROR_CATEGORIES = {
    ClientErrorCategory.TIMEOUT_ERROR,
    ClientErrorCategory.RATE_LIMIT_ERROR,
    ClientErrorCategory.NETWORK_ERROR,
}


def _retry_delay_s(retry_index: int) -> float:
    return min(1.0, 0.25 * retry_index)


def _extract_response_content(response: object) -> str:
    content = getattr(response, "output_text", None)
    if isinstance(content, str) and content.strip():
        return content.strip()
    raise LLMClientRuntimeError(
        category=ClientErrorCategory.RESPONSE_PARSE_ERROR,
        message_summary="OpenAI client returned an empty orchestration response body.",
        retry_count=0,
        failure_after_response=True,
    )


def _classify_client_exception(exc: Exception) -> tuple[ClientErrorCategory, str]:
    import httpx
    import openai

    for item in _iter_exception_chain(exc):
        if isinstance(item, (openai.APITimeoutError, httpx.TimeoutException)):
            return ClientErrorCategory.TIMEOUT_ERROR, _summarize_exception(item)
        if isinstance(item, openai.RateLimitError):
            return ClientErrorCategory.RATE_LIMIT_ERROR, _summarize_exception(item)
        if isinstance(
            item,
            (
                openai.APIConnectionError,
                httpx.NetworkError,
                httpx.ConnectError,
                httpx.ReadError,
                httpx.WriteError,
                httpx.CloseError,
            ),
        ):
            return ClientErrorCategory.NETWORK_ERROR, _summarize_exception(item)
        if isinstance(item, openai.APIError):
            return ClientErrorCategory.API_ERROR, _summarize_exception(item)
    return ClientErrorCategory.UNKNOWN_CLIENT_ERROR, _summarize_exception(exc)


def _iter_exception_chain(exc: BaseException) -> tuple[BaseException, ...]:
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__ or current.__context__
    return tuple(chain)


def _summarize_exception(exc: BaseException) -> str:
    message = str(exc).strip()
    if message:
        return f"{exc.__class__.__name__}: {message}"
    return exc.__class__.__name__


def _should_bypass_proxy_env() -> bool:
    for env_key in ("ALL_PROXY", "HTTPS_PROXY", "HTTP_PROXY"):
        if _is_dead_local_proxy(os.getenv(env_key)):
            return True
    return False


def _is_dead_local_proxy(value: str | None) -> bool:
    if value is None or not value.strip():
        return False
    parsed = urlparse(value)
    host = (parsed.hostname or "").lower()
    return host in {"127.0.0.1", "localhost", "::1"} and parsed.port == 9


__all__ = [
    "FakeLLMClient",
    "LLMClient",
    "LLMClientContext",
    "LLMCompletionRequest",
    "LLMCompletionResponse",
    "LLMClientRuntimeError",
    "LLMMessage",
    "OpenAILLMClient",
]
