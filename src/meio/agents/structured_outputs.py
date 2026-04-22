"""Structured validators for bounded LLM orchestration output."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from meio.contracts import OperationalSubgoal, RegimeLabel, UpdateRequest, UpdateRequestType

_ALLOWED_FIELDS = frozenset(
    {
        "selected_subgoal",
        "candidate_tool_ids",
        "regime_label",
        "confidence",
        "update_request_types",
        "request_replan",
        "rationale",
    }
)
_CONFIDENCE_ALIASES = {
    "low": 0.25,
    "medium": 0.5,
    "high": 0.85,
}


class LLMOutputValidationError(ValueError):
    """Raised when an LLM orchestration payload is malformed or out of scope."""

    def __init__(
        self,
        reason_code: str,
        message: str,
        *,
        recoverable: bool = False,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.recoverable = recoverable
        self.details = dict(details or {})


@dataclass(frozen=True, slots=True)
class StructuredLLMDecision:
    """Validated structured decision returned by an LLM orchestrator."""

    selected_subgoal: OperationalSubgoal
    candidate_tool_ids: tuple[str, ...]
    regime_label: RegimeLabel
    confidence: float
    update_requests: tuple[UpdateRequest, ...] = field(default_factory=tuple)
    request_replan: bool = False
    rationale: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.selected_subgoal, OperationalSubgoal):
            raise TypeError("selected_subgoal must be an OperationalSubgoal.")
        object.__setattr__(self, "candidate_tool_ids", tuple(self.candidate_tool_ids))
        if not isinstance(self.regime_label, RegimeLabel):
            raise TypeError("regime_label must be a RegimeLabel.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be within [0.0, 1.0].")
        object.__setattr__(self, "update_requests", tuple(self.update_requests))
        if not self.rationale.strip():
            raise ValueError("rationale must be a non-empty string.")
        for tool_id in self.candidate_tool_ids:
            if not tool_id.strip():
                raise ValueError("candidate_tool_ids must contain non-empty identifiers.")
        for update_request in self.update_requests:
            if not isinstance(update_request, UpdateRequest):
                raise TypeError("update_requests must contain UpdateRequest values.")
        if self.selected_subgoal in {OperationalSubgoal.NO_ACTION, OperationalSubgoal.ABSTAIN}:
            if self.candidate_tool_ids:
                raise ValueError("candidate_tool_ids must be empty for no_action or abstain.")
            if self.request_replan:
                raise ValueError("request_replan must be False for no_action or abstain.")

    def to_record(self) -> dict[str, object]:
        """Return a compact JSON-serializable decision record."""

        return {
            "selected_subgoal": self.selected_subgoal.value,
            "candidate_tool_ids": list(self.candidate_tool_ids),
            "regime_label": self.regime_label.value,
            "confidence": self.confidence,
            "update_request_types": [
                update_request.request_type.value for update_request in self.update_requests
            ],
            "request_replan": self.request_replan,
            "rationale": self.rationale,
        }


@dataclass(frozen=True, slots=True)
class ParsedLLMOutput:
    """Validated structured output plus recoverable normalization notes."""

    decision: StructuredLLMDecision
    normalization_notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.decision, StructuredLLMDecision):
            raise TypeError("decision must be a StructuredLLMDecision.")
        object.__setattr__(self, "normalization_notes", tuple(self.normalization_notes))
        for note in self.normalization_notes:
            if not note.strip():
                raise ValueError("normalization_notes must contain non-empty strings.")


def parse_llm_output(
    raw_content: str,
    *,
    allowed_tool_ids: tuple[str, ...],
    allowed_update_types: tuple[UpdateRequestType, ...],
) -> StructuredLLMDecision:
    """Parse and validate a bounded LLM orchestration payload."""

    return parse_llm_output_detailed(
        raw_content,
        allowed_tool_ids=allowed_tool_ids,
        allowed_update_types=allowed_update_types,
    ).decision


def parse_llm_output_detailed(
    raw_content: str,
    *,
    allowed_tool_ids: tuple[str, ...],
    allowed_update_types: tuple[UpdateRequestType, ...],
) -> ParsedLLMOutput:
    """Parse and validate a bounded LLM orchestration payload with notes."""

    cleaned_content, preprocessing_notes = sanitize_llm_output(raw_content)
    try:
        payload = json.loads(cleaned_content)
    except json.JSONDecodeError as exc:
        raise LLMOutputValidationError(
            "malformed_json",
            "LLM output must be valid JSON.",
        ) from exc
    if not isinstance(payload, dict):
        raise LLMOutputValidationError(
            "invalid_root_type",
            "LLM output must decode to a JSON object.",
        )
    unexpected_fields = set(payload) - _ALLOWED_FIELDS
    if unexpected_fields:
        raise LLMOutputValidationError(
            "disallowed_fields",
            f"LLM output contains disallowed fields: {sorted(unexpected_fields)}.",
        )

    selected_subgoal = _parse_enum(
        OperationalSubgoal,
        payload,
        "selected_subgoal",
    )
    regime_label = _parse_enum(RegimeLabel, payload, "regime_label")
    confidence, confidence_notes = _parse_confidence(payload)
    request_replan = _parse_boolean(payload, "request_replan")
    rationale = _parse_text(payload, "rationale")
    candidate_tool_ids = _parse_string_list(payload, "candidate_tool_ids")
    unavailable_tool_ids = tuple(
        tool_id for tool_id in candidate_tool_ids if tool_id not in allowed_tool_ids
    )
    if unavailable_tool_ids:
        raise LLMOutputValidationError(
            "requested_missing_tool",
            "LLM selected tool ids outside the advertised available tool set.",
            details={
                "requested_tool_ids": tuple(candidate_tool_ids),
                "unavailable_tool_ids": unavailable_tool_ids,
            },
        )
    update_request_types = _parse_string_list(payload, "update_request_types")
    parsed_update_requests: list[UpdateRequest] = []
    for update_type_name in update_request_types:
        try:
            update_type = UpdateRequestType(update_type_name)
        except ValueError as exc:
            raise LLMOutputValidationError(
                "invalid_update_request_type",
                f"Unsupported update_request_type: {update_type_name!r}.",
            ) from exc
        if update_type not in allowed_update_types:
            raise LLMOutputValidationError(
                "disallowed_update_request_type",
                f"Update request type is outside the configured allowlist: {update_type_name!r}.",
            )
        parsed_update_requests.append(
            UpdateRequest(
                request_type=update_type,
                target="serial_scenarios",
                notes="llm_orchestrator_request",
            )
        )
    return ParsedLLMOutput(
        decision=StructuredLLMDecision(
            selected_subgoal=selected_subgoal,
            candidate_tool_ids=tuple(candidate_tool_ids),
            regime_label=regime_label,
            confidence=confidence,
            update_requests=tuple(parsed_update_requests),
            request_replan=request_replan,
            rationale=rationale,
        ),
        normalization_notes=preprocessing_notes + confidence_notes,
    )


def sanitize_llm_output(raw_content: str) -> tuple[str, tuple[str, ...]]:
    """Return cleaned content plus explicit recoverable normalization notes."""

    if not isinstance(raw_content, str):
        raise LLMOutputValidationError(
            "invalid_raw_content",
            "LLM output must be a string before parsing.",
        )
    cleaned = raw_content.strip()
    notes: list[str] = []
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        if (
            len(lines) >= 3
            and lines[0].lstrip().startswith("```")
            and lines[-1].strip() == "```"
        ):
            cleaned = "\n".join(lines[1:-1]).strip()
            if cleaned.lower().startswith("json\n"):
                cleaned = cleaned[5:].strip()
            notes.append("trimmed_code_fence")
    return cleaned, tuple(notes)


def _parse_enum(enum_type: type[object], payload: dict[str, object], field_name: str) -> object:
    raw_value = _require_field(payload, field_name)
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise LLMOutputValidationError(
            f"invalid_{field_name}",
            f"{field_name} must be a non-empty string.",
        )
    try:
        return enum_type(raw_value)
    except ValueError as exc:
        raise LLMOutputValidationError(
            f"invalid_{field_name}",
            f"{field_name} is out of scope: {raw_value!r}.",
        ) from exc


def _parse_confidence(payload: dict[str, object]) -> tuple[float, tuple[str, ...]]:
    raw_value = _require_field(payload, "confidence")
    notes: list[str] = []
    if isinstance(raw_value, bool):
        raise LLMOutputValidationError(
            "invalid_confidence",
            "confidence must be a numeric value.",
        )
    if isinstance(raw_value, (int, float)):
        confidence = float(raw_value)
    elif isinstance(raw_value, str):
        normalized_value = raw_value.strip().lower()
        if normalized_value in _CONFIDENCE_ALIASES:
            confidence = _CONFIDENCE_ALIASES[normalized_value]
            notes.append(f"normalized_confidence_alias:{normalized_value}")
        else:
            try:
                confidence = float(normalized_value)
                notes.append("normalized_confidence_string")
            except ValueError as exc:
                raise LLMOutputValidationError(
                    "invalid_confidence",
                    "confidence must be a number in [0.0, 1.0].",
                ) from exc
    else:
        raise LLMOutputValidationError(
            "invalid_confidence",
            "confidence must be a numeric value.",
        )
    if not 0.0 <= confidence <= 1.0:
        raise LLMOutputValidationError(
            "invalid_confidence",
            "confidence must be within [0.0, 1.0].",
        )
    return confidence, tuple(notes)


def _parse_boolean(payload: dict[str, object], field_name: str) -> bool:
    raw_value = _require_field(payload, field_name)
    if not isinstance(raw_value, bool):
        raise LLMOutputValidationError(
            f"invalid_{field_name}",
            f"{field_name} must be a boolean.",
        )
    return raw_value


def _parse_text(payload: dict[str, object], field_name: str) -> str:
    raw_value = _require_field(payload, field_name)
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise LLMOutputValidationError(
            f"invalid_{field_name}",
            f"{field_name} must be a non-empty string.",
        )
    return raw_value.strip()


def _parse_string_list(payload: dict[str, object], field_name: str) -> list[str]:
    raw_value = _require_field(payload, field_name)
    if not isinstance(raw_value, list):
        raise LLMOutputValidationError(
            f"invalid_{field_name}",
            f"{field_name} must be a list.",
        )
    values: list[str] = []
    for item in raw_value:
        if not isinstance(item, str) or not item.strip():
            raise LLMOutputValidationError(
                f"invalid_{field_name}",
                f"{field_name} must contain non-empty strings.",
            )
        values.append(item)
    return values


def _require_field(payload: dict[str, object], field_name: str) -> object:
    if field_name not in payload:
        raise LLMOutputValidationError(
            f"missing_{field_name}",
            f"{field_name} is required.",
        )
    return payload[field_name]


__all__ = [
    "LLMOutputValidationError",
    "ParsedLLMOutput",
    "StructuredLLMDecision",
    "parse_llm_output",
    "parse_llm_output_detailed",
    "sanitize_llm_output",
]
