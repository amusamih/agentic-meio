"""Bounded external evidence interpretation tool for the semi-synthetic branch."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from meio.contracts import (
    BoundedTool,
    OperationalSubgoal,
    RegimeLabel,
    ToolClass,
    ToolInvocation,
    ToolResult,
    ToolSpec,
    ToolStatus,
)
from meio.data.external_evidence import (
    ExternalEvidenceBatch,
    ExternalEvidenceRecord,
    ExternalEvidenceSeverity,
)


class ExternalRiskDirection(StrEnum):
    """Bounded interpreted risk directions from external evidence."""

    BENIGN_OR_FALSE_ALARM = "benign_or_false_alarm"
    DEMAND_UPSIDE_RISK = "demand_upside_risk"
    LEADTIME_RISK = "leadtime_risk"
    MIXED_RISK = "mixed_risk"


class ExternalUncertaintyPressure(StrEnum):
    """Bounded uncertainty pressure levels inferred from external evidence."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class ExternalEvidenceInterpretation:
    """Structured bounded interpretation emitted by the external evidence tool."""

    source_ids: tuple[str, ...]
    interpreted_risk_direction: ExternalRiskDirection
    recommended_regime_hint: RegimeLabel
    recommended_uncertainty_pressure: ExternalUncertaintyPressure
    confidence: float
    false_alarm_evidence_detected: bool = False
    summary: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_ids", tuple(self.source_ids))
        for source_id in self.source_ids:
            if not source_id.strip():
                raise ValueError("source_ids must contain non-empty strings.")
        if not isinstance(self.interpreted_risk_direction, ExternalRiskDirection):
            raise TypeError(
                "interpreted_risk_direction must be an ExternalRiskDirection."
            )
        if not isinstance(self.recommended_regime_hint, RegimeLabel):
            raise TypeError("recommended_regime_hint must be a RegimeLabel.")
        if not isinstance(
            self.recommended_uncertainty_pressure,
            ExternalUncertaintyPressure,
        ):
            raise TypeError(
                "recommended_uncertainty_pressure must be an ExternalUncertaintyPressure."
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be within [0.0, 1.0].")
        if not self.summary.strip():
            raise ValueError("summary must be a non-empty string.")
        object.__setattr__(self, "notes", tuple(self.notes))
        for note in self.notes:
            if not note.strip():
                raise ValueError("notes must contain non-empty strings.")


def apply_external_evidence_bias(
    interpretation: ExternalEvidenceInterpretation,
    *,
    demand_outlook: float,
    leadtime_outlook: float,
    safety_buffer_scale: float,
) -> tuple[float, float, float]:
    """Apply a narrow bounded optimizer-input bias from interpreted external evidence."""

    demand_scale = 1.0
    leadtime_scale = 1.0
    buffer_scale = 1.0
    if interpretation.recommended_uncertainty_pressure is ExternalUncertaintyPressure.MEDIUM:
        buffer_scale = 1.03
    elif interpretation.recommended_uncertainty_pressure is ExternalUncertaintyPressure.HIGH:
        buffer_scale = 1.06
    if interpretation.interpreted_risk_direction is ExternalRiskDirection.DEMAND_UPSIDE_RISK:
        demand_scale = 1.03
    elif interpretation.interpreted_risk_direction is ExternalRiskDirection.LEADTIME_RISK:
        leadtime_scale = 1.05
    elif interpretation.interpreted_risk_direction is ExternalRiskDirection.MIXED_RISK:
        demand_scale = 1.03
        leadtime_scale = 1.05
    if interpretation.false_alarm_evidence_detected:
        return demand_outlook, leadtime_outlook, safety_buffer_scale
    return (
        demand_outlook * demand_scale,
        leadtime_outlook * leadtime_scale,
        safety_buffer_scale * buffer_scale,
    )


def _severity_rank(severity: ExternalEvidenceSeverity) -> int:
    return {
        ExternalEvidenceSeverity.LOW: 1,
        ExternalEvidenceSeverity.MEDIUM: 2,
        ExternalEvidenceSeverity.HIGH: 3,
    }[severity]


def _confidence_from_records(records: tuple[ExternalEvidenceRecord, ...]) -> float:
    if not records:
        return 0.5
    explicit = tuple(
        record.credibility for record in records if record.credibility is not None
    )
    if explicit:
        return sum(explicit) / len(explicit)
    severity_score = sum(_severity_rank(record.severity) for record in records) / len(records)
    return min(0.9, 0.45 + 0.12 * severity_score)


def _pressure_from_records(
    records: tuple[ExternalEvidenceRecord, ...],
) -> ExternalUncertaintyPressure:
    max_rank = max((_severity_rank(record.severity) for record in records), default=1)
    if max_rank >= 3:
        return ExternalUncertaintyPressure.HIGH
    if max_rank == 2:
        return ExternalUncertaintyPressure.MEDIUM
    return ExternalUncertaintyPressure.LOW


def _risk_direction_from_records(
    records: tuple[ExternalEvidenceRecord, ...],
) -> ExternalRiskDirection:
    if not records or all(record.false_alarm for record in records):
        return ExternalRiskDirection.BENIGN_OR_FALSE_ALARM
    regime_hints = {
        record.aligned_regime_label
        for record in records
        if record.aligned_regime_label is not None and not record.false_alarm
    }
    if regime_hints & {RegimeLabel.JOINT_DISRUPTION}:
        return ExternalRiskDirection.MIXED_RISK
    if regime_hints & {RegimeLabel.SUPPLY_DISRUPTION}:
        return ExternalRiskDirection.LEADTIME_RISK
    if regime_hints & {
        RegimeLabel.DEMAND_REGIME_SHIFT,
        RegimeLabel.RECOVERY,
    }:
        return ExternalRiskDirection.DEMAND_UPSIDE_RISK
    return ExternalRiskDirection.BENIGN_OR_FALSE_ALARM


def _regime_hint_from_records(
    records: tuple[ExternalEvidenceRecord, ...],
) -> RegimeLabel:
    non_false_alarm = tuple(record for record in records if not record.false_alarm)
    if not non_false_alarm:
        return RegimeLabel.NORMAL
    hint_counts: dict[RegimeLabel, int] = {}
    for record in non_false_alarm:
        hint = record.aligned_regime_label or RegimeLabel.NORMAL
        hint_counts[hint] = hint_counts.get(hint, 0) + _severity_rank(record.severity)
    return max(
        hint_counts.items(),
        key=lambda item: (item[1], item[0].value),
    )[0]


def interpret_external_evidence(
    evidence_batch: ExternalEvidenceBatch | None,
) -> ExternalEvidenceInterpretation:
    """Return a bounded interpretation from one typed external evidence batch."""

    records = evidence_batch.records if evidence_batch is not None else ()
    if not records:
        return ExternalEvidenceInterpretation(
            source_ids=(),
            interpreted_risk_direction=ExternalRiskDirection.BENIGN_OR_FALSE_ALARM,
            recommended_regime_hint=RegimeLabel.NORMAL,
            recommended_uncertainty_pressure=ExternalUncertaintyPressure.LOW,
            confidence=0.5,
            false_alarm_evidence_detected=False,
            summary="No external evidence is attached to this period.",
            notes=("external_evidence_absent",),
        )
    regime_hint = _regime_hint_from_records(records)
    false_alarm_detected = all(record.false_alarm for record in records)
    return ExternalEvidenceInterpretation(
        source_ids=tuple(record.source_id for record in records),
        interpreted_risk_direction=_risk_direction_from_records(records),
        recommended_regime_hint=regime_hint,
        recommended_uncertainty_pressure=_pressure_from_records(records),
        confidence=_confidence_from_records(records),
        false_alarm_evidence_detected=false_alarm_detected,
        summary=(
            "External evidence suggests "
            f"{regime_hint.value} with "
            f"{_pressure_from_records(records).value} uncertainty pressure."
        ),
        notes=tuple(
            sorted(
                {
                    "contains_false_alarm" if record.false_alarm else "contains_relevant_signal"
                    for record in records
                }
            )
        ),
    )


@dataclass(frozen=True, slots=True)
class BoundedExternalEvidenceTool(BoundedTool):
    """Deterministic bounded interpreter for semi-synthetic external evidence."""

    tool_id: str = "external_evidence_tool"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            tool_id=self.tool_id,
            tool_class=ToolClass.LEARNED_NON_LLM,
            supported_subgoals=(
                OperationalSubgoal.QUERY_UNCERTAINTY,
                OperationalSubgoal.UPDATE_UNCERTAINTY,
            ),
            description=(
                "Bounded semi-synthetic external evidence interpreter for early, delayed, "
                "noisy, false-alarm, and relapse signals."
            ),
        )

    def invoke(self, invocation: ToolInvocation) -> ToolResult:
        evidence_batch = (
            invocation.evidence.external_evidence_batch
            if invocation.evidence is not None
            else None
        )
        interpretation = interpret_external_evidence(evidence_batch)
        status = ToolStatus.SUCCESS
        if evidence_batch is None or not evidence_batch.records:
            status = ToolStatus.NO_ACTION
        provenance = "semi_synthetic_external_evidence_tool"
        return ToolResult(
            tool_id=invocation.tool_id,
            tool_class=invocation.tool_class,
            subgoal=invocation.subgoal,
            status=status,
            structured_output={
                "external_evidence_interpretation": interpretation,
            },
            confidence=interpretation.confidence,
            provenance=provenance,
            next_tool_id="forecast_tool",
            next_subgoal=OperationalSubgoal.QUERY_UNCERTAINTY,
            request_replan=False,
        )


__all__ = [
    "BoundedExternalEvidenceTool",
    "ExternalEvidenceInterpretation",
    "ExternalRiskDirection",
    "ExternalUncertaintyPressure",
    "apply_external_evidence_bias",
    "interpret_external_evidence",
]
