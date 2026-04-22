"""Typed semi-synthetic external evidence containers for the Stockpyl branch."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from meio.contracts import RegimeLabel


def _validate_non_empty_text(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")


def _validate_optional_non_empty_text(value: str | None, field_name: str) -> None:
    if value is not None:
        _validate_non_empty_text(value, field_name)


def _validate_optional_probability(value: float | None, field_name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric when provided.")
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{field_name} must be within [0.0, 1.0] when provided.")


class ExternalEvidenceType(StrEnum):
    """Supported coarse evidence-source types for the semi-synthetic branch."""

    NEWS = "news"
    SUPPLIER_ALERT = "supplier_alert"
    PORT_SIGNAL = "port_signal"
    WEATHER = "weather"
    SOCIAL = "social"
    INTERNAL_MEMO = "internal_memo"


class ExternalEvidenceSeverity(StrEnum):
    """Bounded coarse severity levels for external evidence."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ExternalEvidenceTiming(StrEnum):
    """Timing of the evidence relative to the benchmark period it is attached to."""

    LEADING = "leading"
    SAME_PERIOD = "same_period"
    LAGGED = "lagged"


class ExternalEvidenceStrengthTier(StrEnum):
    """Conservative bounded evidence-strength tiers for curated external records."""

    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"


@dataclass(frozen=True, slots=True)
class ExternalEvidenceRecord:
    """One typed semi-synthetic external evidence item."""

    source_id: str
    source_type: ExternalEvidenceType
    headline_or_summary: str
    event_type: str
    location: str | None = None
    organization: str | None = None
    source_domain: str | None = None
    severity: ExternalEvidenceSeverity = ExternalEvidenceSeverity.LOW
    timing: ExternalEvidenceTiming = ExternalEvidenceTiming.SAME_PERIOD
    credibility: float | None = None
    aligned_regime_label: RegimeLabel | None = None
    false_alarm: bool = False
    role_label: str | None = None
    corroboration_group: str | None = None
    evidence_strength_tier: ExternalEvidenceStrengthTier | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _validate_non_empty_text(self.source_id, "source_id")
        if not isinstance(self.source_type, ExternalEvidenceType):
            raise TypeError("source_type must be an ExternalEvidenceType.")
        _validate_non_empty_text(self.headline_or_summary, "headline_or_summary")
        _validate_non_empty_text(self.event_type, "event_type")
        _validate_optional_non_empty_text(self.location, "location")
        _validate_optional_non_empty_text(self.organization, "organization")
        _validate_optional_non_empty_text(self.source_domain, "source_domain")
        if not isinstance(self.severity, ExternalEvidenceSeverity):
            raise TypeError("severity must be an ExternalEvidenceSeverity.")
        if not isinstance(self.timing, ExternalEvidenceTiming):
            raise TypeError("timing must be an ExternalEvidenceTiming.")
        _validate_optional_probability(self.credibility, "credibility")
        if self.aligned_regime_label is not None and not isinstance(
            self.aligned_regime_label,
            RegimeLabel,
        ):
            raise TypeError("aligned_regime_label must be a RegimeLabel when provided.")
        _validate_optional_non_empty_text(self.role_label, "role_label")
        _validate_optional_non_empty_text(
            self.corroboration_group,
            "corroboration_group",
        )
        if self.evidence_strength_tier is not None and not isinstance(
            self.evidence_strength_tier,
            ExternalEvidenceStrengthTier,
        ):
            raise TypeError(
                "evidence_strength_tier must be an ExternalEvidenceStrengthTier when provided."
            )
        object.__setattr__(self, "notes", tuple(self.notes))
        for note in self.notes:
            _validate_non_empty_text(note, "notes")


@dataclass(frozen=True, slots=True)
class ExternalEvidenceBatch:
    """Typed evidence attached to one benchmark period."""

    period_index: int
    records: tuple[ExternalEvidenceRecord, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.period_index < 0:
            raise ValueError("period_index must be non-negative.")
        object.__setattr__(self, "records", tuple(self.records))
        object.__setattr__(self, "notes", tuple(self.notes))
        for record in self.records:
            if not isinstance(record, ExternalEvidenceRecord):
                raise TypeError("records must contain ExternalEvidenceRecord values.")
        for note in self.notes:
            _validate_non_empty_text(note, "notes")

    @property
    def record_count(self) -> int:
        """Return the number of evidence items in the batch."""

        return len(self.records)

    @property
    def contains_false_alarm(self) -> bool:
        """Return whether any attached evidence item is marked as a false alarm."""

        return any(record.false_alarm for record in self.records)


__all__ = [
    "ExternalEvidenceBatch",
    "ExternalEvidenceRecord",
    "ExternalEvidenceSeverity",
    "ExternalEvidenceStrengthTier",
    "ExternalEvidenceTiming",
    "ExternalEvidenceType",
]
