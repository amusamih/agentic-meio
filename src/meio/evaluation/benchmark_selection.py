"""Typed benchmark-qualification summaries for repository-fit judgments."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any
import tomllib


class BenchmarkCandidate(StrEnum):
    """Benchmark candidates currently under qualification."""

    STOCKPYL_SERIAL = "stockpyl_serial"
    OR_GYM_INVENTORY = "or_gym_inventory"
    MABIM = "mabim"


class QualificationCriterion(StrEnum):
    """Criteria used to qualify benchmark fit to the project scope."""

    TOPOLOGY_FIT = "topology_fit"
    REGIME_SHIFT_FIT = "regime_shift_fit"
    OPTIMIZER_BOUNDARY_FIT = "optimizer_boundary_fit"
    EVENT_TRIGGER_COMPATIBILITY = "event_trigger_compatibility"
    IMPLEMENTATION_EFFORT = "implementation_effort"
    EXPECTED_PAPER_VALUE = "expected_paper_value"


class QualificationLevel(StrEnum):
    """Compact ordinal labels for repository-fit judgments."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNVERIFIED = "unverified"


class QualificationDecision(StrEnum):
    """Current shortlist decision for a benchmark candidate."""

    KEEP = "keep"
    DEFER = "defer"
    DROP = "drop"


class QualificationReadiness(StrEnum):
    """Current implementation-readiness state for a candidate wrapper."""

    IMMEDIATELY_SMOKE_TESTABLE = "immediately_smoke_testable"
    PARTIALLY_INTEGRABLE = "partially_integrable"
    DEFERRED_MISSING_EXTERNAL_INTEGRATION = "deferred_missing_external_integration"


@dataclass(frozen=True, slots=True)
class CriterionAssessment:
    """One compact benchmark-fit judgment."""

    criterion: QualificationCriterion
    level: QualificationLevel
    note: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.criterion, QualificationCriterion):
            raise TypeError("criterion must be a QualificationCriterion.")
        if not isinstance(self.level, QualificationLevel):
            raise TypeError("level must be a QualificationLevel.")
        if self.note and not self.note.strip():
            raise ValueError("note must be non-empty when provided.")


@dataclass(frozen=True, slots=True)
class BenchmarkQualificationSpec:
    """Human-authored qualification inputs loaded from TOML."""

    candidate: BenchmarkCandidate
    assessments: tuple[CriterionAssessment, ...]
    rationale: str
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, BenchmarkCandidate):
            raise TypeError("candidate must be a BenchmarkCandidate.")
        if not self.rationale.strip():
            raise ValueError("rationale must be a non-empty string.")
        object.__setattr__(self, "assessments", tuple(self.assessments))
        object.__setattr__(self, "notes", tuple(self.notes))
        seen: set[QualificationCriterion] = set()
        for assessment in self.assessments:
            if not isinstance(assessment, CriterionAssessment):
                raise TypeError("assessments must contain CriterionAssessment values.")
            if assessment.criterion in seen:
                raise ValueError("assessments must not repeat criteria.")
            seen.add(assessment.criterion)
        missing = tuple(
            criterion for criterion in QualificationCriterion if criterion not in seen
        )
        if missing:
            raise ValueError("assessments must cover every QualificationCriterion.")
        for note in self.notes:
            if not note.strip():
                raise ValueError("notes must contain non-empty strings.")


@dataclass(frozen=True, slots=True)
class BenchmarkQualificationSummary:
    """Structured output from one benchmark-qualification pass."""

    candidate: BenchmarkCandidate
    topology_style: str
    readiness: QualificationReadiness
    decision: QualificationDecision
    assessments: tuple[CriterionAssessment, ...]
    smoke_testable_now: bool
    available_modules: tuple[str, ...] = field(default_factory=tuple)
    missing_modules: tuple[str, ...] = field(default_factory=tuple)
    integration_work_remaining: tuple[str, ...] = field(default_factory=tuple)
    rationale: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, BenchmarkCandidate):
            raise TypeError("candidate must be a BenchmarkCandidate.")
        if not self.topology_style.strip():
            raise ValueError("topology_style must be a non-empty string.")
        if not isinstance(self.readiness, QualificationReadiness):
            raise TypeError("readiness must be a QualificationReadiness.")
        if not isinstance(self.decision, QualificationDecision):
            raise TypeError("decision must be a QualificationDecision.")
        if not self.rationale.strip():
            raise ValueError("rationale must be a non-empty string.")
        object.__setattr__(self, "assessments", tuple(self.assessments))
        object.__setattr__(self, "available_modules", tuple(self.available_modules))
        object.__setattr__(self, "missing_modules", tuple(self.missing_modules))
        object.__setattr__(
            self,
            "integration_work_remaining",
            tuple(self.integration_work_remaining),
        )
        object.__setattr__(self, "notes", tuple(self.notes))
        for assessment in self.assessments:
            if not isinstance(assessment, CriterionAssessment):
                raise TypeError("assessments must contain CriterionAssessment values.")
        for value in (
            self.available_modules
            + self.missing_modules
            + self.integration_work_remaining
            + self.notes
        ):
            if not value.strip():
                raise ValueError("summary tuple fields must contain non-empty strings.")

    def level_for(self, criterion: QualificationCriterion) -> QualificationLevel:
        """Return the recorded level for one criterion."""

        for assessment in self.assessments:
            if assessment.criterion is criterion:
                return assessment.level
        raise KeyError(f"Missing assessment for {criterion.value}.")

    def to_record(self) -> dict[str, object]:
        """Return a compact JSON-serializable record."""

        return {
            "candidate": self.candidate.value,
            "topology_style": self.topology_style,
            "readiness": self.readiness.value,
            "decision": self.decision.value,
            "smoke_testable_now": self.smoke_testable_now,
            "available_modules": list(self.available_modules),
            "missing_modules": list(self.missing_modules),
            "criteria": {
                assessment.criterion.value: assessment.level.value
                for assessment in self.assessments
            },
            "integration_work_remaining": list(self.integration_work_remaining),
            "rationale": self.rationale,
            "notes": list(self.notes),
        }


def load_qualification_spec(path: str | Path) -> BenchmarkQualificationSpec:
    """Load one benchmark-qualification TOML document."""

    document = _load_toml_document(path)
    qualification_table = _require_table(document, "qualification")
    return BenchmarkQualificationSpec(
        candidate=_parse_enum(
            BenchmarkCandidate,
            _require_string(qualification_table, "candidate", "qualification"),
            "qualification.candidate",
        ),
        assessments=tuple(
            CriterionAssessment(
                criterion=criterion,
                level=_parse_enum(
                    QualificationLevel,
                    _require_string(qualification_table, criterion.value, "qualification"),
                    f"qualification.{criterion.value}",
                ),
            )
            for criterion in QualificationCriterion
        ),
        rationale=_require_string(qualification_table, "rationale", "qualification"),
        notes=tuple(_optional_string_list(qualification_table, "notes", "qualification")),
    )


def derive_readiness(
    smoke_testable_now: bool,
    available_modules: tuple[str, ...],
) -> QualificationReadiness:
    """Classify readiness from adapter-availability signals."""

    if smoke_testable_now:
        return QualificationReadiness.IMMEDIATELY_SMOKE_TESTABLE
    if available_modules:
        return QualificationReadiness.PARTIALLY_INTEGRABLE
    return QualificationReadiness.DEFERRED_MISSING_EXTERNAL_INTEGRATION


def recommend_decision(
    readiness: QualificationReadiness,
    assessments: tuple[CriterionAssessment, ...],
) -> QualificationDecision:
    """Recommend keep, defer, or drop from compact repository-fit judgments."""

    levels = {assessment.criterion: assessment.level for assessment in assessments}
    topology = levels[QualificationCriterion.TOPOLOGY_FIT]
    optimizer = levels[QualificationCriterion.OPTIMIZER_BOUNDARY_FIT]
    paper_value = levels[QualificationCriterion.EXPECTED_PAPER_VALUE]
    effort = levels[QualificationCriterion.IMPLEMENTATION_EFFORT]

    if paper_value is QualificationLevel.LOW:
        return QualificationDecision.DROP
    if topology is QualificationLevel.LOW or optimizer is QualificationLevel.LOW:
        return QualificationDecision.DROP
    if (
        readiness is QualificationReadiness.DEFERRED_MISSING_EXTERNAL_INTEGRATION
        and topology is QualificationLevel.UNVERIFIED
        and effort is QualificationLevel.HIGH
    ):
        return QualificationDecision.DROP
    if (
        readiness is QualificationReadiness.IMMEDIATELY_SMOKE_TESTABLE
        and topology is QualificationLevel.HIGH
        and paper_value in {QualificationLevel.HIGH, QualificationLevel.MEDIUM}
    ):
        return QualificationDecision.KEEP
    return QualificationDecision.DEFER


def build_qualification_summary(
    spec: BenchmarkQualificationSpec,
    topology_style: str,
    smoke_testable_now: bool,
    available_modules: tuple[str, ...],
    missing_modules: tuple[str, ...],
    integration_work_remaining: tuple[str, ...],
    adapter_notes: tuple[str, ...] = (),
) -> BenchmarkQualificationSummary:
    """Build a structured qualification summary from config and adapter state."""

    readiness = derive_readiness(
        smoke_testable_now=smoke_testable_now,
        available_modules=available_modules,
    )
    decision = recommend_decision(readiness=readiness, assessments=spec.assessments)
    return BenchmarkQualificationSummary(
        candidate=spec.candidate,
        topology_style=topology_style,
        readiness=readiness,
        decision=decision,
        assessments=spec.assessments,
        smoke_testable_now=smoke_testable_now,
        available_modules=available_modules,
        missing_modules=missing_modules,
        integration_work_remaining=integration_work_remaining,
        rationale=spec.rationale,
        notes=spec.notes + tuple(adapter_notes),
    )


def _load_toml_document(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        with path.open("rb") as handle:
            document = tomllib.load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Qualification config file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Invalid TOML in {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"Qualification config {path} did not parse into a TOML table.")
    return document


def _require_table(document: dict[str, Any], key: str) -> dict[str, Any]:
    value = document.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing required table [{key}].")
    return value


def _require_string(document: dict[str, Any], key: str, location: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}.{key} must be a non-empty string.")
    return value


def _optional_string_list(
    document: dict[str, Any],
    key: str,
    location: str,
) -> list[str]:
    if key not in document:
        return []
    value = document[key]
    if not isinstance(value, list):
        raise ValueError(f"{location}.{key} must be a list of strings.")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{location}.{key} must contain non-empty strings.")
        result.append(item)
    return result


def _parse_enum(enum_type: type[Any], raw_value: str, location: str) -> Any:
    try:
        return enum_type(raw_value)
    except ValueError as exc:
        raise ValueError(f"Unsupported value for {location}: {raw_value!r}.") from exc


__all__ = [
    "BenchmarkCandidate",
    "BenchmarkQualificationSpec",
    "BenchmarkQualificationSummary",
    "CriterionAssessment",
    "QualificationCriterion",
    "QualificationDecision",
    "QualificationLevel",
    "QualificationReadiness",
    "build_qualification_summary",
    "derive_readiness",
    "load_qualification_spec",
    "recommend_decision",
]
