"""Typed runtime evidence containers for the first serial smoke path."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from meio.contracts import RegimeLabel

if TYPE_CHECKING:
    from meio.data.external_evidence import ExternalEvidenceBatch


def _coerce_non_negative_series(
    values: tuple[float, ...],
    field_name: str,
    *,
    allow_empty: bool,
) -> tuple[float, ...]:
    if not values:
        if allow_empty:
            return ()
        raise ValueError(f"{field_name} must not be empty.")
    result: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{field_name} must contain numeric values.")
        if float(value) < 0.0:
            raise ValueError(f"{field_name} must contain non-negative values.")
        result.append(float(value))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class DemandEvidence:
    """Compact demand evidence for bounded runtime decisions."""

    history: tuple[float, ...]
    latest_realization: tuple[float, ...] = field(default_factory=tuple)
    stage_index: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "history",
            _coerce_non_negative_series(self.history, "history", allow_empty=False),
        )
        object.__setattr__(
            self,
            "latest_realization",
            _coerce_non_negative_series(
                self.latest_realization,
                "latest_realization",
                allow_empty=True,
            ),
        )
        if self.stage_index is not None and self.stage_index <= 0:
            raise ValueError("stage_index must be positive when provided.")

    @property
    def latest_value(self) -> float:
        """Return the most recent scalar demand value available to the runtime."""

        if self.latest_realization:
            return self.latest_realization[-1]
        return self.history[-1]


@dataclass(frozen=True, slots=True)
class LeadTimeEvidence:
    """Compact lead-time evidence for bounded runtime decisions."""

    history: tuple[float, ...]
    latest_realization: tuple[float, ...] = field(default_factory=tuple)
    upstream_stage_index: int | None = None
    downstream_stage_index: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "history",
            _coerce_non_negative_series(self.history, "history", allow_empty=False),
        )
        object.__setattr__(
            self,
            "latest_realization",
            _coerce_non_negative_series(
                self.latest_realization,
                "latest_realization",
                allow_empty=True,
            ),
        )
        if self.upstream_stage_index is not None and self.upstream_stage_index <= 0:
            raise ValueError("upstream_stage_index must be positive when provided.")
        if self.downstream_stage_index is not None and self.downstream_stage_index <= 0:
            raise ValueError("downstream_stage_index must be positive when provided.")

    @property
    def latest_value(self) -> float:
        """Return the most recent scalar lead-time value available to the runtime."""

        if self.latest_realization:
            return self.latest_realization[-1]
        return self.history[-1]


@dataclass(frozen=True, slots=True)
class RuntimeEvidence:
    """Combined typed evidence envelope for the bounded smoke-path runtime."""

    time_index: int
    demand: DemandEvidence
    leadtime: LeadTimeEvidence
    scenario_families: tuple[RegimeLabel, ...]
    demand_baseline_value: float | None = None
    leadtime_baseline_value: float | None = None
    external_evidence_batch: "ExternalEvidenceBatch | None" = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.time_index < 0:
            raise ValueError("time_index must be non-negative.")
        if not isinstance(self.demand, DemandEvidence):
            raise TypeError("demand must be a DemandEvidence.")
        if not isinstance(self.leadtime, LeadTimeEvidence):
            raise TypeError("leadtime must be a LeadTimeEvidence.")
        object.__setattr__(self, "scenario_families", tuple(self.scenario_families))
        object.__setattr__(self, "notes", tuple(self.notes))
        if (
            self.demand_baseline_value is not None
            and self.demand_baseline_value < 0.0
        ):
            raise ValueError("demand_baseline_value must be non-negative when provided.")
        if (
            self.leadtime_baseline_value is not None
            and self.leadtime_baseline_value <= 0.0
        ):
            raise ValueError("leadtime_baseline_value must be positive when provided.")
        if not self.scenario_families:
            raise ValueError("scenario_families must not be empty.")
        for family in self.scenario_families:
            if not isinstance(family, RegimeLabel):
                raise TypeError("scenario_families must contain RegimeLabel values.")
        if self.external_evidence_batch is not None:
            from meio.data.external_evidence import ExternalEvidenceBatch

            if not isinstance(self.external_evidence_batch, ExternalEvidenceBatch):
                raise TypeError(
                    "external_evidence_batch must be an ExternalEvidenceBatch when provided."
                )
        for note in self.notes:
            if not note.strip():
                raise ValueError("notes must contain non-empty strings.")


__all__ = [
    "DemandEvidence",
    "LeadTimeEvidence",
    "RuntimeEvidence",
]
