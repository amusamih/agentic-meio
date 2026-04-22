"""Typed lead-time contracts for the first MEIO milestone."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


def _coerce_series(values: tuple[float, ...], field_name: str) -> tuple[float, ...]:
    if not values:
        raise ValueError(f"{field_name} must not be empty.")
    result: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{field_name} must contain numeric values.")
        if float(value) < 0.0:
            raise ValueError(f"{field_name} must contain non-negative values.")
        result.append(float(value))
    return tuple(result)


class LeadTimeUncertaintyFormat(StrEnum):
    """Supported compact uncertainty summary formats for lead times."""

    POINT_ONLY = "point_only"
    LOCATION_SCALE = "location_scale"


@dataclass(frozen=True, slots=True)
class LeadTimeRequest:
    """Structured numeric input for bounded lead-time estimation."""

    observed_lead_times: tuple[float, ...]
    horizon: int
    time_index: int = 0
    upstream_stage_index: int | None = None
    downstream_stage_index: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observed_lead_times",
            _coerce_series(self.observed_lead_times, "observed_lead_times"),
        )
        if self.horizon <= 0:
            raise ValueError("horizon must be positive.")
        if self.time_index < 0:
            raise ValueError("time_index must be non-negative.")
        if self.upstream_stage_index is not None and self.upstream_stage_index <= 0:
            raise ValueError("upstream_stage_index must be positive when provided.")
        if self.downstream_stage_index is not None and self.downstream_stage_index <= 0:
            raise ValueError("downstream_stage_index must be positive when provided.")


@dataclass(frozen=True, slots=True)
class LeadTimeResult:
    """Bounded lead-time output with optional uncertainty summary."""

    horizon: int
    expected_lead_time: tuple[float, ...]
    uncertainty_scale: tuple[float, ...] = field(default_factory=tuple)
    uncertainty_format: LeadTimeUncertaintyFormat = LeadTimeUncertaintyFormat.POINT_ONLY
    provenance: str = ""
    metadata: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive.")
        expected_lead_time = _coerce_series(self.expected_lead_time, "expected_lead_time")
        if len(expected_lead_time) != self.horizon:
            raise ValueError("expected_lead_time must have exactly horizon entries.")
        object.__setattr__(self, "expected_lead_time", expected_lead_time)
        object.__setattr__(self, "metadata", tuple(self.metadata))
        if not isinstance(self.uncertainty_format, LeadTimeUncertaintyFormat):
            raise TypeError("uncertainty_format must be a LeadTimeUncertaintyFormat.")
        if self.uncertainty_scale:
            uncertainty_scale = _coerce_series(self.uncertainty_scale, "uncertainty_scale")
            if len(uncertainty_scale) != self.horizon:
                raise ValueError("uncertainty_scale must have exactly horizon entries.")
            object.__setattr__(self, "uncertainty_scale", uncertainty_scale)
        elif self.uncertainty_format is not LeadTimeUncertaintyFormat.POINT_ONLY:
            raise ValueError(
                "uncertainty_scale must be provided when uncertainty_format is not point_only."
            )
        for item in self.metadata:
            if not item.strip():
                raise ValueError("metadata must contain non-empty strings.")


__all__ = [
    "LeadTimeRequest",
    "LeadTimeResult",
    "LeadTimeUncertaintyFormat",
]
