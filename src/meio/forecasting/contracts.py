"""Typed forecasting contracts for the first MEIO milestone."""

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
        result.append(float(value))
    return tuple(result)


def _coerce_non_negative_series(values: tuple[float, ...], field_name: str) -> tuple[float, ...]:
    result = _coerce_series(values, field_name)
    for value in result:
        if value < 0.0:
            raise ValueError(f"{field_name} must contain non-negative values.")
    return result


class ForecastUncertaintyFormat(StrEnum):
    """Supported compact uncertainty summary formats for forecasts."""

    POINT_ONLY = "point_only"
    LOCATION_SCALE = "location_scale"


@dataclass(frozen=True, slots=True)
class ForecastRequest:
    """Structured numeric input for a bounded forecasting tool."""

    demand_history: tuple[float, ...]
    horizon: int
    time_index: int = 0
    stage_index: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "demand_history",
            _coerce_non_negative_series(self.demand_history, "demand_history"),
        )
        if self.horizon <= 0:
            raise ValueError("horizon must be positive.")
        if self.time_index < 0:
            raise ValueError("time_index must be non-negative.")
        if self.stage_index is not None and self.stage_index <= 0:
            raise ValueError("stage_index must be positive when provided.")


@dataclass(frozen=True, slots=True)
class ForecastResult:
    """Bounded forecast output with optional uncertainty summary."""

    horizon: int
    point_forecast: tuple[float, ...]
    uncertainty_scale: tuple[float, ...] = field(default_factory=tuple)
    uncertainty_format: ForecastUncertaintyFormat = ForecastUncertaintyFormat.POINT_ONLY
    provenance: str = ""
    metadata: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive.")
        point_forecast = _coerce_non_negative_series(self.point_forecast, "point_forecast")
        if len(point_forecast) != self.horizon:
            raise ValueError("point_forecast must have exactly horizon entries.")
        object.__setattr__(self, "point_forecast", point_forecast)
        object.__setattr__(self, "metadata", tuple(self.metadata))
        if not isinstance(self.uncertainty_format, ForecastUncertaintyFormat):
            raise TypeError("uncertainty_format must be a ForecastUncertaintyFormat.")
        if self.uncertainty_scale:
            uncertainty_scale = _coerce_non_negative_series(
                self.uncertainty_scale,
                "uncertainty_scale",
            )
            if len(uncertainty_scale) != self.horizon:
                raise ValueError("uncertainty_scale must have exactly horizon entries.")
            object.__setattr__(self, "uncertainty_scale", uncertainty_scale)
        elif self.uncertainty_format is not ForecastUncertaintyFormat.POINT_ONLY:
            raise ValueError(
                "uncertainty_scale must be provided when uncertainty_format is not point_only."
            )
        for item in self.metadata:
            if not item.strip():
                raise ValueError("metadata must contain non-empty strings.")


__all__ = [
    "ForecastRequest",
    "ForecastResult",
    "ForecastUncertaintyFormat",
]
