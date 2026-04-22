"""Explicit bounded update-strength ladder for scenario-update calibration."""

from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import Iterable

from meio.contracts import UpdateRequest, UpdateRequestType


class UpdateStrength(StrEnum):
    """Ordered bounded update-strength labels for the active serial path."""

    KEEP_CURRENT = "keep_current"
    REWEIGHT_SCENARIOS = "reweight_scenarios"
    WIDEN_UNCERTAINTY = "widen_uncertainty"
    SWITCH_DEMAND_REGIME_PLUS_WIDEN_UNCERTAINTY = (
        "switch_demand_regime_plus_widen_uncertainty"
    )
    SWITCH_LEADTIME_REGIME_PLUS_WIDEN_UNCERTAINTY = (
        "switch_leadtime_regime_plus_widen_uncertainty"
    )
    OTHER = "other"


class UpdateStrengthRank(IntEnum):
    """Explicit ordinal ranking for the bounded update ladder."""

    KEEP_CURRENT = 0
    REWEIGHT_SCENARIOS = 1
    WIDEN_UNCERTAINTY = 2
    SWITCH_DEMAND_REGIME_PLUS_WIDEN_UNCERTAINTY = 3
    SWITCH_LEADTIME_REGIME_PLUS_WIDEN_UNCERTAINTY = 3
    OTHER = 99


ORDERED_UPDATE_LADDER: tuple[UpdateStrength, ...] = (
    UpdateStrength.KEEP_CURRENT,
    UpdateStrength.REWEIGHT_SCENARIOS,
    UpdateStrength.WIDEN_UNCERTAINTY,
    UpdateStrength.SWITCH_DEMAND_REGIME_PLUS_WIDEN_UNCERTAINTY,
)

_UPDATE_TYPES_BY_STRENGTH: dict[UpdateStrength, tuple[UpdateRequestType, ...]] = {
    UpdateStrength.KEEP_CURRENT: (UpdateRequestType.KEEP_CURRENT,),
    UpdateStrength.REWEIGHT_SCENARIOS: (UpdateRequestType.REWEIGHT_SCENARIOS,),
    UpdateStrength.WIDEN_UNCERTAINTY: (UpdateRequestType.WIDEN_UNCERTAINTY,),
    UpdateStrength.SWITCH_DEMAND_REGIME_PLUS_WIDEN_UNCERTAINTY: (
        UpdateRequestType.SWITCH_DEMAND_REGIME,
        UpdateRequestType.WIDEN_UNCERTAINTY,
    ),
    UpdateStrength.SWITCH_LEADTIME_REGIME_PLUS_WIDEN_UNCERTAINTY: (
        UpdateRequestType.SWITCH_LEADTIME_REGIME,
        UpdateRequestType.WIDEN_UNCERTAINTY,
    ),
}
_UPDATE_TYPE_ORDER = {
    UpdateRequestType.KEEP_CURRENT: 0,
    UpdateRequestType.REWEIGHT_SCENARIOS: 1,
    UpdateRequestType.WIDEN_UNCERTAINTY: 2,
    UpdateRequestType.SWITCH_DEMAND_REGIME: 3,
    UpdateRequestType.SWITCH_LEADTIME_REGIME: 4,
}


def normalize_update_types(
    update_types: Iterable[UpdateRequestType],
) -> tuple[UpdateRequestType, ...]:
    """Return a stable tuple representation of bounded update-request types."""

    return tuple(
        sorted(
            tuple(update_types),
            key=lambda update_type: _UPDATE_TYPE_ORDER[update_type],
        )
    )


_STRENGTH_BY_UPDATE_TYPES = {
    normalize_update_types(value): key
    for key, value in _UPDATE_TYPES_BY_STRENGTH.items()
}


def infer_update_strength(
    update_types: Iterable[UpdateRequestType],
) -> UpdateStrength:
    """Infer the explicit ladder rung for a bounded update selection."""

    normalized = normalize_update_types(update_types)
    if not normalized:
        return UpdateStrength.KEEP_CURRENT
    return _STRENGTH_BY_UPDATE_TYPES.get(normalized, UpdateStrength.OTHER)


def infer_update_strength_from_requests(
    update_requests: Iterable[UpdateRequest],
) -> UpdateStrength:
    """Infer update strength from repo-native bounded update requests."""

    return infer_update_strength(
        update_request.request_type for update_request in update_requests
    )


def update_types_for_strength(
    strength: UpdateStrength,
) -> tuple[UpdateRequestType, ...]:
    """Return the repo-native bounded update types for one ladder rung."""

    if strength not in _UPDATE_TYPES_BY_STRENGTH:
        raise ValueError(f"Unsupported bounded update strength: {strength.value}.")
    return _UPDATE_TYPES_BY_STRENGTH[strength]


def build_update_requests_for_strength(
    strength: UpdateStrength,
    *,
    target: str = "serial_scenarios",
    notes: str = "update_strength_calibration",
) -> tuple[UpdateRequest, ...]:
    """Build repo-native bounded update requests for one ladder rung."""

    return tuple(
        UpdateRequest(
            request_type=update_type,
            target=target,
            notes=notes,
        )
        for update_type in update_types_for_strength(strength)
    )


def update_strength_rank(strength: UpdateStrength) -> int:
    """Return the explicit ordinal rank for one bounded update strength."""

    return UpdateStrengthRank[strength.name]


def is_strong_demand_escalation(strength: UpdateStrength) -> bool:
    """Return whether the bounded update is the strongest demand-shift rung."""

    return strength is UpdateStrength.SWITCH_DEMAND_REGIME_PLUS_WIDEN_UNCERTAINTY


__all__ = [
    "ORDERED_UPDATE_LADDER",
    "UpdateStrength",
    "build_update_requests_for_strength",
    "infer_update_strength",
    "infer_update_strength_from_requests",
    "is_strong_demand_escalation",
    "normalize_update_types",
    "update_strength_rank",
    "update_types_for_strength",
]
