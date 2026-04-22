"""Typed scenario-update contracts for the first MEIO milestone."""

from __future__ import annotations

from dataclasses import dataclass, field

from meio.contracts import RegimeLabel, UpdateRequest, UpdateRequestType
from meio.forecasting.contracts import ForecastResult
from meio.leadtime.contracts import LeadTimeResult


@dataclass(frozen=True, slots=True)
class ExternalEvidenceFusionSummary:
    """Traceable external-evidence strengthening before and after capping."""

    proposed_demand_outlook: float
    proposed_leadtime_outlook: float
    proposed_safety_buffer_scale: float
    final_demand_outlook: float
    final_leadtime_outlook: float
    final_safety_buffer_scale: float
    corroboration_count: int = 0
    role_labels: tuple[str, ...] = field(default_factory=tuple)
    cap_applied: bool = False
    corroboration_gate_applied: bool = False
    corroborated_family_change_allowed: bool = False
    early_confirmation_gate_applied: bool = False
    family_change_blocked: bool = False
    proposed_update_request_types: tuple[UpdateRequestType, ...] = field(default_factory=tuple)
    final_update_request_types: tuple[UpdateRequestType, ...] = field(default_factory=tuple)
    reason: str | None = None
    corroboration_gate_reason: str | None = None
    early_confirmation_gate_reason: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "proposed_demand_outlook",
            "proposed_leadtime_outlook",
            "proposed_safety_buffer_scale",
            "final_demand_outlook",
            "final_leadtime_outlook",
            "final_safety_buffer_scale",
        ):
            value = getattr(self, field_name)
            if field_name.endswith("leadtime_outlook") and value <= 0.0:
                raise ValueError(f"{field_name} must be positive.")
            if not field_name.endswith("leadtime_outlook") and value < 0.0:
                raise ValueError(f"{field_name} must be non-negative.")
        object.__setattr__(
            self,
            "proposed_update_request_types",
            tuple(self.proposed_update_request_types),
        )
        object.__setattr__(self, "role_labels", tuple(self.role_labels))
        object.__setattr__(
            self,
            "final_update_request_types",
            tuple(self.final_update_request_types),
        )
        if self.corroboration_count < 0:
            raise ValueError("corroboration_count must be non-negative.")
        for role_label in self.role_labels:
            if not role_label.strip():
                raise ValueError("role_labels must contain non-empty strings.")
        for field_name in (
            "proposed_update_request_types",
            "final_update_request_types",
        ):
            for value in getattr(self, field_name):
                if not isinstance(value, UpdateRequestType):
                    raise TypeError(
                        f"{field_name} must contain UpdateRequestType values."
                    )
        if self.reason is not None and not self.reason.strip():
            raise ValueError("reason must be non-empty when provided.")
        if (
            self.corroboration_gate_reason is not None
            and not self.corroboration_gate_reason.strip()
        ):
            raise ValueError(
                "corroboration_gate_reason must be non-empty when provided."
            )
        if (
            self.early_confirmation_gate_reason is not None
            and not self.early_confirmation_gate_reason.strip()
        ):
            raise ValueError(
                "early_confirmation_gate_reason must be non-empty when provided."
            )


@dataclass(frozen=True, slots=True)
class ScenarioAdjustmentSummary:
    """Compact planning adjustment summary produced by bounded scenario logic."""

    demand_outlook: float
    leadtime_outlook: float
    safety_buffer_scale: float = 1.0
    external_evidence_fusion_summary: ExternalEvidenceFusionSummary | None = None

    def __post_init__(self) -> None:
        if self.demand_outlook < 0.0:
            raise ValueError("demand_outlook must be non-negative.")
        if self.leadtime_outlook <= 0.0:
            raise ValueError("leadtime_outlook must be positive.")
        if self.safety_buffer_scale <= 0.0:
            raise ValueError("safety_buffer_scale must be positive.")
        if self.external_evidence_fusion_summary is not None and not isinstance(
            self.external_evidence_fusion_summary,
            ExternalEvidenceFusionSummary,
        ):
            raise TypeError(
                "external_evidence_fusion_summary must be an "
                "ExternalEvidenceFusionSummary when provided."
            )


@dataclass(frozen=True, slots=True)
class ScenarioSummary:
    """Compact scenario representation for the first serial benchmark path."""

    scenario_id: str
    regime_label: RegimeLabel
    weight: float = 1.0
    demand_scale: float = 1.0
    leadtime_scale: float = 1.0

    def __post_init__(self) -> None:
        if not self.scenario_id.strip():
            raise ValueError("scenario_id must be a non-empty string.")
        if not isinstance(self.regime_label, RegimeLabel):
            raise TypeError("regime_label must be a RegimeLabel.")
        if self.weight < 0.0:
            raise ValueError("weight must be non-negative.")
        if self.demand_scale <= 0.0:
            raise ValueError("demand_scale must be positive.")
        if self.leadtime_scale <= 0.0:
            raise ValueError("leadtime_scale must be positive.")


@dataclass(frozen=True, slots=True)
class ScenarioUpdateRequest:
    """Bounded scenario-update request aligned with agent update requests."""

    update_requests: tuple[UpdateRequest, ...]
    current_regime: RegimeLabel = RegimeLabel.NORMAL
    current_scenarios: tuple[ScenarioSummary, ...] = field(default_factory=tuple)
    forecast_result: ForecastResult | None = None
    leadtime_result: LeadTimeResult | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.current_regime, RegimeLabel):
            raise TypeError("current_regime must be a RegimeLabel.")
        object.__setattr__(self, "update_requests", tuple(self.update_requests))
        object.__setattr__(self, "current_scenarios", tuple(self.current_scenarios))
        if not self.update_requests:
            raise ValueError("update_requests must not be empty.")
        request_types = tuple(request.request_type for request in self.update_requests)
        if (
            UpdateRequestType.KEEP_CURRENT in request_types
            and len(request_types) > 1
        ):
            raise ValueError("keep_current must not be combined with other update requests.")
        for request in self.update_requests:
            if not isinstance(request, UpdateRequest):
                raise TypeError("update_requests must contain UpdateRequest values.")
        for scenario in self.current_scenarios:
            if not isinstance(scenario, ScenarioSummary):
                raise TypeError("current_scenarios must contain ScenarioSummary values.")
        if self.forecast_result is not None and not isinstance(self.forecast_result, ForecastResult):
            raise TypeError("forecast_result must be a ForecastResult when provided.")
        if self.leadtime_result is not None and not isinstance(self.leadtime_result, LeadTimeResult):
            raise TypeError("leadtime_result must be a LeadTimeResult when provided.")


@dataclass(frozen=True, slots=True)
class ScenarioUpdateResult:
    """Bounded scenario-update output for downstream optimization."""

    scenarios: tuple[ScenarioSummary, ...]
    applied_update_types: tuple[UpdateRequestType, ...]
    adjustment: ScenarioAdjustmentSummary
    request_replan: bool = False
    provenance: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenarios", tuple(self.scenarios))
        object.__setattr__(self, "applied_update_types", tuple(self.applied_update_types))
        if not self.scenarios:
            raise ValueError("scenarios must not be empty.")
        if not self.applied_update_types:
            raise ValueError("applied_update_types must not be empty.")
        if (
            UpdateRequestType.KEEP_CURRENT in self.applied_update_types
            and len(self.applied_update_types) > 1
        ):
            raise ValueError("keep_current must not be combined with other applied update types.")
        for scenario in self.scenarios:
            if not isinstance(scenario, ScenarioSummary):
                raise TypeError("scenarios must contain ScenarioSummary values.")
        for update_type in self.applied_update_types:
            if not isinstance(update_type, UpdateRequestType):
                raise TypeError("applied_update_types must contain UpdateRequestType values.")
        if not isinstance(self.adjustment, ScenarioAdjustmentSummary):
            raise TypeError("adjustment must be a ScenarioAdjustmentSummary.")


__all__ = [
    "ExternalEvidenceFusionSummary",
    "ScenarioAdjustmentSummary",
    "ScenarioSummary",
    "ScenarioUpdateRequest",
    "ScenarioUpdateResult",
]
