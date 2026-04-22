"""Stateful bounded update-strength calibration for the active serial path."""

from __future__ import annotations

from dataclasses import dataclass, replace

from meio.agents.llm_client import LLMClientContext
from meio.agents.structured_outputs import StructuredLLMDecision
from meio.contracts import RegimeLabel, UpdateRequestType
from meio.scenarios.update_policy import (
    UpdateStrength,
    build_update_requests_for_strength,
    infer_update_strength_from_requests,
    infer_update_strength,
    is_strong_demand_escalation,
    update_strength_rank,
)

_RECENT_STRESS_LOOKBACK = 4
_PIPELINE_ELEVATED_RATIO_THRESHOLD = 4.0
_BACKORDER_ELEVATED_RATIO_THRESHOLD = 1.0
_RECOVERY_HYSTERESIS_LOAD_THRESHOLD = 4.0
_MIN_ABS_DEMAND_WORSENING = 0.5
_BASELINE_DEMAND_WORSENING_FRACTION = 0.05
_MIN_ABS_LEADTIME_WORSENING = 0.25
_BASELINE_LEADTIME_WORSENING_FRACTION = 0.10


@dataclass(frozen=True, slots=True)
class UpdateCalibrationFeatures:
    """Compact stateful features used by the bounded calibration layer."""

    proposed_strength: UpdateStrength
    previous_strength: UpdateStrength | None = None
    strongest_recent_prior_strength: UpdateStrength | None = None
    demand_ratio_to_baseline: float | None = None
    leadtime_ratio_to_baseline: float | None = None
    pipeline_ratio_to_baseline: float | None = None
    backorder_ratio_to_baseline: float | None = None
    consecutive_stressed_periods: int = 0
    time_since_prior_stress: int | None = None
    carry_over_load_high: bool = False
    demand_materially_worsening: bool = False
    demand_materially_stronger_than_recent_stress: bool = False
    leadtime_materially_worsening: bool = False
    recent_prior_stress_detected: bool = False
    prior_intervention_loaded_state: bool = False
    recovery_load_high: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.proposed_strength, UpdateStrength):
            raise TypeError("proposed_strength must be an UpdateStrength.")
        if self.previous_strength is not None and not isinstance(
            self.previous_strength,
            UpdateStrength,
        ):
            raise TypeError("previous_strength must be an UpdateStrength when provided.")
        if self.strongest_recent_prior_strength is not None and not isinstance(
            self.strongest_recent_prior_strength,
            UpdateStrength,
        ):
            raise TypeError(
                "strongest_recent_prior_strength must be an UpdateStrength when provided."
            )
        if self.consecutive_stressed_periods < 0:
            raise ValueError("consecutive_stressed_periods must be non-negative.")
        if self.time_since_prior_stress is not None and self.time_since_prior_stress <= 0:
            raise ValueError("time_since_prior_stress must be positive when provided.")


@dataclass(frozen=True, slots=True)
class UpdateCalibrationResult:
    """LLM-proposed and final bounded update selection after calibration."""

    proposed_decision: StructuredLLMDecision
    final_decision: StructuredLLMDecision
    proposed_strength: UpdateStrength
    final_strength: UpdateStrength
    features: UpdateCalibrationFeatures
    calibration_applied: bool = False
    hysteresis_applied: bool = False
    repeated_stress_moderation_applied: bool = False
    relapse_moderation_applied: bool = False
    unresolved_stress_moderation_applied: bool = False
    calibration_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.proposed_decision, StructuredLLMDecision):
            raise TypeError("proposed_decision must be a StructuredLLMDecision.")
        if not isinstance(self.final_decision, StructuredLLMDecision):
            raise TypeError("final_decision must be a StructuredLLMDecision.")
        if not isinstance(self.proposed_strength, UpdateStrength):
            raise TypeError("proposed_strength must be an UpdateStrength.")
        if not isinstance(self.final_strength, UpdateStrength):
            raise TypeError("final_strength must be an UpdateStrength.")
        if not isinstance(self.features, UpdateCalibrationFeatures):
            raise TypeError("features must be an UpdateCalibrationFeatures.")
        if self.calibration_reason is not None and not self.calibration_reason.strip():
            raise ValueError("calibration_reason must be non-empty when provided.")
        if self.calibration_applied and self.calibration_reason is None:
            raise ValueError(
                "calibration_reason must be provided when calibration_applied is True."
            )


def calibrate_update_decision(
    proposed_decision: StructuredLLMDecision,
    *,
    context: LLMClientContext,
    recent_regime_history: tuple[RegimeLabel, ...] = (),
    recent_stress_reference_demand_value: float | None = None,
    recent_update_request_history: tuple[tuple[UpdateRequestType, ...], ...] = (),
) -> UpdateCalibrationResult:
    """Apply the explicit bounded calibration layer to one LLM proposal."""

    features = build_update_calibration_features(
        proposed_decision=proposed_decision,
        context=context,
        recent_regime_history=recent_regime_history,
        recent_stress_reference_demand_value=recent_stress_reference_demand_value,
        recent_update_request_history=recent_update_request_history,
    )
    proposed_strength = features.proposed_strength
    if _should_apply_recovery_hysteresis(
        proposed_decision=proposed_decision,
        context=context,
        features=features,
    ):
        return _build_calibrated_result(
            proposed_decision,
            features=features,
            final_strength=UpdateStrength.KEEP_CURRENT,
            calibration_reason="recovery_with_high_carry_over_load",
            hysteresis_applied=True,
            unresolved_stress_moderation_applied=True,
        )
    if _should_moderate_relapse_stress(
        proposed_decision=proposed_decision,
        features=features,
    ):
        return _build_calibrated_result(
            proposed_decision,
            features=features,
            final_strength=UpdateStrength.REWEIGHT_SCENARIOS,
            calibration_reason="relapse_with_unresolved_stress_load",
            hysteresis_applied=True,
            relapse_moderation_applied=True,
            unresolved_stress_moderation_applied=True,
        )
    if _should_moderate_repeated_stress(
        proposed_decision=proposed_decision,
        context=context,
        features=features,
    ):
        return _build_calibrated_result(
            proposed_decision,
            features=features,
            final_strength=UpdateStrength.REWEIGHT_SCENARIOS,
            calibration_reason="repeated_stress_not_materially_worsening",
            hysteresis_applied=True,
            repeated_stress_moderation_applied=True,
        )
    return UpdateCalibrationResult(
        proposed_decision=proposed_decision,
        final_decision=proposed_decision,
        proposed_strength=proposed_strength,
        final_strength=proposed_strength,
        features=features,
    )


def build_update_calibration_features(
    *,
    proposed_decision: StructuredLLMDecision,
    context: LLMClientContext,
    recent_regime_history: tuple[RegimeLabel, ...],
    recent_stress_reference_demand_value: float | None,
    recent_update_request_history: tuple[tuple[UpdateRequestType, ...], ...],
) -> UpdateCalibrationFeatures:
    """Build the explicit stateful feature bundle for bounded calibration."""

    previous_strength = None
    if recent_update_request_history:
        previous_strength = infer_update_strength(recent_update_request_history[-1])
    strongest_recent_prior_strength = _strongest_recent_prior_strength(
        recent_update_request_history
    )
    carry_over_load_high = _carry_over_load_high(context)
    return UpdateCalibrationFeatures(
        proposed_strength=infer_update_strength_from_requests(
            proposed_decision.update_requests
        ),
        previous_strength=previous_strength,
        strongest_recent_prior_strength=strongest_recent_prior_strength,
        demand_ratio_to_baseline=context.demand_ratio_to_baseline,
        leadtime_ratio_to_baseline=context.leadtime_ratio_to_baseline,
        pipeline_ratio_to_baseline=context.pipeline_ratio_to_baseline,
        backorder_ratio_to_baseline=context.backorder_ratio_to_baseline,
        consecutive_stressed_periods=_consecutive_stressed_periods(recent_regime_history),
        time_since_prior_stress=_time_since_prior_stress(recent_regime_history),
        carry_over_load_high=carry_over_load_high,
        demand_materially_worsening=_demand_materially_worsening(
            context=context,
            recent_stress_reference_demand_value=recent_stress_reference_demand_value,
        ),
        demand_materially_stronger_than_recent_stress=(
            _demand_materially_stronger_than_recent_stress(
                context=context,
                recent_stress_reference_demand_value=recent_stress_reference_demand_value,
            )
        ),
        leadtime_materially_worsening=_leadtime_materially_worsening(context),
        recent_prior_stress_detected=_recent_prior_stress_detected(recent_regime_history),
        prior_intervention_loaded_state=(
            carry_over_load_high
            and strongest_recent_prior_strength is not None
            and update_strength_rank(strongest_recent_prior_strength)
            >= update_strength_rank(UpdateStrength.REWEIGHT_SCENARIOS)
        ),
        recovery_load_high=(
            context.recovery_with_high_pipeline_load is True
            or context.recovery_with_high_backorder_load is True
        ),
    )


def _build_calibrated_result(
    proposed_decision: StructuredLLMDecision,
    *,
    features: UpdateCalibrationFeatures,
    final_strength: UpdateStrength,
    calibration_reason: str,
    hysteresis_applied: bool = False,
    repeated_stress_moderation_applied: bool = False,
    relapse_moderation_applied: bool = False,
    unresolved_stress_moderation_applied: bool = False,
) -> UpdateCalibrationResult:
    return UpdateCalibrationResult(
        proposed_decision=proposed_decision,
        final_decision=replace(
            proposed_decision,
            update_requests=build_update_requests_for_strength(
                final_strength,
                notes=f"update_strength_calibration:{calibration_reason}",
            ),
        ),
        proposed_strength=features.proposed_strength,
        final_strength=final_strength,
        features=features,
        calibration_applied=True,
        hysteresis_applied=hysteresis_applied,
        repeated_stress_moderation_applied=repeated_stress_moderation_applied,
        relapse_moderation_applied=relapse_moderation_applied,
        unresolved_stress_moderation_applied=unresolved_stress_moderation_applied,
        calibration_reason=calibration_reason,
    )


def _should_moderate_repeated_stress(
    *,
    proposed_decision: StructuredLLMDecision,
    context: LLMClientContext,
    features: UpdateCalibrationFeatures,
) -> bool:
    if proposed_decision.regime_label is not RegimeLabel.DEMAND_REGIME_SHIFT:
        return False
    if context.repeated_stress_detected is not True:
        return False
    if not is_strong_demand_escalation(features.proposed_strength):
        return False
    if features.previous_strength is None:
        return False
    if update_strength_rank(features.previous_strength) < update_strength_rank(
        UpdateStrength.REWEIGHT_SCENARIOS
    ):
        return False
    if features.demand_materially_worsening or features.leadtime_materially_worsening:
        return False
    return True


def _should_moderate_relapse_stress(
    *,
    proposed_decision: StructuredLLMDecision,
    features: UpdateCalibrationFeatures,
) -> bool:
    if proposed_decision.regime_label is not RegimeLabel.DEMAND_REGIME_SHIFT:
        return False
    if not is_strong_demand_escalation(features.proposed_strength):
        return False
    if not features.recent_prior_stress_detected:
        return False
    if features.time_since_prior_stress is None or features.time_since_prior_stress <= 1:
        return False
    if not features.prior_intervention_loaded_state:
        return False
    if (
        features.demand_materially_stronger_than_recent_stress
        or features.leadtime_materially_worsening
    ):
        return False
    return True


def _should_apply_recovery_hysteresis(
    *,
    proposed_decision: StructuredLLMDecision,
    context: LLMClientContext,
    features: UpdateCalibrationFeatures,
) -> bool:
    if proposed_decision.regime_label is not RegimeLabel.RECOVERY:
        return False
    if features.proposed_strength is UpdateStrength.KEEP_CURRENT:
        return False
    if context.recovery_with_high_pipeline_load is not True and context.recovery_with_high_backorder_load is not True:
        return False
    if not features.prior_intervention_loaded_state:
        return False
    if features.leadtime_materially_worsening:
        return False
    if (
        context.pipeline_ratio_to_baseline is not None
        and context.pipeline_ratio_to_baseline < _RECOVERY_HYSTERESIS_LOAD_THRESHOLD
        and context.backorder_ratio_to_baseline is not None
        and context.backorder_ratio_to_baseline < _BACKORDER_ELEVATED_RATIO_THRESHOLD
    ):
        return False
    return True


def _consecutive_stressed_periods(
    recent_regime_history: tuple[RegimeLabel, ...],
) -> int:
    consecutive = 0
    for regime_label in reversed(recent_regime_history):
        if regime_label is not RegimeLabel.DEMAND_REGIME_SHIFT:
            break
        consecutive += 1
    return consecutive


def _time_since_prior_stress(
    recent_regime_history: tuple[RegimeLabel, ...],
) -> int | None:
    for index, regime_label in enumerate(reversed(recent_regime_history), start=1):
        if regime_label is RegimeLabel.DEMAND_REGIME_SHIFT:
            return index
    return None


def _recent_prior_stress_detected(
    recent_regime_history: tuple[RegimeLabel, ...],
) -> bool:
    if not recent_regime_history:
        return False
    return any(
        regime_label is RegimeLabel.DEMAND_REGIME_SHIFT
        for regime_label in recent_regime_history[-_RECENT_STRESS_LOOKBACK:]
    )


def _carry_over_load_high(context: LLMClientContext) -> bool:
    pipeline_elevated = (
        context.pipeline_ratio_to_baseline is not None
        and context.pipeline_ratio_to_baseline >= _PIPELINE_ELEVATED_RATIO_THRESHOLD
    )
    backlog_elevated = (
        context.backorder_ratio_to_baseline is not None
        and context.backorder_ratio_to_baseline >= _BACKORDER_ELEVATED_RATIO_THRESHOLD
    )
    return pipeline_elevated or backlog_elevated


def _strongest_recent_prior_strength(
    recent_update_request_history: tuple[tuple[UpdateRequestType, ...], ...],
) -> UpdateStrength | None:
    strengths = tuple(
        infer_update_strength(update_types)
        for update_types in recent_update_request_history
    )
    if not strengths:
        return None
    return max(strengths, key=update_strength_rank)


def _demand_materially_worsening(
    *,
    context: LLMClientContext,
    recent_stress_reference_demand_value: float | None,
) -> bool:
    threshold = _material_demand_worsening_threshold(context)
    if (
        context.demand_change_from_previous is not None
        and context.demand_change_from_previous > threshold
    ):
        return True
    if recent_stress_reference_demand_value is None:
        return False
    return (context.demand_value - recent_stress_reference_demand_value) > threshold


def _material_demand_worsening_threshold(context: LLMClientContext) -> float:
    baseline_component = 0.0
    if context.demand_baseline_value is not None:
        baseline_component = (
            context.demand_baseline_value * _BASELINE_DEMAND_WORSENING_FRACTION
        )
    return max(_MIN_ABS_DEMAND_WORSENING, baseline_component)


def _demand_materially_stronger_than_recent_stress(
    *,
    context: LLMClientContext,
    recent_stress_reference_demand_value: float | None,
) -> bool:
    if recent_stress_reference_demand_value is None:
        return False
    threshold = _material_demand_worsening_threshold(context)
    return (context.demand_value - recent_stress_reference_demand_value) > threshold


def _leadtime_materially_worsening(context: LLMClientContext) -> bool:
    threshold = _MIN_ABS_LEADTIME_WORSENING
    if context.leadtime_baseline_value is not None:
        threshold = max(
            threshold,
            context.leadtime_baseline_value
            * _BASELINE_LEADTIME_WORSENING_FRACTION,
        )
    if (
        context.leadtime_change_from_previous is not None
        and context.leadtime_change_from_previous > threshold
    ):
        return True
    if (
        context.leadtime_change_from_baseline is not None
        and context.leadtime_change_from_baseline > threshold
    ):
        return True
    return False


__all__ = [
    "UpdateCalibrationFeatures",
    "UpdateCalibrationResult",
    "build_update_calibration_features",
    "calibrate_update_decision",
]
