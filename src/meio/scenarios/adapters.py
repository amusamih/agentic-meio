"""Deterministic scenario-update adapters for the first smoke path."""

from __future__ import annotations

from dataclasses import dataclass

from meio.agents.external_evidence_tool import (
    ExternalEvidenceInterpretation,
    ExternalRiskDirection,
    ExternalUncertaintyPressure,
    apply_external_evidence_bias,
)
from meio.data.external_evidence import (
    ExternalEvidenceRecord,
    ExternalEvidenceStrengthTier,
    ExternalEvidenceTiming,
)
from meio.contracts import (
    AgentAssessment,
    BoundedTool,
    OperationalSubgoal,
    RegimeLabel,
    ToolClass,
    ToolInvocation,
    ToolResult,
    ToolSpec,
    ToolStatus,
    UpdateRequest,
    UpdateRequestType,
)
from meio.forecasting.contracts import ForecastResult
from meio.leadtime.contracts import LeadTimeResult
from meio.scenarios.contracts import (
    ExternalEvidenceFusionSummary,
    ScenarioAdjustmentSummary,
    ScenarioSummary,
    ScenarioUpdateRequest,
    ScenarioUpdateResult,
)

_DEFAULT_UPDATE_TYPES = {
    RegimeLabel.NORMAL: (UpdateRequestType.KEEP_CURRENT,),
    RegimeLabel.DEMAND_REGIME_SHIFT: (
        UpdateRequestType.REWEIGHT_SCENARIOS,
        UpdateRequestType.WIDEN_UNCERTAINTY,
    ),
    RegimeLabel.SUPPLY_DISRUPTION: (
        UpdateRequestType.SWITCH_LEADTIME_REGIME,
        UpdateRequestType.WIDEN_UNCERTAINTY,
    ),
    RegimeLabel.JOINT_DISRUPTION: (
        UpdateRequestType.REWEIGHT_SCENARIOS,
        UpdateRequestType.WIDEN_UNCERTAINTY,
    ),
    RegimeLabel.RECOVERY: (UpdateRequestType.REWEIGHT_SCENARIOS,),
}
_REGIME_BASE_SCALES = {
    RegimeLabel.NORMAL: (1.0, 1.0, 1.0),
    RegimeLabel.DEMAND_REGIME_SHIFT: (1.1, 1.0, 1.05),
    RegimeLabel.SUPPLY_DISRUPTION: (1.0, 1.15, 1.05),
    RegimeLabel.JOINT_DISRUPTION: (1.1, 1.15, 1.08),
    RegimeLabel.RECOVERY: (1.02, 1.0, 1.02),
}
_UPDATE_TYPE_EFFECTS = {
    UpdateRequestType.KEEP_CURRENT: (1.0, 1.0, 1.0),
    UpdateRequestType.REWEIGHT_SCENARIOS: (1.03, 1.0, 1.02),
    UpdateRequestType.SWITCH_DEMAND_REGIME: (1.12, 1.0, 1.05),
    UpdateRequestType.SWITCH_LEADTIME_REGIME: (1.0, 1.15, 1.05),
    UpdateRequestType.WIDEN_UNCERTAINTY: (1.0, 1.0, 1.10),
}
_STRONG_INTERNAL_CONFIDENCE_THRESHOLD = 0.8
_DEMAND_SHIFT_RATIO_THRESHOLD = 1.15
_NORMAL_RATIO_UPPER_BOUND = 1.05
_RECOVERY_RATIO_UPPER_BOUND = 1.08
_LEADTIME_RISK_RATIO_THRESHOLD = 1.1
_EARLY_EVIDENCE_CONFIDENCE_UPPER_BOUND = 0.8
_EARLY_LEADTIME_RATIO_UPPER_BOUND = 1.05
_MATERIAL_DEMAND_MOVE_RATIO_THRESHOLD = 1.08
_MATERIAL_LEADTIME_MOVE_RATIO_THRESHOLD = 1.08
_MIN_CORROBORATION_COUNT = 2

_LEADING_ROLE_LABELS = {"leading_demand_shift"}
_FALSE_ALARM_ROLE_LABELS = {"false_alarm", "false_alarm_or_rumor"}
_RELAPSE_ROLE_LABELS = {"relapse_demand_shift"}
_RECOVERY_ROLE_LABELS = {"recovery_with_load", "recovery_or_easing"}
_SUPPLY_ROLE_LABELS = {"supply_disruption"}


def _external_records(
    invocation: ToolInvocation,
) -> tuple[ExternalEvidenceRecord, ...]:
    if invocation.evidence is None or invocation.evidence.external_evidence_batch is None:
        return ()
    return invocation.evidence.external_evidence_batch.records


def _role_labels(
    records: tuple[ExternalEvidenceRecord, ...],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                record.role_label
                for record in records
                if record.role_label is not None
            }
        )
    )


def _internal_regime(
    invocation: ToolInvocation,
    request: ScenarioUpdateRequest,
) -> RegimeLabel:
    internal_regime = request.current_regime
    if invocation.system_state is not None:
        internal_regime = invocation.system_state.regime_label
    if invocation.observation is not None and invocation.observation.regime_label is not None:
        internal_regime = invocation.observation.regime_label
    return internal_regime


def _evidence_strength_rank(record: ExternalEvidenceRecord) -> int:
    if record.evidence_strength_tier is ExternalEvidenceStrengthTier.STRONG:
        return 3
    if record.evidence_strength_tier is ExternalEvidenceStrengthTier.MODERATE:
        return 2
    if record.evidence_strength_tier is ExternalEvidenceStrengthTier.WEAK:
        return 1
    return {
        "low": 1,
        "medium": 2,
        "high": 3,
    }[record.severity.value]


def _distinct_supporting_source_count(
    records: tuple[ExternalEvidenceRecord, ...],
) -> int:
    if not records:
        return 0
    role_label = next((record.role_label for record in records if record.role_label is not None), None)
    corroboration_group = next(
        (
            record.corroboration_group
            for record in records
            if record.corroboration_group is not None
        ),
        None,
    )
    return len(
        {
            record.source_id
            for record in records
            if (
                role_label is None
                or record.role_label == role_label
            )
            and (
                corroboration_group is None
                or record.corroboration_group == corroboration_group
            )
        }
    )


def _internal_signals_materially_moving(
    invocation: ToolInvocation,
) -> bool:
    if invocation.observation is None or invocation.evidence is None:
        return False
    demand_ratio = _ratio_or_none(
        invocation.observation.demand_realization[-1],
        invocation.evidence.demand_baseline_value,
    )
    leadtime_ratio = _ratio_or_none(
        invocation.observation.leadtime_realization[-1],
        invocation.evidence.leadtime_baseline_value,
    )
    return (
        demand_ratio is not None
        and demand_ratio >= _MATERIAL_DEMAND_MOVE_RATIO_THRESHOLD
    ) or (
        leadtime_ratio is not None
        and leadtime_ratio >= _MATERIAL_LEADTIME_MOVE_RATIO_THRESHOLD
    )


def _strong_external_article_present(
    records: tuple[ExternalEvidenceRecord, ...],
) -> bool:
    return any(
        _evidence_strength_rank(record) >= 3
        or (
            record.credibility is not None
            and record.credibility >= _STRONG_INTERNAL_CONFIDENCE_THRESHOLD
            and _evidence_strength_rank(record) >= 2
        )
        for record in records
    )


def _has_role_label(
    records: tuple[ExternalEvidenceRecord, ...],
    candidate_labels: set[str],
) -> bool:
    return any(record.role_label in candidate_labels for record in records)


def _extract_prior_output(
    invocation: ToolInvocation,
    key: str,
    expected_type: type[object],
) -> object:
    for result in reversed(invocation.prior_results):
        value = result.structured_output.get(key)
        if isinstance(value, expected_type):
            return value
    raise ValueError(f"Scenario adapter requires prior output {key!r}.")


def _fallback_forecast_from_observation(invocation: ToolInvocation) -> ForecastResult:
    if invocation.observation is None:
        raise ValueError("Scenario adapter requires an observation when forecast output is absent.")
    demand_value = invocation.observation.demand_realization[-1]
    return ForecastResult(
        horizon=1,
        point_forecast=(demand_value,),
        uncertainty_scale=(max(1.0, demand_value * 0.1),),
        provenance="scenario_tool_observation_fallback",
        metadata=("forecast_tool_ablated_or_unavailable",),
    )


def _fallback_leadtime_from_observation(invocation: ToolInvocation) -> LeadTimeResult:
    if invocation.observation is None:
        raise ValueError("Scenario adapter requires an observation when lead-time output is absent.")
    leadtime_value = invocation.observation.leadtime_realization[-1]
    return LeadTimeResult(
        horizon=1,
        expected_lead_time=(leadtime_value,),
        uncertainty_scale=(0.25,),
        provenance="scenario_tool_observation_fallback",
        metadata=("leadtime_tool_ablated_or_unavailable",),
    )


def _extract_external_evidence_interpretation(
    invocation: ToolInvocation,
) -> ExternalEvidenceInterpretation | None:
    for result in reversed(invocation.prior_results):
        value = result.structured_output.get("external_evidence_interpretation")
        if isinstance(value, ExternalEvidenceInterpretation):
            return value
    return None


def _ratio_or_none(value: float, baseline_value: float | None) -> float | None:
    if baseline_value is None or baseline_value <= 0.0:
        return None
    return value / baseline_value


def _current_update_types(
    request: ScenarioUpdateRequest,
) -> tuple[UpdateRequestType, ...]:
    return tuple(update_request.request_type for update_request in request.update_requests)


def _leadtime_contradiction_present(
    invocation: ToolInvocation,
    request: ScenarioUpdateRequest,
    interpretation: ExternalEvidenceInterpretation,
) -> bool:
    if interpretation.interpreted_risk_direction in {
        ExternalRiskDirection.LEADTIME_RISK,
        ExternalRiskDirection.MIXED_RISK,
    }:
        return True
    update_types = set(_current_update_types(request))
    if UpdateRequestType.SWITCH_LEADTIME_REGIME in update_types:
        return True
    if request.current_regime in {
        RegimeLabel.SUPPLY_DISRUPTION,
        RegimeLabel.JOINT_DISRUPTION,
    }:
        return True
    if invocation.observation is None or invocation.evidence is None:
        return False
    leadtime_ratio = _ratio_or_none(
        invocation.observation.leadtime_realization[-1],
        invocation.evidence.leadtime_baseline_value,
    )
    return leadtime_ratio is not None and leadtime_ratio >= _LEADTIME_RISK_RATIO_THRESHOLD


def _internal_signal_sufficient(
    invocation: ToolInvocation,
    request: ScenarioUpdateRequest,
    assessment: AgentAssessment | None,
) -> bool:
    if assessment is None or assessment.confidence < _STRONG_INTERNAL_CONFIDENCE_THRESHOLD:
        return False
    if invocation.observation is None or invocation.evidence is None:
        return False
    demand_ratio = _ratio_or_none(
        invocation.observation.demand_realization[-1],
        invocation.evidence.demand_baseline_value,
    )
    if demand_ratio is None:
        return False
    if assessment.regime_label is RegimeLabel.NORMAL:
        return (
            UpdateRequestType.KEEP_CURRENT in _current_update_types(request)
            and demand_ratio <= _NORMAL_RATIO_UPPER_BOUND
        )
    if assessment.regime_label is RegimeLabel.DEMAND_REGIME_SHIFT:
        return demand_ratio >= _DEMAND_SHIFT_RATIO_THRESHOLD
    if assessment.regime_label is RegimeLabel.RECOVERY:
        return (
            UpdateRequestType.REWEIGHT_SCENARIOS in _current_update_types(request)
            and demand_ratio <= _RECOVERY_RATIO_UPPER_BOUND
        )
    return False


def _evidence_is_confirming_or_mild(
    request: ScenarioUpdateRequest,
    assessment: AgentAssessment | None,
    interpretation: ExternalEvidenceInterpretation,
) -> bool:
    reference_regime = assessment.regime_label if assessment is not None else request.current_regime
    if interpretation.recommended_regime_hint is reference_regime:
        return True
    return interpretation.recommended_uncertainty_pressure in {
        ExternalUncertaintyPressure.LOW,
        ExternalUncertaintyPressure.MEDIUM,
    }


def _external_records_are_leading(
    invocation: ToolInvocation,
) -> bool:
    if invocation.evidence is None or invocation.evidence.external_evidence_batch is None:
        return False
    records = invocation.evidence.external_evidence_batch.records
    return bool(records) and all(
        record.timing is ExternalEvidenceTiming.LEADING for record in records
    )


def _internal_signals_still_weak(
    invocation: ToolInvocation,
) -> bool:
    if invocation.observation is None or invocation.evidence is None:
        return False
    demand_ratio = _ratio_or_none(
        invocation.observation.demand_realization[-1],
        invocation.evidence.demand_baseline_value,
    )
    leadtime_ratio = _ratio_or_none(
        invocation.observation.leadtime_realization[-1],
        invocation.evidence.leadtime_baseline_value,
    )
    return (
        (demand_ratio is None or demand_ratio <= _NORMAL_RATIO_UPPER_BOUND)
        and (
            leadtime_ratio is None
            or leadtime_ratio <= _EARLY_LEADTIME_RATIO_UPPER_BOUND
        )
    )


def _evidence_not_strong_enough_for_family_change(
    interpretation: ExternalEvidenceInterpretation,
) -> bool:
    return (
        interpretation.recommended_uncertainty_pressure
        in {
            ExternalUncertaintyPressure.LOW,
            ExternalUncertaintyPressure.MEDIUM,
        }
        and interpretation.confidence <= _EARLY_EVIDENCE_CONFIDENCE_UPPER_BOUND
    )


def _proposed_update_family_is_stronger_than_watchful(
    update_types: tuple[UpdateRequestType, ...],
) -> bool:
    mild_types = {
        UpdateRequestType.KEEP_CURRENT,
        UpdateRequestType.REWEIGHT_SCENARIOS,
    }
    return any(update_type not in mild_types for update_type in update_types)


def _apply_corroboration_gate(
    *,
    invocation: ToolInvocation,
    request: ScenarioUpdateRequest,
    interpretation: ExternalEvidenceInterpretation,
    applied_update_types: tuple[UpdateRequestType, ...],
) -> tuple[
    RegimeLabel,
    tuple[UpdateRequestType, ...],
    int,
    tuple[str, ...],
    bool,
    bool,
    bool,
    str | None,
]:
    records = _external_records(invocation)
    corroboration_count = _distinct_supporting_source_count(records)
    role_labels = _role_labels(records)
    if not records or not _proposed_update_family_is_stronger_than_watchful(applied_update_types):
        return (
            request.current_regime,
            applied_update_types,
            corroboration_count,
            role_labels,
            False,
            False,
            False,
            None,
        )
    if _has_role_label(records, _RELAPSE_ROLE_LABELS):
        return (
            request.current_regime,
            applied_update_types,
            corroboration_count,
            role_labels,
            False,
            False,
            False,
            None,
        )
    if _has_role_label(records, _RECOVERY_ROLE_LABELS):
        return (
            _internal_regime(invocation, request),
            (
                (UpdateRequestType.REWEIGHT_SCENARIOS,)
                if request.current_regime is RegimeLabel.RECOVERY
                else (UpdateRequestType.KEEP_CURRENT,)
            ),
            corroboration_count,
            role_labels,
            True,
            True,
            False,
            "recovery_external_evidence_prevents_unnecessary_escalation",
        )
    if _has_role_label(records, _FALSE_ALARM_ROLE_LABELS):
        if corroboration_count >= _MIN_CORROBORATION_COUNT and _internal_signals_materially_moving(
            invocation
        ):
            return (
                request.current_regime,
                applied_update_types,
                corroboration_count,
                role_labels,
                False,
                False,
                True,
                None,
            )
        return (
            _internal_regime(invocation, request),
            (UpdateRequestType.KEEP_CURRENT,),
            corroboration_count,
            role_labels,
            True,
            True,
            False,
            "uncorroborated_false_alarm_external_evidence",
        )
    if _has_role_label(records, _LEADING_ROLE_LABELS):
        corroborated = corroboration_count >= _MIN_CORROBORATION_COUNT
        strong_with_internal_movement = (
            _strong_external_article_present(records)
            and _internal_signals_materially_moving(invocation)
        )
        if corroborated or strong_with_internal_movement:
            return (
                request.current_regime,
                applied_update_types,
                corroboration_count,
                role_labels,
                False,
                False,
                True,
                None,
            )
        return (
            _internal_regime(invocation, request),
            (UpdateRequestType.KEEP_CURRENT,),
            corroboration_count,
            role_labels,
            True,
            True,
            False,
            "uncorroborated_leading_external_evidence",
        )
    return (
        request.current_regime,
        applied_update_types,
        corroboration_count,
        role_labels,
        False,
        False,
        False,
        None,
    )


def _apply_early_evidence_confirmation_gate(
    *,
    invocation: ToolInvocation,
    request: ScenarioUpdateRequest,
    effective_regime: RegimeLabel,
    interpretation: ExternalEvidenceInterpretation,
    applied_update_types: tuple[UpdateRequestType, ...],
    corroborated_family_change_allowed: bool,
) -> tuple[RegimeLabel, tuple[UpdateRequestType, ...], bool, bool, str | None]:
    if not _proposed_update_family_is_stronger_than_watchful(applied_update_types):
        return effective_regime, applied_update_types, False, False, None
    if corroborated_family_change_allowed:
        return effective_regime, applied_update_types, False, False, None
    if not _external_records_are_leading(invocation):
        return effective_regime, applied_update_types, False, False, None
    if not _internal_signals_still_weak(invocation):
        return effective_regime, applied_update_types, False, False, None
    if not _evidence_not_strong_enough_for_family_change(interpretation):
        return effective_regime, applied_update_types, False, False, None
    if _leadtime_contradiction_present(invocation, request, interpretation):
        return effective_regime, applied_update_types, False, False, None
    internal_regime = request.current_regime
    if invocation.system_state is not None:
        internal_regime = invocation.system_state.regime_label
    if invocation.observation is not None and invocation.observation.regime_label is not None:
        internal_regime = invocation.observation.regime_label
    return (
        internal_regime,
        (UpdateRequestType.KEEP_CURRENT,),
        True,
        True,
        "leading_external_evidence_before_internal_stress",
    )


def _route_external_bias_by_role(
    *,
    invocation: ToolInvocation,
    request: ScenarioUpdateRequest,
    interpretation: ExternalEvidenceInterpretation,
    demand_outlook: float,
    leadtime_outlook: float,
    safety_buffer_scale: float,
    proposed_demand_outlook: float,
    proposed_leadtime_outlook: float,
    proposed_buffer_scale: float,
) -> tuple[float, float, float]:
    records = _external_records(invocation)
    if not records:
        return proposed_demand_outlook, proposed_leadtime_outlook, proposed_buffer_scale
    if _has_role_label(records, _RECOVERY_ROLE_LABELS) and not _leadtime_contradiction_present(
        invocation,
        request,
        interpretation,
    ):
        return (
            min(proposed_demand_outlook, demand_outlook),
            min(proposed_leadtime_outlook, leadtime_outlook),
            min(proposed_buffer_scale, safety_buffer_scale),
        )
    if _has_role_label(records, _SUPPLY_ROLE_LABELS):
        return (
            demand_outlook,
            proposed_leadtime_outlook,
            proposed_buffer_scale,
        )
    return proposed_demand_outlook, proposed_leadtime_outlook, proposed_buffer_scale


def _resolve_external_evidence_fusion(
    *,
    invocation: ToolInvocation,
    request: ScenarioUpdateRequest,
    assessment: AgentAssessment | None,
    interpretation: ExternalEvidenceInterpretation,
    corroboration_count: int,
    role_labels: tuple[str, ...],
    proposed_update_types: tuple[UpdateRequestType, ...],
    final_update_types: tuple[UpdateRequestType, ...],
    corroboration_gate_applied: bool,
    corroborated_family_change_allowed: bool,
    corroboration_gate_reason: str | None,
    early_confirmation_gate_applied: bool,
    family_change_blocked: bool,
    early_confirmation_gate_reason: str | None,
    demand_outlook: float,
    leadtime_outlook: float,
    safety_buffer_scale: float,
) -> tuple[float, float, float, ExternalEvidenceFusionSummary]:
    proposed_demand_outlook, proposed_leadtime_outlook, proposed_buffer_scale = (
        apply_external_evidence_bias(
            interpretation,
            demand_outlook=demand_outlook,
            leadtime_outlook=leadtime_outlook,
            safety_buffer_scale=safety_buffer_scale,
        )
    )
    (
        proposed_demand_outlook,
        proposed_leadtime_outlook,
        proposed_buffer_scale,
    ) = _route_external_bias_by_role(
        invocation=invocation,
        request=request,
        interpretation=interpretation,
        demand_outlook=demand_outlook,
        leadtime_outlook=leadtime_outlook,
        safety_buffer_scale=safety_buffer_scale,
        proposed_demand_outlook=proposed_demand_outlook,
        proposed_leadtime_outlook=proposed_leadtime_outlook,
        proposed_buffer_scale=proposed_buffer_scale,
    )
    cap_applied = (
        _internal_signal_sufficient(invocation, request, assessment)
        and _evidence_is_confirming_or_mild(request, assessment, interpretation)
        and not _leadtime_contradiction_present(invocation, request, interpretation)
        and (
            proposed_demand_outlook > demand_outlook
            or proposed_buffer_scale > safety_buffer_scale
        )
    )
    corroboration_gate_blocks_strengthening = (
        corroboration_gate_applied and family_change_blocked
    )
    confirmation_gate_blocks_strengthening = (
        early_confirmation_gate_applied and family_change_blocked
    )
    final_demand_outlook = (
        demand_outlook
        if (
            cap_applied
            or corroboration_gate_blocks_strengthening
            or confirmation_gate_blocks_strengthening
        )
        else proposed_demand_outlook
    )
    final_leadtime_outlook = (
        leadtime_outlook
        if (
            cap_applied
            or corroboration_gate_blocks_strengthening
            or confirmation_gate_blocks_strengthening
        )
        else proposed_leadtime_outlook
    )
    final_buffer_scale = (
        safety_buffer_scale
        if (
            cap_applied
            or corroboration_gate_blocks_strengthening
            or confirmation_gate_blocks_strengthening
        )
        else proposed_buffer_scale
    )
    return (
        final_demand_outlook,
        final_leadtime_outlook,
        final_buffer_scale,
        ExternalEvidenceFusionSummary(
            proposed_demand_outlook=proposed_demand_outlook,
            proposed_leadtime_outlook=proposed_leadtime_outlook,
            proposed_safety_buffer_scale=proposed_buffer_scale,
            final_demand_outlook=final_demand_outlook,
            final_leadtime_outlook=final_leadtime_outlook,
            final_safety_buffer_scale=final_buffer_scale,
            corroboration_count=corroboration_count,
            role_labels=role_labels,
            cap_applied=cap_applied,
            corroboration_gate_applied=corroboration_gate_applied,
            corroborated_family_change_allowed=corroborated_family_change_allowed,
            early_confirmation_gate_applied=early_confirmation_gate_applied,
            family_change_blocked=family_change_blocked,
            proposed_update_request_types=proposed_update_types,
            final_update_request_types=final_update_types,
            reason=(
                "confirming_external_evidence_with_sufficient_internal_signal"
                if cap_applied
                else None
            ),
            corroboration_gate_reason=corroboration_gate_reason,
            early_confirmation_gate_reason=early_confirmation_gate_reason,
        ),
    )


def build_scenario_update_request(invocation: ToolInvocation) -> ScenarioUpdateRequest:
    """Build a typed scenario-update request from prior bounded tool outputs."""

    try:
        forecast_result = _extract_prior_output(
            invocation,
            "forecast_result",
            ForecastResult,
        )
    except ValueError:
        forecast_result = _fallback_forecast_from_observation(invocation)
    try:
        leadtime_result = _extract_prior_output(
            invocation,
            "leadtime_result",
            LeadTimeResult,
        )
    except ValueError:
        leadtime_result = _fallback_leadtime_from_observation(invocation)
    current_regime = RegimeLabel.NORMAL
    if invocation.system_state is not None:
        current_regime = invocation.system_state.regime_label
    if (
        invocation.observation is not None
        and invocation.observation.regime_label is not None
    ):
        current_regime = invocation.observation.regime_label
    if invocation.agent_assessment is not None:
        current_regime = invocation.agent_assessment.regime_label
        request_types = tuple(
            update_request.request_type
            for update_request in invocation.agent_assessment.update_requests
        ) or (UpdateRequestType.KEEP_CURRENT,)
    else:
        request_types = _DEFAULT_UPDATE_TYPES[current_regime]
    return ScenarioUpdateRequest(
        update_requests=tuple(
            UpdateRequest(
                request_type=request_type,
                target="serial_smoke_scenarios",
                notes="Deterministic placeholder scenario update.",
            )
            for request_type in request_types
        ),
        current_regime=current_regime,
        current_scenarios=(
            ScenarioSummary(
                scenario_id=f"{current_regime.value}_baseline",
                regime_label=current_regime,
            ),
        ),
        forecast_result=forecast_result,  # type: ignore[arg-type]
        leadtime_result=leadtime_result,  # type: ignore[arg-type]
    )


def _resolve_adjustment(
    invocation: ToolInvocation,
    request: ScenarioUpdateRequest,
    assessment: AgentAssessment | None,
    external_evidence_interpretation: ExternalEvidenceInterpretation | None,
) -> tuple[
    RegimeLabel,
    float,
    float,
    float,
    bool,
    tuple[UpdateRequestType, ...],
    ExternalEvidenceFusionSummary | None,
]:
    demand_scale, leadtime_scale, buffer_scale = _REGIME_BASE_SCALES[request.current_regime]
    proposed_update_types = tuple(
        update_request.request_type for update_request in request.update_requests
    )
    effective_regime = request.current_regime
    applied_update_types = proposed_update_types
    corroboration_count = 0
    role_labels: tuple[str, ...] = ()
    corroboration_gate_applied = False
    corroborated_family_change_allowed = False
    family_change_blocked = False
    corroboration_gate_reason = None
    early_confirmation_gate_applied = False
    early_confirmation_gate_reason = None
    if external_evidence_interpretation is not None:
        (
            effective_regime,
            applied_update_types,
            corroboration_count,
            role_labels,
            corroboration_gate_applied,
            family_change_blocked,
            corroborated_family_change_allowed,
            corroboration_gate_reason,
        ) = _apply_corroboration_gate(
            invocation=invocation,
            request=request,
            interpretation=external_evidence_interpretation,
            applied_update_types=proposed_update_types,
        )
        (
            effective_regime,
            applied_update_types,
            early_confirmation_gate_applied,
            early_confirmation_gate_family_blocked,
            early_confirmation_gate_reason,
        ) = _apply_early_evidence_confirmation_gate(
            invocation=invocation,
            request=request,
            effective_regime=effective_regime,
            interpretation=external_evidence_interpretation,
            applied_update_types=applied_update_types,
            corroborated_family_change_allowed=corroborated_family_change_allowed,
        )
        family_change_blocked = (
            family_change_blocked or early_confirmation_gate_family_blocked
        )
    demand_scale, leadtime_scale, buffer_scale = _REGIME_BASE_SCALES[effective_regime]
    for update_type in applied_update_types:
        demand_factor, leadtime_factor, buffer_factor = _UPDATE_TYPE_EFFECTS[update_type]
        demand_scale *= demand_factor
        leadtime_scale *= leadtime_factor
        buffer_scale *= buffer_factor
    if assessment is not None:
        request_replan = assessment.request_replan
    else:
        request_replan = request.current_regime is not RegimeLabel.NORMAL
    external_evidence_fusion_summary = None
    if external_evidence_interpretation is not None:
        demand_outlook, leadtime_outlook, buffer_scale, external_evidence_fusion_summary = (
            _resolve_external_evidence_fusion(
                invocation=invocation,
                request=request,
                assessment=assessment,
                interpretation=external_evidence_interpretation,
                corroboration_count=corroboration_count,
                role_labels=role_labels,
                proposed_update_types=proposed_update_types,
                final_update_types=applied_update_types,
                corroboration_gate_applied=corroboration_gate_applied,
                corroborated_family_change_allowed=corroborated_family_change_allowed,
                corroboration_gate_reason=corroboration_gate_reason,
                early_confirmation_gate_applied=early_confirmation_gate_applied,
                family_change_blocked=family_change_blocked,
                early_confirmation_gate_reason=early_confirmation_gate_reason,
                demand_outlook=request.forecast_result.point_forecast[0] * demand_scale,
                leadtime_outlook=(
                    request.leadtime_result.expected_lead_time[0] * leadtime_scale
                ),
                safety_buffer_scale=buffer_scale,
            )
        )
        demand_scale = demand_outlook / request.forecast_result.point_forecast[0]
        leadtime_scale = leadtime_outlook / request.leadtime_result.expected_lead_time[0]
    return (
        effective_regime,
        demand_scale,
        leadtime_scale,
        buffer_scale,
        request_replan,
        applied_update_types,
        external_evidence_fusion_summary,
    )


@dataclass(frozen=True, slots=True)
class DeterministicScenarioTool(BoundedTool):
    """Deterministic placeholder scenario tool for smoke runs."""

    tool_id: str = "scenario_tool"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            tool_id=self.tool_id,
            tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
            supported_subgoals=(
                OperationalSubgoal.QUERY_UNCERTAINTY,
                OperationalSubgoal.UPDATE_UNCERTAINTY,
            ),
            description=(
                "Deterministic placeholder scenario adapter for bounded "
                "uncertainty inspection and update routing."
            ),
        )

    def invoke(self, invocation: ToolInvocation) -> ToolResult:
        request = build_scenario_update_request(invocation)
        external_evidence_interpretation = _extract_external_evidence_interpretation(
            invocation
        )
        (
            effective_regime,
            demand_scale,
            leadtime_scale,
            buffer_scale,
            request_replan,
            applied_update_types,
            external_evidence_fusion_summary,
        ) = _resolve_adjustment(
            invocation,
            request,
            invocation.agent_assessment,
            external_evidence_interpretation,
        )
        demand_outlook = request.forecast_result.point_forecast[0] * demand_scale
        leadtime_outlook = request.leadtime_result.expected_lead_time[0] * leadtime_scale
        scenario_update_result = ScenarioUpdateResult(
            scenarios=(
                ScenarioSummary(
                    scenario_id="serial_smoke",
                    regime_label=effective_regime,
                    weight=1.0,
                    demand_scale=demand_scale,
                    leadtime_scale=leadtime_scale,
                ),
            ),
            applied_update_types=applied_update_types,
            adjustment=ScenarioAdjustmentSummary(
                demand_outlook=demand_outlook,
                leadtime_outlook=leadtime_outlook,
                safety_buffer_scale=buffer_scale,
                external_evidence_fusion_summary=external_evidence_fusion_summary,
            ),
            request_replan=request_replan,
            provenance=(
                "deterministic_scenario_smoke_adapter_with_external_evidence"
                if external_evidence_interpretation is not None
                else "deterministic_scenario_smoke_adapter"
            ),
        )
        return ToolResult(
            tool_id=invocation.tool_id,
            tool_class=invocation.tool_class,
            subgoal=invocation.subgoal,
            status=ToolStatus.SUCCESS,
            structured_output={"scenario_update_result": scenario_update_result},
            provenance=scenario_update_result.provenance,
            next_subgoal=(
                OperationalSubgoal.REQUEST_REPLAN
                if scenario_update_result.request_replan
                else OperationalSubgoal.NO_ACTION
            ),
            request_replan=scenario_update_result.request_replan,
        )


__all__ = ["DeterministicScenarioTool", "build_scenario_update_request"]
