from __future__ import annotations

from meio.agents.llm_client import LLMClientContext
from meio.agents.structured_outputs import StructuredLLMDecision
from meio.contracts import OperationalSubgoal, RegimeLabel, UpdateRequest, UpdateRequestType
from meio.scenarios.update_calibration import calibrate_update_decision
from meio.scenarios.update_policy import (
    ORDERED_UPDATE_LADDER,
    UpdateStrength,
    build_update_requests_for_strength,
    infer_update_strength,
    infer_update_strength_from_requests,
)


def _decision(
    *,
    regime_label: RegimeLabel,
    update_types: tuple[UpdateRequestType, ...],
    request_replan: bool = True,
) -> StructuredLLMDecision:
    return StructuredLLMDecision(
        selected_subgoal=OperationalSubgoal.QUERY_UNCERTAINTY,
        candidate_tool_ids=("forecast_tool", "leadtime_tool", "scenario_tool"),
        regime_label=regime_label,
        confidence=0.85,
        update_requests=tuple(
            UpdateRequest(
                request_type=update_type,
                target="serial_scenarios",
                notes="unit_test",
            )
            for update_type in update_types
        ),
        request_replan=request_replan,
        rationale="bounded test decision",
    )


def _context(
    *,
    regime_label: RegimeLabel,
    demand_value: float,
    previous_demand_value: float,
    demand_baseline_value: float = 10.0,
    leadtime_value: float = 2.0,
    previous_leadtime_value: float = 2.0,
    leadtime_baseline_value: float = 2.0,
    pipeline_ratio_to_baseline: float | None = None,
    backorder_ratio_to_baseline: float | None = None,
    repeated_stress_detected: bool | None = None,
    recovery_with_high_pipeline_load: bool | None = None,
    recovery_with_high_backorder_load: bool | None = None,
) -> LLMClientContext:
    return LLMClientContext(
        benchmark_id="serial_3_echelon",
        mission_id="serial_mission",
        time_index=2,
        regime_label=regime_label,
        demand_value=demand_value,
        leadtime_value=leadtime_value,
        inventory_level=(10.0, 12.0, 14.0),
        backorder_level=(0.0, 0.0, 0.0),
        available_tool_ids=("forecast_tool", "leadtime_tool", "scenario_tool"),
        max_tool_steps=3,
        demand_baseline_value=demand_baseline_value,
        demand_change_from_baseline=demand_value - demand_baseline_value,
        demand_ratio_to_baseline=demand_value / demand_baseline_value,
        previous_demand_value=previous_demand_value,
        demand_change_from_previous=demand_value - previous_demand_value,
        leadtime_baseline_value=leadtime_baseline_value,
        leadtime_change_from_baseline=leadtime_value - leadtime_baseline_value,
        leadtime_ratio_to_baseline=leadtime_value / leadtime_baseline_value,
        previous_leadtime_value=previous_leadtime_value,
        leadtime_change_from_previous=leadtime_value - previous_leadtime_value,
        downstream_inventory_value=10.0,
        total_inventory_value=36.0,
        pipeline_total_value=(
            None
            if pipeline_ratio_to_baseline is None
            else pipeline_ratio_to_baseline * demand_baseline_value
        ),
        total_backorder_value=(
            None
            if backorder_ratio_to_baseline is None
            else backorder_ratio_to_baseline * demand_baseline_value
        ),
        inventory_gap_to_demand=10.0 - demand_value,
        pipeline_ratio_to_baseline=pipeline_ratio_to_baseline,
        backorder_ratio_to_baseline=backorder_ratio_to_baseline,
        repeated_stress_detected=repeated_stress_detected,
        pipeline_heavy_vs_baseline=(
            None if pipeline_ratio_to_baseline is None else pipeline_ratio_to_baseline >= 12.0
        ),
        backlog_heavy_vs_baseline=(
            None if backorder_ratio_to_baseline is None else backorder_ratio_to_baseline >= 2.0
        ),
        recovery_with_high_pipeline_load=recovery_with_high_pipeline_load,
        recovery_with_high_backorder_load=recovery_with_high_backorder_load,
    )


def test_update_policy_defines_explicit_ordered_ladder() -> None:
    assert ORDERED_UPDATE_LADDER == (
        UpdateStrength.KEEP_CURRENT,
        UpdateStrength.REWEIGHT_SCENARIOS,
        UpdateStrength.WIDEN_UNCERTAINTY,
        UpdateStrength.SWITCH_DEMAND_REGIME_PLUS_WIDEN_UNCERTAINTY,
    )
    assert infer_update_strength((UpdateRequestType.KEEP_CURRENT,)) is UpdateStrength.KEEP_CURRENT
    assert infer_update_strength((UpdateRequestType.REWEIGHT_SCENARIOS,)) is UpdateStrength.REWEIGHT_SCENARIOS
    assert infer_update_strength((UpdateRequestType.WIDEN_UNCERTAINTY,)) is UpdateStrength.WIDEN_UNCERTAINTY
    assert infer_update_strength(
        (
            UpdateRequestType.SWITCH_DEMAND_REGIME,
            UpdateRequestType.WIDEN_UNCERTAINTY,
        )
    ) is UpdateStrength.SWITCH_DEMAND_REGIME_PLUS_WIDEN_UNCERTAINTY
    assert infer_update_strength_from_requests(
        build_update_requests_for_strength(
            UpdateStrength.REWEIGHT_SCENARIOS,
            notes="unit_test",
        )
    ) is UpdateStrength.REWEIGHT_SCENARIOS


def test_calibration_applies_repeated_stress_hysteresis() -> None:
    result = calibrate_update_decision(
        _decision(
            regime_label=RegimeLabel.DEMAND_REGIME_SHIFT,
            update_types=(
                UpdateRequestType.SWITCH_DEMAND_REGIME,
                UpdateRequestType.WIDEN_UNCERTAINTY,
            ),
        ),
        context=_context(
            regime_label=RegimeLabel.DEMAND_REGIME_SHIFT,
            demand_value=14.0,
            previous_demand_value=14.0,
            repeated_stress_detected=True,
            pipeline_ratio_to_baseline=4.2,
        ),
        recent_regime_history=(
            RegimeLabel.NORMAL,
            RegimeLabel.DEMAND_REGIME_SHIFT,
        ),
        recent_stress_reference_demand_value=14.0,
        recent_update_request_history=(
            (UpdateRequestType.SWITCH_DEMAND_REGIME, UpdateRequestType.WIDEN_UNCERTAINTY),
        ),
    )

    assert result.calibration_applied is True
    assert result.hysteresis_applied is True
    assert result.repeated_stress_moderation_applied is True
    assert result.final_strength is UpdateStrength.REWEIGHT_SCENARIOS
    assert tuple(
        request.request_type for request in result.final_decision.update_requests
    ) == (UpdateRequestType.REWEIGHT_SCENARIOS,)


def test_calibration_does_not_fire_when_demand_materially_worsens() -> None:
    result = calibrate_update_decision(
        _decision(
            regime_label=RegimeLabel.DEMAND_REGIME_SHIFT,
            update_types=(
                UpdateRequestType.SWITCH_DEMAND_REGIME,
                UpdateRequestType.WIDEN_UNCERTAINTY,
            ),
        ),
        context=_context(
            regime_label=RegimeLabel.DEMAND_REGIME_SHIFT,
            demand_value=15.0,
            previous_demand_value=14.0,
            repeated_stress_detected=True,
            pipeline_ratio_to_baseline=4.2,
        ),
        recent_regime_history=(
            RegimeLabel.NORMAL,
            RegimeLabel.DEMAND_REGIME_SHIFT,
        ),
        recent_stress_reference_demand_value=14.0,
        recent_update_request_history=(
            (UpdateRequestType.SWITCH_DEMAND_REGIME, UpdateRequestType.WIDEN_UNCERTAINTY),
        ),
    )

    assert result.calibration_applied is False
    assert result.final_strength is UpdateStrength.SWITCH_DEMAND_REGIME_PLUS_WIDEN_UNCERTAINTY


def test_calibration_applies_relapse_moderation_with_unresolved_load() -> None:
    result = calibrate_update_decision(
        _decision(
            regime_label=RegimeLabel.DEMAND_REGIME_SHIFT,
            update_types=(
                UpdateRequestType.SWITCH_DEMAND_REGIME,
                UpdateRequestType.WIDEN_UNCERTAINTY,
            ),
        ),
        context=_context(
            regime_label=RegimeLabel.DEMAND_REGIME_SHIFT,
            demand_value=14.0,
            previous_demand_value=11.0,
            pipeline_ratio_to_baseline=4.8,
            backorder_ratio_to_baseline=1.5,
            repeated_stress_detected=False,
        ),
        recent_regime_history=(
            RegimeLabel.NORMAL,
            RegimeLabel.DEMAND_REGIME_SHIFT,
            RegimeLabel.RECOVERY,
        ),
        recent_stress_reference_demand_value=14.0,
        recent_update_request_history=(
            (UpdateRequestType.SWITCH_DEMAND_REGIME, UpdateRequestType.WIDEN_UNCERTAINTY),
            (UpdateRequestType.KEEP_CURRENT,),
        ),
    )

    assert result.calibration_applied is True
    assert result.relapse_moderation_applied is True
    assert result.unresolved_stress_moderation_applied is True
    assert result.final_strength is UpdateStrength.REWEIGHT_SCENARIOS


def test_calibration_does_not_fire_after_normalization() -> None:
    result = calibrate_update_decision(
        _decision(
            regime_label=RegimeLabel.DEMAND_REGIME_SHIFT,
            update_types=(
                UpdateRequestType.SWITCH_DEMAND_REGIME,
                UpdateRequestType.WIDEN_UNCERTAINTY,
            ),
        ),
        context=_context(
            regime_label=RegimeLabel.DEMAND_REGIME_SHIFT,
            demand_value=14.0,
            previous_demand_value=11.0,
            pipeline_ratio_to_baseline=0.8,
            backorder_ratio_to_baseline=0.0,
            repeated_stress_detected=False,
        ),
        recent_regime_history=(
            RegimeLabel.NORMAL,
            RegimeLabel.DEMAND_REGIME_SHIFT,
            RegimeLabel.RECOVERY,
        ),
        recent_stress_reference_demand_value=14.0,
        recent_update_request_history=(
            (UpdateRequestType.SWITCH_DEMAND_REGIME, UpdateRequestType.WIDEN_UNCERTAINTY),
            (UpdateRequestType.KEEP_CURRENT,),
        ),
    )

    assert result.calibration_applied is False
    assert result.relapse_moderation_applied is False


def test_calibration_applies_recovery_hysteresis_under_high_load() -> None:
    result = calibrate_update_decision(
        _decision(
            regime_label=RegimeLabel.RECOVERY,
            update_types=(UpdateRequestType.REWEIGHT_SCENARIOS,),
            request_replan=True,
        ),
        context=_context(
            regime_label=RegimeLabel.RECOVERY,
            demand_value=11.0,
            previous_demand_value=14.0,
            pipeline_ratio_to_baseline=4.5,
            backorder_ratio_to_baseline=1.2,
            recovery_with_high_pipeline_load=True,
            recovery_with_high_backorder_load=True,
        ),
        recent_regime_history=(
            RegimeLabel.NORMAL,
            RegimeLabel.DEMAND_REGIME_SHIFT,
            RegimeLabel.DEMAND_REGIME_SHIFT,
        ),
        recent_stress_reference_demand_value=14.0,
        recent_update_request_history=(
            (UpdateRequestType.SWITCH_DEMAND_REGIME, UpdateRequestType.WIDEN_UNCERTAINTY),
        ),
    )

    assert result.calibration_applied is True
    assert result.hysteresis_applied is True
    assert result.final_strength is UpdateStrength.KEEP_CURRENT
    assert result.calibration_reason == "recovery_with_high_carry_over_load"
