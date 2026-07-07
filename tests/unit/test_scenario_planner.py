from __future__ import annotations

from meio.agents.scenario_planner import (
    CounterfactualRegretGuardTool,
    DemandUncertaintyDecompositionRecord,
    DemandUncertaintyDecompositionTool,
    DemandUncertaintyType,
    PlannerCandidateScoreRecord,
    RegimeBeliefEntry,
    RegimeBeliefRecord,
    RegimeBeliefTool,
    RegimeDiagnosisRecord,
    RegimeDiagnosisTool,
    RiskSensitiveScenarioEvaluatorTool,
    ScenarioCandidateRecord,
    ScenarioCandidateGeneratorTool,
    ScenarioCandidateSet,
    ScenarioPlannerEvaluationDiagnostics,
    _guarded_candidate_id,
    _spread_aware_candidates,
    _uncertainty_quality_penalty,
)
from meio.contracts import (
    OperationalSubgoal,
    RegimeLabel,
    ToolInvocation,
    UpdateRequestType,
)
from meio.scenarios.contracts import ScenarioUpdateResult
from meio.simulation.evidence import DemandEvidence, LeadTimeEvidence, RuntimeEvidence
from meio.simulation.serial_benchmark import (
    build_initial_simulation_state,
    build_serial_benchmark_case,
)
from meio.simulation.state import Observation


def _spread_context():
    benchmark_case = build_serial_benchmark_case()
    system_state = build_initial_simulation_state(
        benchmark_case,
        regime_label=RegimeLabel.DEMAND_REGIME_SHIFT,
    )
    observation = Observation(
        time_index=0,
        demand_evidence=DemandEvidence(
            history=(2.0, 10.0, 18.0),
            latest_realization=(18.0,),
            stage_index=1,
        ),
        leadtime_evidence=LeadTimeEvidence(
            history=(2.0, 2.0, 2.0),
            latest_realization=(2.0,),
            upstream_stage_index=3,
            downstream_stage_index=2,
        ),
        regime_label=RegimeLabel.DEMAND_REGIME_SHIFT,
    )
    evidence = RuntimeEvidence(
        time_index=0,
        demand=observation.demand_evidence,
        leadtime=observation.leadtime_evidence,
        scenario_families=tuple(RegimeLabel),
        demand_baseline_value=10.0,
        leadtime_baseline_value=2.0,
        notes=("mean_preserving_spread_test",),
    )
    return benchmark_case, system_state, observation, evidence


def _invoke(tool, *, prior_results=()):
    _, system_state, observation, evidence = _spread_context()
    return tool.invoke(
        ToolInvocation(
            tool_id=tool.spec.tool_id,
            tool_class=tool.spec.tool_class,
            subgoal=OperationalSubgoal.QUERY_UNCERTAINTY,
            evidence=evidence,
            system_state=system_state,
            observation=observation,
            prior_results=prior_results,
        )
    )


def test_decomposition_classifies_mean_preserving_spread_increase() -> None:
    diagnosis_result = _invoke(RegimeDiagnosisTool())
    decomposition_result = _invoke(
        DemandUncertaintyDecompositionTool(),
        prior_results=(diagnosis_result,),
    )

    decomposition = decomposition_result.structured_output[
        "demand_uncertainty_decomposition"
    ]

    assert isinstance(decomposition, DemandUncertaintyDecompositionRecord)
    assert decomposition.uncertainty_type is DemandUncertaintyType.SPREAD_INCREASE
    assert decomposition.recommended_demand_outlook == 10.0
    assert decomposition.recommended_safety_buffer_scale > 1.0
    assert decomposition.recommended_update_types == (
        UpdateRequestType.WIDEN_UNCERTAINTY,
    )


def test_candidate_generator_adds_mean_preserving_spread_candidate() -> None:
    benchmark_case, _, _, _ = _spread_context()
    diagnosis_result = _invoke(RegimeDiagnosisTool())
    decomposition_result = _invoke(
        DemandUncertaintyDecompositionTool(),
        prior_results=(diagnosis_result,),
    )
    candidate_result = _invoke(
        ScenarioCandidateGeneratorTool(benchmark_case=benchmark_case),
        prior_results=(diagnosis_result, decomposition_result),
    )

    candidate_set = candidate_result.structured_output["scenario_candidate_set"]

    assert isinstance(candidate_set, ScenarioCandidateSet)
    candidate_ids = {candidate.candidate_id for candidate in candidate_set.candidates}
    assert "agentic_mean_preserving_spread_guard" in candidate_ids
    assert "agentic_spread_service_floor_guard" in candidate_ids
    assert "agentic_spread_tail_cover_guard" in candidate_ids
    assert "agentic_fast_demand_reweight" not in candidate_ids
    spread_candidate = next(
        candidate
        for candidate in candidate_set.candidates
        if candidate.candidate_id == "agentic_mean_preserving_spread_guard"
    )
    service_floor_candidate = next(
        candidate
        for candidate in candidate_set.candidates
        if candidate.candidate_id == "agentic_spread_service_floor_guard"
    )
    tail_cover_candidate = next(
        candidate
        for candidate in candidate_set.candidates
        if candidate.candidate_id == "agentic_spread_tail_cover_guard"
    )
    assert spread_candidate.demand_outlook == 10.0
    assert spread_candidate.safety_buffer_scale > 1.0
    assert service_floor_candidate.demand_outlook >= spread_candidate.demand_outlook
    assert (
        service_floor_candidate.safety_buffer_scale
        > spread_candidate.safety_buffer_scale
    )
    assert tail_cover_candidate.demand_outlook >= spread_candidate.demand_outlook
    assert tail_cover_candidate.demand_outlook <= spread_candidate.demand_outlook * 1.45
    assert UpdateRequestType.WIDEN_UNCERTAINTY in tail_cover_candidate.applied_update_types


def test_full_tool_chain_keeps_spread_update_mean_anchored() -> None:
    benchmark_case, _, _, _ = _spread_context()
    prior_results = ()
    for tool in (
        RegimeDiagnosisTool(),
        DemandUncertaintyDecompositionTool(),
        RegimeBeliefTool(),
        ScenarioCandidateGeneratorTool(benchmark_case=benchmark_case),
        RiskSensitiveScenarioEvaluatorTool(benchmark_case=benchmark_case),
        CounterfactualRegretGuardTool(),
    ):
        result = _invoke(tool, prior_results=prior_results)
        prior_results = prior_results + (result,)

    final_update = prior_results[-1].structured_output["scenario_update_result"]

    assert isinstance(final_update, ScenarioUpdateResult)
    assert final_update.adjustment.demand_outlook <= 11.5
    assert final_update.adjustment.safety_buffer_scale >= 1.0
    assert UpdateRequestType.WIDEN_UNCERTAINTY in final_update.applied_update_types


def test_stable_decomposition_does_not_force_latest_tail_chasing() -> None:
    diagnosis = RegimeDiagnosisRecord(
        regime_label=RegimeLabel.DEMAND_REGIME_SHIFT,
        case_family="sustained_demand_shift",
        demand_ratio_to_baseline=1.7,
        leadtime_ratio_to_baseline=1.0,
        recent_stress_count=1,
        total_backorder=0.0,
        pipeline_total=0.0,
        latest_demand=17.0,
        latest_leadtime=2.0,
        demand_window=(10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 17.0),
        leadtime_window=(2.0, 2.0, 2.0),
        agent_update_request_types=(UpdateRequestType.SWITCH_DEMAND_REGIME,),
        rationale="unit test stable high-tail diagnosis",
    )
    decomposition = DemandUncertaintyDecompositionRecord(
        uncertainty_type=DemandUncertaintyType.STABLE,
        baseline_mean=10.0,
        recent_mean=10.875,
        baseline_cv_reference=0.25,
        recent_std=2.315,
        recent_cv=0.213,
        mean_delta=0.875,
        mean_delta_ratio=0.0875,
        spread_ratio=0.852,
        recommended_demand_outlook=10.875,
        recommended_safety_buffer_scale=1.0,
        recommended_update_types=(UpdateRequestType.KEEP_CURRENT,),
        confidence=0.78,
        rationale="unit test stable decomposition",
    )
    anchored_candidate = ScenarioCandidateRecord(
        candidate_id="anchored",
        provenance="unit_test",
        demand_outlook=10.875,
        leadtime_outlook=2.0,
        safety_buffer_scale=1.0,
        applied_update_types=(UpdateRequestType.KEEP_CURRENT,),
        request_replan=False,
        rationale="anchored candidate",
    )
    tail_chasing_candidate = ScenarioCandidateRecord(
        candidate_id="tail_chasing",
        provenance="unit_test",
        demand_outlook=17.0,
        leadtime_outlook=2.0,
        safety_buffer_scale=1.2,
        applied_update_types=(UpdateRequestType.SWITCH_DEMAND_REGIME,),
        request_replan=True,
        rationale="tail chasing candidate",
    )

    anchored_penalty = _uncertainty_quality_penalty(
        anchored_candidate,
        diagnosis,
        decomposition,
    )
    tail_chasing_penalty = _uncertainty_quality_penalty(
        tail_chasing_candidate,
        diagnosis,
        decomposition,
    )

    assert anchored_penalty == 0.0
    assert tail_chasing_penalty > anchored_penalty

    stable_candidates = _spread_aware_candidates(diagnosis, decomposition)
    assert {candidate.candidate_id for candidate in stable_candidates} == {
        "agentic_stable_reference_guard",
    }
    assert stable_candidates[0].demand_outlook == decomposition.recommended_demand_outlook


def test_mixed_decomposition_anchors_response_to_recent_mean_not_latest_tail() -> None:
    diagnosis = RegimeDiagnosisRecord(
        regime_label=RegimeLabel.DEMAND_REGIME_SHIFT,
        case_family="sustained_demand_shift",
        demand_ratio_to_baseline=5.8,
        leadtime_ratio_to_baseline=1.0,
        recent_stress_count=2,
        total_backorder=0.0,
        pipeline_total=0.0,
        latest_demand=58.0,
        latest_leadtime=2.0,
        demand_window=(8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 58.0),
        leadtime_window=(2.0, 2.0, 2.0),
        agent_update_request_types=(UpdateRequestType.SWITCH_DEMAND_REGIME,),
        rationale="unit test mixed high-tail diagnosis",
    )
    decomposition = DemandUncertaintyDecompositionRecord(
        uncertainty_type=DemandUncertaintyType.MIXED,
        baseline_mean=10.0,
        recent_mean=16.875,
        baseline_cv_reference=0.25,
        recent_std=15.75,
        recent_cv=0.933,
        mean_delta=6.875,
        mean_delta_ratio=0.6875,
        spread_ratio=3.732,
        recommended_demand_outlook=16.875,
        recommended_safety_buffer_scale=1.16,
        recommended_update_types=(
            UpdateRequestType.SWITCH_DEMAND_REGIME,
            UpdateRequestType.WIDEN_UNCERTAINTY,
        ),
        confidence=0.96,
        rationale="unit test mixed decomposition",
    )
    mixed_candidate = ScenarioCandidateRecord(
        candidate_id="mixed_anchor",
        provenance="unit_test",
        demand_outlook=16.875,
        leadtime_outlook=2.0,
        safety_buffer_scale=1.16,
        applied_update_types=(
            UpdateRequestType.SWITCH_DEMAND_REGIME,
            UpdateRequestType.WIDEN_UNCERTAINTY,
        ),
        request_replan=True,
        rationale="mixed anchored candidate",
    )
    latest_tail_candidate = ScenarioCandidateRecord(
        candidate_id="latest_tail",
        provenance="unit_test",
        demand_outlook=58.0,
        leadtime_outlook=2.0,
        safety_buffer_scale=1.2,
        applied_update_types=(UpdateRequestType.SWITCH_DEMAND_REGIME,),
        request_replan=True,
        rationale="latest tail candidate",
    )

    mixed_penalty = _uncertainty_quality_penalty(
        mixed_candidate,
        diagnosis,
        decomposition,
    )
    latest_tail_penalty = _uncertainty_quality_penalty(
        latest_tail_candidate,
        diagnosis,
        decomposition,
    )

    assert mixed_penalty == 0.0
    assert latest_tail_penalty > mixed_penalty


def test_spread_guard_prefers_tail_cover_under_service_pressure_when_close() -> None:
    diagnosis = RegimeDiagnosisRecord(
        regime_label=RegimeLabel.DEMAND_REGIME_SHIFT,
        case_family="initial_demand_shift",
        demand_ratio_to_baseline=1.5,
        leadtime_ratio_to_baseline=1.0,
        recent_stress_count=1,
        total_backorder=0.0,
        pipeline_total=0.0,
        latest_demand=15.0,
        latest_leadtime=2.0,
        demand_window=(8.0, 10.0, 15.0),
        leadtime_window=(2.0, 2.0, 2.0),
        agent_update_request_types=(UpdateRequestType.WIDEN_UNCERTAINTY,),
        rationale="unit test spread service pressure",
    )
    decomposition = DemandUncertaintyDecompositionRecord(
        uncertainty_type=DemandUncertaintyType.SPREAD_INCREASE,
        baseline_mean=10.0,
        recent_mean=11.0,
        baseline_cv_reference=0.25,
        recent_std=2.94,
        recent_cv=0.267,
        mean_delta=1.0,
        mean_delta_ratio=0.1,
        spread_ratio=1.45,
        recommended_demand_outlook=11.0,
        recommended_safety_buffer_scale=1.07,
        recommended_update_types=(UpdateRequestType.WIDEN_UNCERTAINTY,),
        confidence=0.85,
        rationale="unit test spread decomposition",
    )
    belief = RegimeBeliefRecord(
        entries=(
            RegimeBeliefEntry(
                regime_label=RegimeLabel.NORMAL,
                probability=1.0,
                demand_multiplier=1.0,
                leadtime_multiplier=1.0,
                rationale="unit test belief",
            ),
        ),
        dominant_regime_label=RegimeLabel.NORMAL,
        belief_entropy=0.0,
        tail_risk_weight=0.2,
        service_risk_weight=0.3,
        overreaction_weight=0.2,
        rationale="unit test belief",
    )
    evaluation = ScenarioPlannerEvaluationDiagnostics(
        selected_candidate_id="agentic_mean_preserving_spread_guard",
        incumbent_candidate_id="rolling_horizon_incumbent",
        horizon_length=3,
        scenario_count=8,
        candidate_count=2,
        selected_expected_cost=100.0,
        incumbent_expected_cost=105.0,
        candidate_scores=(
            PlannerCandidateScoreRecord(
                candidate_id="agentic_mean_preserving_spread_guard",
                expected_cost=100.0,
                demand_outlook=11.0,
                leadtime_outlook=2.0,
                safety_buffer_scale=1.07,
                mean_cost=80.0,
                tail_cost=90.0,
                service_risk_penalty=20.0,
                overreaction_penalty=0.0,
                selected=True,
            ),
            PlannerCandidateScoreRecord(
                candidate_id="agentic_spread_tail_cover_guard",
                expected_cost=135.0,
                demand_outlook=14.0,
                leadtime_outlook=2.0,
                safety_buffer_scale=1.12,
                mean_cost=88.0,
                tail_cost=96.0,
                service_risk_penalty=18.0,
                overreaction_penalty=1.0,
                selected=False,
            ),
        ),
    )

    selected_id, reason = _guarded_candidate_id(
        diagnosis=diagnosis,
        decomposition=decomposition,
        belief=belief,
        evaluation=evaluation,
        time_index=2,
        immediate_shift_margin=75.0,
        clean_recovery_margin=8.0,
    )

    assert selected_id == "agentic_spread_tail_cover_guard"
    assert reason == "spread_increase_service_pressure_prefers_tail_cover_when_close"
