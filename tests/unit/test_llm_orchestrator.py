from __future__ import annotations

from meio.agents.llm_client import FakeLLMClient
from meio.agents.llm_client import LLMClientRuntimeError
from meio.agents.llm_orchestrator import LLMOrchestrationInput, LLMOrchestrator
from meio.agents.prompts import PROMPT_VERSION, prompt_contract_hash
from meio.agents.telemetry import ClientErrorCategory
from meio.config.schemas import AgentConfig
from meio.contracts import (
    MissionSpec,
    OperationalSubgoal,
    RegimeLabel,
    ToolClass,
    ToolSpec,
    UpdateRequestType,
)
from meio.simulation.evidence import DemandEvidence, LeadTimeEvidence, RuntimeEvidence
from meio.simulation.state import Observation, SimulationState


def test_llm_orchestrator_builds_typed_decision_from_fake_client() -> None:
    system_state = SimulationState(
        benchmark_id="serial_3_echelon",
        time_index=1,
        inventory_level=(20.0, 30.0, 40.0),
        stage_names=("retailer", "regional_dc", "plant"),
        pipeline_inventory=(0.0, 0.0, 0.0),
        backorder_level=(1.0, 0.0, 0.0),
        regime_label=RegimeLabel.DEMAND_REGIME_SHIFT,
    )
    observation = Observation(
        time_index=1,
        demand_evidence=DemandEvidence(
            history=(10.0, 12.0, 14.0),
            latest_realization=(14.0,),
            stage_index=1,
        ),
        leadtime_evidence=LeadTimeEvidence(
            history=(2.0, 2.0, 2.0),
            latest_realization=(2.0,),
            upstream_stage_index=2,
            downstream_stage_index=1,
        ),
        regime_label=RegimeLabel.DEMAND_REGIME_SHIFT,
    )
    evidence = RuntimeEvidence(
        time_index=1,
        demand=observation.demand_evidence,
        leadtime=observation.leadtime_evidence,
        scenario_families=(RegimeLabel.NORMAL, RegimeLabel.DEMAND_REGIME_SHIFT),
    )
    orchestrator = LLMOrchestrator(
        client=FakeLLMClient(),
        agent_config=AgentConfig(
            enabled_regime_labels=(RegimeLabel.NORMAL, RegimeLabel.DEMAND_REGIME_SHIFT),
            allowed_update_types=(
                UpdateRequestType.KEEP_CURRENT,
                UpdateRequestType.SWITCH_DEMAND_REGIME,
                UpdateRequestType.WIDEN_UNCERTAINTY,
            ),
            allowed_tool_classes=(ToolClass.DETERMINISTIC_STATISTICAL,),
            minimum_confidence=0.6,
            max_tool_steps=3,
            llm_client_mode="fake",
            llm_model_name="gpt-4o-mini",
        ),
    )

    decision = orchestrator.decide(
        LLMOrchestrationInput(
            mission=MissionSpec(
                mission_id="serial_mission",
                objective="Allow bounded LLM orchestration only.",
                admissible_tool_ids=("forecast_tool", "leadtime_tool", "scenario_tool"),
            ),
            system_state=system_state,
            observation=observation,
            evidence=evidence,
            candidate_tool_ids=("forecast_tool", "leadtime_tool", "scenario_tool"),
        ),
        (
            ToolSpec(
                tool_id="forecast_tool",
                tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
                supported_subgoals=(OperationalSubgoal.QUERY_UNCERTAINTY,),
                description="forecast",
            ),
            ToolSpec(
                tool_id="leadtime_tool",
                tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
                supported_subgoals=(OperationalSubgoal.QUERY_UNCERTAINTY,),
                description="leadtime",
            ),
            ToolSpec(
                tool_id="scenario_tool",
                tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
                supported_subgoals=(OperationalSubgoal.UPDATE_UNCERTAINTY,),
                description="scenario",
            ),
        ),
    )

    assert decision.assessment.regime_label is RegimeLabel.DEMAND_REGIME_SHIFT
    assert decision.assessment.request_replan is True
    assert decision.selected_tool_ids == ("forecast_tool", "leadtime_tool", "scenario_tool")

    attempt = orchestrator.decide_with_diagnostics(
        LLMOrchestrationInput(
            mission=MissionSpec(
                mission_id="serial_mission",
                objective="Allow bounded LLM orchestration only.",
                admissible_tool_ids=("forecast_tool", "leadtime_tool", "scenario_tool"),
            ),
            system_state=system_state,
            observation=observation,
            evidence=evidence,
            candidate_tool_ids=("forecast_tool", "leadtime_tool", "scenario_tool"),
            recent_regime_history=(
                RegimeLabel.NORMAL,
                RegimeLabel.DEMAND_REGIME_SHIFT,
            ),
            recent_stress_reference_demand_value=14.0,
            recent_update_request_history=(
                (
                    UpdateRequestType.SWITCH_DEMAND_REGIME,
                    UpdateRequestType.WIDEN_UNCERTAINTY,
                ),
            ),
        ),
        (
            ToolSpec(
                tool_id="forecast_tool",
                tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
                supported_subgoals=(OperationalSubgoal.QUERY_UNCERTAINTY,),
                description="forecast",
            ),
            ToolSpec(
                tool_id="leadtime_tool",
                tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
                supported_subgoals=(OperationalSubgoal.QUERY_UNCERTAINTY,),
                description="leadtime",
            ),
            ToolSpec(
                tool_id="scenario_tool",
                tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
                supported_subgoals=(OperationalSubgoal.UPDATE_UNCERTAINTY,),
                description="scenario",
            ),
        ),
    )

    assert attempt.diagnostics.llm_call_telemetry is not None
    assert attempt.diagnostics.llm_call_telemetry.total_tokens == 144
    assert attempt.diagnostics.llm_call_trace is not None
    assert attempt.diagnostics.llm_call_trace.validation_success is True
    assert attempt.diagnostics.llm_call_trace.parsed_output is not None
    assert attempt.diagnostics.prompt_version == PROMPT_VERSION
    assert attempt.diagnostics.prompt_contract_hash == prompt_contract_hash()
    assert attempt.diagnostics.llm_call_trace.prompt_version == PROMPT_VERSION
    assert len(attempt.diagnostics.llm_call_trace.prompt_hash) == 64
    request = orchestrator.build_request(
        LLMOrchestrationInput(
            mission=MissionSpec(
                mission_id="serial_mission",
                objective="Allow bounded LLM orchestration only.",
                admissible_tool_ids=("forecast_tool", "leadtime_tool", "scenario_tool"),
            ),
            system_state=system_state,
            observation=observation,
            evidence=evidence,
            candidate_tool_ids=("forecast_tool", "leadtime_tool", "scenario_tool"),
        ),
        (
            ToolSpec(
                tool_id="forecast_tool",
                tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
                supported_subgoals=(OperationalSubgoal.QUERY_UNCERTAINTY,),
                description="forecast",
            ),
            ToolSpec(
                tool_id="leadtime_tool",
                tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
                supported_subgoals=(OperationalSubgoal.QUERY_UNCERTAINTY,),
                description="leadtime",
            ),
            ToolSpec(
                tool_id="scenario_tool",
                tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
                supported_subgoals=(OperationalSubgoal.UPDATE_UNCERTAINTY,),
                description="scenario",
            ),
        ),
    )
    assert request.context.demand_baseline_value is None
    assert request.context.total_inventory_value == 90.0
    assert request.context.pipeline_total_value == 0.0
    assert request.context.repeated_stress_detected is False
    assert attempt.diagnostics.orchestration_latency_ms is not None
    assert attempt.diagnostics.orchestration_latency_ms >= 0.0


def test_llm_orchestrator_build_request_adds_load_aware_calibration_cues() -> None:
    system_state = SimulationState(
        benchmark_id="serial_3_echelon",
        time_index=3,
        inventory_level=(0.0, 0.0, 0.0),
        stage_names=("retailer", "regional_dc", "plant"),
        pipeline_inventory=(30.0, 40.0, 76.422584),
        backorder_level=(18.0, 19.835024, 22.752536),
        regime_label=RegimeLabel.RECOVERY,
    )
    observation = Observation(
        time_index=3,
        demand_evidence=DemandEvidence(
            history=(10.0, 14.0, 14.0, 11.0),
            latest_realization=(11.0,),
            stage_index=1,
        ),
        leadtime_evidence=LeadTimeEvidence(
            history=(2.0, 2.0, 2.0, 2.0),
            latest_realization=(2.0,),
            upstream_stage_index=2,
            downstream_stage_index=1,
        ),
        regime_label=RegimeLabel.RECOVERY,
    )
    evidence = RuntimeEvidence(
        time_index=3,
        demand=observation.demand_evidence,
        leadtime=observation.leadtime_evidence,
        scenario_families=(RegimeLabel.DEMAND_REGIME_SHIFT, RegimeLabel.RECOVERY),
        demand_baseline_value=10.0,
        leadtime_baseline_value=2.0,
    )
    orchestrator = LLMOrchestrator(
        client=FakeLLMClient(),
        agent_config=AgentConfig(
            enabled_regime_labels=(RegimeLabel.DEMAND_REGIME_SHIFT, RegimeLabel.RECOVERY),
            allowed_update_types=(
                UpdateRequestType.KEEP_CURRENT,
                UpdateRequestType.REWEIGHT_SCENARIOS,
                UpdateRequestType.SWITCH_DEMAND_REGIME,
                UpdateRequestType.WIDEN_UNCERTAINTY,
            ),
            allowed_tool_classes=(ToolClass.DETERMINISTIC_STATISTICAL,),
            minimum_confidence=0.6,
            max_tool_steps=3,
            llm_client_mode="fake",
            llm_model_name="gpt-4o-mini",
        ),
    )

    request = orchestrator.build_request(
        LLMOrchestrationInput(
            mission=MissionSpec(
                mission_id="serial_mission",
                objective="Allow bounded LLM orchestration only.",
                admissible_tool_ids=("forecast_tool", "leadtime_tool", "scenario_tool"),
            ),
            system_state=system_state,
            observation=observation,
            evidence=evidence,
            candidate_tool_ids=("forecast_tool", "leadtime_tool", "scenario_tool"),
        ),
        (
            ToolSpec(
                tool_id="forecast_tool",
                tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
                supported_subgoals=(OperationalSubgoal.QUERY_UNCERTAINTY,),
                description="forecast",
            ),
            ToolSpec(
                tool_id="leadtime_tool",
                tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
                supported_subgoals=(OperationalSubgoal.QUERY_UNCERTAINTY,),
                description="leadtime",
            ),
            ToolSpec(
                tool_id="scenario_tool",
                tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
                supported_subgoals=(OperationalSubgoal.UPDATE_UNCERTAINTY,),
                description="scenario",
            ),
        ),
    )

    assert request.context.pipeline_total_value == 146.422584
    assert request.context.pipeline_ratio_to_baseline == 14.6422584
    assert request.context.backorder_ratio_to_baseline == 6.058756
    assert request.context.pipeline_heavy_vs_baseline is True
    assert request.context.backlog_heavy_vs_baseline is True
    assert request.context.recovery_with_high_pipeline_load is True
    assert request.context.recovery_with_high_backorder_load is True


def test_llm_orchestrator_moderates_repeated_stress_over_escalation() -> None:
    system_state = SimulationState(
        benchmark_id="serial_3_echelon",
        time_index=2,
        inventory_level=(8.0, 18.0, 28.0),
        stage_names=("retailer", "regional_dc", "plant"),
        pipeline_inventory=(12.0, 14.0, 16.0),
        backorder_level=(2.0, 1.0, 0.0),
        regime_label=RegimeLabel.DEMAND_REGIME_SHIFT,
    )
    observation = Observation(
        time_index=2,
        demand_evidence=DemandEvidence(
            history=(10.0, 14.0, 14.0),
            latest_realization=(14.0,),
            stage_index=1,
        ),
        leadtime_evidence=LeadTimeEvidence(
            history=(2.0, 2.0, 2.0),
            latest_realization=(2.0,),
            upstream_stage_index=2,
            downstream_stage_index=1,
        ),
        regime_label=RegimeLabel.DEMAND_REGIME_SHIFT,
    )
    evidence = RuntimeEvidence(
        time_index=2,
        demand=observation.demand_evidence,
        leadtime=observation.leadtime_evidence,
        scenario_families=(RegimeLabel.NORMAL, RegimeLabel.DEMAND_REGIME_SHIFT),
        demand_baseline_value=10.0,
        leadtime_baseline_value=2.0,
    )
    orchestrator = LLMOrchestrator(
        client=FakeLLMClient(),
        agent_config=AgentConfig(
            enabled_regime_labels=(RegimeLabel.NORMAL, RegimeLabel.DEMAND_REGIME_SHIFT),
            allowed_update_types=(
                UpdateRequestType.KEEP_CURRENT,
                UpdateRequestType.REWEIGHT_SCENARIOS,
                UpdateRequestType.SWITCH_DEMAND_REGIME,
                UpdateRequestType.WIDEN_UNCERTAINTY,
            ),
            allowed_tool_classes=(ToolClass.DETERMINISTIC_STATISTICAL,),
            minimum_confidence=0.6,
            max_tool_steps=3,
            llm_client_mode="fake",
            llm_model_name="gpt-4o-mini",
        ),
    )

    attempt = orchestrator.decide_with_diagnostics(
        LLMOrchestrationInput(
            mission=MissionSpec(
                mission_id="serial_mission",
                objective="Allow bounded LLM orchestration only.",
                admissible_tool_ids=("forecast_tool", "leadtime_tool", "scenario_tool"),
            ),
            system_state=system_state,
            observation=observation,
            evidence=evidence,
            candidate_tool_ids=("forecast_tool", "leadtime_tool", "scenario_tool"),
            recent_regime_history=(
                RegimeLabel.NORMAL,
                RegimeLabel.DEMAND_REGIME_SHIFT,
            ),
            recent_stress_reference_demand_value=14.0,
            recent_update_request_history=(
                (
                    UpdateRequestType.SWITCH_DEMAND_REGIME,
                    UpdateRequestType.WIDEN_UNCERTAINTY,
                ),
            ),
        ),
        (
            ToolSpec(
                tool_id="forecast_tool",
                tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
                supported_subgoals=(OperationalSubgoal.QUERY_UNCERTAINTY,),
                description="forecast",
            ),
            ToolSpec(
                tool_id="leadtime_tool",
                tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
                supported_subgoals=(OperationalSubgoal.QUERY_UNCERTAINTY,),
                description="leadtime",
            ),
            ToolSpec(
                tool_id="scenario_tool",
                tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
                supported_subgoals=(OperationalSubgoal.UPDATE_UNCERTAINTY,),
                description="scenario",
            ),
        ),
    )

    assert attempt.decision is not None
    assert tuple(
        request.request_type for request in attempt.decision.assessment.update_requests
    ) == (UpdateRequestType.REWEIGHT_SCENARIOS,)
    assert attempt.diagnostics.proposed_update_request_types == (
        "switch_demand_regime",
        "widen_uncertainty",
    )
    assert (
        attempt.diagnostics.proposed_update_strength
        == "switch_demand_regime_plus_widen_uncertainty"
    )
    assert attempt.diagnostics.final_update_strength == "reweight_scenarios"
    assert attempt.diagnostics.calibration_applied is True
    assert attempt.diagnostics.hysteresis_applied is True
    assert attempt.diagnostics.final_update_request_types == ("reweight_scenarios",)
    assert attempt.diagnostics.repeated_stress_moderation_applied is True
    assert attempt.diagnostics.relapse_moderation_applied is False
    assert attempt.diagnostics.unresolved_stress_moderation_applied is False
    assert (
        attempt.diagnostics.calibration_reason
        == "repeated_stress_not_materially_worsening"
    )
    assert (
        attempt.diagnostics.moderation_reason
        == "repeated_stress_not_materially_worsening"
    )
    assert attempt.diagnostics.llm_call_trace is not None
    assert attempt.diagnostics.llm_call_trace.parsed_output is not None
    assert attempt.diagnostics.llm_call_trace.parsed_output["update_request_types"] == [
        "switch_demand_regime",
        "widen_uncertainty",
    ]


def test_llm_orchestrator_does_not_moderate_when_repeated_stress_is_worsening() -> None:
    system_state = SimulationState(
        benchmark_id="serial_3_echelon",
        time_index=2,
        inventory_level=(8.0, 18.0, 28.0),
        stage_names=("retailer", "regional_dc", "plant"),
        pipeline_inventory=(12.0, 14.0, 16.0),
        backorder_level=(2.0, 1.0, 0.0),
        regime_label=RegimeLabel.DEMAND_REGIME_SHIFT,
    )
    observation = Observation(
        time_index=2,
        demand_evidence=DemandEvidence(
            history=(10.0, 14.0, 15.0),
            latest_realization=(15.0,),
            stage_index=1,
        ),
        leadtime_evidence=LeadTimeEvidence(
            history=(2.0, 2.0, 2.0),
            latest_realization=(2.0,),
            upstream_stage_index=2,
            downstream_stage_index=1,
        ),
        regime_label=RegimeLabel.DEMAND_REGIME_SHIFT,
    )
    evidence = RuntimeEvidence(
        time_index=2,
        demand=observation.demand_evidence,
        leadtime=observation.leadtime_evidence,
        scenario_families=(RegimeLabel.NORMAL, RegimeLabel.DEMAND_REGIME_SHIFT),
        demand_baseline_value=10.0,
        leadtime_baseline_value=2.0,
    )
    orchestrator = LLMOrchestrator(
        client=FakeLLMClient(),
        agent_config=AgentConfig(
            enabled_regime_labels=(RegimeLabel.NORMAL, RegimeLabel.DEMAND_REGIME_SHIFT),
            allowed_update_types=(
                UpdateRequestType.KEEP_CURRENT,
                UpdateRequestType.REWEIGHT_SCENARIOS,
                UpdateRequestType.SWITCH_DEMAND_REGIME,
                UpdateRequestType.WIDEN_UNCERTAINTY,
            ),
            allowed_tool_classes=(ToolClass.DETERMINISTIC_STATISTICAL,),
            minimum_confidence=0.6,
            max_tool_steps=3,
            llm_client_mode="fake",
            llm_model_name="gpt-4o-mini",
        ),
    )

    attempt = orchestrator.decide_with_diagnostics(
        LLMOrchestrationInput(
            mission=MissionSpec(
                mission_id="serial_mission",
                objective="Allow bounded LLM orchestration only.",
                admissible_tool_ids=("forecast_tool", "leadtime_tool", "scenario_tool"),
            ),
            system_state=system_state,
            observation=observation,
            evidence=evidence,
            candidate_tool_ids=("forecast_tool", "leadtime_tool", "scenario_tool"),
        ),
        (
            ToolSpec(
                tool_id="forecast_tool",
                tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
                supported_subgoals=(OperationalSubgoal.QUERY_UNCERTAINTY,),
                description="forecast",
            ),
            ToolSpec(
                tool_id="leadtime_tool",
                tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
                supported_subgoals=(OperationalSubgoal.QUERY_UNCERTAINTY,),
                description="leadtime",
            ),
            ToolSpec(
                tool_id="scenario_tool",
                tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
                supported_subgoals=(OperationalSubgoal.UPDATE_UNCERTAINTY,),
                description="scenario",
            ),
        ),
    )

    assert attempt.decision is not None
    assert tuple(
        request.request_type for request in attempt.decision.assessment.update_requests
    ) == (
        UpdateRequestType.SWITCH_DEMAND_REGIME,
        UpdateRequestType.WIDEN_UNCERTAINTY,
    )
    assert attempt.diagnostics.repeated_stress_moderation_applied is False
    assert attempt.diagnostics.relapse_moderation_applied is False
    assert attempt.diagnostics.unresolved_stress_moderation_applied is False
    assert attempt.diagnostics.calibration_applied is False
    assert attempt.diagnostics.hysteresis_applied is False
    assert (
        attempt.diagnostics.proposed_update_strength
        == "switch_demand_regime_plus_widen_uncertainty"
    )
    assert (
        attempt.diagnostics.final_update_strength
        == "switch_demand_regime_plus_widen_uncertainty"
    )
    assert attempt.diagnostics.proposed_update_request_types == (
        "switch_demand_regime",
        "widen_uncertainty",
    )
    assert attempt.diagnostics.final_update_request_types == (
        "switch_demand_regime",
        "widen_uncertainty",
    )


def test_llm_orchestrator_moderates_relapse_with_unresolved_stress_load() -> None:
    system_state = SimulationState(
        benchmark_id="serial_3_echelon",
        time_index=3,
        inventory_level=(0.0, 0.0, 5.6668),
        stage_names=("retailer", "regional_dc", "plant"),
        pipeline_inventory=(30.0, 34.3332, 43.670048),
        backorder_level=(15.0, 0.8888, 0.0),
        regime_label=RegimeLabel.DEMAND_REGIME_SHIFT,
    )
    observation = Observation(
        time_index=3,
        demand_evidence=DemandEvidence(
            history=(14.0, 11.0, 14.0),
            latest_realization=(14.0,),
            stage_index=1,
        ),
        leadtime_evidence=LeadTimeEvidence(
            history=(2.0, 2.0, 2.0),
            latest_realization=(2.0,),
            upstream_stage_index=2,
            downstream_stage_index=1,
        ),
        regime_label=RegimeLabel.DEMAND_REGIME_SHIFT,
    )
    evidence = RuntimeEvidence(
        time_index=3,
        demand=observation.demand_evidence,
        leadtime=observation.leadtime_evidence,
        scenario_families=(RegimeLabel.NORMAL, RegimeLabel.DEMAND_REGIME_SHIFT),
        demand_baseline_value=10.0,
        leadtime_baseline_value=2.0,
    )
    orchestrator = LLMOrchestrator(
        client=FakeLLMClient(),
        agent_config=AgentConfig(
            enabled_regime_labels=(RegimeLabel.NORMAL, RegimeLabel.DEMAND_REGIME_SHIFT),
            allowed_update_types=(
                UpdateRequestType.KEEP_CURRENT,
                UpdateRequestType.REWEIGHT_SCENARIOS,
                UpdateRequestType.SWITCH_DEMAND_REGIME,
                UpdateRequestType.WIDEN_UNCERTAINTY,
            ),
            allowed_tool_classes=(ToolClass.DETERMINISTIC_STATISTICAL,),
            minimum_confidence=0.6,
            max_tool_steps=3,
            llm_client_mode="fake",
            llm_model_name="gpt-4o-mini",
        ),
    )

    attempt = orchestrator.decide_with_diagnostics(
        LLMOrchestrationInput(
            mission=MissionSpec(
                mission_id="serial_mission",
                objective="Allow bounded LLM orchestration only.",
                admissible_tool_ids=("forecast_tool", "leadtime_tool", "scenario_tool"),
            ),
            system_state=system_state,
            observation=observation,
            evidence=evidence,
            candidate_tool_ids=("forecast_tool", "leadtime_tool", "scenario_tool"),
            recent_regime_history=(
                RegimeLabel.NORMAL,
                RegimeLabel.DEMAND_REGIME_SHIFT,
                RegimeLabel.RECOVERY,
            ),
            recent_stress_reference_demand_value=14.0,
            recent_update_request_history=(
                (
                    UpdateRequestType.SWITCH_DEMAND_REGIME,
                    UpdateRequestType.WIDEN_UNCERTAINTY,
                ),
                (UpdateRequestType.KEEP_CURRENT,),
            ),
        ),
        (
            ToolSpec(
                tool_id="forecast_tool",
                tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
                supported_subgoals=(OperationalSubgoal.QUERY_UNCERTAINTY,),
                description="forecast",
            ),
            ToolSpec(
                tool_id="leadtime_tool",
                tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
                supported_subgoals=(OperationalSubgoal.QUERY_UNCERTAINTY,),
                description="leadtime",
            ),
            ToolSpec(
                tool_id="scenario_tool",
                tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
                supported_subgoals=(OperationalSubgoal.UPDATE_UNCERTAINTY,),
                description="scenario",
            ),
        ),
    )

    assert attempt.decision is not None
    assert tuple(
        request.request_type for request in attempt.decision.assessment.update_requests
    ) == (UpdateRequestType.REWEIGHT_SCENARIOS,)
    assert attempt.diagnostics.repeated_stress_moderation_applied is False
    assert attempt.diagnostics.relapse_moderation_applied is True
    assert attempt.diagnostics.unresolved_stress_moderation_applied is True
    assert attempt.diagnostics.calibration_applied is True
    assert attempt.diagnostics.hysteresis_applied is True
    assert attempt.diagnostics.proposed_update_strength == (
        "switch_demand_regime_plus_widen_uncertainty"
    )
    assert attempt.diagnostics.final_update_strength == "reweight_scenarios"
    assert (
        attempt.diagnostics.calibration_reason
        == "relapse_with_unresolved_stress_load"
    )
    assert attempt.diagnostics.moderation_reason == "relapse_with_unresolved_stress_load"


def test_llm_orchestrator_does_not_moderate_relapse_after_normalization() -> None:
    system_state = SimulationState(
        benchmark_id="serial_3_echelon",
        time_index=3,
        inventory_level=(12.0, 18.0, 22.0),
        stage_names=("retailer", "regional_dc", "plant"),
        pipeline_inventory=(3.0, 2.0, 1.0),
        backorder_level=(0.0, 0.0, 0.0),
        regime_label=RegimeLabel.DEMAND_REGIME_SHIFT,
    )
    observation = Observation(
        time_index=3,
        demand_evidence=DemandEvidence(
            history=(14.0, 11.0, 14.0),
            latest_realization=(14.0,),
            stage_index=1,
        ),
        leadtime_evidence=LeadTimeEvidence(
            history=(2.0, 2.0, 2.0),
            latest_realization=(2.0,),
            upstream_stage_index=2,
            downstream_stage_index=1,
        ),
        regime_label=RegimeLabel.DEMAND_REGIME_SHIFT,
    )
    evidence = RuntimeEvidence(
        time_index=3,
        demand=observation.demand_evidence,
        leadtime=observation.leadtime_evidence,
        scenario_families=(RegimeLabel.NORMAL, RegimeLabel.DEMAND_REGIME_SHIFT),
        demand_baseline_value=10.0,
        leadtime_baseline_value=2.0,
    )
    orchestrator = LLMOrchestrator(
        client=FakeLLMClient(),
        agent_config=AgentConfig(
            enabled_regime_labels=(RegimeLabel.NORMAL, RegimeLabel.DEMAND_REGIME_SHIFT),
            allowed_update_types=(
                UpdateRequestType.KEEP_CURRENT,
                UpdateRequestType.REWEIGHT_SCENARIOS,
                UpdateRequestType.SWITCH_DEMAND_REGIME,
                UpdateRequestType.WIDEN_UNCERTAINTY,
            ),
            allowed_tool_classes=(ToolClass.DETERMINISTIC_STATISTICAL,),
            minimum_confidence=0.6,
            max_tool_steps=3,
            llm_client_mode="fake",
            llm_model_name="gpt-4o-mini",
        ),
    )

    attempt = orchestrator.decide_with_diagnostics(
        LLMOrchestrationInput(
            mission=MissionSpec(
                mission_id="serial_mission",
                objective="Allow bounded LLM orchestration only.",
                admissible_tool_ids=("forecast_tool", "leadtime_tool", "scenario_tool"),
            ),
            system_state=system_state,
            observation=observation,
            evidence=evidence,
            candidate_tool_ids=("forecast_tool", "leadtime_tool", "scenario_tool"),
            recent_regime_history=(
                RegimeLabel.NORMAL,
                RegimeLabel.DEMAND_REGIME_SHIFT,
                RegimeLabel.RECOVERY,
            ),
            recent_stress_reference_demand_value=14.0,
            recent_update_request_history=(
                (
                    UpdateRequestType.SWITCH_DEMAND_REGIME,
                    UpdateRequestType.WIDEN_UNCERTAINTY,
                ),
                (UpdateRequestType.KEEP_CURRENT,),
            ),
        ),
        (
            ToolSpec(
                tool_id="forecast_tool",
                tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
                supported_subgoals=(OperationalSubgoal.QUERY_UNCERTAINTY,),
                description="forecast",
            ),
            ToolSpec(
                tool_id="leadtime_tool",
                tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
                supported_subgoals=(OperationalSubgoal.QUERY_UNCERTAINTY,),
                description="leadtime",
            ),
            ToolSpec(
                tool_id="scenario_tool",
                tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
                supported_subgoals=(OperationalSubgoal.UPDATE_UNCERTAINTY,),
                description="scenario",
            ),
        ),
    )

    assert attempt.decision is not None
    assert tuple(
        request.request_type for request in attempt.decision.assessment.update_requests
    ) == (
        UpdateRequestType.SWITCH_DEMAND_REGIME,
        UpdateRequestType.WIDEN_UNCERTAINTY,
    )
    assert attempt.diagnostics.relapse_moderation_applied is False
    assert attempt.diagnostics.unresolved_stress_moderation_applied is False
    assert attempt.diagnostics.calibration_applied is False


def test_llm_orchestrator_does_not_moderate_relapse_when_demand_is_materially_stronger() -> None:
    system_state = SimulationState(
        benchmark_id="serial_3_echelon",
        time_index=3,
        inventory_level=(0.0, 0.0, 5.6668),
        stage_names=("retailer", "regional_dc", "plant"),
        pipeline_inventory=(30.0, 34.3332, 43.670048),
        backorder_level=(15.0, 0.8888, 0.0),
        regime_label=RegimeLabel.DEMAND_REGIME_SHIFT,
    )
    observation = Observation(
        time_index=3,
        demand_evidence=DemandEvidence(
            history=(14.0, 11.0, 15.0),
            latest_realization=(15.0,),
            stage_index=1,
        ),
        leadtime_evidence=LeadTimeEvidence(
            history=(2.0, 2.0, 2.0),
            latest_realization=(2.0,),
            upstream_stage_index=2,
            downstream_stage_index=1,
        ),
        regime_label=RegimeLabel.DEMAND_REGIME_SHIFT,
    )
    evidence = RuntimeEvidence(
        time_index=3,
        demand=observation.demand_evidence,
        leadtime=observation.leadtime_evidence,
        scenario_families=(RegimeLabel.NORMAL, RegimeLabel.DEMAND_REGIME_SHIFT),
        demand_baseline_value=10.0,
        leadtime_baseline_value=2.0,
    )
    orchestrator = LLMOrchestrator(
        client=FakeLLMClient(),
        agent_config=AgentConfig(
            enabled_regime_labels=(RegimeLabel.NORMAL, RegimeLabel.DEMAND_REGIME_SHIFT),
            allowed_update_types=(
                UpdateRequestType.KEEP_CURRENT,
                UpdateRequestType.REWEIGHT_SCENARIOS,
                UpdateRequestType.SWITCH_DEMAND_REGIME,
                UpdateRequestType.WIDEN_UNCERTAINTY,
            ),
            allowed_tool_classes=(ToolClass.DETERMINISTIC_STATISTICAL,),
            minimum_confidence=0.6,
            max_tool_steps=3,
            llm_client_mode="fake",
            llm_model_name="gpt-4o-mini",
        ),
    )

    attempt = orchestrator.decide_with_diagnostics(
        LLMOrchestrationInput(
            mission=MissionSpec(
                mission_id="serial_mission",
                objective="Allow bounded LLM orchestration only.",
                admissible_tool_ids=("forecast_tool", "leadtime_tool", "scenario_tool"),
            ),
            system_state=system_state,
            observation=observation,
            evidence=evidence,
            candidate_tool_ids=("forecast_tool", "leadtime_tool", "scenario_tool"),
            recent_regime_history=(
                RegimeLabel.NORMAL,
                RegimeLabel.DEMAND_REGIME_SHIFT,
                RegimeLabel.RECOVERY,
            ),
            recent_stress_reference_demand_value=14.0,
            recent_update_request_history=(
                (
                    UpdateRequestType.SWITCH_DEMAND_REGIME,
                    UpdateRequestType.WIDEN_UNCERTAINTY,
                ),
                (UpdateRequestType.KEEP_CURRENT,),
            ),
        ),
        (
            ToolSpec(
                tool_id="forecast_tool",
                tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
                supported_subgoals=(OperationalSubgoal.QUERY_UNCERTAINTY,),
                description="forecast",
            ),
            ToolSpec(
                tool_id="leadtime_tool",
                tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
                supported_subgoals=(OperationalSubgoal.QUERY_UNCERTAINTY,),
                description="leadtime",
            ),
            ToolSpec(
                tool_id="scenario_tool",
                tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
                supported_subgoals=(OperationalSubgoal.UPDATE_UNCERTAINTY,),
                description="scenario",
            ),
        ),
    )

    assert attempt.decision is not None
    assert tuple(
        request.request_type for request in attempt.decision.assessment.update_requests
    ) == (
        UpdateRequestType.SWITCH_DEMAND_REGIME,
        UpdateRequestType.WIDEN_UNCERTAINTY,
    )
    assert attempt.diagnostics.relapse_moderation_applied is False
    assert attempt.diagnostics.unresolved_stress_moderation_applied is False
    assert attempt.diagnostics.calibration_applied is False


def test_llm_orchestrator_counts_invalid_output_and_fallback() -> None:
    class InvalidClient:
        provider = "invalid_client"

        def complete(self, request):
            from meio.agents.llm_client import LLMCompletionResponse

            return LLMCompletionResponse(
                model=request.model,
                provider=self.provider,
                content='{"selected_subgoal":"query_uncertainty","candidate_tool_ids":["bad_tool"],'
                '"regime_label":"normal","confidence":0.8,"update_request_types":["keep_current"],'
                '"request_replan":false,"rationale":"bad tool id"}',
            )

    system_state = SimulationState(
        benchmark_id="serial_3_echelon",
        time_index=0,
        inventory_level=(20.0, 30.0, 40.0),
        stage_names=("retailer", "regional_dc", "plant"),
        pipeline_inventory=(0.0, 0.0, 0.0),
        backorder_level=(0.0, 0.0, 0.0),
        regime_label=RegimeLabel.NORMAL,
    )
    observation = Observation(
        time_index=0,
        demand_evidence=DemandEvidence(
            history=(10.0, 11.0, 10.0),
            latest_realization=(10.0,),
            stage_index=1,
        ),
        leadtime_evidence=LeadTimeEvidence(
            history=(2.0, 2.0, 2.0),
            latest_realization=(2.0,),
            upstream_stage_index=2,
            downstream_stage_index=1,
        ),
        regime_label=RegimeLabel.NORMAL,
    )
    evidence = RuntimeEvidence(
        time_index=0,
        demand=observation.demand_evidence,
        leadtime=observation.leadtime_evidence,
        scenario_families=(RegimeLabel.NORMAL,),
    )
    orchestrator = LLMOrchestrator(
        client=InvalidClient(),
        agent_config=AgentConfig(
            enabled_regime_labels=(RegimeLabel.NORMAL,),
            allowed_update_types=(UpdateRequestType.KEEP_CURRENT,),
            allowed_tool_classes=(ToolClass.DETERMINISTIC_STATISTICAL,),
            minimum_confidence=0.6,
            max_tool_steps=3,
            llm_client_mode="fake",
            llm_model_name="gpt-4o-mini",
        ),
    )

    attempt = orchestrator.decide_with_diagnostics(
        LLMOrchestrationInput(
            mission=MissionSpec(
                mission_id="serial_mission",
                objective="Allow bounded LLM orchestration only.",
                admissible_tool_ids=("forecast_tool",),
            ),
            system_state=system_state,
            observation=observation,
            evidence=evidence,
            candidate_tool_ids=("forecast_tool",),
        ),
        (
            ToolSpec(
                tool_id="forecast_tool",
                tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
                supported_subgoals=(OperationalSubgoal.QUERY_UNCERTAINTY,),
                description="forecast",
            ),
        ),
    )

    assert attempt.decision is None
    assert attempt.diagnostics.invalid_output_count == 1
    assert attempt.diagnostics.fallback_count == 1
    assert attempt.diagnostics.successful_response_count == 0
    assert attempt.diagnostics.llm_call_telemetry is not None
    assert attempt.diagnostics.llm_call_telemetry.total_tokens is None
    assert attempt.diagnostics.llm_call_trace is not None
    assert attempt.diagnostics.llm_call_trace.invalid_output is True
    assert attempt.diagnostics.llm_call_trace.fallback_used is True
    assert attempt.diagnostics.validation_failure_reason == "requested_missing_tool"
    assert attempt.diagnostics.llm_call_trace.fallback_reason == "requested_missing_tool"
    assert attempt.diagnostics.llm_call_trace.requested_tool_ids == ("bad_tool",)
    assert attempt.diagnostics.llm_call_trace.unavailable_tool_ids == ("bad_tool",)


def test_llm_orchestrator_records_precise_client_runtime_failure() -> None:
    class FailingClient:
        provider = "openai"

        def complete(self, request):
            raise LLMClientRuntimeError(
                category=ClientErrorCategory.NETWORK_ERROR,
                message_summary="APIConnectionError: Connection error.",
                retry_count=1,
                failure_after_response=False,
            )

    system_state = SimulationState(
        benchmark_id="serial_3_echelon",
        time_index=0,
        inventory_level=(20.0, 30.0, 40.0),
        stage_names=("retailer", "regional_dc", "plant"),
        pipeline_inventory=(0.0, 0.0, 0.0),
        backorder_level=(0.0, 0.0, 0.0),
        regime_label=RegimeLabel.NORMAL,
    )
    observation = Observation(
        time_index=0,
        demand_evidence=DemandEvidence(
            history=(10.0, 10.0, 10.0),
            latest_realization=(10.0,),
            stage_index=1,
        ),
        leadtime_evidence=LeadTimeEvidence(
            history=(2.0, 2.0, 2.0),
            latest_realization=(2.0,),
            upstream_stage_index=2,
            downstream_stage_index=1,
        ),
        regime_label=RegimeLabel.NORMAL,
    )
    evidence = RuntimeEvidence(
        time_index=0,
        demand=observation.demand_evidence,
        leadtime=observation.leadtime_evidence,
        scenario_families=(RegimeLabel.NORMAL,),
    )
    orchestrator = LLMOrchestrator(
        client=FailingClient(),
        agent_config=AgentConfig(
            enabled_regime_labels=(RegimeLabel.NORMAL,),
            allowed_update_types=(UpdateRequestType.KEEP_CURRENT,),
            allowed_tool_classes=(ToolClass.DETERMINISTIC_STATISTICAL,),
            llm_client_mode="real",
            llm_model_name="gpt-4o-mini",
        ),
    )

    attempt = orchestrator.decide_with_diagnostics(
        LLMOrchestrationInput(
            mission=MissionSpec(
                mission_id="serial_mission",
                objective="Allow bounded LLM orchestration only.",
                admissible_tool_ids=("forecast_tool",),
            ),
            system_state=system_state,
            observation=observation,
            evidence=evidence,
            candidate_tool_ids=("forecast_tool",),
        ),
        (
            ToolSpec(
                tool_id="forecast_tool",
                tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
                supported_subgoals=(OperationalSubgoal.QUERY_UNCERTAINTY,),
                description="forecast",
            ),
        ),
    )

    assert attempt.decision is None
    assert attempt.diagnostics.fallback_count == 1
    assert attempt.diagnostics.validation_failure_reason == "network_error"
    assert attempt.diagnostics.llm_call_telemetry is not None
    assert attempt.diagnostics.llm_call_telemetry.client_error_category is ClientErrorCategory.NETWORK_ERROR
    assert attempt.diagnostics.llm_call_telemetry.retry_count == 1
    assert attempt.diagnostics.llm_call_trace is not None
    assert attempt.diagnostics.llm_call_trace.client_error_category is ClientErrorCategory.NETWORK_ERROR
    assert attempt.diagnostics.llm_call_trace.failure_after_response is False
