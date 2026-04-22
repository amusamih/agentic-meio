from __future__ import annotations

import pytest

from meio.agents.llm_client import FakeLLMClient, LLMClientRuntimeError
from meio.agents.llm_orchestrator import LLMOrchestrator
from meio.agents.telemetry import ClientErrorCategory
from meio.agents.runtime import (
    OrchestrationRequest,
    OrchestrationRuntime,
    RuntimeMode,
)
from meio.config.schemas import AgentConfig
from meio.contracts import (
    BoundedTool,
    MissionSpec,
    OperationalSubgoal,
    RegimeLabel,
    ToolClass,
    ToolInvocation,
    ToolResult,
    ToolSpec,
    ToolStatus,
    UpdateRequestType,
)
from meio.forecasting.contracts import ForecastResult
from meio.simulation.evidence import DemandEvidence, LeadTimeEvidence, RuntimeEvidence
from meio.simulation.serial_benchmark import (
    build_serial_benchmark_case,
    build_serial_orchestration_request,
)
from meio.simulation.state import Observation, SimulationState


class FakeTool(BoundedTool):
    def __init__(self, spec: ToolSpec, result_factory):
        self._spec = spec
        self._result_factory = result_factory

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def invoke(self, invocation: ToolInvocation) -> ToolResult:
        return self._result_factory(invocation)


def build_agent_config() -> AgentConfig:
    return AgentConfig(
        enabled_regime_labels=(RegimeLabel.NORMAL,),
        allowed_update_types=(UpdateRequestType.KEEP_CURRENT,),
        allowed_tool_classes=(
            ToolClass.DETERMINISTIC_STATISTICAL,
            ToolClass.LEARNED_NON_LLM,
        ),
        minimum_confidence=0.6,
        max_tool_steps=3,
        allow_replan_requests=True,
        allow_abstain=True,
    )


def build_runtime_context() -> tuple[SimulationState, Observation, RuntimeEvidence]:
    system_state = SimulationState(
        benchmark_id="serial_3_echelon",
        time_index=0,
        inventory_level=(20.0, 30.0, 40.0),
        stage_names=("retailer", "regional_dc", "plant"),
        pipeline_inventory=(0.0, 0.0, 0.0),
        backorder_level=(0.0, 0.0, 0.0),
    )
    observation = Observation(
        time_index=0,
        demand_evidence=DemandEvidence(
            history=(10.0, 10.0, 11.0),
            latest_realization=(11.0,),
            stage_index=1,
        ),
        leadtime_evidence=LeadTimeEvidence(
            history=(2.0, 2.0, 2.0),
            latest_realization=(2.0,),
            upstream_stage_index=3,
            downstream_stage_index=2,
        ),
        regime_label=RegimeLabel.NORMAL,
    )
    evidence = RuntimeEvidence(
        time_index=0,
        demand=observation.demand_evidence,
        leadtime=observation.leadtime_evidence,
        scenario_families=(RegimeLabel.NORMAL,),
        notes=("runtime_test",),
    )
    return system_state, observation, evidence


def test_runtime_abstains_when_confidence_is_below_threshold() -> None:
    runtime = OrchestrationRuntime(agent_config=build_agent_config(), tools=[])
    request = OrchestrationRequest(
        mission=MissionSpec(
            mission_id="serial_mission",
            objective="Stay within bounded orchestration guardrails.",
        ),
        confidence_hint=0.3,
    )

    response = runtime.run(request)

    assert response.signal.abstained is True
    assert response.tool_results == ()
    assert response.step_telemetry is not None
    assert response.step_telemetry.provider is None
    assert response.step_telemetry.abstain_or_no_action is True


def test_runtime_invokes_admissible_tool_and_returns_structured_output() -> None:
    system_state, observation, evidence = build_runtime_context()
    tool = FakeTool(
        spec=ToolSpec(
            tool_id="evidence_tool",
            tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
            supported_subgoals=(OperationalSubgoal.INSPECT_EVIDENCE,),
        ),
        result_factory=lambda invocation: ToolResult(
            tool_id=invocation.tool_id,
            tool_class=invocation.tool_class,
            subgoal=invocation.subgoal,
            status=ToolStatus.SUCCESS,
            structured_output={"summary": "evidence_checked"},
            confidence=0.8,
            provenance="deterministic check",
        ),
    )
    runtime = OrchestrationRuntime(agent_config=build_agent_config(), tools=[tool])
    request = OrchestrationRequest(
        mission=MissionSpec(
            mission_id="serial_mission",
            objective="Inspect evidence before any replan request.",
            admissible_tool_ids=("evidence_tool",),
        ),
        system_state=system_state,
        observation=observation,
        evidence=evidence,
        requested_subgoal=OperationalSubgoal.INSPECT_EVIDENCE,
    )

    response = runtime.run(request)

    assert response.signal.abstained is False
    assert response.signal.tool_sequence == ("evidence_tool",)
    assert response.tool_results[0].structured_output["summary"] == "evidence_checked"
    assert len(response.tool_call_traces) == 1
    assert response.tool_call_traces[0].tool_id == "evidence_tool"
    assert response.step_telemetry is not None
    assert response.step_telemetry.tool_call_count == 1


def test_runtime_sequences_multiple_tools_within_step_limit() -> None:
    system_state, observation, evidence = build_runtime_context()
    evidence_tool = FakeTool(
        spec=ToolSpec(
            tool_id="evidence_tool",
            tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
            supported_subgoals=(OperationalSubgoal.INSPECT_EVIDENCE,),
        ),
        result_factory=lambda invocation: ToolResult(
            tool_id=invocation.tool_id,
            tool_class=invocation.tool_class,
            subgoal=invocation.subgoal,
            status=ToolStatus.SUCCESS,
            structured_output={"summary": "checked"},
            next_tool_id="uncertainty_tool",
            next_subgoal=OperationalSubgoal.UPDATE_UNCERTAINTY,
        ),
    )
    uncertainty_tool = FakeTool(
        spec=ToolSpec(
            tool_id="uncertainty_tool",
            tool_class=ToolClass.LEARNED_NON_LLM,
            supported_subgoals=(OperationalSubgoal.UPDATE_UNCERTAINTY,),
        ),
        result_factory=lambda invocation: ToolResult(
            tool_id=invocation.tool_id,
            tool_class=invocation.tool_class,
            subgoal=invocation.subgoal,
            status=ToolStatus.NO_ACTION,
            structured_output={"update": "deferred"},
        ),
    )
    runtime = OrchestrationRuntime(
        agent_config=build_agent_config(),
        tools=[evidence_tool, uncertainty_tool],
    )
    request = OrchestrationRequest(
        mission=MissionSpec(
            mission_id="serial_mission",
            objective="Inspect evidence and update uncertainty if needed.",
        ),
        system_state=system_state,
        observation=observation,
        evidence=evidence,
        requested_subgoal=OperationalSubgoal.INSPECT_EVIDENCE,
    )

    response = runtime.run(request)

    assert response.signal.tool_sequence == ("evidence_tool", "uncertainty_tool")
    assert response.memory.step_count == 2
    assert response.signal.no_action is True


def test_runtime_does_not_auto_append_follow_on_tools_for_explicit_queue() -> None:
    system_state, observation, evidence = build_runtime_context()
    evidence_tool = FakeTool(
        spec=ToolSpec(
            tool_id="evidence_tool",
            tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
            supported_subgoals=(OperationalSubgoal.INSPECT_EVIDENCE,),
        ),
        result_factory=lambda invocation: ToolResult(
            tool_id=invocation.tool_id,
            tool_class=invocation.tool_class,
            subgoal=invocation.subgoal,
            status=ToolStatus.SUCCESS,
            structured_output={"summary": "checked"},
            next_tool_id="uncertainty_tool",
            next_subgoal=OperationalSubgoal.UPDATE_UNCERTAINTY,
        ),
    )
    uncertainty_tool = FakeTool(
        spec=ToolSpec(
            tool_id="uncertainty_tool",
            tool_class=ToolClass.LEARNED_NON_LLM,
            supported_subgoals=(OperationalSubgoal.UPDATE_UNCERTAINTY,),
        ),
        result_factory=lambda invocation: ToolResult(
            tool_id=invocation.tool_id,
            tool_class=invocation.tool_class,
            subgoal=invocation.subgoal,
            status=ToolStatus.NO_ACTION,
            structured_output={"update": "deferred"},
        ),
    )
    runtime = OrchestrationRuntime(
        agent_config=build_agent_config(),
        tools=[evidence_tool, uncertainty_tool],
    )
    request = OrchestrationRequest(
        mission=MissionSpec(
            mission_id="serial_mission",
            objective="Respect the explicit tool queue without hidden follow-on chaining.",
        ),
        system_state=system_state,
        observation=observation,
        evidence=evidence,
        requested_subgoal=OperationalSubgoal.INSPECT_EVIDENCE,
        candidate_tool_ids=("evidence_tool",),
    )

    response = runtime.run(request)

    assert response.signal.tool_sequence == ("evidence_tool",)
    assert response.memory.step_count == 1


def test_runtime_rejects_tool_that_claims_order_authority() -> None:
    system_state, observation, evidence = build_runtime_context()
    unsafe_tool = FakeTool(
        spec=ToolSpec(
            tool_id="unsafe_tool",
            tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
            supported_subgoals=(OperationalSubgoal.INSPECT_EVIDENCE,),
            produces_raw_orders=True,
        ),
        result_factory=lambda invocation: ToolResult(
            tool_id=invocation.tool_id,
            tool_class=invocation.tool_class,
            subgoal=invocation.subgoal,
            status=ToolStatus.SUCCESS,
            structured_output={"summary": "should never run"},
        ),
    )
    runtime = OrchestrationRuntime(agent_config=build_agent_config(), tools=[unsafe_tool])
    request = OrchestrationRequest(
        mission=MissionSpec(
            mission_id="serial_mission",
            objective="Preserve optimizer-only order authority.",
        ),
        system_state=system_state,
        observation=observation,
        evidence=evidence,
        requested_subgoal=OperationalSubgoal.INSPECT_EVIDENCE,
    )

    response = runtime.run(request)

    assert response.tool_results == ()
    assert response.admissibility_checks[0].allowed is False
    assert "raw replenishment orders" in response.admissibility_checks[0].reason


def test_runtime_rejects_tool_result_that_emits_raw_orders() -> None:
    system_state, observation, evidence = build_runtime_context()
    unsafe_result_tool = FakeTool(
        spec=ToolSpec(
            tool_id="unsafe_result_tool",
            tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
            supported_subgoals=(OperationalSubgoal.INSPECT_EVIDENCE,),
        ),
        result_factory=lambda invocation: ToolResult(
            tool_id=invocation.tool_id,
            tool_class=invocation.tool_class,
            subgoal=invocation.subgoal,
            status=ToolStatus.SUCCESS,
            structured_output={"summary": "invalid"},
            emits_raw_orders=True,
        ),
    )
    runtime = OrchestrationRuntime(
        agent_config=build_agent_config(),
        tools=[unsafe_result_tool],
    )
    request = OrchestrationRequest(
        mission=MissionSpec(
            mission_id="serial_mission",
            objective="Reject unsafe tool outputs.",
        ),
        system_state=system_state,
        observation=observation,
        evidence=evidence,
        requested_subgoal=OperationalSubgoal.INSPECT_EVIDENCE,
    )

    with pytest.raises(ValueError, match="raw replenishment orders"):
        runtime.run(request)


def test_runtime_honors_explicit_no_action_request() -> None:
    runtime = OrchestrationRuntime(agent_config=build_agent_config(), tools=[])
    request = OrchestrationRequest(
        mission=MissionSpec(
            mission_id="serial_mission",
            objective="Allow bounded explicit no-action outcomes.",
        ),
        requested_subgoal=OperationalSubgoal.NO_ACTION,
    )

    response = runtime.run(request)

    assert response.signal.no_action is True
    assert response.signal.abstained is False
    assert response.tool_results == ()


def test_runtime_blocks_tool_outside_human_admissible_tool_set() -> None:
    system_state, observation, evidence = build_runtime_context()
    tool = FakeTool(
        spec=ToolSpec(
            tool_id="evidence_tool",
            tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
            supported_subgoals=(OperationalSubgoal.INSPECT_EVIDENCE,),
        ),
        result_factory=lambda invocation: ToolResult(
            tool_id=invocation.tool_id,
            tool_class=invocation.tool_class,
            subgoal=invocation.subgoal,
            status=ToolStatus.SUCCESS,
            structured_output={"summary": "should_not_run"},
        ),
    )
    runtime = OrchestrationRuntime(agent_config=build_agent_config(), tools=[tool])
    request = OrchestrationRequest(
        mission=MissionSpec(
            mission_id="serial_mission",
            objective="Keep tool use inside the human-defined admissible set.",
            admissible_tool_ids=("different_tool",),
        ),
        system_state=system_state,
        observation=observation,
        evidence=evidence,
        requested_subgoal=OperationalSubgoal.INSPECT_EVIDENCE,
    )

    response = runtime.run(request)

    assert response.tool_results == ()
    assert response.admissibility_checks[0].allowed is False
    assert "human-defined admissible tool set" in response.admissibility_checks[0].reason


def test_runtime_rejects_tool_result_with_mismatched_subgoal() -> None:
    system_state, observation, evidence = build_runtime_context()
    mismatched_tool = FakeTool(
        spec=ToolSpec(
            tool_id="evidence_tool",
            tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
            supported_subgoals=(OperationalSubgoal.INSPECT_EVIDENCE,),
        ),
        result_factory=lambda invocation: ToolResult(
            tool_id=invocation.tool_id,
            tool_class=invocation.tool_class,
            subgoal=OperationalSubgoal.CLASSIFY_REGIME,
            status=ToolStatus.SUCCESS,
            structured_output={"summary": "invalid"},
        ),
    )
    runtime = OrchestrationRuntime(agent_config=build_agent_config(), tools=[mismatched_tool])
    request = OrchestrationRequest(
        mission=MissionSpec(
            mission_id="serial_mission",
            objective="Reject tools that violate the invoked subgoal boundary.",
        ),
        system_state=system_state,
        observation=observation,
        evidence=evidence,
        requested_subgoal=OperationalSubgoal.INSPECT_EVIDENCE,
    )

    with pytest.raises(ValueError, match="invoked operational subgoal"):
        runtime.run(request)


def test_runtime_runs_against_serial_benchmark_request_boundary() -> None:
    tool = FakeTool(
        spec=ToolSpec(
            tool_id="evidence_tool",
            tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
            supported_subgoals=(OperationalSubgoal.INSPECT_EVIDENCE,),
        ),
        result_factory=lambda invocation: ToolResult(
            tool_id=invocation.tool_id,
            tool_class=invocation.tool_class,
            subgoal=invocation.subgoal,
            status=ToolStatus.SUCCESS,
            structured_output={"scenario_families": invocation.evidence.scenario_families},
        ),
    )
    runtime = OrchestrationRuntime(agent_config=build_agent_config(), tools=[tool])
    case = build_serial_benchmark_case()
    request = build_serial_orchestration_request(
        case=case,
        mission=MissionSpec(
            mission_id="serial_runtime",
            objective="Inspect serial benchmark evidence without emitting orders.",
            admissible_tool_ids=("evidence_tool",),
        ),
    )

    response = runtime.run(request)

    assert response.signal.tool_sequence == ("evidence_tool",)
    assert response.tool_results[0].structured_output["scenario_families"] == (RegimeLabel.NORMAL,)


def test_runtime_accepts_typed_domain_payloads_in_structured_output() -> None:
    system_state, observation, evidence = build_runtime_context()
    tool = FakeTool(
        spec=ToolSpec(
            tool_id="forecast_tool",
            tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
            supported_subgoals=(OperationalSubgoal.QUERY_UNCERTAINTY,),
        ),
        result_factory=lambda invocation: ToolResult(
            tool_id=invocation.tool_id,
            tool_class=invocation.tool_class,
            subgoal=invocation.subgoal,
            status=ToolStatus.SUCCESS,
            structured_output={
                "forecast": ForecastResult(
                    horizon=2,
                    point_forecast=(9.0, 10.0),
                    uncertainty_scale=(1.0, 1.1),
                    provenance="forecast_tool",
                )
            },
        ),
    )
    runtime = OrchestrationRuntime(agent_config=build_agent_config(), tools=[tool])
    request = OrchestrationRequest(
        mission=MissionSpec(
            mission_id="serial_mission",
            objective="Allow typed domain payloads from bounded tools.",
        ),
        system_state=system_state,
        observation=observation,
        evidence=evidence,
        requested_subgoal=OperationalSubgoal.QUERY_UNCERTAINTY,
    )

    response = runtime.run(request)

    assert response.signal.tool_sequence == ("forecast_tool",)
    assert isinstance(response.tool_results[0].structured_output["forecast"], ForecastResult)


def test_runtime_rejects_nested_mapping_output_payloads() -> None:
    system_state, observation, evidence = build_runtime_context()
    tool = FakeTool(
        spec=ToolSpec(
            tool_id="evidence_tool",
            tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
            supported_subgoals=(OperationalSubgoal.INSPECT_EVIDENCE,),
        ),
        result_factory=lambda invocation: ToolResult(
            tool_id=invocation.tool_id,
            tool_class=invocation.tool_class,
            subgoal=invocation.subgoal,
            status=ToolStatus.SUCCESS,
            structured_output={"nested": {"free_form": "not_allowed"}},
        ),
    )
    runtime = OrchestrationRuntime(agent_config=build_agent_config(), tools=[tool])
    request = OrchestrationRequest(
        mission=MissionSpec(
            mission_id="serial_mission",
            objective="Reject ungoverned nested output payloads.",
        ),
        system_state=system_state,
        observation=observation,
        evidence=evidence,
        requested_subgoal=OperationalSubgoal.INSPECT_EVIDENCE,
    )

    with pytest.raises(TypeError, match="typed MEIO contract objects"):
        runtime.run(request)


def test_runtime_supports_llm_orchestration_with_fake_client() -> None:
    system_state, observation, evidence = build_runtime_context()
    forecast_tool = FakeTool(
        spec=ToolSpec(
            tool_id="forecast_tool",
            tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
            supported_subgoals=(OperationalSubgoal.QUERY_UNCERTAINTY,),
        ),
        result_factory=lambda invocation: ToolResult(
            tool_id=invocation.tool_id,
            tool_class=invocation.tool_class,
            subgoal=invocation.subgoal,
            status=ToolStatus.SUCCESS,
            structured_output={"summary": "forecast_checked"},
            next_tool_id="leadtime_tool",
            next_subgoal=OperationalSubgoal.QUERY_UNCERTAINTY,
        ),
    )
    leadtime_tool = FakeTool(
        spec=ToolSpec(
            tool_id="leadtime_tool",
            tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
            supported_subgoals=(OperationalSubgoal.QUERY_UNCERTAINTY,),
        ),
        result_factory=lambda invocation: ToolResult(
            tool_id=invocation.tool_id,
            tool_class=invocation.tool_class,
            subgoal=invocation.subgoal,
            status=ToolStatus.SUCCESS,
            structured_output={"summary": "leadtime_checked"},
            next_tool_id="scenario_tool",
            next_subgoal=OperationalSubgoal.UPDATE_UNCERTAINTY,
        ),
    )
    scenario_tool = FakeTool(
        spec=ToolSpec(
            tool_id="scenario_tool",
            tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
            supported_subgoals=(OperationalSubgoal.UPDATE_UNCERTAINTY,),
        ),
        result_factory=lambda invocation: ToolResult(
            tool_id=invocation.tool_id,
            tool_class=invocation.tool_class,
            subgoal=invocation.subgoal,
            status=ToolStatus.SUCCESS,
            structured_output={"summary": "scenario_checked"},
            request_replan=bool(invocation.agent_assessment and invocation.agent_assessment.request_replan),
        ),
    )
    runtime = OrchestrationRuntime(
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
            allow_replan_requests=True,
            allow_abstain=True,
            llm_client_mode="fake",
            llm_model_name="gpt-4o-mini",
        ),
        tools=[forecast_tool, leadtime_tool, scenario_tool],
        mode=RuntimeMode.LLM_ORCHESTRATION,
        llm_orchestrator=LLMOrchestrator(
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
                allow_replan_requests=True,
                allow_abstain=True,
                llm_client_mode="fake",
                llm_model_name="gpt-4o-mini",
            ),
        ),
    )
    request = OrchestrationRequest(
        mission=MissionSpec(
            mission_id="serial_mission",
            objective="Allow bounded LLM tool orchestration.",
            admissible_tool_ids=("forecast_tool", "leadtime_tool", "scenario_tool"),
        ),
        system_state=SimulationState(
            benchmark_id=system_state.benchmark_id,
            time_index=1,
            inventory_level=system_state.inventory_level,
            stage_names=system_state.stage_names,
            pipeline_inventory=system_state.pipeline_inventory,
            backorder_level=system_state.backorder_level,
            regime_label=RegimeLabel.DEMAND_REGIME_SHIFT,
        ),
        observation=Observation(
            time_index=1,
            demand_evidence=observation.demand_evidence,
            leadtime_evidence=observation.leadtime_evidence,
            regime_label=RegimeLabel.DEMAND_REGIME_SHIFT,
        ),
        evidence=evidence,
    )

    response = runtime.run(request)

    assert response.agent_assessment is not None
    assert response.llm_diagnostics is not None
    assert response.llm_diagnostics.successful_response_count == 1
    assert response.llm_diagnostics.llm_call_trace is not None
    assert response.step_telemetry is not None
    assert response.step_telemetry.provider == "fake_llm_client"
    assert response.step_telemetry.total_tokens == 144
    assert response.signal.tool_sequence == ("forecast_tool", "leadtime_tool", "scenario_tool")
    assert response.signal.request_replan is True
    assert len(response.tool_call_traces) == 3


def test_runtime_falls_back_safely_on_invalid_llm_output() -> None:
    class InvalidClient:
        def complete(self, request):
            from meio.agents.llm_client import LLMCompletionResponse

            return LLMCompletionResponse(
                model=request.model,
                provider="invalid_client",
                content='{"selected_subgoal":"query_uncertainty","candidate_tool_ids":["unsafe_tool"],'
                '"regime_label":"normal","confidence":0.8,"update_request_types":["keep_current"],'
                '"request_replan":false,"rationale":"invalid tool."}',
            )

    system_state, observation, evidence = build_runtime_context()
    runtime = OrchestrationRuntime(
        agent_config=AgentConfig(
            enabled_regime_labels=(RegimeLabel.NORMAL,),
            allowed_update_types=(UpdateRequestType.KEEP_CURRENT,),
            allowed_tool_classes=(ToolClass.DETERMINISTIC_STATISTICAL,),
            minimum_confidence=0.6,
            max_tool_steps=3,
            allow_replan_requests=True,
            allow_abstain=True,
            llm_client_mode="fake",
            llm_model_name="gpt-4o-mini",
        ),
        tools=[],
        mode=RuntimeMode.LLM_ORCHESTRATION,
        llm_orchestrator=LLMOrchestrator(
            client=InvalidClient(),
            agent_config=AgentConfig(
                enabled_regime_labels=(RegimeLabel.NORMAL,),
                allowed_update_types=(UpdateRequestType.KEEP_CURRENT,),
                allowed_tool_classes=(ToolClass.DETERMINISTIC_STATISTICAL,),
                minimum_confidence=0.6,
                max_tool_steps=3,
                allow_replan_requests=True,
                allow_abstain=True,
                llm_client_mode="fake",
                llm_model_name="gpt-4o-mini",
            ),
        ),
    )
    request = OrchestrationRequest(
        mission=MissionSpec(
            mission_id="serial_mission",
            objective="Reject invalid LLM output safely.",
            admissible_tool_ids=("forecast_tool",),
        ),
        system_state=system_state,
        observation=observation,
        evidence=evidence,
    )

    response = runtime.run(request)

    assert response.signal.abstained is True
    assert response.llm_diagnostics is not None
    assert response.llm_diagnostics.fallback_count == 1
    assert response.step_telemetry is not None
    assert response.step_telemetry.fallback_used is True
    assert response.tool_call_traces == ()
    assert response.tool_results == ()


def test_runtime_records_client_runtime_failure_telemetry() -> None:
    class FailingClient:
        provider = "openai"

        def complete(self, request):
            raise LLMClientRuntimeError(
                category=ClientErrorCategory.NETWORK_ERROR,
                message_summary="APIConnectionError: Connection error.",
                retry_count=1,
                failure_after_response=False,
            )

    system_state, observation, evidence = build_runtime_context()
    runtime = OrchestrationRuntime(
        agent_config=AgentConfig(
            enabled_regime_labels=(RegimeLabel.NORMAL,),
            allowed_update_types=(UpdateRequestType.KEEP_CURRENT,),
            allowed_tool_classes=(ToolClass.DETERMINISTIC_STATISTICAL,),
            minimum_confidence=0.6,
            max_tool_steps=3,
            allow_replan_requests=True,
            allow_abstain=True,
            llm_client_mode="real",
            llm_model_name="gpt-4o-mini",
        ),
        tools=[],
        mode=RuntimeMode.LLM_ORCHESTRATION,
        llm_orchestrator=LLMOrchestrator(
            client=FailingClient(),
            agent_config=AgentConfig(
                enabled_regime_labels=(RegimeLabel.NORMAL,),
                allowed_update_types=(UpdateRequestType.KEEP_CURRENT,),
                allowed_tool_classes=(ToolClass.DETERMINISTIC_STATISTICAL,),
                minimum_confidence=0.6,
                max_tool_steps=3,
                allow_replan_requests=True,
                allow_abstain=True,
                llm_client_mode="real",
                llm_model_name="gpt-4o-mini",
            ),
        ),
    )
    request = OrchestrationRequest(
        mission=MissionSpec(
            mission_id="serial_mission",
            objective="Record client runtime failures safely.",
            admissible_tool_ids=("forecast_tool",),
        ),
        system_state=system_state,
        observation=observation,
        evidence=evidence,
        candidate_tool_ids=("forecast_tool",),
    )

    response = runtime.run(request)

    assert response.signal.abstained is True
    assert response.step_telemetry is not None
    assert response.step_telemetry.client_error_category is ClientErrorCategory.NETWORK_ERROR
    assert response.step_telemetry.retry_count == 1
    assert response.step_telemetry.failure_after_response is False


def test_runtime_normalizes_request_replan_with_tools_into_query_path() -> None:
    class ReplanClient:
        provider = "replan_client"

        def complete(self, request):
            from meio.agents.llm_client import LLMCompletionResponse

            return LLMCompletionResponse(
                model=request.model,
                provider=self.provider,
                content=(
                    '{"selected_subgoal":"request_replan","candidate_tool_ids":'
                    '["forecast_tool","leadtime_tool","scenario_tool"],'
                    '"regime_label":"demand_regime_shift","confidence":0.86,'
                    '"update_request_types":["switch_demand_regime","widen_uncertainty"],'
                    '"request_replan":true,'
                    '"rationale":"Demand materially exceeds baseline and planning inputs should change."}'
                ),
            )

    system_state, observation, evidence = build_runtime_context()
    runtime = OrchestrationRuntime(
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
            allow_replan_requests=True,
            allow_abstain=True,
            llm_client_mode="fake",
            llm_model_name="gpt-4o-mini",
        ),
        tools=[
            FakeTool(
                spec=ToolSpec(
                    tool_id="forecast_tool",
                    tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
                    supported_subgoals=(OperationalSubgoal.QUERY_UNCERTAINTY,),
                ),
                result_factory=lambda invocation: ToolResult(
                    tool_id=invocation.tool_id,
                    tool_class=invocation.tool_class,
                    subgoal=invocation.subgoal,
                    status=ToolStatus.SUCCESS,
                    structured_output={"summary": "forecast_checked"},
                    next_tool_id="leadtime_tool",
                    next_subgoal=OperationalSubgoal.QUERY_UNCERTAINTY,
                ),
            ),
            FakeTool(
                spec=ToolSpec(
                    tool_id="leadtime_tool",
                    tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
                    supported_subgoals=(OperationalSubgoal.QUERY_UNCERTAINTY,),
                ),
                result_factory=lambda invocation: ToolResult(
                    tool_id=invocation.tool_id,
                    tool_class=invocation.tool_class,
                    subgoal=invocation.subgoal,
                    status=ToolStatus.SUCCESS,
                    structured_output={"summary": "leadtime_checked"},
                    next_tool_id="scenario_tool",
                    next_subgoal=OperationalSubgoal.UPDATE_UNCERTAINTY,
                ),
            ),
            FakeTool(
                spec=ToolSpec(
                    tool_id="scenario_tool",
                    tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
                    supported_subgoals=(OperationalSubgoal.UPDATE_UNCERTAINTY,),
                ),
                result_factory=lambda invocation: ToolResult(
                    tool_id=invocation.tool_id,
                    tool_class=invocation.tool_class,
                    subgoal=invocation.subgoal,
                    status=ToolStatus.SUCCESS,
                    structured_output={"summary": "scenario_checked"},
                    request_replan=True,
                ),
            ),
        ],
        mode=RuntimeMode.LLM_ORCHESTRATION,
        llm_orchestrator=LLMOrchestrator(
            client=ReplanClient(),
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
                allow_replan_requests=True,
                allow_abstain=True,
                llm_client_mode="fake",
                llm_model_name="gpt-4o-mini",
            ),
        ),
    )
    request = OrchestrationRequest(
        mission=MissionSpec(
            mission_id="serial_mission",
            objective="Normalize bounded live replan requests into the tool path.",
            admissible_tool_ids=("forecast_tool", "leadtime_tool", "scenario_tool"),
        ),
        system_state=SimulationState(
            benchmark_id=system_state.benchmark_id,
            time_index=1,
            inventory_level=system_state.inventory_level,
            stage_names=system_state.stage_names,
            pipeline_inventory=system_state.pipeline_inventory,
            backorder_level=system_state.backorder_level,
            regime_label=RegimeLabel.DEMAND_REGIME_SHIFT,
        ),
        observation=Observation(
            time_index=1,
            demand_evidence=observation.demand_evidence,
            leadtime_evidence=observation.leadtime_evidence,
            regime_label=RegimeLabel.DEMAND_REGIME_SHIFT,
        ),
        evidence=evidence,
    )

    response = runtime.run(request)

    assert response.signal.selected_subgoal is OperationalSubgoal.UPDATE_UNCERTAINTY
    assert response.signal.request_replan is True
    assert response.signal.tool_sequence == ("forecast_tool", "leadtime_tool", "scenario_tool")
    assert len(response.tool_results) == 3
