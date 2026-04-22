"""Bounded orchestration runtime skeleton for the first MEIO milestone."""

from __future__ import annotations

from dataclasses import is_dataclass
from dataclasses import dataclass, field, replace
from enum import StrEnum
from time import perf_counter
from typing import TYPE_CHECKING
from typing import Iterable

from meio.agents.llm_client import LLMClientRuntimeError
from meio.agents.external_evidence_tool import (
    ExternalEvidenceInterpretation,
    apply_external_evidence_bias,
)
from meio.agents.telemetry import (
    LLMCallTelemetry,
    OrchestrationStepTelemetry,
    ToolCallTrace,
)
from meio.config.schemas import AgentConfig
from meio.contracts import (
    AgentAssessment,
    AgentSignal,
    BoundedTool,
    MissionSpec,
    OperationalSubgoal,
    RegimeLabel,
    ToolInvocation,
    ToolResult,
    ToolSpec,
    ToolStatus,
    UpdateRequestType,
)
from meio.forecasting.contracts import ForecastResult
from meio.leadtime.contracts import LeadTimeResult
from meio.optimization.contracts import OptimizationResult
from meio.scenarios.contracts import ScenarioSummary, ScenarioUpdateResult
from meio.simulation.evidence import RuntimeEvidence
from meio.simulation.state import Observation, SimulationState

if TYPE_CHECKING:
    from meio.agents.llm_orchestrator import (
        LLMDecisionAttempt,
        LLMDecisionDiagnostics,
        LLMOrchestrationDecision,
        LLMOrchestrator,
    )


@dataclass(frozen=True, slots=True)
class AdmissibilityCheck:
    """Result of a bounded admissibility or safety check."""

    tool_id: str | None
    allowed: bool
    reason: str

    def __post_init__(self) -> None:
        if self.tool_id is not None and not self.tool_id.strip():
            raise ValueError("tool_id must be non-empty when provided.")
        if not self.reason.strip():
            raise ValueError("reason must be a non-empty string.")


@dataclass(frozen=True, slots=True)
class OrchestrationMemory:
    """Bounded runtime state carried across orchestration steps."""

    step_count: int = 0
    tool_sequence: tuple[str, ...] = field(default_factory=tuple)
    last_subgoal: OperationalSubgoal | None = None
    abstained_last_step: bool = False

    def __post_init__(self) -> None:
        if self.step_count < 0:
            raise ValueError("step_count must be non-negative.")
        object.__setattr__(self, "tool_sequence", tuple(self.tool_sequence))
        for tool_id in self.tool_sequence:
            if not tool_id.strip():
                raise ValueError("tool_sequence must contain non-empty tool identifiers.")
        if self.last_subgoal is not None and not isinstance(
            self.last_subgoal, OperationalSubgoal
        ):
            raise TypeError("last_subgoal must be an OperationalSubgoal when provided.")


@dataclass(frozen=True, slots=True)
class OrchestrationRequest:
    """Inputs for one bounded orchestration-runtime step."""

    mission: MissionSpec
    system_state: SimulationState | None = None
    observation: Observation | None = None
    evidence: RuntimeEvidence | None = None
    requested_subgoal: OperationalSubgoal | None = None
    candidate_tool_ids: tuple[str, ...] = field(default_factory=tuple)
    recent_regime_history: tuple[RegimeLabel, ...] = field(default_factory=tuple)
    recent_stress_reference_demand_value: float | None = None
    recent_update_request_history: tuple[tuple[UpdateRequestType, ...], ...] = field(
        default_factory=tuple
    )
    confidence_hint: float | None = None
    memory: OrchestrationMemory | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mission, MissionSpec):
            raise TypeError("mission must be a MissionSpec.")
        if self.system_state is not None and not isinstance(self.system_state, SimulationState):
            raise TypeError("system_state must be a SimulationState when provided.")
        if self.observation is not None and not isinstance(self.observation, Observation):
            raise TypeError("observation must be an Observation when provided.")
        if self.evidence is not None and not isinstance(self.evidence, RuntimeEvidence):
            raise TypeError("evidence must be a RuntimeEvidence when provided.")
        object.__setattr__(self, "candidate_tool_ids", tuple(self.candidate_tool_ids))
        object.__setattr__(self, "recent_regime_history", tuple(self.recent_regime_history))
        object.__setattr__(
            self,
            "recent_update_request_history",
            tuple(
                tuple(update_types)
                for update_types in self.recent_update_request_history
            ),
        )
        if self.requested_subgoal is not None and not isinstance(
            self.requested_subgoal, OperationalSubgoal
        ):
            raise TypeError("requested_subgoal must be an OperationalSubgoal when provided.")
        for tool_id in self.candidate_tool_ids:
            if not tool_id.strip():
                raise ValueError("candidate_tool_ids must contain non-empty tool identifiers.")
        for regime_label in self.recent_regime_history:
            if not isinstance(regime_label, RegimeLabel):
                raise TypeError(
                    "recent_regime_history must contain RegimeLabel values when provided."
                )
        for update_types in self.recent_update_request_history:
            for update_type in update_types:
                if not isinstance(update_type, UpdateRequestType):
                    raise TypeError(
                        "recent_update_request_history must contain "
                        "UpdateRequestType tuples when provided."
                    )
        if self.recent_stress_reference_demand_value is not None:
            if isinstance(self.recent_stress_reference_demand_value, bool) or not isinstance(
                self.recent_stress_reference_demand_value,
                (int, float),
            ):
                raise TypeError(
                    "recent_stress_reference_demand_value must be numeric when provided."
                )
            if self.recent_stress_reference_demand_value < 0.0:
                raise ValueError(
                    "recent_stress_reference_demand_value must be non-negative when provided."
                )
        if self.confidence_hint is not None and not 0.0 <= self.confidence_hint <= 1.0:
            raise ValueError("confidence_hint must be within [0.0, 1.0] when provided.")
        if self.memory is not None and not isinstance(self.memory, OrchestrationMemory):
            raise TypeError("memory must be an OrchestrationMemory when provided.")


@dataclass(frozen=True, slots=True)
class OrchestrationResponse:
    """Structured bounded output from one orchestration-runtime step."""

    signal: AgentSignal
    agent_assessment: AgentAssessment | None = None
    llm_diagnostics: "LLMDecisionDiagnostics | None" = None
    step_telemetry: OrchestrationStepTelemetry | None = None
    tool_call_traces: tuple[ToolCallTrace, ...] = field(default_factory=tuple)
    tool_results: tuple[ToolResult, ...] = field(default_factory=tuple)
    admissibility_checks: tuple[AdmissibilityCheck, ...] = field(default_factory=tuple)
    memory: OrchestrationMemory = field(default_factory=OrchestrationMemory)

    def __post_init__(self) -> None:
        if not isinstance(self.signal, AgentSignal):
            raise TypeError("signal must be an AgentSignal.")
        if self.agent_assessment is not None and not isinstance(
            self.agent_assessment, AgentAssessment
        ):
            raise TypeError("agent_assessment must be an AgentAssessment when provided.")
        if self.llm_diagnostics is not None:
            from meio.agents.llm_orchestrator import LLMDecisionDiagnostics

            if not isinstance(self.llm_diagnostics, LLMDecisionDiagnostics):
                raise TypeError("llm_diagnostics must be an LLMDecisionDiagnostics when provided.")
        if self.step_telemetry is not None and not isinstance(
            self.step_telemetry,
            OrchestrationStepTelemetry,
        ):
            raise TypeError(
                "step_telemetry must be an OrchestrationStepTelemetry when provided."
            )
        object.__setattr__(self, "tool_call_traces", tuple(self.tool_call_traces))
        object.__setattr__(self, "tool_results", tuple(self.tool_results))
        object.__setattr__(self, "admissibility_checks", tuple(self.admissibility_checks))
        if not isinstance(self.memory, OrchestrationMemory):
            raise TypeError("memory must be an OrchestrationMemory.")
        for trace in self.tool_call_traces:
            if not isinstance(trace, ToolCallTrace):
                raise TypeError("tool_call_traces must contain ToolCallTrace values.")
        for result in self.tool_results:
            if not isinstance(result, ToolResult):
                raise TypeError("tool_results must contain ToolResult values.")
        for check in self.admissibility_checks:
            if not isinstance(check, AdmissibilityCheck):
                raise TypeError("admissibility_checks must contain AdmissibilityCheck values.")


class RuntimeMode(StrEnum):
    """Supported orchestration runtime policies."""

    DETERMINISTIC = "deterministic"
    LLM_ORCHESTRATION = "llm_orchestration"


class OrchestrationRuntime:
    """Minimal bounded runtime for orchestrating tool calls.

    This skeleton supports structured evidence intake, bounded tool selection,
    multi-step tool sequencing, abstain or no-action outcomes, and explicit
    admissibility checks. It does not implement forecasting, optimization, or
    simulation logic.
    """

    def __init__(
        self,
        agent_config: AgentConfig,
        tools: Iterable[BoundedTool],
        *,
        mode: RuntimeMode = RuntimeMode.DETERMINISTIC,
        llm_orchestrator: "LLMOrchestrator | None" = None,
    ) -> None:
        self._agent_config = agent_config
        self._mode = mode
        self._llm_orchestrator = llm_orchestrator
        self._tools: dict[str, BoundedTool] = {}
        for tool in tools:
            spec = tool.spec
            if spec.tool_id in self._tools:
                raise ValueError(f"Duplicate tool_id registered: {spec.tool_id}")
            self._tools[spec.tool_id] = tool
        if self._mode is RuntimeMode.LLM_ORCHESTRATION and self._llm_orchestrator is None:
            raise ValueError("llm_orchestrator must be provided in llm_orchestration mode.")

    def run(self, request: OrchestrationRequest) -> OrchestrationResponse:
        """Run one bounded orchestration step."""

        orchestration_start = perf_counter()
        memory = request.memory or OrchestrationMemory()
        if self._should_abstain(request):
            response = self._build_terminal_response(
                subgoal=OperationalSubgoal.ABSTAIN,
                rationale="Runtime abstained because confidence was below the configured threshold.",
                memory=memory,
                admissibility_checks=(),
                tool_call_traces=(),
                tool_results=(),
                agent_assessment=None,
                llm_diagnostics=None,
            )
            return self._attach_step_telemetry(
                response,
                orchestration_latency_ms=(perf_counter() - orchestration_start) * 1000.0,
            )

        if request.requested_subgoal is OperationalSubgoal.NO_ACTION:
            response = self._build_terminal_response(
                subgoal=OperationalSubgoal.NO_ACTION,
                rationale="Runtime accepted an explicit no-action request.",
                memory=memory,
                admissibility_checks=(),
                tool_call_traces=(),
                tool_results=(),
                agent_assessment=None,
                llm_diagnostics=None,
            )
            return self._attach_step_telemetry(
                response,
                orchestration_latency_ms=(perf_counter() - orchestration_start) * 1000.0,
            )

        if request.requested_subgoal is OperationalSubgoal.ABSTAIN:
            response = self._build_terminal_response(
                subgoal=OperationalSubgoal.ABSTAIN,
                rationale="Runtime accepted an explicit abstain request.",
                memory=memory,
                admissibility_checks=(),
                tool_call_traces=(),
                tool_results=(),
                agent_assessment=None,
                llm_diagnostics=None,
            )
            return self._attach_step_telemetry(
                response,
                orchestration_latency_ms=(perf_counter() - orchestration_start) * 1000.0,
            )

        agent_assessment: AgentAssessment | None = None
        llm_diagnostics = None
        if self._mode is RuntimeMode.LLM_ORCHESTRATION:
            llm_attempt = self._run_llm_policy(request)
            llm_diagnostics = llm_attempt.diagnostics
            if llm_attempt.decision is None:
                response = self._build_terminal_response(
                    subgoal=(
                        OperationalSubgoal.ABSTAIN
                        if self._agent_config.allow_abstain
                        else OperationalSubgoal.NO_ACTION
                    ),
                    rationale=(
                        "Runtime fell back safely because the LLM orchestration output "
                        "was missing, malformed, or out of scope."
                    ),
                    memory=memory,
                    admissibility_checks=(),
                    tool_call_traces=(),
                    tool_results=(),
                    agent_assessment=None,
                    llm_diagnostics=llm_diagnostics,
                )
                return self._attach_step_telemetry(
                    response,
                    orchestration_latency_ms=(perf_counter() - orchestration_start) * 1000.0,
                )
            llm_decision = llm_attempt.decision
            agent_assessment = llm_decision.assessment
            if (
                self._agent_config.allow_abstain
                and agent_assessment.confidence < self._agent_config.minimum_confidence
            ):
                response = self._build_terminal_response(
                    subgoal=OperationalSubgoal.ABSTAIN,
                    rationale=(
                        "Runtime abstained because the LLM orchestration confidence was below "
                        "the configured threshold."
                    ),
                    memory=memory,
                    admissibility_checks=(),
                    tool_call_traces=(),
                    tool_results=(),
                    agent_assessment=agent_assessment,
                    llm_diagnostics=llm_diagnostics,
                )
                return self._attach_step_telemetry(
                    response,
                    orchestration_latency_ms=(perf_counter() - orchestration_start) * 1000.0,
                )
            subgoal = self._normalize_llm_subgoal(
                llm_decision.selected_subgoal,
                llm_decision.selected_tool_ids,
            )
            if subgoal in {OperationalSubgoal.NO_ACTION, OperationalSubgoal.ABSTAIN}:
                response = self._build_terminal_response(
                    subgoal=subgoal,
                    rationale=agent_assessment.rationale,
                    memory=memory,
                    admissibility_checks=(),
                    tool_call_traces=(),
                    tool_results=(),
                    agent_assessment=agent_assessment,
                    llm_diagnostics=llm_diagnostics,
                )
                return self._attach_step_telemetry(
                    response,
                    orchestration_latency_ms=(perf_counter() - orchestration_start) * 1000.0,
                )
            if subgoal is OperationalSubgoal.REQUEST_REPLAN and not llm_decision.selected_tool_ids:
                response = self._build_terminal_response(
                    subgoal=OperationalSubgoal.REQUEST_REPLAN,
                    rationale=agent_assessment.rationale,
                    memory=memory,
                    admissibility_checks=(),
                    tool_call_traces=(),
                    tool_results=(),
                    agent_assessment=agent_assessment,
                    llm_diagnostics=llm_diagnostics,
                    request_replan=agent_assessment.request_replan,
                )
                return self._attach_step_telemetry(
                    response,
                    orchestration_latency_ms=(perf_counter() - orchestration_start) * 1000.0,
                )
            queue = self._seed_tool_queue(subgoal, llm_decision.selected_tool_ids)
            allow_implicit_follow_on = not bool(llm_decision.selected_tool_ids)
        else:
            subgoal = request.requested_subgoal or OperationalSubgoal.INSPECT_EVIDENCE
            queue = self._seed_tool_queue(subgoal, request.candidate_tool_ids)
            allow_implicit_follow_on = not bool(request.candidate_tool_ids)
        max_steps = min(self._agent_config.max_tool_steps, request.mission.max_tool_steps)
        admissibility_checks: list[AdmissibilityCheck] = []
        tool_call_traces: list[ToolCallTrace] = []
        tool_results: list[ToolResult] = []

        while queue and len(tool_results) < max_steps:
            tool_id = queue.pop(0)
            tool = self._tools.get(tool_id)
            check = self._check_admissibility(
                tool_id=tool_id,
                tool=tool,
                mission=request.mission,
                subgoal=subgoal,
            )
            admissibility_checks.append(check)
            if not check.allowed or tool is None:
                continue

            invocation = ToolInvocation(
                tool_id=tool_id,
                tool_class=tool.spec.tool_class,
                subgoal=subgoal,
                evidence=request.evidence,
                system_state=request.system_state,
                observation=request.observation,
                agent_assessment=agent_assessment,
                prior_results=tuple(tool_results),
                step_index=len(tool_results),
            )
            pre_tool_decision = self._tool_decision_snapshot(
                subgoal=subgoal,
                agent_assessment=agent_assessment,
                prior_results=tuple(tool_results),
            )
            pre_tool_optimizer_input = self._tool_optimizer_input_snapshot(
                observation=request.observation,
                prior_results=tuple(tool_results),
            )
            call_start = perf_counter()
            result = tool.invoke(invocation)
            post_tool_decision = self._tool_decision_snapshot(
                subgoal=result.next_subgoal or result.subgoal,
                agent_assessment=agent_assessment,
                prior_results=tuple(tool_results) + (result,),
                current_result=result,
            )
            post_tool_optimizer_input = self._tool_optimizer_input_snapshot(
                observation=request.observation,
                prior_results=tuple(tool_results) + (result,),
            )
            tool_call_traces.append(
                ToolCallTrace(
                    tool_id=tool_id,
                    tool_input=self._summarize_tool_invocation(invocation),
                    tool_output=self._summarize_tool_result(result),
                    success=True,
                    pre_tool_decision=pre_tool_decision,
                    post_tool_decision=post_tool_decision,
                    pre_tool_optimizer_input=pre_tool_optimizer_input,
                    post_tool_optimizer_input=post_tool_optimizer_input,
                    decision_changed=pre_tool_decision != post_tool_decision,
                    optimizer_input_changed=(
                        pre_tool_optimizer_input != post_tool_optimizer_input
                    ),
                    error_type=None,
                    latency_ms=(perf_counter() - call_start) * 1000.0,
                )
            )
            self._validate_tool_result(invocation, tool.spec, result)
            tool_results.append(result)
            memory = OrchestrationMemory(
                step_count=memory.step_count + 1,
                tool_sequence=memory.tool_sequence + (tool_id,),
                last_subgoal=result.next_subgoal or subgoal,
                abstained_last_step=result.status is ToolStatus.ABSTAIN,
            )

            if result.status in {ToolStatus.ABSTAIN, ToolStatus.NO_ACTION}:
                break

            if result.next_subgoal is not None:
                subgoal = result.next_subgoal
            if (
                allow_implicit_follow_on
                and result.next_tool_id
                and result.next_tool_id not in queue
            ):
                queue.append(result.next_tool_id)

        if not tool_results:
            response = self._build_terminal_response(
                subgoal=OperationalSubgoal.NO_ACTION,
                rationale="No admissible bounded tool was available for the requested subgoal.",
                memory=memory,
                admissibility_checks=tuple(admissibility_checks),
                tool_call_traces=tuple(tool_call_traces),
                tool_results=(),
                agent_assessment=agent_assessment,
                llm_diagnostics=llm_diagnostics,
            )
            return self._attach_step_telemetry(
                response,
                orchestration_latency_ms=(perf_counter() - orchestration_start) * 1000.0,
            )

        final_result = tool_results[-1]
        signal = AgentSignal(
            selected_subgoal=final_result.next_subgoal or final_result.subgoal,
            selected_tool_id=tool_results[0].tool_id,
            tool_sequence=tuple(result.tool_id for result in tool_results),
            abstained=final_result.status is ToolStatus.ABSTAIN,
            no_action=final_result.status is ToolStatus.NO_ACTION,
            request_replan=(
                any(result.request_replan for result in tool_results)
                or (agent_assessment.request_replan if agent_assessment is not None else False)
            ),
            rationale=self._build_rationale(tool_results, agent_assessment),
        )
        response = OrchestrationResponse(
            signal=signal,
            agent_assessment=agent_assessment,
            llm_diagnostics=llm_diagnostics,
            tool_call_traces=tuple(tool_call_traces),
            tool_results=tuple(tool_results),
            admissibility_checks=tuple(admissibility_checks),
            memory=memory,
        )
        return self._attach_step_telemetry(
            response,
            orchestration_latency_ms=(perf_counter() - orchestration_start) * 1000.0,
        )

    def _seed_tool_queue(
        self,
        subgoal: OperationalSubgoal,
        candidate_tool_ids: tuple[str, ...],
    ) -> list[str]:
        if candidate_tool_ids:
            return list(candidate_tool_ids)
        return [
            tool_id
            for tool_id, tool in self._tools.items()
            if subgoal in tool.spec.supported_subgoals
        ]

    def _normalize_llm_subgoal(
        self,
        selected_subgoal: OperationalSubgoal,
        selected_tool_ids: tuple[str, ...],
    ) -> OperationalSubgoal:
        if (
            selected_subgoal is OperationalSubgoal.REQUEST_REPLAN
            and selected_tool_ids
        ):
            return OperationalSubgoal.QUERY_UNCERTAINTY
        return selected_subgoal

    def _should_abstain(self, request: OrchestrationRequest) -> bool:
        return (
            self._agent_config.allow_abstain
            and request.confidence_hint is not None
            and request.confidence_hint < self._agent_config.minimum_confidence
        )

    def _check_admissibility(
        self,
        tool_id: str,
        tool: BoundedTool | None,
        mission: MissionSpec,
        subgoal: OperationalSubgoal,
    ) -> AdmissibilityCheck:
        if tool is None:
            return AdmissibilityCheck(tool_id=tool_id, allowed=False, reason="Tool is not registered.")
        if mission.admissible_tool_ids and tool_id not in mission.admissible_tool_ids:
            return AdmissibilityCheck(
                tool_id=tool_id,
                allowed=False,
                reason="Tool is outside the human-defined admissible tool set.",
            )
        if tool.spec.tool_class not in self._agent_config.allowed_tool_classes:
            return AdmissibilityCheck(
                tool_id=tool_id,
                allowed=False,
                reason="Tool class is not enabled in the agent configuration.",
            )
        if tool.spec.produces_raw_orders:
            return AdmissibilityCheck(
                tool_id=tool_id,
                allowed=False,
                reason="Bounded tools may not emit raw replenishment orders.",
            )
        if subgoal not in tool.spec.supported_subgoals:
            return AdmissibilityCheck(
                tool_id=tool_id,
                allowed=False,
                reason="Tool does not support the requested operational subgoal.",
            )
        return AdmissibilityCheck(tool_id=tool_id, allowed=True, reason="Admissible.")

    def _validate_tool_result(
        self,
        invocation: ToolInvocation,
        spec: ToolSpec,
        result: ToolResult,
    ) -> None:
        if result.tool_id != spec.tool_id:
            raise ValueError("ToolResult.tool_id must match the invoked tool.")
        if result.tool_class is not spec.tool_class:
            raise ValueError("ToolResult.tool_class must match the invoked tool class.")
        if result.subgoal is not invocation.subgoal:
            raise ValueError("ToolResult.subgoal must match the invoked operational subgoal.")
        self._validate_structured_output(result.structured_output)
        if result.emits_raw_orders:
            raise ValueError("Bounded tools must not emit raw replenishment orders.")
        if result.request_replan and not self._agent_config.allow_replan_requests:
            raise ValueError("request_replan is disabled in the current agent configuration.")

    def _build_terminal_response(
        self,
        subgoal: OperationalSubgoal,
        rationale: str,
        memory: OrchestrationMemory,
        admissibility_checks: tuple[AdmissibilityCheck, ...],
        tool_call_traces: tuple[ToolCallTrace, ...],
        tool_results: tuple[ToolResult, ...],
        agent_assessment: AgentAssessment | None,
        llm_diagnostics: "LLMDecisionDiagnostics | None",
        request_replan: bool = False,
    ) -> OrchestrationResponse:
        signal = AgentSignal(
            selected_subgoal=subgoal,
            selected_tool_id=None,
            tool_sequence=memory.tool_sequence,
            abstained=subgoal is OperationalSubgoal.ABSTAIN,
            no_action=subgoal is OperationalSubgoal.NO_ACTION,
            request_replan=request_replan,
            rationale=rationale,
        )
        return OrchestrationResponse(
            signal=signal,
            agent_assessment=agent_assessment,
            llm_diagnostics=llm_diagnostics,
            tool_call_traces=tool_call_traces,
            tool_results=tool_results,
            admissibility_checks=admissibility_checks,
            memory=OrchestrationMemory(
                step_count=memory.step_count,
                tool_sequence=memory.tool_sequence,
                last_subgoal=subgoal,
                abstained_last_step=subgoal is OperationalSubgoal.ABSTAIN,
            ),
        )

    def _build_rationale(
        self,
        tool_results: list[ToolResult],
        agent_assessment: AgentAssessment | None,
    ) -> str:
        statuses = ", ".join(f"{result.tool_id}:{result.status.value}" for result in tool_results)
        if agent_assessment is None:
            return f"Runtime executed bounded tool sequence with statuses: {statuses}."
        return (
            f"{agent_assessment.rationale} Runtime executed bounded tool sequence with "
            f"statuses: {statuses}."
        )

    def _attach_step_telemetry(
        self,
        response: OrchestrationResponse,
        *,
        orchestration_latency_ms: float,
    ) -> OrchestrationResponse:
        return replace(
            response,
            step_telemetry=self._build_step_telemetry(
                response,
                orchestration_latency_ms=orchestration_latency_ms,
            ),
        )

    def _summarize_tool_invocation(self, invocation: ToolInvocation) -> dict[str, object]:
        summary: dict[str, object] = {
            "tool_id": invocation.tool_id,
            "tool_class": invocation.tool_class.value,
            "subgoal": invocation.subgoal.value,
            "step_index": invocation.step_index,
            "prior_tool_ids": [result.tool_id for result in invocation.prior_results],
        }
        if invocation.system_state is not None:
            summary["time_index"] = invocation.system_state.time_index
            summary["benchmark_id"] = invocation.system_state.benchmark_id
            summary["inventory_level"] = list(invocation.system_state.inventory_level)
            summary["pipeline_inventory"] = list(invocation.system_state.pipeline_inventory)
            summary["backorder_level"] = list(invocation.system_state.backorder_level)
        if invocation.observation is not None:
            summary["observed_regime_label"] = (
                invocation.observation.regime_label.value
                if invocation.observation.regime_label is not None
                else None
            )
            summary["demand_realization"] = list(invocation.observation.demand_realization)
            summary["leadtime_realization"] = list(invocation.observation.leadtime_realization)
        if invocation.agent_assessment is not None:
            summary["agent_assessment"] = {
                "regime_label": invocation.agent_assessment.regime_label.value,
                "confidence": invocation.agent_assessment.confidence,
                "request_replan": invocation.agent_assessment.request_replan,
            }
        return summary

    def _summarize_tool_result(self, result: ToolResult) -> dict[str, object]:
        return {
            "tool_id": result.tool_id,
            "tool_class": result.tool_class.value,
            "subgoal": result.subgoal.value,
            "status": result.status.value,
            "confidence": result.confidence,
            "provenance": result.provenance or None,
            "next_tool_id": result.next_tool_id,
            "next_subgoal": result.next_subgoal.value if result.next_subgoal is not None else None,
            "request_replan": result.request_replan,
            "structured_output": {
                key: self._to_trace_value(value)
                for key, value in result.structured_output.items()
            },
        }

    def _tool_decision_snapshot(
        self,
        *,
        subgoal: OperationalSubgoal,
        agent_assessment: AgentAssessment | None,
        prior_results: tuple[ToolResult, ...],
        current_result: ToolResult | None = None,
    ) -> dict[str, object]:
        update_request_types: tuple[str, ...] = ()
        if agent_assessment is not None:
            update_request_types = tuple(
                update_request.request_type.value
                for update_request in agent_assessment.update_requests
            )
        scenario_result = self._latest_scenario_update_result(prior_results)
        if scenario_result is not None:
            update_request_types = tuple(
                update_type.value for update_type in scenario_result.applied_update_types
            )
        request_replan = False
        if agent_assessment is not None:
            request_replan = agent_assessment.request_replan
        if scenario_result is not None:
            request_replan = request_replan or scenario_result.request_replan
        if current_result is not None:
            request_replan = request_replan or current_result.request_replan
        return {
            "selected_subgoal": subgoal.value,
            "regime_label": (
                agent_assessment.regime_label.value
                if agent_assessment is not None
                else None
            ),
            "confidence": (
                agent_assessment.confidence if agent_assessment is not None else None
            ),
            "update_request_types": list(update_request_types),
            "request_replan": request_replan,
        }

    def _tool_optimizer_input_snapshot(
        self,
        *,
        observation: Observation | None,
        prior_results: tuple[ToolResult, ...],
    ) -> dict[str, float] | None:
        if observation is None:
            return None
        demand_outlook = observation.demand_realization[-1]
        leadtime_outlook = observation.leadtime_realization[-1]
        safety_buffer_scale = 1.0
        for result in prior_results:
            external_evidence_interpretation = result.structured_output.get(
                "external_evidence_interpretation"
            )
            if isinstance(external_evidence_interpretation, ExternalEvidenceInterpretation):
                (
                    demand_outlook,
                    leadtime_outlook,
                    safety_buffer_scale,
                ) = apply_external_evidence_bias(
                    external_evidence_interpretation,
                    demand_outlook=demand_outlook,
                    leadtime_outlook=leadtime_outlook,
                    safety_buffer_scale=safety_buffer_scale,
                )
            forecast_result = result.structured_output.get("forecast_result")
            if isinstance(forecast_result, ForecastResult):
                demand_outlook = forecast_result.point_forecast[0]
            leadtime_result = result.structured_output.get("leadtime_result")
            if isinstance(leadtime_result, LeadTimeResult):
                leadtime_outlook = leadtime_result.expected_lead_time[0]
            scenario_result = result.structured_output.get("scenario_update_result")
            if isinstance(scenario_result, ScenarioUpdateResult):
                demand_outlook = scenario_result.adjustment.demand_outlook
                leadtime_outlook = scenario_result.adjustment.leadtime_outlook
                safety_buffer_scale = scenario_result.adjustment.safety_buffer_scale
        return {
            "demand_outlook": demand_outlook,
            "leadtime_outlook": leadtime_outlook,
            "safety_buffer_scale": safety_buffer_scale,
        }

    def _latest_scenario_update_result(
        self,
        results: tuple[ToolResult, ...],
    ) -> ScenarioUpdateResult | None:
        for result in reversed(results):
            scenario_result = result.structured_output.get("scenario_update_result")
            if isinstance(scenario_result, ScenarioUpdateResult):
                return scenario_result
        return None

    def _build_step_telemetry(
        self,
        response: OrchestrationResponse,
        *,
        orchestration_latency_ms: float,
    ) -> OrchestrationStepTelemetry:
        llm_call_telemetry = None
        if response.llm_diagnostics is not None:
            llm_call_telemetry = response.llm_diagnostics.llm_call_telemetry
        return OrchestrationStepTelemetry(
            provider=(
                llm_call_telemetry.provider if llm_call_telemetry is not None else None
            ),
            model_name=(
                llm_call_telemetry.model_name if llm_call_telemetry is not None else None
            ),
            prompt_tokens=(
                llm_call_telemetry.prompt_tokens if llm_call_telemetry is not None else None
            ),
            completion_tokens=(
                llm_call_telemetry.completion_tokens
                if llm_call_telemetry is not None
                else None
            ),
            total_tokens=(
                llm_call_telemetry.total_tokens if llm_call_telemetry is not None else None
            ),
            llm_latency_ms=(
                llm_call_telemetry.llm_latency_ms if llm_call_telemetry is not None else None
            ),
            orchestration_latency_ms=orchestration_latency_ms,
            invalid_output=bool(
                response.llm_diagnostics and response.llm_diagnostics.invalid_output_count
            ),
            fallback_used=bool(
                response.llm_diagnostics and response.llm_diagnostics.fallback_count
            ),
            tool_call_count=len(response.signal.tool_sequence),
            selected_tools=response.signal.tool_sequence,
            request_replan=response.signal.request_replan,
            abstain_or_no_action=response.signal.abstained or response.signal.no_action,
            estimated_cost=(
                llm_call_telemetry.estimated_cost if llm_call_telemetry is not None else None
            ),
            client_error_category=(
                llm_call_telemetry.client_error_category
                if llm_call_telemetry is not None
                else None
            ),
            client_error_message=(
                llm_call_telemetry.client_error_message
                if llm_call_telemetry is not None
                else None
            ),
            retry_count=(
                llm_call_telemetry.retry_count if llm_call_telemetry is not None else 0
            ),
            failure_after_response=(
                llm_call_telemetry.failure_after_response
                if llm_call_telemetry is not None
                else None
            ),
        )

    def _run_llm_policy(
        self,
        request: OrchestrationRequest,
    ) -> "LLMDecisionAttempt":
        if (
            request.system_state is None
            or request.observation is None
            or request.evidence is None
            or self._llm_orchestrator is None
        ):
            from meio.agents.llm_orchestrator import LLMDecisionAttempt, LLMDecisionDiagnostics

            return LLMDecisionAttempt(
                decision=None,
                diagnostics=LLMDecisionDiagnostics(
                    provider="unavailable",
                    model_name=self._agent_config.llm_model_name,
                    fallback_count=1,
                    notes=("missing_typed_runtime_inputs",),
                ),
            )
        from meio.agents.llm_orchestrator import (
            LLMDecisionAttempt,
            LLMDecisionDiagnostics,
            LLMOrchestrationInput,
            LLMOutputValidationError,
        )

        candidate_tool_ids = request.candidate_tool_ids or tuple(self._tools)
        try:
            return self._llm_orchestrator.decide_with_diagnostics(
                LLMOrchestrationInput(
                    mission=request.mission,
                    system_state=request.system_state,
                    observation=request.observation,
                    evidence=request.evidence,
                    candidate_tool_ids=candidate_tool_ids,
                    recent_regime_history=request.recent_regime_history,
                    recent_stress_reference_demand_value=(
                        request.recent_stress_reference_demand_value
                    ),
                    recent_update_request_history=(
                        request.recent_update_request_history
                    ),
                ),
                tuple(tool.spec for tool in self._tools.values()),
            )
        except LLMClientRuntimeError as exc:
            return LLMDecisionAttempt(
                decision=None,
                diagnostics=LLMDecisionDiagnostics(
                    provider=getattr(self._llm_orchestrator.client, "provider", "unknown"),
                    model_name=self._agent_config.llm_model_name,
                    llm_call_telemetry=LLMCallTelemetry(
                        provider=getattr(self._llm_orchestrator.client, "provider", "unknown"),
                        model_name=self._agent_config.llm_model_name,
                        client_error_category=exc.category,
                        client_error_message=exc.message_summary,
                        retry_count=exc.retry_count,
                        failure_after_response=exc.failure_after_response,
                    ),
                    validation_failure_reason=exc.category.value,
                    fallback_count=1,
                    notes=("runtime_llm_failure", exc.category.value, exc.message_summary),
                ),
            )
        except (LLMOutputValidationError, RuntimeError, ValueError) as exc:
            return LLMDecisionAttempt(
                decision=None,
                diagnostics=LLMDecisionDiagnostics(
                    provider=getattr(self._llm_orchestrator.client, "provider", "unknown"),
                    model_name=self._agent_config.llm_model_name,
                    fallback_count=1,
                    notes=("runtime_llm_failure", str(exc)),
                ),
            )

    def _validate_structured_output(self, structured_output: dict[str, object]) -> None:
        for key, value in structured_output.items():
            if not key.strip():
                raise ValueError("ToolResult.structured_output keys must be non-empty strings.")
            if not self._is_allowed_output_value(value):
                raise TypeError(
                    "ToolResult.structured_output values must be bounded scalars, tuples, "
                    "or typed MEIO contract objects."
                )

    def _is_allowed_output_value(self, value: object) -> bool:
        if value is None or isinstance(value, (str, int, float, bool)):
            return True
        if isinstance(
            value,
            (
                AgentAssessment,
                ExternalEvidenceInterpretation,
                ForecastResult,
                LeadTimeResult,
                ScenarioSummary,
                ScenarioUpdateResult,
                OptimizationResult,
            ),
        ):
            return True
        if isinstance(value, tuple):
            return all(self._is_allowed_output_value(item) for item in value)
        return is_dataclass(value) and value.__class__.__module__.startswith("meio.")

    def _to_trace_value(self, value: object) -> object:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, StrEnum):
            return value.value
        if isinstance(value, tuple):
            return [self._to_trace_value(item) for item in value]
        if isinstance(value, dict):
            return {
                str(key): self._to_trace_value(item)
                for key, item in value.items()
            }
        if is_dataclass(value):
            return {
                key: self._to_trace_value(getattr(value, key))
                for key in value.__dataclass_fields__
            }
        return repr(value)


__all__ = [
    "AdmissibilityCheck",
    "OrchestrationMemory",
    "OrchestrationRequest",
    "OrchestrationResponse",
    "OrchestrationRuntime",
    "RuntimeMode",
]
