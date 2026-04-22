"""Bounded LLM orchestration helpers for the active MEIO benchmark path."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from time import perf_counter

from meio.agents.llm_client import (
    LLMClient,
    LLMClientContext,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMClientRuntimeError,
)
from meio.agents.prompts import build_prompt_messages
from meio.agents.prompts import PROMPT_VERSION, prompt_contract_hash
from meio.agents.structured_outputs import (
    LLMOutputValidationError,
    ParsedLLMOutput,
    StructuredLLMDecision,
    parse_llm_output_detailed,
)
from meio.agents.telemetry import LLMCallTelemetry, LLMCallTrace
from meio.config.schemas import AgentConfig
from meio.contracts import (
    AgentAssessment,
    MissionSpec,
    OperationalSubgoal,
    RegimeLabel,
    ToolSpec,
    UpdateRequestType,
)
from meio.scenarios.update_calibration import (
    UpdateCalibrationResult,
    calibrate_update_decision,
)
from meio.simulation.external_evidence_alignment import (
    summarize_external_evidence_batch,
)
from meio.simulation.evidence import RuntimeEvidence
from meio.simulation.state import Observation, SimulationState
_PIPELINE_HEAVY_RATIO_THRESHOLD = 12.0
_BACKORDER_HEAVY_RATIO_THRESHOLD = 2.0


@dataclass(frozen=True, slots=True)
class LLMOrchestrationInput:
    """Typed input bundle for one LLM orchestration decision."""

    mission: MissionSpec
    system_state: SimulationState
    observation: Observation
    evidence: RuntimeEvidence
    candidate_tool_ids: tuple[str, ...]
    recent_regime_history: tuple[RegimeLabel, ...] = ()
    recent_stress_reference_demand_value: float | None = None
    recent_update_request_history: tuple[tuple[UpdateRequestType, ...], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.mission, MissionSpec):
            raise TypeError("mission must be a MissionSpec.")
        if not isinstance(self.system_state, SimulationState):
            raise TypeError("system_state must be a SimulationState.")
        if not isinstance(self.observation, Observation):
            raise TypeError("observation must be an Observation.")
        if not isinstance(self.evidence, RuntimeEvidence):
            raise TypeError("evidence must be a RuntimeEvidence.")
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
        for regime_label in self.recent_regime_history:
            if not isinstance(regime_label, RegimeLabel):
                raise TypeError(
                    "recent_regime_history must contain RegimeLabel values."
                )
        for update_types in self.recent_update_request_history:
            for update_type in update_types:
                if not isinstance(update_type, UpdateRequestType):
                    raise TypeError(
                        "recent_update_request_history must contain "
                        "UpdateRequestType tuples."
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


@dataclass(frozen=True, slots=True)
class LLMOrchestrationDecision:
    """Validated bounded decision produced by the LLM orchestrator."""

    assessment: AgentAssessment
    selected_subgoal: OperationalSubgoal
    selected_tool_ids: tuple[str, ...]
    provider: str
    model: str
    raw_content: str

    def __post_init__(self) -> None:
        if not isinstance(self.assessment, AgentAssessment):
            raise TypeError("assessment must be an AgentAssessment.")
        if not isinstance(self.selected_subgoal, OperationalSubgoal):
            raise TypeError("selected_subgoal must be an OperationalSubgoal.")
        object.__setattr__(self, "selected_tool_ids", tuple(self.selected_tool_ids))
        if not self.provider.strip():
            raise ValueError("provider must be a non-empty string.")
        if not self.model.strip():
            raise ValueError("model must be a non-empty string.")
        if not self.raw_content.strip():
            raise ValueError("raw_content must be a non-empty string.")


@dataclass(frozen=True, slots=True)
class LLMDecisionDiagnostics:
    """Compact per-step diagnostics for one bounded LLM decision attempt."""

    provider: str
    model_name: str
    prompt_version: str | None = None
    prompt_contract_hash: str | None = None
    llm_call_telemetry: LLMCallTelemetry | None = None
    llm_call_trace: LLMCallTrace | None = None
    orchestration_latency_ms: float | None = None
    validation_failure_reason: str | None = None
    proposed_update_strength: str | None = None
    final_update_strength: str | None = None
    proposed_update_request_types: tuple[str, ...] = field(default_factory=tuple)
    final_update_request_types: tuple[str, ...] = field(default_factory=tuple)
    calibration_applied: bool = False
    hysteresis_applied: bool = False
    repeated_stress_moderation_applied: bool = False
    relapse_moderation_applied: bool = False
    unresolved_stress_moderation_applied: bool = False
    calibration_reason: str | None = None
    moderation_reason: str | None = None
    invalid_output_count: int = 0
    fallback_count: int = 0
    successful_response_count: int = 0
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("provider must be a non-empty string.")
        if not self.model_name.strip():
            raise ValueError("model_name must be a non-empty string.")
        if self.prompt_version is not None and not self.prompt_version.strip():
            raise ValueError("prompt_version must be non-empty when provided.")
        if self.prompt_contract_hash is not None and not self.prompt_contract_hash.strip():
            raise ValueError("prompt_contract_hash must be non-empty when provided.")
        if self.llm_call_telemetry is not None and not isinstance(
            self.llm_call_telemetry,
            LLMCallTelemetry,
        ):
            raise TypeError("llm_call_telemetry must be an LLMCallTelemetry when provided.")
        if self.llm_call_trace is not None and not isinstance(
            self.llm_call_trace,
            LLMCallTrace,
        ):
            raise TypeError("llm_call_trace must be an LLMCallTrace when provided.")
        if self.orchestration_latency_ms is not None and self.orchestration_latency_ms < 0.0:
            raise ValueError("orchestration_latency_ms must be non-negative when provided.")
        if (
            self.validation_failure_reason is not None
            and not self.validation_failure_reason.strip()
        ):
            raise ValueError("validation_failure_reason must be non-empty when provided.")
        if self.proposed_update_strength is not None and not self.proposed_update_strength.strip():
            raise ValueError("proposed_update_strength must be non-empty when provided.")
        if self.final_update_strength is not None and not self.final_update_strength.strip():
            raise ValueError("final_update_strength must be non-empty when provided.")
        object.__setattr__(
            self,
            "proposed_update_request_types",
            tuple(self.proposed_update_request_types),
        )
        object.__setattr__(
            self,
            "final_update_request_types",
            tuple(self.final_update_request_types),
        )
        object.__setattr__(self, "notes", tuple(self.notes))
        for field_name in (
            "proposed_update_request_types",
            "final_update_request_types",
        ):
            for value in getattr(self, field_name):
                if not value.strip():
                    raise ValueError(f"{field_name} must contain non-empty strings.")
        if self.calibration_reason is not None and not self.calibration_reason.strip():
            raise ValueError("calibration_reason must be non-empty when provided.")
        if self.moderation_reason is not None and not self.moderation_reason.strip():
            raise ValueError("moderation_reason must be non-empty when provided.")
        for field_name in (
            "invalid_output_count",
            "fallback_count",
            "successful_response_count",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative.")
        for note in self.notes:
            if not note.strip():
                raise ValueError("notes must contain non-empty strings.")


@dataclass(frozen=True, slots=True)
class LLMDecisionAttempt:
    """One bounded LLM attempt plus validated decision metadata when available."""

    decision: LLMOrchestrationDecision | None
    diagnostics: LLMDecisionDiagnostics

    def __post_init__(self) -> None:
        if self.decision is not None and not isinstance(
            self.decision, LLMOrchestrationDecision
        ):
            raise TypeError("decision must be an LLMOrchestrationDecision when provided.")
        if not isinstance(self.diagnostics, LLMDecisionDiagnostics):
            raise TypeError("diagnostics must be an LLMDecisionDiagnostics.")


@dataclass(frozen=True, slots=True)
class LLMOrchestrator:
    """Compact bounded LLM orchestration layer for typed runtime decisions."""

    client: LLMClient
    agent_config: AgentConfig

    def build_request(
        self,
        orchestration_input: LLMOrchestrationInput,
        tool_specs: tuple[ToolSpec, ...],
    ) -> LLMCompletionRequest:
        """Build a typed LLM request from runtime inputs."""

        demand_value = orchestration_input.observation.demand_realization[-1]
        leadtime_value = orchestration_input.observation.leadtime_realization[-1]
        previous_demand_value = _previous_series_value(orchestration_input.evidence.demand.history)
        previous_leadtime_value = _previous_series_value(
            orchestration_input.evidence.leadtime.history
        )
        demand_baseline_value = orchestration_input.evidence.demand_baseline_value
        leadtime_baseline_value = orchestration_input.evidence.leadtime_baseline_value
        current_regime = (
            orchestration_input.observation.regime_label
            or orchestration_input.system_state.regime_label
        )
        total_pipeline_value = sum(orchestration_input.system_state.pipeline_inventory)
        total_backorder_value = sum(orchestration_input.system_state.backorder_level)
        previous_demand_ratio_to_baseline = _ratio_or_none(
            previous_demand_value,
            demand_baseline_value,
        )
        pipeline_ratio_to_baseline = _ratio_or_none(
            total_pipeline_value,
            demand_baseline_value,
        )
        backorder_ratio_to_baseline = _ratio_or_none(
            total_backorder_value,
            demand_baseline_value,
        )
        external_evidence_batch = orchestration_input.evidence.external_evidence_batch
        external_evidence_enabled = (
            "external_evidence_tool" in orchestration_input.mission.admissible_tool_ids
        )
        external_evidence_present = (
            external_evidence_batch is not None
            and external_evidence_batch.record_count > 0
        )
        include_external_evidence_context = (
            external_evidence_enabled or external_evidence_present
        )
        pipeline_heavy_vs_baseline = _flag_from_ratio(
            pipeline_ratio_to_baseline,
            threshold=_PIPELINE_HEAVY_RATIO_THRESHOLD,
        )
        backlog_heavy_vs_baseline = _flag_from_ratio(
            backorder_ratio_to_baseline,
            threshold=_BACKORDER_HEAVY_RATIO_THRESHOLD,
        )
        context = LLMClientContext(
            benchmark_id=orchestration_input.system_state.benchmark_id,
            mission_id=orchestration_input.mission.mission_id,
            time_index=orchestration_input.system_state.time_index,
            regime_label=current_regime,
            demand_value=demand_value,
            leadtime_value=leadtime_value,
            inventory_level=orchestration_input.system_state.inventory_level,
            backorder_level=orchestration_input.system_state.backorder_level,
            available_tool_ids=orchestration_input.candidate_tool_ids,
            max_tool_steps=min(
                self.agent_config.max_tool_steps,
                orchestration_input.mission.max_tool_steps,
            ),
            demand_baseline_value=demand_baseline_value,
            demand_change_from_baseline=_delta_or_none(demand_value, demand_baseline_value),
            demand_ratio_to_baseline=_ratio_or_none(demand_value, demand_baseline_value),
            previous_demand_value=previous_demand_value,
            demand_change_from_previous=_delta_or_none(demand_value, previous_demand_value),
            leadtime_baseline_value=leadtime_baseline_value,
            leadtime_change_from_baseline=_delta_or_none(leadtime_value, leadtime_baseline_value),
            leadtime_ratio_to_baseline=_ratio_or_none(leadtime_value, leadtime_baseline_value),
            previous_leadtime_value=previous_leadtime_value,
            leadtime_change_from_previous=_delta_or_none(
                leadtime_value,
                previous_leadtime_value,
            ),
            downstream_inventory_value=orchestration_input.system_state.inventory_level[0],
            total_inventory_value=sum(orchestration_input.system_state.inventory_level),
            pipeline_total_value=total_pipeline_value,
            total_backorder_value=total_backorder_value,
            inventory_gap_to_demand=(
                orchestration_input.system_state.inventory_level[0] - demand_value
            ),
            pipeline_ratio_to_baseline=pipeline_ratio_to_baseline,
            backorder_ratio_to_baseline=backorder_ratio_to_baseline,
            repeated_stress_detected=(
                current_regime is RegimeLabel.DEMAND_REGIME_SHIFT
                and previous_demand_ratio_to_baseline is not None
                and previous_demand_ratio_to_baseline >= 1.15
            ),
            pipeline_heavy_vs_baseline=pipeline_heavy_vs_baseline,
            backlog_heavy_vs_baseline=backlog_heavy_vs_baseline,
            recovery_with_high_pipeline_load=(
                current_regime is RegimeLabel.RECOVERY and pipeline_heavy_vs_baseline
            ),
            recovery_with_high_backorder_load=(
                current_regime is RegimeLabel.RECOVERY and backlog_heavy_vs_baseline
            ),
            external_evidence_present=(
                external_evidence_present
                if include_external_evidence_context
                else None
            ),
            external_evidence_source_count=(
                external_evidence_batch.record_count
                if include_external_evidence_context and external_evidence_batch is not None
                else None
            ),
            external_evidence_false_alarm_present=(
                external_evidence_batch.contains_false_alarm
                if include_external_evidence_context and external_evidence_batch is not None
                else None
            ),
            external_evidence_summaries=(
                summarize_external_evidence_batch(external_evidence_batch)
                if include_external_evidence_context
                else ()
            ),
        )
        return LLMCompletionRequest(
            model=self.agent_config.llm_model_name,
            messages=build_prompt_messages(context, tool_specs),
            context=context,
            temperature=self.agent_config.llm_temperature,
            prompt_version=PROMPT_VERSION,
            prompt_contract_hash=prompt_contract_hash(),
        )

    def parse_response(
        self,
        response: LLMCompletionResponse,
        *,
        allowed_tool_ids: tuple[str, ...],
    ) -> ParsedLLMOutput:
        """Parse and validate one LLM response payload."""

        return parse_llm_output_detailed(
            response.content,
            allowed_tool_ids=allowed_tool_ids,
            allowed_update_types=self.agent_config.allowed_update_types,
        )

    def decide(
        self,
        orchestration_input: LLMOrchestrationInput,
        tool_specs: tuple[ToolSpec, ...],
    ) -> LLMOrchestrationDecision:
        """Run the bounded LLM orchestration policy and return a typed decision."""

        attempt = self.decide_with_diagnostics(orchestration_input, tool_specs)
        if attempt.decision is None:
            raise RuntimeError("LLM orchestration decision could not be validated.")
        return attempt.decision

    def decide_with_diagnostics(
        self,
        orchestration_input: LLMOrchestrationInput,
        tool_specs: tuple[ToolSpec, ...],
    ) -> LLMDecisionAttempt:
        """Run the bounded LLM orchestration policy with typed diagnostics."""

        orchestration_start = perf_counter()
        completion_request = self.build_request(orchestration_input, tool_specs)
        prompt_text = _render_prompt_text(completion_request)
        prompt_hash = _hash_text(prompt_text)
        prompt_version = completion_request.prompt_version
        prompt_contract_hash_value = completion_request.prompt_contract_hash
        provider = getattr(self.client, "provider", "unknown")
        llm_start = perf_counter()
        try:
            completion_response = self.client.complete(completion_request)
            llm_latency_ms = (perf_counter() - llm_start) * 1000.0
        except LLMClientRuntimeError as exc:
            llm_latency_ms = (perf_counter() - llm_start) * 1000.0
            return LLMDecisionAttempt(
                decision=None,
                diagnostics=LLMDecisionDiagnostics(
                    provider=provider,
                    model_name=completion_request.model,
                    prompt_version=prompt_version,
                    prompt_contract_hash=prompt_contract_hash_value,
                    llm_call_telemetry=LLMCallTelemetry(
                        provider=provider,
                        model_name=completion_request.model,
                        llm_latency_ms=llm_latency_ms,
                        client_error_category=exc.category,
                        client_error_message=exc.message_summary,
                        retry_count=exc.retry_count,
                        failure_after_response=exc.failure_after_response,
                    ),
                    llm_call_trace=LLMCallTrace(
                        provider=provider,
                        model_name=completion_request.model,
                        prompt_version=prompt_version,
                        prompt_text=prompt_text,
                        prompt_hash=prompt_hash,
                        raw_output_text=None,
                        parsed_output=None,
                        validation_success=False,
                        invalid_output=False,
                        fallback_used=True,
                        fallback_reason="client_runtime_error",
                        requested_tool_ids=(),
                        unavailable_tool_ids=(),
                        prompt_tokens=None,
                        completion_tokens=None,
                        total_tokens=None,
                        latency_ms=llm_latency_ms,
                        retry_count=exc.retry_count,
                        client_error_category=exc.category,
                        client_error_message=exc.message_summary,
                        failure_after_response=exc.failure_after_response,
                    ),
                    orchestration_latency_ms=(perf_counter() - orchestration_start) * 1000.0,
                    validation_failure_reason=exc.category.value,
                    fallback_count=1,
                    notes=("client_runtime_error", exc.category.value, exc.message_summary),
                ),
            )
        except RuntimeError as exc:
            llm_latency_ms = (perf_counter() - llm_start) * 1000.0
            return LLMDecisionAttempt(
                decision=None,
                diagnostics=LLMDecisionDiagnostics(
                    provider=provider,
                    model_name=completion_request.model,
                    prompt_version=prompt_version,
                    prompt_contract_hash=prompt_contract_hash_value,
                    llm_call_telemetry=LLMCallTelemetry(
                        provider=provider,
                        model_name=completion_request.model,
                        llm_latency_ms=llm_latency_ms,
                    ),
                    llm_call_trace=LLMCallTrace(
                        provider=provider,
                        model_name=completion_request.model,
                        prompt_version=prompt_version,
                        prompt_text=prompt_text,
                        prompt_hash=prompt_hash,
                        raw_output_text=None,
                        parsed_output=None,
                        validation_success=False,
                        invalid_output=False,
                        fallback_used=True,
                        fallback_reason="client_runtime_error",
                        requested_tool_ids=(),
                        unavailable_tool_ids=(),
                        prompt_tokens=None,
                        completion_tokens=None,
                        total_tokens=None,
                        latency_ms=llm_latency_ms,
                        retry_count=0,
                    ),
                    orchestration_latency_ms=(perf_counter() - orchestration_start) * 1000.0,
                    validation_failure_reason="client_runtime_error",
                    fallback_count=1,
                    notes=("client_runtime_error", str(exc)),
                ),
            )
        llm_call_telemetry = completion_response.telemetry or LLMCallTelemetry(
            provider=completion_response.provider,
            model_name=completion_response.model,
        )
        llm_call_telemetry = replace(
            llm_call_telemetry,
            llm_latency_ms=llm_latency_ms,
        )
        try:
            parsed_output = self.parse_response(
                completion_response,
                allowed_tool_ids=orchestration_input.candidate_tool_ids,
            )
        except LLMOutputValidationError as exc:
            requested_tool_ids = tuple(
                str(tool_id) for tool_id in exc.details.get("requested_tool_ids", ())
            )
            unavailable_tool_ids = tuple(
                str(tool_id) for tool_id in exc.details.get("unavailable_tool_ids", ())
            )
            return LLMDecisionAttempt(
                decision=None,
                diagnostics=LLMDecisionDiagnostics(
                    provider=completion_response.provider,
                    model_name=completion_response.model,
                    prompt_version=prompt_version,
                    prompt_contract_hash=prompt_contract_hash_value,
                    llm_call_telemetry=llm_call_telemetry,
                    llm_call_trace=LLMCallTrace(
                        provider=completion_response.provider,
                        model_name=completion_response.model,
                        prompt_version=prompt_version,
                        prompt_text=prompt_text,
                        prompt_hash=prompt_hash,
                        raw_output_text=completion_response.content,
                        parsed_output=None,
                        validation_success=False,
                        invalid_output=True,
                        fallback_used=True,
                        fallback_reason=exc.reason_code,
                        requested_tool_ids=requested_tool_ids,
                        unavailable_tool_ids=unavailable_tool_ids,
                        prompt_tokens=llm_call_telemetry.prompt_tokens,
                        completion_tokens=llm_call_telemetry.completion_tokens,
                        total_tokens=llm_call_telemetry.total_tokens,
                        latency_ms=llm_call_telemetry.llm_latency_ms,
                        retry_count=llm_call_telemetry.retry_count,
                    ),
                    orchestration_latency_ms=(perf_counter() - orchestration_start) * 1000.0,
                    validation_failure_reason=exc.reason_code,
                    invalid_output_count=1,
                    fallback_count=1,
                    notes=("invalid_structured_output", exc.reason_code, str(exc)),
                ),
            )
        structured_decision = parsed_output.decision
        calibrated_decision = calibrate_update_decision(
            structured_decision,
            context=completion_request.context,
            recent_regime_history=orchestration_input.recent_regime_history,
            recent_stress_reference_demand_value=(
                orchestration_input.recent_stress_reference_demand_value
            ),
            recent_update_request_history=(
                orchestration_input.recent_update_request_history
            ),
        )
        final_decision = calibrated_decision.final_decision
        proposed_update_request_types = tuple(
            update_request.request_type.value
            for update_request in structured_decision.update_requests
        )
        final_update_request_types = tuple(
            update_request.request_type.value
            for update_request in final_decision.update_requests
        )
        proposed_update_strength = calibrated_decision.proposed_strength.value
        final_update_strength = calibrated_decision.final_strength.value
        assessment = AgentAssessment(
            regime_label=final_decision.regime_label,
            confidence=final_decision.confidence,
            rationale=final_decision.rationale,
            update_requests=final_decision.update_requests,
            request_replan=final_decision.request_replan,
        )
        return LLMDecisionAttempt(
            decision=LLMOrchestrationDecision(
                assessment=assessment,
                selected_subgoal=final_decision.selected_subgoal,
                selected_tool_ids=final_decision.candidate_tool_ids,
                provider=completion_response.provider,
                model=completion_response.model,
                raw_content=completion_response.content,
            ),
            diagnostics=LLMDecisionDiagnostics(
                provider=completion_response.provider,
                model_name=completion_response.model,
                prompt_version=prompt_version,
                prompt_contract_hash=prompt_contract_hash_value,
                llm_call_telemetry=llm_call_telemetry,
                llm_call_trace=LLMCallTrace(
                    provider=completion_response.provider,
                    model_name=completion_response.model,
                    prompt_version=prompt_version,
                    prompt_text=prompt_text,
                    prompt_hash=prompt_hash,
                    raw_output_text=completion_response.content,
                    parsed_output=structured_decision.to_record(),
                    validation_success=True,
                    invalid_output=False,
                    fallback_used=False,
                    fallback_reason=None,
                    requested_tool_ids=structured_decision.candidate_tool_ids,
                    unavailable_tool_ids=(),
                    prompt_tokens=llm_call_telemetry.prompt_tokens,
                    completion_tokens=llm_call_telemetry.completion_tokens,
                    total_tokens=llm_call_telemetry.total_tokens,
                    latency_ms=llm_call_telemetry.llm_latency_ms,
                    retry_count=llm_call_telemetry.retry_count,
                ),
                orchestration_latency_ms=(perf_counter() - orchestration_start) * 1000.0,
                proposed_update_strength=proposed_update_strength,
                final_update_strength=final_update_strength,
                proposed_update_request_types=proposed_update_request_types,
                final_update_request_types=final_update_request_types,
                calibration_applied=calibrated_decision.calibration_applied,
                hysteresis_applied=calibrated_decision.hysteresis_applied,
                repeated_stress_moderation_applied=(
                    calibrated_decision.repeated_stress_moderation_applied
                ),
                relapse_moderation_applied=calibrated_decision.relapse_moderation_applied,
                unresolved_stress_moderation_applied=(
                    calibrated_decision.unresolved_stress_moderation_applied
                ),
                calibration_reason=calibrated_decision.calibration_reason,
                moderation_reason=calibrated_decision.calibration_reason,
                successful_response_count=1,
                notes=(
                    completion_response.metadata
                    + parsed_output.normalization_notes
                    + _calibration_notes(calibrated_decision)
                ),
            ),
        )


def _render_prompt_text(request: LLMCompletionRequest) -> str:
    return "\n\n".join(
        f"{message.role.upper()}:\n{message.content}"
        for message in request.messages
    )


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _previous_series_value(values: tuple[float, ...]) -> float | None:
    if len(values) < 2:
        return None
    return values[-2]


def _delta_or_none(current_value: float, baseline_value: float | None) -> float | None:
    if baseline_value is None:
        return None
    return current_value - baseline_value


def _ratio_or_none(current_value: float, baseline_value: float | None) -> float | None:
    if baseline_value is None or baseline_value <= 0.0:
        return None
    return current_value / baseline_value


def _flag_from_ratio(value: float | None, *, threshold: float) -> bool | None:
    if value is None:
        return None
    return value >= threshold


def _calibration_notes(
    calibrated_decision: UpdateCalibrationResult,
) -> tuple[str, ...]:
    notes: list[str] = []
    if calibrated_decision.calibration_applied:
        notes.append("applied_update_strength_calibration")
    if calibrated_decision.hysteresis_applied:
        notes.append("applied_update_strength_hysteresis")
    if calibrated_decision.repeated_stress_moderation_applied:
        notes.append("applied_repeated_stress_moderation")
    if calibrated_decision.relapse_moderation_applied:
        notes.append("applied_relapse_moderation")
    if calibrated_decision.unresolved_stress_moderation_applied:
        notes.append("applied_unresolved_stress_moderation")
    return tuple(notes)


__all__ = [
    "LLMDecisionAttempt",
    "LLMDecisionDiagnostics",
    "LLMOrchestrationDecision",
    "LLMOrchestrationInput",
    "LLMOrchestrator",
    "LLMOutputValidationError",
]
