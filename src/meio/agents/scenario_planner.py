"""Explicit Agentic AI scenario-planner tools.

The planner keeps the LLM in a bounded supervisory role: the LLM supplies the
regime-facing assessment, these tools turn that assessment into auditable
scenario candidates, and the final tool evaluates candidates with the trusted
downstream replenishment rule. No tool emits raw replenishment orders.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from statistics import mean

from meio.agents.uncertainty_baselines import (
    RobustUncertaintyPolicy,
    ScenarioRollingHorizonPolicy,
)
from meio.config.schemas import RobustPolicyConfig, ScenarioRollingHorizonPolicyConfig
from meio.contracts import (
    AgentSignal,
    BoundedTool,
    OperationalSubgoal,
    RegimeLabel,
    ToolClass,
    ToolInvocation,
    ToolResult,
    ToolSpec,
    ToolStatus,
    UpdateRequestType,
)
from meio.evaluation.rollout_metrics import compute_period_total_cost
from meio.forecasting.adapters import DeterministicForecastTool
from meio.leadtime.adapters import DeterministicLeadTimeTool
from meio.optimization.adapters import TrustedOptimizerAdapter, build_optimization_request
from meio.scenarios.adapters import DeterministicScenarioTool
from meio.scenarios.contracts import (
    ScenarioAdjustmentSummary,
    ScenarioSummary,
    ScenarioUpdateResult,
)
from meio.simulation.evidence import DemandEvidence, LeadTimeEvidence
from meio.simulation.serial_benchmark import SerialBenchmarkCase, advance_serial_state
from meio.simulation.state import Observation, PeriodTraceRecord, SimulationState


@dataclass(frozen=True, slots=True)
class RegimeDiagnosisRecord:
    """Traceable uncertainty diagnosis used by the candidate generator."""

    regime_label: RegimeLabel
    case_family: str
    demand_ratio_to_baseline: float
    leadtime_ratio_to_baseline: float
    recent_stress_count: int
    total_backorder: float
    pipeline_total: float
    latest_demand: float
    latest_leadtime: float
    demand_window: tuple[float, ...]
    leadtime_window: tuple[float, ...]
    agent_update_request_types: tuple[UpdateRequestType, ...]
    rationale: str

    def __post_init__(self) -> None:
        if not isinstance(self.regime_label, RegimeLabel):
            raise TypeError("regime_label must be a RegimeLabel.")
        if not self.case_family.strip():
            raise ValueError("case_family must be non-empty.")
        if self.demand_ratio_to_baseline < 0.0:
            raise ValueError("demand_ratio_to_baseline must be non-negative.")
        if self.leadtime_ratio_to_baseline < 0.0:
            raise ValueError("leadtime_ratio_to_baseline must be non-negative.")
        if self.recent_stress_count < 0:
            raise ValueError("recent_stress_count must be non-negative.")
        if self.total_backorder < 0.0:
            raise ValueError("total_backorder must be non-negative.")
        if self.pipeline_total < 0.0:
            raise ValueError("pipeline_total must be non-negative.")
        if self.latest_demand < 0.0:
            raise ValueError("latest_demand must be non-negative.")
        if self.latest_leadtime <= 0.0:
            raise ValueError("latest_leadtime must be positive.")
        object.__setattr__(self, "demand_window", tuple(self.demand_window))
        object.__setattr__(self, "leadtime_window", tuple(self.leadtime_window))
        object.__setattr__(
            self,
            "agent_update_request_types",
            tuple(self.agent_update_request_types),
        )
        if not self.demand_window:
            raise ValueError("demand_window must not be empty.")
        if not self.leadtime_window:
            raise ValueError("leadtime_window must not be empty.")
        if not self.rationale.strip():
            raise ValueError("rationale must be non-empty.")


@dataclass(frozen=True, slots=True)
class ScenarioCandidateRecord:
    """One bounded scenario-input candidate, not a replenishment action."""

    candidate_id: str
    provenance: str
    demand_outlook: float
    leadtime_outlook: float
    safety_buffer_scale: float
    applied_update_types: tuple[UpdateRequestType, ...]
    request_replan: bool
    rationale: str

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("candidate_id must be non-empty.")
        if not self.provenance.strip():
            raise ValueError("provenance must be non-empty.")
        if self.demand_outlook < 0.0:
            raise ValueError("demand_outlook must be non-negative.")
        if self.leadtime_outlook <= 0.0:
            raise ValueError("leadtime_outlook must be positive.")
        if self.safety_buffer_scale <= 0.0:
            raise ValueError("safety_buffer_scale must be positive.")
        object.__setattr__(self, "applied_update_types", tuple(self.applied_update_types))
        if not self.applied_update_types:
            raise ValueError("applied_update_types must not be empty.")
        if (
            UpdateRequestType.KEEP_CURRENT in self.applied_update_types
            and len(self.applied_update_types) > 1
        ):
            raise ValueError("keep_current must not be combined with other updates.")
        if not self.rationale.strip():
            raise ValueError("rationale must be non-empty.")


@dataclass(frozen=True, slots=True)
class ScenarioCandidateSet:
    """Candidate set generated from the agent diagnosis and runtime evidence."""

    candidates: tuple[ScenarioCandidateRecord, ...]
    incumbent_candidate_id: str
    generator_notes: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "generator_notes", tuple(self.generator_notes))
        if not self.candidates:
            raise ValueError("candidates must not be empty.")
        ids = tuple(candidate.candidate_id for candidate in self.candidates)
        if len(ids) != len(set(ids)):
            raise ValueError("candidate ids must be unique.")
        if self.incumbent_candidate_id not in ids:
            raise ValueError("incumbent_candidate_id must identify one candidate.")
        for note in self.generator_notes:
            if not note.strip():
                raise ValueError("generator_notes must contain non-empty strings.")


@dataclass(frozen=True, slots=True)
class RegimeBeliefEntry:
    """One hidden-regime belief used for risk-sensitive candidate scoring."""

    regime_label: RegimeLabel
    probability: float
    demand_multiplier: float
    leadtime_multiplier: float
    rationale: str

    def __post_init__(self) -> None:
        if not isinstance(self.regime_label, RegimeLabel):
            raise TypeError("regime_label must be a RegimeLabel.")
        if self.probability < 0.0 or self.probability > 1.0:
            raise ValueError("probability must be within [0.0, 1.0].")
        if self.demand_multiplier <= 0.0:
            raise ValueError("demand_multiplier must be positive.")
        if self.leadtime_multiplier <= 0.0:
            raise ValueError("leadtime_multiplier must be positive.")
        if not self.rationale.strip():
            raise ValueError("rationale must be non-empty.")


@dataclass(frozen=True, slots=True)
class RegimeBeliefRecord:
    """Bounded belief state over possible near-term operating regimes."""

    entries: tuple[RegimeBeliefEntry, ...]
    dominant_regime_label: RegimeLabel
    belief_entropy: float
    tail_risk_weight: float
    service_risk_weight: float
    overreaction_weight: float
    rationale: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", tuple(self.entries))
        if not self.entries:
            raise ValueError("entries must not be empty.")
        if not isinstance(self.dominant_regime_label, RegimeLabel):
            raise TypeError("dominant_regime_label must be a RegimeLabel.")
        for entry in self.entries:
            if not isinstance(entry, RegimeBeliefEntry):
                raise TypeError("entries must contain RegimeBeliefEntry values.")
        probability_sum = sum(entry.probability for entry in self.entries)
        if abs(probability_sum - 1.0) > 1e-6:
            raise ValueError("regime belief probabilities must sum to 1.0.")
        for field_name in (
            "belief_entropy",
            "tail_risk_weight",
            "service_risk_weight",
            "overreaction_weight",
        ):
            if getattr(self, field_name) < 0.0:
                raise ValueError(f"{field_name} must be non-negative.")
        if not self.rationale.strip():
            raise ValueError("rationale must be non-empty.")


@dataclass(frozen=True, slots=True)
class PlannerCandidateScoreRecord:
    """Expected-cost score for one planner candidate."""

    candidate_id: str
    expected_cost: float
    demand_outlook: float
    leadtime_outlook: float
    safety_buffer_scale: float
    mean_cost: float | None = None
    tail_cost: float | None = None
    service_risk_penalty: float | None = None
    overreaction_penalty: float | None = None
    selected: bool = False

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("candidate_id must be non-empty.")
        if self.expected_cost < 0.0:
            raise ValueError("expected_cost must be non-negative.")
        if self.demand_outlook < 0.0:
            raise ValueError("demand_outlook must be non-negative.")
        if self.leadtime_outlook <= 0.0:
            raise ValueError("leadtime_outlook must be positive.")
        if self.safety_buffer_scale <= 0.0:
            raise ValueError("safety_buffer_scale must be positive.")
        for field_name in (
            "mean_cost",
            "tail_cost",
            "service_risk_penalty",
            "overreaction_penalty",
        ):
            value = getattr(self, field_name)
            if value is not None and value < 0.0:
                raise ValueError(f"{field_name} must be non-negative when provided.")


@dataclass(frozen=True, slots=True)
class ScenarioPlannerEvaluationDiagnostics:
    """Auditable evaluation summary for the selected scenario candidate."""

    selected_candidate_id: str
    incumbent_candidate_id: str
    horizon_length: int
    scenario_count: int
    candidate_count: int
    selected_expected_cost: float
    incumbent_expected_cost: float
    candidate_scores: tuple[PlannerCandidateScoreRecord, ...]

    def __post_init__(self) -> None:
        if not self.selected_candidate_id.strip():
            raise ValueError("selected_candidate_id must be non-empty.")
        if not self.incumbent_candidate_id.strip():
            raise ValueError("incumbent_candidate_id must be non-empty.")
        if self.horizon_length <= 0:
            raise ValueError("horizon_length must be positive.")
        if self.scenario_count <= 0:
            raise ValueError("scenario_count must be positive.")
        if self.candidate_count <= 0:
            raise ValueError("candidate_count must be positive.")
        if self.selected_expected_cost < 0.0:
            raise ValueError("selected_expected_cost must be non-negative.")
        if self.incumbent_expected_cost < 0.0:
            raise ValueError("incumbent_expected_cost must be non-negative.")
        object.__setattr__(self, "candidate_scores", tuple(self.candidate_scores))


@dataclass(frozen=True, slots=True)
class CounterfactualRegretGuardRecord:
    """Trace record for a guarded candidate-selection decision."""

    initial_selected_candidate_id: str
    final_selected_candidate_id: str
    guard_changed_selection: bool
    guard_reason: str

    def __post_init__(self) -> None:
        if not self.initial_selected_candidate_id.strip():
            raise ValueError("initial_selected_candidate_id must be non-empty.")
        if not self.final_selected_candidate_id.strip():
            raise ValueError("final_selected_candidate_id must be non-empty.")
        if not self.guard_reason.strip():
            raise ValueError("guard_reason must be non-empty.")


@dataclass(frozen=True, slots=True)
class RegimeDiagnosisTool(BoundedTool):
    """Diagnose the current uncertainty case from structured evidence."""

    tool_id: str = "regime_diagnosis_tool"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            tool_id=self.tool_id,
            tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
            supported_subgoals=(
                OperationalSubgoal.QUERY_UNCERTAINTY,
                OperationalSubgoal.REQUEST_REPLAN,
            ),
            description=(
                "Explicit diagnosis tool. It summarizes demand, lead-time, "
                "backlog, pipeline, and LLM assessment evidence into a bounded "
                "uncertainty-case record for candidate generation."
            ),
            produces_raw_orders=False,
        )

    def invoke(self, invocation: ToolInvocation) -> ToolResult:
        _validate_invocation(invocation, tool_name="RegimeDiagnosisTool")
        diagnosis = _build_diagnosis(invocation)
        return ToolResult(
            tool_id=invocation.tool_id,
            tool_class=invocation.tool_class,
            subgoal=invocation.subgoal,
            status=ToolStatus.SUCCESS,
            structured_output={"regime_diagnosis": diagnosis},
            confidence=0.92,
            provenance="regime_diagnosis_tool",
            next_tool_id="scenario_candidate_generator_tool",
            next_subgoal=OperationalSubgoal.QUERY_UNCERTAINTY,
            request_replan=False,
            emits_raw_orders=False,
        )


@dataclass(frozen=True, slots=True)
class RegimeBeliefTool(BoundedTool):
    """Estimate a bounded hidden-regime belief for risk-sensitive scoring."""

    tool_id: str = "regime_belief_tool"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            tool_id=self.tool_id,
            tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
            supported_subgoals=(
                OperationalSubgoal.QUERY_UNCERTAINTY,
                OperationalSubgoal.REQUEST_REPLAN,
            ),
            description=(
                "Explicit regime-belief tool. It converts the diagnosis and "
                "structured evidence into bounded probabilities over possible "
                "near-term regimes for risk-sensitive scenario evaluation."
            ),
            produces_raw_orders=False,
        )

    def invoke(self, invocation: ToolInvocation) -> ToolResult:
        _validate_invocation(invocation, tool_name="RegimeBeliefTool")
        diagnosis = _latest_diagnosis(invocation)
        belief = _build_regime_belief(diagnosis)
        return ToolResult(
            tool_id=invocation.tool_id,
            tool_class=invocation.tool_class,
            subgoal=invocation.subgoal,
            status=ToolStatus.SUCCESS,
            structured_output={"regime_belief": belief},
            confidence=0.91,
            provenance="regime_belief_tool",
            next_tool_id="scenario_candidate_generator_tool",
            next_subgoal=OperationalSubgoal.QUERY_UNCERTAINTY,
            request_replan=False,
            emits_raw_orders=False,
        )


@dataclass(frozen=True, slots=True)
class ScenarioCandidateGeneratorTool(BoundedTool):
    """Generate bounded regime-conditioned scenario candidates."""

    benchmark_case: SerialBenchmarkCase
    robust_config: RobustPolicyConfig = RobustPolicyConfig()
    rolling_config: ScenarioRollingHorizonPolicyConfig = (
        ScenarioRollingHorizonPolicyConfig()
    )
    tool_id: str = "scenario_candidate_generator_tool"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            tool_id=self.tool_id,
            tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
            supported_subgoals=(OperationalSubgoal.QUERY_UNCERTAINTY,),
            description=(
                "Explicit candidate-generation tool. It combines the LLM-backed "
                "diagnosis with structured evidence to propose bounded scenario "
                "updates, including rolling-horizon as the incumbent comparator."
            ),
            produces_raw_orders=False,
        )

    def invoke(self, invocation: ToolInvocation) -> ToolResult:
        _validate_invocation(invocation, tool_name="ScenarioCandidateGeneratorTool")
        diagnosis = _latest_diagnosis(invocation)
        candidate_set = _build_candidate_set(
            invocation=invocation,
            diagnosis=diagnosis,
            benchmark_case=self.benchmark_case,
            robust_config=self.robust_config,
            rolling_config=self.rolling_config,
        )
        return ToolResult(
            tool_id=invocation.tool_id,
            tool_class=invocation.tool_class,
            subgoal=invocation.subgoal,
            status=ToolStatus.SUCCESS,
            structured_output={"scenario_candidate_set": candidate_set},
            confidence=0.90,
            provenance="scenario_candidate_generator_tool",
            next_tool_id="scenario_evaluator_tool",
            next_subgoal=OperationalSubgoal.QUERY_UNCERTAINTY,
            request_replan=False,
            emits_raw_orders=False,
        )


@dataclass(frozen=True, slots=True)
class ScenarioEvaluatorTool(BoundedTool):
    """Evaluate scenario candidates through the trusted downstream rule."""

    benchmark_case: SerialBenchmarkCase
    rolling_config: ScenarioRollingHorizonPolicyConfig = (
        ScenarioRollingHorizonPolicyConfig()
    )
    tool_id: str = "scenario_evaluator_tool"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            tool_id=self.tool_id,
            tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
            supported_subgoals=(OperationalSubgoal.QUERY_UNCERTAINTY,),
            description=(
                "Explicit deterministic evaluator. It scores generated scenario "
                "candidates over a finite horizon using the same trusted "
                "downstream replenishment rule and cost accounting, then returns "
                "only the selected scenario update."
            ),
            produces_raw_orders=False,
        )

    def invoke(self, invocation: ToolInvocation) -> ToolResult:
        _validate_invocation(invocation, tool_name="ScenarioEvaluatorTool")
        candidate_set = _latest_candidate_set(invocation)
        diagnosis = _latest_diagnosis(invocation)
        regime_label = _resolve_regime(invocation)
        demand_window = _evidence_series(
            invocation.evidence.demand,
            invocation.observation.demand_realization[-1],
        )
        leadtime_window = _evidence_series(
            invocation.evidence.leadtime,
            invocation.observation.leadtime_realization[-1],
        )
        scenario_paths = _scenario_paths(
            demand_window=demand_window,
            leadtime_window=leadtime_window,
            config=self.rolling_config,
            time_index=invocation.system_state.time_index,
        )
        scored = tuple(
            (
                _expected_update_cost(
                    candidate=candidate,
                    diagnosis=diagnosis,
                    system_state=invocation.system_state,
                    benchmark_case=self.benchmark_case,
                    scenario_paths=scenario_paths,
                    regime_label=regime_label,
                ),
                priority,
                candidate,
            )
            for priority, candidate in enumerate(candidate_set.candidates)
        )
        selected_cost, _, selected_candidate = min(scored, key=lambda item: (item[0], item[1]))
        incumbent_cost = next(
            expected_cost
            for expected_cost, _, candidate in scored
            if candidate.candidate_id == candidate_set.incumbent_candidate_id
        )
        score_records = tuple(
            PlannerCandidateScoreRecord(
                candidate_id=candidate.candidate_id,
                expected_cost=expected_cost,
                demand_outlook=candidate.demand_outlook,
                leadtime_outlook=candidate.leadtime_outlook,
                safety_buffer_scale=candidate.safety_buffer_scale,
                selected=candidate.candidate_id == selected_candidate.candidate_id,
            )
            for expected_cost, _, candidate in scored
        )
        diagnostics = ScenarioPlannerEvaluationDiagnostics(
            selected_candidate_id=selected_candidate.candidate_id,
            incumbent_candidate_id=candidate_set.incumbent_candidate_id,
            horizon_length=self.rolling_config.horizon_length,
            scenario_count=self.rolling_config.scenario_count,
            candidate_count=len(candidate_set.candidates),
            selected_expected_cost=selected_cost,
            incumbent_expected_cost=incumbent_cost,
            candidate_scores=score_records,
        )
        selected_update = _candidate_to_update(selected_candidate, regime_label)
        return ToolResult(
            tool_id=invocation.tool_id,
            tool_class=invocation.tool_class,
            subgoal=invocation.subgoal,
            status=ToolStatus.SUCCESS if selected_update.request_replan else ToolStatus.NO_ACTION,
            structured_output={
                "scenario_update_result": selected_update,
                "scenario_planner_evaluation": diagnostics,
            },
            confidence=0.93,
            provenance="scenario_evaluator_tool",
            next_subgoal=(
                OperationalSubgoal.REQUEST_REPLAN
                if selected_update.request_replan
                else OperationalSubgoal.NO_ACTION
            ),
            request_replan=selected_update.request_replan,
            emits_raw_orders=False,
        )


@dataclass(frozen=True, slots=True)
class RiskSensitiveScenarioEvaluatorTool(BoundedTool):
    """Evaluate candidates under regime ambiguity and bad-tail risk."""

    benchmark_case: SerialBenchmarkCase
    rolling_config: ScenarioRollingHorizonPolicyConfig = (
        ScenarioRollingHorizonPolicyConfig()
    )
    tail_fraction: float = 0.25
    tool_id: str = "risk_sensitive_scenario_evaluator_tool"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            tool_id=self.tool_id,
            tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
            supported_subgoals=(OperationalSubgoal.QUERY_UNCERTAINTY,),
            description=(
                "Risk-sensitive deterministic evaluator. It scores generated "
                "scenario candidates using the same trusted downstream rule, "
                "but evaluates mean cost, bad-tail cost, service-risk exposure, "
                "and overreaction under a bounded regime-belief mixture."
            ),
            produces_raw_orders=False,
        )

    def invoke(self, invocation: ToolInvocation) -> ToolResult:
        _validate_invocation(invocation, tool_name="RiskSensitiveScenarioEvaluatorTool")
        candidate_set = _latest_candidate_set(invocation)
        diagnosis = _latest_diagnosis(invocation)
        belief = _latest_regime_belief(invocation)
        scenario_paths = _belief_scenario_paths(
            diagnosis=diagnosis,
            belief=belief,
            config=self.rolling_config,
            time_index=invocation.system_state.time_index,
        )
        scored = tuple(
            (
                _risk_sensitive_candidate_score(
                    candidate=candidate,
                    diagnosis=diagnosis,
                    belief=belief,
                    system_state=invocation.system_state,
                    benchmark_case=self.benchmark_case,
                    scenario_paths=scenario_paths,
                    tail_fraction=self.tail_fraction,
                ),
                priority,
                candidate,
            )
            for priority, candidate in enumerate(candidate_set.candidates)
        )
        selected_score, _, selected_candidate = min(
            scored,
            key=lambda item: (item[0].expected_cost, item[1]),
        )
        incumbent_score = next(
            score
            for score, _, candidate in scored
            if candidate.candidate_id == candidate_set.incumbent_candidate_id
        )
        score_records = tuple(
            PlannerCandidateScoreRecord(
                candidate_id=candidate.candidate_id,
                expected_cost=score.expected_cost,
                demand_outlook=candidate.demand_outlook,
                leadtime_outlook=candidate.leadtime_outlook,
                safety_buffer_scale=candidate.safety_buffer_scale,
                mean_cost=score.mean_cost,
                tail_cost=score.tail_cost,
                service_risk_penalty=score.service_risk_penalty,
                overreaction_penalty=score.overreaction_penalty,
                selected=candidate.candidate_id == selected_candidate.candidate_id,
            )
            for score, _, candidate in scored
        )
        diagnostics = ScenarioPlannerEvaluationDiagnostics(
            selected_candidate_id=selected_candidate.candidate_id,
            incumbent_candidate_id=candidate_set.incumbent_candidate_id,
            horizon_length=self.rolling_config.horizon_length,
            scenario_count=len(scenario_paths),
            candidate_count=len(candidate_set.candidates),
            selected_expected_cost=selected_score.expected_cost,
            incumbent_expected_cost=incumbent_score.expected_cost,
            candidate_scores=score_records,
        )
        selected_update = _candidate_to_update(selected_candidate, belief.dominant_regime_label)
        return ToolResult(
            tool_id=invocation.tool_id,
            tool_class=invocation.tool_class,
            subgoal=invocation.subgoal,
            status=ToolStatus.SUCCESS if selected_update.request_replan else ToolStatus.NO_ACTION,
            structured_output={
                "scenario_update_result": selected_update,
                "scenario_planner_evaluation": diagnostics,
            },
            confidence=0.93,
            provenance="risk_sensitive_scenario_evaluator_tool",
            next_subgoal=(
                OperationalSubgoal.REQUEST_REPLAN
                if selected_update.request_replan
                else OperationalSubgoal.NO_ACTION
            ),
            request_replan=selected_update.request_replan,
            emits_raw_orders=False,
        )


@dataclass(frozen=True, slots=True)
class CounterfactualRegretGuardTool(BoundedTool):
    """Guard against overcomplicated candidate choices in simple contexts."""

    tool_id: str = "counterfactual_regret_guard_tool"
    immediate_shift_margin: float = 75.0
    clean_recovery_margin: float = 8.0

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            tool_id=self.tool_id,
            tool_class=ToolClass.DETERMINISTIC_STATISTICAL,
            supported_subgoals=(
                OperationalSubgoal.QUERY_UNCERTAINTY,
                OperationalSubgoal.REQUEST_REPLAN,
            ),
            description=(
                "Counterfactual regret guard. It inspects the risk-sensitive "
                "candidate scores and may replace the selected scenario input "
                "with a simpler already-scored candidate when the selected "
                "candidate appears to overreact in a clean normal, immediate "
                "shift, or clean recovery context."
            ),
            produces_raw_orders=False,
        )

    def invoke(self, invocation: ToolInvocation) -> ToolResult:
        _validate_invocation(invocation, tool_name="CounterfactualRegretGuardTool")
        candidate_set = _latest_candidate_set(invocation)
        diagnosis = _latest_diagnosis(invocation)
        belief = _latest_regime_belief(invocation)
        evaluation = _latest_planner_evaluation(invocation)
        selected_candidate_id, guard_reason = _guarded_candidate_id(
            diagnosis=diagnosis,
            belief=belief,
            evaluation=evaluation,
            time_index=invocation.system_state.time_index,
            immediate_shift_margin=self.immediate_shift_margin,
            clean_recovery_margin=self.clean_recovery_margin,
        )
        selected_candidate = _candidate_by_id(candidate_set, selected_candidate_id)
        guarded_scores = tuple(
            PlannerCandidateScoreRecord(
                candidate_id=score.candidate_id,
                expected_cost=score.expected_cost,
                demand_outlook=score.demand_outlook,
                leadtime_outlook=score.leadtime_outlook,
                safety_buffer_scale=score.safety_buffer_scale,
                mean_cost=score.mean_cost,
                tail_cost=score.tail_cost,
                service_risk_penalty=score.service_risk_penalty,
                overreaction_penalty=score.overreaction_penalty,
                selected=score.candidate_id == selected_candidate_id,
            )
            for score in evaluation.candidate_scores
        )
        selected_score = _score_by_id(evaluation, selected_candidate_id)
        guarded_evaluation = ScenarioPlannerEvaluationDiagnostics(
            selected_candidate_id=selected_candidate_id,
            incumbent_candidate_id=evaluation.incumbent_candidate_id,
            horizon_length=evaluation.horizon_length,
            scenario_count=evaluation.scenario_count,
            candidate_count=evaluation.candidate_count,
            selected_expected_cost=selected_score.expected_cost,
            incumbent_expected_cost=evaluation.incumbent_expected_cost,
            candidate_scores=guarded_scores,
        )
        selected_update = _candidate_to_update(selected_candidate, belief.dominant_regime_label)
        return ToolResult(
            tool_id=invocation.tool_id,
            tool_class=invocation.tool_class,
            subgoal=invocation.subgoal,
            status=ToolStatus.SUCCESS if selected_update.request_replan else ToolStatus.NO_ACTION,
            structured_output={
                "scenario_update_result": selected_update,
                "scenario_planner_evaluation": guarded_evaluation,
                "counterfactual_regret_guard": CounterfactualRegretGuardRecord(
                    initial_selected_candidate_id=evaluation.selected_candidate_id,
                    final_selected_candidate_id=selected_candidate_id,
                    guard_changed_selection=(
                        selected_candidate_id != evaluation.selected_candidate_id
                    ),
                    guard_reason=guard_reason,
                ),
            },
            confidence=0.91,
            provenance="counterfactual_regret_guard_tool",
            next_subgoal=(
                OperationalSubgoal.REQUEST_REPLAN
                if selected_update.request_replan
                else OperationalSubgoal.NO_ACTION
            ),
            request_replan=selected_update.request_replan,
            emits_raw_orders=False,
        )


def _build_diagnosis(invocation: ToolInvocation) -> RegimeDiagnosisRecord:
    regime_label = _resolve_regime(invocation)
    latest_demand = float(invocation.observation.demand_realization[-1])
    latest_leadtime = float(invocation.observation.leadtime_realization[-1])
    demand_window = _evidence_series(invocation.evidence.demand, latest_demand)
    leadtime_window = _evidence_series(invocation.evidence.leadtime, latest_leadtime)
    demand_baseline = invocation.evidence.demand_baseline_value or max(mean(demand_window), 1.0)
    leadtime_baseline = invocation.evidence.leadtime_baseline_value or max(mean(leadtime_window), 1.0)
    demand_ratio = latest_demand / demand_baseline if demand_baseline > 0.0 else 1.0
    leadtime_ratio = latest_leadtime / leadtime_baseline if leadtime_baseline > 0.0 else 1.0
    recent_ratios = tuple(value / demand_baseline for value in demand_window[-4:])
    recent_stress_count = sum(1 for ratio in recent_ratios if ratio >= 1.15)
    total_backorder = sum(invocation.system_state.backorder_level)
    pipeline_total = sum(invocation.system_state.pipeline_inventory)
    case_family = _case_family(
        regime_label=regime_label,
        demand_ratio=demand_ratio,
        leadtime_ratio=leadtime_ratio,
        recent_stress_count=recent_stress_count,
        total_backorder=total_backorder,
        latest_demand=latest_demand,
    )
    update_types: tuple[UpdateRequestType, ...] = ()
    rationale = "Deterministic diagnosis from structured evidence."
    if invocation.agent_assessment is not None:
        update_types = tuple(
            update_request.request_type
            for update_request in invocation.agent_assessment.update_requests
        )
        rationale = invocation.agent_assessment.rationale
    return RegimeDiagnosisRecord(
        regime_label=regime_label,
        case_family=case_family,
        demand_ratio_to_baseline=demand_ratio,
        leadtime_ratio_to_baseline=leadtime_ratio,
        recent_stress_count=recent_stress_count,
        total_backorder=total_backorder,
        pipeline_total=pipeline_total,
        latest_demand=latest_demand,
        latest_leadtime=latest_leadtime,
        demand_window=demand_window,
        leadtime_window=leadtime_window,
        agent_update_request_types=update_types,
        rationale=rationale,
    )


def _case_family(
    *,
    regime_label: RegimeLabel,
    demand_ratio: float,
    leadtime_ratio: float,
    recent_stress_count: int,
    total_backorder: float,
    latest_demand: float,
) -> str:
    demand_stress = demand_ratio >= 1.15
    leadtime_stress = leadtime_ratio >= 1.15
    if regime_label is RegimeLabel.RECOVERY:
        if total_backorder >= max(1.0, latest_demand * 0.35):
            return "recovery_with_carryover_load"
        return "recovery_or_false_alarm"
    if demand_stress and leadtime_stress:
        return "joint_demand_leadtime_stress"
    if leadtime_stress:
        return "leadtime_stress"
    if demand_stress:
        return "sustained_demand_shift" if recent_stress_count >= 2 else "initial_demand_shift"
    return "stable_or_low_risk"


def _build_regime_belief(diagnosis: RegimeDiagnosisRecord) -> RegimeBeliefRecord:
    """Build a small, normalized hidden-regime belief from the diagnosis."""

    if diagnosis.case_family == "stable_or_low_risk":
        raw_entries = (
            (
                RegimeLabel.NORMAL,
                0.62,
                1.00,
                1.00,
                "Current evidence is close to baseline.",
            ),
            (
                RegimeLabel.DEMAND_REGIME_SHIFT,
                0.18,
                1.22,
                1.00,
                "Keep a small change-point belief for delayed demand shifts.",
            ),
            (
                RegimeLabel.RECOVERY,
                0.10,
                0.92,
                1.00,
                "Allow a low-demand continuation branch.",
            ),
            (
                RegimeLabel.SUPPLY_DISRUPTION,
                0.10,
                1.00,
                1.18,
                "Allow a modest lead-time-risk branch.",
            ),
        )
        tail_weight = 0.10
        service_weight = 0.20
        overreaction_weight = 0.15
        rationale = "Stable evidence with a bounded residual belief over delayed stress."
    elif diagnosis.case_family == "initial_demand_shift":
        raw_entries = (
            (
                RegimeLabel.DEMAND_REGIME_SHIFT,
                0.62,
                1.25,
                1.00,
                "Demand has moved above baseline.",
            ),
            (
                RegimeLabel.NORMAL,
                0.13,
                0.95,
                1.00,
                "Preserve false-alarm possibility.",
            ),
            (
                RegimeLabel.RECOVERY,
                0.10,
                0.90,
                1.00,
                "Allow quick mean reversion.",
            ),
            (
                RegimeLabel.JOINT_DISRUPTION,
                0.15,
                1.25,
                1.15,
                "Demand stress can coincide with lead-time stress.",
            ),
        )
        tail_weight = 0.14
        service_weight = 0.28
        overreaction_weight = 0.08
        rationale = "Initial shift belief balances response with false-alarm risk."
    elif diagnosis.case_family == "sustained_demand_shift":
        raw_entries = (
            (
                RegimeLabel.DEMAND_REGIME_SHIFT,
                0.72,
                1.30,
                1.00,
                "Repeated demand stress supports persistence.",
            ),
            (
                RegimeLabel.JOINT_DISRUPTION,
                0.16,
                1.28,
                1.16,
                "Persistent demand stress may coincide with supply stress.",
            ),
            (
                RegimeLabel.RECOVERY,
                0.08,
                0.92,
                1.00,
                "Retain a small recovery branch.",
            ),
            (
                RegimeLabel.NORMAL,
                0.04,
                0.96,
                1.00,
                "Retain a low normal branch.",
            ),
        )
        tail_weight = 0.18
        service_weight = 0.32
        overreaction_weight = 0.05
        rationale = "Sustained stress belief emphasizes persistence and service risk."
    elif diagnosis.case_family == "recovery_with_carryover_load":
        raw_entries = (
            (
                RegimeLabel.RECOVERY,
                0.40,
                0.92,
                1.00,
                "Observed recovery remains plausible.",
            ),
            (
                RegimeLabel.DEMAND_REGIME_SHIFT,
                0.36,
                1.20,
                1.00,
                "Carry-over load leaves relapse risk.",
            ),
            (
                RegimeLabel.JOINT_DISRUPTION,
                0.14,
                1.16,
                1.15,
                "Carry-over load may expose supply stress.",
            ),
            (
                RegimeLabel.NORMAL,
                0.10,
                1.00,
                1.00,
                "Allow normalization after recovery.",
            ),
        )
        tail_weight = 0.16
        service_weight = 0.30
        overreaction_weight = 0.10
        rationale = "Recovery is credible but carry-over load keeps relapse risk active."
    elif diagnosis.case_family == "recovery_or_false_alarm":
        raw_entries = (
            (
                RegimeLabel.RECOVERY,
                0.56,
                0.88,
                1.00,
                "Recovery evidence is relatively clean.",
            ),
            (
                RegimeLabel.NORMAL,
                0.24,
                1.00,
                1.00,
                "Normal continuation remains plausible.",
            ),
            (
                RegimeLabel.DEMAND_REGIME_SHIFT,
                0.15,
                1.16,
                1.00,
                "Keep a small false-recovery branch.",
            ),
            (
                RegimeLabel.SUPPLY_DISRUPTION,
                0.05,
                1.00,
                1.12,
                "Low residual supply-risk branch.",
            ),
        )
        tail_weight = 0.10
        service_weight = 0.18
        overreaction_weight = 0.24
        rationale = "Clean recovery belief penalizes unnecessary over-protection."
    elif diagnosis.case_family == "leadtime_stress":
        raw_entries = (
            (
                RegimeLabel.SUPPLY_DISRUPTION,
                0.66,
                1.00,
                1.24,
                "Lead-time realization is above baseline.",
            ),
            (
                RegimeLabel.JOINT_DISRUPTION,
                0.18,
                1.15,
                1.22,
                "Supply stress can combine with demand stress.",
            ),
            (
                RegimeLabel.NORMAL,
                0.10,
                1.00,
                1.00,
                "Allow mean reversion.",
            ),
            (
                RegimeLabel.RECOVERY,
                0.06,
                0.92,
                1.00,
                "Low recovery branch.",
            ),
        )
        tail_weight = 0.18
        service_weight = 0.30
        overreaction_weight = 0.06
        rationale = "Lead-time stress belief emphasizes tail service exposure."
    else:
        raw_entries = (
            (
                RegimeLabel.JOINT_DISRUPTION,
                0.54,
                1.24,
                1.20,
                "Demand and lead-time stress appear together.",
            ),
            (
                RegimeLabel.DEMAND_REGIME_SHIFT,
                0.24,
                1.22,
                1.00,
                "Demand stress may dominate.",
            ),
            (
                RegimeLabel.SUPPLY_DISRUPTION,
                0.14,
                1.00,
                1.20,
                "Lead-time stress may dominate.",
            ),
            (
                RegimeLabel.RECOVERY,
                0.08,
                0.92,
                1.00,
                "Retain low recovery branch.",
            ),
        )
        tail_weight = 0.20
        service_weight = 0.34
        overreaction_weight = 0.04
        rationale = "Joint stress belief prioritizes bad-tail and service exposure."
    entries = _normalize_belief_entries(raw_entries)
    dominant = max(entries, key=lambda entry: entry.probability).regime_label
    entropy = -sum(
        entry.probability * math.log(entry.probability)
        for entry in entries
        if entry.probability > 0.0
    )
    return RegimeBeliefRecord(
        entries=entries,
        dominant_regime_label=dominant,
        belief_entropy=entropy,
        tail_risk_weight=tail_weight,
        service_risk_weight=service_weight,
        overreaction_weight=overreaction_weight,
        rationale=rationale,
    )


def _normalize_belief_entries(
    raw_entries: tuple[tuple[RegimeLabel, float, float, float, str], ...],
) -> tuple[RegimeBeliefEntry, ...]:
    total = sum(probability for _, probability, _, _, _ in raw_entries)
    if total <= 0.0:
        raise ValueError("At least one regime-belief probability must be positive.")
    return tuple(
        RegimeBeliefEntry(
            regime_label=regime_label,
            probability=probability / total,
            demand_multiplier=demand_multiplier,
            leadtime_multiplier=leadtime_multiplier,
            rationale=rationale,
        )
        for regime_label, probability, demand_multiplier, leadtime_multiplier, rationale in raw_entries
    )


def _build_candidate_set(
    *,
    invocation: ToolInvocation,
    diagnosis: RegimeDiagnosisRecord,
    benchmark_case: SerialBenchmarkCase,
    robust_config: RobustPolicyConfig,
    rolling_config: ScenarioRollingHorizonPolicyConfig,
) -> ScenarioCandidateSet:
    candidates: list[ScenarioCandidateRecord] = []
    candidates.append(_keep_current_candidate(invocation, diagnosis))
    candidates.append(_original_path_candidate(invocation, diagnosis))
    candidates.append(
        _policy_candidate(
            candidate_id="rolling_horizon_incumbent",
            decision=ScenarioRollingHorizonPolicy(config=rolling_config).decide(
                invocation.system_state,
                invocation.observation,
                invocation.evidence,
                benchmark_case,
            ),
            rationale="Fixed-grid scenario rolling horizon incumbent.",
        )
    )
    candidates.append(
        _policy_candidate(
            candidate_id="robust_quantile_protection",
            decision=RobustUncertaintyPolicy(config=robust_config).decide(
                invocation.system_state,
                invocation.observation,
                invocation.evidence,
                benchmark_case,
            ),
            rationale="Empirical high-quantile robust protection candidate.",
        )
    )
    candidates.extend(_agentic_regime_candidates(diagnosis))
    deduped = _dedupe_candidates(tuple(candidates))
    return ScenarioCandidateSet(
        candidates=deduped,
        incumbent_candidate_id="rolling_horizon_incumbent",
        generator_notes=(
            "rolling_horizon_incumbent_included",
            f"case_family:{diagnosis.case_family}",
            "agentic_candidates_are_regime_conditioned_bounded_scenario_inputs",
        ),
    )


def _keep_current_candidate(
    invocation: ToolInvocation,
    diagnosis: RegimeDiagnosisRecord,
) -> ScenarioCandidateRecord:
    return ScenarioCandidateRecord(
        candidate_id="keep_current",
        provenance="scenario_planner_keep_current",
        demand_outlook=diagnosis.latest_demand,
        leadtime_outlook=diagnosis.latest_leadtime,
        safety_buffer_scale=1.0,
        applied_update_types=(UpdateRequestType.KEEP_CURRENT,),
        request_replan=False,
        rationale="Preserve current observed scenario inputs.",
    )


def _original_path_candidate(
    invocation: ToolInvocation,
    diagnosis: RegimeDiagnosisRecord,
) -> ScenarioCandidateRecord:
    forecast_tool = DeterministicForecastTool()
    leadtime_tool = DeterministicLeadTimeTool()
    scenario_tool = DeterministicScenarioTool()
    forecast_result = forecast_tool.invoke(
        ToolInvocation(
            tool_id=forecast_tool.spec.tool_id,
            tool_class=forecast_tool.spec.tool_class,
            subgoal=invocation.subgoal,
            evidence=invocation.evidence,
            system_state=invocation.system_state,
            observation=invocation.observation,
            agent_assessment=invocation.agent_assessment,
        )
    )
    leadtime_result = leadtime_tool.invoke(
        ToolInvocation(
            tool_id=leadtime_tool.spec.tool_id,
            tool_class=leadtime_tool.spec.tool_class,
            subgoal=invocation.subgoal,
            evidence=invocation.evidence,
            system_state=invocation.system_state,
            observation=invocation.observation,
            agent_assessment=invocation.agent_assessment,
            prior_results=(forecast_result,),
        )
    )
    scenario_result = scenario_tool.invoke(
        ToolInvocation(
            tool_id=scenario_tool.spec.tool_id,
            tool_class=scenario_tool.spec.tool_class,
            subgoal=OperationalSubgoal.UPDATE_UNCERTAINTY,
            evidence=invocation.evidence,
            system_state=invocation.system_state,
            observation=invocation.observation,
            agent_assessment=invocation.agent_assessment,
            prior_results=(forecast_result, leadtime_result),
        )
    )
    value = scenario_result.structured_output.get("scenario_update_result")
    if not isinstance(value, ScenarioUpdateResult):
        raise TypeError("Original scenario path did not return ScenarioUpdateResult.")
    return _update_candidate(
        candidate_id="original_evidence_path",
        scenario_update_result=value,
        rationale=(
            "Original forecast-leadtime-scenario path candidate generated "
            f"for {diagnosis.case_family}."
        ),
    )


def _policy_candidate(
    *,
    candidate_id: str,
    decision,
    rationale: str,
) -> ScenarioCandidateRecord:
    return _update_candidate(
        candidate_id=candidate_id,
        scenario_update_result=decision.scenario_update_result,
        rationale=rationale,
    )


def _update_candidate(
    *,
    candidate_id: str,
    scenario_update_result: ScenarioUpdateResult,
    rationale: str,
) -> ScenarioCandidateRecord:
    return ScenarioCandidateRecord(
        candidate_id=candidate_id,
        provenance=scenario_update_result.provenance or candidate_id,
        demand_outlook=scenario_update_result.adjustment.demand_outlook,
        leadtime_outlook=scenario_update_result.adjustment.leadtime_outlook,
        safety_buffer_scale=scenario_update_result.adjustment.safety_buffer_scale,
        applied_update_types=scenario_update_result.applied_update_types,
        request_replan=scenario_update_result.request_replan,
        rationale=rationale,
    )


def _agentic_regime_candidates(
    diagnosis: RegimeDiagnosisRecord,
) -> tuple[ScenarioCandidateRecord, ...]:
    high_demand = max(
        diagnosis.latest_demand,
        _empirical_quantile(diagnosis.demand_window, 0.75),
        mean(diagnosis.demand_window),
    )
    high_leadtime = max(
        1.0,
        diagnosis.latest_leadtime,
        _empirical_quantile(diagnosis.leadtime_window, 0.75),
    )
    candidates: list[ScenarioCandidateRecord] = []
    if diagnosis.case_family in {
        "initial_demand_shift",
        "sustained_demand_shift",
        "joint_demand_leadtime_stress",
    }:
        candidates.append(
            _bounded_candidate(
                candidate_id="agentic_fast_demand_reweight",
                diagnosis=diagnosis,
                demand_outlook=high_demand * 1.08,
                leadtime_outlook=diagnosis.latest_leadtime,
                safety_buffer_scale=1.02,
                update_types=(UpdateRequestType.REWEIGHT_SCENARIOS,),
                rationale="Fast demand reweighting without excessive buffer widening.",
            )
        )
        candidates.append(
            _bounded_candidate(
                candidate_id="agentic_sustained_shift_guard",
                diagnosis=diagnosis,
                demand_outlook=high_demand * 1.15,
                leadtime_outlook=diagnosis.latest_leadtime,
                safety_buffer_scale=1.00,
                update_types=(UpdateRequestType.SWITCH_DEMAND_REGIME,),
                rationale="Sustained-shift candidate emphasizes demand level over safety inflation.",
            )
        )
    if diagnosis.case_family in {
        "leadtime_stress",
        "joint_demand_leadtime_stress",
        "recovery_with_carryover_load",
    }:
        candidates.append(
            _bounded_candidate(
                candidate_id="agentic_leadtime_guard",
                diagnosis=diagnosis,
                demand_outlook=max(diagnosis.latest_demand, mean(diagnosis.demand_window)),
                leadtime_outlook=high_leadtime * 1.10,
                safety_buffer_scale=1.05,
                update_types=(UpdateRequestType.SWITCH_LEADTIME_REGIME,),
                rationale="Lead-time protection candidate separates supply risk from demand risk.",
            )
        )
    if diagnosis.case_family in {
        "recovery_or_false_alarm",
        "recovery_with_carryover_load",
    }:
        candidates.append(
            _bounded_candidate(
                candidate_id="agentic_recovery_relaxation",
                diagnosis=diagnosis,
                demand_outlook=max(diagnosis.latest_demand, mean(diagnosis.demand_window) * 0.95),
                leadtime_outlook=diagnosis.latest_leadtime,
                safety_buffer_scale=1.00,
                update_types=(UpdateRequestType.REWEIGHT_SCENARIOS,),
                rationale="Recovery candidate relaxes buffer while keeping demand anchored.",
            )
        )
        candidates.append(
            _bounded_candidate(
                candidate_id="agentic_recovery_carryover_guard",
                diagnosis=diagnosis,
                demand_outlook=max(diagnosis.latest_demand, mean(diagnosis.demand_window)),
                leadtime_outlook=high_leadtime,
                safety_buffer_scale=1.04,
                update_types=(UpdateRequestType.REWEIGHT_SCENARIOS,),
                rationale="Recovery guard keeps moderate protection when backlog or pipeline load remains.",
            )
        )
    return tuple(candidates)


def _bounded_candidate(
    *,
    candidate_id: str,
    diagnosis: RegimeDiagnosisRecord,
    demand_outlook: float,
    leadtime_outlook: float,
    safety_buffer_scale: float,
    update_types: tuple[UpdateRequestType, ...],
    rationale: str,
) -> ScenarioCandidateRecord:
    demand_cap = max(diagnosis.latest_demand, mean(diagnosis.demand_window), 1.0) * 1.75
    leadtime_cap = max(diagnosis.latest_leadtime, mean(diagnosis.leadtime_window), 1.0) * 1.50
    return ScenarioCandidateRecord(
        candidate_id=candidate_id,
        provenance="scenario_candidate_generator_tool",
        demand_outlook=min(max(0.0, demand_outlook), demand_cap),
        leadtime_outlook=min(max(1.0, leadtime_outlook), leadtime_cap),
        safety_buffer_scale=min(max(0.85, safety_buffer_scale), 1.25),
        applied_update_types=update_types,
        request_replan=True,
        rationale=rationale,
    )


def _dedupe_candidates(
    candidates: tuple[ScenarioCandidateRecord, ...],
) -> tuple[ScenarioCandidateRecord, ...]:
    by_id: dict[str, ScenarioCandidateRecord] = {}
    for candidate in candidates:
        by_id.setdefault(candidate.candidate_id, candidate)
    return tuple(by_id.values())


def _candidate_to_update(
    candidate: ScenarioCandidateRecord,
    regime_label: RegimeLabel,
) -> ScenarioUpdateResult:
    return ScenarioUpdateResult(
        scenarios=(
            ScenarioSummary(
                scenario_id=f"scenario_planner_{candidate.candidate_id}",
                regime_label=regime_label,
                weight=1.0,
                demand_scale=1.0,
                leadtime_scale=1.0,
            ),
        ),
        applied_update_types=candidate.applied_update_types,
        adjustment=ScenarioAdjustmentSummary(
            demand_outlook=candidate.demand_outlook,
            leadtime_outlook=candidate.leadtime_outlook,
            safety_buffer_scale=candidate.safety_buffer_scale,
        ),
        request_replan=candidate.request_replan,
        provenance=f"scenario_planner_selected:{candidate.candidate_id}",
    )


def _expected_update_cost(
    *,
    candidate: ScenarioCandidateRecord,
    diagnosis: RegimeDiagnosisRecord,
    system_state: SimulationState,
    benchmark_case: SerialBenchmarkCase,
    scenario_paths: tuple[tuple[tuple[float, float], ...], ...],
    regime_label: RegimeLabel,
) -> float:
    optimizer = TrustedOptimizerAdapter()
    scenario_update_result = _candidate_to_update(candidate, regime_label)
    total_cost = 0.0
    for scenario_path in scenario_paths:
        simulated_state = system_state
        for demand_value, leadtime_value in scenario_path:
            observation = Observation(
                time_index=simulated_state.time_index,
                demand_evidence=DemandEvidence(
                    history=(demand_value,),
                    latest_realization=(demand_value,),
                    stage_index=1,
                ),
                leadtime_evidence=LeadTimeEvidence(
                    history=(leadtime_value,),
                    latest_realization=(leadtime_value,),
                    upstream_stage_index=2,
                    downstream_stage_index=1,
                ),
                regime_label=regime_label,
                notes=("scenario_planner_candidate_evaluation",),
            )
            optimization_request = build_optimization_request(
                system_state=simulated_state,
                scenario_update_result=scenario_update_result,
                base_stock_levels=benchmark_case.base_stock_levels,
                planning_horizon=1,
            )
            optimization_result = optimizer.solve(optimization_request)
            transition = advance_serial_state(
                benchmark_case,
                current_state=simulated_state,
                observation=observation,
                optimization_result=optimization_result,
                next_regime=regime_label,
            )
            period_record = PeriodTraceRecord(
                time_index=simulated_state.time_index,
                regime_label=regime_label,
                state=simulated_state,
                observation=observation,
                agent_signal=AgentSignal(
                    selected_subgoal=(
                        OperationalSubgoal.REQUEST_REPLAN
                        if scenario_update_result.request_replan
                        else OperationalSubgoal.NO_ACTION
                    ),
                    request_replan=scenario_update_result.request_replan,
                    no_action=not scenario_update_result.request_replan,
                    rationale="scenario planner candidate evaluation",
                ),
                optimization_result=optimization_result,
                next_state=transition.next_state,
                realized_demand=transition.realized_demand,
                demand_load=transition.demand_load,
                served_demand=transition.served_demand,
                unmet_demand=transition.unmet_demand,
                notes=transition.notes,
            )
            total_cost += compute_period_total_cost(
                period_record,
                benchmark_case.benchmark_config.costs,
                holding_cost_by_stage=benchmark_case.holding_costs,
                stockout_cost_by_stage=benchmark_case.stockout_costs,
            )
            simulated_state = transition.next_state
    expected_cost = total_cost / float(len(scenario_paths))
    return expected_cost + _uncertainty_quality_penalty(
        candidate,
        diagnosis,
    )


def _risk_sensitive_candidate_score(
    *,
    candidate: ScenarioCandidateRecord,
    diagnosis: RegimeDiagnosisRecord,
    belief: RegimeBeliefRecord,
    system_state: SimulationState,
    benchmark_case: SerialBenchmarkCase,
    scenario_paths: tuple[tuple[tuple[float, float, RegimeLabel], ...], ...],
    tail_fraction: float,
) -> PlannerCandidateScoreRecord:
    outcomes = tuple(
        _candidate_path_outcome(
            candidate=candidate,
            system_state=system_state,
            benchmark_case=benchmark_case,
            scenario_path=scenario_path,
        )
        for scenario_path in scenario_paths
    )
    path_costs = tuple(outcome[0] for outcome in outcomes)
    unmet_loads = tuple(outcome[1] for outcome in outcomes)
    final_backorders = tuple(outcome[2] for outcome in outcomes)
    mean_cost = mean(path_costs)
    tail_cost = _upper_tail_mean(path_costs, tail_fraction)
    service_risk = mean(unmet_loads) + 0.25 * mean(final_backorders)
    service_penalty = belief.service_risk_weight * service_risk
    overreaction_penalty = _overreaction_penalty(candidate, diagnosis, belief)
    objective = (
        mean_cost
        + belief.tail_risk_weight * tail_cost
        + service_penalty
        + overreaction_penalty
        + _uncertainty_quality_penalty(candidate, diagnosis)
    )
    return PlannerCandidateScoreRecord(
        candidate_id=candidate.candidate_id,
        expected_cost=objective,
        demand_outlook=candidate.demand_outlook,
        leadtime_outlook=candidate.leadtime_outlook,
        safety_buffer_scale=candidate.safety_buffer_scale,
        mean_cost=mean_cost,
        tail_cost=tail_cost,
        service_risk_penalty=service_penalty,
        overreaction_penalty=overreaction_penalty,
    )


def _candidate_path_outcome(
    *,
    candidate: ScenarioCandidateRecord,
    system_state: SimulationState,
    benchmark_case: SerialBenchmarkCase,
    scenario_path: tuple[tuple[float, float, RegimeLabel], ...],
) -> tuple[float, float, float]:
    optimizer = TrustedOptimizerAdapter()
    simulated_state = system_state
    scenario_update_result = _candidate_to_update(candidate, scenario_path[0][2])
    total_cost = 0.0
    unmet_load = 0.0
    for demand_value, leadtime_value, regime_label in scenario_path:
        observation = Observation(
            time_index=simulated_state.time_index,
            demand_evidence=DemandEvidence(
                history=(demand_value,),
                latest_realization=(demand_value,),
                stage_index=1,
            ),
            leadtime_evidence=LeadTimeEvidence(
                history=(leadtime_value,),
                latest_realization=(leadtime_value,),
                upstream_stage_index=2,
                downstream_stage_index=1,
            ),
            regime_label=regime_label,
            notes=("risk_sensitive_scenario_candidate_evaluation",),
        )
        optimization_request = build_optimization_request(
            system_state=simulated_state,
            scenario_update_result=scenario_update_result,
            base_stock_levels=benchmark_case.base_stock_levels,
            planning_horizon=1,
        )
        optimization_result = optimizer.solve(optimization_request)
        transition = advance_serial_state(
            benchmark_case,
            current_state=simulated_state,
            observation=observation,
            optimization_result=optimization_result,
            next_regime=regime_label,
        )
        period_record = PeriodTraceRecord(
            time_index=simulated_state.time_index,
            regime_label=regime_label,
            state=simulated_state,
            observation=observation,
            agent_signal=AgentSignal(
                selected_subgoal=(
                    OperationalSubgoal.REQUEST_REPLAN
                    if scenario_update_result.request_replan
                    else OperationalSubgoal.NO_ACTION
                ),
                request_replan=scenario_update_result.request_replan,
                no_action=not scenario_update_result.request_replan,
                rationale="risk-sensitive scenario planner candidate evaluation",
            ),
            optimization_result=optimization_result,
            next_state=transition.next_state,
            realized_demand=transition.realized_demand,
            demand_load=transition.demand_load,
            served_demand=transition.served_demand,
            unmet_demand=transition.unmet_demand,
            notes=transition.notes,
        )
        total_cost += compute_period_total_cost(
            period_record,
            benchmark_case.benchmark_config.costs,
            holding_cost_by_stage=benchmark_case.holding_costs,
            stockout_cost_by_stage=benchmark_case.stockout_costs,
        )
        unmet_load += transition.unmet_demand
        simulated_state = transition.next_state
    return total_cost, unmet_load, sum(simulated_state.backorder_level)


def _upper_tail_mean(values: tuple[float, ...], tail_fraction: float) -> float:
    if not values:
        raise ValueError("values must not be empty.")
    if tail_fraction <= 0.0:
        return max(values)
    ordered = tuple(sorted(values, reverse=True))
    count = max(1, int(math.ceil(len(ordered) * min(tail_fraction, 1.0))))
    return mean(ordered[:count])


def _overreaction_penalty(
    candidate: ScenarioCandidateRecord,
    diagnosis: RegimeDiagnosisRecord,
    belief: RegimeBeliefRecord,
) -> float:
    if belief.overreaction_weight <= 0.0:
        return 0.0
    protected_demand = candidate.demand_outlook * candidate.safety_buffer_scale
    demand_anchor = max(diagnosis.latest_demand, mean(diagnosis.demand_window), 1.0)
    leadtime_anchor = max(diagnosis.latest_leadtime, mean(diagnosis.leadtime_window), 1.0)
    demand_excess = max(0.0, protected_demand / demand_anchor - 1.0)
    leadtime_excess = max(0.0, candidate.leadtime_outlook / leadtime_anchor - 1.0)
    conservative_context = belief.dominant_regime_label in {
        RegimeLabel.NORMAL,
        RegimeLabel.RECOVERY,
    }
    scale = 1.0 if conservative_context else 0.35
    return belief.overreaction_weight * scale * (
        18.0 * demand_excess * demand_excess
        + 10.0 * leadtime_excess * leadtime_excess
    )


def _guarded_candidate_id(
    *,
    diagnosis: RegimeDiagnosisRecord,
    belief: RegimeBeliefRecord,
    evaluation: ScenarioPlannerEvaluationDiagnostics,
    time_index: int,
    immediate_shift_margin: float,
    clean_recovery_margin: float,
) -> tuple[str, str]:
    selected_id = evaluation.selected_candidate_id
    selected_score = _score_by_id(evaluation, selected_id)

    if diagnosis.case_family in {"initial_demand_shift", "sustained_demand_shift"}:
        original_score = _optional_score_by_id(evaluation, "original_evidence_path")
        if (
            time_index == 0
            and original_score is not None
            and original_score.expected_cost <= selected_score.expected_cost + immediate_shift_margin
        ):
            return (
                "original_evidence_path",
                "immediate_clean_shift_prefers_simpler_evidence_path_when_close",
            )

    if (
        diagnosis.case_family == "recovery_or_false_alarm"
        and belief.dominant_regime_label is RegimeLabel.RECOVERY
    ):
        robust_score = _optional_score_by_id(evaluation, "robust_quantile_protection")
        if (
            robust_score is not None
            and robust_score.expected_cost <= selected_score.expected_cost + clean_recovery_margin
        ):
            return (
                "robust_quantile_protection",
                "clean_recovery_prefers_conservative_counterfactual_when_close",
            )
        keep_score = _optional_score_by_id(evaluation, "keep_current")
        if (
            keep_score is not None
            and keep_score.expected_cost <= selected_score.expected_cost + clean_recovery_margin
        ):
            return (
                "keep_current",
                "clean_recovery_prefers_keep_current_when_close",
            )

    return selected_id, "risk_sensitive_selection_retained"


def _uncertainty_quality_penalty(
    candidate: ScenarioCandidateRecord,
    diagnosis: RegimeDiagnosisRecord,
) -> float:
    """Penalize scenario inputs that contradict high-confidence regime evidence."""

    penalty = 0.0
    if diagnosis.case_family in {
        "initial_demand_shift",
        "sustained_demand_shift",
        "joint_demand_leadtime_stress",
    }:
        shift_response_target = _demand_shift_response_target(diagnosis)
        if candidate.demand_outlook < shift_response_target:
            shortfall = shift_response_target - candidate.demand_outlook
            penalty += 40.0 * shortfall * shortfall
        if (
            UpdateRequestType.KEEP_CURRENT in candidate.applied_update_types
            and diagnosis.demand_ratio_to_baseline >= 1.25
        ):
            penalty += 15.0
    if diagnosis.case_family in {"leadtime_stress", "joint_demand_leadtime_stress"}:
        if candidate.leadtime_outlook < diagnosis.latest_leadtime:
            shortfall = diagnosis.latest_leadtime - candidate.leadtime_outlook
            penalty += 6.0 * shortfall * shortfall
    if diagnosis.case_family == "recovery_or_false_alarm":
        overshoot = max(0.0, candidate.demand_outlook - diagnosis.latest_demand)
        penalty += 2.0 * overshoot * overshoot
    return penalty


def _demand_shift_response_target(diagnosis: RegimeDiagnosisRecord) -> float:
    """Minimum credible demand outlook for a diagnosed demand-shift response."""

    base_level = max(
        diagnosis.latest_demand,
        _empirical_quantile(diagnosis.demand_window, 0.75),
        mean(diagnosis.demand_window),
        1.0,
    )
    multiplier = 1.18
    if diagnosis.case_family == "sustained_demand_shift":
        multiplier = 1.22
    if diagnosis.case_family == "joint_demand_leadtime_stress":
        multiplier = 1.20
    if UpdateRequestType.SWITCH_DEMAND_REGIME in diagnosis.agent_update_request_types:
        multiplier += 0.07
    return base_level * multiplier


def _scenario_paths(
    *,
    demand_window: tuple[float, ...],
    leadtime_window: tuple[float, ...],
    config: ScenarioRollingHorizonPolicyConfig,
    time_index: int,
) -> tuple[tuple[tuple[float, float], ...], ...]:
    rng = random.Random(config.random_seed + time_index * 1009)
    return tuple(
        tuple(
            (
                float(rng.choice(demand_window)),
                max(1.0, float(rng.choice(leadtime_window))),
            )
            for _ in range(config.horizon_length)
        )
        for _ in range(config.scenario_count)
    )


def _belief_scenario_paths(
    *,
    diagnosis: RegimeDiagnosisRecord,
    belief: RegimeBeliefRecord,
    config: ScenarioRollingHorizonPolicyConfig,
    time_index: int,
) -> tuple[tuple[tuple[float, float, RegimeLabel], ...], ...]:
    rng = random.Random(config.random_seed + time_index * 1009 + 7919)
    counts = _belief_scenario_counts(belief.entries, config.scenario_count)
    paths: list[tuple[tuple[float, float, RegimeLabel], ...]] = []
    demand_anchor = max(
        diagnosis.latest_demand,
        mean(diagnosis.demand_window),
        _empirical_quantile(diagnosis.demand_window, 0.75),
        1.0,
    )
    leadtime_anchor = max(
        diagnosis.latest_leadtime,
        mean(diagnosis.leadtime_window),
        _empirical_quantile(diagnosis.leadtime_window, 0.75),
        1.0,
    )
    for entry, count in zip(belief.entries, counts, strict=True):
        for path_index in range(count):
            path: list[tuple[float, float, RegimeLabel]] = []
            for horizon_index in range(config.horizon_length):
                demand_value = float(rng.choice(diagnosis.demand_window))
                leadtime_value = max(1.0, float(rng.choice(diagnosis.leadtime_window)))
                if entry.regime_label in {
                    RegimeLabel.DEMAND_REGIME_SHIFT,
                    RegimeLabel.JOINT_DISRUPTION,
                }:
                    demand_value = max(demand_value, demand_anchor)
                    demand_value *= entry.demand_multiplier * (1.0 + 0.03 * horizon_index)
                elif entry.regime_label is RegimeLabel.RECOVERY:
                    demand_value = min(demand_value, demand_anchor)
                    demand_value *= entry.demand_multiplier * max(0.82, 1.0 - 0.03 * horizon_index)
                else:
                    demand_value *= entry.demand_multiplier
                if entry.regime_label in {
                    RegimeLabel.SUPPLY_DISRUPTION,
                    RegimeLabel.JOINT_DISRUPTION,
                }:
                    leadtime_value = max(leadtime_value, leadtime_anchor)
                    leadtime_value *= entry.leadtime_multiplier
                else:
                    leadtime_value *= entry.leadtime_multiplier
                jitter = 1.0 + 0.015 * ((path_index + horizon_index) % 3 - 1)
                path.append(
                    (
                        max(0.0, demand_value * jitter),
                        max(1.0, leadtime_value),
                        entry.regime_label,
                    )
                )
            paths.append(tuple(path))
    return tuple(paths)


def _belief_scenario_counts(
    entries: tuple[RegimeBeliefEntry, ...],
    scenario_count: int,
) -> tuple[int, ...]:
    raw_counts = [entry.probability * scenario_count for entry in entries]
    counts = [int(math.floor(value)) for value in raw_counts]
    remainder = scenario_count - sum(counts)
    fractions = sorted(
        enumerate(value - math.floor(value) for value in raw_counts),
        key=lambda item: item[1],
        reverse=True,
    )
    for index, _ in fractions[:remainder]:
        counts[index] += 1
    for index, count in enumerate(counts):
        if count == 0 and entries[index].probability > 0.0:
            donor = max(range(len(counts)), key=lambda item: counts[item])
            if counts[donor] > 1:
                counts[donor] -= 1
                counts[index] = 1
    return tuple(counts)


def _latest_diagnosis(invocation: ToolInvocation) -> RegimeDiagnosisRecord:
    for result in reversed(invocation.prior_results):
        value = result.structured_output.get("regime_diagnosis")
        if isinstance(value, RegimeDiagnosisRecord):
            return value
    raise ValueError("Scenario candidate generator requires a prior regime diagnosis.")


def _latest_regime_belief(invocation: ToolInvocation) -> RegimeBeliefRecord:
    for result in reversed(invocation.prior_results):
        value = result.structured_output.get("regime_belief")
        if isinstance(value, RegimeBeliefRecord):
            return value
    raise ValueError("Risk-sensitive evaluator requires a prior regime belief.")


def _latest_planner_evaluation(
    invocation: ToolInvocation,
) -> ScenarioPlannerEvaluationDiagnostics:
    for result in reversed(invocation.prior_results):
        value = result.structured_output.get("scenario_planner_evaluation")
        if isinstance(value, ScenarioPlannerEvaluationDiagnostics):
            return value
    raise ValueError("Counterfactual regret guard requires a prior planner evaluation.")


def _latest_candidate_set(invocation: ToolInvocation) -> ScenarioCandidateSet:
    for result in reversed(invocation.prior_results):
        value = result.structured_output.get("scenario_candidate_set")
        if isinstance(value, ScenarioCandidateSet):
            return value
    raise ValueError("Scenario evaluator requires a prior scenario candidate set.")


def _candidate_by_id(
    candidate_set: ScenarioCandidateSet,
    candidate_id: str,
) -> ScenarioCandidateRecord:
    for candidate in candidate_set.candidates:
        if candidate.candidate_id == candidate_id:
            return candidate
    raise ValueError(f"Unknown scenario candidate id: {candidate_id!r}.")


def _score_by_id(
    evaluation: ScenarioPlannerEvaluationDiagnostics,
    candidate_id: str,
) -> PlannerCandidateScoreRecord:
    score = _optional_score_by_id(evaluation, candidate_id)
    if score is None:
        raise ValueError(f"Unknown scenario candidate score id: {candidate_id!r}.")
    return score


def _optional_score_by_id(
    evaluation: ScenarioPlannerEvaluationDiagnostics,
    candidate_id: str,
) -> PlannerCandidateScoreRecord | None:
    for score in evaluation.candidate_scores:
        if score.candidate_id == candidate_id:
            return score
    return None


def _resolve_regime(invocation: ToolInvocation) -> RegimeLabel:
    if invocation.agent_assessment is not None:
        return invocation.agent_assessment.regime_label
    return (
        invocation.observation.regime_label
        or invocation.system_state.regime_label
        or RegimeLabel.NORMAL
    )


def _evidence_series(evidence, latest_value: float) -> tuple[float, ...]:
    values = tuple(float(value) for value in evidence.history)
    if not values or values[-1] != float(latest_value):
        values = values + (float(latest_value),)
    return values


def _empirical_quantile(values: tuple[float, ...], quantile: float) -> float:
    ordered = tuple(sorted(float(value) for value in values))
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * (len(ordered) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return ordered[lower_index] + fraction * (ordered[upper_index] - ordered[lower_index])


def _validate_invocation(invocation: ToolInvocation, *, tool_name: str) -> None:
    if invocation.system_state is None:
        raise ValueError(f"{tool_name} requires system_state.")
    if invocation.observation is None:
        raise ValueError(f"{tool_name} requires observation.")
    if invocation.evidence is None:
        raise ValueError(f"{tool_name} requires runtime evidence.")


__all__ = [
    "CounterfactualRegretGuardRecord",
    "CounterfactualRegretGuardTool",
    "PlannerCandidateScoreRecord",
    "RegimeBeliefEntry",
    "RegimeBeliefRecord",
    "RegimeBeliefTool",
    "RegimeDiagnosisRecord",
    "RegimeDiagnosisTool",
    "RiskSensitiveScenarioEvaluatorTool",
    "ScenarioCandidateGeneratorTool",
    "ScenarioCandidateRecord",
    "ScenarioCandidateSet",
    "ScenarioEvaluatorTool",
    "ScenarioPlannerEvaluationDiagnostics",
]
