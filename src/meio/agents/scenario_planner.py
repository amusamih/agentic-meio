"""Explicit Agentic AI scenario-planner tools.

The planner keeps the LLM in a bounded supervisory role: the LLM supplies the
regime-facing assessment, these tools turn that assessment into auditable
scenario candidates, and the final tool evaluates candidates with the trusted
downstream replenishment rule. No tool emits raw replenishment orders.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
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


class DemandUncertaintyType(StrEnum):
    """Mean/spread interpretation used before scenario-candidate generation."""

    STABLE = "stable"
    MEAN_SHIFT = "mean_shift"
    SPREAD_INCREASE = "spread_increase"
    SPREAD_DECREASE = "spread_decrease"
    MIXED = "mixed"


@dataclass(frozen=True, slots=True)
class DemandUncertaintyDecompositionRecord:
    """Traceable decomposition of demand mean movement versus spread movement."""

    uncertainty_type: DemandUncertaintyType
    baseline_mean: float
    recent_mean: float
    baseline_cv_reference: float
    recent_std: float
    recent_cv: float
    mean_delta: float
    mean_delta_ratio: float
    spread_ratio: float
    recommended_demand_outlook: float
    recommended_safety_buffer_scale: float
    recommended_update_types: tuple[UpdateRequestType, ...]
    confidence: float
    rationale: str

    def __post_init__(self) -> None:
        if not isinstance(self.uncertainty_type, DemandUncertaintyType):
            raise TypeError("uncertainty_type must be a DemandUncertaintyType.")
        for field_name in (
            "baseline_mean",
            "recent_mean",
            "baseline_cv_reference",
            "recent_std",
            "recent_cv",
            "spread_ratio",
            "recommended_demand_outlook",
            "recommended_safety_buffer_scale",
        ):
            if getattr(self, field_name) < 0.0:
                raise ValueError(f"{field_name} must be non-negative.")
        if self.baseline_mean <= 0.0:
            raise ValueError("baseline_mean must be positive.")
        if self.baseline_cv_reference <= 0.0:
            raise ValueError("baseline_cv_reference must be positive.")
        if self.recommended_demand_outlook <= 0.0:
            raise ValueError("recommended_demand_outlook must be positive.")
        if self.recommended_safety_buffer_scale <= 0.0:
            raise ValueError("recommended_safety_buffer_scale must be positive.")
        object.__setattr__(
            self,
            "recommended_update_types",
            tuple(self.recommended_update_types),
        )
        if not self.recommended_update_types:
            raise ValueError("recommended_update_types must not be empty.")
        for update_type in self.recommended_update_types:
            if not isinstance(update_type, UpdateRequestType):
                raise TypeError(
                    "recommended_update_types must contain UpdateRequestType values."
                )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be within [0.0, 1.0].")
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
            next_tool_id="demand_uncertainty_decomposition_tool",
            next_subgoal=OperationalSubgoal.QUERY_UNCERTAINTY,
            request_replan=False,
            emits_raw_orders=False,
        )


@dataclass(frozen=True, slots=True)
class DemandUncertaintyDecompositionTool(BoundedTool):
    """Separate demand-level movement from demand-spread movement."""

    baseline_cv_reference: float = 0.25
    mean_shift_relative_threshold: float = 0.12
    mean_shift_absolute_threshold: float = 1.0
    spread_increase_ratio_threshold: float = 1.35
    spread_decrease_ratio_threshold: float = 0.60
    tool_id: str = "demand_uncertainty_decomposition_tool"

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
                "Demand uncertainty decomposition tool. It distinguishes demand "
                "mean shifts from spread changes and returns bounded scenario-input "
                "guidance for candidate generation."
            ),
            produces_raw_orders=False,
        )

    def invoke(self, invocation: ToolInvocation) -> ToolResult:
        _validate_invocation(
            invocation,
            tool_name="DemandUncertaintyDecompositionTool",
        )
        diagnosis = _latest_diagnosis(invocation)
        decomposition = _build_demand_uncertainty_decomposition(
            invocation=invocation,
            diagnosis=diagnosis,
            baseline_cv_reference=self.baseline_cv_reference,
            mean_shift_relative_threshold=self.mean_shift_relative_threshold,
            mean_shift_absolute_threshold=self.mean_shift_absolute_threshold,
            spread_increase_ratio_threshold=self.spread_increase_ratio_threshold,
            spread_decrease_ratio_threshold=self.spread_decrease_ratio_threshold,
        )
        return ToolResult(
            tool_id=invocation.tool_id,
            tool_class=invocation.tool_class,
            subgoal=invocation.subgoal,
            status=ToolStatus.SUCCESS,
            structured_output={
                "demand_uncertainty_decomposition": decomposition,
            },
            confidence=decomposition.confidence,
            provenance="demand_uncertainty_decomposition_tool",
            next_tool_id="regime_belief_tool",
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
        decomposition = _latest_demand_uncertainty_decomposition(invocation)
        belief = _build_regime_belief(diagnosis, decomposition)
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
        decomposition = _latest_demand_uncertainty_decomposition(invocation)
        candidate_set = _build_candidate_set(
            invocation=invocation,
            diagnosis=diagnosis,
            decomposition=decomposition,
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
            next_tool_id="risk_sensitive_scenario_evaluator_tool",
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
        decomposition = _latest_demand_uncertainty_decomposition(invocation)
        belief = _latest_regime_belief(invocation)
        scenario_paths = _belief_scenario_paths(
            diagnosis=diagnosis,
            decomposition=decomposition,
            belief=belief,
            config=self.rolling_config,
            time_index=invocation.system_state.time_index,
        )
        scored = tuple(
            (
                _risk_sensitive_candidate_score(
                    candidate=candidate,
                    diagnosis=diagnosis,
                    decomposition=decomposition,
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
        decomposition = _latest_demand_uncertainty_decomposition(invocation)
        belief = _latest_regime_belief(invocation)
        evaluation = _latest_planner_evaluation(invocation)
        selected_candidate_id, guard_reason = _guarded_candidate_id(
            diagnosis=diagnosis,
            decomposition=decomposition,
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


def _build_demand_uncertainty_decomposition(
    *,
    invocation: ToolInvocation,
    diagnosis: RegimeDiagnosisRecord,
    baseline_cv_reference: float,
    mean_shift_relative_threshold: float,
    mean_shift_absolute_threshold: float,
    spread_increase_ratio_threshold: float,
    spread_decrease_ratio_threshold: float,
) -> DemandUncertaintyDecompositionRecord:
    """Classify whether recent demand movement is mostly level, spread, or both."""

    if baseline_cv_reference <= 0.0:
        raise ValueError("baseline_cv_reference must be positive.")
    baseline_mean = (
        float(invocation.evidence.demand_baseline_value)
        if invocation.evidence.demand_baseline_value is not None
        and invocation.evidence.demand_baseline_value > 0.0
        else max(mean(diagnosis.demand_window), 1.0)
    )
    recent_mean = max(mean(diagnosis.demand_window), 0.0)
    recent_std = _population_std(diagnosis.demand_window)
    recent_cv = recent_std / max(recent_mean, 1e-9)
    mean_delta = recent_mean - baseline_mean
    mean_delta_ratio = mean_delta / baseline_mean
    spread_ratio = recent_cv / baseline_cv_reference
    mean_threshold = max(
        mean_shift_absolute_threshold,
        baseline_mean * mean_shift_relative_threshold,
    )
    mean_shift_detected = abs(mean_delta) >= mean_threshold
    spread_increase_detected = spread_ratio >= spread_increase_ratio_threshold
    spread_decrease_detected = spread_ratio <= spread_decrease_ratio_threshold

    if mean_shift_detected and spread_increase_detected:
        uncertainty_type = DemandUncertaintyType.MIXED
    elif mean_shift_detected:
        uncertainty_type = DemandUncertaintyType.MEAN_SHIFT
    elif spread_increase_detected:
        uncertainty_type = DemandUncertaintyType.SPREAD_INCREASE
    elif spread_decrease_detected:
        uncertainty_type = DemandUncertaintyType.SPREAD_DECREASE
    else:
        uncertainty_type = DemandUncertaintyType.STABLE

    if uncertainty_type is DemandUncertaintyType.MEAN_SHIFT:
        recommended_demand_outlook = max(
            diagnosis.latest_demand,
            recent_mean,
            baseline_mean + mean_delta,
            1.0,
        )
        safety_buffer_scale = 1.02
        update_types = (UpdateRequestType.SWITCH_DEMAND_REGIME,)
        rationale = "Recent mean is materially displaced from the baseline."
    elif uncertainty_type is DemandUncertaintyType.MIXED:
        recommended_demand_outlook = max(recent_mean, baseline_mean + mean_delta, 1.0)
        safety_buffer_scale = _spread_safety_buffer_scale(spread_ratio, base=1.06)
        update_types = (
            UpdateRequestType.SWITCH_DEMAND_REGIME,
            UpdateRequestType.WIDEN_UNCERTAINTY,
        )
        rationale = "Recent evidence combines a mean movement with wider dispersion."
    elif uncertainty_type is DemandUncertaintyType.SPREAD_INCREASE:
        recommended_demand_outlook = max(recent_mean, baseline_mean, 1.0)
        safety_buffer_scale = _spread_safety_buffer_scale(spread_ratio, base=1.04)
        update_types = (UpdateRequestType.WIDEN_UNCERTAINTY,)
        rationale = (
            "Recent dispersion increased while the mean remains close to baseline; "
            "protect service through the buffer rather than a demand-mean jump."
        )
    elif uncertainty_type is DemandUncertaintyType.SPREAD_DECREASE:
        recommended_demand_outlook = max(recent_mean, min(baseline_mean, recent_mean), 1.0)
        safety_buffer_scale = 0.98
        update_types = (UpdateRequestType.REWEIGHT_SCENARIOS,)
        rationale = "Recent dispersion is lower than the reference spread."
    else:
        recommended_demand_outlook = max(recent_mean, baseline_mean, 1.0)
        safety_buffer_scale = 1.0
        update_types = (UpdateRequestType.KEEP_CURRENT,)
        rationale = "Recent mean and spread are close to the reference operating level."

    signal_strength = max(
        abs(mean_delta_ratio) / max(mean_shift_relative_threshold, 1e-9),
        abs(spread_ratio - 1.0),
    )
    confidence = min(0.96, 0.72 + 0.10 * signal_strength)
    return DemandUncertaintyDecompositionRecord(
        uncertainty_type=uncertainty_type,
        baseline_mean=baseline_mean,
        recent_mean=recent_mean,
        baseline_cv_reference=baseline_cv_reference,
        recent_std=recent_std,
        recent_cv=recent_cv,
        mean_delta=mean_delta,
        mean_delta_ratio=mean_delta_ratio,
        spread_ratio=spread_ratio,
        recommended_demand_outlook=recommended_demand_outlook,
        recommended_safety_buffer_scale=safety_buffer_scale,
        recommended_update_types=update_types,
        confidence=confidence,
        rationale=rationale,
    )


def _spread_safety_buffer_scale(spread_ratio: float, *, base: float) -> float:
    spread_excess = max(0.0, spread_ratio - 1.0)
    return min(1.18, max(0.95, base + 0.055 * spread_excess))


def _population_std(values: tuple[float, ...]) -> float:
    if len(values) <= 1:
        return 0.0
    center = mean(values)
    return math.sqrt(sum((float(value) - center) ** 2 for value in values) / len(values))


def _build_regime_belief(
    diagnosis: RegimeDiagnosisRecord,
    decomposition: DemandUncertaintyDecompositionRecord | None = None,
) -> RegimeBeliefRecord:
    """Build a small, normalized hidden-regime belief from the diagnosis."""

    if (
        decomposition is not None
        and decomposition.uncertainty_type is DemandUncertaintyType.SPREAD_INCREASE
    ):
        raw_entries = (
            (
                RegimeLabel.NORMAL,
                0.44,
                1.00,
                1.00,
                "Mean remains close to baseline despite wider demand spread.",
            ),
            (
                RegimeLabel.DEMAND_REGIME_SHIFT,
                0.26,
                1.08,
                1.00,
                "Retain a bounded mean-shift branch for high-demand tails.",
            ),
            (
                RegimeLabel.JOINT_DISRUPTION,
                0.18,
                1.05,
                1.12,
                "Wider demand spread can coincide with lead-time risk.",
            ),
            (
                RegimeLabel.RECOVERY,
                0.12,
                0.92,
                1.00,
                "Allow mean-preserving reversion after high tail realizations.",
            ),
        )
        tail_weight = 0.20
        service_weight = 0.34
        overreaction_weight = 0.24
        rationale = (
            "Spread-aware belief emphasizes service tails while penalizing "
            "demand-mean overreaction."
        )
    elif (
        decomposition is not None
        and decomposition.uncertainty_type is DemandUncertaintyType.MIXED
    ):
        raw_entries = (
            (
                RegimeLabel.DEMAND_REGIME_SHIFT,
                0.38,
                1.12,
                1.00,
                "Recent mean movement is present but should remain bounded.",
            ),
            (
                RegimeLabel.JOINT_DISRUPTION,
                0.24,
                1.08,
                1.12,
                "Wide demand spread can coincide with lead-time risk.",
            ),
            (
                RegimeLabel.NORMAL,
                0.24,
                1.00,
                1.00,
                "Part of the signal may be high-tail variation rather than a new mean.",
            ),
            (
                RegimeLabel.RECOVERY,
                0.14,
                0.94,
                1.00,
                "Retain a mean-reversion branch after tail-heavy observations.",
            ),
        )
        tail_weight = 0.22
        service_weight = 0.32
        overreaction_weight = 0.18
        rationale = (
            "Mixed mean-and-spread belief protects service while limiting "
            "conversion of a tail realization into a durable demand mean."
        )
    elif (
        decomposition is not None
        and decomposition.uncertainty_type is DemandUncertaintyType.SPREAD_DECREASE
    ):
        raw_entries = (
            (
                RegimeLabel.RECOVERY,
                0.46,
                0.94,
                1.00,
                "Low dispersion supports calmer demand uncertainty.",
            ),
            (
                RegimeLabel.NORMAL,
                0.36,
                1.00,
                1.00,
                "Normal continuation remains plausible.",
            ),
            (
                RegimeLabel.DEMAND_REGIME_SHIFT,
                0.12,
                1.10,
                1.00,
                "Retain a small delayed-shift branch.",
            ),
            (
                RegimeLabel.SUPPLY_DISRUPTION,
                0.06,
                1.00,
                1.12,
                "Retain low lead-time-risk branch.",
            ),
        )
        tail_weight = 0.08
        service_weight = 0.16
        overreaction_weight = 0.28
        rationale = "Calmer spread-aware belief penalizes unnecessary protection."
    elif (
        decomposition is not None
        and decomposition.uncertainty_type is DemandUncertaintyType.STABLE
    ):
        raw_entries = (
            (
                RegimeLabel.NORMAL,
                0.58,
                1.00,
                1.00,
                "Mean and spread evidence remain close to the operating reference.",
            ),
            (
                RegimeLabel.RECOVERY,
                0.18,
                0.94,
                1.00,
                "Allow a calmer continuation after isolated high observations.",
            ),
            (
                RegimeLabel.DEMAND_REGIME_SHIFT,
                0.14,
                1.10,
                1.00,
                "Retain a bounded delayed-shift branch.",
            ),
            (
                RegimeLabel.SUPPLY_DISRUPTION,
                0.10,
                1.00,
                1.12,
                "Retain a modest lead-time-risk branch.",
            ),
        )
        tail_weight = 0.08
        service_weight = 0.18
        overreaction_weight = 0.30
        rationale = (
            "Stable decomposition dampens latest-tail overreaction while "
            "retaining a small delayed-stress belief."
        )
    elif diagnosis.case_family == "stable_or_low_risk":
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
    decomposition: DemandUncertaintyDecompositionRecord | None,
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
    candidates.extend(_spread_aware_candidates(diagnosis, decomposition))
    candidates.extend(_agentic_regime_candidates(diagnosis, decomposition))
    deduped = _dedupe_candidates(tuple(candidates))
    generator_notes = [
        "rolling_horizon_incumbent_included",
        f"case_family:{diagnosis.case_family}",
        "agentic_candidates_are_regime_conditioned_bounded_scenario_inputs",
    ]
    if decomposition is not None:
        generator_notes.append(f"demand_uncertainty_type:{decomposition.uncertainty_type.value}")
    return ScenarioCandidateSet(
        candidates=deduped,
        incumbent_candidate_id="rolling_horizon_incumbent",
        generator_notes=tuple(generator_notes),
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
    decomposition: DemandUncertaintyDecompositionRecord | None = None,
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
    spread_only = (
        decomposition is not None
        and decomposition.uncertainty_type is DemandUncertaintyType.SPREAD_INCREASE
    )
    if diagnosis.case_family in {
        "initial_demand_shift",
        "sustained_demand_shift",
        "joint_demand_leadtime_stress",
    } and not spread_only:
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


def _spread_aware_candidates(
    diagnosis: RegimeDiagnosisRecord,
    decomposition: DemandUncertaintyDecompositionRecord | None,
) -> tuple[ScenarioCandidateRecord, ...]:
    if decomposition is None:
        return ()
    if decomposition.uncertainty_type is DemandUncertaintyType.STABLE:
        return (
            _bounded_candidate(
                candidate_id="agentic_stable_reference_guard",
                diagnosis=diagnosis,
                demand_outlook=decomposition.recommended_demand_outlook,
                leadtime_outlook=max(
                    diagnosis.latest_leadtime,
                    mean(diagnosis.leadtime_window),
                ),
                safety_buffer_scale=1.0,
                update_types=(UpdateRequestType.REWEIGHT_SCENARIOS,),
                rationale=(
                    "Stable reference candidate follows the decomposed operating "
                    "level instead of treating an isolated tail as a regime shift."
                ),
            ),
        )
    if decomposition.uncertainty_type is DemandUncertaintyType.SPREAD_INCREASE:
        return (
            _bounded_candidate(
                candidate_id="agentic_mean_preserving_spread_guard",
                diagnosis=diagnosis,
                demand_outlook=decomposition.recommended_demand_outlook,
                leadtime_outlook=diagnosis.latest_leadtime,
                safety_buffer_scale=decomposition.recommended_safety_buffer_scale,
                update_types=decomposition.recommended_update_types,
                rationale=(
                    "Mean-preserving spread candidate widens uncertainty protection "
                    "without converting high-tail demand into a demand-mean shift."
                ),
            ),
            _bounded_candidate(
                candidate_id="agentic_spread_service_floor_guard",
                diagnosis=diagnosis,
                demand_outlook=_service_floor_demand_outlook(
                    decomposition,
                    max_multiplier=1.08,
                ),
                leadtime_outlook=max(
                    diagnosis.latest_leadtime,
                    _empirical_quantile(diagnosis.leadtime_window, 0.75),
                ),
                safety_buffer_scale=_service_floor_safety_buffer_scale(
                    decomposition,
                    additive=0.065,
                ),
                update_types=decomposition.recommended_update_types,
                rationale=(
                    "Spread service-floor candidate preserves the demand mean "
                    "while adding bounded buffer and lead-time cover for tail risk."
                ),
            ),
            _bounded_candidate(
                candidate_id="agentic_spread_tail_cover_guard",
                diagnosis=diagnosis,
                demand_outlook=_tail_cover_demand_outlook(
                    diagnosis,
                    decomposition,
                    cap_multiplier=1.45,
                ),
                leadtime_outlook=max(
                    diagnosis.latest_leadtime,
                    _empirical_quantile(diagnosis.leadtime_window, 0.75),
                ),
                safety_buffer_scale=_service_floor_safety_buffer_scale(
                    decomposition,
                    additive=0.045,
                ),
                update_types=decomposition.recommended_update_types,
                rationale=(
                    "Spread tail-cover candidate converts high dispersion into a "
                    "bounded tail-cover demand equivalent for the downstream rule."
                ),
            ),
        )
    if decomposition.uncertainty_type is DemandUncertaintyType.MIXED:
        return (
            _bounded_candidate(
                candidate_id="agentic_mixed_mean_spread_guard",
                diagnosis=diagnosis,
                demand_outlook=decomposition.recommended_demand_outlook,
                leadtime_outlook=diagnosis.latest_leadtime,
                safety_buffer_scale=decomposition.recommended_safety_buffer_scale,
                update_types=decomposition.recommended_update_types,
                rationale=(
                    "Mixed mean-and-spread candidate combines bounded mean response "
                    "with explicit uncertainty widening."
                ),
            ),
            _bounded_candidate(
                candidate_id="agentic_mixed_service_floor_guard",
                diagnosis=diagnosis,
                demand_outlook=_service_floor_demand_outlook(
                    decomposition,
                    max_multiplier=1.12,
                ),
                leadtime_outlook=max(
                    diagnosis.latest_leadtime,
                    _empirical_quantile(diagnosis.leadtime_window, 0.75),
                ),
                safety_buffer_scale=_service_floor_safety_buffer_scale(
                    decomposition,
                    additive=0.055,
                ),
                update_types=decomposition.recommended_update_types,
                rationale=(
                    "Mixed service-floor candidate balances bounded mean response "
                    "with additional spread protection under service-risk pressure."
                ),
            ),
            _bounded_candidate(
                candidate_id="agentic_mixed_tail_cover_guard",
                diagnosis=diagnosis,
                demand_outlook=_tail_cover_demand_outlook(
                    diagnosis,
                    decomposition,
                    cap_multiplier=2.15,
                ),
                leadtime_outlook=max(
                    diagnosis.latest_leadtime,
                    _empirical_quantile(diagnosis.leadtime_window, 0.75),
                ),
                safety_buffer_scale=_service_floor_safety_buffer_scale(
                    decomposition,
                    additive=0.040,
                ),
                update_types=decomposition.recommended_update_types,
                rationale=(
                    "Mixed tail-cover candidate protects against severe high-tail "
                    "dispersion without treating the latest realization as the mean."
                ),
            ),
        )
    if decomposition.uncertainty_type is DemandUncertaintyType.SPREAD_DECREASE:
        return (
            _bounded_candidate(
                candidate_id="agentic_spread_relaxation",
                diagnosis=diagnosis,
                demand_outlook=decomposition.recommended_demand_outlook,
                leadtime_outlook=diagnosis.latest_leadtime,
                safety_buffer_scale=decomposition.recommended_safety_buffer_scale,
                update_types=decomposition.recommended_update_types,
                rationale=(
                    "Lower-spread candidate relaxes uncertainty protection while "
                    "keeping the demand outlook anchored."
                ),
            ),
        )
    return ()


def _service_floor_demand_outlook(
    decomposition: DemandUncertaintyDecompositionRecord,
    *,
    max_multiplier: float,
) -> float:
    severity_multiplier = 1.0 + min(
        max_multiplier - 1.0,
        0.025 * max(0.0, decomposition.spread_ratio - 1.0),
    )
    return decomposition.recommended_demand_outlook * severity_multiplier


def _service_floor_safety_buffer_scale(
    decomposition: DemandUncertaintyDecompositionRecord,
    *,
    additive: float,
) -> float:
    severity_additive = additive + min(
        0.055,
        0.025 * max(0.0, decomposition.spread_ratio - 1.0),
    )
    return min(
        1.25,
        max(
            decomposition.recommended_safety_buffer_scale,
            decomposition.recommended_safety_buffer_scale + severity_additive,
        ),
    )


def _tail_cover_demand_outlook(
    diagnosis: RegimeDiagnosisRecord,
    decomposition: DemandUncertaintyDecompositionRecord,
    *,
    cap_multiplier: float,
) -> float:
    tail_cover = _upper_tail_mean(diagnosis.demand_window, 0.25)
    cap = decomposition.recommended_demand_outlook * cap_multiplier
    return max(
        decomposition.recommended_demand_outlook,
        min(tail_cover, cap),
    )


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
        None,
    )


def _risk_sensitive_candidate_score(
    *,
    candidate: ScenarioCandidateRecord,
    diagnosis: RegimeDiagnosisRecord,
    decomposition: DemandUncertaintyDecompositionRecord | None,
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
    overreaction_penalty = _overreaction_penalty(
        candidate,
        diagnosis,
        decomposition,
        belief,
    )
    objective = (
        mean_cost
        + belief.tail_risk_weight * tail_cost
        + service_penalty
        + overreaction_penalty
        + _uncertainty_quality_penalty(candidate, diagnosis, decomposition)
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
    decomposition: DemandUncertaintyDecompositionRecord | None,
    belief: RegimeBeliefRecord,
) -> float:
    if belief.overreaction_weight <= 0.0:
        return 0.0
    protected_demand = candidate.demand_outlook * candidate.safety_buffer_scale
    if (
        decomposition is not None
        and decomposition.uncertainty_type
        in {
            DemandUncertaintyType.STABLE,
            DemandUncertaintyType.SPREAD_INCREASE,
            DemandUncertaintyType.SPREAD_DECREASE,
            DemandUncertaintyType.MIXED,
        }
    ):
        demand_anchor = max(decomposition.recommended_demand_outlook, 1.0)
    else:
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
    decomposition: DemandUncertaintyDecompositionRecord | None,
    belief: RegimeBeliefRecord,
    evaluation: ScenarioPlannerEvaluationDiagnostics,
    time_index: int,
    immediate_shift_margin: float,
    clean_recovery_margin: float,
) -> tuple[str, str]:
    selected_id = evaluation.selected_candidate_id
    selected_score = _score_by_id(evaluation, selected_id)

    if (
        decomposition is not None
        and decomposition.uncertainty_type is DemandUncertaintyType.SPREAD_INCREASE
    ):
        spread_candidate = _best_spread_preserving_score(
            evaluation,
            decomposition=decomposition,
        )
        demand_cap = decomposition.recommended_demand_outlook * 1.15
        selected_overstates_mean = selected_score.demand_outlook > demand_cap
        if (
            spread_candidate is not None
            and selected_overstates_mean
            and spread_candidate.expected_cost <= selected_score.expected_cost + 30.0
        ):
            return (
                spread_candidate.candidate_id,
                "spread_increase_prefers_mean_preserving_candidate_when_cost_close",
            )
        tail_cover_score = _optional_score_by_id(
            evaluation,
            "agentic_spread_tail_cover_guard",
        )
        if (
            tail_cover_score is not None
            and time_index >= 2
            and selected_score.service_risk_penalty >= 12.0
            and tail_cover_score.expected_cost <= selected_score.expected_cost + 70.0
        ):
            return (
                "agentic_spread_tail_cover_guard",
                "spread_increase_service_pressure_prefers_tail_cover_when_close",
            )

    if (
        decomposition is not None
        and decomposition.uncertainty_type is DemandUncertaintyType.MIXED
    ):
        tail_cover_score = _optional_score_by_id(
            evaluation,
            "agentic_mixed_tail_cover_guard",
        )
        if (
            tail_cover_score is not None
            and time_index >= 2
            and selected_score.service_risk_penalty >= 35.0
            and tail_cover_score.expected_cost <= selected_score.expected_cost + 90.0
        ):
            return (
                "agentic_mixed_tail_cover_guard",
                "mixed_service_pressure_prefers_tail_cover_when_close",
            )

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
    decomposition: DemandUncertaintyDecompositionRecord | None,
) -> float:
    """Penalize scenario inputs that contradict high-confidence regime evidence."""

    penalty = 0.0
    if (
        decomposition is not None
        and decomposition.uncertainty_type is DemandUncertaintyType.STABLE
    ):
        demand_anchor = max(decomposition.recommended_demand_outlook, 1.0)
        protected_demand = candidate.demand_outlook * candidate.safety_buffer_scale
        demand_excess = max(0.0, protected_demand / demand_anchor - 1.30)
        penalty += 45.0 * demand_excess * demand_excess
        if UpdateRequestType.SWITCH_DEMAND_REGIME in candidate.applied_update_types:
            penalty += 5.0
    elif (
        decomposition is not None
        and decomposition.uncertainty_type is DemandUncertaintyType.SPREAD_INCREASE
    ):
        demand_anchor = max(decomposition.recommended_demand_outlook, 1.0)
        demand_excess = max(0.0, candidate.demand_outlook / demand_anchor - 1.12)
        penalty += 80.0 * demand_excess * demand_excess
        buffer_shortfall = max(
            0.0,
            decomposition.recommended_safety_buffer_scale
            - candidate.safety_buffer_scale,
        )
        penalty += 35.0 * buffer_shortfall * buffer_shortfall
        if UpdateRequestType.SWITCH_DEMAND_REGIME in candidate.applied_update_types:
            penalty += 4.0
    elif (
        decomposition is not None
        and decomposition.uncertainty_type is DemandUncertaintyType.MIXED
    ):
        demand_anchor = max(decomposition.recommended_demand_outlook, 1.0)
        target_protection = (
            demand_anchor * decomposition.recommended_safety_buffer_scale
        )
        protected_demand = candidate.demand_outlook * candidate.safety_buffer_scale
        protection_shortfall = max(0.0, target_protection - protected_demand)
        penalty += 16.0 * protection_shortfall * protection_shortfall
        demand_excess = max(0.0, protected_demand / target_protection - 1.85)
        penalty += 120.0 * demand_excess * demand_excess
        if (
            UpdateRequestType.WIDEN_UNCERTAINTY
            not in candidate.applied_update_types
            and candidate.safety_buffer_scale
            < decomposition.recommended_safety_buffer_scale
        ):
            buffer_shortfall = (
                decomposition.recommended_safety_buffer_scale
                - candidate.safety_buffer_scale
            )
            penalty += 20.0 * buffer_shortfall * buffer_shortfall
    elif (
        decomposition is not None
        and decomposition.uncertainty_type is DemandUncertaintyType.SPREAD_DECREASE
    ):
        demand_anchor = max(decomposition.recommended_demand_outlook, 1.0)
        protected_demand = candidate.demand_outlook * candidate.safety_buffer_scale
        demand_excess = max(0.0, protected_demand / demand_anchor - 1.18)
        penalty += 55.0 * demand_excess * demand_excess
        if UpdateRequestType.SWITCH_DEMAND_REGIME in candidate.applied_update_types:
            penalty += 6.0
    elif diagnosis.case_family in {
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
    decomposition: DemandUncertaintyDecompositionRecord | None,
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
                spread_only = (
                    decomposition is not None
                    and decomposition.uncertainty_type
                    is DemandUncertaintyType.SPREAD_INCREASE
                )
                if spread_only:
                    if entry.regime_label is RegimeLabel.RECOVERY:
                        demand_value = min(
                            demand_value,
                            decomposition.recommended_demand_outlook,
                        )
                    elif entry.regime_label in {
                        RegimeLabel.DEMAND_REGIME_SHIFT,
                        RegimeLabel.JOINT_DISRUPTION,
                    }:
                        demand_value = max(
                            demand_value,
                            decomposition.recommended_demand_outlook,
                        )
                    else:
                        demand_value *= entry.demand_multiplier
                elif entry.regime_label in {
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


def _best_spread_preserving_score(
    evaluation: ScenarioPlannerEvaluationDiagnostics,
    *,
    decomposition: DemandUncertaintyDecompositionRecord,
) -> PlannerCandidateScoreRecord | None:
    demand_cap = decomposition.recommended_demand_outlook * 1.15
    eligible = tuple(
        score
        for score in evaluation.candidate_scores
        if score.demand_outlook <= demand_cap and score.safety_buffer_scale >= 1.0
    )
    if not eligible:
        return None
    return min(eligible, key=lambda score: score.expected_cost)


def _latest_diagnosis(invocation: ToolInvocation) -> RegimeDiagnosisRecord:
    for result in reversed(invocation.prior_results):
        value = result.structured_output.get("regime_diagnosis")
        if isinstance(value, RegimeDiagnosisRecord):
            return value
    raise ValueError("Scenario candidate generator requires a prior regime diagnosis.")


def _latest_demand_uncertainty_decomposition(
    invocation: ToolInvocation,
) -> DemandUncertaintyDecompositionRecord | None:
    for result in reversed(invocation.prior_results):
        value = result.structured_output.get("demand_uncertainty_decomposition")
        if isinstance(value, DemandUncertaintyDecompositionRecord):
            return value
    return None


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
    "DemandUncertaintyDecompositionRecord",
    "DemandUncertaintyDecompositionTool",
    "DemandUncertaintyType",
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
