"""Non-agentic uncertainty-handling comparison policies.

These policies intentionally do not use the LLM-backed orchestration runtime.
They only construct bounded scenario inputs for the existing trusted
downstream replenishment rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import random
from statistics import mean
from typing import Protocol

from meio.config.schemas import (
    RobustPolicyConfig,
    ScenarioRollingHorizonPolicyConfig,
)
from meio.contracts import AgentSignal, OperationalSubgoal, RegimeLabel, UpdateRequestType
from meio.evaluation.rollout_metrics import compute_period_total_cost
from meio.optimization.adapters import TrustedOptimizerAdapter, build_optimization_request
from meio.scenarios.contracts import (
    ScenarioAdjustmentSummary,
    ScenarioSummary,
    ScenarioUpdateResult,
)
from meio.simulation.evidence import DemandEvidence, LeadTimeEvidence, RuntimeEvidence
from meio.simulation.serial_benchmark import SerialBenchmarkCase, advance_serial_state
from meio.simulation.state import Observation, PeriodTraceRecord, SimulationState


PolicyDiagnostics = dict[str, object]


@dataclass(frozen=True, slots=True)
class UncertaintyPolicyDecision:
    """Bounded policy output with no raw replenishment-order content."""

    signal: AgentSignal
    scenario_update_result: ScenarioUpdateResult
    diagnostics: PolicyDiagnostics

    def __post_init__(self) -> None:
        if not isinstance(self.signal, AgentSignal):
            raise TypeError("signal must be an AgentSignal.")
        if not isinstance(self.scenario_update_result, ScenarioUpdateResult):
            raise TypeError("scenario_update_result must be a ScenarioUpdateResult.")


class UncertaintyPolicy(Protocol):
    """Protocol for non-agentic policies that supply scenario inputs only."""

    mode_name: str

    def decide(
        self,
        system_state: SimulationState,
        observation: Observation,
        evidence: RuntimeEvidence,
        benchmark_case: SerialBenchmarkCase,
    ) -> UncertaintyPolicyDecision:
        """Return bounded scenario inputs for the trusted downstream rule."""


@dataclass(frozen=True, slots=True)
class RobustUncertaintyPolicy:
    """Empirical uncertainty-set baseline using conservative evidence quantiles."""

    config: RobustPolicyConfig = RobustPolicyConfig()
    mode_name: str = "robust_policy"

    def decide(
        self,
        system_state: SimulationState,
        observation: Observation,
        evidence: RuntimeEvidence,
        benchmark_case: SerialBenchmarkCase,
    ) -> UncertaintyPolicyDecision:
        _validate_inputs(system_state, observation, evidence, benchmark_case)
        regime_label = _resolve_regime(system_state, observation)
        demand_window = _bounded_window(
            _evidence_series(evidence.demand, observation.demand_realization[-1]),
            self.config.window_length,
        )
        leadtime_window = _bounded_window(
            _evidence_series(evidence.leadtime, observation.leadtime_realization[-1]),
            self.config.window_length,
        )
        demand_outlook = max(
            observation.demand_realization[-1],
            _empirical_quantile(demand_window, self.config.quantile),
        )
        leadtime_outlook = max(
            1.0,
            observation.leadtime_realization[-1],
            _empirical_quantile(leadtime_window, self.config.quantile),
        )
        scenario_update_result = _scenario_update(
            mode_name=self.mode_name,
            regime_label=regime_label,
            demand_outlook=demand_outlook,
            leadtime_outlook=leadtime_outlook,
            safety_buffer_scale=self.config.safety_buffer_scale,
            scenario_id=f"robust_q{int(round(self.config.quantile * 100)):02d}",
            applied_update_types=(UpdateRequestType.WIDEN_UNCERTAINTY,),
            provenance="robust_uncertainty_protected_policy",
        )
        diagnostics: PolicyDiagnostics = {
            "policy_name": self.mode_name,
            "window_length": self.config.window_length,
            "quantile": self.config.quantile,
            "safety_buffer_scale": self.config.safety_buffer_scale,
            "demand_window": list(demand_window),
            "leadtime_window": list(leadtime_window),
            "demand_outlook": demand_outlook,
            "leadtime_outlook": leadtime_outlook,
        }
        return UncertaintyPolicyDecision(
            signal=_policy_signal(
                selected_subgoal=OperationalSubgoal.UPDATE_UNCERTAINTY,
                rationale=(
                    "Robust policy supplied conservative empirical uncertainty "
                    "inputs to the trusted downstream rule."
                ),
            ),
            scenario_update_result=scenario_update_result,
            diagnostics=diagnostics,
        )


@dataclass(frozen=True, slots=True)
class ScenarioRollingHorizonPolicy:
    """Scenario-based rolling-horizon baseline over bounded candidate inputs."""

    config: ScenarioRollingHorizonPolicyConfig = ScenarioRollingHorizonPolicyConfig()
    mode_name: str = "scenario_rolling_horizon_policy"

    def decide(
        self,
        system_state: SimulationState,
        observation: Observation,
        evidence: RuntimeEvidence,
        benchmark_case: SerialBenchmarkCase,
    ) -> UncertaintyPolicyDecision:
        _validate_inputs(system_state, observation, evidence, benchmark_case)
        regime_label = _resolve_regime(system_state, observation)
        demand_window = _bounded_window(
            _evidence_series(evidence.demand, observation.demand_realization[-1]),
            max(1, len(evidence.demand.history)),
        )
        leadtime_window = _bounded_window(
            _evidence_series(evidence.leadtime, observation.leadtime_realization[-1]),
            max(1, len(evidence.leadtime.history)),
        )
        scenario_paths = _scenario_paths(
            demand_window=demand_window,
            leadtime_window=leadtime_window,
            config=self.config,
            time_index=system_state.time_index,
        )
        candidates = tuple(
            _Candidate(
                demand_multiplier=demand_multiplier,
                leadtime_multiplier=leadtime_multiplier,
                safety_buffer_scale=safety_buffer_scale,
                demand_outlook=max(0.0, mean(demand_window) * demand_multiplier),
                leadtime_outlook=max(1.0, mean(leadtime_window) * leadtime_multiplier),
            )
            for demand_multiplier, leadtime_multiplier, safety_buffer_scale in product(
                self.config.demand_multipliers,
                self.config.leadtime_multipliers,
                self.config.safety_buffer_scales,
            )
        )
        ranked = tuple(
            (
                _expected_candidate_cost(
                    candidate=candidate,
                    system_state=system_state,
                    regime_label=regime_label,
                    benchmark_case=benchmark_case,
                    scenario_paths=scenario_paths,
                ),
                candidate,
            )
            for candidate in candidates
        )
        expected_cost, selected = min(ranked, key=lambda item: item[0])
        scenario_update_result = _scenario_update(
            mode_name=self.mode_name,
            regime_label=regime_label,
            demand_outlook=selected.demand_outlook,
            leadtime_outlook=selected.leadtime_outlook,
            safety_buffer_scale=selected.safety_buffer_scale,
            scenario_id="rolling_horizon_selected_candidate",
            applied_update_types=(UpdateRequestType.REWEIGHT_SCENARIOS,),
            provenance="scenario_rolling_horizon_policy",
        )
        diagnostics: PolicyDiagnostics = {
            "policy_name": self.mode_name,
            "horizon_length": self.config.horizon_length,
            "scenario_count": self.config.scenario_count,
            "random_seed": self.config.random_seed,
            "candidate_grid_size": len(candidates),
            "selected_expected_scenario_cost": expected_cost,
            "selected_demand_multiplier": selected.demand_multiplier,
            "selected_leadtime_multiplier": selected.leadtime_multiplier,
            "selected_safety_buffer_scale": selected.safety_buffer_scale,
            "demand_outlook": selected.demand_outlook,
            "leadtime_outlook": selected.leadtime_outlook,
        }
        return UncertaintyPolicyDecision(
            signal=_policy_signal(
                selected_subgoal=OperationalSubgoal.REQUEST_REPLAN,
                rationale=(
                    "Scenario rolling-horizon policy selected the lowest expected "
                    "cost candidate over bounded empirical scenarios."
                ),
            ),
            scenario_update_result=scenario_update_result,
            diagnostics=diagnostics,
        )


@dataclass(frozen=True, slots=True)
class _Candidate:
    demand_multiplier: float
    leadtime_multiplier: float
    safety_buffer_scale: float
    demand_outlook: float
    leadtime_outlook: float


def build_uncertainty_policy(
    mode_name: str,
    *,
    robust_config: RobustPolicyConfig,
    rolling_config: ScenarioRollingHorizonPolicyConfig,
) -> UncertaintyPolicy:
    """Build a non-agentic uncertainty baseline by mode name."""

    if mode_name == "robust_policy":
        return RobustUncertaintyPolicy(config=robust_config)
    if mode_name == "scenario_rolling_horizon_policy":
        return ScenarioRollingHorizonPolicy(config=rolling_config)
    raise ValueError(f"Unsupported uncertainty baseline mode: {mode_name!r}.")


def is_uncertainty_policy_mode(mode_name: str) -> bool:
    """Return whether the mode is handled by the non-agentic policy module."""

    return mode_name in {"robust_policy", "scenario_rolling_horizon_policy"}


def _validate_inputs(
    system_state: SimulationState,
    observation: Observation,
    evidence: RuntimeEvidence,
    benchmark_case: SerialBenchmarkCase,
) -> None:
    if not isinstance(system_state, SimulationState):
        raise TypeError("system_state must be a SimulationState.")
    if not isinstance(observation, Observation):
        raise TypeError("observation must be an Observation.")
    if not isinstance(evidence, RuntimeEvidence):
        raise TypeError("evidence must be a RuntimeEvidence.")
    if not isinstance(benchmark_case, SerialBenchmarkCase):
        raise TypeError("benchmark_case must be a SerialBenchmarkCase.")


def _resolve_regime(
    system_state: SimulationState,
    observation: Observation,
) -> RegimeLabel:
    return observation.regime_label or system_state.regime_label


def _evidence_series(evidence, latest_value: float) -> tuple[float, ...]:
    values = tuple(float(value) for value in evidence.history)
    if not values or values[-1] != float(latest_value):
        values = values + (float(latest_value),)
    return values


def _bounded_window(values: tuple[float, ...], window_length: int) -> tuple[float, ...]:
    if window_length <= 0:
        raise ValueError("window_length must be positive.")
    if not values:
        raise ValueError("values must not be empty.")
    return tuple(values[-window_length:])


def _empirical_quantile(values: tuple[float, ...], quantile: float) -> float:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be within [0.0, 1.0].")
    ordered = tuple(sorted(float(value) for value in values))
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * (len(ordered) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return ordered[lower_index] + fraction * (ordered[upper_index] - ordered[lower_index])


def _scenario_update(
    *,
    mode_name: str,
    regime_label: RegimeLabel,
    demand_outlook: float,
    leadtime_outlook: float,
    safety_buffer_scale: float,
    scenario_id: str,
    applied_update_types: tuple[UpdateRequestType, ...],
    provenance: str,
) -> ScenarioUpdateResult:
    return ScenarioUpdateResult(
        scenarios=(
            ScenarioSummary(
                scenario_id=f"{mode_name}_{scenario_id}",
                regime_label=regime_label,
                weight=1.0,
                demand_scale=1.0,
                leadtime_scale=1.0,
            ),
        ),
        applied_update_types=applied_update_types,
        adjustment=ScenarioAdjustmentSummary(
            demand_outlook=demand_outlook,
            leadtime_outlook=leadtime_outlook,
            safety_buffer_scale=safety_buffer_scale,
        ),
        request_replan=True,
        provenance=provenance,
    )


def _policy_signal(
    *,
    selected_subgoal: OperationalSubgoal,
    rationale: str,
) -> AgentSignal:
    return AgentSignal(
        selected_subgoal=selected_subgoal,
        selected_tool_id=None,
        tool_sequence=(),
        abstained=False,
        no_action=False,
        request_replan=True,
        rationale=rationale,
    )


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


def _expected_candidate_cost(
    *,
    candidate: _Candidate,
    system_state: SimulationState,
    regime_label: RegimeLabel,
    benchmark_case: SerialBenchmarkCase,
    scenario_paths: tuple[tuple[tuple[float, float], ...], ...],
) -> float:
    optimizer = TrustedOptimizerAdapter()
    total_cost = 0.0
    scenario_update_result = _scenario_update(
        mode_name="scenario_rolling_horizon_policy",
        regime_label=regime_label,
        demand_outlook=candidate.demand_outlook,
        leadtime_outlook=candidate.leadtime_outlook,
        safety_buffer_scale=candidate.safety_buffer_scale,
        scenario_id="candidate",
        applied_update_types=(UpdateRequestType.REWEIGHT_SCENARIOS,),
        provenance="scenario_rolling_horizon_candidate_evaluation",
    )
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
                notes=("scenario_rolling_horizon_candidate_evaluation",),
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
                agent_signal=_policy_signal(
                    selected_subgoal=OperationalSubgoal.REQUEST_REPLAN,
                    rationale="scenario candidate evaluation",
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
    return total_cost / float(len(scenario_paths))


__all__ = [
    "RobustUncertaintyPolicy",
    "ScenarioRollingHorizonPolicy",
    "UncertaintyPolicy",
    "UncertaintyPolicyDecision",
    "build_uncertainty_policy",
    "is_uncertainty_policy_mode",
]
