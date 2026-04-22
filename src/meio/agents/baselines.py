"""Non-agentic deterministic comparison paths for controlled rollouts."""

from __future__ import annotations

from dataclasses import dataclass

from meio.contracts import AgentSignal, OperationalSubgoal, RegimeLabel, UpdateRequestType
from meio.scenarios.contracts import (
    ScenarioAdjustmentSummary,
    ScenarioSummary,
    ScenarioUpdateResult,
)
from meio.simulation.evidence import RuntimeEvidence
from meio.simulation.state import Observation, SimulationState

_BASELINE_BEHAVIOR = {
    RegimeLabel.NORMAL: (
        1.0,
        1.0,
        1.0,
        False,
        OperationalSubgoal.NO_ACTION,
        (UpdateRequestType.KEEP_CURRENT,),
    ),
    RegimeLabel.DEMAND_REGIME_SHIFT: (
        1.05,
        1.0,
        1.0,
        True,
        OperationalSubgoal.REQUEST_REPLAN,
        (UpdateRequestType.REWEIGHT_SCENARIOS,),
    ),
    RegimeLabel.SUPPLY_DISRUPTION: (
        1.0,
        1.15,
        1.0,
        True,
        OperationalSubgoal.REQUEST_REPLAN,
        (UpdateRequestType.SWITCH_LEADTIME_REGIME,),
    ),
    RegimeLabel.JOINT_DISRUPTION: (
        1.05,
        1.15,
        1.0,
        True,
        OperationalSubgoal.REQUEST_REPLAN,
        (UpdateRequestType.REWEIGHT_SCENARIOS,),
    ),
    RegimeLabel.RECOVERY: (
        1.0,
        1.0,
        1.0,
        False,
        OperationalSubgoal.NO_ACTION,
        (UpdateRequestType.KEEP_CURRENT,),
    ),
}


@dataclass(frozen=True, slots=True)
class DeterministicBaselineDecision:
    """Bounded deterministic comparison output with no orchestration semantics."""

    signal: AgentSignal
    scenario_update_result: ScenarioUpdateResult

    def __post_init__(self) -> None:
        if not isinstance(self.signal, AgentSignal):
            raise TypeError("signal must be an AgentSignal.")
        if not isinstance(self.scenario_update_result, ScenarioUpdateResult):
            raise TypeError("scenario_update_result must be a ScenarioUpdateResult.")


@dataclass(frozen=True, slots=True)
class DeterministicBaselinePolicy:
    """Simple non-agentic path for controlled comparison against orchestration."""

    provenance: str = "deterministic_baseline_policy"

    def decide(
        self,
        system_state: SimulationState,
        observation: Observation,
        evidence: RuntimeEvidence,
    ) -> DeterministicBaselineDecision:
        if not isinstance(system_state, SimulationState):
            raise TypeError("system_state must be a SimulationState.")
        if not isinstance(observation, Observation):
            raise TypeError("observation must be an Observation.")
        if not isinstance(evidence, RuntimeEvidence):
            raise TypeError("evidence must be a RuntimeEvidence.")
        regime_label = observation.regime_label or system_state.regime_label
        (
            demand_scale,
            leadtime_scale,
            buffer_scale,
            request_replan,
            selected_subgoal,
            applied_update_types,
        ) = _BASELINE_BEHAVIOR[regime_label]
        if request_replan:
            rationale = (
                "Deterministic baseline requested a bounded scenario update under the "
                "scheduled regime."
            )
        else:
            rationale = (
                "Deterministic baseline kept the current scenario under the scheduled regime."
            )
        scenario_update_result = ScenarioUpdateResult(
            scenarios=(
                ScenarioSummary(
                    scenario_id=f"baseline_{regime_label.value}",
                    regime_label=regime_label,
                    weight=1.0,
                    demand_scale=demand_scale,
                    leadtime_scale=leadtime_scale,
                ),
            ),
            applied_update_types=applied_update_types,
            adjustment=ScenarioAdjustmentSummary(
                demand_outlook=observation.demand_realization[-1] * demand_scale,
                leadtime_outlook=observation.leadtime_realization[-1] * leadtime_scale,
                safety_buffer_scale=buffer_scale,
            ),
            request_replan=request_replan,
            provenance=self.provenance,
        )
        signal = AgentSignal(
            selected_subgoal=selected_subgoal,
            selected_tool_id=None,
            tool_sequence=(),
            abstained=False,
            no_action=selected_subgoal is OperationalSubgoal.NO_ACTION,
            request_replan=request_replan,
            rationale=rationale,
        )
        return DeterministicBaselineDecision(
            signal=signal,
            scenario_update_result=scenario_update_result,
        )


__all__ = ["DeterministicBaselineDecision", "DeterministicBaselinePolicy"]
