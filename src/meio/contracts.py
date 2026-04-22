"""Shared typed boundary contracts for the MEIO package.

These types intentionally describe only bounded orchestration-agent, tool, and
benchmark-facing signals. Raw replenishment orders do not belong here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from meio.simulation.evidence import RuntimeEvidence
    from meio.simulation.state import Observation, SimulationState


class RegimeLabel(StrEnum):
    """Allowed outer-loop regime labels for the first milestone."""

    NORMAL = "normal"
    DEMAND_REGIME_SHIFT = "demand_regime_shift"
    SUPPLY_DISRUPTION = "supply_disruption"
    JOINT_DISRUPTION = "joint_disruption"
    RECOVERY = "recovery"


class UpdateRequestType(StrEnum):
    """Bounded uncertainty-management update requests."""

    KEEP_CURRENT = "keep_current"
    REWEIGHT_SCENARIOS = "reweight_scenarios"
    SWITCH_DEMAND_REGIME = "switch_demand_regime"
    SWITCH_LEADTIME_REGIME = "switch_leadtime_regime"
    WIDEN_UNCERTAINTY = "widen_uncertainty"


class BackorderPolicy(StrEnum):
    """Supported service model for the first official benchmark."""

    BACKORDERS = "backorders"


class BenchmarkFamily(StrEnum):
    """Supported benchmark family identifiers."""

    SERIAL = "serial"


class ToolClass(StrEnum):
    """Supported classes of bounded tools."""

    DETERMINISTIC_STATISTICAL = "deterministic_statistical"
    LEARNED_NON_LLM = "learned_non_llm"
    LLM_BACKED = "llm_backed"


class ToolStatus(StrEnum):
    """Status values returned by bounded tool calls."""

    SUCCESS = "success"
    NO_ACTION = "no_action"
    ABSTAIN = "abstain"


class OperationalSubgoal(StrEnum):
    """Bounded operational subgoals managed by the orchestration runtime."""

    INSPECT_EVIDENCE = "inspect_evidence"
    CLASSIFY_REGIME = "classify_regime"
    QUERY_UNCERTAINTY = "query_uncertainty"
    UPDATE_UNCERTAINTY = "update_uncertainty"
    REQUEST_REPLAN = "request_replan"
    NO_ACTION = "no_action"
    ABSTAIN = "abstain"


@dataclass(frozen=True, slots=True)
class MissionSpec:
    """Human-defined mission, objective, and hard constraints for a run."""

    mission_id: str
    objective: str
    hard_constraints: tuple[str, ...] = field(default_factory=tuple)
    admissible_tool_ids: tuple[str, ...] = field(default_factory=tuple)
    max_tool_steps: int = 3

    def __post_init__(self) -> None:
        if not self.mission_id.strip():
            raise ValueError("mission_id must be a non-empty string.")
        if not self.objective.strip():
            raise ValueError("objective must be a non-empty string.")
        if self.max_tool_steps <= 0:
            raise ValueError("max_tool_steps must be positive.")
        object.__setattr__(self, "hard_constraints", tuple(self.hard_constraints))
        object.__setattr__(self, "admissible_tool_ids", tuple(self.admissible_tool_ids))
        for constraint in self.hard_constraints:
            if not constraint.strip():
                raise ValueError("hard_constraints must contain non-empty strings.")
        for tool_id in self.admissible_tool_ids:
            if not tool_id.strip():
                raise ValueError("admissible_tool_ids must contain non-empty strings.")


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Metadata for a bounded tool exposed to the orchestration runtime."""

    tool_id: str
    tool_class: ToolClass
    supported_subgoals: tuple[OperationalSubgoal, ...]
    description: str = ""
    produces_raw_orders: bool = False

    def __post_init__(self) -> None:
        if not self.tool_id.strip():
            raise ValueError("tool_id must be a non-empty string.")
        if not isinstance(self.tool_class, ToolClass):
            raise TypeError("tool_class must be a ToolClass.")
        object.__setattr__(self, "supported_subgoals", tuple(self.supported_subgoals))
        if not self.supported_subgoals:
            raise ValueError("supported_subgoals must not be empty.")
        for subgoal in self.supported_subgoals:
            if not isinstance(subgoal, OperationalSubgoal):
                raise TypeError("supported_subgoals must contain OperationalSubgoal values.")


@dataclass(frozen=True, slots=True)
class ToolInvocation:
    """A bounded request from the orchestration runtime to a tool."""

    tool_id: str
    tool_class: ToolClass
    subgoal: OperationalSubgoal
    evidence: RuntimeEvidence | None = None
    system_state: SimulationState | None = None
    observation: Observation | None = None
    agent_assessment: AgentAssessment | None = None
    prior_results: tuple["ToolResult", ...] = field(default_factory=tuple)
    step_index: int = 0

    def __post_init__(self) -> None:
        if not self.tool_id.strip():
            raise ValueError("tool_id must be a non-empty string.")
        if not isinstance(self.tool_class, ToolClass):
            raise TypeError("tool_class must be a ToolClass.")
        if not isinstance(self.subgoal, OperationalSubgoal):
            raise TypeError("subgoal must be an OperationalSubgoal.")
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative.")
        object.__setattr__(self, "prior_results", tuple(self.prior_results))
        if self.evidence is not None:
            from meio.simulation.evidence import RuntimeEvidence

            if not isinstance(self.evidence, RuntimeEvidence):
                raise TypeError("evidence must be a RuntimeEvidence when provided.")
        if self.system_state is not None:
            from meio.simulation.state import SimulationState

            if not isinstance(self.system_state, SimulationState):
                raise TypeError("system_state must be a SimulationState when provided.")
        if self.observation is not None:
            from meio.simulation.state import Observation

            if not isinstance(self.observation, Observation):
                raise TypeError("observation must be an Observation when provided.")
        if self.agent_assessment is not None and not isinstance(
            self.agent_assessment, AgentAssessment
        ):
            raise TypeError("agent_assessment must be an AgentAssessment when provided.")
        for result in self.prior_results:
            if not isinstance(result, ToolResult):
                raise TypeError("prior_results must contain ToolResult values.")


@dataclass(frozen=True, slots=True)
class ToolResult:
    """A bounded structured result returned by a tool.

    The runtime-level envelope remains generic because it spans heterogeneous
    tool families, but domain payloads should prefer typed contract objects
    rather than nested ad hoc dictionaries.
    """

    tool_id: str
    tool_class: ToolClass
    subgoal: OperationalSubgoal
    status: ToolStatus = ToolStatus.SUCCESS
    structured_output: dict[str, object] = field(default_factory=dict)
    confidence: float | None = None
    provenance: str = ""
    next_tool_id: str | None = None
    next_subgoal: OperationalSubgoal | None = None
    request_replan: bool = False
    emits_raw_orders: bool = False

    def __post_init__(self) -> None:
        if not self.tool_id.strip():
            raise ValueError("tool_id must be a non-empty string.")
        if not isinstance(self.tool_class, ToolClass):
            raise TypeError("tool_class must be a ToolClass.")
        if not isinstance(self.subgoal, OperationalSubgoal):
            raise TypeError("subgoal must be an OperationalSubgoal.")
        if not isinstance(self.status, ToolStatus):
            raise TypeError("status must be a ToolStatus.")
        object.__setattr__(self, "structured_output", dict(self.structured_output))
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be within [0.0, 1.0] when provided.")
        if self.next_tool_id is not None and not self.next_tool_id.strip():
            raise ValueError("next_tool_id must be non-empty when provided.")
        if self.next_subgoal is not None and not isinstance(
            self.next_subgoal, OperationalSubgoal
        ):
            raise TypeError("next_subgoal must be an OperationalSubgoal when provided.")


@dataclass(frozen=True, slots=True)
class AgentSignal:
    """A bounded orchestration-runtime output with no raw order content."""

    selected_subgoal: OperationalSubgoal
    selected_tool_id: str | None = None
    tool_sequence: tuple[str, ...] = field(default_factory=tuple)
    abstained: bool = False
    no_action: bool = False
    request_replan: bool = False
    rationale: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.selected_subgoal, OperationalSubgoal):
            raise TypeError("selected_subgoal must be an OperationalSubgoal.")
        object.__setattr__(self, "tool_sequence", tuple(self.tool_sequence))
        for tool_id in self.tool_sequence:
            if not tool_id.strip():
                raise ValueError("tool_sequence must contain non-empty tool identifiers.")
        if self.selected_tool_id is not None and not self.selected_tool_id.strip():
            raise ValueError("selected_tool_id must be non-empty when provided.")
        if self.abstained and self.no_action:
            raise ValueError("abstained and no_action cannot both be True.")


@dataclass(frozen=True, slots=True)
class UpdateRequest:
    """A bounded orchestration-agent request to revise uncertainty handling.

    This is not an order request and must not encode raw replenishment actions.
    """

    request_type: UpdateRequestType
    target: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.request_type, UpdateRequestType):
            raise TypeError("request_type must be an UpdateRequestType.")
        if self.target is not None and not self.target.strip():
            raise ValueError("target must be non-empty when provided.")


@dataclass(frozen=True, slots=True)
class AgentAssessment:
    """A bounded orchestration-agent assessment with no raw order output.

    Empty update requests together with ``request_replan=False`` can represent
    abstain or no-action behavior under the current mission and guardrails.
    """

    regime_label: RegimeLabel
    confidence: float
    rationale: str
    update_requests: tuple[UpdateRequest, ...] = field(default_factory=tuple)
    request_replan: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.regime_label, RegimeLabel):
            raise TypeError("regime_label must be a RegimeLabel.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be within [0.0, 1.0].")
        if not self.rationale.strip():
            raise ValueError("rationale must be a non-empty string.")
        object.__setattr__(self, "update_requests", tuple(self.update_requests))
        for request in self.update_requests:
            if not isinstance(request, UpdateRequest):
                raise TypeError("update_requests must contain UpdateRequest values.")


@runtime_checkable
class BoundedTool(Protocol):
    """Protocol for bounded tools callable by the orchestration runtime."""

    @property
    def spec(self) -> ToolSpec:
        """Return static metadata describing the tool."""

    def invoke(self, invocation: ToolInvocation) -> ToolResult:
        """Run a bounded tool call and return a structured result."""


__all__ = [
    "AgentAssessment",
    "AgentSignal",
    "BackorderPolicy",
    "BenchmarkFamily",
    "BoundedTool",
    "MissionSpec",
    "OperationalSubgoal",
    "RegimeLabel",
    "ToolClass",
    "ToolInvocation",
    "ToolResult",
    "ToolSpec",
    "ToolStatus",
    "UpdateRequest",
    "UpdateRequestType",
]
