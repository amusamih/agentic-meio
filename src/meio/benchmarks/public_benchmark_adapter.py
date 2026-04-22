"""Public-benchmark inspection and bounded execution helpers."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from importlib import import_module
from importlib import invalidate_caches
from pathlib import Path
import sys
from typing import Any
from typing import Iterator

import numpy as np
import yaml

from meio.agents.baselines import DeterministicBaselinePolicy
from meio.agents.llm_client import FakeLLMClient, OpenAILLMClient
from meio.agents.runtime import (
    OrchestrationMemory,
    OrchestrationRequest,
    OrchestrationResponse,
    OrchestrationRuntime,
    RuntimeMode,
)
from meio.agents.telemetry import summarize_episode_telemetry
from meio.benchmarks.replenishmentenv_support import locate_package_root
from meio.config.loaders import load_agent_config
from meio.config.schemas import AgentConfig, PublicBenchmarkEvalConfig
from meio.contracts import MissionSpec, OperationalSubgoal, RegimeLabel, UpdateRequestType
from meio.evaluation.aggregate_results import (
    BatchAggregateSummary,
    aggregate_batch_episode_summaries,
)
from meio.evaluation.logging_io import hash_jsonable
from meio.evaluation.logging_schema import (
    CostBreakdownRecord,
    EpisodeSummaryRecord,
    LLMCallTraceRecord,
    StepTraceRecord,
    ToolCallTraceRecord,
)
from meio.evaluation.summaries import (
    build_benchmark_run_summary,
    build_episode_summary_record,
)
from meio.forecasting.adapters import DeterministicForecastTool
from meio.leadtime.adapters import DeterministicLeadTimeTool
from meio.optimization.adapters import TrustedOptimizerAdapter
from meio.optimization.contracts import OptimizationRequest, OptimizationResult, OptimizationStatus
from meio.scenarios.adapters import DeterministicScenarioTool
from meio.scenarios.contracts import (
    ScenarioAdjustmentSummary,
    ScenarioSummary,
    ScenarioUpdateResult,
)
from meio.simulation.evidence import DemandEvidence, LeadTimeEvidence, RuntimeEvidence
from meio.simulation.state import EpisodeTrace, Observation, PeriodTraceRecord, SimulationState


VALIDATION_LANE = "public_benchmark"
_REPLENISHMENTENV_REQUIRED_PATHS: tuple[str, ...] = (
    "env/replenishment_env.py",
    "wrapper/default_wrapper.py",
    "wrapper/dynamic_wrapper.py",
    "wrapper/history_wrapper.py",
    "wrapper/observation_wrapper.py",
    "wrapper/observation_wrapper_for_old_code.py",
    "wrapper/flatten_wrapper.py",
    "wrapper/oracle_wrapper.py",
)
_REPO_ROOT = Path(__file__).resolve().parents[3]
_LOCAL_REPLENISHMENTENV_ROOT = _REPO_ROOT / "third_party" / "ReplenishmentEnv"
_SETUP_ROOT_CAUSE = (
    "setup.py uses find_packages(include=['ReplenishmentEnv']) and excludes "
    "ReplenishmentEnv.* subpackages from built installs."
)
_PUBLIC_BENCHMARK_NOTES = (
    "benchmark_reward_not_directly_comparable_to_stockpyl_total_cost",
    "single_store_mapping_only",
    "aggregate_observation_with_per_sku_optimizer_execution",
)


def _non_empty(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")


@dataclass(frozen=True, slots=True)
class ReplenishmentEnvDemoSummary:
    """Compact view over the packaged ReplenishmentEnv demo config."""

    demo_config_path: str
    mode_names: tuple[str, ...]
    warehouse_names: tuple[str, ...]
    lookback_len: int | None
    horizon: int | None
    referenced_files: tuple[str, ...] = ()
    missing_files: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _non_empty(self.demo_config_path, "demo_config_path")
        object.__setattr__(self, "mode_names", tuple(self.mode_names))
        object.__setattr__(self, "warehouse_names", tuple(self.warehouse_names))
        object.__setattr__(self, "referenced_files", tuple(self.referenced_files))
        object.__setattr__(self, "missing_files", tuple(self.missing_files))


@dataclass(frozen=True, slots=True)
class PublicBenchmarkSmokeSummary:
    """Compact environment smoke-execution summary."""

    config_name: str
    wrapper_names: tuple[str, ...]
    benchmark_mode: str
    smoke_horizon_steps: int
    attempted: bool
    succeeded: bool
    completed_steps: int = 0
    observation_shape: tuple[int, ...] = ()
    action_shape: tuple[int, ...] = ()
    reward_shape: tuple[int, ...] = ()
    reward_sum: float | None = None
    done_reached: bool | None = None
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        _non_empty(self.config_name, "config_name")
        _non_empty(self.benchmark_mode, "benchmark_mode")
        object.__setattr__(self, "wrapper_names", tuple(self.wrapper_names))
        object.__setattr__(self, "observation_shape", tuple(self.observation_shape))
        object.__setattr__(self, "action_shape", tuple(self.action_shape))
        object.__setattr__(self, "reward_shape", tuple(self.reward_shape))
        if self.smoke_horizon_steps <= 0:
            raise ValueError("smoke_horizon_steps must be positive.")
        if self.completed_steps < 0:
            raise ValueError("completed_steps must be non-negative.")
        if self.failure_reason is not None:
            _non_empty(self.failure_reason, "failure_reason")


@dataclass(frozen=True, slots=True)
class PublicBenchmarkAdapterSummary:
    """Auditable adapter status for a public benchmark candidate."""

    benchmark_candidate: str
    benchmark_source: str
    validation_lane: str
    module_name: str
    module_discovered: bool
    import_succeeds: bool
    package_layout_complete: bool
    environment_runnable: bool
    benchmark_root: str | None
    blocked_reason: str | None = None
    demo_summary: ReplenishmentEnvDemoSummary | None = None
    smoke_summary: PublicBenchmarkSmokeSummary | None = None
    using_local_source_checkout: bool = False
    source_checkout_root: str | None = None
    packaging_root_cause: str | None = None
    bounded_orchestrator_integration_supported: bool = False
    missing_required_paths: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for field_name in (
            "benchmark_candidate",
            "benchmark_source",
            "validation_lane",
            "module_name",
        ):
            _non_empty(getattr(self, field_name), field_name)
        if self.blocked_reason is not None:
            _non_empty(self.blocked_reason, "blocked_reason")
        object.__setattr__(self, "missing_required_paths", tuple(self.missing_required_paths))
        object.__setattr__(self, "notes", tuple(self.notes))
        for relative_path in self.missing_required_paths:
            _non_empty(relative_path, "missing_required_paths")
        for note in self.notes:
            _non_empty(note, "notes")


@dataclass(frozen=True, slots=True)
class PublicBenchmarkActionContract:
    """Concrete public-benchmark action and observation contract."""

    environment_config_name: str
    wrapper_names: tuple[str, ...]
    benchmark_mode: str
    warehouse_names: tuple[str, ...]
    sku_count: int
    action_mode: str
    advertised_action_shape: tuple[int, ...]
    expected_action_shape: tuple[int, ...]
    observation_shape: tuple[int, ...]
    reward_shape: tuple[int, ...]
    action_units: str
    reward_semantics: str

    def __post_init__(self) -> None:
        for field_name in (
            "environment_config_name",
            "benchmark_mode",
            "action_mode",
            "action_units",
            "reward_semantics",
        ):
            _non_empty(getattr(self, field_name), field_name)
        object.__setattr__(self, "wrapper_names", tuple(self.wrapper_names))
        object.__setattr__(self, "warehouse_names", tuple(self.warehouse_names))
        object.__setattr__(self, "advertised_action_shape", tuple(self.advertised_action_shape))
        object.__setattr__(self, "expected_action_shape", tuple(self.expected_action_shape))
        object.__setattr__(self, "observation_shape", tuple(self.observation_shape))
        object.__setattr__(self, "reward_shape", tuple(self.reward_shape))
        if self.sku_count <= 0:
            raise ValueError("sku_count must be positive.")


@dataclass(frozen=True, slots=True)
class PublicBenchmarkModeSummary:
    """Per-mode public-benchmark execution summary."""

    mode: str
    completed: bool
    step_count: int
    total_reward: float
    average_step_reward: float
    average_fill_rate: float | None
    average_inventory_units: float | None
    total_order_quantity: float
    invalid_output_count: int
    fallback_count: int
    tool_call_count: int
    llm_provider: str | None = None
    llm_model_name: str | None = None

    def __post_init__(self) -> None:
        _non_empty(self.mode, "mode")
        if self.step_count < 0:
            raise ValueError("step_count must be non-negative.")
        for field_name in (
            "total_reward",
            "average_step_reward",
            "total_order_quantity",
        ):
            if not isinstance(getattr(self, field_name), (int, float)):
                raise TypeError(f"{field_name} must be numeric.")
        for field_name in ("invalid_output_count", "fallback_count", "tool_call_count"):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative.")
        if self.average_fill_rate is not None and not 0.0 <= self.average_fill_rate <= 1.0:
            raise ValueError("average_fill_rate must be within [0.0, 1.0] when provided.")
        if self.average_inventory_units is not None and self.average_inventory_units < 0.0:
            raise ValueError("average_inventory_units must be non-negative when provided.")
        if self.llm_provider is not None:
            _non_empty(self.llm_provider, "llm_provider")
        if self.llm_model_name is not None:
            _non_empty(self.llm_model_name, "llm_model_name")


@dataclass(frozen=True, slots=True)
class PublicBenchmarkEvaluationSummary:
    """Top-level public-benchmark execution summary."""

    benchmark_candidate: str
    benchmark_source: str
    validation_lane: str
    benchmark_root: str | None
    blocked_reason: str | None
    environment_runnable: bool
    using_local_source_checkout: bool
    source_checkout_root: str | None
    packaging_root_cause: str | None
    bounded_orchestrator_integration_supported: bool
    mapping_identity: str | None = None
    environment_contract: PublicBenchmarkActionContract | None = None
    mode_summaries: tuple[PublicBenchmarkModeSummary, ...] = ()
    comparability_notes: tuple[str, ...] = field(default_factory=tuple)
    mapping_assumptions: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for field_name in ("benchmark_candidate", "benchmark_source", "validation_lane"):
            _non_empty(getattr(self, field_name), field_name)
        if self.blocked_reason is not None:
            _non_empty(self.blocked_reason, "blocked_reason")
        if self.mapping_identity is not None:
            _non_empty(self.mapping_identity, "mapping_identity")
        object.__setattr__(self, "mode_summaries", tuple(self.mode_summaries))
        object.__setattr__(self, "comparability_notes", tuple(self.comparability_notes))
        object.__setattr__(self, "mapping_assumptions", tuple(self.mapping_assumptions))
        object.__setattr__(self, "notes", tuple(self.notes))
        for item in self.comparability_notes + self.mapping_assumptions + self.notes:
            _non_empty(item, "summary_notes")


@dataclass(frozen=True, slots=True)
class PublicBenchmarkModeArtifacts:
    """One completed public-benchmark mode run plus structured artifacts."""

    mode_summary: PublicBenchmarkModeSummary
    episode_summary_record: EpisodeSummaryRecord
    step_trace_records: tuple[StepTraceRecord, ...]
    llm_call_trace_records: tuple[LLMCallTraceRecord, ...]
    tool_call_trace_records: tuple[ToolCallTraceRecord, ...]


@dataclass(frozen=True, slots=True)
class PublicBenchmarkExecutionBatch:
    """Structured batch result for the public benchmark lane."""

    inspection_summary: PublicBenchmarkAdapterSummary
    evaluation_summary: PublicBenchmarkEvaluationSummary
    mode_artifacts: tuple[PublicBenchmarkModeArtifacts, ...]
    aggregate_summary: BatchAggregateSummary | None


@dataclass(frozen=True, slots=True)
class _BenchmarkSnapshot:
    warehouse_names: tuple[str, ...]
    sku_ids: tuple[str, ...]
    action_mode: str
    demand_history_by_sku: np.ndarray
    leadtime_history_by_sku: np.ndarray
    demand_mean_by_sku: np.ndarray
    leadtime_mean_by_sku: np.ndarray
    in_stock_by_sku: np.ndarray
    in_transit_by_sku: np.ndarray


@dataclass(frozen=True, slots=True)
class _AggregatedSnapshot:
    demand_history: tuple[float, ...]
    leadtime_history: tuple[float, ...]
    inventory_mean: float
    pipeline_mean: float


@dataclass(frozen=True, slots=True)
class _StepOutcome:
    reward_sum: float
    fill_rate: float | None
    total_demand: float
    total_sale: float
    ending_inventory_total: float


def inspect_replenishmentenv_installation(
    *,
    module_name: str = "ReplenishmentEnv",
    benchmark_root: Path | None = None,
    demo_config_path: Path = Path("config/demo.yml"),
    run_smoke_execution: bool = False,
    environment_config_name: str = "sku100.single_store.standard",
    wrapper_names: tuple[str, ...] = ("DefaultWrapper",),
    benchmark_mode: str = "test",
    smoke_horizon_steps: int = 1,
    vis_path: Path | None = None,
) -> PublicBenchmarkAdapterSummary:
    """Inspect the local ReplenishmentEnv installation and optionally run it."""

    package_root = locate_package_root(module_name=module_name, explicit_root=benchmark_root)
    module_discovered = package_root is not None
    import_succeeds = False
    blocked_reason: str | None = None
    missing_required_paths: tuple[str, ...] = ()
    package_layout_complete = False
    import_error_text: str | None = None
    demo_summary = None
    smoke_summary: PublicBenchmarkSmokeSummary | None = None
    notes: list[str] = []
    using_local_source_checkout = False
    source_checkout_root: str | None = None
    packaging_root_cause: str | None = None

    if package_root is not None:
        using_local_source_checkout = _is_local_source_checkout(package_root)
        source_checkout_root = str(_source_checkout_root(package_root))
        packaging_root_cause = _detect_packaging_root_cause(package_root)
        missing_required_paths = _missing_required_paths(package_root)
        package_layout_complete = not missing_required_paths
        try:
            _import_module_from_package_root(module_name=module_name, package_root=package_root)
            import_succeeds = True
        except Exception as exc:  # pragma: no cover - local import state.
            import_error_text = f"{exc.__class__.__name__}: {exc}"
        resolved_demo_path = package_root / demo_config_path
        if resolved_demo_path.exists():
            demo_summary = _read_demo_summary(package_root, resolved_demo_path)
        else:
            notes.append("demo_config_missing")
    else:
        blocked_reason = "module_not_found"

    if import_error_text is not None:
        notes.append(f"import_error={import_error_text}")
    if using_local_source_checkout:
        notes.append("local_source_checkout_used")
    if missing_required_paths:
        notes.append("installed_package_layout_incomplete")
        blocked_reason = "incomplete_installation: missing " + ", ".join(missing_required_paths)
    elif blocked_reason is None and import_error_text is not None:
        blocked_reason = import_error_text

    if (
        run_smoke_execution
        and package_root is not None
        and import_succeeds
        and package_layout_complete
    ):
        smoke_summary = execute_replenishmentenv_smoke(
            module_name=module_name,
            package_root=package_root,
            environment_config_name=environment_config_name,
            wrapper_names=wrapper_names,
            benchmark_mode=benchmark_mode,
            smoke_horizon_steps=smoke_horizon_steps,
            vis_path=vis_path,
        )
        if smoke_summary.succeeded:
            notes.append("environment_smoke_execution_succeeded")
        elif smoke_summary.failure_reason is not None:
            blocked_reason = smoke_summary.failure_reason

    environment_runnable = (
        smoke_summary.succeeded
        if smoke_summary is not None
        else import_succeeds and package_layout_complete
    )
    if environment_runnable:
        notes.append("bounded_orchestrator_mapping_still_partial")
    elif blocked_reason is None:
        blocked_reason = "environment_not_runnable"

    return PublicBenchmarkAdapterSummary(
        benchmark_candidate="replenishment_env",
        benchmark_source=VALIDATION_LANE,
        validation_lane=VALIDATION_LANE,
        module_name=module_name,
        module_discovered=module_discovered,
        import_succeeds=import_succeeds,
        package_layout_complete=package_layout_complete,
        environment_runnable=environment_runnable,
        benchmark_root=(str(package_root) if package_root is not None else None),
        blocked_reason=blocked_reason,
        demo_summary=demo_summary,
        smoke_summary=smoke_summary,
        using_local_source_checkout=using_local_source_checkout,
        source_checkout_root=source_checkout_root,
        packaging_root_cause=packaging_root_cause,
        bounded_orchestrator_integration_supported=False,
        missing_required_paths=missing_required_paths,
        notes=tuple(notes),
    )


def execute_replenishmentenv_smoke(
    *,
    module_name: str,
    package_root: Path,
    environment_config_name: str,
    wrapper_names: tuple[str, ...],
    benchmark_mode: str,
    smoke_horizon_steps: int,
    vis_path: Path | None,
) -> PublicBenchmarkSmokeSummary:
    """Run a tiny deterministic smoke execution against the public environment."""

    env = None
    observation_shape: tuple[int, ...] = ()
    action_shape: tuple[int, ...] = ()
    reward_shape: tuple[int, ...] = ()
    completed_steps = 0
    reward_sum = 0.0
    done_reached = False
    try:
        module = _import_module_from_package_root(
            module_name=module_name,
            package_root=package_root,
        )
        make_env = getattr(module, "make_env")
        env = make_env(
            environment_config_name,
            wrapper_names=list(wrapper_names),
            mode=benchmark_mode,
            vis_path=(str(vis_path) if vis_path is not None else None),
        )
        observation = np.asarray(env.reset())
        observation_shape = tuple(int(dimension) for dimension in observation.shape)
        raw_action_shape = getattr(env.action_space, "shape", ()) or ()
        action_shape = tuple(int(dimension) for dimension in raw_action_shape)
        step_action_shape = _expected_step_action_shape(env)
        action = np.zeros(step_action_shape, dtype=np.float32)
        for step_index in range(smoke_horizon_steps):
            observation, reward, done_flag, _info = env.step(action)
            completed_steps = step_index + 1
            observation_shape = tuple(int(dimension) for dimension in np.asarray(observation).shape)
            reward_array = np.asarray(reward)
            reward_shape = tuple(int(dimension) for dimension in reward_array.shape)
            reward_sum += float(reward_array.sum())
            done_reached = bool(done_flag)
            if done_reached:
                break
        return PublicBenchmarkSmokeSummary(
            config_name=environment_config_name,
            wrapper_names=wrapper_names,
            benchmark_mode=benchmark_mode,
            smoke_horizon_steps=smoke_horizon_steps,
            attempted=True,
            succeeded=True,
            completed_steps=completed_steps,
            observation_shape=observation_shape,
            action_shape=action_shape,
            reward_shape=reward_shape,
            reward_sum=reward_sum,
            done_reached=done_reached,
        )
    except Exception as exc:  # pragma: no cover - depends on local runtime state.
        return PublicBenchmarkSmokeSummary(
            config_name=environment_config_name,
            wrapper_names=wrapper_names,
            benchmark_mode=benchmark_mode,
            smoke_horizon_steps=smoke_horizon_steps,
            attempted=True,
            succeeded=False,
            completed_steps=completed_steps,
            observation_shape=observation_shape,
            action_shape=action_shape,
            reward_shape=reward_shape,
            reward_sum=(reward_sum if completed_steps > 0 else None),
            done_reached=(done_reached if completed_steps > 0 else None),
            failure_reason=f"{exc.__class__.__name__}: {exc}",
        )
    finally:
        if env is not None and hasattr(env, "close"):
            env.close()


def run_public_benchmark_execution(
    *,
    config: PublicBenchmarkEvalConfig,
    selected_modes: tuple[str, ...] | None = None,
    llm_client_mode_override: str | None = None,
    vis_root: Path | None = None,
) -> PublicBenchmarkExecutionBatch:
    """Run the first bounded-method public benchmark batch."""

    inspection_summary = inspect_replenishmentenv_installation(
        module_name=config.discovery_module,
        benchmark_root=config.benchmark_root,
        demo_config_path=config.demo_config_path,
        run_smoke_execution=False,
        environment_config_name=config.environment_config_name,
        wrapper_names=config.wrapper_names,
        benchmark_mode=config.benchmark_mode,
    )
    if not inspection_summary.environment_runnable or inspection_summary.benchmark_root is None:
        evaluation_summary = PublicBenchmarkEvaluationSummary(
            benchmark_candidate=config.benchmark_candidate,
            benchmark_source=VALIDATION_LANE,
            validation_lane=VALIDATION_LANE,
            benchmark_root=inspection_summary.benchmark_root,
            blocked_reason=inspection_summary.blocked_reason,
            environment_runnable=False,
            using_local_source_checkout=inspection_summary.using_local_source_checkout,
            source_checkout_root=inspection_summary.source_checkout_root,
            packaging_root_cause=inspection_summary.packaging_root_cause,
            bounded_orchestrator_integration_supported=False,
            mapping_identity=None,
            comparability_notes=("environment_import_or_layout_blocked",),
            mapping_assumptions=(),
            notes=inspection_summary.notes,
        )
        return PublicBenchmarkExecutionBatch(
            inspection_summary=inspection_summary,
            evaluation_summary=evaluation_summary,
            mode_artifacts=(),
            aggregate_summary=None,
        )

    package_root = Path(inspection_summary.benchmark_root)
    module = _import_module_from_package_root(
        module_name=config.discovery_module,
        package_root=package_root,
    )
    agent_config = load_agent_config(config.agent_config_path)
    if llm_client_mode_override is not None:
        agent_config = replace(agent_config, llm_client_mode=llm_client_mode_override)
    active_modes = config.mode_set if selected_modes is None else selected_modes
    mode_artifacts = tuple(
        _run_public_benchmark_mode(
            module=module,
            config=config,
            agent_config=agent_config,
            mode_name=mode_name,
            vis_path=(vis_root / mode_name if vis_root is not None else None),
        )
        for mode_name in active_modes
    )
    environment_contract = _read_environment_contract(
        module=module,
        config=config,
    )
    evaluation_summary = PublicBenchmarkEvaluationSummary(
        benchmark_candidate=config.benchmark_candidate,
        benchmark_source=VALIDATION_LANE,
        validation_lane=VALIDATION_LANE,
        benchmark_root=inspection_summary.benchmark_root,
        blocked_reason=None,
        environment_runnable=True,
        using_local_source_checkout=inspection_summary.using_local_source_checkout,
        source_checkout_root=inspection_summary.source_checkout_root,
        packaging_root_cause=inspection_summary.packaging_root_cause,
        bounded_orchestrator_integration_supported=True,
        mapping_identity=_build_mapping_identity(
            environment_contract=environment_contract,
            comparability_notes=(
                "public_benchmark_reward_is_profit_like_not_stockpyl_total_cost",
                "public_benchmark_fill_rate_is_comparable_as_service_proxy",
                "multi_store_public_benchmark_mapping_not_attempted_in_this_pass",
            ),
            mapping_assumptions=(
                "orchestrator_observation_uses_aggregate_per_sku_means",
                "trusted_optimizer_executes_one_single_stage_request_per_sku",
                "optimizer_quantities_are_converted_to_benchmark_actions_by_dividing_by_lookback_demand_mean",
            ),
        ),
        environment_contract=environment_contract,
        mode_summaries=tuple(item.mode_summary for item in mode_artifacts),
        comparability_notes=(
            "public_benchmark_reward_is_profit_like_not_stockpyl_total_cost",
            "public_benchmark_fill_rate_is_comparable_as_service_proxy",
            "multi_store_public_benchmark_mapping_not_attempted_in_this_pass",
        ),
        mapping_assumptions=(
            "orchestrator_observation_uses_aggregate_per_sku_means",
            "trusted_optimizer_executes_one_single_stage_request_per_sku",
            "optimizer_quantities_are_converted_to_benchmark_actions_by_dividing_by_lookback_demand_mean",
        ),
        notes=inspection_summary.notes + ("bounded_method_execution_succeeded",),
    )
    records_by_mode = {
        artifact.mode_summary.mode: (artifact.episode_summary_record,)
        for artifact in mode_artifacts
    }
    step_trace_records_by_mode = {
        artifact.mode_summary.mode: artifact.step_trace_records
        for artifact in mode_artifacts
    }
    tool_call_records_by_mode = {
        artifact.mode_summary.mode: artifact.tool_call_trace_records
        for artifact in mode_artifacts
    }
    aggregate_summary = aggregate_batch_episode_summaries(
        benchmark_id=f"{config.benchmark_candidate}:{config.environment_config_name}",
        validation_lane=VALIDATION_LANE,
        records_by_mode=records_by_mode,
        step_trace_records_by_mode=step_trace_records_by_mode,
        tool_call_records_by_mode=tool_call_records_by_mode,
    )
    return PublicBenchmarkExecutionBatch(
        inspection_summary=inspection_summary,
        evaluation_summary=evaluation_summary,
        mode_artifacts=mode_artifacts,
        aggregate_summary=aggregate_summary,
    )


def map_optimizer_orders_to_public_benchmark_actions(
    *,
    replenishment_quantities: np.ndarray,
    demand_mean_by_sku: np.ndarray,
    action_mode: str,
    demand_scale_epsilon: float,
) -> np.ndarray:
    """Convert trusted optimizer quantities into benchmark action semantics."""

    if demand_scale_epsilon <= 0.0:
        raise ValueError("demand_scale_epsilon must be positive.")
    replenishment_quantities = np.asarray(replenishment_quantities, dtype=np.float32)
    demand_mean_by_sku = np.asarray(demand_mean_by_sku, dtype=np.float32)
    if replenishment_quantities.shape != demand_mean_by_sku.shape:
        raise ValueError("replenishment_quantities must match demand_mean_by_sku shape.")
    clipped_quantities = np.maximum(replenishment_quantities, 0.0)
    if action_mode == "continuous":
        return clipped_quantities
    if action_mode == "demand_mean_continuous":
        safe_mean = np.maximum(demand_mean_by_sku, demand_scale_epsilon)
        return clipped_quantities / safe_mean
    raise ValueError(f"Unsupported benchmark action_mode for honest mapping: {action_mode!r}.")


def _run_public_benchmark_mode(
    *,
    module: object,
    config: PublicBenchmarkEvalConfig,
    agent_config: AgentConfig,
    mode_name: str,
    vis_path: Path | None,
) -> PublicBenchmarkModeArtifacts:
    env = getattr(module, "make_env")(
        config.environment_config_name,
        wrapper_names=list(config.wrapper_names),
        mode=config.benchmark_mode,
        vis_path=(str(vis_path) if vis_path is not None else None),
    )
    try:
        warehouse_count = len(tuple(env.get_warehouse_list()))
        if warehouse_count != 1:
            raise ValueError(
                "Only single-store ReplenishmentEnv configs are supported for the "
                "current thin action-mapping adapter."
            )
        mission = _build_public_benchmark_mission()
        runtime: OrchestrationRuntime | None = None
        llm_provider: str | None = None
        if mode_name != "deterministic_baseline":
            runtime, llm_provider = _build_runtime(agent_config, mode=mode_name)
        optimizer = TrustedOptimizerAdapter()
        baseline_policy = DeterministicBaselinePolicy()
        env.reset()
        initial_snapshot = _capture_snapshot(env)
        demand_baseline_value = float(np.asarray(initial_snapshot.demand_history_by_sku).mean())
        leadtime_baseline_value = float(np.asarray(initial_snapshot.leadtime_history_by_sku).mean())
        benchmark_id = f"{config.benchmark_candidate}:{config.environment_config_name}"
        previous_regime = RegimeLabel.NORMAL
        period_records: list[PeriodTraceRecord] = []
        step_trace_records: list[StepTraceRecord] = []
        llm_call_trace_records: list[LLMCallTraceRecord] = []
        tool_call_trace_records: list[ToolCallTraceRecord] = []
        responses: list[OrchestrationResponse] = []
        inferred_regime_schedule: list[str] = []
        total_reward = 0.0
        total_demand = 0.0
        total_sale = 0.0
        ending_inventory_totals: list[float] = []
        total_order_quantity = 0.0
        run_id = f"{mode_name}_{config.environment_config_name}"
        for step_index in range(config.episode_horizon_steps):
            snapshot = _capture_snapshot(env)
            aggregate = _aggregate_snapshot(snapshot)
            inferred_regime = _infer_public_regime(
                demand_value=aggregate.demand_history[-1],
                leadtime_value=aggregate.leadtime_history[-1],
                demand_baseline_value=demand_baseline_value,
                leadtime_baseline_value=leadtime_baseline_value,
                previous_regime=previous_regime,
            )
            previous_regime = inferred_regime
            inferred_regime_schedule.append(inferred_regime.value)
            system_state = SimulationState(
                benchmark_id=benchmark_id,
                time_index=step_index,
                inventory_level=(aggregate.inventory_mean,),
                stage_names=("public_store",),
                pipeline_inventory=(aggregate.pipeline_mean,),
                backorder_level=(0.0,),
                regime_label=inferred_regime,
            )
            observation = Observation(
                time_index=step_index,
                demand_evidence=DemandEvidence(
                    history=aggregate.demand_history,
                    latest_realization=(aggregate.demand_history[-1],),
                    stage_index=1,
                ),
                leadtime_evidence=LeadTimeEvidence(
                    history=aggregate.leadtime_history,
                    latest_realization=(aggregate.leadtime_history[-1],),
                    upstream_stage_index=2,
                    downstream_stage_index=1,
                ),
                regime_label=inferred_regime,
                notes=("public_benchmark", config.environment_config_name),
            )
            evidence = RuntimeEvidence(
                time_index=step_index,
                demand=observation.demand_evidence,
                leadtime=observation.leadtime_evidence,
                scenario_families=tuple(label for label in RegimeLabel),
                demand_baseline_value=demand_baseline_value,
                leadtime_baseline_value=leadtime_baseline_value,
                notes=("public_benchmark", config.environment_config_name),
            )
            response: OrchestrationResponse | None = None
            if mode_name == "deterministic_baseline":
                decision = baseline_policy.decide(system_state, observation, evidence)
                scenario_update_result = decision.scenario_update_result
                agent_signal = decision.signal
            else:
                request = OrchestrationRequest(
                    mission=mission,
                    system_state=system_state,
                    observation=observation,
                    evidence=evidence,
                    requested_subgoal=OperationalSubgoal.INSPECT_EVIDENCE,
                    candidate_tool_ids=mission.admissible_tool_ids,
                    recent_regime_history=tuple(
                        record.ending_state.regime_label for record in period_records[-3:]
                    ),
                    recent_stress_reference_demand_value=demand_baseline_value,
                    recent_update_request_history=tuple(
                        (UpdateRequestType.KEEP_CURRENT,) for _ in period_records[-3:]
                    ),
                    memory=OrchestrationMemory(),
                )
                response = runtime.run(request)
                responses.append(response)
                scenario_update_result = _extract_scenario_update_result(response)
                if scenario_update_result is None:
                    scenario_update_result = _default_keep_current_update(
                        inferred_regime,
                        observation,
                        provenance=f"{mode_name}_keep_current_fallback",
                    )
                agent_signal = response.signal
                llm_call_trace_records.extend(
                    _build_llm_call_trace_records(
                        episode_id=run_id,
                        mode=mode_name,
                        run_seed=0,
                        response=response,
                        schedule_name=config.environment_config_name,
                        period_index=step_index,
                    )
                )
                tool_call_trace_records.extend(
                    _build_tool_call_trace_records(
                        episode_id=run_id,
                        mode=mode_name,
                        run_seed=0,
                        response=response,
                        schedule_name=config.environment_config_name,
                        period_index=step_index,
                    )
                )
            order_quantities = _solve_skuwise_orders(
                optimizer=optimizer,
                snapshot=snapshot,
                scenario_update_result=scenario_update_result,
                base_stock_multiplier=config.base_stock_multiplier,
                time_index=step_index,
            )
            counterfactual_quantities = _solve_skuwise_orders(
                optimizer=optimizer,
                snapshot=snapshot,
                scenario_update_result=_default_keep_current_update(
                    inferred_regime,
                    observation,
                    provenance=f"{mode_name}_counterfactual_keep_current",
                ),
                base_stock_multiplier=config.base_stock_multiplier,
                time_index=step_index,
            )
            action_matrix = map_optimizer_orders_to_public_benchmark_actions(
                replenishment_quantities=order_quantities,
                demand_mean_by_sku=snapshot.demand_mean_by_sku,
                action_mode=snapshot.action_mode,
                demand_scale_epsilon=config.demand_scale_epsilon,
            )
            _next_observation, reward, done_flag, _info = env.step(action_matrix)
            step_outcome = _capture_step_outcome(env, reward)
            next_snapshot = _capture_snapshot(env)
            next_aggregate = _aggregate_snapshot(next_snapshot)
            next_state = SimulationState(
                benchmark_id=benchmark_id,
                time_index=step_index + 1,
                inventory_level=(next_aggregate.inventory_mean,),
                stage_names=("public_store",),
                pipeline_inventory=(next_aggregate.pipeline_mean,),
                backorder_level=(0.0,),
                regime_label=inferred_regime,
            )
            aggregate_optimization_result = OptimizationResult(
                replenishment_orders=(float(order_quantities.sum()),),
                planning_horizon=1,
                status=OptimizationStatus.SUCCESS,
                objective_value=float(order_quantities.sum()),
                provenance="public_benchmark_skuwise_trusted_optimizer_aggregate",
                metadata=("skuwise_orders_aggregated_for_trace",),
            )
            period_record = PeriodTraceRecord(
                time_index=step_index,
                regime_label=inferred_regime,
                state=system_state,
                observation=observation,
                agent_signal=agent_signal,
                optimization_result=aggregate_optimization_result,
                next_state=next_state,
                realized_demand=step_outcome.total_demand,
                demand_load=step_outcome.total_demand,
                served_demand=step_outcome.total_sale,
                unmet_demand=max(0.0, step_outcome.total_demand - step_outcome.total_sale),
                step_telemetry=(response.step_telemetry if response is not None else None),
                notes=("public_benchmark",),
            )
            period_records.append(period_record)
            total_reward += step_outcome.reward_sum
            total_demand += step_outcome.total_demand
            total_sale += step_outcome.total_sale
            ending_inventory_totals.append(step_outcome.ending_inventory_total)
            total_order_quantity += float(order_quantities.sum())
            step_trace_records.append(
                StepTraceRecord(
                    episode_id=run_id,
                    mode=mode_name,
                    tool_ablation_variant="full",
                    schedule_name=config.environment_config_name,
                    run_seed=0,
                    period_index=step_index,
                    true_regime_label="unlabeled_public_benchmark",
                    predicted_regime_label=(
                        response.agent_assessment.regime_label.value
                        if response is not None and response.agent_assessment is not None
                        else None
                    ),
                    confidence=(
                        response.agent_assessment.confidence
                        if response is not None and response.agent_assessment is not None
                        else None
                    ),
                    selected_subgoal=agent_signal.selected_subgoal.value,
                    selected_tools=agent_signal.tool_sequence,
                    update_requests=tuple(
                        update_type.value
                        for update_type in scenario_update_result.applied_update_types
                    ),
                    request_replan=agent_signal.request_replan,
                    abstain_or_no_action=agent_signal.abstained or agent_signal.no_action,
                    demand_outlook=scenario_update_result.adjustment.demand_outlook,
                    leadtime_outlook=scenario_update_result.adjustment.leadtime_outlook,
                    scenario_adjustment_summary={
                        "demand_outlook": scenario_update_result.adjustment.demand_outlook,
                        "leadtime_outlook": scenario_update_result.adjustment.leadtime_outlook,
                        "safety_buffer_scale": (
                            scenario_update_result.adjustment.safety_buffer_scale
                        ),
                    },
                    optimizer_orders=(float(order_quantities.sum()),),
                    inventory_by_echelon=system_state.inventory_level,
                    pipeline_by_echelon=system_state.pipeline_inventory,
                    backorders_by_echelon=(0.0,),
                    per_period_cost=None,
                    per_period_fill_rate=step_outcome.fill_rate,
                    decision_changed_optimizer_input=not np.allclose(
                        order_quantities,
                        counterfactual_quantities,
                    ),
                    optimizer_output_changed_state=bool(float(order_quantities.sum()) > 0.0),
                    intervention_changed_outcome=False,
                    validation_lane=VALIDATION_LANE,
                )
            )
            if bool(done_flag):
                break
        trace = EpisodeTrace(
            run_id=run_id,
            benchmark_id=benchmark_id,
            period_records=tuple(period_records),
        )
        telemetry_summary = summarize_episode_telemetry(trace.step_telemetry)
        average_fill_rate = (total_sale / total_demand) if total_demand > 0.0 else None
        average_inventory_total = (
            float(sum(ending_inventory_totals) / len(ending_inventory_totals))
            if ending_inventory_totals
            else None
        )
        run_summary = replace(
            build_benchmark_run_summary(
                run_id=run_id,
                benchmark_id=benchmark_id,
                benchmark_source=VALIDATION_LANE,
                topology="single_store_public_benchmark",
                echelon_count=1,
                traces=(trace,),
                optimizer_order_boundary_preserved=_optimizer_boundary_preserved(tuple(responses)),
                notes=_PUBLIC_BENCHMARK_NOTES,
            ),
            total_cost=None,
            fill_rate=average_fill_rate,
            average_inventory=average_inventory_total,
            average_backorder_level=None,
            episode_telemetry=telemetry_summary,
        )
        episode_summary_record = build_episode_summary_record(
            run_summary,
            mode=mode_name,
            validation_lane=VALIDATION_LANE,
            tool_ablation_variant="full",
            schedule_name=config.environment_config_name,
            run_seed=0,
            regime_schedule=tuple(inferred_regime_schedule),
            cost_breakdown=CostBreakdownRecord(
                holding_cost=0.0,
                backlog_cost=0.0,
                ordering_cost=0.0,
                other_cost=0.0,
            ),
            intervention_count=sum(
                1
                for record in step_trace_records
                if record.request_replan
                or bool(record.selected_tools)
                or record.decision_changed_optimizer_input
            ),
            invalid_output_count=telemetry_summary.invalid_output_count,
            fallback_count=telemetry_summary.fallback_count,
            rollout_fidelity_gate_passed=True,
            operational_metrics_gate_passed=True,
        )
        mode_summary = PublicBenchmarkModeSummary(
            mode=mode_name,
            completed=True,
            step_count=trace.trace_length,
            total_reward=total_reward,
            average_step_reward=(total_reward / trace.trace_length if trace.trace_length else 0.0),
            average_fill_rate=average_fill_rate,
            average_inventory_units=average_inventory_total,
            total_order_quantity=total_order_quantity,
            invalid_output_count=telemetry_summary.invalid_output_count,
            fallback_count=telemetry_summary.fallback_count,
            tool_call_count=telemetry_summary.total_tool_call_count,
            llm_provider=telemetry_summary.provider,
            llm_model_name=telemetry_summary.model_name,
        )
        return PublicBenchmarkModeArtifacts(
            mode_summary=mode_summary,
            episode_summary_record=episode_summary_record,
            step_trace_records=tuple(step_trace_records),
            llm_call_trace_records=tuple(llm_call_trace_records),
            tool_call_trace_records=tuple(tool_call_trace_records),
        )
    finally:
        if hasattr(env, "close"):
            env.close()


def _capture_snapshot(env: object) -> _BenchmarkSnapshot:
    base_env = env.unwrapped
    warehouse_names = tuple(str(name) for name in env.get_warehouse_list())
    sku_ids = tuple(str(item) for item in env.get_sku_list())
    demand_history = np.asarray(
        base_env.agent_states["all_warehouses", "demand", "lookback_with_current"],
        dtype=np.float32,
    )
    leadtime_history = np.asarray(
        base_env.agent_states["all_warehouses", "vlt", "lookback_with_current"],
        dtype=np.float32,
    )
    return _BenchmarkSnapshot(
        warehouse_names=warehouse_names,
        sku_ids=sku_ids,
        action_mode=str(base_env.action_mode),
        demand_history_by_sku=demand_history,
        leadtime_history_by_sku=leadtime_history,
        demand_mean_by_sku=np.asarray(env.get_demand_mean(), dtype=np.float32),
        leadtime_mean_by_sku=np.asarray(env.get_average_vlt(), dtype=np.float32),
        in_stock_by_sku=np.asarray(env.get_in_stock(), dtype=np.float32),
        in_transit_by_sku=np.asarray(env.get_in_transit(), dtype=np.float32),
    )


def _aggregate_snapshot(snapshot: _BenchmarkSnapshot) -> _AggregatedSnapshot:
    demand_history = tuple(
        float(value)
        for value in snapshot.demand_history_by_sku.mean(axis=(0, 2))
    )
    leadtime_history = tuple(
        float(value)
        for value in snapshot.leadtime_history_by_sku.mean(axis=(0, 2))
    )
    return _AggregatedSnapshot(
        demand_history=demand_history,
        leadtime_history=leadtime_history,
        inventory_mean=float(snapshot.in_stock_by_sku.mean()),
        pipeline_mean=float(snapshot.in_transit_by_sku.mean()),
    )


def _capture_step_outcome(env: object, reward: object) -> _StepOutcome:
    base_env = env.unwrapped
    reward_array = np.asarray(reward, dtype=np.float32)
    sale = np.asarray(base_env.agent_states["all_warehouses", "sale", "yesterday"], dtype=np.float32)
    demand = np.asarray(base_env.agent_states["all_warehouses", "demand", "yesterday"], dtype=np.float32)
    in_stock = np.asarray(base_env.agent_states["all_warehouses", "in_stock", "yesterday"], dtype=np.float32)
    total_demand = float(demand.sum())
    total_sale = float(sale.sum())
    fill_rate = (total_sale / total_demand) if total_demand > 0.0 else None
    return _StepOutcome(
        reward_sum=float(reward_array.sum()),
        fill_rate=fill_rate,
        total_demand=total_demand,
        total_sale=total_sale,
        ending_inventory_total=float(in_stock.sum()),
    )


def _infer_public_regime(
    *,
    demand_value: float,
    leadtime_value: float,
    demand_baseline_value: float,
    leadtime_baseline_value: float,
    previous_regime: RegimeLabel,
) -> RegimeLabel:
    demand_ratio = demand_value / demand_baseline_value if demand_baseline_value > 0.0 else 1.0
    leadtime_ratio = (
        leadtime_value / leadtime_baseline_value if leadtime_baseline_value > 0.0 else 1.0
    )
    if demand_ratio >= 1.15 and leadtime_ratio >= 1.15:
        return RegimeLabel.JOINT_DISRUPTION
    if leadtime_ratio >= 1.15:
        return RegimeLabel.SUPPLY_DISRUPTION
    if demand_ratio >= 1.12:
        return RegimeLabel.DEMAND_REGIME_SHIFT
    if previous_regime is not RegimeLabel.NORMAL and demand_ratio <= 1.02 and leadtime_ratio <= 1.05:
        return RegimeLabel.RECOVERY
    return RegimeLabel.NORMAL


def _solve_skuwise_orders(
    *,
    optimizer: TrustedOptimizerAdapter,
    snapshot: _BenchmarkSnapshot,
    scenario_update_result: ScenarioUpdateResult,
    base_stock_multiplier: float,
    time_index: int,
) -> np.ndarray:
    replenishment_quantities = np.zeros_like(snapshot.demand_mean_by_sku, dtype=np.float32)
    for warehouse_index in range(snapshot.demand_mean_by_sku.shape[0]):
        for sku_index in range(snapshot.demand_mean_by_sku.shape[1]):
            local_demand_mean = float(snapshot.demand_mean_by_sku[warehouse_index, sku_index])
            local_leadtime_mean = float(snapshot.leadtime_mean_by_sku[warehouse_index, sku_index])
            base_stock_level = max(
                1.0,
                local_demand_mean * max(1.0, local_leadtime_mean) * base_stock_multiplier,
            )
            request = OptimizationRequest(
                inventory_level=(float(snapshot.in_stock_by_sku[warehouse_index, sku_index]),),
                planning_horizon=1,
                base_stock_levels=(base_stock_level,),
                scenario_adjustment=scenario_update_result.adjustment,
                pipeline_inventory=(float(snapshot.in_transit_by_sku[warehouse_index, sku_index]),),
                backorder_level=(0.0,),
                scenario_summaries=scenario_update_result.scenarios,
                time_index=time_index,
            )
            result = optimizer.solve(request)
            replenishment_quantities[warehouse_index, sku_index] = float(
                result.replenishment_orders[0]
            )
    return replenishment_quantities


def _read_environment_contract(
    *,
    module: object,
    config: PublicBenchmarkEvalConfig,
) -> PublicBenchmarkActionContract:
    env = getattr(module, "make_env")(
        config.environment_config_name,
        wrapper_names=list(config.wrapper_names),
        mode=config.benchmark_mode,
        vis_path=None,
    )
    try:
        observation = np.asarray(env.reset())
        warehouse_names = tuple(str(name) for name in env.get_warehouse_list())
        sku_count = len(tuple(env.get_sku_list()))
        action_shape = _expected_step_action_shape(env)
        reward_shape = (len(warehouse_names), sku_count)
        action_mode = str(env.unwrapped.action_mode)
        return PublicBenchmarkActionContract(
            environment_config_name=config.environment_config_name,
            wrapper_names=config.wrapper_names,
            benchmark_mode=config.benchmark_mode,
            warehouse_names=warehouse_names,
            sku_count=sku_count,
            action_mode=action_mode,
            advertised_action_shape=tuple(int(item) for item in (env.action_space.shape or ())),
            expected_action_shape=action_shape,
            observation_shape=tuple(int(item) for item in observation.shape),
            reward_shape=reward_shape,
            action_units=_action_units(action_mode),
            reward_semantics=f"{env.unwrapped.config['reward_function']}_sum_over_warehouse_and_sku",
        )
    finally:
        if hasattr(env, "close"):
            env.close()


def _action_units(action_mode: str) -> str:
    if action_mode == "continuous":
        return "direct_replenishment_quantity"
    if action_mode == "demand_mean_continuous":
        return "replenishment_quantity_divided_by_lookback_demand_mean"
    if action_mode == "discrete":
        return "discrete_index_into_action_space"
    if action_mode == "demand_mean_discrete":
        return "discrete_multiplier_index_times_lookback_demand_mean"
    return f"unknown_action_mode:{action_mode}"


def _build_mapping_identity(
    *,
    environment_contract: PublicBenchmarkActionContract,
    comparability_notes: tuple[str, ...],
    mapping_assumptions: tuple[str, ...],
) -> str:
    return hash_jsonable(
        {
            "environment_contract": {
                "environment_config_name": environment_contract.environment_config_name,
                "wrapper_names": environment_contract.wrapper_names,
                "benchmark_mode": environment_contract.benchmark_mode,
                "warehouse_names": environment_contract.warehouse_names,
                "sku_count": environment_contract.sku_count,
                "action_mode": environment_contract.action_mode,
                "advertised_action_shape": environment_contract.advertised_action_shape,
                "expected_action_shape": environment_contract.expected_action_shape,
                "observation_shape": environment_contract.observation_shape,
                "reward_shape": environment_contract.reward_shape,
                "action_units": environment_contract.action_units,
                "reward_semantics": environment_contract.reward_semantics,
            },
            "comparability_notes": comparability_notes,
            "mapping_assumptions": mapping_assumptions,
        }
    )


def _expected_step_action_shape(env: object) -> tuple[int, ...]:
    warehouse_count = len(tuple(env.get_warehouse_list()))
    sku_count = len(tuple(env.get_sku_list()))
    return (warehouse_count, sku_count)


def _build_public_benchmark_mission() -> MissionSpec:
    return MissionSpec(
        mission_id="public_benchmark_eval",
        objective=(
            "Inspect bounded benchmark evidence, manage uncertainty conservatively, "
            "and keep replenishment orders inside the trusted optimizer boundary."
        ),
        admissible_tool_ids=("forecast_tool", "leadtime_tool", "scenario_tool"),
    )


def _build_runtime(
    agent_config: AgentConfig,
    *,
    mode: str,
) -> tuple[OrchestrationRuntime, str | None]:
    from meio.agents.llm_orchestrator import LLMOrchestrator

    tools = (
        DeterministicForecastTool(),
        DeterministicLeadTimeTool(),
        DeterministicScenarioTool(),
    )
    if mode == "deterministic_orchestrator":
        return OrchestrationRuntime(agent_config=agent_config, tools=tools), None
    if mode == "llm_orchestrator":
        client_mode = (
            "real" if agent_config.llm_client_mode == "openai" else agent_config.llm_client_mode
        )
        if client_mode == "real":
            client = OpenAILLMClient(
                provider=agent_config.llm_provider,
                request_timeout_s=agent_config.llm_request_timeout_s,
                max_retries=agent_config.llm_max_retries,
            )
            client.ensure_available()
            llm_provider = agent_config.llm_provider
        else:
            client = FakeLLMClient()
            llm_provider = client.provider
        llm_orchestrator = LLMOrchestrator(client=client, agent_config=agent_config)
        return (
            OrchestrationRuntime(
                agent_config=agent_config,
                tools=tools,
                mode=RuntimeMode.LLM_ORCHESTRATION,
                llm_orchestrator=llm_orchestrator,
            ),
            llm_provider,
        )
    raise ValueError(f"Unsupported public benchmark mode: {mode!r}.")


def _extract_scenario_update_result(response: OrchestrationResponse) -> ScenarioUpdateResult | None:
    for tool_result in reversed(response.tool_results):
        value = tool_result.structured_output.get("scenario_update_result")
        if isinstance(value, ScenarioUpdateResult):
            return value
    return None


def _default_keep_current_update(
    regime_label: RegimeLabel,
    observation: Observation,
    provenance: str,
) -> ScenarioUpdateResult:
    return ScenarioUpdateResult(
        scenarios=(
            ScenarioSummary(
                scenario_id=f"{regime_label.value}_keep_current",
                regime_label=regime_label,
                weight=1.0,
                demand_scale=1.0,
                leadtime_scale=1.0,
            ),
        ),
        applied_update_types=(UpdateRequestType.KEEP_CURRENT,),
        adjustment=ScenarioAdjustmentSummary(
            demand_outlook=observation.demand_realization[-1],
            leadtime_outlook=observation.leadtime_realization[-1],
            safety_buffer_scale=1.0,
        ),
        request_replan=False,
        provenance=provenance,
    )


def _optimizer_boundary_preserved(
    responses: tuple[OrchestrationResponse, ...],
) -> bool:
    for response in responses:
        for tool_result in response.tool_results:
            if tool_result.emits_raw_orders:
                return False
            if any(
                isinstance(value, OptimizationResult)
                for value in tool_result.structured_output.values()
            ):
                return False
    return True


def _build_llm_call_trace_records(
    *,
    episode_id: str,
    mode: str,
    run_seed: int,
    response: OrchestrationResponse,
    schedule_name: str,
    period_index: int,
) -> tuple[LLMCallTraceRecord, ...]:
    diagnostics = response.llm_diagnostics
    if diagnostics is None or diagnostics.llm_call_trace is None:
        return ()
    trace = diagnostics.llm_call_trace
    return (
        LLMCallTraceRecord(
            episode_id=episode_id,
            mode=mode,
            tool_ablation_variant="full",
            schedule_name=schedule_name,
            run_seed=run_seed,
            period_index=period_index,
            call_index=0,
            provider=trace.provider,
            model_name=trace.model_name,
            prompt_version=trace.prompt_version,
            prompt_text=trace.prompt_text,
            prompt_hash=trace.prompt_hash,
            raw_output_text=trace.raw_output_text,
            parsed_output=trace.parsed_output,
            validation_success=trace.validation_success,
            invalid_output=trace.invalid_output,
            fallback_used=trace.fallback_used,
            fallback_reason=trace.fallback_reason,
            prompt_tokens=trace.prompt_tokens,
            completion_tokens=trace.completion_tokens,
            total_tokens=trace.total_tokens,
            latency_ms=trace.latency_ms,
            requested_tool_ids=trace.requested_tool_ids,
            unavailable_tool_ids=trace.unavailable_tool_ids,
            retry_count=trace.retry_count,
            client_error_category=(
                trace.client_error_category.value
                if trace.client_error_category is not None
                else None
            ),
            client_error_message=trace.client_error_message,
            failure_after_response=trace.failure_after_response,
            validation_lane=VALIDATION_LANE,
        ),
    )


def _build_tool_call_trace_records(
    *,
    episode_id: str,
    mode: str,
    run_seed: int,
    response: OrchestrationResponse,
    schedule_name: str,
    period_index: int,
) -> tuple[ToolCallTraceRecord, ...]:
    return tuple(
        ToolCallTraceRecord(
            episode_id=episode_id,
            mode=mode,
            tool_ablation_variant="full",
            schedule_name=schedule_name,
            run_seed=run_seed,
            period_index=period_index,
            call_index=call_index,
            tool_id=trace.tool_id,
            tool_input=trace.tool_input,
            tool_output=trace.tool_output,
            success=trace.success,
            error_type=trace.error_type,
            latency_ms=trace.latency_ms,
            pre_tool_decision=trace.pre_tool_decision,
            post_tool_decision=trace.post_tool_decision,
            pre_tool_optimizer_input=trace.pre_tool_optimizer_input,
            post_tool_optimizer_input=trace.post_tool_optimizer_input,
            decision_changed=trace.decision_changed,
            optimizer_input_changed=trace.optimizer_input_changed,
            validation_lane=VALIDATION_LANE,
        )
        for call_index, trace in enumerate(response.tool_call_traces)
    )


def _missing_required_paths(package_root: Path) -> tuple[str, ...]:
    return tuple(
        relative_path
        for relative_path in _REPLENISHMENTENV_REQUIRED_PATHS
        if not (package_root / relative_path).exists()
    )


def _read_demo_summary(
    package_root: Path,
    demo_config_path: Path,
) -> ReplenishmentEnvDemoSummary:
    payload = yaml.safe_load(demo_config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("demo config must parse into a mapping.")
    env_payload = payload.get("env", {})
    warehouse_payload = payload.get("warehouse", [])
    mode_names = tuple(
        str(item.get("name"))
        for item in env_payload.get("mode", [])
        if isinstance(item, dict) and item.get("name") is not None
    )
    warehouse_names = tuple(
        str(item.get("name"))
        for item in warehouse_payload
        if isinstance(item, dict) and item.get("name") is not None
    )
    referenced_files = _referenced_demo_files(payload)
    missing_files = tuple(
        relative_path
        for relative_path in referenced_files
        if not (package_root / relative_path).exists()
    )
    return ReplenishmentEnvDemoSummary(
        demo_config_path=str(demo_config_path),
        mode_names=mode_names,
        warehouse_names=warehouse_names,
        lookback_len=_optional_int(env_payload.get("lookback_len")),
        horizon=_optional_int(env_payload.get("horizon")),
        referenced_files=referenced_files,
        missing_files=missing_files,
    )


def _referenced_demo_files(payload: dict[str, Any]) -> tuple[str, ...]:
    referenced: list[str] = []
    env_payload = payload.get("env", {})
    sku_list_value = env_payload.get("sku_list")
    if isinstance(sku_list_value, str) and sku_list_value.strip():
        referenced.append(_trim_replenishmentenv_prefix(sku_list_value))
    for warehouse_payload in payload.get("warehouse", []):
        if not isinstance(warehouse_payload, dict):
            continue
        sku_payload = warehouse_payload.get("sku", {})
        if not isinstance(sku_payload, dict):
            continue
        static_path = sku_payload.get("static_data")
        if isinstance(static_path, str) and static_path.strip():
            referenced.append(_trim_replenishmentenv_prefix(static_path))
        for dynamic_item in sku_payload.get("dynamic_data", []):
            if not isinstance(dynamic_item, dict):
                continue
            file_path = dynamic_item.get("file")
            if isinstance(file_path, str) and file_path.strip():
                referenced.append(_trim_replenishmentenv_prefix(file_path))
    deduped: list[str] = []
    for item in referenced:
        if item not in deduped:
            deduped.append(item)
    return tuple(deduped)


def _trim_replenishmentenv_prefix(value: str) -> str:
    normalized = value.replace("\\", "/").strip()
    prefix = "ReplenishmentEnv/"
    if normalized.startswith(prefix):
        return normalized[len(prefix) :]
    return normalized


def _is_local_source_checkout(package_root: Path) -> bool:
    try:
        package_root.resolve().relative_to(_LOCAL_REPLENISHMENTENV_ROOT.resolve())
    except ValueError:
        return False
    return True


def _source_checkout_root(package_root: Path) -> Path:
    return package_root.parent if package_root.name == "ReplenishmentEnv" else package_root


def _detect_packaging_root_cause(package_root: Path) -> str | None:
    source_root = _source_checkout_root(package_root)
    patch_notes_path = source_root / "LOCAL_PATCH_NOTES.md"
    if patch_notes_path.exists():
        return _SETUP_ROOT_CAUSE
    setup_path = source_root / "setup.py"
    if not setup_path.exists():
        return None
    setup_text = setup_path.read_text(encoding="utf-8", errors="ignore")
    if 'find_packages(include=["ReplenishmentEnv"])' in setup_text:
        return _SETUP_ROOT_CAUSE
    return None


@contextmanager
def _temporary_import_root(package_root: Path) -> Iterator[None]:
    import_root = str(_source_checkout_root(package_root))
    inserted = import_root not in sys.path
    if inserted:
        sys.path.insert(0, import_root)
    try:
        yield
    finally:
        if inserted and import_root in sys.path:
            sys.path.remove(import_root)


def _import_module_from_package_root(*, module_name: str, package_root: Path) -> object:
    invalidate_caches()
    _clear_module_cache(module_name)
    with _temporary_import_root(package_root):
        return import_module(module_name)


def _clear_module_cache(module_name: str) -> None:
    keys = tuple(
        key for key in sys.modules if key == module_name or key.startswith(f"{module_name}.")
    )
    for key in keys:
        sys.modules.pop(key, None)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("Expected an integer value when present.")
    return value


__all__ = [
    "PublicBenchmarkActionContract",
    "PublicBenchmarkAdapterSummary",
    "PublicBenchmarkEvaluationSummary",
    "PublicBenchmarkExecutionBatch",
    "PublicBenchmarkModeArtifacts",
    "PublicBenchmarkModeSummary",
    "PublicBenchmarkSmokeSummary",
    "ReplenishmentEnvDemoSummary",
    "VALIDATION_LANE",
    "execute_replenishmentenv_smoke",
    "inspect_replenishmentenv_installation",
    "locate_package_root",
    "map_optimizer_orders_to_public_benchmark_actions",
    "run_public_benchmark_execution",
]
