"""Typed configuration schemas for the first MEIO milestone."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from meio.contracts import (
    BackorderPolicy,
    BenchmarkFamily,
    RegimeLabel,
    ToolClass,
    UpdateRequestType,
)

ALLOWED_RUNTIME_MODES = (
    "deterministic_baseline",
    "deterministic_orchestrator",
    "llm_orchestrator",
    "llm_orchestrator_internal_only",
    "llm_orchestrator_with_external_evidence",
)
DEFAULT_RUNTIME_MODE_SET = (
    "deterministic_baseline",
    "deterministic_orchestrator",
    "llm_orchestrator",
)
ALLOWED_TOOL_ABLATION_VARIANTS = (
    "full",
    "no_forecast_tool",
    "no_leadtime_tool",
    "no_scenario_tool",
)
ALLOWED_EXTERNAL_EVIDENCE_SOURCES = (
    "semi_synthetic",
)
ALLOWED_VALIDATION_LANES = (
    "stockpyl_internal",
    "public_benchmark",
    "real_demand_backtest",
    "official_event_replay",
)


def _validate_probability(value: float, field_name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be within [0.0, 1.0].")


def _validate_positive_int(value: int, field_name: str) -> None:
    if value <= 0:
        raise ValueError(f"{field_name} must be positive.")


def _validate_non_negative_int(value: int, field_name: str) -> None:
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative.")


def _validate_non_negative_number(value: float, field_name: str) -> None:
    if value < 0.0:
        raise ValueError(f"{field_name} must be non-negative.")


def _validate_non_empty_text(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty.")


def _validate_choice(value: str, field_name: str, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ValueError(f"{field_name} must be one of {allowed}.")


@dataclass(frozen=True, slots=True)
class SerialStageConfig:
    """Boundary configuration for a single serial stage."""

    stage_index: int
    stage_name: str
    initial_inventory: int = 0
    shipment_lead_time: int = 2
    base_stock_level: int | None = None

    def __post_init__(self) -> None:
        _validate_positive_int(self.stage_index, "stage_index")
        _validate_non_empty_text(self.stage_name, "stage_name")
        _validate_non_negative_int(self.initial_inventory, "initial_inventory")
        _validate_positive_int(self.shipment_lead_time, "shipment_lead_time")
        if self.base_stock_level is not None:
            _validate_non_negative_int(self.base_stock_level, "base_stock_level")


@dataclass(frozen=True, slots=True)
class SerialSystemConfig:
    """Configuration for a serial multi-echelon system."""

    topology: str = "serial"
    echelon_count: int = 3
    stages: tuple[SerialStageConfig, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _validate_positive_int(self.echelon_count, "echelon_count")
        if self.topology != "serial":
            raise ValueError("Only serial topology is supported in this milestone.")
        object.__setattr__(self, "stages", tuple(self.stages))
        if self.stages and len(self.stages) != self.echelon_count:
            raise ValueError("stages must match echelon_count when provided.")
        if self.stages:
            expected_indices = tuple(range(1, self.echelon_count + 1))
            actual_indices = tuple(stage.stage_index for stage in self.stages)
            if actual_indices != expected_indices:
                raise ValueError("stage_index values must be consecutive starting at 1.")


@dataclass(frozen=True, slots=True)
class CostConfig:
    """Cost inputs for the benchmark boundary."""

    holding_cost: float
    backorder_cost: float
    ordering_cost: float = 0.0

    def __post_init__(self) -> None:
        _validate_non_negative_number(self.holding_cost, "holding_cost")
        _validate_non_negative_number(self.backorder_cost, "backorder_cost")
        _validate_non_negative_number(self.ordering_cost, "ordering_cost")


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Benchmark configuration for the first serial milestone."""

    benchmark_family: BenchmarkFamily
    system: SerialSystemConfig
    costs: CostConfig
    service_model: BackorderPolicy = BackorderPolicy.BACKORDERS
    scenario_families: tuple[str, ...] = (RegimeLabel.NORMAL.value,)
    random_seed: int = 0
    demand_mean: float = 10.0

    @property
    def topology(self) -> str:
        return self.system.topology

    @property
    def echelon_count(self) -> int:
        return self.system.echelon_count

    def __post_init__(self) -> None:
        if not isinstance(self.benchmark_family, BenchmarkFamily):
            raise TypeError("benchmark_family must be a BenchmarkFamily.")
        if not isinstance(self.service_model, BackorderPolicy):
            raise TypeError("service_model must be a BackorderPolicy.")
        object.__setattr__(self, "scenario_families", tuple(self.scenario_families))
        if not self.scenario_families:
            raise ValueError("scenario_families must contain at least one identifier.")
        for scenario_family in self.scenario_families:
            _validate_non_empty_text(scenario_family, "scenario_families entries")
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative.")
        _validate_non_negative_number(self.demand_mean, "demand_mean")


@dataclass(frozen=True, slots=True)
class RegimeScheduleConfig:
    """Named fixed regime schedule for controlled benchmark evaluation."""

    name: str
    labels: tuple[RegimeLabel, ...]

    def __post_init__(self) -> None:
        _validate_non_empty_text(self.name, "name")
        object.__setattr__(self, "labels", tuple(self.labels))
        if not self.labels:
            raise ValueError("labels must not be empty.")
        for label in self.labels:
            if not isinstance(label, RegimeLabel):
                raise TypeError("labels must contain RegimeLabel values.")

    @property
    def rollout_horizon(self) -> int:
        """Return the fixed horizon implied by the schedule."""

        return len(self.labels)


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Configuration for a reproducible experiment run."""

    experiment_name: str
    benchmark_config_path: Path
    agent_config_path: Path | None = None
    episode_count: int = 1
    rollout_horizon: int | None = None
    regime_schedule: tuple[RegimeLabel, ...] = (RegimeLabel.NORMAL,)
    regime_schedules: tuple[RegimeScheduleConfig, ...] = field(default_factory=tuple)
    seed_set: tuple[int, ...] = field(default_factory=tuple)
    mode_set: tuple[str, ...] = field(default_factory=lambda: DEFAULT_RUNTIME_MODE_SET)
    tool_ablation_variants: tuple[str, ...] = field(default_factory=lambda: ("full",))
    semi_synthetic_external_evidence: bool = False
    external_evidence_source: str | None = None
    results_dir: Path = Path("results")

    def __post_init__(self) -> None:
        _validate_non_empty_text(self.experiment_name, "experiment_name")
        _validate_positive_int(self.episode_count, "episode_count")
        object.__setattr__(self, "regime_schedule", tuple(self.regime_schedule))
        object.__setattr__(self, "regime_schedules", tuple(self.regime_schedules))
        object.__setattr__(self, "seed_set", tuple(self.seed_set))
        object.__setattr__(self, "mode_set", tuple(self.mode_set))
        object.__setattr__(
            self,
            "tool_ablation_variants",
            tuple(self.tool_ablation_variants),
        )
        if self.rollout_horizon is not None:
            _validate_positive_int(self.rollout_horizon, "rollout_horizon")
        if not self.regime_schedules and not self.regime_schedule:
            raise ValueError("At least one regime schedule must be configured.")
        for regime_label in self.regime_schedule:
            if not isinstance(regime_label, RegimeLabel):
                raise TypeError("regime_schedule must contain RegimeLabel values.")
        for schedule in self.regime_schedules:
            if not isinstance(schedule, RegimeScheduleConfig):
                raise TypeError("regime_schedules must contain RegimeScheduleConfig values.")
        schedule_names = tuple(schedule.name for schedule in self.regime_schedules)
        if len(schedule_names) != len(set(schedule_names)):
            raise ValueError("regime_schedules must use unique names.")
        resolved_schedule_set = self.resolved_schedule_set()
        if self.rollout_horizon is not None:
            if any(schedule.rollout_horizon != self.rollout_horizon for schedule in resolved_schedule_set):
                raise ValueError("All configured schedules must match rollout_horizon.")
        for seed in self.seed_set:
            _validate_non_negative_int(seed, "seed_set")
        if self.seed_set and len(self.seed_set) != self.episode_count:
            raise ValueError("seed_set must match episode_count when provided.")
        if not self.mode_set:
            raise ValueError("mode_set must not be empty.")
        for mode in self.mode_set:
            _validate_choice(mode, "mode_set", ALLOWED_RUNTIME_MODES)
        if not self.tool_ablation_variants:
            raise ValueError("tool_ablation_variants must not be empty.")
        for variant in self.tool_ablation_variants:
            _validate_choice(
                variant,
                "tool_ablation_variants",
                ALLOWED_TOOL_ABLATION_VARIANTS,
            )
        if self.external_evidence_source is not None:
            _validate_choice(
                self.external_evidence_source,
                "external_evidence_source",
                ALLOWED_EXTERNAL_EVIDENCE_SOURCES,
            )
        resolved_evidence_source = self.resolved_external_evidence_source()
        if resolved_evidence_source not in {None, "semi_synthetic"}:
            raise ValueError(
                "Only semi_synthetic external evidence is supported after cleanup."
            )
        if not isinstance(self.results_dir, Path):
            raise TypeError("results_dir must be a Path.")

    def resolved_schedule_set(self) -> tuple[RegimeScheduleConfig, ...]:
        """Return the explicit schedule set for the experiment."""

        if self.regime_schedules:
            return self.regime_schedules
        return (RegimeScheduleConfig(name="default", labels=self.regime_schedule),)

    def resolved_rollout_horizon(self) -> int:
        """Return the rollout horizon implied by the schedule configuration."""

        if self.rollout_horizon is not None:
            return self.rollout_horizon
        return self.resolved_schedule_set()[0].rollout_horizon

    def resolved_seed_set(self, default_seed: int) -> tuple[int, ...]:
        """Return the explicit seed set for the experiment."""

        if self.seed_set:
            return self.seed_set
        return tuple(default_seed + offset for offset in range(self.episode_count))

    def resolved_mode_set(self) -> tuple[str, ...]:
        """Return the explicit mode set for the experiment."""

        return self.mode_set

    def resolved_tool_ablation_variants(self) -> tuple[str, ...]:
        """Return the explicit tool-ablation variants for the experiment."""

        return self.tool_ablation_variants

    def resolved_external_evidence_source(self) -> str | None:
        """Return the explicit external evidence source for this experiment."""

        if self.external_evidence_source is not None:
            return self.external_evidence_source
        if self.semi_synthetic_external_evidence:
            return "semi_synthetic"
        return None


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """Bounded orchestration-agent configuration with no raw control logic."""

    enabled_regime_labels: tuple[RegimeLabel, ...] = field(default_factory=lambda: tuple(RegimeLabel))
    allowed_update_types: tuple[UpdateRequestType, ...] = field(
        default_factory=lambda: tuple(UpdateRequestType)
    )
    allowed_tool_classes: tuple[ToolClass, ...] = field(default_factory=lambda: tuple(ToolClass))
    minimum_confidence: float = 0.0
    max_tool_steps: int = 3
    allow_replan_requests: bool = True
    allow_abstain: bool = True
    llm_provider: str = "openai"
    llm_client_mode: str = "fake"
    llm_model_name: str = "gpt-4o-mini"
    llm_temperature: float | None = None
    llm_request_timeout_s: float = 20.0
    llm_max_retries: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled_regime_labels", tuple(self.enabled_regime_labels))
        object.__setattr__(self, "allowed_update_types", tuple(self.allowed_update_types))
        object.__setattr__(self, "allowed_tool_classes", tuple(self.allowed_tool_classes))
        if not self.enabled_regime_labels:
            raise ValueError("enabled_regime_labels must not be empty.")
        if not self.allowed_update_types:
            raise ValueError("allowed_update_types must not be empty.")
        if not self.allowed_tool_classes:
            raise ValueError("allowed_tool_classes must not be empty.")
        for label in self.enabled_regime_labels:
            if not isinstance(label, RegimeLabel):
                raise TypeError("enabled_regime_labels must contain RegimeLabel values.")
        for update_type in self.allowed_update_types:
            if not isinstance(update_type, UpdateRequestType):
                raise TypeError("allowed_update_types must contain UpdateRequestType values.")
        for tool_class in self.allowed_tool_classes:
            if not isinstance(tool_class, ToolClass):
                raise TypeError("allowed_tool_classes must contain ToolClass values.")
        _validate_probability(self.minimum_confidence, "minimum_confidence")
        _validate_positive_int(self.max_tool_steps, "max_tool_steps")
        _validate_choice(self.llm_provider, "llm_provider", ("openai",))
        _validate_choice(self.llm_client_mode, "llm_client_mode", ("fake", "real", "openai"))
        _validate_non_empty_text(self.llm_model_name, "llm_model_name")
        if self.llm_temperature is not None:
            if not 0.0 <= self.llm_temperature <= 2.0:
                raise ValueError("llm_temperature must be within [0.0, 2.0] when provided.")
        if self.llm_request_timeout_s <= 0.0:
            raise ValueError("llm_request_timeout_s must be positive.")
        _validate_non_negative_int(self.llm_max_retries, "llm_max_retries")


@dataclass(frozen=True, slots=True)
class PublicBenchmarkEvalConfig:
    """Configuration for public-benchmark integration checks."""

    experiment_name: str
    benchmark_candidate: str
    discovery_module: str = "ReplenishmentEnv"
    benchmark_root: Path | None = None
    demo_config_path: Path = Path("config/demo.yml")
    agent_config_path: Path = Path("configs/agent/base.toml")
    environment_config_name: str = "sku100.single_store.standard"
    wrapper_names: tuple[str, ...] = ("DefaultWrapper",)
    benchmark_mode: str = "test"
    smoke_horizon_steps: int = 1
    mode_set: tuple[str, ...] = field(default_factory=lambda: DEFAULT_RUNTIME_MODE_SET)
    episode_horizon_steps: int = 10
    base_stock_multiplier: float = 1.0
    demand_scale_epsilon: float = 1e-6
    results_dir: Path = Path("results/public_benchmark_eval")

    def __post_init__(self) -> None:
        _validate_non_empty_text(self.experiment_name, "experiment_name")
        _validate_non_empty_text(self.benchmark_candidate, "benchmark_candidate")
        _validate_non_empty_text(self.discovery_module, "discovery_module")
        if self.benchmark_root is not None and not isinstance(self.benchmark_root, Path):
            raise TypeError("benchmark_root must be a Path when provided.")
        if not isinstance(self.demo_config_path, Path):
            raise TypeError("demo_config_path must be a Path.")
        if not isinstance(self.agent_config_path, Path):
            raise TypeError("agent_config_path must be a Path.")
        _validate_non_empty_text(self.environment_config_name, "environment_config_name")
        object.__setattr__(self, "wrapper_names", tuple(self.wrapper_names))
        if not self.wrapper_names:
            raise ValueError("wrapper_names must not be empty.")
        for wrapper_name in self.wrapper_names:
            _validate_non_empty_text(wrapper_name, "wrapper_names")
        _validate_non_empty_text(self.benchmark_mode, "benchmark_mode")
        _validate_positive_int(self.smoke_horizon_steps, "smoke_horizon_steps")
        object.__setattr__(self, "mode_set", tuple(self.mode_set))
        if not self.mode_set:
            raise ValueError("mode_set must not be empty.")
        for mode in self.mode_set:
            _validate_choice(mode, "mode_set", ALLOWED_RUNTIME_MODES)
        _validate_positive_int(self.episode_horizon_steps, "episode_horizon_steps")
        _validate_non_negative_number(self.base_stock_multiplier, "base_stock_multiplier")
        if self.base_stock_multiplier <= 0.0:
            raise ValueError("base_stock_multiplier must be positive.")
        _validate_non_negative_number(self.demand_scale_epsilon, "demand_scale_epsilon")
        if self.demand_scale_epsilon <= 0.0:
            raise ValueError("demand_scale_epsilon must be positive.")
        if not isinstance(self.results_dir, Path):
            raise TypeError("results_dir must be a Path.")


@dataclass(frozen=True, slots=True)
class RealDemandBacktestSliceConfig:
    """One explicit slice inside a bounded real-demand validation panel."""

    name: str
    dataset_name: str
    demand_csv_path: Path
    leadtime_csv_path: Path
    sku_list_path: Path
    selected_skus: tuple[str, ...] = field(default_factory=tuple)
    selected_sku_count: int = 5
    subset_selection: str = "nearest_benchmark_mean"
    training_window_days: int = 180
    history_window_days: int = 28
    forecast_update_window_days: int = 1
    evaluation_horizon_days: int = 60
    evaluation_start_date: str | None = None
    roll_forward_stride_days: int = 1

    def __post_init__(self) -> None:
        _validate_non_empty_text(self.name, "name")
        _validate_non_empty_text(self.dataset_name, "dataset_name")
        _validate_non_empty_text(self.subset_selection, "subset_selection")
        for field_name in ("demand_csv_path", "leadtime_csv_path", "sku_list_path"):
            if not isinstance(getattr(self, field_name), Path):
                raise TypeError(f"{field_name} must be a Path.")
        object.__setattr__(self, "selected_skus", tuple(self.selected_skus))
        for sku in self.selected_skus:
            _validate_non_empty_text(sku, "selected_skus")
        for field_name in (
            "selected_sku_count",
            "training_window_days",
            "history_window_days",
            "forecast_update_window_days",
            "evaluation_horizon_days",
            "roll_forward_stride_days",
        ):
            _validate_positive_int(getattr(self, field_name), field_name)
        if self.evaluation_start_date is not None:
            _validate_non_empty_text(self.evaluation_start_date, "evaluation_start_date")


@dataclass(frozen=True, slots=True)
class RealDemandBacktestConfig:
    """Configuration for bounded real-demand backtesting."""

    experiment_name: str
    benchmark_config_path: Path
    agent_config_path: Path
    dataset_name: str
    slice_name: str | None = None
    discovery_module: str = "ReplenishmentEnv"
    dataset_root: Path | None = None
    demand_csv_path: Path = Path("data/sku2778/sku2778.demand.csv")
    leadtime_csv_path: Path = Path("data/sku2778/sku2778.store2.dynamic_vlt.csv")
    sku_list_path: Path = Path("data/sku2778/sku50.sku_list.csv")
    selected_skus: tuple[str, ...] = field(default_factory=tuple)
    selected_sku_count: int = 5
    subset_selection: str = "nearest_benchmark_mean"
    training_window_days: int = 180
    history_window_days: int = 28
    forecast_update_window_days: int = 1
    evaluation_horizon_days: int = 60
    evaluation_start_date: str | None = None
    roll_forward_stride_days: int = 1
    mode_set: tuple[str, ...] = field(default_factory=lambda: DEFAULT_RUNTIME_MODE_SET)
    results_dir: Path = Path("results/real_demand_backtest")

    def __post_init__(self) -> None:
        _validate_non_empty_text(self.experiment_name, "experiment_name")
        _validate_non_empty_text(self.dataset_name, "dataset_name")
        _validate_non_empty_text(self.discovery_module, "discovery_module")
        _validate_non_empty_text(self.subset_selection, "subset_selection")
        if self.slice_name is not None:
            _validate_non_empty_text(self.slice_name, "slice_name")
        if self.evaluation_start_date is not None:
            _validate_non_empty_text(self.evaluation_start_date, "evaluation_start_date")
        if self.dataset_root is not None and not isinstance(self.dataset_root, Path):
            raise TypeError("dataset_root must be a Path when provided.")
        for field_name in (
            "benchmark_config_path",
            "agent_config_path",
            "demand_csv_path",
            "leadtime_csv_path",
            "sku_list_path",
            "results_dir",
        ):
            if not isinstance(getattr(self, field_name), Path):
                raise TypeError(f"{field_name} must be a Path.")
        object.__setattr__(self, "selected_skus", tuple(self.selected_skus))
        for sku in self.selected_skus:
            _validate_non_empty_text(sku, "selected_skus")
        for field_name in (
            "selected_sku_count",
            "training_window_days",
            "history_window_days",
            "forecast_update_window_days",
            "evaluation_horizon_days",
            "roll_forward_stride_days",
        ):
            _validate_positive_int(getattr(self, field_name), field_name)
        object.__setattr__(self, "mode_set", tuple(self.mode_set))
        if not self.mode_set:
            raise ValueError("mode_set must not be empty.")
        for mode in self.mode_set:
            _validate_choice(mode, "mode_set", ALLOWED_RUNTIME_MODES)


@dataclass(frozen=True, slots=True)
class RealDemandBacktestPanelConfig:
    """Configuration for a fixed panel of explicit real-demand slices."""

    experiment_name: str
    benchmark_config_path: Path
    agent_config_path: Path
    discovery_module: str = "ReplenishmentEnv"
    dataset_root: Path | None = None
    mode_set: tuple[str, ...] = field(default_factory=lambda: DEFAULT_RUNTIME_MODE_SET)
    results_dir: Path = Path("results/real_demand_backtest")
    slices: tuple[RealDemandBacktestSliceConfig, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _validate_non_empty_text(self.experiment_name, "experiment_name")
        _validate_non_empty_text(self.discovery_module, "discovery_module")
        if self.dataset_root is not None and not isinstance(self.dataset_root, Path):
            raise TypeError("dataset_root must be a Path when provided.")
        for field_name in ("benchmark_config_path", "agent_config_path", "results_dir"):
            if not isinstance(getattr(self, field_name), Path):
                raise TypeError(f"{field_name} must be a Path.")
        object.__setattr__(self, "mode_set", tuple(self.mode_set))
        object.__setattr__(self, "slices", tuple(self.slices))
        if not self.mode_set:
            raise ValueError("mode_set must not be empty.")
        if not self.slices:
            raise ValueError("slices must not be empty.")
        for mode in self.mode_set:
            _validate_choice(mode, "mode_set", ALLOWED_RUNTIME_MODES)
        slice_names = tuple(slice_config.name for slice_config in self.slices)
        if len(slice_names) != len(set(slice_names)):
            raise ValueError("slices must use unique names.")
        for slice_config in self.slices:
            if not isinstance(slice_config, RealDemandBacktestSliceConfig):
                raise TypeError("slices must contain RealDemandBacktestSliceConfig values.")

    def resolved_slice_configs(self) -> tuple[RealDemandBacktestConfig, ...]:
        """Project the panel into explicit per-slice backtest configs."""

        return tuple(
            RealDemandBacktestConfig(
                experiment_name=self.experiment_name,
                benchmark_config_path=self.benchmark_config_path,
                agent_config_path=self.agent_config_path,
                dataset_name=slice_config.dataset_name,
                slice_name=slice_config.name,
                discovery_module=self.discovery_module,
                dataset_root=self.dataset_root,
                demand_csv_path=slice_config.demand_csv_path,
                leadtime_csv_path=slice_config.leadtime_csv_path,
                sku_list_path=slice_config.sku_list_path,
                selected_skus=slice_config.selected_skus,
                selected_sku_count=slice_config.selected_sku_count,
                subset_selection=slice_config.subset_selection,
                training_window_days=slice_config.training_window_days,
                history_window_days=slice_config.history_window_days,
                forecast_update_window_days=slice_config.forecast_update_window_days,
                evaluation_horizon_days=slice_config.evaluation_horizon_days,
                evaluation_start_date=slice_config.evaluation_start_date,
                roll_forward_stride_days=slice_config.roll_forward_stride_days,
                mode_set=self.mode_set,
                results_dir=self.results_dir,
            )
            for slice_config in self.slices
        )


__all__ = [
    "ALLOWED_RUNTIME_MODES",
    "DEFAULT_RUNTIME_MODE_SET",
    "ALLOWED_TOOL_ABLATION_VARIANTS",
    "ALLOWED_VALIDATION_LANES",
    "AgentConfig",
    "BenchmarkConfig",
    "CostConfig",
    "ExperimentConfig",
    "PublicBenchmarkEvalConfig",
    "RealDemandBacktestConfig",
    "RealDemandBacktestPanelConfig",
    "RealDemandBacktestSliceConfig",
    "RegimeScheduleConfig",
    "SerialStageConfig",
    "SerialSystemConfig",
]
