"""TOML loaders for the first MEIO milestone configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import tomllib

from meio.config.schemas import (
    ALLOWED_RUNTIME_MODES,
    AgentConfig,
    BenchmarkConfig,
    CostConfig,
    DEFAULT_RUNTIME_MODE_SET,
    ExperimentConfig,
    PublicBenchmarkEvalConfig,
    RealDemandBacktestConfig,
    RealDemandBacktestPanelConfig,
    RealDemandBacktestSliceConfig,
    RegimeScheduleConfig,
    SerialStageConfig,
    SerialSystemConfig,
)
from meio.contracts import (
    BackorderPolicy,
    BenchmarkFamily,
    RegimeLabel,
    ToolClass,
    UpdateRequestType,
)
from meio.utils.env import load_env_value


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    """Load a benchmark TOML file into a typed benchmark config."""

    document = _load_toml_document(path)
    benchmark_table = _require_table(document, "benchmark")
    system_table = _require_table(document, "system")
    costs_table = _require_table(document, "costs")
    stage_tables = _optional_list(document, "stages")

    stages = tuple(
        SerialStageConfig(
            stage_index=_require_int(stage_table, "stage_index", "stages"),
            stage_name=_require_string(stage_table, "stage_name", "stages"),
            initial_inventory=_optional_int(stage_table, "initial_inventory", "stages", default=0),
            shipment_lead_time=_optional_int(
                stage_table,
                "shipment_lead_time",
                "stages",
                default=2,
            ),
            base_stock_level=_optional_int_or_none(stage_table, "base_stock_level", "stages"),
        )
        for stage_table in stage_tables
    )

    return BenchmarkConfig(
        benchmark_family=_parse_enum(
            BenchmarkFamily,
            _require_string(benchmark_table, "family", "benchmark"),
            "benchmark.family",
        ),
        system=SerialSystemConfig(
            topology=_require_string(system_table, "topology", "system"),
            echelon_count=_require_int(system_table, "echelon_count", "system"),
            stages=stages,
        ),
        costs=CostConfig(
            holding_cost=_require_number(costs_table, "holding_cost", "costs"),
            backorder_cost=_require_number(costs_table, "backorder_cost", "costs"),
            ordering_cost=_optional_number(costs_table, "ordering_cost", "costs", default=0.0),
        ),
        service_model=_parse_enum(
            BackorderPolicy,
            _require_string(benchmark_table, "service_model", "benchmark"),
            "benchmark.service_model",
        ),
        scenario_families=tuple(
            _require_string_list(benchmark_table, "scenario_families", "benchmark")
        ),
        random_seed=_require_int(benchmark_table, "random_seed", "benchmark"),
        demand_mean=_optional_number(benchmark_table, "demand_mean", "benchmark", default=10.0),
    )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment TOML file into a typed experiment config."""

    document = _load_toml_document(path)
    experiment_table = _require_table(document, "experiment")
    regime_schedule_tables = _optional_list(document, "regime_schedules")

    agent_path = experiment_table.get("agent_config")
    if agent_path is not None and not isinstance(agent_path, str):
        raise ValueError("experiment.agent_config must be a string when provided.")

    return ExperimentConfig(
        experiment_name=_require_string(experiment_table, "name", "experiment"),
        benchmark_config_path=Path(
            _require_string(experiment_table, "benchmark_config", "experiment")
        ),
        agent_config_path=Path(agent_path) if agent_path is not None else None,
        episode_count=_require_int(experiment_table, "episode_count", "experiment"),
        rollout_horizon=_optional_int_or_none(
            experiment_table,
            "rollout_horizon",
            "experiment",
        ),
        regime_schedule=tuple(
            _parse_enum_list(
                RegimeLabel,
                _optional_string_list(
                    experiment_table,
                    "regime_schedule",
                    "experiment",
                    default=[RegimeLabel.NORMAL.value],
                ),
                "experiment.regime_schedule",
            )
        ),
        regime_schedules=tuple(
            RegimeScheduleConfig(
                name=_require_string(schedule_table, "name", "regime_schedules"),
                labels=tuple(
                    _parse_enum_list(
                        RegimeLabel,
                        _require_string_list(schedule_table, "labels", "regime_schedules"),
                        "regime_schedules.labels",
                    )
                ),
            )
            for schedule_table in regime_schedule_tables
        ),
        seed_set=tuple(
            _optional_int_list(
                experiment_table,
                "seed_set",
                "experiment",
                default=[],
            )
        ),
        mode_set=tuple(
            _optional_string_list(
                experiment_table,
                "mode_set",
                "experiment",
                default=list(DEFAULT_RUNTIME_MODE_SET),
            )
        ),
        tool_ablation_variants=tuple(
            _optional_string_list(
                experiment_table,
                "tool_ablation_variants",
                "experiment",
                default=["full"],
            )
        ),
        semi_synthetic_external_evidence=_optional_bool(
            experiment_table,
            "semi_synthetic_external_evidence",
            "experiment",
            default=False,
        ),
        external_evidence_source=_optional_optional_string(
            experiment_table,
            "external_evidence_source",
            "experiment",
            default=None,
        ),
        results_dir=Path(_require_string(experiment_table, "results_dir", "experiment")),
    )


def load_agent_config(path: str | Path) -> AgentConfig:
    """Load a bounded outer-loop TOML file into a typed agent config."""

    document = _load_toml_document(path)
    agent_table = _require_table(document, "agent")
    configured_model_name = _optional_string(
        agent_table,
        "llm_model_name",
        "agent",
        default="gpt-4o-mini",
    )
    env_model_name = load_env_value("MEIO_LLM_ORCHESTRATOR_MODEL")

    return AgentConfig(
        enabled_regime_labels=tuple(
            _parse_enum_list(
                RegimeLabel,
                _require_string_list(agent_table, "enabled_regime_labels", "agent"),
                "agent.enabled_regime_labels",
            )
        ),
        allowed_update_types=tuple(
            _parse_enum_list(
                UpdateRequestType,
                _require_string_list(agent_table, "allowed_update_types", "agent"),
                "agent.allowed_update_types",
            )
        ),
        allowed_tool_classes=tuple(
            _parse_enum_list(
                ToolClass,
                _optional_string_list(
                    agent_table,
                    "allowed_tool_classes",
                    "agent",
                    default=[tool_class.value for tool_class in ToolClass],
                ),
                "agent.allowed_tool_classes",
            )
        ),
        minimum_confidence=_optional_number(
            agent_table, "minimum_confidence", "agent", default=0.0
        ),
        max_tool_steps=_optional_int(agent_table, "max_tool_steps", "agent", default=3),
        allow_replan_requests=_optional_bool(
            agent_table, "allow_replan_requests", "agent", default=True
        ),
        allow_abstain=_optional_bool(agent_table, "allow_abstain", "agent", default=True),
        llm_provider=_optional_string(
            agent_table,
            "llm_provider",
            "agent",
            default="openai",
        ),
        llm_client_mode=_optional_string(
            agent_table,
            "llm_client_mode",
            "agent",
            default="fake",
        ),
        llm_model_name=env_model_name or configured_model_name,
        llm_temperature=_optional_optional_number(
            agent_table,
            "llm_temperature",
            "agent",
            default=None,
        ),
        llm_request_timeout_s=_optional_number(
            agent_table,
            "llm_request_timeout_s",
            "agent",
            default=20.0,
        ),
        llm_max_retries=_optional_int(
            agent_table,
            "llm_max_retries",
            "agent",
            default=1,
        ),
    )


def load_public_benchmark_eval_config(path: str | Path) -> PublicBenchmarkEvalConfig:
    """Load a public-benchmark evaluation TOML file."""

    document = _load_toml_document(path)
    experiment_table = _require_table(document, "experiment")
    return PublicBenchmarkEvalConfig(
        experiment_name=_require_string(experiment_table, "name", "experiment"),
        benchmark_candidate=_require_string(
            experiment_table,
            "benchmark_candidate",
            "experiment",
        ),
        discovery_module=_optional_string(
            experiment_table,
            "discovery_module",
            "experiment",
            default="ReplenishmentEnv",
        ),
        benchmark_root=_optional_path_or_none(
            experiment_table,
            "benchmark_root",
            "experiment",
        ),
        demo_config_path=Path(
            _optional_string(
                experiment_table,
                "demo_config_path",
                "experiment",
                default="config/demo.yml",
            )
        ),
        agent_config_path=Path(
            _optional_string(
                experiment_table,
                "agent_config",
                "experiment",
                default="configs/agent/base.toml",
            )
        ),
        environment_config_name=_optional_string(
            experiment_table,
            "environment_config_name",
            "experiment",
            default="sku100.single_store.standard",
        ),
        wrapper_names=tuple(
            _optional_string_list(
                experiment_table,
                "wrapper_names",
                "experiment",
                default=["DefaultWrapper"],
            )
        ),
        benchmark_mode=_optional_string(
            experiment_table,
            "benchmark_mode",
            "experiment",
            default="test",
        ),
        smoke_horizon_steps=_optional_int(
            experiment_table,
            "smoke_horizon_steps",
            "experiment",
            default=1,
        ),
        mode_set=tuple(
            _optional_string_list(
                experiment_table,
                "mode_set",
                "experiment",
                default=list(DEFAULT_RUNTIME_MODE_SET),
            )
        ),
        episode_horizon_steps=_optional_int(
            experiment_table,
            "episode_horizon_steps",
            "experiment",
            default=10,
        ),
        base_stock_multiplier=_optional_number(
            experiment_table,
            "base_stock_multiplier",
            "experiment",
            default=1.0,
        ),
        demand_scale_epsilon=_optional_number(
            experiment_table,
            "demand_scale_epsilon",
            "experiment",
            default=1e-6,
        ),
        results_dir=Path(_require_string(experiment_table, "results_dir", "experiment")),
    )


def load_real_demand_backtest_config(path: str | Path) -> RealDemandBacktestConfig:
    """Load a real-demand backtest TOML file."""

    document = _load_toml_document(path)
    experiment_table = _require_table(document, "experiment")
    return RealDemandBacktestConfig(
        experiment_name=_require_string(experiment_table, "name", "experiment"),
        benchmark_config_path=Path(
            _require_string(experiment_table, "benchmark_config", "experiment")
        ),
        agent_config_path=Path(
            _require_string(experiment_table, "agent_config", "experiment")
        ),
        dataset_name=_require_string(experiment_table, "dataset_name", "experiment"),
        slice_name=_optional_optional_string(
            experiment_table,
            "slice_name",
            "experiment",
            default=None,
        ),
        discovery_module=_optional_string(
            experiment_table,
            "discovery_module",
            "experiment",
            default="ReplenishmentEnv",
        ),
        dataset_root=_optional_path_or_none(
            experiment_table,
            "dataset_root",
            "experiment",
        ),
        demand_csv_path=Path(
            _require_string(experiment_table, "demand_csv_path", "experiment")
        ),
        leadtime_csv_path=Path(
            _require_string(experiment_table, "leadtime_csv_path", "experiment")
        ),
        sku_list_path=Path(
            _require_string(experiment_table, "sku_list_path", "experiment")
        ),
        selected_skus=tuple(
            _optional_string_list(
                experiment_table,
                "selected_skus",
                "experiment",
                default=[],
            )
        ),
        selected_sku_count=_optional_int(
            experiment_table,
            "selected_sku_count",
            "experiment",
            default=5,
        ),
        subset_selection=_optional_string(
            experiment_table,
            "subset_selection",
            "experiment",
            default="nearest_benchmark_mean",
        ),
        training_window_days=_optional_int(
            experiment_table,
            "training_window_days",
            "experiment",
            default=180,
        ),
        history_window_days=_optional_int(
            experiment_table,
            "history_window_days",
            "experiment",
            default=28,
        ),
        forecast_update_window_days=_optional_int(
            experiment_table,
            "forecast_update_window_days",
            "experiment",
            default=1,
        ),
        evaluation_horizon_days=_optional_int(
            experiment_table,
            "evaluation_horizon_days",
            "experiment",
            default=60,
        ),
        evaluation_start_date=_optional_optional_string(
            experiment_table,
            "evaluation_start_date",
            "experiment",
            default=None,
        ),
        roll_forward_stride_days=_optional_int(
            experiment_table,
            "roll_forward_stride_days",
            "experiment",
            default=1,
        ),
        mode_set=tuple(
            _optional_string_list(
                experiment_table,
                "mode_set",
                "experiment",
                default=list(DEFAULT_RUNTIME_MODE_SET),
            )
        ),
        results_dir=Path(_require_string(experiment_table, "results_dir", "experiment")),
    )


def load_real_demand_backtest_panel_config(path: str | Path) -> RealDemandBacktestPanelConfig:
    """Load a fixed real-demand panel TOML file."""

    document = _load_toml_document(path)
    experiment_table = _require_table(document, "experiment")
    slice_tables = _optional_list(document, "slices")
    return RealDemandBacktestPanelConfig(
        experiment_name=_require_string(experiment_table, "name", "experiment"),
        benchmark_config_path=Path(
            _require_string(experiment_table, "benchmark_config", "experiment")
        ),
        agent_config_path=Path(
            _require_string(experiment_table, "agent_config", "experiment")
        ),
        discovery_module=_optional_string(
            experiment_table,
            "discovery_module",
            "experiment",
            default="ReplenishmentEnv",
        ),
        dataset_root=_optional_path_or_none(
            experiment_table,
            "dataset_root",
            "experiment",
        ),
        mode_set=tuple(
            _optional_string_list(
                experiment_table,
                "mode_set",
                "experiment",
                default=list(DEFAULT_RUNTIME_MODE_SET),
            )
        ),
        results_dir=Path(_require_string(experiment_table, "results_dir", "experiment")),
        slices=tuple(
            RealDemandBacktestSliceConfig(
                name=_require_string(slice_table, "name", "slices"),
                dataset_name=_require_string(slice_table, "dataset_name", "slices"),
                demand_csv_path=Path(
                    _require_string(slice_table, "demand_csv_path", "slices")
                ),
                leadtime_csv_path=Path(
                    _require_string(slice_table, "leadtime_csv_path", "slices")
                ),
                sku_list_path=Path(
                    _require_string(slice_table, "sku_list_path", "slices")
                ),
                selected_skus=tuple(
                    _optional_string_list(
                        slice_table,
                        "selected_skus",
                        "slices",
                        default=[],
                    )
                ),
                selected_sku_count=_optional_int(
                    slice_table,
                    "selected_sku_count",
                    "slices",
                    default=5,
                ),
                subset_selection=_optional_string(
                    slice_table,
                    "subset_selection",
                    "slices",
                    default="nearest_benchmark_mean",
                ),
                training_window_days=_optional_int(
                    slice_table,
                    "training_window_days",
                    "slices",
                    default=180,
                ),
                history_window_days=_optional_int(
                    slice_table,
                    "history_window_days",
                    "slices",
                    default=28,
                ),
                forecast_update_window_days=_optional_int(
                    slice_table,
                    "forecast_update_window_days",
                    "slices",
                    default=1,
                ),
                evaluation_horizon_days=_optional_int(
                    slice_table,
                    "evaluation_horizon_days",
                    "slices",
                    default=60,
                ),
                evaluation_start_date=_optional_optional_string(
                    slice_table,
                    "evaluation_start_date",
                    "slices",
                    default=None,
                ),
                roll_forward_stride_days=_optional_int(
                    slice_table,
                    "roll_forward_stride_days",
                    "slices",
                    default=1,
                ),
            )
            for slice_table in slice_tables
        ),
    )


def _load_toml_document(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        with path.open("rb") as handle:
            document = tomllib.load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Config file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Invalid TOML in {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"Config file {path} did not parse into a TOML table.")
    return document


def _require_table(document: dict[str, Any], key: str) -> dict[str, Any]:
    value = document.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing required table [{key}].")
    return value


def _optional_list(document: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = document.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a TOML array.")
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"{key} entries must be TOML tables.")
    return value


def _require_string(document: dict[str, Any], key: str, location: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}.{key} must be a non-empty string.")
    return value


def _require_int(document: dict[str, Any], key: str, location: str) -> int:
    value = document.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{location}.{key} must be an integer.")
    return value


def _optional_int(document: dict[str, Any], key: str, location: str, default: int) -> int:
    if key not in document:
        return default
    return _require_int(document, key, location)


def _optional_int_or_none(document: dict[str, Any], key: str, location: str) -> int | None:
    if key not in document:
        return None
    return _require_int(document, key, location)


def _require_number(document: dict[str, Any], key: str, location: str) -> float:
    value = document.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location}.{key} must be a number.")
    return float(value)


def _optional_number(
    document: dict[str, Any], key: str, location: str, default: float
) -> float:
    if key not in document:
        return default
    return _require_number(document, key, location)


def _optional_optional_number(
    document: dict[str, Any],
    key: str,
    location: str,
    default: float | None,
) -> float | None:
    if key not in document:
        return default
    value = document[key]
    if value is None:
        return None
    return _require_number(document, key, location)


def _optional_bool(document: dict[str, Any], key: str, location: str, default: bool) -> bool:
    if key not in document:
        return default
    value = document[key]
    if not isinstance(value, bool):
        raise ValueError(f"{location}.{key} must be a boolean.")
    return value


def _optional_string(document: dict[str, Any], key: str, location: str, default: str) -> str:
    if key not in document:
        return default
    return _require_string(document, key, location)


def _optional_optional_string(
    document: dict[str, Any],
    key: str,
    location: str,
    default: str | None,
) -> str | None:
    if key not in document:
        return default
    value = document[key]
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}.{key} must be a non-empty string when provided.")
    return value


def _optional_path_or_none(
    document: dict[str, Any],
    key: str,
    location: str,
) -> Path | None:
    raw_value = _optional_optional_string(document, key, location, default=None)
    if raw_value is None:
        return None
    return Path(raw_value)


def _require_string_list(document: dict[str, Any], key: str, location: str) -> list[str]:
    value = document.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{location}.{key} must be a non-empty list of strings.")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{location}.{key} must contain non-empty strings.")
        result.append(item)
    return result


def _optional_string_list(
    document: dict[str, Any],
    key: str,
    location: str,
    default: list[str],
) -> list[str]:
    if key not in document:
        return list(default)
    return _require_string_list(document, key, location)


def _optional_int_list(
    document: dict[str, Any],
    key: str,
    location: str,
    default: list[int],
) -> list[int]:
    if key not in document:
        return list(default)
    value = document.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{location}.{key} must be a list of integers.")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"{location}.{key} must contain integers.")
        result.append(item)
    return result


def _parse_enum(enum_type: type[Any], raw_value: str, location: str) -> Any:
    try:
        return enum_type(raw_value)
    except ValueError as exc:
        raise ValueError(f"Unsupported value for {location}: {raw_value!r}.") from exc


def _parse_enum_list(enum_type: type[Any], values: list[str], location: str) -> list[Any]:
    return [_parse_enum(enum_type, value, location) for value in values]


__all__ = [
    "load_agent_config",
    "load_benchmark_config",
    "load_experiment_config",
    "load_public_benchmark_eval_config",
    "load_real_demand_backtest_config",
]
