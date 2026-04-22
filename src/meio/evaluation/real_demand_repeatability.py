"""Repeatability analysis for repeated real-demand backtest runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

DEFAULT_REAL_DEMAND_RESULTS_ROOT = Path("results/real_demand_backtest")
_NEUTRAL_TOLERANCE = 1e-9
_LLM_MODE = "llm_orchestrator"
_DETERMINISTIC_ORCHESTRATOR_MODE = "deterministic_orchestrator"
_DETERMINISTIC_BASELINE_MODE = "deterministic_baseline"


def _non_empty(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Expected a numeric value when present.")
    return float(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("Expected an integer value when present.")
    return value


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Expected a string value when present.")
    return value


@dataclass(frozen=True, slots=True)
class NumericSpreadSummary:
    """Compact numeric spread summary over repeated runs."""

    mean_value: float | None
    minimum_value: float | None
    maximum_value: float | None
    standard_deviation: float | None


@dataclass(frozen=True, slots=True)
class RealDemandRepeatabilityModeRunSummary:
    """Per-mode summary extracted from one saved real-demand run."""

    mode: str
    average_total_cost: float | None
    average_fill_rate: float | None
    fallback_count: int
    invalid_output_count: int
    tool_call_count: int | None
    replan_rate: float | None
    no_action_rate: float | None
    llm_provider: str | None = None
    llm_model_name: str | None = None

    def __post_init__(self) -> None:
        _non_empty(self.mode, "mode")


@dataclass(frozen=True, slots=True)
class RealDemandRepeatabilityRunSummary:
    """Per-run real-demand repeatability summary."""

    run_dir: str
    panel_run_dir: str | None
    panel_config_hash: str | None
    config_hash: str | None
    artifact_use_class: str | None
    validity_gate_passed: bool | None
    slice_name: str | None
    dataset_name: str | None
    date_range: tuple[str, ...]
    selected_skus: tuple[str, ...]
    mode_summaries: tuple[RealDemandRepeatabilityModeRunSummary, ...]

    def __post_init__(self) -> None:
        _non_empty(self.run_dir, "run_dir")


@dataclass(frozen=True, slots=True)
class RealDemandRepeatabilityModeSummary:
    """Repeated-run summary for one real-demand mode."""

    mode: str
    run_count: int
    average_total_cost: NumericSpreadSummary
    average_fill_rate: NumericSpreadSummary
    total_fallback_count: int
    total_invalid_output_count: int
    tool_call_count: NumericSpreadSummary
    replan_rate: NumericSpreadSummary
    no_action_rate: NumericSpreadSummary
    llm_providers: tuple[str, ...]

    def __post_init__(self) -> None:
        _non_empty(self.mode, "mode")


@dataclass(frozen=True, slots=True)
class RealDemandProviderComparison:
    """Provider-group comparison for real-demand LLM runs."""

    fake_run_count: int
    real_run_count: int
    fake_cost_mean: float | None
    real_cost_mean: float | None
    real_minus_fake_cost_mean: float | None
    fake_fill_rate_mean: float | None
    real_fill_rate_mean: float | None


@dataclass(frozen=True, slots=True)
class RealDemandSliceRepeatabilitySummary:
    """Per-slice repeatability summary across repeated panel reruns."""

    slice_name: str
    dataset_name: str | None
    date_range: tuple[str, ...]
    selected_skus: tuple[str, ...]
    panel_config_hashes: tuple[str, ...]
    run_count: int
    mode_summaries: tuple[RealDemandRepeatabilityModeSummary, ...]
    llm_minus_deterministic_orchestrator_cost_delta: NumericSpreadSummary
    deterministic_orchestrator_minus_baseline_cost_delta: NumericSpreadSummary
    llm_vs_deterministic_orchestrator_label: str
    deterministic_orchestrator_vs_baseline_label: str
    llm_validity_clean: bool

    def __post_init__(self) -> None:
        _non_empty(self.slice_name, "slice_name")


@dataclass(frozen=True, slots=True)
class RealDemandRepeatabilityAnalysis:
    """Top-level repeatability summary for real-demand backtests."""

    run_dirs: tuple[str, ...]
    run_count: int
    panel_run_dirs: tuple[str, ...]
    panel_run_count: int
    panel_config_hashes: tuple[str, ...]
    config_hashes: tuple[str, ...]
    slice_names: tuple[str, ...]
    dataset_names: tuple[str, ...]
    selected_sku_sets: tuple[tuple[str, ...], ...]
    run_summaries: tuple[RealDemandRepeatabilityRunSummary, ...]
    mode_summaries: tuple[RealDemandRepeatabilityModeSummary, ...]
    slice_summaries: tuple[RealDemandSliceRepeatabilitySummary, ...]
    llm_vs_deterministic_orchestrator_label: str
    deterministic_orchestrator_vs_baseline_label: str
    llm_validity_clean: bool
    fake_vs_real_llm_comparison: RealDemandProviderComparison | None
    slices_favoring_deterministic_orchestrator: tuple[str, ...]
    slices_favoring_llm_orchestrator: tuple[str, ...]
    competitive_slices: tuple[str, ...]
    mixed_slices: tuple[str, ...]


def list_real_demand_run_dirs(
    *,
    results_root: str | Path = DEFAULT_REAL_DEMAND_RESULTS_ROOT,
    latest_n: int | None = None,
) -> tuple[Path, ...]:
    """List saved real-demand run directories ordered by modification time."""

    root = Path(results_root)
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist.")
    run_groups = tuple(
        path
        for path in root.iterdir()
        if path.is_dir()
        and (
            (path / "aggregate_summary.json").exists()
            or (path / "panel_summary.json").exists()
        )
    )
    if not run_groups:
        raise FileNotFoundError(f"No saved real-demand runs found under {root}.")
    ordered = tuple(sorted(run_groups, key=lambda path: path.stat().st_mtime))
    if latest_n is None:
        selected_groups = ordered
    else:
        if latest_n <= 0:
            raise ValueError("latest_n must be positive when provided.")
        selected_groups = ordered[-latest_n:]
    run_dirs = tuple(
        run_dir
        for group_dir in selected_groups
        for run_dir in _expand_panel_run_dir(group_dir)
    )
    if not run_dirs:
        raise FileNotFoundError(f"No saved real-demand runs found under {root}.")
    return run_dirs


def analyze_real_demand_repeatability(
    run_dirs: tuple[str | Path, ...] | None = None,
    *,
    results_root: str | Path = DEFAULT_REAL_DEMAND_RESULTS_ROOT,
    latest_n: int | None = None,
) -> RealDemandRepeatabilityAnalysis:
    """Summarize repeatability across saved real-demand runs."""

    resolved_run_dirs = _resolve_run_dirs(
        run_dirs=run_dirs,
        results_root=results_root,
        latest_n=latest_n,
    )
    run_summaries = tuple(_summarize_run_dir(path) for path in resolved_run_dirs)
    slice_summaries = _build_slice_summaries(run_summaries)
    mode_summaries = tuple(
        summary
        for summary in (
            _build_mode_summary(run_summaries, _DETERMINISTIC_BASELINE_MODE),
            _build_mode_summary(run_summaries, _DETERMINISTIC_ORCHESTRATOR_MODE),
            _build_mode_summary(run_summaries, _LLM_MODE),
        )
        if summary is not None
    )
    llm_deltas = _mode_cost_deltas(
        run_summaries,
        left_mode=_LLM_MODE,
        right_mode=_DETERMINISTIC_ORCHESTRATOR_MODE,
    )
    deterministic_deltas = _mode_cost_deltas(
        run_summaries,
        left_mode=_DETERMINISTIC_ORCHESTRATOR_MODE,
        right_mode=_DETERMINISTIC_BASELINE_MODE,
    )
    return RealDemandRepeatabilityAnalysis(
        run_dirs=tuple(str(path) for path in resolved_run_dirs),
        run_count=len(run_summaries),
        panel_run_dirs=tuple(
            sorted(
                {
                    summary.panel_run_dir
                    for summary in run_summaries
                    if summary.panel_run_dir is not None
                }
            )
        ),
        panel_run_count=len(
            {
                summary.panel_run_dir
                for summary in run_summaries
                if summary.panel_run_dir is not None
            }
        ),
        panel_config_hashes=tuple(
            sorted(
                {
                    summary.panel_config_hash
                    for summary in run_summaries
                    if summary.panel_config_hash is not None
                }
            )
        ),
        config_hashes=tuple(
            sorted(
                {
                    summary.config_hash
                    for summary in run_summaries
                    if summary.config_hash is not None
                }
            )
        ),
        slice_names=tuple(
            sorted(
                {
                    summary.slice_name
                    for summary in run_summaries
                    if summary.slice_name is not None
                }
            )
        ),
        dataset_names=tuple(
            sorted(
                {
                    summary.dataset_name
                    for summary in run_summaries
                    if summary.dataset_name is not None
                }
            )
        ),
        selected_sku_sets=tuple(
            sorted(
                {
                    summary.selected_skus
                    for summary in run_summaries
                    if summary.selected_skus
                }
            )
        ),
        run_summaries=run_summaries,
        mode_summaries=mode_summaries,
        slice_summaries=slice_summaries,
        llm_vs_deterministic_orchestrator_label=_cost_stability_label(llm_deltas),
        deterministic_orchestrator_vs_baseline_label=_cost_stability_label(
            deterministic_deltas
        ),
        llm_validity_clean=all(
            mode.fallback_count == 0 and mode.invalid_output_count == 0
            for summary in run_summaries
            for mode in summary.mode_summaries
            if mode.mode == _LLM_MODE
        ),
        fake_vs_real_llm_comparison=_build_fake_vs_real_comparison(run_summaries),
        slices_favoring_deterministic_orchestrator=_slice_names_with_label(
            slice_summaries,
            "stably_worse",
        ),
        slices_favoring_llm_orchestrator=_slice_names_with_label(
            slice_summaries,
            "stably_better",
        ),
        competitive_slices=_slice_names_with_label(slice_summaries, "stably_neutral"),
        mixed_slices=_slice_names_with_label(slice_summaries, "unstable"),
    )


def _resolve_run_dirs(
    *,
    run_dirs: tuple[str | Path, ...] | None,
    results_root: str | Path,
    latest_n: int | None,
) -> tuple[Path, ...]:
    if run_dirs:
        return tuple(
            run_dir
            for path in run_dirs
            for run_dir in _expand_panel_run_dir(Path(path))
        )
    return list_real_demand_run_dirs(results_root=results_root, latest_n=latest_n)


def _expand_panel_run_dir(path: Path) -> tuple[Path, ...]:
    if _is_single_run_dir(path):
        slice_root = path / "slices"
        if not slice_root.exists():
            return (path,)
        slice_runs = tuple(
            candidate
            for candidate in sorted(slice_root.iterdir())
            if _is_single_run_dir(candidate)
        )
        if slice_runs:
            return slice_runs
        return (path,)
    if path.is_dir() and (path / "slices").exists():
        slice_runs = tuple(
            candidate
            for candidate in sorted((path / "slices").iterdir())
            if _is_single_run_dir(candidate)
        )
        if slice_runs:
            return slice_runs
    raise FileNotFoundError(f"{path} is not a saved real-demand run directory.")


def _is_single_run_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "aggregate_summary.json").exists()
        and (path / "experiment_metadata.json").exists()
        and (path / "dataset_summary.json").exists()
    )


def _summarize_run_dir(run_dir: Path) -> RealDemandRepeatabilityRunSummary:
    metadata_payload = _load_json(run_dir / "experiment_metadata.json")
    aggregate_payload = _load_json(run_dir / "aggregate_summary.json")
    dataset_payload = _load_json(run_dir / "dataset_summary.json")
    panel_run_dir, panel_config_hash = _resolve_panel_identity(run_dir)
    mode_summaries: list[RealDemandRepeatabilityModeRunSummary] = []
    provider = _optional_str(metadata_payload.get("provider"))
    model_name = _optional_str(metadata_payload.get("model_name"))
    for item in aggregate_payload.get("mode_summaries", []):
        if not isinstance(item, dict):
            continue
        mode_name = _optional_str(item.get("mode"))
        if mode_name is None:
            continue
        performance_summary = item.get("performance_summary", {})
        decision_quality = item.get("decision_quality", {})
        validity_summary = item.get("validity_summary", {})
        tool_use_summary = item.get("tool_use_summary", {})
        benchmark_metrics = item.get("benchmark_metrics", {})
        tool_call_count = _resolve_tool_call_count(
            tool_use_summary=tool_use_summary,
            benchmark_metrics=benchmark_metrics,
        )
        mode_summaries.append(
            RealDemandRepeatabilityModeRunSummary(
                mode=mode_name,
                average_total_cost=_optional_float(
                    performance_summary.get("average_total_cost")
                ),
                average_fill_rate=_optional_float(
                    performance_summary.get("average_fill_rate")
                ),
                fallback_count=_optional_int(validity_summary.get("fallback_count")) or 0,
                invalid_output_count=(
                    _optional_int(validity_summary.get("invalid_output_count")) or 0
                ),
                tool_call_count=tool_call_count,
                replan_rate=_optional_float(decision_quality.get("replan_rate")),
                no_action_rate=_optional_float(decision_quality.get("no_action_rate")),
                llm_provider=provider if mode_name == _LLM_MODE else None,
                llm_model_name=model_name if mode_name == _LLM_MODE else None,
            )
        )
    selected_skus = dataset_payload.get("selected_skus", [])
    if not isinstance(selected_skus, list):
        raise ValueError("selected_skus must be a list when present.")
    return RealDemandRepeatabilityRunSummary(
        run_dir=str(run_dir),
        panel_run_dir=panel_run_dir,
        panel_config_hash=panel_config_hash,
        config_hash=_optional_str(metadata_payload.get("config_hash")),
        artifact_use_class=_optional_str(metadata_payload.get("artifact_use_class")),
        validity_gate_passed=(
            bool(metadata_payload.get("validity_gate_passed"))
            if "validity_gate_passed" in metadata_payload
            else None
        ),
        slice_name=_optional_str(dataset_payload.get("slice_name")),
        dataset_name=_optional_str(dataset_payload.get("dataset_name")),
        date_range=tuple(str(item) for item in dataset_payload.get("date_range", [])),
        selected_skus=tuple(str(item) for item in selected_skus),
        mode_summaries=tuple(mode_summaries),
    )


def _resolve_panel_identity(run_dir: Path) -> tuple[str | None, str | None]:
    if run_dir.parent.name != "slices":
        return None, None
    panel_root = run_dir.parent.parent
    return str(panel_root), _read_panel_config_hash(panel_root)


def _read_panel_config_hash(panel_root: Path) -> str | None:
    dataset_summary_path = panel_root / "dataset_summary.json"
    if dataset_summary_path.exists():
        payload = _load_json(dataset_summary_path)
        panel_config_hash = _optional_str(payload.get("panel_config_hash"))
        if panel_config_hash is not None:
            return panel_config_hash
    panel_summary_path = panel_root / "panel_summary.json"
    if panel_summary_path.exists():
        payload = _load_json(panel_summary_path)
        return _optional_str(payload.get("panel_config_hash"))
    return None


def _build_mode_summary(
    run_summaries: tuple[RealDemandRepeatabilityRunSummary, ...],
    mode_name: str,
) -> RealDemandRepeatabilityModeSummary | None:
    selected = tuple(
        mode
        for run_summary in run_summaries
        for mode in run_summary.mode_summaries
        if mode.mode == mode_name
    )
    if not selected:
        return None
    return RealDemandRepeatabilityModeSummary(
        mode=mode_name,
        run_count=len(selected),
        average_total_cost=_numeric_spread(
            tuple(
                value
                for value in (item.average_total_cost for item in selected)
                if value is not None
            )
        ),
        average_fill_rate=_numeric_spread(
            tuple(
                value
                for value in (item.average_fill_rate for item in selected)
                if value is not None
            )
        ),
        total_fallback_count=sum(item.fallback_count for item in selected),
        total_invalid_output_count=sum(item.invalid_output_count for item in selected),
        tool_call_count=_numeric_spread(
            tuple(
                float(value)
                for value in (item.tool_call_count for item in selected)
                if value is not None
            )
        ),
        replan_rate=_numeric_spread(
            tuple(value for value in (item.replan_rate for item in selected) if value is not None)
        ),
        no_action_rate=_numeric_spread(
            tuple(value for value in (item.no_action_rate for item in selected) if value is not None)
        ),
        llm_providers=tuple(
            sorted(
                {
                    item.llm_provider
                    for item in selected
                    if item.llm_provider is not None
                }
            )
        ),
    )


def _mode_cost_deltas(
    run_summaries: tuple[RealDemandRepeatabilityRunSummary, ...],
    *,
    left_mode: str,
    right_mode: str,
) -> tuple[float, ...]:
    deltas: list[float] = []
    for run_summary in run_summaries:
        left = _find_mode(run_summary, left_mode)
        right = _find_mode(run_summary, right_mode)
        if left is None or right is None:
            continue
        if left.average_total_cost is None or right.average_total_cost is None:
            continue
        deltas.append(left.average_total_cost - right.average_total_cost)
    return tuple(deltas)


def _find_mode(
    run_summary: RealDemandRepeatabilityRunSummary,
    mode_name: str,
) -> RealDemandRepeatabilityModeRunSummary | None:
    for mode_summary in run_summary.mode_summaries:
        if mode_summary.mode == mode_name:
            return mode_summary
    return None


def _resolve_tool_call_count(
    *,
    tool_use_summary: object,
    benchmark_metrics: object,
) -> int | None:
    if isinstance(tool_use_summary, dict):
        tool_use_count = _optional_float(tool_use_summary.get("average_tool_call_count"))
        if tool_use_count is not None:
            return int(round(tool_use_count))
    if isinstance(benchmark_metrics, dict):
        benchmark_count = _optional_float(benchmark_metrics.get("average_tool_call_count"))
        if benchmark_count is not None:
            return int(round(benchmark_count))
    return None


def _slice_names_with_label(
    slice_summaries: tuple[RealDemandSliceRepeatabilitySummary, ...],
    label: str,
) -> tuple[str, ...]:
    return tuple(
        summary.slice_name
        for summary in slice_summaries
        if summary.llm_vs_deterministic_orchestrator_label == label
    )


def _slice_cost_label(run_summary: RealDemandRepeatabilityRunSummary) -> str:
    llm_summary = _find_mode(run_summary, _LLM_MODE)
    deterministic_summary = _find_mode(run_summary, _DETERMINISTIC_ORCHESTRATOR_MODE)
    if llm_summary is None or deterministic_summary is None:
        return "unknown"
    if llm_summary.average_total_cost is None or deterministic_summary.average_total_cost is None:
        return "unknown"
    delta = llm_summary.average_total_cost - deterministic_summary.average_total_cost
    if abs(delta) <= _NEUTRAL_TOLERANCE:
        return "competitive"
    if delta < 0.0:
        return "llm_orchestrator"
    return "deterministic_orchestrator"


def _slice_identity(run_summary: RealDemandRepeatabilityRunSummary) -> str:
    if run_summary.slice_name is not None:
        return run_summary.slice_name
    if run_summary.dataset_name is not None:
        return run_summary.dataset_name
    return run_summary.run_dir


def _build_slice_summaries(
    run_summaries: tuple[RealDemandRepeatabilityRunSummary, ...],
) -> tuple[RealDemandSliceRepeatabilitySummary, ...]:
    grouped: dict[str, list[RealDemandRepeatabilityRunSummary]] = {}
    for run_summary in run_summaries:
        grouped.setdefault(_slice_identity(run_summary), []).append(run_summary)
    return tuple(
        _build_slice_summary(slice_name, tuple(grouped[slice_name]))
        for slice_name in sorted(grouped)
    )


def _build_slice_summary(
    slice_name: str,
    run_summaries: tuple[RealDemandRepeatabilityRunSummary, ...],
) -> RealDemandSliceRepeatabilitySummary:
    reference = run_summaries[0]
    llm_deltas = _mode_cost_deltas(
        run_summaries,
        left_mode=_LLM_MODE,
        right_mode=_DETERMINISTIC_ORCHESTRATOR_MODE,
    )
    deterministic_deltas = _mode_cost_deltas(
        run_summaries,
        left_mode=_DETERMINISTIC_ORCHESTRATOR_MODE,
        right_mode=_DETERMINISTIC_BASELINE_MODE,
    )
    mode_summaries = tuple(
        summary
        for summary in (
            _build_mode_summary(run_summaries, _DETERMINISTIC_BASELINE_MODE),
            _build_mode_summary(run_summaries, _DETERMINISTIC_ORCHESTRATOR_MODE),
            _build_mode_summary(run_summaries, _LLM_MODE),
        )
        if summary is not None
    )
    return RealDemandSliceRepeatabilitySummary(
        slice_name=slice_name,
        dataset_name=reference.dataset_name,
        date_range=reference.date_range,
        selected_skus=reference.selected_skus,
        panel_config_hashes=tuple(
            sorted(
                {
                    summary.panel_config_hash
                    for summary in run_summaries
                    if summary.panel_config_hash is not None
                }
            )
        ),
        run_count=len(run_summaries),
        mode_summaries=mode_summaries,
        llm_minus_deterministic_orchestrator_cost_delta=_numeric_spread(llm_deltas),
        deterministic_orchestrator_minus_baseline_cost_delta=_numeric_spread(
            deterministic_deltas
        ),
        llm_vs_deterministic_orchestrator_label=_cost_stability_label(llm_deltas),
        deterministic_orchestrator_vs_baseline_label=_cost_stability_label(
            deterministic_deltas
        ),
        llm_validity_clean=all(
            mode.fallback_count == 0 and mode.invalid_output_count == 0
            for summary in run_summaries
            for mode in summary.mode_summaries
            if mode.mode == _LLM_MODE
        ),
    )


def _build_fake_vs_real_comparison(
    run_summaries: tuple[RealDemandRepeatabilityRunSummary, ...],
) -> RealDemandProviderComparison | None:
    fake_costs: list[float] = []
    real_costs: list[float] = []
    fake_fill_rates: list[float] = []
    real_fill_rates: list[float] = []
    for run_summary in run_summaries:
        llm_summary = _find_mode(run_summary, _LLM_MODE)
        if llm_summary is None or llm_summary.llm_provider is None:
            continue
        if llm_summary.llm_provider == "fake_llm_client":
            if llm_summary.average_total_cost is not None:
                fake_costs.append(llm_summary.average_total_cost)
            if llm_summary.average_fill_rate is not None:
                fake_fill_rates.append(llm_summary.average_fill_rate)
        elif llm_summary.llm_provider == "openai":
            if llm_summary.average_total_cost is not None:
                real_costs.append(llm_summary.average_total_cost)
            if llm_summary.average_fill_rate is not None:
                real_fill_rates.append(llm_summary.average_fill_rate)
    if not fake_costs and not real_costs:
        return None
    fake_cost_mean = float(mean(fake_costs)) if fake_costs else None
    real_cost_mean = float(mean(real_costs)) if real_costs else None
    return RealDemandProviderComparison(
        fake_run_count=len(fake_costs),
        real_run_count=len(real_costs),
        fake_cost_mean=fake_cost_mean,
        real_cost_mean=real_cost_mean,
        real_minus_fake_cost_mean=(
            real_cost_mean - fake_cost_mean
            if real_cost_mean is not None and fake_cost_mean is not None
            else None
        ),
        fake_fill_rate_mean=(float(mean(fake_fill_rates)) if fake_fill_rates else None),
        real_fill_rate_mean=(float(mean(real_fill_rates)) if real_fill_rates else None),
    )


def _numeric_spread(values: tuple[float, ...]) -> NumericSpreadSummary:
    if not values:
        return NumericSpreadSummary(None, None, None, None)
    return NumericSpreadSummary(
        mean_value=float(mean(values)),
        minimum_value=float(min(values)),
        maximum_value=float(max(values)),
        standard_deviation=0.0 if len(values) == 1 else float(pstdev(values)),
    )


def _cost_stability_label(values: tuple[float, ...]) -> str:
    if not values:
        return "unknown"
    if all(abs(value) <= _NEUTRAL_TOLERANCE for value in values):
        return "stably_neutral"
    if all(value <= _NEUTRAL_TOLERANCE for value in values) and any(
        value < -_NEUTRAL_TOLERANCE for value in values
    ):
        return "stably_better"
    if all(value >= -_NEUTRAL_TOLERANCE for value in values) and any(
        value > _NEUTRAL_TOLERANCE for value in values
    ):
        return "stably_worse"
    return "unstable"


__all__ = [
    "DEFAULT_REAL_DEMAND_RESULTS_ROOT",
    "NumericSpreadSummary",
    "RealDemandProviderComparison",
    "RealDemandRepeatabilityAnalysis",
    "RealDemandRepeatabilityModeRunSummary",
    "RealDemandRepeatabilityModeSummary",
    "RealDemandRepeatabilityRunSummary",
    "RealDemandSliceRepeatabilitySummary",
    "analyze_real_demand_repeatability",
    "list_real_demand_run_dirs",
]
