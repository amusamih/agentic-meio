"""Public real-demand dataset loading for bounded backtests."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from meio.benchmarks.replenishmentenv_support import locate_package_root


def _non_empty(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")


@dataclass(frozen=True, slots=True)
class RealDemandDatasetPaths:
    """Resolved file locations for one public demand dataset."""

    dataset_name: str
    source_root: Path
    demand_csv_path: Path
    leadtime_csv_path: Path
    sku_list_path: Path

    def __post_init__(self) -> None:
        _non_empty(self.dataset_name, "dataset_name")
        for field_name in (
            "source_root",
            "demand_csv_path",
            "leadtime_csv_path",
            "sku_list_path",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, Path):
                raise TypeError(f"{field_name} must be a Path.")


@dataclass(frozen=True, slots=True)
class BacktestWindow:
    """Resolved rolling-window boundaries for one bounded backtest."""

    training_start_index: int
    training_end_index: int
    evaluation_start_index: int
    evaluation_end_index: int
    history_window_days: int
    forecast_update_window_days: int
    evaluation_horizon_days: int

    def __post_init__(self) -> None:
        for field_name in (
            "training_start_index",
            "training_end_index",
            "evaluation_start_index",
            "evaluation_end_index",
            "history_window_days",
            "forecast_update_window_days",
            "evaluation_horizon_days",
        ):
            value = getattr(self, field_name)
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative.")
        if self.training_end_index < self.training_start_index:
            raise ValueError("training_end_index must be >= training_start_index.")
        if self.evaluation_end_index < self.evaluation_start_index:
            raise ValueError("evaluation_end_index must be >= evaluation_start_index.")


@dataclass(frozen=True, slots=True)
class RealDemandSeries:
    """Bounded aggregate demand and lead-time series for backtesting."""

    dataset_name: str
    source_root: Path
    dates: tuple[str, ...]
    selected_skus: tuple[str, ...]
    demand_values: tuple[float, ...]
    leadtime_values: tuple[float, ...]
    demand_target_mean: float
    realized_subset_mean: float
    realized_leadtime_mean: float
    selection_strategy: str
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _non_empty(self.dataset_name, "dataset_name")
        _non_empty(self.selection_strategy, "selection_strategy")
        if not isinstance(self.source_root, Path):
            raise TypeError("source_root must be a Path.")
        object.__setattr__(self, "dates", tuple(self.dates))
        object.__setattr__(self, "selected_skus", tuple(self.selected_skus))
        object.__setattr__(self, "demand_values", tuple(self.demand_values))
        object.__setattr__(self, "leadtime_values", tuple(self.leadtime_values))
        object.__setattr__(self, "notes", tuple(self.notes))
        if not self.dates:
            raise ValueError("dates must not be empty.")
        if len(self.dates) != len(self.demand_values) or len(self.dates) != len(self.leadtime_values):
            raise ValueError("dates, demand_values, and leadtime_values must have matching lengths.")
        if not self.selected_skus:
            raise ValueError("selected_skus must not be empty.")
        for sku in self.selected_skus:
            _non_empty(sku, "selected_skus")
        for value in self.demand_values:
            if value < 0.0:
                raise ValueError("demand_values must be non-negative.")
        for value in self.leadtime_values:
            if value <= 0.0:
                raise ValueError("leadtime_values must be positive.")
        if self.demand_target_mean <= 0.0:
            raise ValueError("demand_target_mean must be positive.")
        if self.realized_subset_mean < 0.0:
            raise ValueError("realized_subset_mean must be non-negative.")
        if self.realized_leadtime_mean <= 0.0:
            raise ValueError("realized_leadtime_mean must be positive.")
        for note in self.notes:
            _non_empty(note, "notes")


def resolve_real_demand_dataset_paths(
    *,
    dataset_name: str,
    discovery_module: str,
    dataset_root: Path | None,
    demand_csv_path: Path,
    leadtime_csv_path: Path,
    sku_list_path: Path,
) -> RealDemandDatasetPaths:
    """Resolve public dataset files from an explicit or discovered package root."""

    source_root = locate_package_root(
        module_name=discovery_module,
        explicit_root=dataset_root,
    )
    if source_root is None:
        raise FileNotFoundError(
            f"Could not locate package root for {discovery_module!r}; "
            "set dataset_root explicitly."
        )
    resolved = RealDemandDatasetPaths(
        dataset_name=dataset_name,
        source_root=source_root,
        demand_csv_path=source_root / demand_csv_path,
        leadtime_csv_path=source_root / leadtime_csv_path,
        sku_list_path=source_root / sku_list_path,
    )
    for field_name in ("demand_csv_path", "leadtime_csv_path", "sku_list_path"):
        path = getattr(resolved, field_name)
        if not path.exists():
            raise FileNotFoundError(f"Required dataset file not found: {path}")
    return resolved


def read_sku_list(path: str | Path) -> tuple[str, ...]:
    """Read a comment-tolerant SKU list file."""

    rows: list[str] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line == "SKU":
            continue
        rows.append(line)
    return tuple(rows)


def construct_backtest_window(
    *,
    total_length: int,
    training_window_days: int,
    history_window_days: int,
    forecast_update_window_days: int,
    evaluation_horizon_days: int,
    evaluation_start_index: int | None = None,
) -> BacktestWindow:
    """Resolve one bounded rolling backtest window."""

    if training_window_days <= 0:
        raise ValueError("training_window_days must be positive.")
    if history_window_days <= 0:
        raise ValueError("history_window_days must be positive.")
    if forecast_update_window_days <= 0:
        raise ValueError("forecast_update_window_days must be positive.")
    if evaluation_horizon_days <= 0:
        raise ValueError("evaluation_horizon_days must be positive.")
    required_length = training_window_days + evaluation_horizon_days
    resolved_evaluation_start_index = (
        training_window_days if evaluation_start_index is None else evaluation_start_index
    )
    if resolved_evaluation_start_index < training_window_days:
        raise ValueError(
            "evaluation_start_index must be at least training_window_days "
            "to preserve a full training window."
        )
    if total_length < required_length and evaluation_start_index is None:
        raise ValueError(
            f"Dataset length {total_length} is shorter than the required "
            f"training + evaluation horizon {required_length}."
        )
    training_start_index = resolved_evaluation_start_index - training_window_days
    evaluation_end_index = resolved_evaluation_start_index + evaluation_horizon_days - 1
    if training_start_index < 0 or evaluation_end_index >= total_length:
        raise ValueError(
            "Resolved backtest window exceeds the available dataset range."
        )
    return BacktestWindow(
        training_start_index=training_start_index,
        training_end_index=resolved_evaluation_start_index - 1,
        evaluation_start_index=resolved_evaluation_start_index,
        evaluation_end_index=evaluation_end_index,
        history_window_days=history_window_days,
        forecast_update_window_days=forecast_update_window_days,
        evaluation_horizon_days=evaluation_horizon_days,
    )


def load_real_demand_series(
    *,
    paths: RealDemandDatasetPaths,
    selected_sku_count: int,
    demand_target_mean: float,
    subset_selection: str,
    selected_skus: tuple[str, ...] = (),
) -> RealDemandSeries:
    """Load one bounded aggregate demand/lead-time series from public data."""

    if selected_sku_count <= 0:
        raise ValueError("selected_sku_count must be positive.")
    demand_frame = pd.read_csv(paths.demand_csv_path, comment="#")
    leadtime_frame = pd.read_csv(paths.leadtime_csv_path, comment="#")
    sku_candidates = tuple(
        sku
        for sku in read_sku_list(paths.sku_list_path)
        if sku in demand_frame.columns and sku in leadtime_frame.columns
    )
    if len(sku_candidates) < selected_sku_count:
        raise ValueError(
            "Not enough shared SKU columns exist for the requested subset size."
        )
    if selected_skus:
        selected_skus = _resolve_explicit_skus(
            requested_skus=selected_skus,
            sku_candidates=sku_candidates,
        )
        selection_strategy = "explicit_selected_skus"
    else:
        selected_skus = _select_skus(
            demand_frame=demand_frame,
            sku_candidates=sku_candidates,
            selected_sku_count=selected_sku_count,
            demand_target_mean=demand_target_mean,
            subset_selection=subset_selection,
        )
        selection_strategy = subset_selection
    demand_series = tuple(
        float(value) for value in demand_frame[list(selected_skus)].mean(axis=1)
    )
    leadtime_series = tuple(
        max(1.0, float(value))
        for value in leadtime_frame[list(selected_skus)].mean(axis=1)
    )
    dates = tuple(str(value) for value in demand_frame["Date"].tolist())
    return RealDemandSeries(
        dataset_name=paths.dataset_name,
        source_root=paths.source_root,
        dates=dates,
        selected_skus=selected_skus,
        demand_values=demand_series,
        leadtime_values=leadtime_series,
        demand_target_mean=demand_target_mean,
        realized_subset_mean=float(sum(demand_series) / len(demand_series)),
        realized_leadtime_mean=float(sum(leadtime_series) / len(leadtime_series)),
        selection_strategy=selection_strategy,
        notes=(
            f"demand_csv={paths.demand_csv_path.name}",
            f"leadtime_csv={paths.leadtime_csv_path.name}",
            f"sku_list={paths.sku_list_path.name}",
        ),
    )


def resolve_evaluation_start_index(
    *,
    dates: tuple[str, ...],
    evaluation_start_date: str | None,
) -> int | None:
    """Resolve an optional evaluation start date to a dataset index."""

    if evaluation_start_date is None:
        return None
    try:
        return dates.index(evaluation_start_date)
    except ValueError as exc:
        raise ValueError(
            f"evaluation_start_date {evaluation_start_date!r} is not present in the dataset."
        ) from exc


def _select_skus(
    *,
    demand_frame: pd.DataFrame,
    sku_candidates: tuple[str, ...],
    selected_sku_count: int,
    demand_target_mean: float,
    subset_selection: str,
) -> tuple[str, ...]:
    if subset_selection == "nearest_benchmark_mean":
        ranked = sorted(
            sku_candidates,
            key=lambda sku: abs(float(demand_frame[sku].mean()) - demand_target_mean),
        )
        return tuple(ranked[:selected_sku_count])
    if subset_selection == "first_available":
        return tuple(sku_candidates[:selected_sku_count])
    raise ValueError(f"Unsupported subset_selection: {subset_selection!r}.")


def _resolve_explicit_skus(
    *,
    requested_skus: tuple[str, ...],
    sku_candidates: tuple[str, ...],
) -> tuple[str, ...]:
    missing = tuple(sku for sku in requested_skus if sku not in sku_candidates)
    if missing:
        raise ValueError(
            f"Explicit selected_skus are missing from the shared dataset columns: {missing}."
        )
    return requested_skus


__all__ = [
    "BacktestWindow",
    "RealDemandDatasetPaths",
    "RealDemandSeries",
    "construct_backtest_window",
    "load_real_demand_series",
    "read_sku_list",
    "resolve_evaluation_start_index",
    "resolve_real_demand_dataset_paths",
]
