from __future__ import annotations

from pathlib import Path
import shutil
from uuid import uuid4

from meio.data.real_demand_loader import (
    construct_backtest_window,
    load_real_demand_series,
    resolve_evaluation_start_index,
    resolve_real_demand_dataset_paths,
)


def test_load_real_demand_series_selects_bounded_subset_from_temp_csvs() -> None:
    tmp_path = Path(".tmp_real_demand_loader_tests") / uuid4().hex
    data_dir = tmp_path / "data"
    try:
        data_dir.mkdir(parents=True)
        (data_dir / "demand.csv").write_text(
            "Date,SKU1,SKU2,SKU3\n"
            "2020-01-01,10,4,20\n"
            "2020-01-02,12,5,18\n"
            "2020-01-03,8,6,22\n"
            "2020-01-04,10,5,19\n",
            encoding="utf-8",
        )
        (data_dir / "leadtime.csv").write_text(
            "Date,SKU1,SKU2,SKU3\n"
            "2020-01-01,2,3,5\n"
            "2020-01-02,2,3,5\n"
            "2020-01-03,2,4,5\n"
            "2020-01-04,2,3,6\n",
            encoding="utf-8",
        )
        (data_dir / "sku_list.csv").write_text(
            "# comment\nSKU\nSKU1\nSKU2\nSKU3\n",
            encoding="utf-8",
        )

        paths = resolve_real_demand_dataset_paths(
            dataset_name="temp_public_series",
            discovery_module="ignored_with_explicit_root",
            dataset_root=tmp_path,
            demand_csv_path=Path("data/demand.csv"),
            leadtime_csv_path=Path("data/leadtime.csv"),
            sku_list_path=Path("data/sku_list.csv"),
        )
        series = load_real_demand_series(
            paths=paths,
            selected_sku_count=2,
            demand_target_mean=10.0,
            subset_selection="nearest_benchmark_mean",
        )
        window = construct_backtest_window(
            total_length=len(series.dates),
            training_window_days=2,
            history_window_days=2,
            forecast_update_window_days=1,
            evaluation_horizon_days=2,
        )

        assert series.dataset_name == "temp_public_series"
        assert series.selected_skus == ("SKU1", "SKU2")
        assert len(series.demand_values) == 4
        assert window.evaluation_start_index == 2
        assert window.evaluation_end_index == 3
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_load_real_demand_series_accepts_explicit_selected_skus_and_start_date() -> None:
    tmp_path = Path(".tmp_real_demand_loader_tests") / uuid4().hex
    data_dir = tmp_path / "data"
    try:
        data_dir.mkdir(parents=True)
        (data_dir / "demand.csv").write_text(
            "Date,SKU1,SKU2,SKU3\n"
            "2020-01-01,10,4,20\n"
            "2020-01-02,12,5,18\n"
            "2020-01-03,8,6,22\n"
            "2020-01-04,10,5,19\n"
            "2020-01-05,11,4,18\n",
            encoding="utf-8",
        )
        (data_dir / "leadtime.csv").write_text(
            "Date,SKU1,SKU2,SKU3\n"
            "2020-01-01,2,3,5\n"
            "2020-01-02,2,3,5\n"
            "2020-01-03,2,4,5\n"
            "2020-01-04,2,3,6\n"
            "2020-01-05,2,3,5\n",
            encoding="utf-8",
        )
        (data_dir / "sku_list.csv").write_text("SKU\nSKU1\nSKU2\nSKU3\n", encoding="utf-8")

        paths = resolve_real_demand_dataset_paths(
            dataset_name="temp_public_series",
            discovery_module="ignored_with_explicit_root",
            dataset_root=tmp_path,
            demand_csv_path=Path("data/demand.csv"),
            leadtime_csv_path=Path("data/leadtime.csv"),
            sku_list_path=Path("data/sku_list.csv"),
        )
        series = load_real_demand_series(
            paths=paths,
            selected_sku_count=2,
            demand_target_mean=10.0,
            subset_selection="nearest_benchmark_mean",
            selected_skus=("SKU3", "SKU1"),
        )
        window = construct_backtest_window(
            total_length=len(series.dates),
            training_window_days=2,
            history_window_days=2,
            forecast_update_window_days=1,
            evaluation_horizon_days=2,
            evaluation_start_index=resolve_evaluation_start_index(
                dates=series.dates,
                evaluation_start_date="2020-01-04",
            ),
        )

        assert series.selected_skus == ("SKU3", "SKU1")
        assert series.selection_strategy == "explicit_selected_skus"
        assert window.training_start_index == 1
        assert window.evaluation_start_index == 3
        assert window.evaluation_end_index == 4
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
