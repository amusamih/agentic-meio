from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import pytest

from meio.config.loaders import load_benchmark_config
from meio.data.stockpyl_adapter import StockpylSerialAdapter


REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(find_spec("stockpyl") is None, reason="stockpyl not installed")


def test_stockpyl_serial_adapter_builds_typed_instance_from_config() -> None:
    config = load_benchmark_config(REPO_ROOT / "configs/benchmark/serial_3_echelon.toml")

    instance = StockpylSerialAdapter().build_instance(config)

    assert instance.adapter_name == "stockpyl_serial"
    assert instance.stage_names == ("retailer", "regional_dc", "plant")
    assert instance.initial_inventory == (20, 30, 40)
    assert instance.demand_mean == 10.0
    assert instance.primary_inbound_stage_index == 2
    assert instance.primary_inbound_lead_time == 2.0
    assert instance.scenario_families[0].value == "normal"
