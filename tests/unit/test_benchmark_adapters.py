from __future__ import annotations

import meio.data.benchmark_adapters as benchmark_adapters


def test_stockpyl_adapter_reports_smoke_testable_when_module_is_available(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        benchmark_adapters,
        "_module_available",
        lambda module_name: module_name == "stockpyl",
    )

    status = benchmark_adapters.StockpylSerialAdapterBoundary().describe()

    assert status.candidate_id == "stockpyl_serial"
    assert status.smoke_testable_now is True
    assert status.available_modules == ("stockpyl",)
    assert status.missing_modules == ()


def test_primary_benchmark_adapter_returns_stockpyl_adapter() -> None:
    adapter = benchmark_adapters.primary_benchmark_adapter()

    assert adapter.adapter_name == "stockpyl_serial"


def test_orgym_adapter_reports_partial_integration_when_related_module_is_present(
    monkeypatch,
) -> None:
    availability = {"or_gym": False, "or_gym_inventory": True}
    monkeypatch.setattr(
        benchmark_adapters,
        "_module_available",
        lambda module_name: availability.get(module_name, False),
    )

    status = benchmark_adapters.OrGymInventoryAdapterBoundary().describe()

    assert status.smoke_testable_now is False
    assert status.available_modules == ("or_gym_inventory",)
    assert status.missing_modules == ("or_gym",)


def test_mabim_adapter_reports_unavailable_when_module_is_missing(monkeypatch) -> None:
    monkeypatch.setattr(benchmark_adapters, "_module_available", lambda module_name: False)

    status = benchmark_adapters.MabimAdapterBoundary().describe()

    assert status.smoke_testable_now is False
    assert status.available_modules == ()
    assert status.missing_modules == ("mabim",)
