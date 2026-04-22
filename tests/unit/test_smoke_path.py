from __future__ import annotations

from scripts.run_smoke_path import run_smoke_path
from meio.optimization.contracts import OptimizationResult


def test_smoke_path_runs_end_to_end_with_typed_boundaries() -> None:
    smoke_run = run_smoke_path()

    assert smoke_run.benchmark_case.benchmark_id == "serial_3_echelon"
    assert smoke_run.orchestration_response.signal.tool_sequence == (
        "forecast_tool",
        "leadtime_tool",
        "scenario_tool",
    )
    assert smoke_run.orchestration_response.signal.request_replan is False
    assert smoke_run.trace.optimization_results[0].replenishment_orders == (0.0, 0.0, 0.0)


def test_smoke_path_keeps_raw_orders_inside_optimization_results() -> None:
    smoke_run = run_smoke_path()

    assert all(tool_result.emits_raw_orders is False for tool_result in smoke_run.orchestration_response.tool_results)
    assert all(
        not any(isinstance(value, OptimizationResult) for value in tool_result.structured_output.values())
        for tool_result in smoke_run.orchestration_response.tool_results
    )
