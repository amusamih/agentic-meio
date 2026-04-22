from __future__ import annotations

import json
from pathlib import Path
import shutil
import time
from uuid import uuid4

from meio.evaluation.failure_analysis import (
    analyze_paper_candidate_batch,
    latest_paper_candidate_batch_dir,
)
from meio.evaluation.logging_io import write_json, write_jsonl
import scripts.analyze_paper_candidate_failures as failure_script


def _write_minimal_batch(run_dir: Path) -> None:
    write_json(
        run_dir / "aggregate_summary.json",
        {
            "benchmark_id": "serial_3_echelon",
            "mode_summaries": [
                {"mode": "deterministic_baseline"},
                {"mode": "deterministic_orchestrator"},
                {"mode": "llm_orchestrator"},
            ],
        },
    )
    write_json(
        run_dir / "run_manifest.json",
        {
            "experiment_id": "paper_candidate_demo",
            "run_group_id": run_dir.name,
            "benchmark_id": "serial_3_echelon",
            "benchmark_source": "stockpyl_serial",
        },
    )
    write_jsonl(
        run_dir / "episode_summaries.jsonl",
        [
            {
                "schedule_name": "sustained_shift",
                "run_seed": 1,
                "mode": "deterministic_baseline",
                "total_cost": 100.0,
                "fill_rate": 0.5,
                "average_inventory": 12.0,
                "average_backorder_level": 3.0,
                "replan_count": 2,
                "no_action_count": 1,
                "intervention_count": 3,
                "tool_call_count": 0,
                "cost_breakdown": {
                    "holding_cost": 40.0,
                    "backlog_cost": 30.0,
                    "ordering_cost": 30.0,
                    "other_cost": 0.0,
                },
                "decision_quality": {"regime_prediction_accuracy": None},
            },
            {
                "schedule_name": "sustained_shift",
                "run_seed": 1,
                "mode": "deterministic_orchestrator",
                "total_cost": 105.0,
                "fill_rate": 0.5,
                "average_inventory": 10.0,
                "average_backorder_level": 4.0,
                "replan_count": 2,
                "no_action_count": 0,
                "intervention_count": 3,
                "tool_call_count": 6,
                "cost_breakdown": {
                    "holding_cost": 35.0,
                    "backlog_cost": 30.0,
                    "ordering_cost": 40.0,
                    "other_cost": 0.0,
                },
                "decision_quality": {"regime_prediction_accuracy": None},
            },
            {
                "schedule_name": "sustained_shift",
                "run_seed": 1,
                "mode": "llm_orchestrator",
                "total_cost": 110.0,
                "fill_rate": 0.5,
                "average_inventory": 9.0,
                "average_backorder_level": 5.0,
                "replan_count": 2,
                "no_action_count": 1,
                "intervention_count": 2,
                "tool_call_count": 4,
                "cost_breakdown": {
                    "holding_cost": 32.0,
                    "backlog_cost": 30.0,
                    "ordering_cost": 48.0,
                    "other_cost": 0.0,
                },
                "decision_quality": {"regime_prediction_accuracy": 1.0},
            },
        ],
    )
    write_jsonl(
        run_dir / "step_traces.jsonl",
        [
            {
                "schedule_name": "sustained_shift",
                "mode": "deterministic_baseline",
                "period_index": 1,
                "true_regime_label": "demand_regime_shift",
                "predicted_regime_label": None,
                "selected_subgoal": "request_replan",
                "selected_tools": [],
                "update_requests": ["reweight_scenarios"],
                "request_replan": True,
                "demand_outlook": 14.7,
                "leadtime_outlook": 2.0,
                "scenario_adjustment_summary": {"safety_buffer_scale": 1.0},
                "optimizer_orders": [10.0, 10.0, 10.0],
                "pipeline_by_echelon": [0.0, 0.0, 0.0],
                "inventory_by_echelon": [10.0, 20.0, 30.0],
                "backorders_by_echelon": [0.0, 0.0, 0.0],
                "per_period_cost": 50.0,
                "per_period_fill_rate": 0.5,
                "decision_changed_optimizer_input": True,
                "optimizer_output_changed_state": True,
                "intervention_changed_outcome": True,
            },
            {
                "schedule_name": "sustained_shift",
                "mode": "deterministic_orchestrator",
                "period_index": 1,
                "true_regime_label": "demand_regime_shift",
                "predicted_regime_label": None,
                "selected_subgoal": "request_replan",
                "selected_tools": ["forecast_tool", "leadtime_tool", "scenario_tool"],
                "update_requests": ["reweight_scenarios", "widen_uncertainty"],
                "request_replan": True,
                "demand_outlook": 15.862,
                "leadtime_outlook": 2.0,
                "scenario_adjustment_summary": {"safety_buffer_scale": 1.1781},
                "optimizer_orders": [12.0, 14.0, 18.0],
                "pipeline_by_echelon": [0.0, 0.0, 0.0],
                "inventory_by_echelon": [10.0, 20.0, 30.0],
                "backorders_by_echelon": [0.0, 0.0, 0.0],
                "per_period_cost": 52.0,
                "per_period_fill_rate": 0.5,
                "decision_changed_optimizer_input": True,
                "optimizer_output_changed_state": True,
                "intervention_changed_outcome": True,
            },
            {
                "schedule_name": "sustained_shift",
                "mode": "llm_orchestrator",
                "period_index": 1,
                "true_regime_label": "demand_regime_shift",
                "predicted_regime_label": "demand_regime_shift",
                "selected_subgoal": "request_replan",
                "selected_tools": ["forecast_tool", "scenario_tool"],
                "update_requests": ["switch_demand_regime", "widen_uncertainty"],
                "request_replan": True,
                "demand_outlook": 17.248,
                "leadtime_outlook": 2.0,
                "scenario_adjustment_summary": {"safety_buffer_scale": 1.21275},
                "optimizer_orders": [15.0, 18.0, 22.0],
                "pipeline_by_echelon": [0.0, 0.0, 0.0],
                "inventory_by_echelon": [10.0, 20.0, 30.0],
                "backorders_by_echelon": [0.0, 0.0, 0.0],
                "per_period_cost": 55.0,
                "per_period_fill_rate": 0.5,
                "decision_changed_optimizer_input": True,
                "optimizer_output_changed_state": True,
                "intervention_changed_outcome": True,
            },
        ],
    )
    write_jsonl(
        run_dir / "tool_call_traces.jsonl",
        [
            {
                "schedule_name": "sustained_shift",
                "mode": "deterministic_orchestrator",
                "tool_id": "scenario_tool",
                "optimizer_input_changed": True,
            },
            {
                "schedule_name": "sustained_shift",
                "mode": "llm_orchestrator",
                "tool_id": "forecast_tool",
                "optimizer_input_changed": False,
            },
            {
                "schedule_name": "sustained_shift",
                "mode": "llm_orchestrator",
                "tool_id": "scenario_tool",
                "optimizer_input_changed": True,
            },
        ],
    )


def test_latest_paper_candidate_batch_dir_returns_latest_directory() -> None:
    root = Path(".tmp_failure_analysis_tests") / uuid4().hex / "results"
    first = root / "batch_a"
    second = root / "batch_b"
    try:
        first.mkdir(parents=True)
        second.mkdir(parents=True)
        time.sleep(0.02)
        (second / ".marker").write_text("latest", encoding="utf-8")

        latest = latest_paper_candidate_batch_dir(root)

        assert latest == second
    finally:
        if root.parent.exists():
            shutil.rmtree(root.parent, ignore_errors=True)


def test_analyze_paper_candidate_batch_extracts_schedule_level_deltas() -> None:
    run_dir = Path(".tmp_failure_analysis_tests") / uuid4().hex / "results" / "paper_candidate_demo"
    try:
        run_dir.mkdir(parents=True)
        _write_minimal_batch(run_dir)

        analysis = analyze_paper_candidate_batch(run_dir)

        assert analysis.benchmark_id == "serial_3_echelon"
        assert analysis.schedule_names == ("sustained_shift",)
        comparison = analysis.schedule_comparisons[0]
        assert comparison.schedule_name == "sustained_shift"
        assert comparison.llm_cost_rank == 3
        assert comparison.llm_vs_baseline_total_cost_delta == 10.0
        assert comparison.llm_vs_deterministic_orchestrator_total_cost_delta == 5.0
        assert comparison.llm_vs_deterministic_orchestrator_ordering_cost_delta == 8.0
        assert comparison.period_highlights[0].llm_order_sum_delta_vs_baseline == 25.0
    finally:
        if run_dir.parents[2].exists():
            shutil.rmtree(run_dir.parents[2], ignore_errors=True)


def test_analyze_paper_candidate_failures_script_prints_json(monkeypatch, capsys) -> None:
    run_dir = Path(".tmp_failure_analysis_tests") / uuid4().hex / "results" / "paper_candidate_demo"
    try:
        run_dir.mkdir(parents=True)
        _write_minimal_batch(run_dir)
        monkeypatch.setattr(
            "sys.argv",
            [
                "analyze_paper_candidate_failures.py",
                "--run-dir",
                str(run_dir),
            ],
        )

        failure_script.main()
        output = capsys.readouterr().out
        payload = json.loads(output)

        assert payload["benchmark_id"] == "serial_3_echelon"
        assert payload["schedule_comparisons"][0]["schedule_name"] == "sustained_shift"
        assert payload["schedule_comparisons"][0]["llm_vs_baseline_total_cost_delta"] == 10.0
    finally:
        if run_dir.parents[2].exists():
            shutil.rmtree(run_dir.parents[2], ignore_errors=True)
