from __future__ import annotations

import json
from pathlib import Path
import shutil
from uuid import uuid4

from meio.evaluation.logging_io import write_json
from meio.evaluation.real_demand_repeatability import (
    analyze_real_demand_repeatability,
    list_real_demand_run_dirs,
)
import scripts.analyze_real_demand_repeatability as repeatability_script


def _write_real_demand_run(
    run_dir: Path,
    *,
    config_hash: str,
    provider: str,
    baseline_cost: float,
    deterministic_cost: float,
    llm_cost: float,
    llm_fill_rate: float,
) -> None:
    write_json(
        run_dir / "experiment_metadata.json",
        {
            "config_hash": config_hash,
            "artifact_use_class": "internal_only",
            "validity_gate_passed": True,
            "provider": provider,
            "model_name": "gpt-4o-mini",
        },
    )
    write_json(
        run_dir / "dataset_summary.json",
        {
            "slice_name": run_dir.name,
            "dataset_name": "replenishmentenv_sku2778_store2",
            "date_range": ["2019/2/26", "2019/3/27"],
            "selected_skus": ["SKU27", "SKU47"],
        },
    )
    write_json(
        run_dir / "aggregate_summary.json",
        {
            "mode_summaries": [
                {
                    "mode": "deterministic_baseline",
                    "performance_summary": {
                        "average_total_cost": baseline_cost,
                        "average_fill_rate": 0.46,
                    },
                    "decision_quality": {"replan_rate": 0.6, "no_action_rate": 0.4},
                    "tool_use_summary": {"average_tool_call_count": 0.0},
                    "validity_summary": {
                        "fallback_count": 0,
                        "invalid_output_count": 0,
                    },
                },
                {
                    "mode": "deterministic_orchestrator",
                    "performance_summary": {
                        "average_total_cost": deterministic_cost,
                        "average_fill_rate": 0.53,
                    },
                    "decision_quality": {"replan_rate": 0.75, "no_action_rate": 0.0},
                    "tool_use_summary": {"average_tool_call_count": 90.0},
                    "validity_summary": {
                        "fallback_count": 0,
                        "invalid_output_count": 0,
                    },
                },
                {
                    "mode": "llm_orchestrator",
                    "performance_summary": {
                        "average_total_cost": llm_cost,
                        "average_fill_rate": llm_fill_rate,
                    },
                    "decision_quality": {"replan_rate": 0.83, "no_action_rate": 0.1},
                    "tool_use_summary": {"average_tool_call_count": 77.0},
                    "validity_summary": {
                        "fallback_count": 0,
                        "invalid_output_count": 0,
                    },
                },
            ]
        },
    )


def test_analyze_real_demand_repeatability_summarizes_costs_and_providers() -> None:
    tmp_root = Path(".tmp_real_demand_repeatability_tests") / uuid4().hex
    try:
        run_a = tmp_root / "run_a"
        run_b = tmp_root / "run_b"
        run_c = tmp_root / "run_c"
        run_a.mkdir(parents=True)
        run_b.mkdir(parents=True)
        run_c.mkdir(parents=True)
        _write_real_demand_run(
            run_a,
            config_hash="config123",
            provider="fake_llm_client",
            baseline_cost=8000.0,
            deterministic_cost=7400.0,
            llm_cost=7200.0,
            llm_fill_rate=0.55,
        )
        _write_real_demand_run(
            run_b,
            config_hash="config123",
            provider="openai",
            baseline_cost=8000.0,
            deterministic_cost=7400.0,
            llm_cost=7350.0,
            llm_fill_rate=0.54,
        )
        _write_real_demand_run(
            run_c,
            config_hash="config123",
            provider="openai",
            baseline_cost=8000.0,
            deterministic_cost=7400.0,
            llm_cost=7380.0,
            llm_fill_rate=0.545,
        )

        summary = analyze_real_demand_repeatability((run_a, run_b, run_c))

        assert summary.run_count == 3
        assert summary.panel_run_count == 0
        assert summary.panel_config_hashes == ()
        assert summary.config_hashes == ("config123",)
        assert summary.slice_names == ("run_a", "run_b", "run_c")
        assert summary.dataset_names == ("replenishmentenv_sku2778_store2",)
        assert summary.selected_sku_sets == (("SKU27", "SKU47"),)
        assert summary.deterministic_orchestrator_vs_baseline_label == "stably_better"
        assert summary.llm_vs_deterministic_orchestrator_label == "stably_better"
        assert summary.llm_validity_clean is True
        assert summary.slices_favoring_llm_orchestrator == ("run_a", "run_b", "run_c")
        assert summary.slices_favoring_deterministic_orchestrator == ()
        assert summary.competitive_slices == ()
        assert summary.mixed_slices == ()
        assert summary.fake_vs_real_llm_comparison is not None
        assert summary.fake_vs_real_llm_comparison.fake_run_count == 1
        assert summary.fake_vs_real_llm_comparison.real_run_count == 2
        assert summary.fake_vs_real_llm_comparison.fake_cost_mean == 7200.0
        assert summary.fake_vs_real_llm_comparison.real_cost_mean == 7365.0
        assert summary.fake_vs_real_llm_comparison.real_minus_fake_cost_mean == 165.0
        assert len(summary.slice_summaries) == 3

        llm_mode_summary = next(
            item for item in summary.mode_summaries if item.mode == "llm_orchestrator"
        )
        assert llm_mode_summary.tool_call_count.mean_value == 77.0
        assert llm_mode_summary.llm_providers == ("fake_llm_client", "openai")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_analyze_real_demand_repeatability_script_prints_json(
    monkeypatch,
    capsys,
) -> None:
    tmp_root = Path(".tmp_real_demand_repeatability_script_tests") / uuid4().hex
    try:
        run_dir = tmp_root / "run_a"
        run_dir.mkdir(parents=True)
        _write_real_demand_run(
            run_dir,
            config_hash="config123",
            provider="openai",
            baseline_cost=8000.0,
            deterministic_cost=7400.0,
            llm_cost=7350.0,
            llm_fill_rate=0.54,
        )
        monkeypatch.setattr(
            "sys.argv",
            [
                "analyze_real_demand_repeatability.py",
                "--run-dir",
                str(run_dir),
            ],
        )

        repeatability_script.main()
        payload = json.loads(capsys.readouterr().out)

        assert payload["run_count"] == 1
        assert payload["config_hashes"] == ["config123"]
        assert payload["llm_validity_clean"] is True
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_list_real_demand_run_dirs_expands_panel_root_into_slice_runs() -> None:
    tmp_root = Path(".tmp_real_demand_repeatability_panel_tests") / uuid4().hex
    try:
        panel_root = tmp_root / "panel_run"
        slice_a = panel_root / "slices" / "slice_a"
        slice_b = panel_root / "slices" / "slice_b"
        slice_a.mkdir(parents=True)
        slice_b.mkdir(parents=True)
        write_json(
            panel_root / "panel_summary.json",
            {"slice_count": 2, "panel_config_hash": "panel_hash_123"},
        )
        write_json(
            panel_root / "dataset_summary.json",
            {"panel_config_hash": "panel_hash_123", "slice_count": 2},
        )
        _write_real_demand_run(
            slice_a,
            config_hash="panel_config",
            provider="openai",
            baseline_cost=8000.0,
            deterministic_cost=7400.0,
            llm_cost=7350.0,
            llm_fill_rate=0.54,
        )
        _write_real_demand_run(
            slice_b,
            config_hash="panel_config",
            provider="openai",
            baseline_cost=8100.0,
            deterministic_cost=7450.0,
            llm_cost=7425.0,
            llm_fill_rate=0.541,
        )

        run_dirs = list_real_demand_run_dirs(results_root=tmp_root)

        assert run_dirs == (slice_a, slice_b)
        summary = analyze_real_demand_repeatability(results_root=tmp_root)
        assert summary.run_count == 2
        assert summary.panel_run_count == 1
        assert summary.panel_config_hashes == ("panel_hash_123",)
        assert summary.slice_names == ("slice_a", "slice_b")
        assert len(summary.slice_summaries) == 2
        first_slice = next(item for item in summary.slice_summaries if item.slice_name == "slice_a")
        assert first_slice.panel_config_hashes == ("panel_hash_123",)
        assert first_slice.llm_vs_deterministic_orchestrator_label == "stably_better"
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_analyze_real_demand_repeatability_groups_repeated_panel_slices() -> None:
    tmp_root = Path(".tmp_real_demand_repeatability_grouped_panel_tests") / uuid4().hex
    try:
        panel_a = tmp_root / "panel_a"
        panel_b = tmp_root / "panel_b"
        for panel_root in (panel_a, panel_b):
            write_json(
                panel_root / "panel_summary.json",
                {"slice_count": 2, "panel_config_hash": "panel_hash_456"},
            )
            write_json(
                panel_root / "dataset_summary.json",
                {"panel_config_hash": "panel_hash_456", "slice_count": 2},
            )
        slice_a_1 = panel_a / "slices" / "slice_a"
        slice_b_1 = panel_a / "slices" / "slice_b"
        slice_a_2 = panel_b / "slices" / "slice_a"
        slice_b_2 = panel_b / "slices" / "slice_b"
        for path in (slice_a_1, slice_b_1, slice_a_2, slice_b_2):
            path.mkdir(parents=True)

        _write_real_demand_run(
            slice_a_1,
            config_hash="slice_a_cfg",
            provider="openai",
            baseline_cost=100.0,
            deterministic_cost=90.0,
            llm_cost=80.0,
            llm_fill_rate=0.6,
        )
        _write_real_demand_run(
            slice_a_2,
            config_hash="slice_a_cfg",
            provider="openai",
            baseline_cost=105.0,
            deterministic_cost=92.0,
            llm_cost=84.0,
            llm_fill_rate=0.61,
        )
        _write_real_demand_run(
            slice_b_1,
            config_hash="slice_b_cfg",
            provider="openai",
            baseline_cost=100.0,
            deterministic_cost=90.0,
            llm_cost=95.0,
            llm_fill_rate=0.55,
        )
        _write_real_demand_run(
            slice_b_2,
            config_hash="slice_b_cfg",
            provider="openai",
            baseline_cost=100.0,
            deterministic_cost=88.0,
            llm_cost=86.0,
            llm_fill_rate=0.56,
        )

        summary = analyze_real_demand_repeatability((panel_a, panel_b))

        assert summary.panel_run_count == 2
        assert summary.panel_config_hashes == ("panel_hash_456",)
        assert summary.run_count == 4
        assert len(summary.slice_summaries) == 2
        assert summary.slices_favoring_llm_orchestrator == ("slice_a",)
        assert summary.slices_favoring_deterministic_orchestrator == ()
        assert summary.mixed_slices == ("slice_b",)
        slice_a = next(item for item in summary.slice_summaries if item.slice_name == "slice_a")
        slice_b = next(item for item in summary.slice_summaries if item.slice_name == "slice_b")
        assert slice_a.run_count == 2
        assert slice_a.llm_vs_deterministic_orchestrator_label == "stably_better"
        assert slice_b.run_count == 2
        assert slice_b.llm_vs_deterministic_orchestrator_label == "unstable"
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
