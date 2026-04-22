from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
from uuid import uuid4

import pytest

from meio.evaluation.heldout_analysis import analyze_heldout_batch
from scripts import analyze_heldout_results


def test_analyze_heldout_batch_labels_help_hurt_and_neutral() -> None:
    aggregate_payload = {
        "benchmark_id": "serial_3_echelon",
        "mode_names": [
            "deterministic_baseline",
            "deterministic_orchestrator",
            "llm_orchestrator",
        ],
        "schedule_names": [
            "help_case",
            "hurt_case",
            "neutral_case",
        ],
        "seed_values": [1, 2],
        "mode_summaries": [
            {
                "mode": "deterministic_baseline",
                "schedule_breakdown": [
                    {
                        "schedule_name": "help_case",
                        "performance_summary": {
                            "average_total_cost": 220.0,
                            "average_fill_rate": 0.6,
                            "average_inventory": 10.0,
                            "average_backorder_level": 4.0,
                        },
                        "decision_quality": {
                            "regime_prediction_accuracy": None,
                            "no_action_rate": 0.5,
                            "replan_rate": 0.5,
                            "intervention_rate": 1.0,
                        },
                        "telemetry_metrics": {
                            "fallback_count": 0,
                            "invalid_output_count": 0,
                        },
                        "tool_use_summary": {
                            "average_tool_call_count": 0.0,
                            "repeated_stress_moderation_count": 0,
                            "moderated_update_count": 0,
                        },
                    },
                    {
                        "schedule_name": "hurt_case",
                        "performance_summary": {
                            "average_total_cost": 210.0,
                            "average_fill_rate": 0.6,
                            "average_inventory": 10.0,
                            "average_backorder_level": 4.0,
                        },
                        "decision_quality": {
                            "regime_prediction_accuracy": None,
                            "no_action_rate": 0.5,
                            "replan_rate": 0.5,
                            "intervention_rate": 1.0,
                        },
                        "telemetry_metrics": {
                            "fallback_count": 0,
                            "invalid_output_count": 0,
                        },
                        "tool_use_summary": {
                            "average_tool_call_count": 0.0,
                            "repeated_stress_moderation_count": 0,
                            "moderated_update_count": 0,
                        },
                    },
                    {
                        "schedule_name": "neutral_case",
                        "performance_summary": {
                            "average_total_cost": 200.0,
                            "average_fill_rate": 0.6,
                            "average_inventory": 10.0,
                            "average_backorder_level": 4.0,
                        },
                        "decision_quality": {
                            "regime_prediction_accuracy": None,
                            "no_action_rate": 0.5,
                            "replan_rate": 0.5,
                            "intervention_rate": 1.0,
                        },
                        "telemetry_metrics": {
                            "fallback_count": 0,
                            "invalid_output_count": 0,
                        },
                        "tool_use_summary": {
                            "average_tool_call_count": 0.0,
                            "repeated_stress_moderation_count": 0,
                            "moderated_update_count": 0,
                        },
                    },
                ],
            },
            {
                "mode": "deterministic_orchestrator",
                "schedule_breakdown": [
                    {
                        "schedule_name": "help_case",
                        "performance_summary": {
                            "average_total_cost": 215.0,
                            "average_fill_rate": 0.6,
                            "average_inventory": 10.0,
                            "average_backorder_level": 4.0,
                        },
                        "decision_quality": {
                            "regime_prediction_accuracy": None,
                            "no_action_rate": 0.0,
                            "replan_rate": 0.5,
                            "intervention_rate": 1.0,
                        },
                        "telemetry_metrics": {
                            "fallback_count": 0,
                            "invalid_output_count": 0,
                        },
                        "tool_use_summary": {
                            "average_tool_call_count": 9.0,
                            "repeated_stress_moderation_count": 0,
                            "moderated_update_count": 0,
                        },
                    },
                    {
                        "schedule_name": "hurt_case",
                        "performance_summary": {
                            "average_total_cost": 230.0,
                            "average_fill_rate": 0.6,
                            "average_inventory": 10.0,
                            "average_backorder_level": 4.0,
                        },
                        "decision_quality": {
                            "regime_prediction_accuracy": None,
                            "no_action_rate": 0.0,
                            "replan_rate": 0.5,
                            "intervention_rate": 1.0,
                        },
                        "telemetry_metrics": {
                            "fallback_count": 0,
                            "invalid_output_count": 0,
                        },
                        "tool_use_summary": {
                            "average_tool_call_count": 9.0,
                            "repeated_stress_moderation_count": 0,
                            "moderated_update_count": 0,
                        },
                    },
                    {
                        "schedule_name": "neutral_case",
                        "performance_summary": {
                            "average_total_cost": 180.0,
                            "average_fill_rate": 0.6,
                            "average_inventory": 10.0,
                            "average_backorder_level": 4.0,
                        },
                        "decision_quality": {
                            "regime_prediction_accuracy": None,
                            "no_action_rate": 0.0,
                            "replan_rate": 0.5,
                            "intervention_rate": 1.0,
                        },
                        "telemetry_metrics": {
                            "fallback_count": 0,
                            "invalid_output_count": 0,
                        },
                        "tool_use_summary": {
                            "average_tool_call_count": 9.0,
                            "repeated_stress_moderation_count": 0,
                            "moderated_update_count": 0,
                        },
                    },
                ],
            },
            {
                "mode": "llm_orchestrator",
                "schedule_breakdown": [
                    {
                        "schedule_name": "help_case",
                        "performance_summary": {
                            "average_total_cost": 200.0,
                            "average_fill_rate": 0.6,
                            "average_inventory": 9.0,
                            "average_backorder_level": 4.0,
                        },
                        "decision_quality": {
                            "regime_prediction_accuracy": 1.0,
                            "no_action_rate": 0.3,
                            "replan_rate": 0.4,
                            "intervention_rate": 0.7,
                        },
                        "telemetry_metrics": {
                            "fallback_count": 0,
                            "invalid_output_count": 0,
                        },
                        "tool_use_summary": {
                            "average_tool_call_count": 6.0,
                            "repeated_stress_moderation_count": 1,
                            "moderated_update_count": 1,
                        },
                    },
                    {
                        "schedule_name": "hurt_case",
                        "performance_summary": {
                            "average_total_cost": 250.0,
                            "average_fill_rate": 0.6,
                            "average_inventory": 12.0,
                            "average_backorder_level": 4.0,
                        },
                        "decision_quality": {
                            "regime_prediction_accuracy": 1.0,
                            "no_action_rate": 0.3,
                            "replan_rate": 0.4,
                            "intervention_rate": 0.7,
                        },
                        "telemetry_metrics": {
                            "fallback_count": 0,
                            "invalid_output_count": 0,
                        },
                        "tool_use_summary": {
                            "average_tool_call_count": 6.0,
                            "repeated_stress_moderation_count": 2,
                            "moderated_update_count": 2,
                        },
                    },
                    {
                        "schedule_name": "neutral_case",
                        "performance_summary": {
                            "average_total_cost": 190.0,
                            "average_fill_rate": 0.6,
                            "average_inventory": 11.0,
                            "average_backorder_level": 4.0,
                        },
                        "decision_quality": {
                            "regime_prediction_accuracy": 1.0,
                            "no_action_rate": 0.3,
                            "replan_rate": 0.4,
                            "intervention_rate": 0.7,
                        },
                        "telemetry_metrics": {
                            "fallback_count": 0,
                            "invalid_output_count": 0,
                        },
                        "tool_use_summary": {
                            "average_tool_call_count": 6.0,
                            "repeated_stress_moderation_count": 1,
                            "moderated_update_count": 1,
                        },
                    },
                ],
            },
        ],
    }
    manifest_payload = {
        "artifact_use_class": "paper_candidate",
        "mode_artifact_use_classes": [
            ["deterministic_baseline", "paper_candidate"],
            ["deterministic_orchestrator", "paper_candidate"],
            ["llm_orchestrator", "paper_candidate"],
        ],
    }
    output_root = Path(".tmp_heldout_analysis_tests") / uuid4().hex
    try:
        run_dir = output_root / "heldout_run"
        run_dir.mkdir(parents=True)
        (run_dir / "aggregate_summary.json").write_text(
            json.dumps(aggregate_payload),
            encoding="utf-8",
        )
        (run_dir / "run_manifest.json").write_text(
            json.dumps(manifest_payload),
            encoding="utf-8",
        )

        analysis = analyze_heldout_batch(run_dir)

        assert analysis.artifact_use_class == "paper_candidate"
        assert analysis.llm_help_schedules == ("help_case",)
        assert analysis.llm_hurt_schedules == ("hurt_case",)
        assert analysis.llm_neutral_schedules == ("neutral_case",)
        help_summary = next(
            comparison
            for comparison in analysis.schedule_comparisons
            if comparison.schedule_name == "help_case"
        )
        assert help_summary.llm_outcome_label == "helps"
        assert help_summary.llm_cost_rank == 1
    finally:
        if output_root.exists():
            shutil.rmtree(output_root, ignore_errors=True)


def test_analyze_heldout_results_script_prints_json_summary(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = Path(".tmp_heldout_analysis_tests") / uuid4().hex
    try:
        run_dir = output_root / "heldout_run"
        run_dir.mkdir(parents=True)
        (run_dir / "aggregate_summary.json").write_text(
            json.dumps(
                {
                    "benchmark_id": "serial_3_echelon",
                    "mode_names": [
                        "deterministic_baseline",
                        "deterministic_orchestrator",
                        "llm_orchestrator",
                    ],
                    "schedule_names": ["single_case"],
                    "seed_values": [1, 2],
                    "mode_summaries": [
                        {
                            "mode": "deterministic_baseline",
                            "schedule_breakdown": [
                                {
                                    "schedule_name": "single_case",
                                    "performance_summary": {
                                        "average_total_cost": 200.0,
                                        "average_fill_rate": 0.5,
                                        "average_inventory": 10.0,
                                        "average_backorder_level": 2.0,
                                    },
                                    "decision_quality": {
                                        "regime_prediction_accuracy": None,
                                        "no_action_rate": 0.5,
                                        "replan_rate": 0.5,
                                        "intervention_rate": 1.0,
                                    },
                                    "telemetry_metrics": {
                                        "fallback_count": 0,
                                        "invalid_output_count": 0,
                                    },
                                    "tool_use_summary": {
                                        "average_tool_call_count": 0.0,
                                        "repeated_stress_moderation_count": 0,
                                        "moderated_update_count": 0,
                                    },
                                }
                            ],
                        },
                        {
                            "mode": "deterministic_orchestrator",
                            "schedule_breakdown": [
                                {
                                    "schedule_name": "single_case",
                                    "performance_summary": {
                                        "average_total_cost": 190.0,
                                        "average_fill_rate": 0.5,
                                        "average_inventory": 9.0,
                                        "average_backorder_level": 2.0,
                                    },
                                    "decision_quality": {
                                        "regime_prediction_accuracy": None,
                                        "no_action_rate": 0.0,
                                        "replan_rate": 0.5,
                                        "intervention_rate": 1.0,
                                    },
                                    "telemetry_metrics": {
                                        "fallback_count": 0,
                                        "invalid_output_count": 0,
                                    },
                                    "tool_use_summary": {
                                        "average_tool_call_count": 9.0,
                                        "repeated_stress_moderation_count": 0,
                                        "moderated_update_count": 0,
                                    },
                                }
                            ],
                        },
                        {
                            "mode": "llm_orchestrator",
                            "schedule_breakdown": [
                                {
                                    "schedule_name": "single_case",
                                    "performance_summary": {
                                        "average_total_cost": 210.0,
                                        "average_fill_rate": 0.5,
                                        "average_inventory": 11.0,
                                        "average_backorder_level": 2.0,
                                    },
                                    "decision_quality": {
                                        "regime_prediction_accuracy": 1.0,
                                        "no_action_rate": 0.2,
                                        "replan_rate": 0.3,
                                        "intervention_rate": 0.8,
                                    },
                                    "telemetry_metrics": {
                                        "fallback_count": 0,
                                        "invalid_output_count": 0,
                                    },
                                    "tool_use_summary": {
                                        "average_tool_call_count": 6.0,
                                        "repeated_stress_moderation_count": 1,
                                        "moderated_update_count": 1,
                                    },
                                }
                            ],
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "artifact_use_class": "paper_candidate",
                    "mode_artifact_use_classes": [
                        ["deterministic_baseline", "paper_candidate"],
                        ["deterministic_orchestrator", "paper_candidate"],
                        ["llm_orchestrator", "paper_candidate"],
                    ],
                }
            ),
            encoding="utf-8",
        )

        monkeypatch.setattr(
            sys,
            "argv",
            ["analyze_heldout_results.py", "--run-dir", str(run_dir)],
        )

        analyze_heldout_results.main()
        output = json.loads(capsys.readouterr().out)

        assert output["artifact_use_class"] == "paper_candidate"
        assert output["llm_hurt_schedules"] == ["single_case"]
    finally:
        if output_root.exists():
            shutil.rmtree(output_root, ignore_errors=True)
