from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
from uuid import uuid4

from meio.evaluation.broad_eval_analysis import analyze_frozen_broad_batch
from scripts import analyze_frozen_broad_eval


def test_analyze_frozen_broad_batch_groups_schedule_families_and_outcomes() -> None:
    run_dir = Path(".tmp_broad_eval_analysis") / uuid4().hex
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        aggregate_payload = {
            "benchmark_id": "serial_3_echelon",
            "mode_names": [
                "deterministic_baseline",
                "deterministic_orchestrator",
                "llm_orchestrator",
            ],
            "schedule_names": [
                "adjacent_case",
                "relapse_case",
                "false_alarm_case",
            ],
            "seed_values": [1, 2, 3],
            "mode_summaries": [
                {
                    "mode": "deterministic_baseline",
                    "artifact_use_class": "paper_candidate",
                    "performance_summary": {
                        "average_total_cost": 100.0,
                        "average_fill_rate": 0.5,
                        "average_inventory": 10.0,
                        "average_backorder_level": 4.0,
                    },
                    "decision_quality": {
                        "regime_prediction_accuracy": None,
                        "no_action_rate": 0.6,
                        "replan_rate": 0.4,
                        "intervention_rate": 1.0,
                    },
                    "telemetry_metrics": {
                        "fallback_count": 0,
                        "invalid_output_count": 0,
                    },
                    "tool_use_summary": {
                        "average_tool_call_count": 0.0,
                        "repeated_stress_moderation_count": 0,
                        "relapse_moderation_count": 0,
                        "unresolved_stress_moderation_count": 0,
                        "moderated_update_count": 0,
                    },
                    "schedule_breakdown": [
                        _schedule_summary("adjacent_case", 120.0, 0.5, 0.0, 0, 0, 0, 0, None),
                        _schedule_summary("relapse_case", 130.0, 0.5, 0.0, 0, 0, 0, 0, None),
                        _schedule_summary("false_alarm_case", 110.0, 0.6, 0.0, 0, 0, 0, 0, None),
                    ],
                },
                {
                    "mode": "deterministic_orchestrator",
                    "artifact_use_class": "paper_candidate",
                    "performance_summary": {
                        "average_total_cost": 105.0,
                        "average_fill_rate": 0.5,
                        "average_inventory": 11.0,
                        "average_backorder_level": 4.5,
                    },
                    "decision_quality": {
                        "regime_prediction_accuracy": None,
                        "no_action_rate": 0.0,
                        "replan_rate": 0.6,
                        "intervention_rate": 1.0,
                    },
                    "telemetry_metrics": {
                        "fallback_count": 0,
                        "invalid_output_count": 0,
                    },
                    "tool_use_summary": {
                        "average_tool_call_count": 9.0,
                        "repeated_stress_moderation_count": 0,
                        "relapse_moderation_count": 0,
                        "unresolved_stress_moderation_count": 0,
                        "moderated_update_count": 0,
                    },
                    "schedule_breakdown": [
                        _schedule_summary("adjacent_case", 115.0, 0.5, 9.0, 1, 0, 0, 1, None),
                        _schedule_summary("relapse_case", 125.0, 0.5, 9.0, 0, 0, 0, 0, None),
                        _schedule_summary("false_alarm_case", 111.0, 0.6, 9.0, 0, 0, 0, 0, None),
                    ],
                },
                {
                    "mode": "llm_orchestrator",
                    "artifact_use_class": "paper_candidate",
                    "performance_summary": {
                        "average_total_cost": 102.0,
                        "average_fill_rate": 0.5,
                        "average_inventory": 10.5,
                        "average_backorder_level": 4.1,
                    },
                    "decision_quality": {
                        "regime_prediction_accuracy": 1.0,
                        "no_action_rate": 0.4,
                        "replan_rate": 0.5,
                        "intervention_rate": 0.6,
                    },
                    "telemetry_metrics": {
                        "fallback_count": 0,
                        "invalid_output_count": 0,
                    },
                    "tool_use_summary": {
                        "average_tool_call_count": 5.0,
                        "repeated_stress_moderation_count": 2,
                        "relapse_moderation_count": 1,
                        "unresolved_stress_moderation_count": 1,
                        "moderated_update_count": 3,
                    },
                    "schedule_breakdown": [
                        _schedule_summary("adjacent_case", 110.0, 0.5, 5.0, 2, 0, 0, 2, 1.0),
                        _schedule_summary("relapse_case", 135.0, 0.5, 6.0, 0, 2, 2, 2, 1.0),
                        _schedule_summary("false_alarm_case", 111.0, 0.6, 4.0, 0, 0, 0, 0, 1.0),
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
        episode_rows = [
            _episode_row("adjacent_case", ["normal", "demand_regime_shift", "demand_regime_shift"]),
            _episode_row(
                "relapse_case",
                ["normal", "demand_regime_shift", "recovery", "demand_regime_shift"],
            ),
            _episode_row(
                "false_alarm_case",
                ["normal", "recovery", "demand_regime_shift", "recovery"],
            ),
        ]
        (run_dir / "aggregate_summary.json").write_text(
            json.dumps(aggregate_payload),
            encoding="utf-8",
        )
        (run_dir / "run_manifest.json").write_text(
            json.dumps(manifest_payload),
            encoding="utf-8",
        )
        (run_dir / "episode_summaries.jsonl").write_text(
            "\n".join(json.dumps(row) for row in episode_rows),
            encoding="utf-8",
        )

        analysis = analyze_frozen_broad_batch(run_dir)

        assert analysis.artifact_use_class == "paper_candidate"
        assert analysis.llm_help_schedules == ("adjacent_case",)
        assert analysis.llm_hurt_schedules == ("relapse_case",)
        assert analysis.llm_neutral_schedules == ("false_alarm_case",)
        family_by_name = {
            summary.schedule_family: summary for summary in analysis.family_summaries
        }
        assert family_by_name["adjacent_repeated_stress"].llm_help_count == 1
        assert family_by_name["relapse_or_gap"].llm_hurt_count == 1
        assert family_by_name["false_alarm_mixed"].llm_neutral_count == 1
    finally:
        if run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)


def test_analyze_frozen_broad_eval_script_prints_json(capsys: object) -> None:
    run_dir = Path(".tmp_broad_eval_analysis") / uuid4().hex
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "aggregate_summary.json").write_text(
            json.dumps(
                {
                    "benchmark_id": "serial_3_echelon",
                    "mode_names": ["deterministic_baseline", "deterministic_orchestrator", "llm_orchestrator"],
                    "schedule_names": ["simple_case"],
                    "seed_values": [1],
                    "mode_summaries": [
                        {
                            "mode": "deterministic_baseline",
                            "artifact_use_class": "paper_candidate",
                            "performance_summary": {"average_total_cost": 10.0},
                            "decision_quality": {"no_action_rate": 1.0, "replan_rate": 0.0, "intervention_rate": 0.0},
                            "telemetry_metrics": {"fallback_count": 0, "invalid_output_count": 0},
                            "tool_use_summary": {
                                "average_tool_call_count": 0.0,
                                "repeated_stress_moderation_count": 0,
                                "relapse_moderation_count": 0,
                                "unresolved_stress_moderation_count": 0,
                                "moderated_update_count": 0,
                            },
                            "schedule_breakdown": [_schedule_summary("simple_case", 10.0, 0.5, 0.0, 0, 0, 0, 0, None)],
                        },
                        {
                            "mode": "deterministic_orchestrator",
                            "artifact_use_class": "paper_candidate",
                            "performance_summary": {"average_total_cost": 11.0},
                            "decision_quality": {"no_action_rate": 0.0, "replan_rate": 1.0, "intervention_rate": 1.0},
                            "telemetry_metrics": {"fallback_count": 0, "invalid_output_count": 0},
                            "tool_use_summary": {
                                "average_tool_call_count": 9.0,
                                "repeated_stress_moderation_count": 0,
                                "relapse_moderation_count": 0,
                                "unresolved_stress_moderation_count": 0,
                                "moderated_update_count": 0,
                            },
                            "schedule_breakdown": [_schedule_summary("simple_case", 11.0, 0.5, 9.0, 0, 0, 0, 0, None)],
                        },
                        {
                            "mode": "llm_orchestrator",
                            "artifact_use_class": "paper_candidate",
                            "performance_summary": {"average_total_cost": 9.0},
                            "decision_quality": {
                                "regime_prediction_accuracy": 1.0,
                                "no_action_rate": 0.5,
                                "replan_rate": 0.5,
                                "intervention_rate": 0.5,
                            },
                            "telemetry_metrics": {"fallback_count": 0, "invalid_output_count": 0},
                            "tool_use_summary": {
                                "average_tool_call_count": 3.0,
                                "repeated_stress_moderation_count": 0,
                                "relapse_moderation_count": 0,
                                "unresolved_stress_moderation_count": 0,
                                "moderated_update_count": 0,
                            },
                            "schedule_breakdown": [_schedule_summary("simple_case", 9.0, 0.5, 3.0, 0, 0, 0, 0, 1.0)],
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
        (run_dir / "episode_summaries.jsonl").write_text(
            json.dumps(_episode_row("simple_case", ["normal", "demand_regime_shift", "recovery"])),
            encoding="utf-8",
        )

        original_argv = sys.argv
        sys.argv = ["analyze_frozen_broad_eval.py", "--run-dir", str(run_dir)]
        try:
            analyze_frozen_broad_eval.main()
        finally:
            sys.argv = original_argv

        payload = json.loads(capsys.readouterr().out)
        assert payload["artifact_use_class"] == "paper_candidate"
        assert payload["llm_help_schedules"] == ["simple_case"]
    finally:
        if run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)


def _schedule_summary(
    schedule_name: str,
    total_cost: float,
    fill_rate: float,
    tool_call_count: float,
    repeated_count: int,
    relapse_count: int,
    unresolved_count: int,
    moderated_count: int,
    regime_accuracy: float | None,
) -> dict[str, object]:
    return {
        "schedule_name": schedule_name,
        "performance_summary": {
            "average_total_cost": total_cost,
            "average_fill_rate": fill_rate,
            "average_inventory": 10.0,
            "average_backorder_level": 4.0,
        },
        "decision_quality": {
            "regime_prediction_accuracy": regime_accuracy,
            "no_action_rate": 0.5,
            "replan_rate": 0.5,
            "intervention_rate": 0.5,
        },
        "telemetry_metrics": {
            "fallback_count": 0,
            "invalid_output_count": 0,
        },
        "tool_use_summary": {
            "average_tool_call_count": tool_call_count,
            "repeated_stress_moderation_count": repeated_count,
            "relapse_moderation_count": relapse_count,
            "unresolved_stress_moderation_count": unresolved_count,
            "moderated_update_count": moderated_count,
        },
    }


def _episode_row(schedule_name: str, labels: list[str]) -> dict[str, object]:
    return {
        "schedule_name": schedule_name,
        "regime_schedule": labels,
    }
