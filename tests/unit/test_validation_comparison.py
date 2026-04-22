from __future__ import annotations

from pathlib import Path
import shutil
from uuid import uuid4

from meio.evaluation.logging_io import write_json
from meio.evaluation.validation_comparison import summarize_validation_stack


def test_summarize_validation_stack_reads_internal_and_public_lane_artifacts(
) -> None:
    tmp_path = Path(".tmp_validation_comparison_tests") / uuid4().hex
    internal_dir = tmp_path / "results" / "stockpyl_serial_paper_candidate" / "run_a"
    public_dir = tmp_path / "results" / "public_benchmark_eval" / "run_b"
    try:
        internal_dir.mkdir(parents=True)
        public_dir.mkdir(parents=True)

        write_json(
            internal_dir / "run_manifest.json",
            {
                "run_group_id": "run_a",
                "experiment_id": "paper_candidate",
                "benchmark_id": "serial_3_echelon",
                "benchmark_source": "stockpyl_serial",
            },
        )
        write_json(
            internal_dir / "experiment_metadata.json",
            {
                "experiment_id": "paper_candidate",
                "benchmark_source": "stockpyl_serial",
                "validation_lane": "stockpyl_internal",
                "artifact_use_class": "paper_candidate",
                "validity_gate_passed": True,
            },
        )
        write_json(
            internal_dir / "aggregate_summary.json",
            {
                "artifact_use_class": "paper_candidate",
                "validity_gate_passed": True,
                "mode_summaries": [
                    {
                        "mode": "llm_orchestrator",
                        "performance_summary": {"average_total_cost": 320.0, "average_fill_rate": 1.0},
                        "decision_quality": {"regime_prediction_accuracy": 0.95},
                        "validity_summary": {"invalid_output_count": 0, "fallback_count": 0},
                    }
                ],
            },
        )

        write_json(
            public_dir / "run_manifest.json",
            {
                "run_group_id": "run_b",
                "experiment_id": "public_benchmark_eval",
                "benchmark_id": "replenishment_env",
                "benchmark_source": "public_benchmark",
            },
        )
        write_json(
            public_dir / "experiment_metadata.json",
            {
                "experiment_id": "public_benchmark_eval",
                "benchmark_source": "public_benchmark",
                "validation_lane": "public_benchmark",
                "artifact_use_class": "internal_only",
                "validity_gate_passed": False,
            },
        )
        write_json(
            public_dir / "public_benchmark_summary.json",
            {
                "environment_runnable": False,
                "blocked_reason": "ModuleNotFoundError: demo",
            },
        )

        summary = summarize_validation_stack((internal_dir, public_dir))

        assert {item.lane for item in summary.artifact_summaries} == {
            "stockpyl_internal",
            "public_benchmark",
        }
        assert {item.lane: item.status for item in summary.artifact_summaries} == {
            "stockpyl_internal": "completed",
            "public_benchmark": "blocked_partial",
        }
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_summarize_validation_stack_reads_completed_public_benchmark_rewards() -> None:
    tmp_path = Path(".tmp_validation_comparison_tests") / uuid4().hex
    public_dir = tmp_path / "results" / "public_benchmark_eval" / "run_c"
    try:
        public_dir.mkdir(parents=True)
        write_json(
            public_dir / "run_manifest.json",
            {
                "run_group_id": "run_c",
                "experiment_id": "public_benchmark_eval",
                "benchmark_id": "replenishment_env:sku50.single_store.standard",
                "benchmark_source": "public_benchmark",
            },
        )
        write_json(
            public_dir / "experiment_metadata.json",
            {
                "experiment_id": "public_benchmark_eval",
                "benchmark_source": "public_benchmark",
                "validation_lane": "public_benchmark",
                "artifact_use_class": "internal_only",
                "validity_gate_passed": True,
            },
        )
        write_json(
            public_dir / "aggregate_summary.json",
            {
                "artifact_use_class": "internal_only",
                "validity_gate_passed": True,
                "mode_summaries": [
                    {
                        "mode": "deterministic_orchestrator",
                        "performance_summary": {"average_fill_rate": 0.63},
                        "decision_quality": {},
                        "validity_summary": {"invalid_output_count": 0, "fallback_count": 0},
                    }
                ],
            },
        )
        write_json(
            public_dir / "public_benchmark_summary.json",
            {
                "bounded_orchestrator_integration_supported": True,
                "blocked_reason": None,
                "mode_summaries": [
                    {
                        "mode": "deterministic_orchestrator",
                        "total_reward": 31086.72,
                    }
                ],
            },
        )

        summary = summarize_validation_stack((public_dir,))

        assert summary.artifact_summaries[0].lane == "public_benchmark"
        assert summary.artifact_summaries[0].status == "completed"
        assert summary.artifact_summaries[0].mode_summaries[0].average_total_reward == 31086.72
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
