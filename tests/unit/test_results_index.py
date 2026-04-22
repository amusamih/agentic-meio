from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

from meio.evaluation.logging_io import write_json
from meio.evaluation.logging_schema import ArtifactUseClass
from meio.evaluation.results_index import (
    IndexedRunRecord,
    classify_artifact_governance,
    compute_validity_gate_passed,
    index_result_runs,
    summarize_directory_governance,
)
import scripts.list_paper_eligible_runs as list_runs_script


def test_classify_artifact_governance_marks_fake_runs_internal_only() -> None:
    decision = classify_artifact_governance(
        experiment_name="stockpyl_serial_live_llm_eval",
        benchmark_source="stockpyl_serial",
        provider="fake_llm_client",
        validity_gate_passed=True,
    )

    assert decision.artifact_use_class is ArtifactUseClass.INTERNAL_ONLY
    assert "fake_llm_internal_only" in decision.eligibility_notes
    assert "provisional_stockpyl_rollout" in decision.eligibility_notes


def test_classify_artifact_governance_keeps_live_runs_internal_while_benchmark_is_provisional() -> None:
    decision = classify_artifact_governance(
        experiment_name="stockpyl_serial_live_llm_eval",
        benchmark_source="stockpyl_serial",
        provider="openai",
        validity_gate_passed=True,
    )

    assert decision.validity_gate_passed is True
    assert decision.artifact_use_class is ArtifactUseClass.INTERNAL_ONLY
    assert decision.eligibility_notes == (
        "provisional_stockpyl_rollout",
        "provisional_operational_metrics",
    )


def test_classify_artifact_governance_marks_live_runs_paper_candidate_once_gates_pass() -> None:
    decision = classify_artifact_governance(
        experiment_name="stockpyl_serial_live_llm_eval",
        benchmark_source="stockpyl_serial",
        provider="openai",
        validity_gate_passed=True,
        rollout_fidelity_gate_passed=True,
        operational_metrics_gate_passed=True,
    )

    assert decision.validity_gate_passed is True
    assert decision.artifact_use_class is ArtifactUseClass.PAPER_CANDIDATE
    assert decision.eligibility_notes == ()


def test_classify_artifact_governance_marks_semi_synthetic_external_evidence_internal_only() -> None:
    decision = classify_artifact_governance(
        experiment_name="stockpyl_serial_external_evidence",
        benchmark_source="stockpyl_serial",
        provider="openai",
        validity_gate_passed=True,
        rollout_fidelity_gate_passed=True,
        operational_metrics_gate_passed=True,
        semi_synthetic_external_evidence=True,
    )

    assert decision.artifact_use_class is ArtifactUseClass.INTERNAL_ONLY
    assert decision.eligibility_notes == ("semi_synthetic_external_evidence_branch",)


def test_summarize_directory_governance_collapses_mode_decisions() -> None:
    live_decision = classify_artifact_governance(
        experiment_name="stockpyl_serial_live_llm_eval",
        benchmark_source="stockpyl_serial",
        provider="openai",
        validity_gate_passed=True,
    )
    fake_decision = classify_artifact_governance(
        experiment_name="stockpyl_serial_live_llm_eval",
        benchmark_source="stockpyl_serial",
        provider="fake_llm_client",
        validity_gate_passed=True,
    )

    directory_decision = summarize_directory_governance((live_decision, fake_decision))

    assert directory_decision.artifact_use_class is ArtifactUseClass.INTERNAL_ONLY
    assert "mixed_mode_directory_contains_internal_only_results" in (
        directory_decision.eligibility_notes
    )


def test_compute_validity_gate_passed_requires_clean_validity_counts() -> None:
    assert compute_validity_gate_passed(
        optimizer_order_boundary_preserved=True,
        invalid_output_count=0,
        fallback_count=0,
    )
    assert not compute_validity_gate_passed(
        optimizer_order_boundary_preserved=True,
        invalid_output_count=1,
        fallback_count=0,
    )


def test_index_result_runs_reads_saved_mode_level_governance() -> None:
    scratch_dir = Path(".tmp_results_index_tests") / uuid4().hex
    run_dir = scratch_dir / "results" / "demo" / "demo_run"
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            run_dir / "run_manifest.json",
            {
                "run_group_id": "demo_run",
                "experiment_id": "demo_experiment",
                "benchmark_id": "serial_3_echelon",
                "benchmark_source": "stockpyl_serial",
            },
        )
        write_json(
            run_dir / "experiment_metadata.json",
            {
                "mode": "all",
                "provider": "mixed",
                "model_name": "mixed",
                "validation_lane": "stockpyl_internal",
            },
        )
        write_json(
            run_dir / "aggregate_summary.json",
            {
                "mode_summaries": [
                    {
                        "mode": "deterministic_orchestrator",
                        "artifact_use_class": "internal_only",
                        "validity_gate_passed": True,
                        "eligibility_notes": [
                            "provisional_stockpyl_rollout",
                            "provisional_operational_metrics",
                        ],
                        "performance_summary": {
                            "average_total_cost": 320.0,
                            "average_fill_rate": 1.0,
                        },
                    }
                ]
            },
        )

        records = index_result_runs(scratch_dir / "results")

        assert len(records) == 1
        assert records[0].mode == "deterministic_orchestrator"
        assert records[0].artifact_use_class is ArtifactUseClass.INTERNAL_ONLY
        assert records[0].validity_gate_passed is True
        assert records[0].average_total_cost == 320.0
        assert records[0].validation_lane == "stockpyl_internal"
    finally:
        if scratch_dir.exists():
            shutil.rmtree(scratch_dir, ignore_errors=True)


def test_list_paper_eligible_runs_filters_indexed_output(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        list_runs_script,
        "index_result_runs",
        lambda _root: (
            IndexedRunRecord(
                results_dir="results/demo/demo_run",
                run_group_id="demo_run",
                experiment_id="demo_experiment",
                benchmark_id="serial_3_echelon",
                benchmark_source="stockpyl_serial",
                validation_lane="stockpyl_internal",
                mode="llm_orchestrator",
                provider="fake_llm_client",
                model_name="gpt-4o-mini",
                external_evidence_source=None,
                artifact_use_class=ArtifactUseClass.INTERNAL_ONLY,
                validity_gate_passed=True,
                eligibility_notes=("fake_llm_internal_only",),
                average_total_cost=320.0,
                average_fill_rate=1.0,
            ),
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "list_paper_eligible_runs.py",
            "--artifact-use-class",
            "internal_only",
        ],
    )

    list_runs_script.main()
    output = capsys.readouterr().out

    assert "demo_run" in output
    assert "internal_only" in output
