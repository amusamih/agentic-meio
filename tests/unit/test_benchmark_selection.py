from __future__ import annotations

from meio.evaluation.benchmark_selection import (
    BenchmarkCandidate,
    BenchmarkQualificationSpec,
    CriterionAssessment,
    QualificationCriterion,
    QualificationLevel,
    QualificationDecision,
    QualificationReadiness,
    build_qualification_summary,
    recommend_decision,
)


def build_assessments(
    topology: QualificationLevel,
    paper_value: QualificationLevel,
) -> tuple[CriterionAssessment, ...]:
    return (
        CriterionAssessment(QualificationCriterion.TOPOLOGY_FIT, topology),
        CriterionAssessment(QualificationCriterion.REGIME_SHIFT_FIT, QualificationLevel.MEDIUM),
        CriterionAssessment(QualificationCriterion.OPTIMIZER_BOUNDARY_FIT, QualificationLevel.HIGH),
        CriterionAssessment(
            QualificationCriterion.EVENT_TRIGGER_COMPATIBILITY,
            QualificationLevel.MEDIUM,
        ),
        CriterionAssessment(QualificationCriterion.IMPLEMENTATION_EFFORT, QualificationLevel.MEDIUM),
        CriterionAssessment(QualificationCriterion.EXPECTED_PAPER_VALUE, paper_value),
    )


def test_benchmark_qualification_summary_construction() -> None:
    spec = BenchmarkQualificationSpec(
        candidate=BenchmarkCandidate.STOCKPYL_SERIAL,
        assessments=build_assessments(
            topology=QualificationLevel.HIGH,
            paper_value=QualificationLevel.HIGH,
        ),
        rationale="Closest current fit to the canonical serial path.",
        notes=("living_judgment",),
    )

    summary = build_qualification_summary(
        spec=spec,
        topology_style="canonical_serial_multi_echelon",
        smoke_testable_now=True,
        available_modules=("stockpyl",),
        missing_modules=(),
        integration_work_remaining=("Wrap typed state objects.",),
        adapter_notes=("availability_checked",),
    )

    assert summary.readiness is QualificationReadiness.IMMEDIATELY_SMOKE_TESTABLE
    assert summary.decision is QualificationDecision.KEEP
    assert summary.level_for(QualificationCriterion.TOPOLOGY_FIT) is QualificationLevel.HIGH
    assert summary.notes == ("living_judgment", "availability_checked")


def test_recommend_decision_drops_low_paper_value_candidate() -> None:
    decision = recommend_decision(
        readiness=QualificationReadiness.DEFERRED_MISSING_EXTERNAL_INTEGRATION,
        assessments=build_assessments(
            topology=QualificationLevel.UNVERIFIED,
            paper_value=QualificationLevel.LOW,
        ),
    )

    assert decision is QualificationDecision.DROP
