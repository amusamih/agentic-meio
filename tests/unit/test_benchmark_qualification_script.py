from __future__ import annotations

import meio.data.benchmark_adapters as benchmark_adapters
from meio.evaluation.benchmark_selection import (
    BenchmarkCandidate,
    QualificationDecision,
    QualificationReadiness,
)
from scripts.run_benchmark_qualification import run_benchmark_qualification


def test_benchmark_qualification_script_returns_expected_shortlist(monkeypatch) -> None:
    availability = {
        "stockpyl": True,
        "or_gym": False,
        "or_gym_inventory": True,
        "mabim": False,
    }
    monkeypatch.setattr(
        benchmark_adapters,
        "_module_available",
        lambda module_name: availability.get(module_name, False),
    )

    summaries = run_benchmark_qualification()
    summary_by_candidate = {summary.candidate: summary for summary in summaries}

    stockpyl = summary_by_candidate[BenchmarkCandidate.STOCKPYL_SERIAL]
    assert stockpyl.readiness is QualificationReadiness.IMMEDIATELY_SMOKE_TESTABLE
    assert stockpyl.decision is QualificationDecision.KEEP

    orgym = summary_by_candidate[BenchmarkCandidate.OR_GYM_INVENTORY]
    assert orgym.readiness is QualificationReadiness.PARTIALLY_INTEGRABLE
    assert orgym.decision is QualificationDecision.DEFER

    mabim = summary_by_candidate[BenchmarkCandidate.MABIM]
    assert mabim.readiness is QualificationReadiness.DEFERRED_MISSING_EXTERNAL_INTEGRATION
    assert mabim.decision is QualificationDecision.DROP
