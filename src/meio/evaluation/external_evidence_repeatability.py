"""Repeatability analysis for repeated live external-evidence branch runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

from meio.evaluation.external_evidence_analysis import (
    DEFAULT_EXTERNAL_EVIDENCE_RESULTS_ROOT,
    ExternalEvidenceBatchAnalysis,
    ExternalEvidenceModeSummary,
    ExternalEvidenceScheduleComparison,
    analyze_external_evidence_batch,
)

INTERNAL_ONLY_MODE = "llm_orchestrator_internal_only"
EXTERNAL_EVIDENCE_MODE = "llm_orchestrator_with_external_evidence"
_NEUTRAL_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class NumericSpreadSummary:
    """Compact numeric spread summary over repeated runs."""

    mean_value: float | None
    minimum_value: float | None
    maximum_value: float | None
    standard_deviation: float | None


@dataclass(frozen=True, slots=True)
class ExternalEvidenceRepeatabilityRunSummary:
    """Per-run comparison of the internal-only and external-evidence live branches."""

    run_dir: str
    artifact_use_class: str | None
    internal_average_total_cost: float | None
    external_average_total_cost: float | None
    external_minus_internal_average_total_cost: float | None
    internal_regime_prediction_accuracy: float | None
    external_regime_prediction_accuracy: float | None
    internal_fallback_count: int
    external_fallback_count: int
    internal_invalid_output_count: int
    external_invalid_output_count: int
    evidence_fusion_cap_count: int
    early_evidence_confirmation_gate_count: int


@dataclass(frozen=True, slots=True)
class ExternalEvidenceRepeatabilityModeSummary:
    """Repeated-run summary for one live LLM branch."""

    mode: str
    run_count: int
    average_total_cost: NumericSpreadSummary
    regime_prediction_accuracy: NumericSpreadSummary
    total_fallback_count: int
    total_invalid_output_count: int
    total_evidence_fusion_cap_count: int = 0
    total_early_evidence_confirmation_gate_count: int = 0


@dataclass(frozen=True, slots=True)
class ExternalEvidenceRepeatabilityScheduleSummary:
    """Repeated-run schedule-level delta summary for external vs internal."""

    schedule_name: str
    run_count: int
    external_minus_internal_cost: NumericSpreadSummary
    better_count: int
    worse_count: int
    neutral_count: int
    stability_label: str


@dataclass(frozen=True, slots=True)
class ExternalEvidenceRepeatabilityAnalysis:
    """Top-level repeated-run summary for live external-evidence comparisons."""

    run_dirs: tuple[str, ...]
    run_count: int
    artifact_use_classes: tuple[str, ...]
    internal_mode_summary: ExternalEvidenceRepeatabilityModeSummary | None
    external_mode_summary: ExternalEvidenceRepeatabilityModeSummary | None
    run_summaries: tuple[ExternalEvidenceRepeatabilityRunSummary, ...]
    schedule_summaries: tuple[ExternalEvidenceRepeatabilityScheduleSummary, ...]
    external_branch_label: str
    early_evidence_gate_ever_fired: bool
    shift_recovery_stability_label: str | None
    double_shift_with_gap_stability_label: str | None
    recovery_then_relapse_stability_label: str | None


def list_external_evidence_run_dirs(
    *,
    results_root: str | Path = DEFAULT_EXTERNAL_EVIDENCE_RESULTS_ROOT,
    latest_n: int | None = None,
) -> tuple[Path, ...]:
    """List saved external-evidence run directories ordered by modification time."""

    root = Path(results_root)
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist.")
    candidates = tuple(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "aggregate_summary.json").exists()
    )
    if not candidates:
        raise FileNotFoundError(
            f"No saved external-evidence run directories found under {root}."
        )
    ordered = tuple(sorted(candidates, key=lambda path: path.stat().st_mtime))
    if latest_n is None:
        return ordered
    if latest_n <= 0:
        raise ValueError("latest_n must be positive when provided.")
    return ordered[-latest_n:]


def analyze_external_evidence_repeatability(
    run_dirs: tuple[str | Path, ...] | None = None,
    *,
    results_root: str | Path = DEFAULT_EXTERNAL_EVIDENCE_RESULTS_ROOT,
    latest_n: int | None = None,
) -> ExternalEvidenceRepeatabilityAnalysis:
    """Summarize repeated live external-evidence runs."""

    resolved_run_dirs = _resolve_run_dirs(
        run_dirs=run_dirs,
        results_root=results_root,
        latest_n=latest_n,
    )
    analyses = tuple(analyze_external_evidence_batch(run_dir) for run_dir in resolved_run_dirs)
    run_summaries = tuple(_build_run_summary(analysis) for analysis in analyses)
    internal_mode_summary = _build_mode_repeatability_summary(
        analyses,
        INTERNAL_ONLY_MODE,
    )
    external_mode_summary = _build_mode_repeatability_summary(
        analyses,
        EXTERNAL_EVIDENCE_MODE,
    )
    schedule_summaries = _build_schedule_summaries(analyses)
    overall_deltas = tuple(
        summary.external_minus_internal_average_total_cost
        for summary in run_summaries
        if summary.external_minus_internal_average_total_cost is not None
    )
    return ExternalEvidenceRepeatabilityAnalysis(
        run_dirs=tuple(str(path) for path in resolved_run_dirs),
        run_count=len(analyses),
        artifact_use_classes=tuple(
            sorted(
                {
                    analysis.artifact_use_class
                    for analysis in analyses
                    if analysis.artifact_use_class is not None
                }
            )
        ),
        internal_mode_summary=internal_mode_summary,
        external_mode_summary=external_mode_summary,
        run_summaries=run_summaries,
        schedule_summaries=schedule_summaries,
        external_branch_label=_stability_label(overall_deltas),
        early_evidence_gate_ever_fired=any(
            summary.early_evidence_confirmation_gate_count > 0
            for summary in run_summaries
        ),
        shift_recovery_stability_label=_schedule_label(
            schedule_summaries,
            "shift_recovery",
        ),
        double_shift_with_gap_stability_label=_schedule_label(
            schedule_summaries,
            "double_shift_with_gap",
        ),
        recovery_then_relapse_stability_label=_schedule_label(
            schedule_summaries,
            "recovery_then_relapse",
        ),
    )


def _resolve_run_dirs(
    *,
    run_dirs: tuple[str | Path, ...] | None,
    results_root: str | Path,
    latest_n: int | None,
) -> tuple[Path, ...]:
    if run_dirs:
        return tuple(Path(run_dir) for run_dir in run_dirs)
    return list_external_evidence_run_dirs(results_root=results_root, latest_n=latest_n)


def _build_run_summary(
    analysis: ExternalEvidenceBatchAnalysis,
) -> ExternalEvidenceRepeatabilityRunSummary:
    internal = _mode_summary(analysis, INTERNAL_ONLY_MODE)
    external = _mode_summary(analysis, EXTERNAL_EVIDENCE_MODE)
    external_metrics = external.external_evidence_summary if external is not None else None
    return ExternalEvidenceRepeatabilityRunSummary(
        run_dir=analysis.run_dir,
        artifact_use_class=analysis.artifact_use_class,
        internal_average_total_cost=(
            internal.average_total_cost if internal is not None else None
        ),
        external_average_total_cost=(
            external.average_total_cost if external is not None else None
        ),
        external_minus_internal_average_total_cost=_difference(
            external.average_total_cost if external is not None else None,
            internal.average_total_cost if internal is not None else None,
        ),
        internal_regime_prediction_accuracy=(
            internal.regime_prediction_accuracy if internal is not None else None
        ),
        external_regime_prediction_accuracy=(
            external.regime_prediction_accuracy if external is not None else None
        ),
        internal_fallback_count=internal.fallback_count if internal is not None else 0,
        external_fallback_count=external.fallback_count if external is not None else 0,
        internal_invalid_output_count=(
            internal.invalid_output_count if internal is not None else 0
        ),
        external_invalid_output_count=(
            external.invalid_output_count if external is not None else 0
        ),
        evidence_fusion_cap_count=(
            external_metrics.evidence_fusion_cap_count
            if external_metrics is not None
            else 0
        ),
        early_evidence_confirmation_gate_count=(
            external_metrics.early_evidence_confirmation_gate_count
            if external_metrics is not None
            else 0
        ),
    )


def _build_mode_repeatability_summary(
    analyses: tuple[ExternalEvidenceBatchAnalysis, ...],
    mode_name: str,
) -> ExternalEvidenceRepeatabilityModeSummary | None:
    mode_summaries = tuple(
        summary
        for analysis in analyses
        for summary in analysis.mode_summaries
        if summary.mode == mode_name
    )
    if not mode_summaries:
        return None
    external_metric_summaries = tuple(
        summary.external_evidence_summary
        for summary in mode_summaries
        if summary.external_evidence_summary is not None
    )
    return ExternalEvidenceRepeatabilityModeSummary(
        mode=mode_name,
        run_count=len(mode_summaries),
        average_total_cost=_numeric_spread(
            tuple(
                summary.average_total_cost
                for summary in mode_summaries
                if summary.average_total_cost is not None
            )
        ),
        regime_prediction_accuracy=_numeric_spread(
            tuple(
                summary.regime_prediction_accuracy
                for summary in mode_summaries
                if summary.regime_prediction_accuracy is not None
            )
        ),
        total_fallback_count=sum(summary.fallback_count for summary in mode_summaries),
        total_invalid_output_count=sum(
            summary.invalid_output_count for summary in mode_summaries
        ),
        total_evidence_fusion_cap_count=sum(
            item.evidence_fusion_cap_count for item in external_metric_summaries
        ),
        total_early_evidence_confirmation_gate_count=sum(
            item.early_evidence_confirmation_gate_count
            for item in external_metric_summaries
        ),
    )


def _build_schedule_summaries(
    analyses: tuple[ExternalEvidenceBatchAnalysis, ...],
) -> tuple[ExternalEvidenceRepeatabilityScheduleSummary, ...]:
    schedule_names = tuple(
        sorted(
            {
                comparison.schedule_name
                for analysis in analyses
                for comparison in analysis.schedule_comparisons
            }
        )
    )
    summaries: list[ExternalEvidenceRepeatabilityScheduleSummary] = []
    for schedule_name in schedule_names:
        deltas = tuple(
            comparison.llm_external_minus_internal_cost
            for analysis in analyses
            for comparison in analysis.schedule_comparisons
            if comparison.schedule_name == schedule_name
            and comparison.llm_external_minus_internal_cost is not None
        )
        if not deltas:
            continue
        summaries.append(
            ExternalEvidenceRepeatabilityScheduleSummary(
                schedule_name=schedule_name,
                run_count=len(deltas),
                external_minus_internal_cost=_numeric_spread(deltas),
                better_count=sum(
                    1 for value in deltas if value < -_NEUTRAL_TOLERANCE
                ),
                worse_count=sum(
                    1 for value in deltas if value > _NEUTRAL_TOLERANCE
                ),
                neutral_count=sum(
                    1 for value in deltas if abs(value) <= _NEUTRAL_TOLERANCE
                ),
                stability_label=_stability_label(deltas),
            )
        )
    return tuple(summaries)


def _mode_summary(
    analysis: ExternalEvidenceBatchAnalysis,
    mode_name: str,
) -> ExternalEvidenceModeSummary | None:
    for summary in analysis.mode_summaries:
        if summary.mode == mode_name:
            return summary
    return None


def _numeric_spread(values: tuple[float, ...]) -> NumericSpreadSummary:
    if not values:
        return NumericSpreadSummary(None, None, None, None)
    return NumericSpreadSummary(
        mean_value=float(mean(values)),
        minimum_value=float(min(values)),
        maximum_value=float(max(values)),
        standard_deviation=0.0 if len(values) == 1 else float(pstdev(values)),
    )


def _difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _stability_label(values: tuple[float, ...]) -> str:
    if not values:
        return "unknown"
    if all(abs(value) <= _NEUTRAL_TOLERANCE for value in values):
        return "stably_neutral"
    if all(value <= _NEUTRAL_TOLERANCE for value in values) and any(
        value < -_NEUTRAL_TOLERANCE for value in values
    ):
        return "stably_better"
    if all(value >= -_NEUTRAL_TOLERANCE for value in values) and any(
        value > _NEUTRAL_TOLERANCE for value in values
    ):
        return "stably_worse"
    return "unstable"


def _schedule_label(
    schedule_summaries: tuple[ExternalEvidenceRepeatabilityScheduleSummary, ...],
    schedule_name: str,
) -> str | None:
    for summary in schedule_summaries:
        if summary.schedule_name == schedule_name:
            return summary.stability_label
    return None


__all__ = [
    "DEFAULT_EXTERNAL_EVIDENCE_RESULTS_ROOT",
    "EXTERNAL_EVIDENCE_MODE",
    "INTERNAL_ONLY_MODE",
    "ExternalEvidenceRepeatabilityAnalysis",
    "ExternalEvidenceRepeatabilityModeSummary",
    "ExternalEvidenceRepeatabilityRunSummary",
    "ExternalEvidenceRepeatabilityScheduleSummary",
    "NumericSpreadSummary",
    "analyze_external_evidence_repeatability",
    "list_external_evidence_run_dirs",
]
