"""Held-out schedule analysis helpers for frozen Stockpyl comparisons."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


DEFAULT_HELDOUT_RESULTS_ROOT = Path("results/stockpyl_serial_heldout_eval")
DEFAULT_MODE_NAMES = (
    "deterministic_baseline",
    "deterministic_orchestrator",
    "llm_orchestrator",
)


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _payload_float(payload: dict[str, object], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric when present.")
    return float(value)


def _nested_float(payload: object, key: str) -> float | None:
    if not isinstance(payload, dict):
        return None
    return _payload_float(payload, key)


def latest_heldout_batch_dir(
    results_root: str | Path = DEFAULT_HELDOUT_RESULTS_ROOT,
) -> Path:
    """Return the latest saved held-out batch directory."""

    root = Path(results_root)
    candidates = tuple(path for path in root.iterdir() if path.is_dir()) if root.exists() else ()
    if not candidates:
        raise FileNotFoundError(f"No saved held-out batch directories found under {root}.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


@dataclass(frozen=True, slots=True)
class HeldoutModeScheduleSummary:
    """Compact per-mode summary for one held-out schedule."""

    schedule_name: str
    mode: str
    average_total_cost: float | None
    average_fill_rate: float | None
    average_inventory: float | None
    average_backorder_level: float | None
    regime_prediction_accuracy: float | None
    no_action_rate: float | None
    replan_rate: float | None
    intervention_rate: float | None
    average_tool_call_count: float | None
    fallback_count: int
    invalid_output_count: int
    repeated_stress_moderation_count: int
    moderated_update_count: int


@dataclass(frozen=True, slots=True)
class HeldoutScheduleComparison:
    """Schedule-level comparison for the three frozen modes."""

    schedule_name: str
    mode_summaries: tuple[HeldoutModeScheduleSummary, ...]
    llm_cost_rank: int | None
    llm_vs_baseline_total_cost_delta: float | None = None
    llm_vs_deterministic_orchestrator_total_cost_delta: float | None = None
    llm_vs_baseline_fill_rate_delta: float | None = None
    llm_vs_deterministic_orchestrator_fill_rate_delta: float | None = None
    llm_outcome_label: str = "neutral"


@dataclass(frozen=True, slots=True)
class HeldoutBatchAnalysis:
    """Top-level held-out comparison analysis for one saved batch."""

    run_dir: str
    benchmark_id: str | None
    artifact_use_class: str | None
    mode_artifact_use_classes: tuple[tuple[str, str], ...]
    mode_names: tuple[str, ...]
    schedule_names: tuple[str, ...]
    seed_values: tuple[int, ...]
    schedule_comparisons: tuple[HeldoutScheduleComparison, ...]
    llm_help_schedules: tuple[str, ...] = ()
    llm_hurt_schedules: tuple[str, ...] = ()
    llm_neutral_schedules: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _HeldoutArtifacts:
    run_dir: Path
    aggregate_summary: dict[str, object]
    run_manifest: dict[str, object]


def load_heldout_batch(
    run_dir: str | Path | None = None,
    *,
    results_root: str | Path = DEFAULT_HELDOUT_RESULTS_ROOT,
) -> _HeldoutArtifacts:
    """Load saved held-out aggregate and manifest artifacts."""

    resolved_dir = Path(run_dir) if run_dir is not None else latest_heldout_batch_dir(results_root)
    return _HeldoutArtifacts(
        run_dir=resolved_dir,
        aggregate_summary=_load_json(resolved_dir / "aggregate_summary.json"),
        run_manifest=_load_json(resolved_dir / "run_manifest.json"),
    )


def analyze_heldout_batch(
    run_dir: str | Path | None = None,
    *,
    results_root: str | Path = DEFAULT_HELDOUT_RESULTS_ROOT,
) -> HeldoutBatchAnalysis:
    """Summarize held-out frozen-system performance by schedule and mode."""

    artifacts = load_heldout_batch(run_dir, results_root=results_root)
    mode_names = tuple(
        str(item)
        for item in artifacts.aggregate_summary.get("mode_names", ())
        if isinstance(item, str)
    ) or DEFAULT_MODE_NAMES
    schedule_names = tuple(
        str(item)
        for item in artifacts.aggregate_summary.get("schedule_names", ())
        if isinstance(item, str)
    )
    seed_values = tuple(
        int(item)
        for item in artifacts.aggregate_summary.get("seed_values", ())
        if isinstance(item, int)
    )
    schedule_comparisons = tuple(
        _build_schedule_comparison(artifacts.aggregate_summary, schedule_name, mode_names)
        for schedule_name in schedule_names
    )
    llm_help_schedules = tuple(
        comparison.schedule_name
        for comparison in schedule_comparisons
        if comparison.llm_outcome_label == "helps"
    )
    llm_hurt_schedules = tuple(
        comparison.schedule_name
        for comparison in schedule_comparisons
        if comparison.llm_outcome_label == "hurts"
    )
    llm_neutral_schedules = tuple(
        comparison.schedule_name
        for comparison in schedule_comparisons
        if comparison.llm_outcome_label == "neutral"
    )
    mode_artifact_use_classes = tuple(
        (str(mode_name), str(use_class))
        for mode_name, use_class in artifacts.run_manifest.get("mode_artifact_use_classes", ())
        if isinstance(mode_name, str) and isinstance(use_class, str)
    )
    benchmark_id = artifacts.aggregate_summary.get("benchmark_id")
    return HeldoutBatchAnalysis(
        run_dir=str(artifacts.run_dir),
        benchmark_id=benchmark_id if isinstance(benchmark_id, str) else None,
        artifact_use_class=(
            artifacts.run_manifest.get("artifact_use_class")
            if isinstance(artifacts.run_manifest.get("artifact_use_class"), str)
            else None
        ),
        mode_artifact_use_classes=mode_artifact_use_classes,
        mode_names=mode_names,
        schedule_names=schedule_names,
        seed_values=seed_values,
        schedule_comparisons=schedule_comparisons,
        llm_help_schedules=llm_help_schedules,
        llm_hurt_schedules=llm_hurt_schedules,
        llm_neutral_schedules=llm_neutral_schedules,
    )


def _build_schedule_comparison(
    aggregate_summary: dict[str, object],
    schedule_name: str,
    mode_names: tuple[str, ...],
) -> HeldoutScheduleComparison:
    mode_summaries = tuple(
        _build_mode_schedule_summary(aggregate_summary, schedule_name, mode_name)
        for mode_name in mode_names
    )
    summary_by_mode = {summary.mode: summary for summary in mode_summaries}
    baseline = summary_by_mode.get("deterministic_baseline")
    deterministic = summary_by_mode.get("deterministic_orchestrator")
    llm = summary_by_mode.get("llm_orchestrator")
    comparable_costs = [
        (summary.mode, summary.average_total_cost)
        for summary in mode_summaries
        if summary.average_total_cost is not None
    ]
    llm_cost_rank = None
    if llm is not None and llm.average_total_cost is not None and comparable_costs:
        sorted_costs = sorted(comparable_costs, key=lambda item: float(item[1]))
        llm_cost_rank = next(
            index
            for index, (mode_name, _) in enumerate(sorted_costs, start=1)
            if mode_name == "llm_orchestrator"
        )
    return HeldoutScheduleComparison(
        schedule_name=schedule_name,
        mode_summaries=mode_summaries,
        llm_cost_rank=llm_cost_rank,
        llm_vs_baseline_total_cost_delta=_difference(
            llm.average_total_cost if llm is not None else None,
            baseline.average_total_cost if baseline is not None else None,
        ),
        llm_vs_deterministic_orchestrator_total_cost_delta=_difference(
            llm.average_total_cost if llm is not None else None,
            deterministic.average_total_cost if deterministic is not None else None,
        ),
        llm_vs_baseline_fill_rate_delta=_difference(
            llm.average_fill_rate if llm is not None else None,
            baseline.average_fill_rate if baseline is not None else None,
        ),
        llm_vs_deterministic_orchestrator_fill_rate_delta=_difference(
            llm.average_fill_rate if llm is not None else None,
            deterministic.average_fill_rate if deterministic is not None else None,
        ),
        llm_outcome_label=_classify_llm_outcome(
            llm,
            baseline,
            deterministic,
        ),
    )


def _build_mode_schedule_summary(
    aggregate_summary: dict[str, object],
    schedule_name: str,
    mode_name: str,
) -> HeldoutModeScheduleSummary:
    mode_summary = _find_mode_summary(aggregate_summary, mode_name)
    schedule_breakdown = _find_schedule_breakdown(mode_summary, schedule_name)
    performance = schedule_breakdown.get("performance_summary", {})
    decision_quality = schedule_breakdown.get("decision_quality", {})
    telemetry = schedule_breakdown.get("telemetry_metrics", {})
    tool_use = schedule_breakdown.get("tool_use_summary", {})
    return HeldoutModeScheduleSummary(
        schedule_name=schedule_name,
        mode=mode_name,
        average_total_cost=_nested_float(performance, "average_total_cost"),
        average_fill_rate=_nested_float(performance, "average_fill_rate"),
        average_inventory=_nested_float(performance, "average_inventory"),
        average_backorder_level=_nested_float(performance, "average_backorder_level"),
        regime_prediction_accuracy=_nested_float(decision_quality, "regime_prediction_accuracy"),
        no_action_rate=_nested_float(decision_quality, "no_action_rate"),
        replan_rate=_nested_float(decision_quality, "replan_rate"),
        intervention_rate=_nested_float(decision_quality, "intervention_rate"),
        average_tool_call_count=_nested_float(tool_use, "average_tool_call_count"),
        fallback_count=_int_from_payload(telemetry, "fallback_count"),
        invalid_output_count=_int_from_payload(telemetry, "invalid_output_count"),
        repeated_stress_moderation_count=_int_from_payload(
            tool_use,
            "repeated_stress_moderation_count",
        ),
        moderated_update_count=_int_from_payload(tool_use, "moderated_update_count"),
    )


def _find_mode_summary(
    aggregate_summary: dict[str, object],
    mode_name: str,
) -> dict[str, object]:
    for mode_summary in aggregate_summary.get("mode_summaries", ()):
        if isinstance(mode_summary, dict) and mode_summary.get("mode") == mode_name:
            return mode_summary
    raise ValueError(f"Mode summary not found for {mode_name!r}.")


def _find_schedule_breakdown(
    mode_summary: dict[str, object],
    schedule_name: str,
) -> dict[str, object]:
    for schedule_summary in mode_summary.get("schedule_breakdown", ()):
        if isinstance(schedule_summary, dict) and schedule_summary.get("schedule_name") == schedule_name:
            return schedule_summary
    raise ValueError(f"Schedule breakdown not found for {schedule_name!r}.")


def _classify_llm_outcome(
    llm: HeldoutModeScheduleSummary | None,
    baseline: HeldoutModeScheduleSummary | None,
    deterministic: HeldoutModeScheduleSummary | None,
) -> str:
    if llm is None or baseline is None or deterministic is None:
        return "neutral"
    if (
        llm.average_total_cost is None
        or baseline.average_total_cost is None
        or deterministic.average_total_cost is None
    ):
        return "neutral"
    if (
        llm.average_fill_rate is not None
        and baseline.average_fill_rate is not None
        and deterministic.average_fill_rate is not None
    ):
        if llm.average_fill_rate < min(baseline.average_fill_rate, deterministic.average_fill_rate):
            return "hurts"
    if llm.average_total_cost < min(
        baseline.average_total_cost,
        deterministic.average_total_cost,
    ):
        return "helps"
    if llm.average_total_cost > max(
        baseline.average_total_cost,
        deterministic.average_total_cost,
    ):
        return "hurts"
    return "neutral"


def _int_from_payload(payload: object, key: str) -> int:
    if not isinstance(payload, dict):
        return 0
    value = payload.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer when present.")
    return value


__all__ = [
    "DEFAULT_HELDOUT_RESULTS_ROOT",
    "DEFAULT_MODE_NAMES",
    "HeldoutBatchAnalysis",
    "HeldoutModeScheduleSummary",
    "HeldoutScheduleComparison",
    "analyze_heldout_batch",
    "latest_heldout_batch_dir",
    "load_heldout_batch",
]
