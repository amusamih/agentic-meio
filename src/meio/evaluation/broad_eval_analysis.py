"""Frozen broad-evaluation analysis helpers for Stockpyl batch results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


DEFAULT_FROZEN_BROAD_RESULTS_ROOT = Path("results/stockpyl_serial_frozen_broad_eval")
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


def _load_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    if not path.exists():
        return ()
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} must contain JSON object rows.")
        rows.append(payload)
    return tuple(rows)


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


def _payload_int(payload: dict[str, object], key: str) -> int:
    value = payload.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer when present.")
    return value


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / float(len(values))


def _difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def latest_frozen_broad_batch_dir(
    results_root: str | Path = DEFAULT_FROZEN_BROAD_RESULTS_ROOT,
) -> Path:
    """Return the latest saved frozen broad-eval batch directory."""

    root = Path(results_root)
    candidates = tuple(path for path in root.iterdir() if path.is_dir()) if root.exists() else ()
    if not candidates:
        raise FileNotFoundError(
            f"No saved frozen broad-eval batch directories found under {root}."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


@dataclass(frozen=True, slots=True)
class BroadEvalModeSummary:
    """Compact mode-level summary from one frozen broad-eval batch."""

    mode: str
    artifact_use_class: str | None
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
    relapse_moderation_count: int
    unresolved_stress_moderation_count: int
    moderated_update_count: int


@dataclass(frozen=True, slots=True)
class BroadEvalModeScheduleSummary:
    """Compact per-mode summary for one broad-eval schedule."""

    schedule_name: str
    schedule_family: str
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
    relapse_moderation_count: int
    unresolved_stress_moderation_count: int
    moderated_update_count: int


@dataclass(frozen=True, slots=True)
class BroadEvalScheduleComparison:
    """Schedule-level comparison for the three frozen modes."""

    schedule_name: str
    schedule_family: str
    mode_summaries: tuple[BroadEvalModeScheduleSummary, ...]
    llm_cost_rank: int | None
    llm_vs_baseline_total_cost_delta: float | None = None
    llm_vs_deterministic_orchestrator_total_cost_delta: float | None = None
    llm_vs_baseline_fill_rate_delta: float | None = None
    llm_vs_deterministic_orchestrator_fill_rate_delta: float | None = None
    llm_outcome_label: str = "neutral"


@dataclass(frozen=True, slots=True)
class BroadEvalFamilySummary:
    """Grouped summary for one schedule family."""

    schedule_family: str
    schedule_names: tuple[str, ...]
    llm_help_count: int
    llm_hurt_count: int
    llm_neutral_count: int
    llm_average_cost_delta_vs_baseline: float | None
    llm_average_cost_delta_vs_deterministic_orchestrator: float | None


@dataclass(frozen=True, slots=True)
class FrozenBroadEvalAnalysis:
    """Top-level frozen broad-eval comparison summary."""

    run_dir: str
    benchmark_id: str | None
    artifact_use_class: str | None
    mode_artifact_use_classes: tuple[tuple[str, str], ...]
    mode_names: tuple[str, ...]
    schedule_names: tuple[str, ...]
    seed_values: tuple[int, ...]
    mode_summaries: tuple[BroadEvalModeSummary, ...]
    schedule_comparisons: tuple[BroadEvalScheduleComparison, ...]
    family_summaries: tuple[BroadEvalFamilySummary, ...]
    llm_help_schedules: tuple[str, ...] = ()
    llm_hurt_schedules: tuple[str, ...] = ()
    llm_neutral_schedules: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _BroadEvalArtifacts:
    run_dir: Path
    aggregate_summary: dict[str, object]
    run_manifest: dict[str, object]
    episode_rows: tuple[dict[str, object], ...]


def load_frozen_broad_batch(
    run_dir: str | Path | None = None,
    *,
    results_root: str | Path = DEFAULT_FROZEN_BROAD_RESULTS_ROOT,
) -> _BroadEvalArtifacts:
    """Load saved artifacts for one frozen broad-eval batch directory."""

    resolved_dir = (
        Path(run_dir) if run_dir is not None else latest_frozen_broad_batch_dir(results_root)
    )
    return _BroadEvalArtifacts(
        run_dir=resolved_dir,
        aggregate_summary=_load_json(resolved_dir / "aggregate_summary.json"),
        run_manifest=_load_json(resolved_dir / "run_manifest.json"),
        episode_rows=_load_jsonl(resolved_dir / "episode_summaries.jsonl"),
    )


def analyze_frozen_broad_batch(
    run_dir: str | Path | None = None,
    *,
    results_root: str | Path = DEFAULT_FROZEN_BROAD_RESULTS_ROOT,
) -> FrozenBroadEvalAnalysis:
    """Summarize a frozen broad-eval batch by mode, schedule, and family."""

    artifacts = load_frozen_broad_batch(run_dir, results_root=results_root)
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
    schedule_labels = _schedule_labels_by_name(artifacts.episode_rows)
    mode_summaries = tuple(
        _build_mode_summary(artifacts.aggregate_summary, mode_name)
        for mode_name in mode_names
    )
    schedule_comparisons = tuple(
        _build_schedule_comparison(
            artifacts.aggregate_summary,
            schedule_name,
            mode_names,
            schedule_labels.get(schedule_name, ()),
        )
        for schedule_name in schedule_names
    )
    family_summaries = _build_family_summaries(schedule_comparisons)
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
    artifact_use_class = artifacts.run_manifest.get("artifact_use_class")
    return FrozenBroadEvalAnalysis(
        run_dir=str(artifacts.run_dir),
        benchmark_id=benchmark_id if isinstance(benchmark_id, str) else None,
        artifact_use_class=artifact_use_class if isinstance(artifact_use_class, str) else None,
        mode_artifact_use_classes=mode_artifact_use_classes,
        mode_names=mode_names,
        schedule_names=schedule_names,
        seed_values=seed_values,
        mode_summaries=mode_summaries,
        schedule_comparisons=schedule_comparisons,
        family_summaries=family_summaries,
        llm_help_schedules=llm_help_schedules,
        llm_hurt_schedules=llm_hurt_schedules,
        llm_neutral_schedules=llm_neutral_schedules,
    )


def _schedule_labels_by_name(
    episode_rows: tuple[dict[str, object], ...],
) -> dict[str, tuple[str, ...]]:
    labels_by_name: dict[str, tuple[str, ...]] = {}
    for row in episode_rows:
        schedule_name = row.get("schedule_name")
        regime_schedule = row.get("regime_schedule")
        if not isinstance(schedule_name, str) or not isinstance(regime_schedule, list):
            continue
        if schedule_name in labels_by_name:
            continue
        labels = tuple(label for label in regime_schedule if isinstance(label, str))
        if labels:
            labels_by_name[schedule_name] = labels
    return labels_by_name


def _build_mode_summary(
    aggregate_summary: dict[str, object],
    mode_name: str,
) -> BroadEvalModeSummary:
    mode_summary = _find_mode_summary(aggregate_summary, mode_name)
    performance = mode_summary.get("performance_summary", {})
    decision_quality = mode_summary.get("decision_quality", {})
    telemetry = mode_summary.get("telemetry_metrics", {})
    tool_use = mode_summary.get("tool_use_summary", {})
    artifact_use_class = mode_summary.get("artifact_use_class")
    return BroadEvalModeSummary(
        mode=mode_name,
        artifact_use_class=artifact_use_class if isinstance(artifact_use_class, str) else None,
        average_total_cost=_nested_float(performance, "average_total_cost"),
        average_fill_rate=_nested_float(performance, "average_fill_rate"),
        average_inventory=_nested_float(performance, "average_inventory"),
        average_backorder_level=_nested_float(performance, "average_backorder_level"),
        regime_prediction_accuracy=_nested_float(decision_quality, "regime_prediction_accuracy"),
        no_action_rate=_nested_float(decision_quality, "no_action_rate"),
        replan_rate=_nested_float(decision_quality, "replan_rate"),
        intervention_rate=_nested_float(decision_quality, "intervention_rate"),
        average_tool_call_count=_nested_float(tool_use, "average_tool_call_count"),
        fallback_count=_payload_int(telemetry if isinstance(telemetry, dict) else {}, "fallback_count"),
        invalid_output_count=_payload_int(
            telemetry if isinstance(telemetry, dict) else {},
            "invalid_output_count",
        ),
        repeated_stress_moderation_count=_payload_int(
            tool_use if isinstance(tool_use, dict) else {},
            "repeated_stress_moderation_count",
        ),
        relapse_moderation_count=_payload_int(
            tool_use if isinstance(tool_use, dict) else {},
            "relapse_moderation_count",
        ),
        unresolved_stress_moderation_count=_payload_int(
            tool_use if isinstance(tool_use, dict) else {},
            "unresolved_stress_moderation_count",
        ),
        moderated_update_count=_payload_int(
            tool_use if isinstance(tool_use, dict) else {},
            "moderated_update_count",
        ),
    )


def _build_schedule_comparison(
    aggregate_summary: dict[str, object],
    schedule_name: str,
    mode_names: tuple[str, ...],
    schedule_labels: tuple[str, ...],
) -> BroadEvalScheduleComparison:
    schedule_family = _schedule_family(schedule_labels)
    mode_summaries = tuple(
        _build_mode_schedule_summary(
            aggregate_summary,
            schedule_name,
            mode_name,
            schedule_family,
        )
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
    return BroadEvalScheduleComparison(
        schedule_name=schedule_name,
        schedule_family=schedule_family,
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
        llm_outcome_label=_classify_llm_outcome(llm, baseline, deterministic),
    )


def _build_mode_schedule_summary(
    aggregate_summary: dict[str, object],
    schedule_name: str,
    mode_name: str,
    schedule_family: str,
) -> BroadEvalModeScheduleSummary:
    mode_summary = _find_mode_summary(aggregate_summary, mode_name)
    schedule_breakdown = _find_schedule_breakdown(mode_summary, schedule_name)
    performance = schedule_breakdown.get("performance_summary", {})
    decision_quality = schedule_breakdown.get("decision_quality", {})
    telemetry = schedule_breakdown.get("telemetry_metrics", {})
    tool_use = schedule_breakdown.get("tool_use_summary", {})
    return BroadEvalModeScheduleSummary(
        schedule_name=schedule_name,
        schedule_family=schedule_family,
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
        fallback_count=_payload_int(telemetry if isinstance(telemetry, dict) else {}, "fallback_count"),
        invalid_output_count=_payload_int(
            telemetry if isinstance(telemetry, dict) else {},
            "invalid_output_count",
        ),
        repeated_stress_moderation_count=_payload_int(
            tool_use if isinstance(tool_use, dict) else {},
            "repeated_stress_moderation_count",
        ),
        relapse_moderation_count=_payload_int(
            tool_use if isinstance(tool_use, dict) else {},
            "relapse_moderation_count",
        ),
        unresolved_stress_moderation_count=_payload_int(
            tool_use if isinstance(tool_use, dict) else {},
            "unresolved_stress_moderation_count",
        ),
        moderated_update_count=_payload_int(
            tool_use if isinstance(tool_use, dict) else {},
            "moderated_update_count",
        ),
    )


def _build_family_summaries(
    schedule_comparisons: tuple[BroadEvalScheduleComparison, ...],
) -> tuple[BroadEvalFamilySummary, ...]:
    family_names = tuple(sorted({comparison.schedule_family for comparison in schedule_comparisons}))
    summaries: list[BroadEvalFamilySummary] = []
    for family_name in family_names:
        family_comparisons = tuple(
            comparison
            for comparison in schedule_comparisons
            if comparison.schedule_family == family_name
        )
        summaries.append(
            BroadEvalFamilySummary(
                schedule_family=family_name,
                schedule_names=tuple(
                    comparison.schedule_name for comparison in family_comparisons
                ),
                llm_help_count=sum(
                    1 for comparison in family_comparisons if comparison.llm_outcome_label == "helps"
                ),
                llm_hurt_count=sum(
                    1 for comparison in family_comparisons if comparison.llm_outcome_label == "hurts"
                ),
                llm_neutral_count=sum(
                    1
                    for comparison in family_comparisons
                    if comparison.llm_outcome_label == "neutral"
                ),
                llm_average_cost_delta_vs_baseline=_mean(
                    [
                        delta
                        for delta in (
                            comparison.llm_vs_baseline_total_cost_delta
                            for comparison in family_comparisons
                        )
                        if delta is not None
                    ]
                ),
                llm_average_cost_delta_vs_deterministic_orchestrator=_mean(
                    [
                        delta
                        for delta in (
                            comparison.llm_vs_deterministic_orchestrator_total_cost_delta
                            for comparison in family_comparisons
                        )
                        if delta is not None
                    ]
                ),
            )
        )
    return tuple(summaries)


def _schedule_family(labels: tuple[str, ...]) -> str:
    if not labels:
        return "other"
    shift_indices = [index for index, label in enumerate(labels) if label == "demand_regime_shift"]
    recovery_indices = [index for index, label in enumerate(labels) if label == "recovery"]
    if not shift_indices:
        return "false_alarm_only" if recovery_indices else "other"
    first_shift = shift_indices[0]
    if any(index < first_shift for index in recovery_indices):
        return "false_alarm_mixed"
    if len(shift_indices) == 1:
        if any(index > first_shift for index in recovery_indices):
            return "single_shift_recovery"
        return "single_shift_only"
    if any(later == earlier + 1 for earlier, later in zip(shift_indices, shift_indices[1:])):
        return "adjacent_repeated_stress"
    return "relapse_or_gap"


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
    llm: BroadEvalModeScheduleSummary | None,
    baseline: BroadEvalModeScheduleSummary | None,
    deterministic: BroadEvalModeScheduleSummary | None,
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
        and llm.average_fill_rate < min(
            baseline.average_fill_rate,
            deterministic.average_fill_rate,
        )
    ):
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


__all__ = [
    "BroadEvalFamilySummary",
    "BroadEvalModeScheduleSummary",
    "BroadEvalModeSummary",
    "BroadEvalScheduleComparison",
    "DEFAULT_FROZEN_BROAD_RESULTS_ROOT",
    "FrozenBroadEvalAnalysis",
    "analyze_frozen_broad_batch",
    "latest_frozen_broad_batch_dir",
    "load_frozen_broad_batch",
]
