"""Schedule-level comparison helpers for paper-candidate failure analysis."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


DEFAULT_PAPER_CANDIDATE_RESULTS_ROOT = Path("results/stockpyl_serial_paper_candidate")
DEFAULT_MODE_NAMES = (
    "deterministic_baseline",
    "deterministic_orchestrator",
    "llm_orchestrator",
)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / float(len(values))


def _sorted_counts(counter: Counter[str]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(counter.items()))


def _average_sequence(values: list[tuple[float, ...]]) -> tuple[float, ...]:
    if not values:
        return ()
    max_length = max(len(value) for value in values)
    totals = [0.0] * max_length
    for value in values:
        for index in range(max_length):
            totals[index] += value[index] if index < len(value) else 0.0
    denominator = float(len(values))
    return tuple(total / denominator for total in totals)


def _sequence_sum(values: tuple[float, ...]) -> float:
    return sum(values)


def _counter_from_iterables(values: list[list[str] | tuple[str, ...]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for group in values:
        counter.update(group)
    return counter


def _dominant_label(counts: tuple[tuple[str, int], ...]) -> str | None:
    if not counts:
        return None
    return max(counts, key=lambda item: (item[1], item[0]))[0]


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


def latest_paper_candidate_batch_dir(
    results_root: str | Path = DEFAULT_PAPER_CANDIDATE_RESULTS_ROOT,
) -> Path:
    """Return the latest saved paper-candidate batch directory."""

    root = Path(results_root)
    candidates = tuple(path for path in root.iterdir() if path.is_dir()) if root.exists() else ()
    if not candidates:
        raise FileNotFoundError(f"No saved batch directories found under {root}.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


@dataclass(frozen=True, slots=True)
class PeriodBehaviorSummary:
    """Compact averaged view of one schedule-period behavior by mode."""

    period_index: int
    true_regime_label: str | None
    predicted_regime_counts: tuple[tuple[str, int], ...] = ()
    selected_subgoal_counts: tuple[tuple[str, int], ...] = ()
    selected_tool_counts: tuple[tuple[str, int], ...] = ()
    update_request_counts: tuple[tuple[str, int], ...] = ()
    average_replan_rate: float | None = None
    average_no_action_rate: float | None = None
    average_demand_outlook: float | None = None
    average_leadtime_outlook: float | None = None
    average_safety_buffer_scale: float | None = None
    average_orders_by_echelon: tuple[float, ...] = ()
    average_pipeline_by_echelon: tuple[float, ...] = ()
    average_inventory_by_echelon: tuple[float, ...] = ()
    average_backorders_by_echelon: tuple[float, ...] = ()
    average_order_sum: float | None = None
    average_per_period_cost: float | None = None
    average_per_period_fill_rate: float | None = None


@dataclass(frozen=True, slots=True)
class ModeScheduleSummary:
    """Schedule-level summary for one mode."""

    schedule_name: str
    mode: str
    run_count: int
    average_total_cost: float | None
    average_fill_rate: float | None
    average_inventory: float | None
    average_backorder_level: float | None
    average_holding_cost: float | None
    average_backlog_cost: float | None
    average_ordering_cost: float | None
    average_replan_count: float | None
    average_no_action_count: float | None
    average_intervention_count: float | None
    average_tool_call_count: float | None
    regime_prediction_accuracy: float | None
    tool_usage_counts: tuple[tuple[str, int], ...] = ()
    optimizer_input_change_count: int = 0
    decision_changed_optimizer_input_count: int = 0
    optimizer_output_changed_state_count: int = 0
    intervention_changed_outcome_count: int = 0
    period_summaries: tuple[PeriodBehaviorSummary, ...] = ()


@dataclass(frozen=True, slots=True)
class PeriodDifferenceHighlight:
    """Compact per-period difference view for the live LLM path."""

    period_index: int
    true_regime_label: str | None
    llm_predicted_regime_label: str | None
    llm_selected_tool_counts: tuple[tuple[str, int], ...] = ()
    llm_update_request_counts: tuple[tuple[str, int], ...] = ()
    llm_average_replan_rate: float | None = None
    llm_average_demand_outlook: float | None = None
    deterministic_baseline_average_demand_outlook: float | None = None
    deterministic_orchestrator_average_demand_outlook: float | None = None
    llm_average_safety_buffer_scale: float | None = None
    deterministic_baseline_average_safety_buffer_scale: float | None = None
    deterministic_orchestrator_average_safety_buffer_scale: float | None = None
    llm_average_order_sum: float | None = None
    deterministic_baseline_average_order_sum: float | None = None
    deterministic_orchestrator_average_order_sum: float | None = None
    llm_order_sum_delta_vs_baseline: float | None = None
    llm_order_sum_delta_vs_deterministic_orchestrator: float | None = None


@dataclass(frozen=True, slots=True)
class ScheduleComparisonSummary:
    """Comparison summary for one schedule across the three modes."""

    schedule_name: str
    mode_summaries: tuple[ModeScheduleSummary, ...]
    llm_cost_rank: int | None
    llm_vs_baseline_total_cost_delta: float | None = None
    llm_vs_deterministic_orchestrator_total_cost_delta: float | None = None
    llm_vs_baseline_fill_rate_delta: float | None = None
    llm_vs_deterministic_orchestrator_fill_rate_delta: float | None = None
    llm_vs_baseline_ordering_cost_delta: float | None = None
    llm_vs_deterministic_orchestrator_ordering_cost_delta: float | None = None
    period_highlights: tuple[PeriodDifferenceHighlight, ...] = ()


@dataclass(frozen=True, slots=True)
class PaperCandidateFailureAnalysis:
    """Top-level paper-candidate batch analysis from saved artifacts."""

    run_dir: str
    benchmark_id: str | None
    mode_names: tuple[str, ...]
    schedule_names: tuple[str, ...]
    schedule_comparisons: tuple[ScheduleComparisonSummary, ...]
    llm_best_schedule_by_cost: str | None = None
    llm_worst_schedule_by_cost: str | None = None


@dataclass(frozen=True, slots=True)
class _BatchArtifacts:
    run_dir: Path
    aggregate_summary: dict[str, object]
    episode_rows: tuple[dict[str, object], ...]
    step_rows: tuple[dict[str, object], ...]
    tool_rows: tuple[dict[str, object], ...]


def load_paper_candidate_batch(
    run_dir: str | Path | None = None,
    *,
    results_root: str | Path = DEFAULT_PAPER_CANDIDATE_RESULTS_ROOT,
) -> _BatchArtifacts:
    """Load the saved artifacts for one paper-candidate batch directory."""

    resolved_dir = Path(run_dir) if run_dir is not None else latest_paper_candidate_batch_dir(results_root)
    return _BatchArtifacts(
        run_dir=resolved_dir,
        aggregate_summary=_load_json(resolved_dir / "aggregate_summary.json"),
        episode_rows=_load_jsonl(resolved_dir / "episode_summaries.jsonl"),
        step_rows=_load_jsonl(resolved_dir / "step_traces.jsonl"),
        tool_rows=_load_jsonl(resolved_dir / "tool_call_traces.jsonl"),
    )


def analyze_paper_candidate_batch(
    run_dir: str | Path | None = None,
    *,
    results_root: str | Path = DEFAULT_PAPER_CANDIDATE_RESULTS_ROOT,
) -> PaperCandidateFailureAnalysis:
    """Build schedule-level comparison summaries from saved paper-candidate artifacts."""

    artifacts = load_paper_candidate_batch(run_dir, results_root=results_root)
    schedule_names = tuple(
        sorted(
            {
                str(row["schedule_name"])
                for row in artifacts.episode_rows
                if row.get("schedule_name") is not None
            }
        )
    )
    mode_names = tuple(
        mode_summary.get("mode", "")
        for mode_summary in artifacts.aggregate_summary.get("mode_summaries", [])
        if isinstance(mode_summary, dict) and isinstance(mode_summary.get("mode"), str)
    ) or DEFAULT_MODE_NAMES
    schedule_comparisons = tuple(
        _build_schedule_comparison(artifacts, schedule_name, mode_names)
        for schedule_name in schedule_names
    )
    llm_schedule_costs = {
        comparison.schedule_name: _mode_summary_by_name(comparison, "llm_orchestrator").average_total_cost
        for comparison in schedule_comparisons
    }
    comparable_costs = {
        name: value for name, value in llm_schedule_costs.items() if value is not None
    }
    llm_best_schedule = min(comparable_costs, key=comparable_costs.__getitem__) if comparable_costs else None
    llm_worst_schedule = max(comparable_costs, key=comparable_costs.__getitem__) if comparable_costs else None
    benchmark_id = artifacts.aggregate_summary.get("benchmark_id")
    return PaperCandidateFailureAnalysis(
        run_dir=str(artifacts.run_dir),
        benchmark_id=benchmark_id if isinstance(benchmark_id, str) else None,
        mode_names=mode_names,
        schedule_names=schedule_names,
        schedule_comparisons=schedule_comparisons,
        llm_best_schedule_by_cost=llm_best_schedule,
        llm_worst_schedule_by_cost=llm_worst_schedule,
    )


def _build_schedule_comparison(
    artifacts: _BatchArtifacts,
    schedule_name: str,
    mode_names: tuple[str, ...],
) -> ScheduleComparisonSummary:
    mode_summaries = tuple(
        _build_mode_schedule_summary(artifacts, schedule_name, mode_name)
        for mode_name in mode_names
    )
    baseline = next((summary for summary in mode_summaries if summary.mode == "deterministic_baseline"), None)
    deterministic = next(
        (summary for summary in mode_summaries if summary.mode == "deterministic_orchestrator"),
        None,
    )
    llm = next((summary for summary in mode_summaries if summary.mode == "llm_orchestrator"), None)
    comparable_costs = [
        (summary.mode, summary.average_total_cost)
        for summary in mode_summaries
        if summary.average_total_cost is not None
    ]
    llm_rank = None
    if llm is not None and llm.average_total_cost is not None and comparable_costs:
        sorted_costs = sorted(comparable_costs, key=lambda item: float(item[1]))
        llm_rank = next(
            index
            for index, (mode_name, _) in enumerate(sorted_costs, start=1)
            if mode_name == "llm_orchestrator"
        )
    return ScheduleComparisonSummary(
        schedule_name=schedule_name,
        mode_summaries=mode_summaries,
        llm_cost_rank=llm_rank,
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
        llm_vs_baseline_ordering_cost_delta=_difference(
            llm.average_ordering_cost if llm is not None else None,
            baseline.average_ordering_cost if baseline is not None else None,
        ),
        llm_vs_deterministic_orchestrator_ordering_cost_delta=_difference(
            llm.average_ordering_cost if llm is not None else None,
            deterministic.average_ordering_cost if deterministic is not None else None,
        ),
        period_highlights=_build_period_highlights(mode_summaries),
    )


def _build_mode_schedule_summary(
    artifacts: _BatchArtifacts,
    schedule_name: str,
    mode_name: str,
) -> ModeScheduleSummary:
    episode_rows = [
        row
        for row in artifacts.episode_rows
        if row.get("schedule_name") == schedule_name and row.get("mode") == mode_name
    ]
    step_rows = [
        row
        for row in artifacts.step_rows
        if row.get("schedule_name") == schedule_name and row.get("mode") == mode_name
    ]
    tool_rows = [
        row
        for row in artifacts.tool_rows
        if row.get("schedule_name") == schedule_name and row.get("mode") == mode_name
    ]
    period_summaries = tuple(
        _build_period_behavior_summary(period_index, period_rows)
        for period_index, period_rows in _group_step_rows_by_period(step_rows).items()
    )
    tool_usage_counts = Counter(str(row["tool_id"]) for row in tool_rows if row.get("tool_id"))
    optimizer_input_change_count = sum(
        1 for row in tool_rows if bool(row.get("optimizer_input_changed"))
    )
    decision_changed_optimizer_input_count = sum(
        1 for row in step_rows if bool(row.get("decision_changed_optimizer_input"))
    )
    optimizer_output_changed_state_count = sum(
        1 for row in step_rows if bool(row.get("optimizer_output_changed_state"))
    )
    intervention_changed_outcome_count = sum(
        1 for row in step_rows if bool(row.get("intervention_changed_outcome"))
    )
    return ModeScheduleSummary(
        schedule_name=schedule_name,
        mode=mode_name,
        run_count=len(episode_rows),
        average_total_cost=_mean([float(row["total_cost"]) for row in episode_rows if row.get("total_cost") is not None]),
        average_fill_rate=_mean([float(row["fill_rate"]) for row in episode_rows if row.get("fill_rate") is not None]),
        average_inventory=_mean(
            [float(row["average_inventory"]) for row in episode_rows if row.get("average_inventory") is not None]
        ),
        average_backorder_level=_mean(
            [
                float(row["average_backorder_level"])
                for row in episode_rows
                if row.get("average_backorder_level") is not None
            ]
        ),
        average_holding_cost=_mean(
            [
                float(row["cost_breakdown"]["holding_cost"])
                for row in episode_rows
                if isinstance(row.get("cost_breakdown"), dict)
                and row["cost_breakdown"].get("holding_cost") is not None
            ]
        ),
        average_backlog_cost=_mean(
            [
                float(row["cost_breakdown"]["backlog_cost"])
                for row in episode_rows
                if isinstance(row.get("cost_breakdown"), dict)
                and row["cost_breakdown"].get("backlog_cost") is not None
            ]
        ),
        average_ordering_cost=_mean(
            [
                float(row["cost_breakdown"]["ordering_cost"])
                for row in episode_rows
                if isinstance(row.get("cost_breakdown"), dict)
                and row["cost_breakdown"].get("ordering_cost") is not None
            ]
        ),
        average_replan_count=_mean([float(row["replan_count"]) for row in episode_rows if row.get("replan_count") is not None]),
        average_no_action_count=_mean(
            [float(row["no_action_count"]) for row in episode_rows if row.get("no_action_count") is not None]
        ),
        average_intervention_count=_mean(
            [float(row["intervention_count"]) for row in episode_rows if row.get("intervention_count") is not None]
        ),
        average_tool_call_count=_mean(
            [float(row["tool_call_count"]) for row in episode_rows if row.get("tool_call_count") is not None]
        ),
        regime_prediction_accuracy=_mean(
            [
                float(row["decision_quality"]["regime_prediction_accuracy"])
                for row in episode_rows
                if isinstance(row.get("decision_quality"), dict)
                and row["decision_quality"].get("regime_prediction_accuracy") is not None
            ]
        ),
        tool_usage_counts=_sorted_counts(tool_usage_counts),
        optimizer_input_change_count=optimizer_input_change_count,
        decision_changed_optimizer_input_count=decision_changed_optimizer_input_count,
        optimizer_output_changed_state_count=optimizer_output_changed_state_count,
        intervention_changed_outcome_count=intervention_changed_outcome_count,
        period_summaries=period_summaries,
    )


def _group_step_rows_by_period(rows: list[dict[str, object]]) -> dict[int, list[dict[str, object]]]:
    grouped: defaultdict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        period_index = row.get("period_index")
        if isinstance(period_index, int):
            grouped[period_index].append(row)
    return dict(sorted(grouped.items()))


def _build_period_behavior_summary(
    period_index: int,
    rows: list[dict[str, object]],
) -> PeriodBehaviorSummary:
    predicted_regime_counts = Counter(
        str(row["predicted_regime_label"])
        for row in rows
        if row.get("predicted_regime_label")
    )
    subgoal_counts = Counter(
        str(row["selected_subgoal"])
        for row in rows
        if row.get("selected_subgoal")
    )
    tool_counts = _counter_from_iterables(
        [
            tuple(str(item) for item in row.get("selected_tools", []) if isinstance(item, str))
            for row in rows
            if isinstance(row.get("selected_tools"), list)
        ]
    )
    update_request_counts = _counter_from_iterables(
        [
            tuple(str(item) for item in row.get("update_requests", []) if isinstance(item, str))
            for row in rows
            if isinstance(row.get("update_requests"), list)
        ]
    )
    safety_buffer_values = [
        float(row["scenario_adjustment_summary"]["safety_buffer_scale"])
        for row in rows
        if isinstance(row.get("scenario_adjustment_summary"), dict)
        and row["scenario_adjustment_summary"].get("safety_buffer_scale") is not None
    ]
    orders = [
        tuple(float(value) for value in row.get("optimizer_orders", []))
        for row in rows
        if isinstance(row.get("optimizer_orders"), list)
    ]
    pipelines = [
        tuple(float(value) for value in row.get("pipeline_by_echelon", []))
        for row in rows
        if isinstance(row.get("pipeline_by_echelon"), list)
    ]
    inventories = [
        tuple(float(value) for value in row.get("inventory_by_echelon", []))
        for row in rows
        if isinstance(row.get("inventory_by_echelon"), list)
    ]
    backorders = [
        tuple(float(value) for value in row.get("backorders_by_echelon", []))
        for row in rows
        if isinstance(row.get("backorders_by_echelon"), list)
    ]
    true_regime_label = next(
        (
            str(row["true_regime_label"])
            for row in rows
            if row.get("true_regime_label") is not None
        ),
        None,
    )
    average_orders = _average_sequence(orders)
    return PeriodBehaviorSummary(
        period_index=period_index,
        true_regime_label=true_regime_label,
        predicted_regime_counts=_sorted_counts(predicted_regime_counts),
        selected_subgoal_counts=_sorted_counts(subgoal_counts),
        selected_tool_counts=_sorted_counts(tool_counts),
        update_request_counts=_sorted_counts(update_request_counts),
        average_replan_rate=_mean([1.0 if bool(row.get("request_replan")) else 0.0 for row in rows]),
        average_no_action_rate=_mean(
            [1.0 if row.get("selected_subgoal") == "no_action" else 0.0 for row in rows]
        ),
        average_demand_outlook=_mean(
            [float(row["demand_outlook"]) for row in rows if row.get("demand_outlook") is not None]
        ),
        average_leadtime_outlook=_mean(
            [float(row["leadtime_outlook"]) for row in rows if row.get("leadtime_outlook") is not None]
        ),
        average_safety_buffer_scale=_mean(safety_buffer_values),
        average_orders_by_echelon=average_orders,
        average_pipeline_by_echelon=_average_sequence(pipelines),
        average_inventory_by_echelon=_average_sequence(inventories),
        average_backorders_by_echelon=_average_sequence(backorders),
        average_order_sum=_sequence_sum(average_orders) if average_orders else None,
        average_per_period_cost=_mean(
            [float(row["per_period_cost"]) for row in rows if row.get("per_period_cost") is not None]
        ),
        average_per_period_fill_rate=_mean(
            [float(row["per_period_fill_rate"]) for row in rows if row.get("per_period_fill_rate") is not None]
        ),
    )


def _build_period_highlights(
    mode_summaries: tuple[ModeScheduleSummary, ...],
) -> tuple[PeriodDifferenceHighlight, ...]:
    mode_lookup = {summary.mode: summary for summary in mode_summaries}
    llm = mode_lookup.get("llm_orchestrator")
    baseline = mode_lookup.get("deterministic_baseline")
    deterministic = mode_lookup.get("deterministic_orchestrator")
    if llm is None:
        return ()
    baseline_periods = {summary.period_index: summary for summary in baseline.period_summaries} if baseline else {}
    deterministic_periods = {summary.period_index: summary for summary in deterministic.period_summaries} if deterministic else {}
    highlights: list[PeriodDifferenceHighlight] = []
    for llm_period in llm.period_summaries:
        baseline_period = baseline_periods.get(llm_period.period_index)
        deterministic_period = deterministic_periods.get(llm_period.period_index)
        highlights.append(
            PeriodDifferenceHighlight(
                period_index=llm_period.period_index,
                true_regime_label=llm_period.true_regime_label,
                llm_predicted_regime_label=_dominant_label(llm_period.predicted_regime_counts),
                llm_selected_tool_counts=llm_period.selected_tool_counts,
                llm_update_request_counts=llm_period.update_request_counts,
                llm_average_replan_rate=llm_period.average_replan_rate,
                llm_average_demand_outlook=llm_period.average_demand_outlook,
                deterministic_baseline_average_demand_outlook=baseline_period.average_demand_outlook if baseline_period else None,
                deterministic_orchestrator_average_demand_outlook=deterministic_period.average_demand_outlook if deterministic_period else None,
                llm_average_safety_buffer_scale=llm_period.average_safety_buffer_scale,
                deterministic_baseline_average_safety_buffer_scale=baseline_period.average_safety_buffer_scale if baseline_period else None,
                deterministic_orchestrator_average_safety_buffer_scale=deterministic_period.average_safety_buffer_scale if deterministic_period else None,
                llm_average_order_sum=llm_period.average_order_sum,
                deterministic_baseline_average_order_sum=baseline_period.average_order_sum if baseline_period else None,
                deterministic_orchestrator_average_order_sum=deterministic_period.average_order_sum if deterministic_period else None,
                llm_order_sum_delta_vs_baseline=_difference(
                    llm_period.average_order_sum,
                    baseline_period.average_order_sum if baseline_period else None,
                ),
                llm_order_sum_delta_vs_deterministic_orchestrator=_difference(
                    llm_period.average_order_sum,
                    deterministic_period.average_order_sum if deterministic_period else None,
                ),
            )
        )
    return tuple(highlights)


def _difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _mode_summary_by_name(
    comparison: ScheduleComparisonSummary,
    mode_name: str,
) -> ModeScheduleSummary:
    return next(summary for summary in comparison.mode_summaries if summary.mode == mode_name)


__all__ = [
    "DEFAULT_PAPER_CANDIDATE_RESULTS_ROOT",
    "DEFAULT_MODE_NAMES",
    "ModeScheduleSummary",
    "PaperCandidateFailureAnalysis",
    "PeriodBehaviorSummary",
    "PeriodDifferenceHighlight",
    "ScheduleComparisonSummary",
    "analyze_paper_candidate_batch",
    "latest_paper_candidate_batch_dir",
    "load_paper_candidate_batch",
]
