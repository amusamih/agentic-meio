"""Live external-evidence failure analysis helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from meio.evaluation.external_evidence_analysis import (
    DEFAULT_EXTERNAL_EVIDENCE_RESULTS_ROOT,
    latest_external_evidence_batch_dir,
)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / float(len(values))


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


def _as_float(value: object) -> float | None:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _as_int(value: object) -> int | None:
    if value is None or isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _as_str_tuple(values: object) -> tuple[str, ...]:
    if not isinstance(values, list):
        return ()
    return tuple(str(value) for value in values if isinstance(value, str))


def _difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _schedule_summary_payload(
    aggregate_summary: dict[str, object],
    mode_name: str,
    schedule_name: str,
) -> dict[str, object] | None:
    mode_summaries = aggregate_summary.get("mode_summaries")
    if not isinstance(mode_summaries, list):
        return None
    for mode_payload in mode_summaries:
        if not isinstance(mode_payload, dict) or mode_payload.get("mode") != mode_name:
            continue
        schedule_breakdown = mode_payload.get("schedule_breakdown")
        if not isinstance(schedule_breakdown, list):
            return None
        for schedule_payload in schedule_breakdown:
            if (
                isinstance(schedule_payload, dict)
                and schedule_payload.get("schedule_name") == schedule_name
            ):
                return schedule_payload
        return None
    return None


def _schedule_metric(
    aggregate_summary: dict[str, object],
    mode_name: str,
    schedule_name: str,
    section_name: str,
    metric_name: str,
) -> float | None:
    schedule_payload = _schedule_summary_payload(aggregate_summary, mode_name, schedule_name)
    if schedule_payload is None:
        return None
    section = schedule_payload.get(section_name)
    if not isinstance(section, dict):
        return None
    return _as_float(section.get(metric_name))


def _schedule_int_metric(
    aggregate_summary: dict[str, object],
    mode_name: str,
    schedule_name: str,
    section_name: str,
    metric_name: str,
) -> int:
    schedule_payload = _schedule_summary_payload(aggregate_summary, mode_name, schedule_name)
    if schedule_payload is None:
        return 0
    section = schedule_payload.get(section_name)
    if not isinstance(section, dict):
        return 0
    value = _as_int(section.get(metric_name))
    return value if value is not None else 0


@dataclass(frozen=True, slots=True)
class ExternalEvidenceToolHint:
    """Compact extracted external-evidence interpretation."""

    recommended_regime_hint: str | None
    recommended_uncertainty_pressure: str | None
    false_alarm_evidence_detected: bool
    confidence: float | None


@dataclass(frozen=True, slots=True)
class ExternalEvidencePeriodComparison:
    """Side-by-side period comparison between the two live LLM branches."""

    schedule_name: str
    run_seed: int
    period_index: int
    true_regime_label: str | None
    internal_predicted_regime_label: str | None
    external_predicted_regime_label: str | None
    regime_changed: bool
    internal_request_replan: bool
    external_request_replan: bool
    replan_changed: bool
    internal_selected_subgoal: str | None
    external_selected_subgoal: str | None
    selected_subgoal_changed: bool
    internal_proposed_update_requests: tuple[str, ...]
    external_proposed_update_requests: tuple[str, ...]
    proposed_update_changed: bool
    internal_final_update_requests: tuple[str, ...]
    external_final_update_requests: tuple[str, ...]
    final_update_changed: bool
    internal_proposed_update_strength: str | None
    external_proposed_update_strength: str | None
    proposed_strength_changed: bool
    internal_final_update_strength: str | None
    external_final_update_strength: str | None
    final_strength_changed: bool
    internal_demand_outlook: float | None
    external_demand_outlook: float | None
    demand_outlook_delta: float | None
    internal_safety_buffer_scale: float | None
    external_safety_buffer_scale: float | None
    safety_buffer_scale_delta: float | None
    internal_per_period_cost: float | None
    external_per_period_cost: float | None
    per_period_cost_delta: float | None
    external_evidence_present: bool
    external_evidence_tool_used: bool
    external_evidence_supported_intervention: bool
    external_evidence_changed_optimizer_input: bool | None
    external_evidence_false_alarm: bool
    evidence_present_but_should_ignore: bool
    evidence_worsened_step_cost: bool
    external_evidence_hint: ExternalEvidenceToolHint | None = None


@dataclass(frozen=True, slots=True)
class ExternalEvidenceSeedComparison:
    """Per-seed schedule outcome comparison."""

    schedule_name: str
    run_seed: int
    internal_total_cost: float | None
    external_total_cost: float | None
    total_cost_delta: float | None
    internal_regime_accuracy: float | None
    external_regime_accuracy: float | None
    internal_replan_count: int | None
    external_replan_count: int | None
    internal_intervention_count: int | None
    external_intervention_count: int | None


@dataclass(frozen=True, slots=True)
class ExternalEvidenceFailureScheduleSummary:
    """Schedule-level live failure summary."""

    schedule_name: str
    internal_average_total_cost: float | None
    external_average_total_cost: float | None
    total_cost_delta: float | None
    internal_regime_accuracy: float | None
    external_regime_accuracy: float | None
    internal_replan_rate: float | None
    external_replan_rate: float | None
    internal_intervention_rate: float | None
    external_intervention_rate: float | None
    external_evidence_period_count: int
    external_evidence_tool_use_count: int
    false_alarm_evidence_count: int
    evidence_changed_regime_count: int
    evidence_changed_replan_count: int
    evidence_changed_proposed_update_count: int
    evidence_changed_final_update_count: int
    external_evidence_changed_optimizer_input_count: int
    evidence_strength_only_change_count: int
    evidence_present_but_should_ignore_count: int
    evidence_worsened_step_cost_count: int
    seed_comparisons: tuple[ExternalEvidenceSeedComparison, ...]
    period_comparisons: tuple[ExternalEvidencePeriodComparison, ...]


@dataclass(frozen=True, slots=True)
class ExternalEvidenceFailureAnalysis:
    """Top-level side-by-side comparison for one live external-evidence batch."""

    run_dir: str
    benchmark_id: str | None
    artifact_use_class: str | None
    schedule_names: tuple[str, ...]
    schedule_summaries: tuple[ExternalEvidenceFailureScheduleSummary, ...]
    llm_help_schedules: tuple[str, ...]
    llm_hurt_schedules: tuple[str, ...]
    llm_neutral_schedules: tuple[str, ...]
    total_external_evidence_period_count: int
    total_evidence_changed_regime_count: int
    total_evidence_changed_replan_count: int
    total_evidence_changed_proposed_update_count: int
    total_evidence_changed_final_update_count: int
    total_external_evidence_changed_optimizer_input_count: int
    total_evidence_strength_only_change_count: int
    total_evidence_present_but_should_ignore_count: int
    total_evidence_worsened_step_cost_count: int


@dataclass(frozen=True, slots=True)
class _FailureArtifacts:
    run_dir: Path
    aggregate_summary: dict[str, object]
    run_manifest: dict[str, object]
    episode_rows: tuple[dict[str, object], ...]
    step_rows: tuple[dict[str, object], ...]
    tool_rows: tuple[dict[str, object], ...]


def load_external_evidence_failure_batch(
    run_dir: str | Path | None = None,
    *,
    results_root: str | Path = DEFAULT_EXTERNAL_EVIDENCE_RESULTS_ROOT,
) -> _FailureArtifacts:
    """Load saved artifacts for side-by-side live external-evidence analysis."""

    resolved_dir = Path(run_dir) if run_dir is not None else latest_external_evidence_batch_dir(results_root)
    return _FailureArtifacts(
        run_dir=resolved_dir,
        aggregate_summary=_load_json(resolved_dir / "aggregate_summary.json"),
        run_manifest=_load_json(resolved_dir / "run_manifest.json"),
        episode_rows=_load_jsonl(resolved_dir / "episode_summaries.jsonl"),
        step_rows=_load_jsonl(resolved_dir / "step_traces.jsonl"),
        tool_rows=_load_jsonl(resolved_dir / "tool_call_traces.jsonl"),
    )


def analyze_external_evidence_failures(
    run_dir: str | Path | None = None,
    *,
    results_root: str | Path = DEFAULT_EXTERNAL_EVIDENCE_RESULTS_ROOT,
) -> ExternalEvidenceFailureAnalysis:
    """Compare the live internal-only and live external-evidence branches."""

    artifacts = load_external_evidence_failure_batch(run_dir, results_root=results_root)
    schedule_names = tuple(
        sorted(
            {
                str(row["schedule_name"])
                for row in artifacts.episode_rows
                if row.get("schedule_name") is not None
            }
        )
    )
    schedule_summaries = tuple(
        _build_schedule_summary(artifacts, schedule_name) for schedule_name in schedule_names
    )
    llm_help_schedules = tuple(
        summary.schedule_name
        for summary in schedule_summaries
        if summary.total_cost_delta is not None and summary.total_cost_delta < -1e-9
    )
    llm_hurt_schedules = tuple(
        summary.schedule_name
        for summary in schedule_summaries
        if summary.total_cost_delta is not None and summary.total_cost_delta > 1e-9
    )
    llm_neutral_schedules = tuple(
        summary.schedule_name
        for summary in schedule_summaries
        if summary.total_cost_delta is None or abs(summary.total_cost_delta) <= 1e-9
    )
    benchmark_id = artifacts.aggregate_summary.get("benchmark_id")
    return ExternalEvidenceFailureAnalysis(
        run_dir=str(artifacts.run_dir),
        benchmark_id=benchmark_id if isinstance(benchmark_id, str) else None,
        artifact_use_class=_artifact_use_class(artifacts.run_manifest),
        schedule_names=schedule_names,
        schedule_summaries=schedule_summaries,
        llm_help_schedules=llm_help_schedules,
        llm_hurt_schedules=llm_hurt_schedules,
        llm_neutral_schedules=llm_neutral_schedules,
        total_external_evidence_period_count=sum(
            summary.external_evidence_period_count for summary in schedule_summaries
        ),
        total_evidence_changed_regime_count=sum(
            summary.evidence_changed_regime_count for summary in schedule_summaries
        ),
        total_evidence_changed_replan_count=sum(
            summary.evidence_changed_replan_count for summary in schedule_summaries
        ),
        total_evidence_changed_proposed_update_count=sum(
            summary.evidence_changed_proposed_update_count for summary in schedule_summaries
        ),
        total_evidence_changed_final_update_count=sum(
            summary.evidence_changed_final_update_count for summary in schedule_summaries
        ),
        total_external_evidence_changed_optimizer_input_count=sum(
            summary.external_evidence_changed_optimizer_input_count
            for summary in schedule_summaries
        ),
        total_evidence_strength_only_change_count=sum(
            summary.evidence_strength_only_change_count for summary in schedule_summaries
        ),
        total_evidence_present_but_should_ignore_count=sum(
            summary.evidence_present_but_should_ignore_count for summary in schedule_summaries
        ),
        total_evidence_worsened_step_cost_count=sum(
            summary.evidence_worsened_step_cost_count for summary in schedule_summaries
        ),
    )


def _build_schedule_summary(
    artifacts: _FailureArtifacts,
    schedule_name: str,
) -> ExternalEvidenceFailureScheduleSummary:
    period_comparisons = tuple(
        _build_period_comparison(artifacts, schedule_name, run_seed, period_index)
        for run_seed, period_index in _schedule_period_keys(artifacts.step_rows, schedule_name)
    )
    seed_comparisons = tuple(
        _build_seed_comparison(artifacts, schedule_name, run_seed)
        for run_seed in _schedule_seed_values(artifacts.episode_rows, schedule_name)
    )
    return ExternalEvidenceFailureScheduleSummary(
        schedule_name=schedule_name,
        internal_average_total_cost=_schedule_metric(
            artifacts.aggregate_summary,
            "llm_orchestrator_internal_only",
            schedule_name,
            "performance_summary",
            "average_total_cost",
        ),
        external_average_total_cost=_schedule_metric(
            artifacts.aggregate_summary,
            "llm_orchestrator_with_external_evidence",
            schedule_name,
            "performance_summary",
            "average_total_cost",
        ),
        total_cost_delta=_difference(
            _schedule_metric(
                artifacts.aggregate_summary,
                "llm_orchestrator_with_external_evidence",
                schedule_name,
                "performance_summary",
                "average_total_cost",
            ),
            _schedule_metric(
                artifacts.aggregate_summary,
                "llm_orchestrator_internal_only",
                schedule_name,
                "performance_summary",
                "average_total_cost",
            ),
        ),
        internal_regime_accuracy=_schedule_metric(
            artifacts.aggregate_summary,
            "llm_orchestrator_internal_only",
            schedule_name,
            "decision_quality",
            "regime_prediction_accuracy",
        ),
        external_regime_accuracy=_schedule_metric(
            artifacts.aggregate_summary,
            "llm_orchestrator_with_external_evidence",
            schedule_name,
            "decision_quality",
            "regime_prediction_accuracy",
        ),
        internal_replan_rate=_schedule_metric(
            artifacts.aggregate_summary,
            "llm_orchestrator_internal_only",
            schedule_name,
            "decision_quality",
            "replan_rate",
        ),
        external_replan_rate=_schedule_metric(
            artifacts.aggregate_summary,
            "llm_orchestrator_with_external_evidence",
            schedule_name,
            "decision_quality",
            "replan_rate",
        ),
        internal_intervention_rate=_schedule_metric(
            artifacts.aggregate_summary,
            "llm_orchestrator_internal_only",
            schedule_name,
            "decision_quality",
            "intervention_rate",
        ),
        external_intervention_rate=_schedule_metric(
            artifacts.aggregate_summary,
            "llm_orchestrator_with_external_evidence",
            schedule_name,
            "decision_quality",
            "intervention_rate",
        ),
        external_evidence_period_count=sum(
            1 for comparison in period_comparisons if comparison.external_evidence_present
        ),
        external_evidence_tool_use_count=sum(
            1 for comparison in period_comparisons if comparison.external_evidence_tool_used
        ),
        false_alarm_evidence_count=sum(
            1 for comparison in period_comparisons if comparison.external_evidence_false_alarm
        ),
        evidence_changed_regime_count=sum(
            1
            for comparison in period_comparisons
            if comparison.external_evidence_present and comparison.regime_changed
        ),
        evidence_changed_replan_count=sum(
            1
            for comparison in period_comparisons
            if comparison.external_evidence_present and comparison.replan_changed
        ),
        evidence_changed_proposed_update_count=sum(
            1
            for comparison in period_comparisons
            if comparison.external_evidence_present and comparison.proposed_update_changed
        ),
        evidence_changed_final_update_count=sum(
            1
            for comparison in period_comparisons
            if comparison.external_evidence_present and comparison.final_update_changed
        ),
        external_evidence_changed_optimizer_input_count=sum(
            1
            for comparison in period_comparisons
            if comparison.external_evidence_changed_optimizer_input is True
        ),
        evidence_strength_only_change_count=sum(
            1
            for comparison in period_comparisons
            if comparison.external_evidence_changed_optimizer_input is True
            and not comparison.regime_changed
            and not comparison.replan_changed
            and not comparison.final_update_changed
            and not comparison.final_strength_changed
        ),
        evidence_present_but_should_ignore_count=sum(
            1
            for comparison in period_comparisons
            if comparison.evidence_present_but_should_ignore
        ),
        evidence_worsened_step_cost_count=sum(
            1
            for comparison in period_comparisons
            if comparison.evidence_worsened_step_cost
        ),
        seed_comparisons=seed_comparisons,
        period_comparisons=period_comparisons,
    )


def _schedule_seed_values(
    episode_rows: tuple[dict[str, object], ...],
    schedule_name: str,
) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                int(row["run_seed"])
                for row in episode_rows
                if row.get("schedule_name") == schedule_name and isinstance(row.get("run_seed"), int)
            }
        )
    )


def _schedule_period_keys(
    step_rows: tuple[dict[str, object], ...],
    schedule_name: str,
) -> tuple[tuple[int, int], ...]:
    return tuple(
        sorted(
            {
                (int(row["run_seed"]), int(row["period_index"]))
                for row in step_rows
                if row.get("schedule_name") == schedule_name
                and row.get("mode") == "llm_orchestrator_with_external_evidence"
                and isinstance(row.get("run_seed"), int)
                and isinstance(row.get("period_index"), int)
            }
        )
    )


def _build_seed_comparison(
    artifacts: _FailureArtifacts,
    schedule_name: str,
    run_seed: int,
) -> ExternalEvidenceSeedComparison:
    internal_row = _episode_row(
        artifacts.episode_rows,
        "llm_orchestrator_internal_only",
        schedule_name,
        run_seed,
    )
    external_row = _episode_row(
        artifacts.episode_rows,
        "llm_orchestrator_with_external_evidence",
        schedule_name,
        run_seed,
    )
    internal_accuracy = _seed_regime_accuracy(
        artifacts.step_rows,
        "llm_orchestrator_internal_only",
        schedule_name,
        run_seed,
    )
    external_accuracy = _seed_regime_accuracy(
        artifacts.step_rows,
        "llm_orchestrator_with_external_evidence",
        schedule_name,
        run_seed,
    )
    internal_total_cost = _as_float(internal_row.get("total_cost")) if internal_row else None
    external_total_cost = _as_float(external_row.get("total_cost")) if external_row else None
    return ExternalEvidenceSeedComparison(
        schedule_name=schedule_name,
        run_seed=run_seed,
        internal_total_cost=internal_total_cost,
        external_total_cost=external_total_cost,
        total_cost_delta=_difference(external_total_cost, internal_total_cost),
        internal_regime_accuracy=internal_accuracy,
        external_regime_accuracy=external_accuracy,
        internal_replan_count=_as_int(internal_row.get("replan_count")) if internal_row else None,
        external_replan_count=_as_int(external_row.get("replan_count")) if external_row else None,
        internal_intervention_count=_as_int(internal_row.get("intervention_count")) if internal_row else None,
        external_intervention_count=_as_int(external_row.get("intervention_count")) if external_row else None,
    )


def _build_period_comparison(
    artifacts: _FailureArtifacts,
    schedule_name: str,
    run_seed: int,
    period_index: int,
) -> ExternalEvidencePeriodComparison:
    internal_row = _step_row(
        artifacts.step_rows,
        "llm_orchestrator_internal_only",
        schedule_name,
        run_seed,
        period_index,
    )
    external_row = _step_row(
        artifacts.step_rows,
        "llm_orchestrator_with_external_evidence",
        schedule_name,
        run_seed,
        period_index,
    )
    if internal_row is None or external_row is None:
        raise ValueError(
            "Matching live LLM step rows are required for external-evidence failure analysis."
        )
    tool_hint = _external_evidence_tool_hint(
        artifacts.tool_rows,
        schedule_name,
        run_seed,
        period_index,
    )
    internal_proposed_updates = _as_str_tuple(internal_row.get("proposed_update_requests"))
    external_proposed_updates = _as_str_tuple(external_row.get("proposed_update_requests"))
    internal_final_updates = _as_str_tuple(internal_row.get("update_requests"))
    external_final_updates = _as_str_tuple(external_row.get("update_requests"))
    internal_safety = _safety_buffer_scale(internal_row)
    external_safety = _safety_buffer_scale(external_row)
    evidence_present = bool(external_row.get("external_evidence_present"))
    evidence_false_alarm = bool(external_row.get("external_evidence_false_alarm"))
    internal_regime = _string_or_none(internal_row.get("predicted_regime_label"))
    external_regime = _string_or_none(external_row.get("predicted_regime_label"))
    external_final_strength = _string_or_none(external_row.get("final_update_strength"))
    evidence_should_ignore = (
        evidence_present
        and _string_or_none(external_row.get("true_regime_label")) == "normal"
        and internal_regime == "normal"
        and (
            external_regime != "normal"
            or bool(external_row.get("request_replan"))
            or external_final_strength not in (None, "keep_current")
        )
    )
    per_period_cost_delta = _difference(
        _as_float(external_row.get("per_period_cost")),
        _as_float(internal_row.get("per_period_cost")),
    )
    return ExternalEvidencePeriodComparison(
        schedule_name=schedule_name,
        run_seed=run_seed,
        period_index=period_index,
        true_regime_label=_string_or_none(external_row.get("true_regime_label")),
        internal_predicted_regime_label=internal_regime,
        external_predicted_regime_label=external_regime,
        regime_changed=internal_regime != external_regime,
        internal_request_replan=bool(internal_row.get("request_replan")),
        external_request_replan=bool(external_row.get("request_replan")),
        replan_changed=bool(internal_row.get("request_replan")) != bool(external_row.get("request_replan")),
        internal_selected_subgoal=_string_or_none(internal_row.get("selected_subgoal")),
        external_selected_subgoal=_string_or_none(external_row.get("selected_subgoal")),
        selected_subgoal_changed=_string_or_none(internal_row.get("selected_subgoal"))
        != _string_or_none(external_row.get("selected_subgoal")),
        internal_proposed_update_requests=internal_proposed_updates,
        external_proposed_update_requests=external_proposed_updates,
        proposed_update_changed=internal_proposed_updates != external_proposed_updates,
        internal_final_update_requests=internal_final_updates,
        external_final_update_requests=external_final_updates,
        final_update_changed=internal_final_updates != external_final_updates,
        internal_proposed_update_strength=_string_or_none(
            internal_row.get("proposed_update_strength")
        ),
        external_proposed_update_strength=_string_or_none(
            external_row.get("proposed_update_strength")
        ),
        proposed_strength_changed=_string_or_none(internal_row.get("proposed_update_strength"))
        != _string_or_none(external_row.get("proposed_update_strength")),
        internal_final_update_strength=_string_or_none(internal_row.get("final_update_strength")),
        external_final_update_strength=external_final_strength,
        final_strength_changed=_string_or_none(internal_row.get("final_update_strength"))
        != external_final_strength,
        internal_demand_outlook=_as_float(internal_row.get("demand_outlook")),
        external_demand_outlook=_as_float(external_row.get("demand_outlook")),
        demand_outlook_delta=_difference(
            _as_float(external_row.get("demand_outlook")),
            _as_float(internal_row.get("demand_outlook")),
        ),
        internal_safety_buffer_scale=internal_safety,
        external_safety_buffer_scale=external_safety,
        safety_buffer_scale_delta=_difference(external_safety, internal_safety),
        internal_per_period_cost=_as_float(internal_row.get("per_period_cost")),
        external_per_period_cost=_as_float(external_row.get("per_period_cost")),
        per_period_cost_delta=per_period_cost_delta,
        external_evidence_present=evidence_present,
        external_evidence_tool_used=bool(external_row.get("external_evidence_tool_used")),
        external_evidence_supported_intervention=bool(
            external_row.get("external_evidence_supported_intervention")
        ),
        external_evidence_changed_optimizer_input=_bool_or_none(
            external_row.get("external_evidence_changed_optimizer_input")
        ),
        external_evidence_false_alarm=evidence_false_alarm,
        evidence_present_but_should_ignore=evidence_should_ignore,
        evidence_worsened_step_cost=bool(
            evidence_present
            and per_period_cost_delta is not None
            and per_period_cost_delta > 1e-9
        ),
        external_evidence_hint=tool_hint,
    )


def _seed_regime_accuracy(
    step_rows: tuple[dict[str, object], ...],
    mode_name: str,
    schedule_name: str,
    run_seed: int,
) -> float | None:
    rows = [
        row
        for row in step_rows
        if row.get("mode") == mode_name
        and row.get("schedule_name") == schedule_name
        and row.get("run_seed") == run_seed
    ]
    if not rows:
        return None
    values = [
        1.0
        for row in rows
        if row.get("predicted_regime_label") == row.get("true_regime_label")
    ]
    return float(len(values)) / float(len(rows))


def _episode_row(
    episode_rows: tuple[dict[str, object], ...],
    mode_name: str,
    schedule_name: str,
    run_seed: int,
) -> dict[str, object] | None:
    for row in episode_rows:
        if (
            row.get("mode") == mode_name
            and row.get("schedule_name") == schedule_name
            and row.get("run_seed") == run_seed
        ):
            return row
    return None


def _step_row(
    step_rows: tuple[dict[str, object], ...],
    mode_name: str,
    schedule_name: str,
    run_seed: int,
    period_index: int,
) -> dict[str, object] | None:
    for row in step_rows:
        if (
            row.get("mode") == mode_name
            and row.get("schedule_name") == schedule_name
            and row.get("run_seed") == run_seed
            and row.get("period_index") == period_index
        ):
            return row
    return None


def _external_evidence_tool_hint(
    tool_rows: tuple[dict[str, object], ...],
    schedule_name: str,
    run_seed: int,
    period_index: int,
) -> ExternalEvidenceToolHint | None:
    for row in tool_rows:
        if (
            row.get("mode") == "llm_orchestrator_with_external_evidence"
            and row.get("tool_id") == "external_evidence_tool"
            and row.get("schedule_name") == schedule_name
            and row.get("run_seed") == run_seed
            and row.get("period_index") == period_index
        ):
            tool_output = row.get("tool_output")
            if not isinstance(tool_output, dict):
                return None
            structured_output = tool_output.get("structured_output")
            if not isinstance(structured_output, dict):
                return None
            interpretation = structured_output.get("external_evidence_interpretation")
            if not isinstance(interpretation, dict):
                return None
            return ExternalEvidenceToolHint(
                recommended_regime_hint=_string_or_none(
                    interpretation.get("recommended_regime_hint")
                ),
                recommended_uncertainty_pressure=_string_or_none(
                    interpretation.get("recommended_uncertainty_pressure")
                ),
                false_alarm_evidence_detected=bool(
                    interpretation.get("false_alarm_evidence_detected")
                ),
                confidence=_as_float(interpretation.get("confidence")),
            )
    return None


def _artifact_use_class(run_manifest: dict[str, object]) -> str | None:
    value = run_manifest.get("artifact_use_class")
    return value if isinstance(value, str) else None


def _safety_buffer_scale(row: dict[str, object]) -> float | None:
    summary = row.get("scenario_adjustment_summary")
    if not isinstance(summary, dict):
        return None
    return _as_float(summary.get("safety_buffer_scale"))


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _bool_or_none(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


__all__ = [
    "DEFAULT_EXTERNAL_EVIDENCE_RESULTS_ROOT",
    "ExternalEvidenceFailureAnalysis",
    "ExternalEvidenceFailureScheduleSummary",
    "ExternalEvidencePeriodComparison",
    "ExternalEvidenceSeedComparison",
    "ExternalEvidenceToolHint",
    "analyze_external_evidence_failures",
    "load_external_evidence_failure_batch",
]
