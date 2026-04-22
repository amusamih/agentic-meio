"""Analysis helpers for the semi-synthetic external-evidence branch."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from meio.evaluation.logging_schema import EpisodeSummaryRecord, StepTraceRecord


DEFAULT_EXTERNAL_EVIDENCE_RESULTS_ROOT = Path("results/stockpyl_serial_external_evidence")


@dataclass(frozen=True, slots=True)
class ExternalEvidenceMetrics:
    """Minimal honest metrics for external-evidence use in one mode."""

    external_evidence_period_count: int
    external_evidence_tool_call_count: int
    false_alarm_evidence_count: int
    evidence_supported_intervention_count: int
    external_evidence_changed_optimizer_input_count: int
    evidence_fusion_cap_count: int = 0
    capped_external_strengthening_count: int = 0
    early_evidence_confirmation_gate_count: int = 0
    early_evidence_family_change_block_count: int = 0

    def __post_init__(self) -> None:
        for field_name in (
            "external_evidence_period_count",
            "external_evidence_tool_call_count",
            "false_alarm_evidence_count",
            "evidence_supported_intervention_count",
            "external_evidence_changed_optimizer_input_count",
            "evidence_fusion_cap_count",
            "capped_external_strengthening_count",
            "early_evidence_confirmation_gate_count",
            "early_evidence_family_change_block_count",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative.")


@dataclass(frozen=True, slots=True)
class ExternalEvidenceModeSummary:
    """Compact mode-level summary for one external-evidence batch."""

    mode: str
    artifact_use_class: str | None
    average_total_cost: float | None
    average_fill_rate: float | None
    regime_prediction_accuracy: float | None
    no_action_rate: float | None
    replan_rate: float | None
    intervention_rate: float | None
    average_tool_call_count: float | None
    fallback_count: int
    invalid_output_count: int
    external_evidence_summary: ExternalEvidenceMetrics | None = None


@dataclass(frozen=True, slots=True)
class ExternalEvidenceScheduleComparison:
    """Schedule-level comparison between the two LLM branches."""

    schedule_name: str
    deterministic_baseline_cost: float | None
    deterministic_orchestrator_cost: float | None
    llm_internal_only_cost: float | None
    llm_with_external_evidence_cost: float | None
    llm_external_minus_internal_cost: float | None
    llm_outcome_label: str
    false_alarm_like: bool


@dataclass(frozen=True, slots=True)
class ExternalEvidenceBatchAnalysis:
    """Top-level summary for one saved external-evidence batch directory."""

    run_dir: str
    benchmark_id: str | None
    artifact_use_class: str | None
    mode_artifact_use_classes: tuple[tuple[str, str], ...]
    mode_names: tuple[str, ...]
    schedule_names: tuple[str, ...]
    seed_values: tuple[int, ...]
    mode_summaries: tuple[ExternalEvidenceModeSummary, ...]
    schedule_comparisons: tuple[ExternalEvidenceScheduleComparison, ...]
    llm_help_schedules: tuple[str, ...] = ()
    llm_hurt_schedules: tuple[str, ...] = ()
    llm_neutral_schedules: tuple[str, ...] = ()


def summarize_external_evidence_metrics(
    *,
    episode_records: tuple[EpisodeSummaryRecord, ...],
    step_records: tuple[StepTraceRecord, ...],
) -> ExternalEvidenceMetrics | None:
    """Aggregate the current supportable external-evidence metrics."""

    period_count = sum(record.external_evidence_period_count for record in episode_records)
    tool_call_count = sum(
        record.external_evidence_tool_call_count for record in episode_records
    )
    false_alarm_count = sum(record.false_alarm_evidence_count for record in episode_records)
    intervention_count = sum(
        record.evidence_supported_intervention_count for record in episode_records
    )
    optimizer_input_change_count = sum(
        1
        for record in step_records
        if record.external_evidence_changed_optimizer_input is True
    )
    evidence_fusion_cap_count = sum(
        1
        for record in step_records
        if record.external_evidence_fusion_cap_applied
    )
    capped_external_strengthening_count = sum(
        1
        for record in step_records
        if record.proposed_external_strengthening is not None
        and record.final_external_strengthening is not None
        and record.proposed_external_strengthening != record.final_external_strengthening
    )
    early_evidence_confirmation_gate_count = sum(
        1
        for record in step_records
        if record.early_evidence_confirmation_gate_applied
    )
    early_evidence_family_change_block_count = sum(
        1
        for record in step_records
        if record.early_evidence_family_change_blocked
    )
    if (
        period_count == 0
        and tool_call_count == 0
        and false_alarm_count == 0
        and intervention_count == 0
        and optimizer_input_change_count == 0
        and evidence_fusion_cap_count == 0
        and capped_external_strengthening_count == 0
        and early_evidence_confirmation_gate_count == 0
        and early_evidence_family_change_block_count == 0
    ):
        return None
    return ExternalEvidenceMetrics(
        external_evidence_period_count=period_count,
        external_evidence_tool_call_count=tool_call_count,
        false_alarm_evidence_count=false_alarm_count,
        evidence_supported_intervention_count=intervention_count,
        external_evidence_changed_optimizer_input_count=optimizer_input_change_count,
        evidence_fusion_cap_count=evidence_fusion_cap_count,
        capped_external_strengthening_count=capped_external_strengthening_count,
        early_evidence_confirmation_gate_count=early_evidence_confirmation_gate_count,
        early_evidence_family_change_block_count=early_evidence_family_change_block_count,
    )


def latest_external_evidence_batch_dir(
    results_root: str | Path = DEFAULT_EXTERNAL_EVIDENCE_RESULTS_ROOT,
) -> Path:
    """Return the latest saved external-evidence batch directory."""

    root = Path(results_root)
    candidates = tuple(path for path in root.iterdir() if path.is_dir()) if root.exists() else ()
    if not candidates:
        raise FileNotFoundError(
            f"No saved external-evidence batch directories found under {root}."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_external_evidence_batch(
    run_dir: str | Path | None = None,
    *,
    results_root: str | Path = DEFAULT_EXTERNAL_EVIDENCE_RESULTS_ROOT,
) -> tuple[Path, dict[str, object], dict[str, object]]:
    """Load the saved aggregate summary and run manifest for one batch."""

    resolved_dir = Path(run_dir) if run_dir is not None else latest_external_evidence_batch_dir(results_root)
    return (
        resolved_dir,
        _load_json(resolved_dir / "aggregate_summary.json"),
        _load_json(resolved_dir / "run_manifest.json"),
    )


def analyze_external_evidence_batch(
    run_dir: str | Path | None = None,
    *,
    results_root: str | Path = DEFAULT_EXTERNAL_EVIDENCE_RESULTS_ROOT,
) -> ExternalEvidenceBatchAnalysis:
    """Compare the internal-only and external-evidence LLM branches."""

    resolved_dir, aggregate_summary, run_manifest = load_external_evidence_batch(
        run_dir,
        results_root=results_root,
    )
    mode_payloads = tuple(
        payload
        for payload in aggregate_summary.get("mode_summaries", ())
        if isinstance(payload, dict)
    )
    mode_names = tuple(
        str(item)
        for item in aggregate_summary.get("mode_names", ())
        if isinstance(item, str)
    )
    schedule_names = tuple(
        str(item)
        for item in aggregate_summary.get("schedule_names", ())
        if isinstance(item, str)
    )
    seed_values = tuple(
        int(item)
        for item in aggregate_summary.get("seed_values", ())
        if isinstance(item, int)
    )
    mode_summaries = tuple(_build_mode_summary(payload) for payload in mode_payloads)
    schedule_comparisons = tuple(
        _build_schedule_comparison(schedule_name, mode_payloads)
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
    return ExternalEvidenceBatchAnalysis(
        run_dir=str(resolved_dir),
        benchmark_id=_payload_string(aggregate_summary, "benchmark_id"),
        artifact_use_class=_payload_string(run_manifest, "artifact_use_class"),
        mode_artifact_use_classes=tuple(
            (str(mode_name), str(use_class))
            for mode_name, use_class in run_manifest.get("mode_artifact_use_classes", ())
            if isinstance(mode_name, str) and isinstance(use_class, str)
        ),
        mode_names=mode_names,
        schedule_names=schedule_names,
        seed_values=seed_values,
        mode_summaries=mode_summaries,
        schedule_comparisons=schedule_comparisons,
        llm_help_schedules=llm_help_schedules,
        llm_hurt_schedules=llm_hurt_schedules,
        llm_neutral_schedules=llm_neutral_schedules,
    )


def _build_mode_summary(mode_payload: dict[str, object]) -> ExternalEvidenceModeSummary:
    decision_quality = mode_payload.get("decision_quality")
    performance_summary = mode_payload.get("performance_summary")
    telemetry_metrics = mode_payload.get("telemetry_metrics")
    tool_use_summary = mode_payload.get("tool_use_summary")
    external_summary = None
    if isinstance(mode_payload.get("external_evidence_summary"), dict):
        payload = mode_payload["external_evidence_summary"]
        external_summary = ExternalEvidenceMetrics(
            external_evidence_period_count=_payload_int(
                payload,
                "external_evidence_period_count",
            ),
            external_evidence_tool_call_count=_payload_int(
                payload,
                "external_evidence_tool_call_count",
            ),
            false_alarm_evidence_count=_payload_int(
                payload,
                "false_alarm_evidence_count",
            ),
            evidence_supported_intervention_count=_payload_int(
                payload,
                "evidence_supported_intervention_count",
            ),
            external_evidence_changed_optimizer_input_count=_payload_int(
                payload,
                "external_evidence_changed_optimizer_input_count",
            ),
            evidence_fusion_cap_count=_payload_int(
                payload,
                "evidence_fusion_cap_count",
            ),
            capped_external_strengthening_count=_payload_int(
                payload,
                "capped_external_strengthening_count",
            ),
            early_evidence_confirmation_gate_count=_payload_int(
                payload,
                "early_evidence_confirmation_gate_count",
            ),
            early_evidence_family_change_block_count=_payload_int(
                payload,
                "early_evidence_family_change_block_count",
            ),
        )
    return ExternalEvidenceModeSummary(
        mode=_payload_string(mode_payload, "mode") or "unknown",
        artifact_use_class=_payload_string(mode_payload, "artifact_use_class"),
        average_total_cost=_payload_float(performance_summary, "average_total_cost"),
        average_fill_rate=_payload_float(performance_summary, "average_fill_rate"),
        regime_prediction_accuracy=_payload_float(
            decision_quality,
            "regime_prediction_accuracy",
        ),
        no_action_rate=_payload_float(decision_quality, "no_action_rate"),
        replan_rate=_payload_float(decision_quality, "replan_rate"),
        intervention_rate=_payload_float(decision_quality, "intervention_rate"),
        average_tool_call_count=_payload_float(tool_use_summary, "average_tool_call_count"),
        fallback_count=_payload_int(telemetry_metrics, "fallback_count"),
        invalid_output_count=_payload_int(telemetry_metrics, "invalid_output_count"),
        external_evidence_summary=external_summary,
    )


def _build_schedule_comparison(
    schedule_name: str,
    mode_payloads: tuple[dict[str, object], ...],
) -> ExternalEvidenceScheduleComparison:
    cost_by_mode: dict[str, float | None] = {}
    for mode_payload in mode_payloads:
        mode_name = _payload_string(mode_payload, "mode")
        if mode_name is None:
            continue
        cost_by_mode[mode_name] = _schedule_cost(mode_payload, schedule_name)
    llm_internal_cost = cost_by_mode.get("llm_orchestrator_internal_only")
    llm_external_cost = cost_by_mode.get("llm_orchestrator_with_external_evidence")
    delta = _difference(llm_external_cost, llm_internal_cost)
    if delta is None:
        outcome_label = "neutral"
    elif delta < -1e-9:
        outcome_label = "helps"
    elif delta > 1e-9:
        outcome_label = "hurts"
    else:
        outcome_label = "neutral"
    return ExternalEvidenceScheduleComparison(
        schedule_name=schedule_name,
        deterministic_baseline_cost=cost_by_mode.get("deterministic_baseline"),
        deterministic_orchestrator_cost=cost_by_mode.get("deterministic_orchestrator"),
        llm_internal_only_cost=llm_internal_cost,
        llm_with_external_evidence_cost=llm_external_cost,
        llm_external_minus_internal_cost=delta,
        llm_outcome_label=outcome_label,
        false_alarm_like="false_alarm" in schedule_name,
    )


def _schedule_cost(mode_payload: dict[str, object], schedule_name: str) -> float | None:
    schedule_breakdown = mode_payload.get("schedule_breakdown")
    if not isinstance(schedule_breakdown, list):
        return None
    for item in schedule_breakdown:
        if not isinstance(item, dict):
            continue
        if _payload_string(item, "schedule_name") != schedule_name:
            continue
        return _payload_float(item.get("performance_summary"), "average_total_cost")
    return None


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _payload_string(payload: object, key: str) -> str | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string when present.")
    return value


def _payload_int(payload: object, key: str) -> int:
    if not isinstance(payload, dict):
        return 0
    value = payload.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer when present.")
    return value


def _payload_float(payload: object, key: str) -> float | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric when present.")
    return float(value)


def _difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


__all__ = [
    "DEFAULT_EXTERNAL_EVIDENCE_RESULTS_ROOT",
    "ExternalEvidenceBatchAnalysis",
    "ExternalEvidenceMetrics",
    "ExternalEvidenceModeSummary",
    "ExternalEvidenceScheduleComparison",
    "analyze_external_evidence_batch",
    "latest_external_evidence_batch_dir",
    "load_external_evidence_batch",
    "summarize_external_evidence_metrics",
]
