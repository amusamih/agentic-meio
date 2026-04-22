"""Human-inspectable external-evidence views for saved live comparison runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from meio.contracts import RegimeLabel
from meio.data.external_evidence import ExternalEvidenceBatch
from meio.evaluation.external_evidence_analysis import (
    DEFAULT_EXTERNAL_EVIDENCE_RESULTS_ROOT,
    latest_external_evidence_batch_dir,
)
from meio.evaluation.external_evidence_failure_analysis import (
    ExternalEvidencePeriodComparison,
    ExternalEvidenceToolHint,
    analyze_external_evidence_failures,
)
from meio.simulation.external_evidence_alignment import build_external_evidence_batch


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


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _float_or_none(value: object) -> float | None:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


@dataclass(frozen=True, slots=True)
class EvidenceRecordView:
    """One attached external-evidence record rendered for inspection."""

    source_id: str
    source_type: str
    headline_or_summary: str
    event_type: str
    severity: str
    timing: str
    aligned_regime_label: str | None
    false_alarm: bool
    credibility: float | None
    location: str | None
    organization: str | None
    notes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DecisionInspection:
    """Compact bounded decision snapshot for one branch."""

    predicted_regime_label: str | None
    request_replan: bool
    selected_subgoal: str | None
    proposed_update_requests: tuple[str, ...]
    final_update_requests: tuple[str, ...]
    proposed_update_strength: str | None
    final_update_strength: str | None


@dataclass(frozen=True, slots=True)
class OptimizerInputInspection:
    """Compact optimizer-facing input snapshot for one branch."""

    demand_outlook: float | None
    leadtime_outlook: float | None
    safety_buffer_scale: float | None


@dataclass(frozen=True, slots=True)
class ExternalEvidenceInspectionRecord:
    """Human-inspectable evidence-to-decision mapping for one period."""

    schedule_name: str
    run_seed: int
    period_index: int
    true_regime_label: str | None
    schedule_total_cost_delta: float | None
    per_period_cost_delta: float | None
    evidence_records: tuple[EvidenceRecordView, ...]
    evidence_source_summary: tuple[str, ...]
    external_evidence_hint: ExternalEvidenceToolHint | None
    internal_decision: DecisionInspection
    external_decision: DecisionInspection
    internal_optimizer_input: OptimizerInputInspection
    external_optimizer_input: OptimizerInputInspection
    evidence_changed_regime: bool
    evidence_changed_replan: bool
    evidence_changed_final_update: bool
    evidence_changed_optimizer_input: bool
    evidence_strength_only_change: bool
    evidence_present_but_should_ignore: bool
    early_evidence_confirmation_gate_applied: bool
    early_evidence_family_change_blocked: bool
    proposed_external_update_requests: tuple[str, ...]
    final_external_update_requests: tuple[str, ...]
    early_evidence_confirmation_gate_reason: str | None


@dataclass(frozen=True, slots=True)
class ExternalEvidenceInspectionSummary:
    """Top-level inspection view for one live external-evidence run directory."""

    run_dir: str
    artifact_use_class: str | None
    benchmark_id: str | None
    inspection_records: tuple[ExternalEvidenceInspectionRecord, ...]


def inspect_external_evidence_run(
    run_dir: str | Path | None = None,
    *,
    results_root: str | Path = DEFAULT_EXTERNAL_EVIDENCE_RESULTS_ROOT,
) -> ExternalEvidenceInspectionSummary:
    """Build a human-inspectable side-by-side evidence view from saved artifacts."""

    resolved_dir = Path(run_dir) if run_dir is not None else latest_external_evidence_batch_dir(results_root)
    aggregate_summary = _load_json(resolved_dir / "aggregate_summary.json")
    run_manifest = _load_json(resolved_dir / "run_manifest.json")
    episode_rows = _load_jsonl(resolved_dir / "episode_summaries.jsonl")
    step_rows = _load_jsonl(resolved_dir / "step_traces.jsonl")
    failure_analysis = analyze_external_evidence_failures(resolved_dir)

    inspection_records = tuple(
        _build_inspection_record(
            comparison=comparison,
            schedule_total_cost_delta=schedule_summary.total_cost_delta,
            episode_rows=episode_rows,
            step_rows=step_rows,
        )
        for schedule_summary in failure_analysis.schedule_summaries
        for comparison in schedule_summary.period_comparisons
        if comparison.external_evidence_present
    )
    benchmark_id = aggregate_summary.get("benchmark_id")
    return ExternalEvidenceInspectionSummary(
        run_dir=str(resolved_dir),
        artifact_use_class=_string_or_none(run_manifest.get("artifact_use_class")),
        benchmark_id=benchmark_id if isinstance(benchmark_id, str) else None,
        inspection_records=inspection_records,
    )


def export_external_evidence_inspection_json(
    inspection: ExternalEvidenceInspectionSummary,
    output_path: str | Path,
) -> Path:
    """Write the compact inspection view to JSON for later notebook use."""

    from dataclasses import asdict

    resolved_path = Path(output_path)
    resolved_path.write_text(
        json.dumps(asdict(inspection), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return resolved_path


def _build_inspection_record(
    *,
    comparison: ExternalEvidencePeriodComparison,
    schedule_total_cost_delta: float | None,
    episode_rows: tuple[dict[str, object], ...],
    step_rows: tuple[dict[str, object], ...],
) -> ExternalEvidenceInspectionRecord:
    regime_schedule = _regime_schedule_for_seed(
        episode_rows=episode_rows,
        schedule_name=comparison.schedule_name,
        run_seed=comparison.run_seed,
    )
    evidence_batch = build_external_evidence_batch(
        schedule_name=comparison.schedule_name,
        regime_schedule=regime_schedule,
        period_index=comparison.period_index,
        run_seed=comparison.run_seed,
    )
    evidence_records = _record_views(evidence_batch)
    evidence_source_summary = tuple(
        f"{record.source_id}:{record.event_type}:{record.severity}"
        for record in evidence_records
    )
    internal_row = _step_row(
        step_rows=step_rows,
        mode_name="llm_orchestrator_internal_only",
        schedule_name=comparison.schedule_name,
        run_seed=comparison.run_seed,
        period_index=comparison.period_index,
    )
    external_row = _step_row(
        step_rows=step_rows,
        mode_name="llm_orchestrator_with_external_evidence",
        schedule_name=comparison.schedule_name,
        run_seed=comparison.run_seed,
        period_index=comparison.period_index,
    )
    return ExternalEvidenceInspectionRecord(
        schedule_name=comparison.schedule_name,
        run_seed=comparison.run_seed,
        period_index=comparison.period_index,
        true_regime_label=comparison.true_regime_label,
        schedule_total_cost_delta=schedule_total_cost_delta,
        per_period_cost_delta=comparison.per_period_cost_delta,
        evidence_records=evidence_records,
        evidence_source_summary=evidence_source_summary,
        external_evidence_hint=comparison.external_evidence_hint,
        internal_decision=DecisionInspection(
            predicted_regime_label=comparison.internal_predicted_regime_label,
            request_replan=comparison.internal_request_replan,
            selected_subgoal=comparison.internal_selected_subgoal,
            proposed_update_requests=comparison.internal_proposed_update_requests,
            final_update_requests=comparison.internal_final_update_requests,
            proposed_update_strength=comparison.internal_proposed_update_strength,
            final_update_strength=comparison.internal_final_update_strength,
        ),
        external_decision=DecisionInspection(
            predicted_regime_label=comparison.external_predicted_regime_label,
            request_replan=comparison.external_request_replan,
            selected_subgoal=comparison.external_selected_subgoal,
            proposed_update_requests=comparison.external_proposed_update_requests,
            final_update_requests=comparison.external_final_update_requests,
            proposed_update_strength=comparison.external_proposed_update_strength,
            final_update_strength=comparison.external_final_update_strength,
        ),
        internal_optimizer_input=OptimizerInputInspection(
            demand_outlook=comparison.internal_demand_outlook,
            leadtime_outlook=(
                _float_or_none(internal_row.get("leadtime_outlook"))
                if internal_row is not None
                else None
            ),
            safety_buffer_scale=comparison.internal_safety_buffer_scale,
        ),
        external_optimizer_input=OptimizerInputInspection(
            demand_outlook=comparison.external_demand_outlook,
            leadtime_outlook=(
                _float_or_none(external_row.get("leadtime_outlook"))
                if external_row is not None
                else None
            ),
            safety_buffer_scale=comparison.external_safety_buffer_scale,
        ),
        evidence_changed_regime=comparison.regime_changed,
        evidence_changed_replan=comparison.replan_changed,
        evidence_changed_final_update=comparison.final_update_changed,
        evidence_changed_optimizer_input=(
            comparison.external_evidence_changed_optimizer_input is True
        ),
        evidence_strength_only_change=comparison.external_evidence_changed_optimizer_input
        is True
        and not comparison.regime_changed
        and not comparison.replan_changed
        and not comparison.final_update_changed,
        evidence_present_but_should_ignore=comparison.evidence_present_but_should_ignore,
        early_evidence_confirmation_gate_applied=bool(
            external_row is not None
            and external_row.get("early_evidence_confirmation_gate_applied")
        ),
        early_evidence_family_change_blocked=bool(
            external_row is not None
            and external_row.get("early_evidence_family_change_blocked")
        ),
        proposed_external_update_requests=(
            tuple(
                str(value)
                for value in external_row.get("proposed_external_update_requests", ())
                if isinstance(value, str)
            )
            if external_row is not None
            else ()
        ),
        final_external_update_requests=(
            tuple(
                str(value)
                for value in external_row.get("final_external_update_requests", ())
                if isinstance(value, str)
            )
            if external_row is not None
            else ()
        ),
        early_evidence_confirmation_gate_reason=(
            _string_or_none(
                external_row.get("early_evidence_confirmation_gate_reason")
            )
            if external_row is not None
            else None
        ),
    )


def _regime_schedule_for_seed(
    *,
    episode_rows: tuple[dict[str, object], ...],
    schedule_name: str,
    run_seed: int,
) -> tuple[RegimeLabel, ...]:
    for row in episode_rows:
        if (
            row.get("schedule_name") == schedule_name
            and row.get("run_seed") == run_seed
            and row.get("mode") == "llm_orchestrator_with_external_evidence"
        ):
            raw_schedule = row.get("regime_schedule")
            if not isinstance(raw_schedule, list):
                raise ValueError("regime_schedule must be a list when present.")
            return tuple(RegimeLabel(str(value)) for value in raw_schedule)
    raise ValueError(
        "Missing saved regime_schedule for external-evidence inspection: "
        f"{schedule_name} seed {run_seed}."
    )


def _record_views(evidence_batch: ExternalEvidenceBatch | None) -> tuple[EvidenceRecordView, ...]:
    if evidence_batch is None:
        return ()
    return tuple(
        EvidenceRecordView(
            source_id=record.source_id,
            source_type=record.source_type.value,
            headline_or_summary=record.headline_or_summary,
            event_type=record.event_type,
            severity=record.severity.value,
            timing=record.timing.value,
            aligned_regime_label=(
                record.aligned_regime_label.value
                if record.aligned_regime_label is not None
                else None
            ),
            false_alarm=record.false_alarm,
            credibility=record.credibility,
            location=record.location,
            organization=record.organization,
            notes=tuple(record.notes),
        )
        for record in evidence_batch.records
    )


def _step_row(
    *,
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


__all__ = [
    "DecisionInspection",
    "EvidenceRecordView",
    "ExternalEvidenceInspectionRecord",
    "ExternalEvidenceInspectionSummary",
    "OptimizerInputInspection",
    "export_external_evidence_inspection_json",
    "inspect_external_evidence_run",
]
