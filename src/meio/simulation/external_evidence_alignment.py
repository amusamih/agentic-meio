"""Alignment of external evidence onto fixed Stockpyl schedules."""

from __future__ import annotations

from meio.contracts import RegimeLabel
from meio.data.external_evidence import (
    ExternalEvidenceBatch,
    ExternalEvidenceRecord,
    ExternalEvidenceSeverity,
    ExternalEvidenceTiming,
    ExternalEvidenceType,
)


def _record(
    *,
    source_id: str,
    source_type: ExternalEvidenceType,
    headline_or_summary: str,
    event_type: str,
    severity: ExternalEvidenceSeverity,
    timing: ExternalEvidenceTiming,
    aligned_regime_label: RegimeLabel | None,
    false_alarm: bool = False,
    location: str | None = None,
    organization: str | None = None,
    credibility: float | None = None,
    notes: tuple[str, ...] = (),
) -> ExternalEvidenceRecord:
    return ExternalEvidenceRecord(
        source_id=source_id,
        source_type=source_type,
        headline_or_summary=headline_or_summary,
        event_type=event_type,
        location=location,
        organization=organization,
        severity=severity,
        timing=timing,
        credibility=credibility,
        aligned_regime_label=aligned_regime_label,
        false_alarm=false_alarm,
        notes=notes,
    )


def build_external_evidence_batch(
    *,
    schedule_name: str,
    regime_schedule: tuple[RegimeLabel, ...],
    period_index: int,
    run_seed: int,
) -> ExternalEvidenceBatch | None:
    """Return the reproducible external evidence batch for one schedule-period pair."""

    if period_index < 0:
        raise ValueError("period_index must be non-negative.")
    if period_index >= len(regime_schedule):
        raise ValueError("period_index must be within the regime schedule horizon.")
    records = _schedule_records(
        schedule_name=schedule_name,
        regime_schedule=regime_schedule,
        period_index=period_index,
        run_seed=run_seed,
    )
    if not records:
        return None
    return ExternalEvidenceBatch(
        period_index=period_index,
        records=records,
        notes=("semi_synthetic_external_evidence", schedule_name),
    )


def summarize_external_evidence_batch(
    evidence_batch: ExternalEvidenceBatch | None,
) -> tuple[str, ...]:
    """Return compact prompt-safe summaries for an attached evidence batch."""

    if evidence_batch is None:
        return ()
    return tuple(
        (
            f"{record.source_type.value}:{record.event_type}:"
            f"{record.severity.value}:{record.timing.value}:"
            f"{'false_alarm' if record.false_alarm else 'relevant'}:"
            f"{record.headline_or_summary}"
        )
        for record in evidence_batch.records
    )


def _schedule_records(
    *,
    schedule_name: str,
    regime_schedule: tuple[RegimeLabel, ...],
    period_index: int,
    run_seed: int,
) -> tuple[ExternalEvidenceRecord, ...]:
    current_label = regime_schedule[period_index]
    previous_label = regime_schedule[period_index - 1] if period_index > 0 else None
    next_label = regime_schedule[period_index + 1] if period_index + 1 < len(regime_schedule) else None
    seed_suffix = str(run_seed)[-3:]
    records: list[ExternalEvidenceRecord] = []

    if schedule_name in {"shift_recovery", "sustained_shift", "long_shift_recovery"}:
        if current_label is RegimeLabel.NORMAL and next_label is RegimeLabel.DEMAND_REGIME_SHIFT:
            records.append(
                _record(
                    source_id=f"early-demand-{seed_suffix}-{period_index}",
                    source_type=ExternalEvidenceType.NEWS,
                    headline_or_summary="Supplier demand bulletin signals abrupt near-term order surge.",
                    event_type="demand_uplift_signal",
                    severity=ExternalEvidenceSeverity.MEDIUM,
                    timing=ExternalEvidenceTiming.LEADING,
                    aligned_regime_label=RegimeLabel.DEMAND_REGIME_SHIFT,
                    credibility=0.74,
                    notes=("early_relevant_evidence",),
                )
            )
    if schedule_name in {"delayed_shift_recovery", "delayed_sustained_shift"}:
        if current_label is RegimeLabel.DEMAND_REGIME_SHIFT and previous_label is RegimeLabel.NORMAL:
            records.append(
                _record(
                    source_id=f"delayed-demand-{seed_suffix}-{period_index}",
                    source_type=ExternalEvidenceType.SOCIAL,
                    headline_or_summary="Market chatter confirms demand pressure only after internal jump appears.",
                    event_type="delayed_demand_confirmation",
                    severity=ExternalEvidenceSeverity.MEDIUM,
                    timing=ExternalEvidenceTiming.SAME_PERIOD,
                    aligned_regime_label=RegimeLabel.DEMAND_REGIME_SHIFT,
                    credibility=0.62,
                    notes=("delayed_evidence", "noisy_but_relevant"),
                )
            )
    if schedule_name in {"recovery_false_alarm", "false_alarm_then_real_shift", "false_alarm_then_shift_recovery"}:
        if period_index == 0:
            records.append(
                _record(
                    source_id=f"false-alarm-{seed_suffix}-{period_index}",
                    source_type=ExternalEvidenceType.NEWS,
                    headline_or_summary="Strike rumor circulates, but carrier denies material disruption.",
                    event_type="rumored_disruption",
                    severity=ExternalEvidenceSeverity.LOW,
                    timing=ExternalEvidenceTiming.LEADING,
                    aligned_regime_label=RegimeLabel.SUPPLY_DISRUPTION,
                    false_alarm=True,
                    credibility=0.41,
                    notes=("false_alarm_evidence",),
                )
            )
        if current_label is RegimeLabel.NORMAL and next_label is RegimeLabel.DEMAND_REGIME_SHIFT:
            records.append(
                _record(
                    source_id=f"real-shift-{seed_suffix}-{period_index}",
                    source_type=ExternalEvidenceType.INTERNAL_MEMO,
                    headline_or_summary="Key account planning memo now confirms real downstream pull-forward risk.",
                    event_type="customer_demand_signal",
                    severity=ExternalEvidenceSeverity.MEDIUM,
                    timing=ExternalEvidenceTiming.LEADING,
                    aligned_regime_label=RegimeLabel.DEMAND_REGIME_SHIFT,
                    credibility=0.78,
                    notes=("relevant_after_false_alarm",),
                )
            )
    if schedule_name in {"double_shift_with_gap", "recovery_then_relapse"}:
        if (
            current_label is RegimeLabel.DEMAND_REGIME_SHIFT
            and previous_label is not RegimeLabel.DEMAND_REGIME_SHIFT
            and any(label is RegimeLabel.DEMAND_REGIME_SHIFT for label in regime_schedule[:period_index])
        ):
            records.append(
                _record(
                    source_id=f"relapse-{seed_suffix}-{period_index}",
                    source_type=ExternalEvidenceType.SUPPLIER_ALERT,
                    headline_or_summary="Channel reports renewed demand stress before the prior surge fully clears.",
                    event_type="relapse_signal",
                    severity=ExternalEvidenceSeverity.HIGH,
                    timing=ExternalEvidenceTiming.SAME_PERIOD,
                    aligned_regime_label=RegimeLabel.DEMAND_REGIME_SHIFT,
                    credibility=0.81,
                    notes=("relapse_evidence", "unresolved_load_relevant"),
                )
            )
    if schedule_name in {"long_shift_recovery", "delayed_long_shift_recovery"}:
        if current_label is RegimeLabel.RECOVERY and previous_label is RegimeLabel.DEMAND_REGIME_SHIFT:
            records.append(
                _record(
                    source_id=f"recovery-{seed_suffix}-{period_index}",
                    source_type=ExternalEvidenceType.PORT_SIGNAL,
                    headline_or_summary="Inbound flow normalizes, but congestion clearance is gradual rather than immediate.",
                    event_type="recovery_but_loaded_signal",
                    severity=ExternalEvidenceSeverity.MEDIUM,
                    timing=ExternalEvidenceTiming.SAME_PERIOD,
                    aligned_regime_label=RegimeLabel.RECOVERY,
                    credibility=0.76,
                    notes=("recovery_with_load",),
                )
            )
    return tuple(records)


__all__ = [
    "build_external_evidence_batch",
    "summarize_external_evidence_batch",
]
