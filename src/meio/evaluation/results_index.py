"""Helpers for artifact-governance classification and saved-run indexing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from meio.evaluation.logging_schema import ArtifactUseClass

_INTERNAL_ONLY_EXPERIMENT_MARKERS = (
    "smoke",
    "first_milestone",
    "qualification",
    "schema",
    "debug",
)


def _dedupe_preserving_order(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class ArtifactGovernanceDecision:
    """Explicit governance classification for one saved artifact scope."""

    artifact_use_class: ArtifactUseClass
    validity_gate_passed: bool
    eligibility_notes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class IndexedRunRecord:
    """Compact indexed view over one saved mode-level result set."""

    results_dir: str
    run_group_id: str
    experiment_id: str | None
    benchmark_id: str | None
    benchmark_source: str | None
    validation_lane: str | None
    mode: str
    provider: str | None
    model_name: str | None
    external_evidence_source: str | None
    artifact_use_class: ArtifactUseClass
    validity_gate_passed: bool
    eligibility_notes: tuple[str, ...]
    average_total_cost: float | None = None
    average_fill_rate: float | None = None


def compute_validity_gate_passed(
    *,
    optimizer_order_boundary_preserved: bool,
    invalid_output_count: int,
    fallback_count: int,
) -> bool:
    """Return the current conservative validity-gate pass flag."""

    return (
        optimizer_order_boundary_preserved
        and invalid_output_count == 0
        and fallback_count == 0
    )


def classify_artifact_governance(
    *,
    experiment_name: str,
    benchmark_source: str,
    provider: str | None,
    validity_gate_passed: bool,
    rollout_fidelity_gate_passed: bool = False,
    operational_metrics_gate_passed: bool = False,
    semi_synthetic_external_evidence: bool = False,
    external_evidence_source: str | None = None,
) -> ArtifactGovernanceDecision:
    """Classify one mode-level or run-level artifact scope."""

    notes: list[str] = []
    lowered_name = experiment_name.lower()
    if any(marker in lowered_name for marker in _INTERNAL_ONLY_EXPERIMENT_MARKERS):
        notes.append("schema_or_smoke_validation_run")
    normalized_external_evidence_source = external_evidence_source
    if normalized_external_evidence_source is None and semi_synthetic_external_evidence:
        normalized_external_evidence_source = "semi_synthetic"
    if provider == "fake_llm_client":
        notes.append("fake_llm_internal_only")
    if normalized_external_evidence_source == "semi_synthetic":
        notes.append("semi_synthetic_external_evidence_branch")
    if not validity_gate_passed:
        notes.append("validity_gates_failed")
    if benchmark_source == "stockpyl_serial":
        if not rollout_fidelity_gate_passed:
            notes.append("provisional_stockpyl_rollout")
        if not operational_metrics_gate_passed:
            notes.append("provisional_operational_metrics")
    elif benchmark_source == "public_benchmark":
        notes.append("public_benchmark_partial_integration")
    elif benchmark_source == "real_demand_backtest":
        notes.append("real_demand_backtest_provisional")
    elif benchmark_source == "official_event_replay":
        notes.append("official_event_replay_provisional")
    deduped_notes = _dedupe_preserving_order(notes)
    use_class = (
        ArtifactUseClass.PAPER_CANDIDATE
        if not deduped_notes
        else ArtifactUseClass.INTERNAL_ONLY
    )
    return ArtifactGovernanceDecision(
        artifact_use_class=use_class,
        validity_gate_passed=validity_gate_passed,
        eligibility_notes=deduped_notes,
    )


def summarize_directory_governance(
    mode_decisions: tuple[ArtifactGovernanceDecision, ...],
) -> ArtifactGovernanceDecision:
    """Collapse per-mode governance into one directory-level classification."""

    if not mode_decisions:
        raise ValueError("mode_decisions must not be empty.")
    all_paper_candidate = all(
        decision.artifact_use_class is ArtifactUseClass.PAPER_CANDIDATE
        for decision in mode_decisions
    )
    notes = list(
        note
        for decision in mode_decisions
        for note in decision.eligibility_notes
    )
    if len(mode_decisions) > 1 and not all_paper_candidate:
        notes.append("mixed_mode_directory_contains_internal_only_results")
    return ArtifactGovernanceDecision(
        artifact_use_class=(
            ArtifactUseClass.PAPER_CANDIDATE
            if all_paper_candidate
            else ArtifactUseClass.INTERNAL_ONLY
        ),
        validity_gate_passed=all(
            decision.validity_gate_passed for decision in mode_decisions
        ),
        eligibility_notes=_dedupe_preserving_order(notes),
    )


def index_result_runs(results_root: str | Path = "results") -> tuple[IndexedRunRecord, ...]:
    """Scan saved result directories and return compact mode-level records."""

    root = Path(results_root)
    if not root.exists():
        return ()
    records: list[IndexedRunRecord] = []
    for manifest_path in sorted(root.rglob("run_manifest.json")):
        run_dir = manifest_path.parent
        manifest_payload = _load_json(manifest_path)
        metadata_payload = _load_optional_json(run_dir / "experiment_metadata.json")
        aggregate_payload = _load_optional_json(run_dir / "aggregate_summary.json")
        provider = _payload_string(metadata_payload, "provider")
        model_name = _payload_string(metadata_payload, "model_name")
        external_evidence_source = _payload_string(
            metadata_payload,
            "external_evidence_source",
        )
        benchmark_source = _payload_string(manifest_payload, "benchmark_source")
        validation_lane = _resolve_validation_lane(
            metadata_payload=metadata_payload,
            benchmark_source=benchmark_source,
        )
        run_group_id = _payload_string(manifest_payload, "run_group_id") or run_dir.name
        experiment_id = _payload_string(manifest_payload, "experiment_id")
        benchmark_id = _payload_string(manifest_payload, "benchmark_id")
        if aggregate_payload is not None and isinstance(aggregate_payload.get("mode_summaries"), list):
            for mode_summary in aggregate_payload["mode_summaries"]:
                if not isinstance(mode_summary, dict):
                    continue
                mode_name = _payload_string(mode_summary, "mode") or "unknown"
                validity_gate_passed = _mode_validity_gate_passed(mode_summary)
                governance = classify_artifact_governance(
                    experiment_name=experiment_id or run_group_id,
                    benchmark_source=benchmark_source or "",
                    provider=provider,
                    validity_gate_passed=validity_gate_passed,
                    rollout_fidelity_gate_passed=_mode_rollout_fidelity_passed(mode_summary),
                    operational_metrics_gate_passed=_mode_operational_metrics_passed(
                        mode_summary
                    ),
                    external_evidence_source=(
                        _nested_payload_string(
                            mode_summary.get("external_evidence_summary"),
                            "external_evidence_source",
                        )
                        or external_evidence_source
                    ),
                )
                performance_summary = mode_summary.get("performance_summary")
                records.append(
                    IndexedRunRecord(
                        results_dir=str(run_dir),
                        run_group_id=run_group_id,
                        experiment_id=experiment_id,
                        benchmark_id=benchmark_id,
                        benchmark_source=benchmark_source,
                        validation_lane=validation_lane,
                        mode=mode_name,
                        provider=provider,
                        model_name=model_name,
                        external_evidence_source=(
                            _nested_payload_string(
                                mode_summary.get("external_evidence_summary"),
                                "external_evidence_source",
                            )
                            or external_evidence_source
                        ),
                        artifact_use_class=ArtifactUseClass(
                            _payload_string(mode_summary, "artifact_use_class")
                            or governance.artifact_use_class.value
                        ),
                        validity_gate_passed=bool(
                            mode_summary.get("validity_gate_passed", validity_gate_passed)
                        ),
                        eligibility_notes=tuple(
                            mode_summary.get("eligibility_notes", governance.eligibility_notes)
                        ),
                        average_total_cost=_nested_payload_float(
                            performance_summary,
                            "average_total_cost",
                        ),
                        average_fill_rate=_nested_payload_float(
                            performance_summary,
                            "average_fill_rate",
                        ),
                    )
                )
            continue
        legacy_validity_gate_passed = bool(
            metadata_payload.get("validity_gate_passed", False)
            if isinstance(metadata_payload, dict)
            else False
        )
        governance = classify_artifact_governance(
            experiment_name=experiment_id or run_group_id,
            benchmark_source=benchmark_source or "",
            provider=provider,
            validity_gate_passed=legacy_validity_gate_passed,
            rollout_fidelity_gate_passed=bool(
                metadata_payload.get("rollout_fidelity_gate_passed", False)
                if isinstance(metadata_payload, dict)
                else False
            ),
            operational_metrics_gate_passed=bool(
                metadata_payload.get("operational_metrics_gate_passed", False)
                if isinstance(metadata_payload, dict)
                else False
            ),
            external_evidence_source=external_evidence_source,
        )
        records.append(
            IndexedRunRecord(
                results_dir=str(run_dir),
                run_group_id=run_group_id,
                experiment_id=experiment_id,
                benchmark_id=benchmark_id,
                benchmark_source=benchmark_source,
                validation_lane=validation_lane,
                mode=_payload_string(metadata_payload, "mode") or "unknown",
                provider=provider,
                model_name=model_name,
                external_evidence_source=external_evidence_source,
                artifact_use_class=ArtifactUseClass(
                    _payload_string(metadata_payload, "artifact_use_class")
                    or governance.artifact_use_class.value
                ),
                validity_gate_passed=legacy_validity_gate_passed,
                eligibility_notes=tuple(
                    metadata_payload.get("eligibility_notes", governance.eligibility_notes)
                    if isinstance(metadata_payload, dict)
                    else governance.eligibility_notes
                ),
            )
        )
    return tuple(records)


def _resolve_validation_lane(
    *,
    metadata_payload: dict[str, object] | None,
    benchmark_source: str | None,
) -> str | None:
    explicit_lane = _payload_string(metadata_payload, "validation_lane")
    if explicit_lane is not None:
        return explicit_lane
    if benchmark_source == "stockpyl_serial":
        return "stockpyl_internal"
    if benchmark_source in {
        "public_benchmark",
        "real_demand_backtest",
        "official_event_replay",
    }:
        return benchmark_source
    return None


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _load_optional_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return _load_json(path)


def _payload_string(payload: dict[str, object] | None, key: str) -> str | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string when present.")
    return value


def _mode_validity_gate_passed(mode_summary: dict[str, object]) -> bool:
    if "validity_gate_passed" in mode_summary:
        return bool(mode_summary["validity_gate_passed"])
    validity_summary = mode_summary.get("validity_summary")
    if not isinstance(validity_summary, dict):
        return False
    optimizer_boundary = bool(validity_summary.get("optimizer_order_boundary_preserved", False))
    invalid_output_count = _int_from_payload(validity_summary, "invalid_output_count")
    fallback_count = _int_from_payload(validity_summary, "fallback_count")
    return compute_validity_gate_passed(
        optimizer_order_boundary_preserved=optimizer_boundary,
        invalid_output_count=invalid_output_count,
        fallback_count=fallback_count,
    )


def _mode_rollout_fidelity_passed(mode_summary: dict[str, object]) -> bool:
    validity_summary = mode_summary.get("validity_summary")
    if not isinstance(validity_summary, dict):
        return False
    return bool(validity_summary.get("rollout_fidelity_gate_passed", False))


def _mode_operational_metrics_passed(mode_summary: dict[str, object]) -> bool:
    validity_summary = mode_summary.get("validity_summary")
    if not isinstance(validity_summary, dict):
        return False
    return bool(validity_summary.get("operational_metrics_gate_passed", False))


def _nested_payload_float(payload: object, key: str) -> float | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric when present.")
    return float(value)


def _nested_payload_string(payload: object, key: str) -> str | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _int_from_payload(payload: dict[str, object], key: str) -> int:
    value = payload.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer when present.")
    return value


__all__ = [
    "ArtifactGovernanceDecision",
    "IndexedRunRecord",
    "classify_artifact_governance",
    "compute_validity_gate_passed",
    "index_result_runs",
    "summarize_directory_governance",
]
