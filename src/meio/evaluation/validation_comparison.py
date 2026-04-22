"""Compact summaries across the repo's validation lanes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


def _non_empty(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")


def _latest_run_dir(root: Path) -> Path | None:
    if not root.exists():
        return None
    candidates = tuple(
        path for path in root.iterdir() if path.is_dir() and (path / "run_manifest.json").exists()
    )
    if not candidates:
        return None
    preferred = tuple(path for path in candidates if "_all_" in path.name)
    selection = preferred if preferred else candidates
    return sorted(selection)[-1]


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _optional_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return _load_json(path)


def _payload_str(payload: dict[str, object] | None, key: str) -> str | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string when present.")
    return value


def _payload_float(payload: dict[str, object] | None, key: str) -> float | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric when present.")
    return float(value)


def _payload_int(payload: dict[str, object] | None, key: str) -> int | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer when present.")
    return value


def _infer_validation_lane(
    *,
    metadata_payload: dict[str, object] | None,
    benchmark_source: str | None,
) -> str:
    explicit_lane = _payload_str(metadata_payload, "validation_lane")
    if explicit_lane is not None:
        return explicit_lane
    if benchmark_source == "stockpyl_serial":
        return "stockpyl_internal"
    if benchmark_source in {"public_benchmark", "real_demand_backtest", "official_event_replay"}:
        return benchmark_source
    return "unknown"


def default_validation_run_dirs(results_root: str | Path = "results") -> tuple[Path, ...]:
    """Return the latest saved run directory for each current validation lane."""

    root = Path(results_root)
    candidates = (
        root / "stockpyl_serial_paper_candidate",
        root / "stockpyl_serial_heldout_eval",
        root / "stockpyl_serial_frozen_broad_eval",
        root / "public_benchmark_eval",
        root / "real_demand_backtest",
    )
    run_dirs = tuple(
        run_dir
        for candidate_root in candidates
        if (run_dir := _latest_run_dir(candidate_root)) is not None
    )
    return run_dirs


@dataclass(frozen=True, slots=True)
class ValidationModeSummary:
    """Compact summary for one mode within one validation artifact."""

    mode: str
    average_total_cost: float | None
    average_total_reward: float | None
    average_fill_rate: float | None
    regime_prediction_accuracy: float | None
    invalid_output_count: int | None
    fallback_count: int | None

    def __post_init__(self) -> None:
        _non_empty(self.mode, "mode")


@dataclass(frozen=True, slots=True)
class ValidationArtifactSummary:
    """Compact summary for one saved validation artifact directory."""

    results_dir: str
    lane: str
    benchmark_source: str | None
    experiment_id: str | None
    artifact_use_class: str | None
    validity_gate_passed: bool | None
    status: str
    validated_scope: str
    still_synthetic: bool
    mode_summaries: tuple[ValidationModeSummary, ...] = ()
    eligibility_notes: tuple[str, ...] = ()
    public_benchmark_blocked_reason: str | None = None
    bounded_orchestrator_integration_supported: bool | None = None

    def __post_init__(self) -> None:
        _non_empty(self.results_dir, "results_dir")
        _non_empty(self.lane, "lane")
        _non_empty(self.status, "status")
        _non_empty(self.validated_scope, "validated_scope")


@dataclass(frozen=True, slots=True)
class ValidationLaneCoverage:
    """Aggregate lane-level view across one or more saved artifacts."""

    lane: str
    artifact_count: int
    completed_artifact_count: int
    latest_status: str
    still_synthetic: bool
    validated_scope: str

    def __post_init__(self) -> None:
        _non_empty(self.lane, "lane")
        _non_empty(self.latest_status, "latest_status")
        _non_empty(self.validated_scope, "validated_scope")


@dataclass(frozen=True, slots=True)
class ValidationStackSummary:
    """Top-level cross-lane validation summary."""

    artifact_summaries: tuple[ValidationArtifactSummary, ...]
    lane_coverage: tuple[ValidationLaneCoverage, ...]


def summarize_validation_stack(
    run_dirs: tuple[str | Path, ...],
) -> ValidationStackSummary:
    """Summarize the available validation-lane artifacts."""

    artifact_summaries = tuple(_summarize_run_dir(Path(path)) for path in run_dirs)
    lane_coverage = tuple(
        _summarize_lane(artifact_summaries, lane)
        for lane in sorted({summary.lane for summary in artifact_summaries})
    )
    return ValidationStackSummary(
        artifact_summaries=artifact_summaries,
        lane_coverage=lane_coverage,
    )


def _summarize_run_dir(run_dir: Path) -> ValidationArtifactSummary:
    metadata_payload = _optional_json(run_dir / "experiment_metadata.json")
    manifest_payload = _optional_json(run_dir / "run_manifest.json")
    aggregate_payload = _optional_json(run_dir / "aggregate_summary.json")
    public_benchmark_payload = _optional_json(run_dir / "public_benchmark_summary.json")
    benchmark_source = (
        _payload_str(metadata_payload, "benchmark_source")
        or _payload_str(manifest_payload, "benchmark_source")
    )
    lane = _infer_validation_lane(
        metadata_payload=metadata_payload,
        benchmark_source=benchmark_source,
    )
    if public_benchmark_payload is not None:
        integration_supported = bool(
            public_benchmark_payload.get("bounded_orchestrator_integration_supported", False)
        )
        status = "completed" if integration_supported or aggregate_payload is not None else "blocked_partial"
    elif aggregate_payload is not None:
        status = "completed"
    else:
        status = "metadata_only"
    mode_summaries = _mode_summaries_from_aggregate(
        aggregate_payload,
        reward_by_mode=_mode_reward_map_from_public_summary(public_benchmark_payload),
    )
    return ValidationArtifactSummary(
        results_dir=str(run_dir),
        lane=lane,
        benchmark_source=benchmark_source,
        experiment_id=(
            _payload_str(metadata_payload, "experiment_id")
            or _payload_str(manifest_payload, "experiment_id")
        ),
        artifact_use_class=(
            _payload_str(metadata_payload, "artifact_use_class")
            or _payload_str(aggregate_payload, "artifact_use_class")
            or _payload_str(manifest_payload, "artifact_use_class")
        ),
        validity_gate_passed=(
            bool(metadata_payload.get("validity_gate_passed"))
            if isinstance(metadata_payload, dict) and "validity_gate_passed" in metadata_payload
            else (
                bool(aggregate_payload.get("validity_gate_passed"))
                if isinstance(aggregate_payload, dict) and "validity_gate_passed" in aggregate_payload
                else None
            )
        ),
        status=status,
        validated_scope=_validated_scope_for_lane(lane),
        still_synthetic=_still_synthetic_for_lane(lane),
        mode_summaries=mode_summaries,
        eligibility_notes=_resolve_eligibility_notes(
            metadata_payload=metadata_payload,
            aggregate_payload=aggregate_payload,
            manifest_payload=manifest_payload,
        ),
        public_benchmark_blocked_reason=_payload_str(public_benchmark_payload, "blocked_reason"),
        bounded_orchestrator_integration_supported=(
            bool(public_benchmark_payload.get("bounded_orchestrator_integration_supported"))
            if isinstance(public_benchmark_payload, dict)
            else None
        ),
    )


def _mode_summaries_from_aggregate(
    aggregate_payload: dict[str, object] | None,
    *,
    reward_by_mode: dict[str, float] | None = None,
) -> tuple[ValidationModeSummary, ...]:
    if not isinstance(aggregate_payload, dict):
        return ()
    raw_mode_summaries = aggregate_payload.get("mode_summaries")
    if not isinstance(raw_mode_summaries, list):
        return ()
    summaries: list[ValidationModeSummary] = []
    for item in raw_mode_summaries:
        if not isinstance(item, dict):
            continue
        performance_summary = item.get("performance_summary")
        decision_quality = item.get("decision_quality")
        validity_summary = item.get("validity_summary")
        mode_name = _payload_str(item, "mode")
        if mode_name is None:
            continue
        summaries.append(
            ValidationModeSummary(
                mode=mode_name,
                average_total_cost=_payload_float(performance_summary, "average_total_cost"),
                average_total_reward=(
                    reward_by_mode.get(mode_name)
                    if reward_by_mode is not None
                    else None
                ),
                average_fill_rate=_payload_float(performance_summary, "average_fill_rate"),
                regime_prediction_accuracy=_payload_float(
                    decision_quality,
                    "regime_prediction_accuracy",
                ),
                invalid_output_count=_payload_int(validity_summary, "invalid_output_count"),
                fallback_count=_payload_int(validity_summary, "fallback_count"),
            )
    )
    return tuple(summaries)


def _mode_reward_map_from_public_summary(
    public_benchmark_payload: dict[str, object] | None,
) -> dict[str, float]:
    if not isinstance(public_benchmark_payload, dict):
        return {}
    raw_mode_summaries = public_benchmark_payload.get("mode_summaries")
    if not isinstance(raw_mode_summaries, list):
        return {}
    reward_by_mode: dict[str, float] = {}
    for item in raw_mode_summaries:
        if not isinstance(item, dict):
            continue
        mode_name = _payload_str(item, "mode")
        total_reward = _payload_float(item, "total_reward")
        if mode_name is None or total_reward is None:
            continue
        reward_by_mode[mode_name] = total_reward
    return reward_by_mode


def _validated_scope_for_lane(lane: str) -> str:
    if lane == "stockpyl_internal":
        return "bounded orchestrator behavior on the internal Stockpyl serial simulator"
    if lane == "public_benchmark":
        return "portability to a recognized public benchmark package and its environment assumptions"
    if lane == "real_demand_backtest":
        return "bounded inventory control against public demand and lead-time observations"
    if lane == "official_event_replay":
        return "exogenous-event replay against official event records"
    return "unclassified validation scope"


def _still_synthetic_for_lane(lane: str) -> bool:
    return lane == "stockpyl_internal"


def _summarize_lane(
    artifact_summaries: tuple[ValidationArtifactSummary, ...],
    lane: str,
) -> ValidationLaneCoverage:
    selected = tuple(summary for summary in artifact_summaries if summary.lane == lane)
    latest = selected[-1]
    return ValidationLaneCoverage(
        lane=lane,
        artifact_count=len(selected),
        completed_artifact_count=sum(1 for summary in selected if summary.status == "completed"),
        latest_status=latest.status,
        still_synthetic=latest.still_synthetic,
        validated_scope=latest.validated_scope,
    )


def _resolve_eligibility_notes(
    *,
    metadata_payload: dict[str, object] | None,
    aggregate_payload: dict[str, object] | None,
    manifest_payload: dict[str, object] | None,
) -> tuple[str, ...]:
    for payload in (metadata_payload, aggregate_payload, manifest_payload):
        if not isinstance(payload, dict):
            continue
        value = payload.get("eligibility_notes")
        if value is None:
            continue
        if not isinstance(value, list):
            raise ValueError("eligibility_notes must be a list when present.")
        return tuple(str(item) for item in value)
    return ()


__all__ = [
    "ValidationArtifactSummary",
    "ValidationLaneCoverage",
    "ValidationModeSummary",
    "ValidationStackSummary",
    "default_validation_run_dirs",
    "summarize_validation_stack",
]
