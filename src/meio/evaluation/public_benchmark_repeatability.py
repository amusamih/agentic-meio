"""Repeatability analysis for repeated public-benchmark runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

from meio.evaluation.logging_io import hash_jsonable

DEFAULT_PUBLIC_BENCHMARK_RESULTS_ROOT = Path("results/public_benchmark_eval")
_NEUTRAL_TOLERANCE = 1e-9
_LLM_MODE = "llm_orchestrator"
_DETERMINISTIC_ORCHESTRATOR_MODE = "deterministic_orchestrator"
_DETERMINISTIC_BASELINE_MODE = "deterministic_baseline"


def _non_empty(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _require_dict(value: object, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a JSON object.")
    return value


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Expected a numeric value when present.")
    return float(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("Expected an integer value when present.")
    return value


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Expected a string value when present.")
    return value


@dataclass(frozen=True, slots=True)
class NumericSpreadSummary:
    """Compact numeric spread summary over repeated runs."""

    mean_value: float | None
    minimum_value: float | None
    maximum_value: float | None
    standard_deviation: float | None


@dataclass(frozen=True, slots=True)
class PublicBenchmarkRepeatabilityModeRunSummary:
    """Per-mode summary extracted from one saved public-benchmark run."""

    mode: str
    total_reward: float | None
    average_fill_rate: float | None
    fallback_count: int
    invalid_output_count: int
    tool_call_count: int | None
    replan_rate: float | None
    no_action_rate: float | None
    llm_provider: str | None = None
    llm_model_name: str | None = None

    def __post_init__(self) -> None:
        _non_empty(self.mode, "mode")


@dataclass(frozen=True, slots=True)
class PublicBenchmarkRepeatabilityRunSummary:
    """Per-run public-benchmark summary for repeatability analysis."""

    run_dir: str
    config_hash: str | None
    mapping_identity: str
    artifact_use_class: str | None
    validity_gate_passed: bool | None
    mapping_assumptions: tuple[str, ...]
    comparability_notes: tuple[str, ...]
    mode_summaries: tuple[PublicBenchmarkRepeatabilityModeRunSummary, ...]

    def __post_init__(self) -> None:
        _non_empty(self.run_dir, "run_dir")
        _non_empty(self.mapping_identity, "mapping_identity")


@dataclass(frozen=True, slots=True)
class PublicBenchmarkRepeatabilityModeSummary:
    """Repeated-run summary for one public-benchmark mode."""

    mode: str
    run_count: int
    reward: NumericSpreadSummary
    fill_rate: NumericSpreadSummary
    total_fallback_count: int
    total_invalid_output_count: int
    tool_call_count: NumericSpreadSummary
    replan_rate: NumericSpreadSummary
    no_action_rate: NumericSpreadSummary
    llm_providers: tuple[str, ...]

    def __post_init__(self) -> None:
        _non_empty(self.mode, "mode")


@dataclass(frozen=True, slots=True)
class PublicBenchmarkProviderComparison:
    """Provider-group comparison for the LLM public-benchmark mode."""

    fake_run_count: int
    real_run_count: int
    fake_reward_mean: float | None
    real_reward_mean: float | None
    real_minus_fake_reward_mean: float | None
    fake_fill_rate_mean: float | None
    real_fill_rate_mean: float | None


@dataclass(frozen=True, slots=True)
class PublicBenchmarkRepeatabilityAnalysis:
    """Top-level repeated-run summary for the public benchmark lane."""

    run_dirs: tuple[str, ...]
    run_count: int
    config_hashes: tuple[str, ...]
    mapping_identities: tuple[str, ...]
    run_summaries: tuple[PublicBenchmarkRepeatabilityRunSummary, ...]
    mode_summaries: tuple[PublicBenchmarkRepeatabilityModeSummary, ...]
    llm_vs_deterministic_orchestrator_label: str
    deterministic_orchestrator_vs_baseline_label: str
    llm_validity_clean: bool
    fake_vs_real_llm_comparison: PublicBenchmarkProviderComparison | None


def list_public_benchmark_run_dirs(
    *,
    results_root: str | Path = DEFAULT_PUBLIC_BENCHMARK_RESULTS_ROOT,
    latest_n: int | None = None,
) -> tuple[Path, ...]:
    """List saved public-benchmark run directories ordered by modification time."""

    root = Path(results_root)
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist.")
    candidates = tuple(
        path
        for path in root.iterdir()
        if path.is_dir()
        and (path / "public_benchmark_summary.json").exists()
        and (path / "aggregate_summary.json").exists()
    )
    if not candidates:
        raise FileNotFoundError(f"No saved public-benchmark runs found under {root}.")
    ordered = tuple(sorted(candidates, key=lambda path: path.stat().st_mtime))
    if latest_n is None:
        return ordered
    if latest_n <= 0:
        raise ValueError("latest_n must be positive when provided.")
    return ordered[-latest_n:]


def analyze_public_benchmark_repeatability(
    run_dirs: tuple[str | Path, ...] | None = None,
    *,
    results_root: str | Path = DEFAULT_PUBLIC_BENCHMARK_RESULTS_ROOT,
    latest_n: int | None = None,
) -> PublicBenchmarkRepeatabilityAnalysis:
    """Summarize repeatability across saved public-benchmark runs."""

    resolved_run_dirs = _resolve_run_dirs(
        run_dirs=run_dirs,
        results_root=results_root,
        latest_n=latest_n,
    )
    run_summaries = tuple(_summarize_run_dir(path) for path in resolved_run_dirs)
    mode_summaries = tuple(
        _build_mode_summary(run_summaries, mode_name)
        for mode_name in (
            _DETERMINISTIC_BASELINE_MODE,
            _DETERMINISTIC_ORCHESTRATOR_MODE,
            _LLM_MODE,
        )
        if _build_mode_summary(run_summaries, mode_name) is not None
    )
    llm_deltas = _mode_reward_deltas(
        run_summaries,
        left_mode=_LLM_MODE,
        right_mode=_DETERMINISTIC_ORCHESTRATOR_MODE,
    )
    deterministic_deltas = _mode_reward_deltas(
        run_summaries,
        left_mode=_DETERMINISTIC_ORCHESTRATOR_MODE,
        right_mode=_DETERMINISTIC_BASELINE_MODE,
    )
    return PublicBenchmarkRepeatabilityAnalysis(
        run_dirs=tuple(str(path) for path in resolved_run_dirs),
        run_count=len(run_summaries),
        config_hashes=tuple(
            sorted(
                {
                    summary.config_hash
                    for summary in run_summaries
                    if summary.config_hash is not None
                }
            )
        ),
        mapping_identities=tuple(
            sorted({summary.mapping_identity for summary in run_summaries})
        ),
        run_summaries=run_summaries,
        mode_summaries=mode_summaries,
        llm_vs_deterministic_orchestrator_label=_stability_label(llm_deltas),
        deterministic_orchestrator_vs_baseline_label=_stability_label(
            deterministic_deltas
        ),
        llm_validity_clean=all(
            mode.fallback_count == 0 and mode.invalid_output_count == 0
            for summary in run_summaries
            for mode in summary.mode_summaries
            if mode.mode == _LLM_MODE
        ),
        fake_vs_real_llm_comparison=_build_fake_vs_real_comparison(run_summaries),
    )


def _resolve_run_dirs(
    *,
    run_dirs: tuple[str | Path, ...] | None,
    results_root: str | Path,
    latest_n: int | None,
) -> tuple[Path, ...]:
    if run_dirs:
        return tuple(Path(path) for path in run_dirs)
    return list_public_benchmark_run_dirs(results_root=results_root, latest_n=latest_n)


def _summarize_run_dir(run_dir: Path) -> PublicBenchmarkRepeatabilityRunSummary:
    metadata_payload = _load_json(run_dir / "experiment_metadata.json")
    public_payload = _load_json(run_dir / "public_benchmark_summary.json")
    aggregate_payload = _load_json(run_dir / "aggregate_summary.json")
    mode_summaries_by_mode = _mode_payload_by_name(public_payload.get("mode_summaries"))
    aggregate_summaries_by_mode = _mode_payload_by_name(aggregate_payload.get("mode_summaries"))
    mode_summaries: list[PublicBenchmarkRepeatabilityModeRunSummary] = []
    for mode_name, public_mode_payload in mode_summaries_by_mode.items():
        aggregate_mode_payload = aggregate_summaries_by_mode.get(mode_name, {})
        decision_quality = _require_dict(
            aggregate_mode_payload.get("decision_quality", {}),
            "decision_quality",
        )
        validity_summary = _require_dict(
            aggregate_mode_payload.get("validity_summary", {}),
            "validity_summary",
        )
        mode_summaries.append(
            PublicBenchmarkRepeatabilityModeRunSummary(
                mode=mode_name,
                total_reward=_optional_float(public_mode_payload.get("total_reward")),
                average_fill_rate=_optional_float(
                    public_mode_payload.get("average_fill_rate")
                ),
                fallback_count=_optional_int(validity_summary.get("fallback_count")) or 0,
                invalid_output_count=(
                    _optional_int(validity_summary.get("invalid_output_count")) or 0
                ),
                tool_call_count=_optional_int(public_mode_payload.get("tool_call_count")),
                replan_rate=_optional_float(decision_quality.get("replan_rate")),
                no_action_rate=_optional_float(decision_quality.get("no_action_rate")),
                llm_provider=_optional_str(public_mode_payload.get("llm_provider")),
                llm_model_name=_optional_str(public_mode_payload.get("llm_model_name")),
            )
        )
    mapping_payload = {
        "environment_contract": public_payload.get("environment_contract"),
        "mapping_assumptions": public_payload.get("mapping_assumptions"),
        "comparability_notes": public_payload.get("comparability_notes"),
    }
    return PublicBenchmarkRepeatabilityRunSummary(
        run_dir=str(run_dir),
        config_hash=_optional_str(metadata_payload.get("config_hash")),
        mapping_identity=(
            _optional_str(public_payload.get("mapping_identity"))
            or hash_jsonable(mapping_payload)
        ),
        artifact_use_class=_optional_str(metadata_payload.get("artifact_use_class")),
        validity_gate_passed=(
            bool(metadata_payload.get("validity_gate_passed"))
            if "validity_gate_passed" in metadata_payload
            else None
        ),
        mapping_assumptions=tuple(
            str(item) for item in public_payload.get("mapping_assumptions", [])
        ),
        comparability_notes=tuple(
            str(item) for item in public_payload.get("comparability_notes", [])
        ),
        mode_summaries=tuple(mode_summaries),
    )


def _mode_payload_by_name(raw_value: object) -> dict[str, dict[str, object]]:
    if not isinstance(raw_value, list):
        return {}
    payload_by_name: dict[str, dict[str, object]] = {}
    for item in raw_value:
        if not isinstance(item, dict):
            continue
        mode_name = _optional_str(item.get("mode"))
        if mode_name is None:
            continue
        payload_by_name[mode_name] = item
    return payload_by_name


def _build_mode_summary(
    run_summaries: tuple[PublicBenchmarkRepeatabilityRunSummary, ...],
    mode_name: str,
) -> PublicBenchmarkRepeatabilityModeSummary | None:
    selected = tuple(
        mode
        for run_summary in run_summaries
        for mode in run_summary.mode_summaries
        if mode.mode == mode_name
    )
    if not selected:
        return None
    return PublicBenchmarkRepeatabilityModeSummary(
        mode=mode_name,
        run_count=len(selected),
        reward=_numeric_spread(
            tuple(value for value in (item.total_reward for item in selected) if value is not None)
        ),
        fill_rate=_numeric_spread(
            tuple(
                value
                for value in (item.average_fill_rate for item in selected)
                if value is not None
            )
        ),
        total_fallback_count=sum(item.fallback_count for item in selected),
        total_invalid_output_count=sum(item.invalid_output_count for item in selected),
        tool_call_count=_numeric_spread(
            tuple(
                float(value)
                for value in (item.tool_call_count for item in selected)
                if value is not None
            )
        ),
        replan_rate=_numeric_spread(
            tuple(value for value in (item.replan_rate for item in selected) if value is not None)
        ),
        no_action_rate=_numeric_spread(
            tuple(
                value for value in (item.no_action_rate for item in selected) if value is not None
            )
        ),
        llm_providers=tuple(
            sorted(
                {
                    item.llm_provider
                    for item in selected
                    if item.llm_provider is not None
                }
            )
        ),
    )


def _mode_reward_deltas(
    run_summaries: tuple[PublicBenchmarkRepeatabilityRunSummary, ...],
    *,
    left_mode: str,
    right_mode: str,
) -> tuple[float, ...]:
    deltas: list[float] = []
    for run_summary in run_summaries:
        left = _find_mode(run_summary, left_mode)
        right = _find_mode(run_summary, right_mode)
        if left is None or right is None:
            continue
        if left.total_reward is None or right.total_reward is None:
            continue
        deltas.append(left.total_reward - right.total_reward)
    return tuple(deltas)


def _find_mode(
    run_summary: PublicBenchmarkRepeatabilityRunSummary,
    mode_name: str,
) -> PublicBenchmarkRepeatabilityModeRunSummary | None:
    for mode_summary in run_summary.mode_summaries:
        if mode_summary.mode == mode_name:
            return mode_summary
    return None


def _build_fake_vs_real_comparison(
    run_summaries: tuple[PublicBenchmarkRepeatabilityRunSummary, ...],
) -> PublicBenchmarkProviderComparison | None:
    fake_rewards: list[float] = []
    real_rewards: list[float] = []
    fake_fill_rates: list[float] = []
    real_fill_rates: list[float] = []
    for run_summary in run_summaries:
        llm_summary = _find_mode(run_summary, _LLM_MODE)
        if llm_summary is None or llm_summary.llm_provider is None:
            continue
        if llm_summary.llm_provider == "fake_llm_client":
            if llm_summary.total_reward is not None:
                fake_rewards.append(llm_summary.total_reward)
            if llm_summary.average_fill_rate is not None:
                fake_fill_rates.append(llm_summary.average_fill_rate)
        elif llm_summary.llm_provider == "openai":
            if llm_summary.total_reward is not None:
                real_rewards.append(llm_summary.total_reward)
            if llm_summary.average_fill_rate is not None:
                real_fill_rates.append(llm_summary.average_fill_rate)
    if not fake_rewards and not real_rewards:
        return None
    fake_reward_mean = float(mean(fake_rewards)) if fake_rewards else None
    real_reward_mean = float(mean(real_rewards)) if real_rewards else None
    return PublicBenchmarkProviderComparison(
        fake_run_count=len(fake_rewards),
        real_run_count=len(real_rewards),
        fake_reward_mean=fake_reward_mean,
        real_reward_mean=real_reward_mean,
        real_minus_fake_reward_mean=(
            real_reward_mean - fake_reward_mean
            if real_reward_mean is not None and fake_reward_mean is not None
            else None
        ),
        fake_fill_rate_mean=(float(mean(fake_fill_rates)) if fake_fill_rates else None),
        real_fill_rate_mean=(float(mean(real_fill_rates)) if real_fill_rates else None),
    )


def _numeric_spread(values: tuple[float, ...]) -> NumericSpreadSummary:
    if not values:
        return NumericSpreadSummary(None, None, None, None)
    return NumericSpreadSummary(
        mean_value=float(mean(values)),
        minimum_value=float(min(values)),
        maximum_value=float(max(values)),
        standard_deviation=0.0 if len(values) == 1 else float(pstdev(values)),
    )


def _stability_label(values: tuple[float, ...]) -> str:
    if not values:
        return "unknown"
    if all(abs(value) <= _NEUTRAL_TOLERANCE for value in values):
        return "stably_neutral"
    if all(value >= -_NEUTRAL_TOLERANCE for value in values) and any(
        value > _NEUTRAL_TOLERANCE for value in values
    ):
        return "stably_better"
    if all(value <= _NEUTRAL_TOLERANCE for value in values) and any(
        value < -_NEUTRAL_TOLERANCE for value in values
    ):
        return "stably_worse"
    return "unstable"


__all__ = [
    "DEFAULT_PUBLIC_BENCHMARK_RESULTS_ROOT",
    "NumericSpreadSummary",
    "PublicBenchmarkProviderComparison",
    "PublicBenchmarkRepeatabilityAnalysis",
    "PublicBenchmarkRepeatabilityModeRunSummary",
    "PublicBenchmarkRepeatabilityModeSummary",
    "PublicBenchmarkRepeatabilityRunSummary",
    "analyze_public_benchmark_repeatability",
    "list_public_benchmark_run_dirs",
]
