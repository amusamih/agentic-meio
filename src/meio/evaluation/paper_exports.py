"""Paper-packaging exports derived from frozen MEIO result artifacts."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from meio.evaluation.logging_io import hash_jsonable, jsonable, write_json
from meio.evaluation.public_benchmark_repeatability import (
    PublicBenchmarkRepeatabilityAnalysis,
    PublicBenchmarkRepeatabilityModeSummary,
    analyze_public_benchmark_repeatability,
)
from meio.evaluation.real_demand_repeatability import (
    RealDemandRepeatabilityAnalysis,
    RealDemandRepeatabilityModeSummary,
    RealDemandSliceRepeatabilitySummary,
    analyze_real_demand_repeatability,
)
from meio.evaluation.validation_comparison import (
    ValidationStackSummary,
    default_validation_run_dirs,
    summarize_validation_stack,
)

DEFAULT_RESULTS_ROOT = Path("results")
DEFAULT_PAPER_EXPORTS_DIR = DEFAULT_RESULTS_ROOT / "paper_exports"
_MODE_BASELINE = "deterministic_baseline"
_MODE_DETERMINISTIC = "deterministic_orchestrator"
_MODE_LLM = "llm_orchestrator"


@dataclass(frozen=True, slots=True)
class PaperExportBundle:
    """Manifest for the generated paper-packaging artifacts."""

    output_dir: str
    created_files: tuple[str, ...]
    export_identity: str
    source_run_dirs: dict[str, object]


def export_paper_packaging(
    *,
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    output_dir: str | Path = DEFAULT_PAPER_EXPORTS_DIR,
) -> PaperExportBundle:
    """Export compact paper-ready tables and figure inputs."""

    results_root_path = Path(results_root)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    stockpyl_paper_dir = _latest_run_dir(results_root_path / "stockpyl_serial_paper_candidate")
    stockpyl_heldout_dir = _latest_run_dir(results_root_path / "stockpyl_serial_heldout_eval")
    stockpyl_broad_dir = _latest_run_dir(results_root_path / "stockpyl_serial_frozen_broad_eval")
    public_latest_dir = _latest_run_dir(results_root_path / "public_benchmark_eval")
    real_latest_dir = _latest_run_dir(results_root_path / "real_demand_backtest")

    public_repeatability = analyze_public_benchmark_repeatability(
        results_root=results_root_path / "public_benchmark_eval",
        latest_n=6,
    )
    real_repeatability = analyze_real_demand_repeatability(
        results_root=results_root_path / "real_demand_backtest",
        latest_n=3,
    )
    validation_summary = summarize_validation_stack(
        default_validation_run_dirs(results_root_path)
    )

    created_files: list[str] = []

    stockpyl_rows = _build_stockpyl_rows(
        {
            "paper_candidate": stockpyl_paper_dir,
            "heldout_eval": stockpyl_heldout_dir,
            "frozen_broad_eval": stockpyl_broad_dir,
        }
    )
    created_files.append(
        str(
            _write_csv(
                output_dir_path / "main_stockpyl_results.csv",
                stockpyl_rows,
                fieldnames=(
                    "evaluation_group",
                    "validation_lane",
                    "artifact_use_class",
                    "source_run_dir",
                    "mode",
                    "average_total_cost",
                    "average_fill_rate",
                    "regime_prediction_accuracy",
                    "average_tool_call_count",
                    "fallback_count",
                    "invalid_output_count",
                    "schedule_count",
                    "seed_count",
                ),
            )
        )
    )

    public_rows = _build_public_benchmark_rows(
        public_latest_dir=public_latest_dir,
        repeatability=public_repeatability,
    )
    created_files.append(
        str(
            _write_csv(
                output_dir_path / "public_benchmark_results.csv",
                public_rows,
                fieldnames=(
                    "validation_lane",
                    "artifact_use_class",
                    "source_latest_run_dir",
                    "repeatability_run_dirs",
                    "config_hash",
                    "mapping_identity",
                    "mode",
                    "latest_total_reward",
                    "latest_average_fill_rate",
                    "repeatability_mean_total_reward",
                    "repeatability_reward_std",
                    "repeatability_mean_fill_rate",
                    "repeatability_fill_rate_std",
                    "repeatability_total_fallback_count",
                    "repeatability_total_invalid_output_count",
                    "repeatability_mean_tool_call_count",
                    "repeatability_mean_replan_rate",
                    "repeatability_mean_no_action_rate",
                    "comparison_label",
                    "llm_validity_clean",
                    "provider_set",
                    "real_minus_fake_reward_mean",
                    "comparability_notes",
                ),
            )
        )
    )

    real_rows = _build_real_demand_rows(
        latest_panel_dir=real_latest_dir,
        repeatability=real_repeatability,
    )
    created_files.append(
        str(
            _write_csv(
                output_dir_path / "real_demand_panel_results.csv",
                real_rows,
                fieldnames=(
                    "scope",
                    "validation_lane",
                    "artifact_use_class",
                    "source_latest_panel_run_dir",
                    "source_panel_run_dirs",
                    "panel_config_hash",
                    "slice_name",
                    "dataset_name",
                    "date_range_start",
                    "date_range_end",
                    "selected_skus",
                    "run_count",
                    "baseline_cost_mean",
                    "deterministic_orchestrator_cost_mean",
                    "llm_cost_mean",
                    "baseline_fill_rate_mean",
                    "deterministic_orchestrator_fill_rate_mean",
                    "llm_fill_rate_mean",
                    "llm_minus_deterministic_orchestrator_cost_delta_mean",
                    "deterministic_orchestrator_minus_baseline_cost_delta_mean",
                    "llm_vs_deterministic_orchestrator_label",
                    "deterministic_orchestrator_vs_baseline_label",
                    "llm_validity_clean",
                    "llm_tool_call_mean",
                    "llm_replan_rate_mean",
                    "llm_no_action_rate_mean",
                    "real_minus_fake_cost_mean",
                ),
            )
        )
    )

    safety_rows = _build_safety_validity_rows(
        stockpyl_run_dirs=(
            ("stockpyl_paper_candidate", stockpyl_paper_dir),
            ("stockpyl_heldout_eval", stockpyl_heldout_dir),
            ("stockpyl_frozen_broad_eval", stockpyl_broad_dir),
        ),
        public_latest_dir=public_latest_dir,
        real_latest_dir=real_latest_dir,
    )
    created_files.append(
        str(
            _write_csv(
                output_dir_path / "safety_validity_summary.csv",
                safety_rows,
                fieldnames=(
                    "evaluation_group",
                    "validation_lane",
                    "artifact_use_class",
                    "source_run_dir",
                    "mode",
                    "validity_gate_passed",
                    "optimizer_order_boundary_preserved",
                    "operational_metrics_gate_passed",
                    "rollout_fidelity_gate_passed",
                    "fallback_count",
                    "invalid_output_count",
                    "average_tool_call_count",
                ),
            )
        )
    )

    lane_rows = _build_lane_comparison_rows(
        stockpyl_rows=stockpyl_rows,
        public_repeatability=public_repeatability,
        real_repeatability=real_repeatability,
        validation_summary=validation_summary,
        stockpyl_run_dirs={
            "paper_candidate": stockpyl_paper_dir,
            "heldout_eval": stockpyl_heldout_dir,
            "frozen_broad_eval": stockpyl_broad_dir,
        },
        public_latest_dir=public_latest_dir,
        real_latest_dir=real_latest_dir,
    )
    created_files.append(
        str(
            _write_csv(
                output_dir_path / "lane_comparison_summary.csv",
                lane_rows,
                fieldnames=(
                    "lane_or_group",
                    "role_in_paper",
                    "source_run_dir",
                    "repeatability_reference",
                    "primary_metric_name",
                    "primary_metric_direction",
                    "baseline_value",
                    "deterministic_orchestrator_value",
                    "llm_value",
                    "llm_minus_deterministic_orchestrator_delta",
                    "deterministic_orchestrator_minus_baseline_delta",
                    "stability_summary",
                    "validity_summary",
                    "still_synthetic",
                    "comparability_limit",
                    "recommended_weight",
                ),
            )
        )
    )

    created_files.append(
        str(
            write_json(
                output_dir_path / "figure_data_validation_stack.json",
                _build_validation_stack_figure_data(
                    validation_summary=validation_summary,
                    public_repeatability=public_repeatability,
                    real_repeatability=real_repeatability,
                ),
            )
        )
    )
    created_files.append(
        str(
            write_json(
                output_dir_path / "figure_data_mode_comparison.json",
                _build_mode_comparison_figure_data(
                    stockpyl_rows=stockpyl_rows,
                    public_repeatability=public_repeatability,
                    real_repeatability=real_repeatability,
                ),
            )
        )
    )
    created_files.append(
        str(
            write_json(
                output_dir_path / "figure_data_real_demand_slice_effects.json",
                _build_real_demand_slice_effect_figure_data(real_repeatability),
            )
        )
    )
    created_files.append(
        str(
            write_json(
                output_dir_path / "figure_data_safety_validity.json",
                {"rows": safety_rows},
            )
        )
    )

    manifest_path = output_dir_path / "paper_export_manifest.json"
    manifest = PaperExportBundle(
        output_dir=str(output_dir_path),
        created_files=tuple(sorted(created_files + [str(manifest_path)])),
        export_identity=hash_jsonable(
            {
                "stockpyl_run_dirs": {
                    "paper_candidate": str(stockpyl_paper_dir),
                    "heldout_eval": str(stockpyl_heldout_dir),
                    "frozen_broad_eval": str(stockpyl_broad_dir),
                },
                "public_latest_dir": str(public_latest_dir),
                "real_latest_dir": str(real_latest_dir),
                "public_repeatability_run_dirs": public_repeatability.run_dirs,
                "real_repeatability_run_dirs": real_repeatability.run_dirs,
            }
        ),
        source_run_dirs={
            "stockpyl_paper_candidate": str(stockpyl_paper_dir),
            "stockpyl_heldout_eval": str(stockpyl_heldout_dir),
            "stockpyl_frozen_broad_eval": str(stockpyl_broad_dir),
            "public_benchmark_latest": str(public_latest_dir),
            "public_benchmark_repeatability": list(public_repeatability.run_dirs),
            "real_demand_latest": str(real_latest_dir),
            "real_demand_repeatability": list(real_repeatability.panel_run_dirs),
        },
    )
    write_json(manifest_path, manifest)
    return manifest


def build_public_benchmark_rows(
    *,
    public_latest_dir: Path,
    repeatability: PublicBenchmarkRepeatabilityAnalysis,
) -> list[dict[str, object]]:
    """Build rows for the compact public-benchmark export."""

    return _build_public_benchmark_rows(
        public_latest_dir=public_latest_dir,
        repeatability=repeatability,
    )


def build_real_demand_rows(
    *,
    latest_panel_dir: Path,
    repeatability: RealDemandRepeatabilityAnalysis,
) -> list[dict[str, object]]:
    """Build rows for the compact real-demand export."""

    return _build_real_demand_rows(
        latest_panel_dir=latest_panel_dir,
        repeatability=repeatability,
    )


def build_lane_comparison_rows(
    *,
    stockpyl_rows: list[dict[str, object]],
    public_repeatability: PublicBenchmarkRepeatabilityAnalysis,
    real_repeatability: RealDemandRepeatabilityAnalysis,
    validation_summary: ValidationStackSummary,
    stockpyl_run_dirs: dict[str, Path],
    public_latest_dir: Path,
    real_latest_dir: Path,
) -> list[dict[str, object]]:
    """Build the compact lane-comparison export rows."""

    return _build_lane_comparison_rows(
        stockpyl_rows=stockpyl_rows,
        public_repeatability=public_repeatability,
        real_repeatability=real_repeatability,
        validation_summary=validation_summary,
        stockpyl_run_dirs=stockpyl_run_dirs,
        public_latest_dir=public_latest_dir,
        real_latest_dir=real_latest_dir,
    )


def _latest_run_dir(root: Path) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist.")
    candidates = tuple(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "run_manifest.json").exists()
    )
    if not candidates:
        raise FileNotFoundError(f"No saved runs found under {root}.")
    preferred = tuple(path for path in candidates if "_all_" in path.name)
    selection = preferred if preferred else candidates
    return sorted(selection)[-1]


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _build_stockpyl_rows(run_dirs: dict[str, Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for evaluation_group, run_dir in run_dirs.items():
        aggregate_payload = _load_json(run_dir / "aggregate_summary.json")
        for mode_payload in _mode_payloads(aggregate_payload):
            performance = _dict_payload(mode_payload.get("performance_summary"))
            decision = _dict_payload(mode_payload.get("decision_quality"))
            validity = _dict_payload(mode_payload.get("validity_summary"))
            tool_use = _dict_payload(mode_payload.get("tool_use_summary"))
            robustness = _dict_payload(mode_payload.get("robustness_summary"))
            rows.append(
                {
                    "evaluation_group": evaluation_group,
                    "validation_lane": "stockpyl_internal",
                    "artifact_use_class": _string(aggregate_payload.get("artifact_use_class")),
                    "source_run_dir": str(run_dir),
                    "mode": _string(mode_payload.get("mode")),
                    "average_total_cost": _number(performance.get("average_total_cost")),
                    "average_fill_rate": _number(performance.get("average_fill_rate")),
                    "regime_prediction_accuracy": _number(
                        decision.get("regime_prediction_accuracy")
                    ),
                    "average_tool_call_count": _number(
                        tool_use.get("average_tool_call_count")
                    ),
                    "fallback_count": _integer(validity.get("fallback_count")),
                    "invalid_output_count": _integer(
                        validity.get("invalid_output_count")
                    ),
                    "schedule_count": _integer(robustness.get("schedule_count")),
                    "seed_count": _integer(robustness.get("seed_count")),
                }
            )
    return rows


def _build_public_benchmark_rows(
    *,
    public_latest_dir: Path,
    repeatability: PublicBenchmarkRepeatabilityAnalysis,
) -> list[dict[str, object]]:
    latest_public_payload = _load_json(public_latest_dir / "public_benchmark_summary.json")
    latest_payloads = {
        _string(payload.get("mode")): payload
        for payload in _list_of_dicts(latest_public_payload.get("mode_summaries"))
        if _string(payload.get("mode")) is not None
    }
    mode_summaries = {
        summary.mode: summary for summary in repeatability.mode_summaries
    }
    providers = {
        summary.mode: ";".join(summary.llm_providers)
        for summary in repeatability.mode_summaries
    }
    rows: list[dict[str, object]] = []
    for mode_name in (_MODE_BASELINE, _MODE_DETERMINISTIC, _MODE_LLM):
        latest_mode = latest_payloads.get(mode_name, {})
        repeated_mode = mode_summaries.get(mode_name)
        comparison_label = ""
        if mode_name == _MODE_DETERMINISTIC:
            comparison_label = repeatability.deterministic_orchestrator_vs_baseline_label
        elif mode_name == _MODE_LLM:
            comparison_label = repeatability.llm_vs_deterministic_orchestrator_label
        rows.append(
            {
                "validation_lane": "public_benchmark",
                "artifact_use_class": "internal_only",
                "source_latest_run_dir": str(public_latest_dir),
                "repeatability_run_dirs": ";".join(repeatability.run_dirs),
                "config_hash": ";".join(repeatability.config_hashes),
                "mapping_identity": ";".join(repeatability.mapping_identities),
                "mode": mode_name,
                "latest_total_reward": _number(latest_mode.get("total_reward")),
                "latest_average_fill_rate": _number(
                    latest_mode.get("average_fill_rate")
                ),
                "repeatability_mean_total_reward": _spread_mean(
                    repeated_mode.reward if repeated_mode is not None else None
                ),
                "repeatability_reward_std": _spread_std(
                    repeated_mode.reward if repeated_mode is not None else None
                ),
                "repeatability_mean_fill_rate": _spread_mean(
                    repeated_mode.fill_rate if repeated_mode is not None else None
                ),
                "repeatability_fill_rate_std": _spread_std(
                    repeated_mode.fill_rate if repeated_mode is not None else None
                ),
                "repeatability_total_fallback_count": (
                    repeated_mode.total_fallback_count if repeated_mode is not None else None
                ),
                "repeatability_total_invalid_output_count": (
                    repeated_mode.total_invalid_output_count
                    if repeated_mode is not None
                    else None
                ),
                "repeatability_mean_tool_call_count": _spread_mean(
                    repeated_mode.tool_call_count if repeated_mode is not None else None
                ),
                "repeatability_mean_replan_rate": _spread_mean(
                    repeated_mode.replan_rate if repeated_mode is not None else None
                ),
                "repeatability_mean_no_action_rate": _spread_mean(
                    repeated_mode.no_action_rate if repeated_mode is not None else None
                ),
                "comparison_label": comparison_label,
                "llm_validity_clean": repeatability.llm_validity_clean,
                "provider_set": providers.get(mode_name, ""),
                "real_minus_fake_reward_mean": (
                    repeatability.fake_vs_real_llm_comparison.real_minus_fake_reward_mean
                    if mode_name == _MODE_LLM
                    and repeatability.fake_vs_real_llm_comparison is not None
                    else None
                ),
                "comparability_notes": ";".join(
                    str(item)
                    for item in latest_public_payload.get("comparability_notes", [])
                ),
            }
        )
    return rows


def _build_real_demand_rows(
    *,
    latest_panel_dir: Path,
    repeatability: RealDemandRepeatabilityAnalysis,
) -> list[dict[str, object]]:
    latest_aggregate_payload = _load_json(latest_panel_dir / "aggregate_summary.json")
    mode_summaries = {
        summary.mode: summary for summary in repeatability.mode_summaries
    }
    aggregate_row = {
        "scope": "panel_repeatability",
        "validation_lane": "real_demand_backtest",
        "artifact_use_class": _string(latest_aggregate_payload.get("artifact_use_class")),
        "source_latest_panel_run_dir": str(latest_panel_dir),
        "source_panel_run_dirs": ";".join(repeatability.panel_run_dirs),
        "panel_config_hash": ";".join(repeatability.panel_config_hashes),
        "slice_name": "",
        "dataset_name": "",
        "date_range_start": "",
        "date_range_end": "",
        "selected_skus": "",
        "run_count": repeatability.panel_run_count,
        "baseline_cost_mean": _mode_cost_mean(mode_summaries, _MODE_BASELINE),
        "deterministic_orchestrator_cost_mean": _mode_cost_mean(
            mode_summaries, _MODE_DETERMINISTIC
        ),
        "llm_cost_mean": _mode_cost_mean(mode_summaries, _MODE_LLM),
        "baseline_fill_rate_mean": _mode_fill_mean(mode_summaries, _MODE_BASELINE),
        "deterministic_orchestrator_fill_rate_mean": _mode_fill_mean(
            mode_summaries, _MODE_DETERMINISTIC
        ),
        "llm_fill_rate_mean": _mode_fill_mean(mode_summaries, _MODE_LLM),
        "llm_minus_deterministic_orchestrator_cost_delta_mean": _mode_delta_mean(
            repeatability, left_mode=_MODE_LLM, right_mode=_MODE_DETERMINISTIC
        ),
        "deterministic_orchestrator_minus_baseline_cost_delta_mean": _mode_delta_mean(
            repeatability, left_mode=_MODE_DETERMINISTIC, right_mode=_MODE_BASELINE
        ),
        "llm_vs_deterministic_orchestrator_label": (
            repeatability.llm_vs_deterministic_orchestrator_label
        ),
        "deterministic_orchestrator_vs_baseline_label": (
            repeatability.deterministic_orchestrator_vs_baseline_label
        ),
        "llm_validity_clean": repeatability.llm_validity_clean,
        "llm_tool_call_mean": _mode_tool_calls(mode_summaries, _MODE_LLM),
        "llm_replan_rate_mean": _mode_replan_mean(mode_summaries, _MODE_LLM),
        "llm_no_action_rate_mean": _mode_no_action_mean(mode_summaries, _MODE_LLM),
        "real_minus_fake_cost_mean": (
            repeatability.fake_vs_real_llm_comparison.real_minus_fake_cost_mean
            if repeatability.fake_vs_real_llm_comparison is not None
            else None
        ),
    }
    rows = [aggregate_row]
    for slice_summary in repeatability.slice_summaries:
        slice_mode_map = {summary.mode: summary for summary in slice_summary.mode_summaries}
        rows.append(
            {
                "scope": "slice_repeatability",
                "validation_lane": "real_demand_backtest",
                "artifact_use_class": _string(
                    latest_aggregate_payload.get("artifact_use_class")
                ),
                "source_latest_panel_run_dir": str(latest_panel_dir),
                "source_panel_run_dirs": ";".join(repeatability.panel_run_dirs),
                "panel_config_hash": ";".join(slice_summary.panel_config_hashes),
                "slice_name": slice_summary.slice_name,
                "dataset_name": slice_summary.dataset_name or "",
                "date_range_start": slice_summary.date_range[0]
                if slice_summary.date_range
                else "",
                "date_range_end": slice_summary.date_range[-1]
                if slice_summary.date_range
                else "",
                "selected_skus": ";".join(slice_summary.selected_skus),
                "run_count": slice_summary.run_count,
                "baseline_cost_mean": _mode_cost_mean(slice_mode_map, _MODE_BASELINE),
                "deterministic_orchestrator_cost_mean": _mode_cost_mean(
                    slice_mode_map, _MODE_DETERMINISTIC
                ),
                "llm_cost_mean": _mode_cost_mean(slice_mode_map, _MODE_LLM),
                "baseline_fill_rate_mean": _mode_fill_mean(
                    slice_mode_map, _MODE_BASELINE
                ),
                "deterministic_orchestrator_fill_rate_mean": _mode_fill_mean(
                    slice_mode_map, _MODE_DETERMINISTIC
                ),
                "llm_fill_rate_mean": _mode_fill_mean(slice_mode_map, _MODE_LLM),
                "llm_minus_deterministic_orchestrator_cost_delta_mean": _spread_mean(
                    slice_summary.llm_minus_deterministic_orchestrator_cost_delta
                ),
                "deterministic_orchestrator_minus_baseline_cost_delta_mean": _spread_mean(
                    slice_summary.deterministic_orchestrator_minus_baseline_cost_delta
                ),
                "llm_vs_deterministic_orchestrator_label": (
                    slice_summary.llm_vs_deterministic_orchestrator_label
                ),
                "deterministic_orchestrator_vs_baseline_label": (
                    slice_summary.deterministic_orchestrator_vs_baseline_label
                ),
                "llm_validity_clean": slice_summary.llm_validity_clean,
                "llm_tool_call_mean": _mode_tool_calls(slice_mode_map, _MODE_LLM),
                "llm_replan_rate_mean": _mode_replan_mean(slice_mode_map, _MODE_LLM),
                "llm_no_action_rate_mean": _mode_no_action_mean(slice_mode_map, _MODE_LLM),
                "real_minus_fake_cost_mean": "",
            }
        )
    return rows


def _build_safety_validity_rows(
    *,
    stockpyl_run_dirs: tuple[tuple[str, Path], ...],
    public_latest_dir: Path,
    real_latest_dir: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for evaluation_group, run_dir in stockpyl_run_dirs:
        aggregate_payload = _load_json(run_dir / "aggregate_summary.json")
        rows.extend(
            _safety_rows_from_aggregate(
                evaluation_group=evaluation_group,
                validation_lane="stockpyl_internal",
                run_dir=run_dir,
                aggregate_payload=aggregate_payload,
            )
        )
    rows.extend(
        _safety_rows_from_aggregate(
            evaluation_group="public_benchmark_latest",
            validation_lane="public_benchmark",
            run_dir=public_latest_dir,
            aggregate_payload=_load_json(public_latest_dir / "aggregate_summary.json"),
        )
    )
    rows.extend(
        _safety_rows_from_aggregate(
            evaluation_group="real_demand_panel_latest",
            validation_lane="real_demand_backtest",
            run_dir=real_latest_dir,
            aggregate_payload=_load_json(real_latest_dir / "aggregate_summary.json"),
        )
    )
    return rows


def _safety_rows_from_aggregate(
    *,
    evaluation_group: str,
    validation_lane: str,
    run_dir: Path,
    aggregate_payload: dict[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for mode_payload in _mode_payloads(aggregate_payload):
        validity = _dict_payload(mode_payload.get("validity_summary"))
        tool_use = _dict_payload(mode_payload.get("tool_use_summary"))
        rows.append(
            {
                "evaluation_group": evaluation_group,
                "validation_lane": validation_lane,
                "artifact_use_class": _string(aggregate_payload.get("artifact_use_class")),
                "source_run_dir": str(run_dir),
                "mode": _string(mode_payload.get("mode")),
                "validity_gate_passed": aggregate_payload.get("validity_gate_passed"),
                "optimizer_order_boundary_preserved": validity.get(
                    "optimizer_order_boundary_preserved"
                ),
                "operational_metrics_gate_passed": validity.get(
                    "operational_metrics_gate_passed"
                ),
                "rollout_fidelity_gate_passed": validity.get(
                    "rollout_fidelity_gate_passed"
                ),
                "fallback_count": _integer(validity.get("fallback_count")),
                "invalid_output_count": _integer(validity.get("invalid_output_count")),
                "average_tool_call_count": _number(
                    tool_use.get("average_tool_call_count")
                ),
            }
        )
    return rows


def _build_lane_comparison_rows(
    *,
    stockpyl_rows: list[dict[str, object]],
    public_repeatability: PublicBenchmarkRepeatabilityAnalysis,
    real_repeatability: RealDemandRepeatabilityAnalysis,
    validation_summary: ValidationStackSummary,
    stockpyl_run_dirs: dict[str, Path],
    public_latest_dir: Path,
    real_latest_dir: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    stockpyl_groups = (
        ("paper_candidate", "main_evidence"),
        ("heldout_eval", "main_evidence"),
        ("frozen_broad_eval", "main_evidence"),
    )
    for evaluation_group, role in stockpyl_groups:
        group_rows = [
            row for row in stockpyl_rows if row["evaluation_group"] == evaluation_group
        ]
        baseline_value = _row_value(group_rows, _MODE_BASELINE, "average_total_cost")
        deterministic_value = _row_value(
            group_rows, _MODE_DETERMINISTIC, "average_total_cost"
        )
        llm_value = _row_value(group_rows, _MODE_LLM, "average_total_cost")
        rows.append(
            {
                "lane_or_group": f"stockpyl_internal_{evaluation_group}",
                "role_in_paper": role,
                "source_run_dir": str(stockpyl_run_dirs[evaluation_group]),
                "repeatability_reference": "latest_frozen_internal_screen_only",
                "primary_metric_name": "average_total_cost",
                "primary_metric_direction": "lower_is_better",
                "baseline_value": baseline_value,
                "deterministic_orchestrator_value": deterministic_value,
                "llm_value": llm_value,
                "llm_minus_deterministic_orchestrator_delta": _delta(
                    llm_value, deterministic_value
                ),
                "deterministic_orchestrator_minus_baseline_delta": _delta(
                    deterministic_value, baseline_value
                ),
                "stability_summary": "latest_frozen_internal_screen_only",
                "validity_summary": "validity_clean_latest_run",
                "still_synthetic": True,
                "comparability_limit": "synthetic_internal_simulator_lane",
                "recommended_weight": (
                    "highest" if evaluation_group == "paper_candidate" else "high"
                ),
            }
        )
    public_mode_map = {summary.mode: summary for summary in public_repeatability.mode_summaries}
    public_baseline = _mode_reward_mean(public_mode_map, _MODE_BASELINE)
    public_deterministic = _mode_reward_mean(public_mode_map, _MODE_DETERMINISTIC)
    public_llm = _mode_reward_mean(public_mode_map, _MODE_LLM)
    rows.append(
        {
            "lane_or_group": "public_benchmark",
            "role_in_paper": "supporting_external",
            "source_run_dir": str(public_latest_dir),
            "repeatability_reference": ";".join(public_repeatability.run_dirs),
            "primary_metric_name": "total_reward",
            "primary_metric_direction": "higher_is_better",
            "baseline_value": public_baseline,
            "deterministic_orchestrator_value": public_deterministic,
            "llm_value": public_llm,
            "llm_minus_deterministic_orchestrator_delta": _delta(
                public_llm, public_deterministic
            ),
            "deterministic_orchestrator_minus_baseline_delta": _delta(
                public_deterministic, public_baseline
            ),
            "stability_summary": (
                f"deterministic_vs_baseline={public_repeatability.deterministic_orchestrator_vs_baseline_label};"
                f"llm_vs_deterministic={public_repeatability.llm_vs_deterministic_orchestrator_label}"
            ),
            "validity_summary": (
                "llm_validity_clean" if public_repeatability.llm_validity_clean else "llm_validity_issue"
            ),
            "still_synthetic": False,
            "comparability_limit": "reward_not_cost_partial_single_store_mapping",
            "recommended_weight": "supporting",
        }
    )
    real_mode_map = {summary.mode: summary for summary in real_repeatability.mode_summaries}
    real_baseline = _mode_cost_mean(real_mode_map, _MODE_BASELINE)
    real_deterministic = _mode_cost_mean(real_mode_map, _MODE_DETERMINISTIC)
    real_llm = _mode_cost_mean(real_mode_map, _MODE_LLM)
    rows.append(
        {
            "lane_or_group": "real_demand_backtest",
            "role_in_paper": "supporting_external",
            "source_run_dir": str(real_latest_dir),
            "repeatability_reference": ";".join(real_repeatability.panel_run_dirs),
            "primary_metric_name": "average_total_cost",
            "primary_metric_direction": "lower_is_better",
            "baseline_value": real_baseline,
            "deterministic_orchestrator_value": real_deterministic,
            "llm_value": real_llm,
            "llm_minus_deterministic_orchestrator_delta": _delta(
                llm_value=real_llm,
                other_value=real_deterministic,
            ),
            "deterministic_orchestrator_minus_baseline_delta": _delta(
                llm_value=real_deterministic,
                other_value=real_baseline,
            ),
            "stability_summary": (
                f"aggregate={real_repeatability.llm_vs_deterministic_orchestrator_label};"
                f"slices_llm={';'.join(real_repeatability.slices_favoring_llm_orchestrator)};"
                f"slices_det={';'.join(real_repeatability.slices_favoring_deterministic_orchestrator)}"
            ),
            "validity_summary": (
                "llm_validity_clean" if real_repeatability.llm_validity_clean else "llm_validity_issue"
            ),
            "still_synthetic": False,
            "comparability_limit": "stockpyl_cost_backbone_no_native_regime_labels",
            "recommended_weight": "supporting",
        }
    )
    return rows


def _build_validation_stack_figure_data(
    *,
    validation_summary: ValidationStackSummary,
    public_repeatability: PublicBenchmarkRepeatabilityAnalysis,
    real_repeatability: RealDemandRepeatabilityAnalysis,
) -> dict[str, object]:
    return {
        "main_method": "bounded_orchestrator_with_trusted_optimizer",
        "hierarchy": [
            {
                "lane": "stockpyl_internal",
                "role": "main_evidence",
                "status": "completed",
            },
            {
                "lane": "public_benchmark",
                "role": "supporting_external",
                "status": "completed",
            },
            {
                "lane": "real_demand_backtest",
                "role": "supporting_external",
                "status": "completed",
            },
        ],
        "lane_coverage": jsonable(validation_summary.lane_coverage),
        "stability_highlights": {
            "public_benchmark_llm_vs_deterministic": (
                public_repeatability.llm_vs_deterministic_orchestrator_label
            ),
            "real_demand_llm_vs_deterministic": (
                real_repeatability.llm_vs_deterministic_orchestrator_label
            ),
            "real_demand_slice_favors_llm": list(
                real_repeatability.slices_favoring_llm_orchestrator
            ),
            "real_demand_slice_favors_deterministic": list(
                real_repeatability.slices_favoring_deterministic_orchestrator
            ),
        },
    }


def _build_mode_comparison_figure_data(
    *,
    stockpyl_rows: list[dict[str, object]],
    public_repeatability: PublicBenchmarkRepeatabilityAnalysis,
    real_repeatability: RealDemandRepeatabilityAnalysis,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for stockpyl_row in stockpyl_rows:
        rows.append(
            {
                "lane_or_group": stockpyl_row["evaluation_group"],
                "metric_name": "average_total_cost",
                "higher_is_better": False,
                "mode": stockpyl_row["mode"],
                "value": stockpyl_row["average_total_cost"],
                "fill_rate": stockpyl_row["average_fill_rate"],
            }
        )
    public_mode_map = {summary.mode: summary for summary in public_repeatability.mode_summaries}
    for mode_name in (_MODE_BASELINE, _MODE_DETERMINISTIC, _MODE_LLM):
        mode_summary = public_mode_map[mode_name]
        rows.append(
            {
                "lane_or_group": "public_benchmark_repeatability",
                "metric_name": "total_reward",
                "higher_is_better": True,
                "mode": mode_name,
                "value": _spread_mean(mode_summary.reward),
                "fill_rate": _spread_mean(mode_summary.fill_rate),
            }
        )
    real_mode_map = {summary.mode: summary for summary in real_repeatability.mode_summaries}
    for mode_name in (_MODE_BASELINE, _MODE_DETERMINISTIC, _MODE_LLM):
        mode_summary = real_mode_map[mode_name]
        rows.append(
            {
                "lane_or_group": "real_demand_panel_repeatability",
                "metric_name": "average_total_cost",
                "higher_is_better": False,
                "mode": mode_name,
                "value": _spread_mean(mode_summary.average_total_cost),
                "fill_rate": _spread_mean(mode_summary.average_fill_rate),
            }
        )
    return {"rows": rows}


def _build_real_demand_slice_effect_figure_data(
    repeatability: RealDemandRepeatabilityAnalysis,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for slice_summary in repeatability.slice_summaries:
        mode_map = {summary.mode: summary for summary in slice_summary.mode_summaries}
        rows.append(
            {
                "slice_name": slice_summary.slice_name,
                "dataset_name": slice_summary.dataset_name,
                "baseline_cost_mean": _mode_cost_mean(mode_map, _MODE_BASELINE),
                "deterministic_orchestrator_cost_mean": _mode_cost_mean(
                    mode_map, _MODE_DETERMINISTIC
                ),
                "llm_cost_mean": _mode_cost_mean(mode_map, _MODE_LLM),
                "llm_minus_deterministic_cost_delta_mean": _spread_mean(
                    slice_summary.llm_minus_deterministic_orchestrator_cost_delta
                ),
                "deterministic_minus_baseline_cost_delta_mean": _spread_mean(
                    slice_summary.deterministic_orchestrator_minus_baseline_cost_delta
                ),
                "llm_vs_deterministic_label": (
                    slice_summary.llm_vs_deterministic_orchestrator_label
                ),
                "deterministic_vs_baseline_label": (
                    slice_summary.deterministic_orchestrator_vs_baseline_label
                ),
            }
        )
    return {
        "panel_run_dirs": list(repeatability.panel_run_dirs),
        "panel_config_hashes": list(repeatability.panel_config_hashes),
        "rows": rows,
    }


def _write_csv(
    path: Path,
    rows: list[dict[str, object]],
    *,
    fieldnames: tuple[str, ...],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_cell(row.get(name)) for name in fieldnames})
    return path


def _csv_cell(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, str)):
        return value
    return json.dumps(jsonable(value), sort_keys=True)


def _mode_payloads(payload: dict[str, object]) -> list[dict[str, object]]:
    raw_payload = payload.get("mode_summaries")
    return _list_of_dicts(raw_payload)


def _list_of_dicts(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _dict_payload(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Expected a string when present.")
    return value


def _number(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Expected a number when present.")
    return float(value)


def _integer(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("Expected an integer when present.")
    return value


def _row_value(
    rows: list[dict[str, object]],
    mode_name: str,
    field_name: str,
) -> float | None:
    for row in rows:
        if row.get("mode") == mode_name:
            value = row.get(field_name)
            return float(value) if isinstance(value, (int, float)) else None
    return None


def _spread_mean(spread: object) -> float | None:
    if spread is None:
        return None
    payload = getattr(spread, "mean_value", None)
    return float(payload) if isinstance(payload, (int, float)) else None


def _spread_std(spread: object) -> float | None:
    if spread is None:
        return None
    payload = getattr(spread, "standard_deviation", None)
    return float(payload) if isinstance(payload, (int, float)) else None


def _mode_reward_mean(
    mode_map: dict[str, PublicBenchmarkRepeatabilityModeSummary],
    mode_name: str,
) -> float | None:
    summary = mode_map.get(mode_name)
    return _spread_mean(summary.reward if summary is not None else None)


def _mode_cost_mean(
    mode_map: dict[str, object],
    mode_name: str,
) -> float | None:
    summary = mode_map.get(mode_name)
    if summary is None:
        return None
    return _spread_mean(getattr(summary, "average_total_cost", None))


def _mode_fill_mean(
    mode_map: dict[str, object],
    mode_name: str,
) -> float | None:
    summary = mode_map.get(mode_name)
    if summary is None:
        return None
    return _spread_mean(getattr(summary, "average_fill_rate", None))


def _mode_tool_calls(
    mode_map: dict[str, object],
    mode_name: str,
) -> float | None:
    summary = mode_map.get(mode_name)
    if summary is None:
        return None
    return _spread_mean(getattr(summary, "tool_call_count", None))


def _mode_replan_mean(
    mode_map: dict[str, object],
    mode_name: str,
) -> float | None:
    summary = mode_map.get(mode_name)
    if summary is None:
        return None
    return _spread_mean(getattr(summary, "replan_rate", None))


def _mode_no_action_mean(
    mode_map: dict[str, object],
    mode_name: str,
) -> float | None:
    summary = mode_map.get(mode_name)
    if summary is None:
        return None
    return _spread_mean(getattr(summary, "no_action_rate", None))


def _mode_delta_mean(
    repeatability: RealDemandRepeatabilityAnalysis,
    *,
    left_mode: str,
    right_mode: str,
) -> float | None:
    mode_map = {summary.mode: summary for summary in repeatability.mode_summaries}
    left_value = _mode_cost_mean(mode_map, left_mode)
    right_value = _mode_cost_mean(mode_map, right_mode)
    return _delta(left_value, right_value)


def _delta(
    llm_value: float | None = None,
    other_value: float | None = None,
) -> float | None:
    if llm_value is None or other_value is None:
        return None
    return llm_value - other_value


__all__ = [
    "DEFAULT_PAPER_EXPORTS_DIR",
    "PaperExportBundle",
    "build_lane_comparison_rows",
    "build_public_benchmark_rows",
    "build_real_demand_rows",
    "export_paper_packaging",
]
