"""Run bounded Stockpyl serial rollouts across deterministic and LLM orchestrators."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from meio.agents.baselines import DeterministicBaselinePolicy
from meio.agents.external_evidence_tool import (
    BoundedExternalEvidenceTool,
)
from meio.agents.llm_client import FakeLLMClient, OpenAILLMClient
from meio.agents.llm_orchestrator import LLMOrchestrator
from meio.agents.prompts import PROMPT_VERSION, prompt_contract_hash
from meio.agents.runtime import OrchestrationResponse, OrchestrationRuntime, RuntimeMode
from meio.agents.telemetry import (
    EpisodeTelemetrySummary,
    OrchestrationStepTelemetry,
    summarize_episode_telemetry,
)
from meio.config.loaders import load_agent_config, load_benchmark_config, load_experiment_config
from meio.config.schemas import (
    AgentConfig,
    ExperimentConfig,
)
from meio.contracts import MissionSpec, OperationalSubgoal, RegimeLabel, UpdateRequestType
from meio.data.stockpyl_adapter import StockpylSerialAdapter
from meio.evaluation.aggregate_results import (
    BatchAggregateSummary,
    ModeAggregateSummary,
    aggregate_batch_episode_summaries,
    aggregate_mode_episode_summaries,
)
from meio.evaluation.decision_quality import DecisionQualitySummary, compute_decision_quality
from meio.evaluation.logging_io import (
    ensure_output_dir,
    hash_jsonable,
    jsonable,
    utc_timestamp,
    write_episode_summaries_jsonl,
    write_experiment_metadata_json,
    write_json,
    write_llm_call_traces_jsonl,
    write_run_manifest_json,
    write_step_traces_jsonl,
    write_tool_call_traces_jsonl,
)
from meio.evaluation.logging_schema import (
    CostBreakdownRecord,
    EpisodeSummaryRecord,
    ExperimentMetadata,
    LLMCallTraceRecord,
    RunManifestRecord,
    StepTraceRecord,
    ToolCallTraceRecord,
)
from meio.evaluation.results_index import (
    classify_artifact_governance,
    compute_validity_gate_passed,
    summarize_directory_governance,
)
from meio.evaluation.llm_run_summaries import ComparisonRunSummary, LLMRunDiagnostics
from meio.evaluation.tool_attribution import (
    ToolAblationSummary,
    ToolUsageSummary,
)
from meio.evaluation.rollout_metrics import (
    RolloutMetrics,
    compute_period_cost_breakdown,
    compute_period_fill_rate,
    compute_period_total_cost,
    compute_rollout_metrics,
)
from meio.evaluation.summaries import (
    BenchmarkRunSummary,
    InterventionSummary,
    TraceSummary,
    build_episode_summary_record,
)
from meio.forecasting.adapters import DeterministicForecastTool
from meio.leadtime.adapters import DeterministicLeadTimeTool
from meio.optimization.adapters import TrustedOptimizerAdapter, build_optimization_request
from meio.optimization.contracts import OptimizationRequest, OptimizationResult
from meio.scenarios.adapters import DeterministicScenarioTool
from meio.scenarios.contracts import (
    ScenarioAdjustmentSummary,
    ScenarioSummary,
    ScenarioUpdateResult,
)
from meio.simulation.serial_benchmark import (
    SerialBenchmarkCase,
    advance_serial_state,
    build_initial_simulation_state,
    build_period_observation,
    build_runtime_evidence,
    build_serial_benchmark_case,
    build_serial_orchestration_request,
    regime_for_period,
)
from meio.simulation.external_evidence_alignment import (
    build_external_evidence_batch,
)
from meio.simulation.state import EpisodeTrace, PeriodTraceRecord


DEFAULT_EXPERIMENT_CONFIG = Path("configs/experiment/stockpyl_serial.toml")
LEGACY_COMPARISON_MODES = (
    "deterministic_baseline",
    "deterministic_orchestrator",
    "llm_orchestrator",
)
LLM_INTERNAL_ONLY_MODE = "llm_orchestrator_internal_only"
LLM_WITH_EXTERNAL_EVIDENCE_MODE = "llm_orchestrator_with_external_evidence"
EXTERNAL_EVIDENCE_TOOL_ID = "external_evidence_tool"
VALIDATION_LANE = "stockpyl_internal"
TOOL_IDS_BY_ABLATION = {
    "full": ("forecast_tool", "leadtime_tool", "scenario_tool"),
    "no_forecast_tool": ("leadtime_tool", "scenario_tool"),
    "no_leadtime_tool": ("forecast_tool", "scenario_tool"),
    "no_scenario_tool": ("forecast_tool", "leadtime_tool"),
}


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """One explicit schedule/seed case for controlled comparison."""

    case_index: int
    schedule_name: str
    regime_schedule: tuple[RegimeLabel, ...]
    run_seed: int
    tool_ablation_variant: str = "full"

    def __post_init__(self) -> None:
        if self.case_index < 0:
            raise ValueError("case_index must be non-negative.")
        if not self.schedule_name.strip():
            raise ValueError("schedule_name must be non-empty.")
        object.__setattr__(self, "regime_schedule", tuple(self.regime_schedule))
        if not self.regime_schedule:
            raise ValueError("regime_schedule must not be empty.")
        if self.run_seed < 0:
            raise ValueError("run_seed must be non-negative.")
        if not self.tool_ablation_variant.strip():
            raise ValueError("tool_ablation_variant must be non-empty.")


@dataclass(frozen=True, slots=True)
class StockpylSerialRun:
    """Structured result from one controlled Stockpyl serial rollout."""

    mode: str
    run_index: int
    schedule_name: str
    run_seed: int
    tool_ablation_variant: str
    benchmark_case: SerialBenchmarkCase
    trace: EpisodeTrace
    rollout_metrics: RolloutMetrics
    summary: BenchmarkRunSummary
    regime_schedule: tuple[RegimeLabel, ...]
    episode_summary_record: EpisodeSummaryRecord | None = None
    step_trace_records: tuple[StepTraceRecord, ...] = ()
    llm_call_trace_records: tuple[LLMCallTraceRecord, ...] = ()
    tool_call_trace_records: tuple[ToolCallTraceRecord, ...] = ()
    orchestration_responses: tuple[OrchestrationResponse, ...] = ()
    llm_provider: str | None = None
    llm_model_name: str | None = None
    prompt_version: str | None = None
    prompt_contract_hash: str | None = None
    invalid_output_count: int = 0
    fallback_count: int = 0
    successful_response_count: int = 0
    validation_failure_counts: tuple[tuple[str, int], ...] = ()
    client_error_counts: tuple[tuple[str, int], ...] = ()
    total_retry_count: int = 0
    failure_before_response_count: int = 0
    failure_after_response_count: int = 0

    def to_summary(self) -> dict[str, object]:
        """Return a compact JSON-serializable summary."""

        summary = self.summary.to_record()
        summary["mode"] = self.mode
        summary["run_index"] = self.run_index
        summary["schedule_name"] = self.schedule_name
        summary["run_seed"] = self.run_seed
        summary["tool_ablation_variant"] = self.tool_ablation_variant
        summary["regime_schedule"] = [label.value for label in self.regime_schedule]
        summary["trace_length"] = self.trace.trace_length
        summary["orders_by_period"] = [
            list(result.replenishment_orders) for result in self.trace.optimization_results
        ]
        summary["llm_provider"] = self.llm_provider
        summary["llm_model_name"] = self.llm_model_name
        summary["prompt_version"] = self.prompt_version
        summary["prompt_hash"] = self.prompt_contract_hash
        summary["invalid_output_count"] = self.invalid_output_count
        summary["fallback_count"] = self.fallback_count
        summary["successful_response_count"] = self.successful_response_count
        summary["validation_failure_counts"] = [
            [reason, count] for reason, count in self.validation_failure_counts
        ]
        summary["client_error_counts"] = [
            [reason, count] for reason, count in self.client_error_counts
        ]
        summary["total_retry_count"] = self.total_retry_count
        summary["failure_before_response_count"] = self.failure_before_response_count
        summary["failure_after_response_count"] = self.failure_after_response_count
        summary["step_trace_count"] = len(self.step_trace_records)
        summary["llm_call_trace_count"] = len(self.llm_call_trace_records)
        summary["tool_call_trace_count"] = len(self.tool_call_trace_records)
        if self.episode_summary_record is not None:
            summary["rollout_fidelity_gate_passed"] = (
                self.episode_summary_record.rollout_fidelity_gate_passed
            )
            summary["operational_metrics_gate_passed"] = (
                self.episode_summary_record.operational_metrics_gate_passed
            )
            summary["unavailable_tool_request_count"] = (
                self.episode_summary_record.unavailable_tool_request_count
            )
            summary["disabled_tool_fallback_count"] = (
                self.episode_summary_record.disabled_tool_fallback_count
            )
            summary["sequencing_blocked_tool_request_count"] = (
                self.episode_summary_record.sequencing_blocked_tool_request_count
            )
            summary["clean_intervention_count"] = (
                self.episode_summary_record.clean_intervention_count
            )
            summary["clean_optimizer_input_change_count"] = (
                self.episode_summary_record.clean_optimizer_input_change_count
            )
            summary["repeated_stress_moderation_count"] = (
                self.episode_summary_record.repeated_stress_moderation_count
            )
            summary["relapse_moderation_count"] = (
                self.episode_summary_record.relapse_moderation_count
            )
            summary["unresolved_stress_moderation_count"] = (
                self.episode_summary_record.unresolved_stress_moderation_count
            )
            summary["moderated_update_count"] = (
                self.episode_summary_record.moderated_update_count
            )
            summary["hysteresis_application_count"] = (
                self.episode_summary_record.hysteresis_application_count
            )
        return summary


@dataclass(frozen=True, slots=True)
class StockpylSerialComparison:
    """Structured single-run comparison across the supported rollout modes."""

    benchmark_case: SerialBenchmarkCase
    schedule_name: str
    run_seed: int
    regime_schedule: tuple[RegimeLabel, ...]
    deterministic_baseline: StockpylSerialRun
    deterministic_orchestrator: StockpylSerialRun
    llm_orchestrator: StockpylSerialRun
    outcomes_differ: bool
    optimizer_order_boundary_preserved: bool

    def to_summary(self) -> dict[str, object]:
        """Return a compact JSON-serializable comparison summary."""

        return {
            "benchmark_id": self.benchmark_case.benchmark_id,
            "benchmark_source": self.benchmark_case.adapter_name,
            "schedule_name": self.schedule_name,
            "run_seed": self.run_seed,
            "regime_schedule": [label.value for label in self.regime_schedule],
            "outcomes_differ": self.outcomes_differ,
            "optimizer_order_boundary_preserved": self.optimizer_order_boundary_preserved,
            "deterministic_baseline": self.deterministic_baseline.to_summary(),
            "deterministic_orchestrator": self.deterministic_orchestrator.to_summary(),
            "llm_orchestrator": self.llm_orchestrator.to_summary(),
        }


@dataclass(frozen=True, slots=True)
class StockpylSerialModeBatch:
    """Repeated-run aggregate for one rollout mode."""

    mode: str
    runs: tuple[StockpylSerialRun, ...]
    aggregate_summary: ComparisonRunSummary
    aggregate_metrics: ModeAggregateSummary | None = None
    tool_usage_by_ablation: tuple[ToolUsageSummary, ...] = ()
    tool_ablation_summaries: tuple[ToolAblationSummary, ...] = ()

    def to_summary(self) -> dict[str, object]:
        """Return a compact JSON-serializable batch summary."""

        aggregate = {
            "mode": self.aggregate_summary.mode,
            "run_count": self.aggregate_summary.run_count,
            "average_tool_call_count": self.aggregate_summary.average_tool_call_count,
            "average_replan_count": self.aggregate_summary.average_replan_count,
            "average_abstain_count": self.aggregate_summary.average_abstain_count,
            "average_no_action_count": self.aggregate_summary.average_no_action_count,
            "average_total_cost": self.aggregate_summary.average_total_cost,
            "average_fill_rate": self.aggregate_summary.average_fill_rate,
            "optimizer_order_boundary_preserved": (
                self.aggregate_summary.optimizer_order_boundary_preserved
            ),
        }
        if self.aggregate_summary.decision_quality is not None:
            aggregate["decision_quality"] = self.aggregate_summary.decision_quality.to_record()
        if self.aggregate_summary.llm_diagnostics is not None:
            aggregate["llm_diagnostics"] = {
                "model_name": self.aggregate_summary.llm_diagnostics.model_name,
                "provider": self.aggregate_summary.llm_diagnostics.provider,
                "run_count": self.aggregate_summary.llm_diagnostics.run_count,
                "prompt_version": self.aggregate_summary.llm_diagnostics.prompt_version,
                "prompt_hash": self.aggregate_summary.llm_diagnostics.prompt_hash,
                "average_prompt_tokens": (
                    self.aggregate_summary.llm_diagnostics.average_prompt_tokens
                ),
                "average_completion_tokens": (
                    self.aggregate_summary.llm_diagnostics.average_completion_tokens
                ),
                "average_total_tokens": (
                    self.aggregate_summary.llm_diagnostics.average_total_tokens
                ),
                "average_llm_latency_ms": (
                    self.aggregate_summary.llm_diagnostics.average_llm_latency_ms
                ),
                "average_orchestration_latency_ms": (
                    self.aggregate_summary.llm_diagnostics.average_orchestration_latency_ms
                ),
                "invalid_output_count": self.aggregate_summary.llm_diagnostics.invalid_output_count,
                "fallback_count": self.aggregate_summary.llm_diagnostics.fallback_count,
                "successful_response_count": (
                    self.aggregate_summary.llm_diagnostics.successful_response_count
                ),
                "validation_failure_counts": [
                    [reason, count]
                    for reason, count in (
                        self.aggregate_summary.llm_diagnostics.validation_failure_counts
                    )
                ],
                "client_error_counts": [
                    [reason, count]
                    for reason, count in (
                        self.aggregate_summary.llm_diagnostics.client_error_counts
                    )
                ],
                "total_retry_count": self.aggregate_summary.llm_diagnostics.total_retry_count,
                "failure_before_response_count": (
                    self.aggregate_summary.llm_diagnostics.failure_before_response_count
                ),
                "failure_after_response_count": (
                    self.aggregate_summary.llm_diagnostics.failure_after_response_count
                ),
                "optimizer_order_boundary_preserved": (
                    self.aggregate_summary.llm_diagnostics.optimizer_order_boundary_preserved
                ),
            }
        if self.aggregate_metrics is not None:
            aggregate["artifact_use_class"] = self.aggregate_metrics.artifact_use_class.value
            aggregate["validity_gate_passed"] = self.aggregate_metrics.validity_gate_passed
            aggregate["eligibility_notes"] = list(self.aggregate_metrics.eligibility_notes)
            aggregate["schedule_names"] = list(self.aggregate_metrics.schedule_names)
            aggregate["seed_values"] = list(self.aggregate_metrics.seed_values)
            aggregate["tool_ablation_variants"] = list(
                self.aggregate_metrics.tool_ablation_variants
            )
            aggregate["decision_quality"] = {
                **aggregate.get("decision_quality", {}),
                "no_action_rate": self.aggregate_metrics.decision_quality.no_action_rate,
                "replan_rate": self.aggregate_metrics.decision_quality.replan_rate,
                "intervention_rate": self.aggregate_metrics.decision_quality.intervention_rate,
                "missed_intervention_count": (
                    self.aggregate_metrics.decision_quality.missed_intervention_count
                ),
                "unnecessary_intervention_count": (
                    self.aggregate_metrics.decision_quality.unnecessary_intervention_count
                ),
            }
            aggregate["telemetry_metrics"] = jsonable(self.aggregate_metrics.telemetry_metrics)
            aggregate["schedule_breakdown"] = [
                jsonable(item) for item in self.aggregate_metrics.schedule_breakdown
            ]
            aggregate["ablation_breakdown"] = [
                jsonable(item) for item in self.aggregate_metrics.ablation_breakdown
            ]
            if self.aggregate_metrics.external_evidence_summary is not None:
                aggregate["external_evidence_summary"] = jsonable(
                    self.aggregate_metrics.external_evidence_summary
                )
            aggregate["rubric_summary"] = {
                "validity": jsonable(self.aggregate_metrics.validity_summary),
                "decision_quality": jsonable(self.aggregate_metrics.decision_quality),
                "performance": jsonable(self.aggregate_metrics.performance_summary),
                "telemetry": jsonable(self.aggregate_metrics.telemetry_metrics),
                "tool_use": jsonable(self.aggregate_metrics.tool_use_summary),
                "robustness": jsonable(self.aggregate_metrics.robustness_summary),
            }
            aggregate["calibration_summary"] = {
                "calibration_layer_application_count": (
                    self.aggregate_metrics.tool_use_summary.moderated_update_count
                ),
                "repeated_stress_moderation_count": (
                    self.aggregate_metrics.tool_use_summary.repeated_stress_moderation_count
                ),
                "relapse_moderation_count": (
                    self.aggregate_metrics.tool_use_summary.relapse_moderation_count
                ),
                "unresolved_stress_moderation_count": (
                    self.aggregate_metrics.tool_use_summary.unresolved_stress_moderation_count
                ),
                "moderated_update_count": (
                    self.aggregate_metrics.tool_use_summary.moderated_update_count
                ),
                "hysteresis_application_count": (
                    self.aggregate_metrics.tool_use_summary.hysteresis_application_count
                ),
            }
            aggregate["ablation_quality"] = [
                {
                    "tool_ablation_variant": item.tool_ablation_variant,
                    "unavailable_tool_request_count": (
                        item.tool_use_summary.unavailable_tool_request_count
                    ),
                    "disabled_tool_fallback_count": (
                        item.tool_use_summary.disabled_tool_fallback_count
                    ),
                    "sequencing_blocked_tool_request_count": (
                        item.tool_use_summary.sequencing_blocked_tool_request_count
                    ),
                    "clean_intervention_count": (
                        item.tool_use_summary.clean_intervention_count
                    ),
                    "clean_optimizer_input_change_count": (
                        item.tool_use_summary.clean_optimizer_input_change_count
                    ),
                    "regime_prediction_accuracy": (
                        item.decision_quality.regime_prediction_accuracy
                    ),
                    "replan_rate": item.decision_quality.replan_rate,
                    "average_total_cost": item.performance_summary.average_total_cost,
                }
                for item in self.aggregate_metrics.ablation_breakdown
            ]
        if self.tool_usage_by_ablation:
            aggregate["tool_usage_by_ablation"] = [
                jsonable(summary) for summary in self.tool_usage_by_ablation
            ]
        if self.tool_ablation_summaries:
            aggregate["tool_ablation_summaries"] = [
                jsonable(summary) for summary in self.tool_ablation_summaries
            ]
        return {
            "mode": self.mode,
            "aggregate": aggregate,
            "runs": [run.to_summary() for run in self.runs],
        }


@dataclass(frozen=True, slots=True)
class StockpylSerialComparisonBatch:
    """Repeated-run comparison across deterministic and LLM paths."""

    benchmark_case: SerialBenchmarkCase
    schedule_names: tuple[str, ...]
    seed_values: tuple[int, ...]
    tool_ablation_variants: tuple[str, ...]
    deterministic_baseline: StockpylSerialModeBatch
    deterministic_orchestrator: StockpylSerialModeBatch
    llm_orchestrator: StockpylSerialModeBatch
    outcomes_differ: bool
    optimizer_order_boundary_preserved: bool
    aggregate_results: BatchAggregateSummary | None = None

    def to_summary(self) -> dict[str, object]:
        """Return a compact JSON-serializable repeated-run comparison summary."""

        return {
            "benchmark_id": self.benchmark_case.benchmark_id,
            "benchmark_source": self.benchmark_case.adapter_name,
            "schedule_names": list(self.schedule_names),
            "seed_values": list(self.seed_values),
            "tool_ablation_variants": list(self.tool_ablation_variants),
            "outcomes_differ": self.outcomes_differ,
            "optimizer_order_boundary_preserved": self.optimizer_order_boundary_preserved,
            "artifact_use_class": (
                self.aggregate_results.artifact_use_class.value
                if self.aggregate_results is not None
                else None
            ),
            "validity_gate_passed": (
                self.aggregate_results.validity_gate_passed
                if self.aggregate_results is not None
                else None
            ),
            "eligibility_notes": (
                list(self.aggregate_results.eligibility_notes)
                if self.aggregate_results is not None
                else []
            ),
            "deterministic_baseline": self.deterministic_baseline.to_summary(),
            "deterministic_orchestrator": self.deterministic_orchestrator.to_summary(),
            "llm_orchestrator": self.llm_orchestrator.to_summary(),
            "aggregate_results": (
                jsonable(self.aggregate_results) if self.aggregate_results is not None else None
            ),
        }


@dataclass(frozen=True, slots=True)
class StockpylSerialModeSweep:
    """Generic repeated-run sweep across an explicit configured mode set."""

    benchmark_case: SerialBenchmarkCase
    schedule_names: tuple[str, ...]
    seed_values: tuple[int, ...]
    tool_ablation_variants: tuple[str, ...]
    mode_batches: tuple[StockpylSerialModeBatch, ...]
    outcomes_differ: bool
    optimizer_order_boundary_preserved: bool
    aggregate_results: BatchAggregateSummary | None = None

    def to_summary(self) -> dict[str, object]:
        """Return a compact JSON-serializable repeated-run mode sweep summary."""

        return {
            "benchmark_id": self.benchmark_case.benchmark_id,
            "benchmark_source": self.benchmark_case.adapter_name,
            "schedule_names": list(self.schedule_names),
            "seed_values": list(self.seed_values),
            "tool_ablation_variants": list(self.tool_ablation_variants),
            "mode_names": [batch.mode for batch in self.mode_batches],
            "outcomes_differ": self.outcomes_differ,
            "optimizer_order_boundary_preserved": self.optimizer_order_boundary_preserved,
            "artifact_use_class": (
                self.aggregate_results.artifact_use_class.value
                if self.aggregate_results is not None
                else None
            ),
            "validity_gate_passed": (
                self.aggregate_results.validity_gate_passed
                if self.aggregate_results is not None
                else None
            ),
            "eligibility_notes": (
                list(self.aggregate_results.eligibility_notes)
                if self.aggregate_results is not None
                else []
            ),
            "mode_batches": [batch.to_summary() for batch in self.mode_batches],
            "aggregate_results": (
                jsonable(self.aggregate_results) if self.aggregate_results is not None else None
            ),
        }


def _extract_scenario_update_result(response: OrchestrationResponse) -> ScenarioUpdateResult | None:
    for tool_result in reversed(response.tool_results):
        value = tool_result.structured_output.get("scenario_update_result")
        if isinstance(value, ScenarioUpdateResult):
            return value
    return None


def _default_keep_current_update(
    regime_label: RegimeLabel,
    observation,
    provenance: str,
) -> ScenarioUpdateResult:
    return ScenarioUpdateResult(
        scenarios=(
            ScenarioSummary(
                scenario_id=f"{regime_label.value}_keep_current",
                regime_label=regime_label,
                weight=1.0,
                demand_scale=1.0,
                leadtime_scale=1.0,
            ),
        ),
        applied_update_types=(UpdateRequestType.KEEP_CURRENT,),
        adjustment=ScenarioAdjustmentSummary(
            demand_outlook=observation.demand_realization[-1],
            leadtime_outlook=observation.leadtime_realization[-1],
            safety_buffer_scale=1.0,
        ),
        request_replan=False,
        provenance=provenance,
    )


def _optimizer_boundary_preserved(
    responses: tuple[OrchestrationResponse, ...],
) -> bool:
    for response in responses:
        for tool_result in response.tool_results:
            if tool_result.emits_raw_orders:
                return False
            if any(
                isinstance(value, OptimizationResult)
                for value in tool_result.structured_output.values()
            ):
                return False
    return True


def _normalize_llm_client_mode(value: str) -> str:
    return "real" if value == "openai" else value


def _normalize_mode_alias(mode: str) -> str:
    if mode == LLM_INTERNAL_ONLY_MODE:
        return "llm_orchestrator"
    if mode == LLM_WITH_EXTERNAL_EVIDENCE_MODE:
        return "llm_orchestrator"
    return mode


def _mode_uses_llm(mode: str) -> bool:
    return _normalize_mode_alias(mode) == "llm_orchestrator"


def _mode_enables_external_evidence(mode: str) -> bool:
    return mode == LLM_WITH_EXTERNAL_EVIDENCE_MODE


def _resolved_external_evidence_source(experiment_config: ExperimentConfig) -> str | None:
    return experiment_config.resolved_external_evidence_source()


def _build_external_evidence_batch_for_period(
    *,
    experiment_config: ExperimentConfig,
    evaluation_case: EvaluationCase,
    time_index: int,
):
    source = _resolved_external_evidence_source(experiment_config)
    if source == "semi_synthetic":
        return build_external_evidence_batch(
            schedule_name=evaluation_case.schedule_name,
            regime_schedule=evaluation_case.regime_schedule,
            period_index=time_index,
            run_seed=evaluation_case.run_seed,
        )
    return None


def _resolve_agent_config(
    agent_config: AgentConfig,
    *,
    llm_client_mode_override: str | None = None,
    llm_model_name_override: str | None = None,
) -> AgentConfig:
    updates: dict[str, object] = {}
    if llm_client_mode_override is not None and llm_client_mode_override != "config":
        updates["llm_client_mode"] = llm_client_mode_override
    if llm_model_name_override is not None:
        updates["llm_model_name"] = llm_model_name_override
    if not updates:
        return agent_config
    return replace(agent_config, **updates)


def _load_runtime_components(
    experiment_config_path: str | Path,
    *,
    llm_client_mode_override: str | None = None,
    llm_model_name_override: str | None = None,
) -> tuple[ExperimentConfig, AgentConfig, SerialBenchmarkCase]:
    experiment_config = load_experiment_config(experiment_config_path)
    benchmark_config = load_benchmark_config(experiment_config.benchmark_config_path)
    if experiment_config.agent_config_path is None:
        raise ValueError("Stockpyl serial rollout requires an agent configuration path.")
    agent_config = load_agent_config(experiment_config.agent_config_path)
    agent_config = _resolve_agent_config(
        agent_config,
        llm_client_mode_override=llm_client_mode_override,
        llm_model_name_override=llm_model_name_override,
    )
    benchmark_case = build_serial_benchmark_case(
        benchmark_config,
        adapter=StockpylSerialAdapter(),
    )
    return experiment_config, agent_config, benchmark_case


def _candidate_tool_ids_for_ablation(tool_ablation_variant: str) -> tuple[str, ...]:
    try:
        return TOOL_IDS_BY_ABLATION[tool_ablation_variant]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported tool_ablation_variant: {tool_ablation_variant!r}."
        ) from exc


def _candidate_tool_ids_for_mode(
    tool_ablation_variant: str,
    *,
    external_evidence_enabled: bool,
) -> tuple[str, ...]:
    candidate_tool_ids = _candidate_tool_ids_for_ablation(tool_ablation_variant)
    if external_evidence_enabled:
        return candidate_tool_ids + (EXTERNAL_EVIDENCE_TOOL_ID,)
    return candidate_tool_ids


def _build_mission(
    tool_ablation_variant: str = "full",
    *,
    external_evidence_enabled: bool = False,
) -> MissionSpec:
    return MissionSpec(
        mission_id="stockpyl_serial_controlled_rollout",
        objective=(
            "Inspect Stockpyl-backed serial evidence, manage bounded uncertainty, and "
            "keep replenishment orders inside the trusted optimizer boundary."
        ),
        admissible_tool_ids=_candidate_tool_ids_for_mode(
            tool_ablation_variant,
            external_evidence_enabled=external_evidence_enabled,
        ),
    )


def _build_evaluation_cases(
    experiment_config: ExperimentConfig,
    *,
    default_seed: int,
) -> tuple[EvaluationCase, ...]:
    schedule_set = experiment_config.resolved_schedule_set()
    seed_set = experiment_config.resolved_seed_set(default_seed)
    cases: list[EvaluationCase] = []
    case_index = 0
    for schedule in schedule_set:
        for seed in seed_set:
            for tool_ablation_variant in experiment_config.resolved_tool_ablation_variants():
                cases.append(
                    EvaluationCase(
                        case_index=case_index,
                        schedule_name=schedule.name,
                        regime_schedule=schedule.labels,
                        run_seed=seed,
                        tool_ablation_variant=tool_ablation_variant,
                    )
                )
                case_index += 1
    return tuple(cases)


def _limit_evaluation_cases(
    cases: tuple[EvaluationCase, ...],
    *,
    max_runs: int | None,
) -> tuple[EvaluationCase, ...]:
    if max_runs is None:
        return cases
    if max_runs <= 0:
        raise ValueError("max_runs must be positive when provided.")
    return cases[:max_runs]


def _experiment_config_for_case(
    experiment_config: ExperimentConfig,
    evaluation_case: EvaluationCase,
) -> ExperimentConfig:
    return replace(
        experiment_config,
        rollout_horizon=len(evaluation_case.regime_schedule),
        regime_schedule=evaluation_case.regime_schedule,
        regime_schedules=(),
    )


def _benchmark_case_for_seed(
    benchmark_case: SerialBenchmarkCase,
    *,
    run_seed: int,
) -> SerialBenchmarkCase:
    seeded_benchmark_config = replace(benchmark_case.benchmark_config, random_seed=run_seed)
    return build_serial_benchmark_case(
        seeded_benchmark_config,
        adapter=StockpylSerialAdapter(),
    )


def _build_runtime(agent_config: AgentConfig, *, mode: str) -> tuple[OrchestrationRuntime, str | None]:
    normalized_mode = _normalize_mode_alias(mode)
    external_evidence_enabled = _mode_enables_external_evidence(mode)
    base_tools = (
        DeterministicForecastTool(),
        DeterministicLeadTimeTool(),
        DeterministicScenarioTool(),
    )
    if external_evidence_enabled:
        tools = (BoundedExternalEvidenceTool(),) + base_tools
    else:
        tools = base_tools
    if normalized_mode == "deterministic_orchestrator":
        return OrchestrationRuntime(agent_config=agent_config, tools=tools), None
    if normalized_mode == "llm_orchestrator":
        client_mode = _normalize_llm_client_mode(agent_config.llm_client_mode)
        if client_mode == "real":
            if agent_config.llm_provider != "openai":
                raise ValueError(
                    f"Unsupported real LLM provider for this milestone: {agent_config.llm_provider!r}."
                )
            client = OpenAILLMClient(
                provider=agent_config.llm_provider,
                request_timeout_s=agent_config.llm_request_timeout_s,
                max_retries=agent_config.llm_max_retries,
            )
            client.ensure_available()
            llm_provider = agent_config.llm_provider
        else:
            client = FakeLLMClient()
            llm_provider = client.provider
        llm_orchestrator = LLMOrchestrator(client=client, agent_config=agent_config)
        return (
            OrchestrationRuntime(
                agent_config=agent_config,
                tools=tools,
                mode=RuntimeMode.LLM_ORCHESTRATION,
                llm_orchestrator=llm_orchestrator,
            ),
            llm_provider,
        )
    raise ValueError(f"Unsupported runtime mode: {mode!r}.")


def _build_summary(
    mode: str,
    run_id: str,
    benchmark_case: SerialBenchmarkCase,
    trace: EpisodeTrace,
    rollout_metrics: RolloutMetrics,
    episode_telemetry: EpisodeTelemetrySummary | None,
    decision_quality: DecisionQualitySummary,
    optimizer_order_boundary_preserved: bool,
) -> BenchmarkRunSummary:
    return BenchmarkRunSummary(
        run_id=run_id,
        benchmark_id=benchmark_case.benchmark_id,
        benchmark_source=benchmark_case.adapter_name,
        topology=benchmark_case.topology,
        echelon_count=benchmark_case.echelon_count,
        intervention_summary=InterventionSummary(
            tool_call_count=rollout_metrics.total_tool_call_count,
            replan_count=rollout_metrics.total_replan_count,
            abstain_count=rollout_metrics.abstain_count,
            no_action_count=sum(1 for signal in trace.agent_signals if signal.no_action),
        ),
        trace_summary=TraceSummary(
            episode_count=1,
            trace_length=trace.trace_length,
            state_count=len(trace.states),
            observation_count=len(trace.observations),
            agent_signal_count=len(trace.agent_signals),
            optimization_call_count=len(trace.optimization_results),
        ),
        total_cost=rollout_metrics.total_cost,
        fill_rate=rollout_metrics.fill_rate,
        average_inventory=rollout_metrics.average_inventory,
        average_backorder_level=rollout_metrics.average_backorder_level,
        optimizer_order_boundary_preserved=optimizer_order_boundary_preserved,
        episode_telemetry=episode_telemetry,
        decision_quality=decision_quality,
        notes=(
            "stockpyl_serial_controlled_rollout",
            mode,
            "leadtime_queue_rollout_with_explicit_cost_breakdown",
        ),
    )


def _collect_llm_counts(
    responses: tuple[OrchestrationResponse, ...],
) -> tuple[int, int, int]:
    invalid_output_count = 0
    fallback_count = 0
    successful_response_count = 0
    for response in responses:
        if response.llm_diagnostics is None:
            continue
        invalid_output_count += response.llm_diagnostics.invalid_output_count
        fallback_count += response.llm_diagnostics.fallback_count
        successful_response_count += response.llm_diagnostics.successful_response_count
    return invalid_output_count, fallback_count, successful_response_count


def _build_baseline_step_telemetry(
    *,
    signal,
    orchestration_latency_ms: float,
) -> OrchestrationStepTelemetry:
    return OrchestrationStepTelemetry(
        provider=None,
        model_name=None,
        llm_latency_ms=None,
        orchestration_latency_ms=orchestration_latency_ms,
        invalid_output=False,
        fallback_used=False,
        tool_call_count=len(signal.tool_sequence),
        selected_tools=signal.tool_sequence,
        request_replan=signal.request_replan,
        abstain_or_no_action=signal.abstained or signal.no_action,
        estimated_cost=None,
    )


def _cost_breakdown_record(rollout_metrics: RolloutMetrics) -> CostBreakdownRecord:
    return CostBreakdownRecord(
        holding_cost=rollout_metrics.holding_cost,
        backlog_cost=rollout_metrics.backlog_cost,
        ordering_cost=rollout_metrics.ordering_cost,
        other_cost=rollout_metrics.other_cost,
    )


def _scenario_adjustment_record(
    scenario_update_result: ScenarioUpdateResult,
) -> dict[str, float]:
    adjustment = scenario_update_result.adjustment
    return {
        "demand_outlook": adjustment.demand_outlook,
        "leadtime_outlook": adjustment.leadtime_outlook,
        "safety_buffer_scale": adjustment.safety_buffer_scale,
    }


def _decision_changed_optimizer_input(
    optimization_request: OptimizationRequest,
    counterfactual_request: OptimizationRequest,
) -> bool:
    return _optimization_request_snapshot(optimization_request) != _optimization_request_snapshot(
        counterfactual_request
    )


def _optimizer_output_changed_state(
    transition,
    zero_order_transition,
) -> bool:
    return _transition_outcomes_differ(transition, zero_order_transition)


def _intervention_count(step_trace_records: tuple[StepTraceRecord, ...]) -> int:
    return sum(
        1
        for record in step_trace_records
        if record.request_replan
        or bool(record.selected_tools)
        or record.decision_changed_optimizer_input
    )


def _is_intervention_step(
    *,
    request_replan: bool,
    selected_tools: tuple[str, ...],
    decision_changed_optimizer_input: bool,
) -> bool:
    return request_replan or bool(selected_tools) or decision_changed_optimizer_input


def _optimization_request_snapshot(
    request: OptimizationRequest,
) -> dict[str, object]:
    return {
        "inventory_level": list(request.inventory_level),
        "pipeline_inventory": list(request.pipeline_inventory),
        "backorder_level": list(request.backorder_level),
        "base_stock_levels": list(request.base_stock_levels),
        "demand_outlook": request.scenario_adjustment.demand_outlook,
        "leadtime_outlook": request.scenario_adjustment.leadtime_outlook,
        "safety_buffer_scale": request.scenario_adjustment.safety_buffer_scale,
        "scenario_ids": [scenario.scenario_id for scenario in request.scenario_summaries],
        "planning_horizon": request.planning_horizon,
        "time_index": request.time_index,
    }


def _transition_outcomes_differ(left_transition, right_transition) -> bool:
    return (
        left_transition.next_state.inventory_level != right_transition.next_state.inventory_level
        or left_transition.next_state.pipeline_inventory
        != right_transition.next_state.pipeline_inventory
        or left_transition.next_state.backorder_level
        != right_transition.next_state.backorder_level
        or left_transition.served_demand != right_transition.served_demand
        or left_transition.unmet_demand != right_transition.unmet_demand
        or left_transition.outbound_shipments != right_transition.outbound_shipments
        or left_transition.scheduled_arrivals != right_transition.scheduled_arrivals
    )


def _state_has_consistent_pipeline(
    state: SimulationState,
    benchmark_case: SerialBenchmarkCase,
) -> bool:
    if len(state.inventory_level) != benchmark_case.echelon_count:
        return False
    if not state.in_transit_inventory:
        return False
    if len(state.in_transit_inventory) != benchmark_case.echelon_count:
        return False
    expected_lengths = tuple(
        max(0, lead_time) for lead_time in benchmark_case.shipment_lead_times
    )
    actual_lengths = tuple(len(queue) for queue in state.in_transit_inventory)
    if actual_lengths != expected_lengths:
        return False
    derived_pipeline = tuple(
        float(sum(queue)) for queue in state.in_transit_inventory
    )
    return derived_pipeline == state.pipeline_inventory


def _rollout_fidelity_gate_passed(
    trace: EpisodeTrace,
    benchmark_case: SerialBenchmarkCase,
) -> bool:
    if not trace.period_records:
        return False
    for record in trace.period_records:
        if record.next_state is None:
            return False
        if not _state_has_consistent_pipeline(record.state, benchmark_case):
            return False
        if not _state_has_consistent_pipeline(record.next_state, benchmark_case):
            return False
        if len(record.next_state.backorder_level) != benchmark_case.echelon_count:
            return False
    return True


def _operational_metrics_gate_passed(
    summary: BenchmarkRunSummary,
    rollout_metrics: RolloutMetrics,
    step_trace_records: tuple[StepTraceRecord, ...],
) -> bool:
    if any(
        value is None
        for value in (
            summary.total_cost,
            summary.fill_rate,
            summary.average_inventory,
            summary.average_backorder_level,
        )
    ):
        return False
    if any(record.per_period_cost is None for record in step_trace_records):
        return False
    if any(
        (record.per_period_fill_rate is None)
        for record in step_trace_records
        if record.true_regime_label.strip()
    ):
        return False
    total_cost = rollout_metrics.total_cost
    if total_cost is None:
        return False
    expected_total_cost = (
        rollout_metrics.holding_cost
        + rollout_metrics.backlog_cost
        + rollout_metrics.ordering_cost
        + rollout_metrics.other_cost
    )
    if abs(total_cost - expected_total_cost) > 1e-9:
        return False
    per_period_total_cost = sum(record.per_period_cost or 0.0 for record in step_trace_records)
    if abs(total_cost - per_period_total_cost) > 1e-9:
        return False
    fill_rate = summary.fill_rate
    if fill_rate is None or not 0.0 <= fill_rate <= 1.0:
        return False
    return True


def _llm_trace_from_response(response: OrchestrationResponse | None):
    if response is None or response.llm_diagnostics is None:
        return None
    return response.llm_diagnostics.llm_call_trace


def _unavailable_tool_request(response: OrchestrationResponse | None) -> bool:
    trace = _llm_trace_from_response(response)
    return bool(trace is not None and trace.unavailable_tool_ids)


def _disabled_tool_fallback(response: OrchestrationResponse | None) -> bool:
    trace = _llm_trace_from_response(response)
    return bool(
        trace is not None
        and trace.fallback_used
        and trace.unavailable_tool_ids
    )


def _build_step_trace_record(
    *,
    episode_id: str,
    mode: str,
    tool_ablation_variant: str,
    schedule_name: str,
    run_seed: int,
    period_record: PeriodTraceRecord,
    optimization_request: OptimizationRequest,
    counterfactual_request: OptimizationRequest,
    transition,
    counterfactual_transition,
    zero_order_transition,
    response: OrchestrationResponse | None,
    scenario_update_result: ScenarioUpdateResult,
    cost_config,
    holding_cost_by_stage: tuple[float, ...],
    stockout_cost_by_stage: tuple[float, ...],
    external_evidence_source: str | None,
    external_evidence_batch,
) -> StepTraceRecord:
    predicted_regime_label = None
    confidence = None
    proposed_update_requests: tuple[str, ...] = ()
    proposed_update_strength = None
    final_update_strength = None
    calibration_applied = False
    hysteresis_applied = False
    repeated_stress_moderation_applied = False
    relapse_moderation_applied = False
    unresolved_stress_moderation_applied = False
    calibration_reason = None
    moderation_reason = None
    external_evidence_present = (
        external_evidence_batch is not None and external_evidence_batch.record_count > 0
    )
    external_evidence_false_alarm = (
        external_evidence_batch.contains_false_alarm
        if external_evidence_batch is not None
        else False
    )
    external_evidence_role_labels = (
        tuple(
            sorted(
                {
                    record.role_label
                    for record in external_evidence_batch.records
                    if record.role_label is not None
                }
            )
        )
        if external_evidence_batch is not None
        else ()
    )
    external_evidence_tool_used = False
    external_evidence_corroboration_count = 0
    external_evidence_changed_optimizer_input = None
    external_evidence_fusion_cap_applied = False
    corroboration_gate_applied = False
    corroborated_family_change_allowed = False
    proposed_external_strengthening = None
    final_external_strengthening = None
    external_evidence_fusion_cap_reason = None
    corroboration_gate_reason = None
    early_evidence_confirmation_gate_applied = False
    early_evidence_family_change_blocked = False
    proposed_external_update_requests: tuple[str, ...] = ()
    final_external_update_requests: tuple[str, ...] = ()
    early_evidence_confirmation_gate_reason = None
    if response is not None and response.agent_assessment is not None:
        predicted_regime_label = response.agent_assessment.regime_label.value
        confidence = response.agent_assessment.confidence
    if response is not None and response.llm_diagnostics is not None:
        proposed_update_requests = response.llm_diagnostics.proposed_update_request_types
        proposed_update_strength = response.llm_diagnostics.proposed_update_strength
        final_update_strength = response.llm_diagnostics.final_update_strength
        calibration_applied = response.llm_diagnostics.calibration_applied
        hysteresis_applied = response.llm_diagnostics.hysteresis_applied
        repeated_stress_moderation_applied = (
            response.llm_diagnostics.repeated_stress_moderation_applied
        )
        relapse_moderation_applied = (
            response.llm_diagnostics.relapse_moderation_applied
        )
        unresolved_stress_moderation_applied = (
            response.llm_diagnostics.unresolved_stress_moderation_applied
        )
        calibration_reason = response.llm_diagnostics.calibration_reason
        moderation_reason = response.llm_diagnostics.moderation_reason
    if response is not None:
        external_tool_traces = tuple(
            trace for trace in response.tool_call_traces if trace.tool_id == EXTERNAL_EVIDENCE_TOOL_ID
        )
        external_evidence_tool_used = bool(external_tool_traces)
        if external_tool_traces:
            external_evidence_changed_optimizer_input = any(
                trace.optimizer_input_changed is True for trace in external_tool_traces
            )
    fusion_summary = scenario_update_result.adjustment.external_evidence_fusion_summary
    if fusion_summary is not None:
        external_evidence_corroboration_count = fusion_summary.corroboration_count
        external_evidence_fusion_cap_applied = fusion_summary.cap_applied
        corroboration_gate_applied = fusion_summary.corroboration_gate_applied
        corroborated_family_change_allowed = (
            fusion_summary.corroborated_family_change_allowed
        )
        proposed_external_strengthening = {
            "demand_outlook": fusion_summary.proposed_demand_outlook,
            "leadtime_outlook": fusion_summary.proposed_leadtime_outlook,
            "safety_buffer_scale": fusion_summary.proposed_safety_buffer_scale,
        }
        final_external_strengthening = {
            "demand_outlook": fusion_summary.final_demand_outlook,
            "leadtime_outlook": fusion_summary.final_leadtime_outlook,
            "safety_buffer_scale": fusion_summary.final_safety_buffer_scale,
        }
        external_evidence_fusion_cap_reason = fusion_summary.reason
        corroboration_gate_reason = fusion_summary.corroboration_gate_reason
        if fusion_summary.role_labels:
            external_evidence_role_labels = fusion_summary.role_labels
        early_evidence_confirmation_gate_applied = (
            fusion_summary.early_confirmation_gate_applied
        )
        early_evidence_family_change_blocked = fusion_summary.family_change_blocked
        proposed_external_update_requests = tuple(
            update_type.value for update_type in fusion_summary.proposed_update_request_types
        )
        final_external_update_requests = tuple(
            update_type.value for update_type in fusion_summary.final_update_request_types
        )
        early_evidence_confirmation_gate_reason = (
            fusion_summary.early_confirmation_gate_reason
        )
    unavailable_tool_request = _unavailable_tool_request(response)
    disabled_tool_fallback = _disabled_tool_fallback(response)
    sequencing_blocked_tool_request_count = _sequencing_blocked_tool_request_count(
        response
    )
    decision_changed_optimizer_input = _decision_changed_optimizer_input(
        optimization_request,
        counterfactual_request,
    )
    clean_intervention = _is_intervention_step(
        request_replan=period_record.agent_signal.request_replan,
        selected_tools=period_record.agent_signal.tool_sequence,
        decision_changed_optimizer_input=decision_changed_optimizer_input,
    ) and not unavailable_tool_request and sequencing_blocked_tool_request_count == 0
    external_evidence_supported_intervention = external_evidence_present and (
        clean_intervention or external_evidence_tool_used
    )
    optimizer_output_changed_state = _optimizer_output_changed_state(
        transition,
        zero_order_transition,
    )
    return StepTraceRecord(
        episode_id=episode_id,
        mode=mode,
        tool_ablation_variant=tool_ablation_variant,
        schedule_name=schedule_name,
        run_seed=run_seed,
        period_index=period_record.time_index,
        true_regime_label=period_record.regime_label.value,
        predicted_regime_label=predicted_regime_label,
        confidence=confidence,
        selected_subgoal=period_record.agent_signal.selected_subgoal.value,
        selected_tools=period_record.agent_signal.tool_sequence,
        proposed_update_requests=proposed_update_requests,
        proposed_update_strength=proposed_update_strength,
        final_update_strength=final_update_strength,
        calibration_applied=calibration_applied,
        hysteresis_applied=hysteresis_applied,
        update_requests=tuple(
            update_type.value for update_type in scenario_update_result.applied_update_types
        ),
        request_replan=period_record.agent_signal.request_replan,
        abstain_or_no_action=(
            period_record.agent_signal.abstained or period_record.agent_signal.no_action
        ),
        demand_outlook=scenario_update_result.adjustment.demand_outlook,
        leadtime_outlook=scenario_update_result.adjustment.leadtime_outlook,
        scenario_adjustment_summary=_scenario_adjustment_record(scenario_update_result),
        optimizer_orders=period_record.optimization_result.replenishment_orders,
        inventory_by_echelon=period_record.state.inventory_level,
        pipeline_by_echelon=period_record.state.pipeline_inventory,
        backorders_by_echelon=period_record.state.backorder_level,
        per_period_cost=compute_period_total_cost(
            period_record,
            cost_config,
            holding_cost_by_stage=holding_cost_by_stage,
            stockout_cost_by_stage=stockout_cost_by_stage,
        ),
        per_period_fill_rate=compute_period_fill_rate(period_record),
        decision_changed_optimizer_input=decision_changed_optimizer_input,
        optimizer_output_changed_state=optimizer_output_changed_state,
        intervention_changed_outcome=_transition_outcomes_differ(
            transition,
            counterfactual_transition,
        ),
        unavailable_tool_request=unavailable_tool_request,
        disabled_tool_fallback=disabled_tool_fallback,
        sequencing_blocked_tool_request_count=sequencing_blocked_tool_request_count,
        clean_intervention=clean_intervention,
        repeated_stress_moderation_applied=repeated_stress_moderation_applied,
        relapse_moderation_applied=relapse_moderation_applied,
        unresolved_stress_moderation_applied=unresolved_stress_moderation_applied,
        calibration_reason=calibration_reason,
        moderation_reason=moderation_reason,
        external_evidence_source=external_evidence_source,
        external_evidence_present=external_evidence_present,
        external_evidence_false_alarm=external_evidence_false_alarm,
        external_evidence_tool_used=external_evidence_tool_used,
        external_evidence_supported_intervention=external_evidence_supported_intervention,
        external_evidence_role_labels=external_evidence_role_labels,
        external_evidence_corroboration_count=external_evidence_corroboration_count,
        external_evidence_changed_optimizer_input=external_evidence_changed_optimizer_input,
        external_evidence_fusion_cap_applied=external_evidence_fusion_cap_applied,
        corroboration_gate_applied=corroboration_gate_applied,
        corroborated_family_change_allowed=corroborated_family_change_allowed,
        proposed_external_strengthening=proposed_external_strengthening,
        final_external_strengthening=final_external_strengthening,
        external_evidence_fusion_cap_reason=external_evidence_fusion_cap_reason,
        corroboration_gate_reason=corroboration_gate_reason,
        early_evidence_confirmation_gate_applied=(
            early_evidence_confirmation_gate_applied
        ),
        early_evidence_family_change_blocked=early_evidence_family_change_blocked,
        proposed_external_update_requests=proposed_external_update_requests,
        final_external_update_requests=final_external_update_requests,
        early_evidence_confirmation_gate_reason=(
            early_evidence_confirmation_gate_reason
        ),
        validation_lane=VALIDATION_LANE,
    )


def _build_llm_call_trace_records(
    *,
    episode_id: str,
    mode: str,
    tool_ablation_variant: str,
    schedule_name: str,
    run_seed: int,
    period_index: int,
    response: OrchestrationResponse,
    external_evidence_batch,
    external_evidence_enabled: bool,
) -> tuple[LLMCallTraceRecord, ...]:
    if (
        response.llm_diagnostics is None
        or response.llm_diagnostics.llm_call_trace is None
    ):
        return ()
    trace = response.llm_diagnostics.llm_call_trace
    return (
        LLMCallTraceRecord(
            episode_id=episode_id,
            mode=mode,
            tool_ablation_variant=tool_ablation_variant,
            schedule_name=schedule_name,
            run_seed=run_seed,
            period_index=period_index,
            call_index=0,
            provider=trace.provider,
            model_name=trace.model_name,
            prompt_version=trace.prompt_version,
            prompt_text=trace.prompt_text,
            prompt_hash=trace.prompt_hash,
            raw_output_text=trace.raw_output_text,
            parsed_output=trace.parsed_output,
            validation_success=trace.validation_success,
            invalid_output=trace.invalid_output,
            fallback_used=trace.fallback_used,
            fallback_reason=trace.fallback_reason,
            requested_tool_ids=trace.requested_tool_ids,
            unavailable_tool_ids=trace.unavailable_tool_ids,
            violated_available_tool_set=bool(trace.unavailable_tool_ids),
            proposed_update_strength=(
                response.llm_diagnostics.proposed_update_strength
            ),
            final_update_strength=(
                response.llm_diagnostics.final_update_strength
            ),
            calibration_applied=response.llm_diagnostics.calibration_applied,
            hysteresis_applied=response.llm_diagnostics.hysteresis_applied,
            final_update_request_types=response.llm_diagnostics.final_update_request_types,
            repeated_stress_moderation_applied=(
                response.llm_diagnostics.repeated_stress_moderation_applied
            ),
            relapse_moderation_applied=(
                response.llm_diagnostics.relapse_moderation_applied
            ),
            unresolved_stress_moderation_applied=(
                response.llm_diagnostics.unresolved_stress_moderation_applied
            ),
            calibration_reason=response.llm_diagnostics.calibration_reason,
            moderation_reason=response.llm_diagnostics.moderation_reason,
            prompt_tokens=trace.prompt_tokens,
            completion_tokens=trace.completion_tokens,
            total_tokens=trace.total_tokens,
            latency_ms=trace.latency_ms,
            retry_count=trace.retry_count,
            client_error_category=(
                trace.client_error_category.value
                if trace.client_error_category is not None
                else None
            ),
            client_error_message=trace.client_error_message,
            failure_after_response=trace.failure_after_response,
            external_evidence_present=bool(
                external_evidence_batch is not None and external_evidence_batch.record_count > 0
            ),
            external_evidence_tool_available=external_evidence_enabled,
            validation_lane=VALIDATION_LANE,
        ),
    )


def _build_tool_call_trace_records(
    *,
    episode_id: str,
    mode: str,
    tool_ablation_variant: str,
    schedule_name: str,
    run_seed: int,
    period_index: int,
    response: OrchestrationResponse,
) -> tuple[ToolCallTraceRecord, ...]:
    return tuple(
        ToolCallTraceRecord(
            episode_id=episode_id,
            mode=mode,
            tool_ablation_variant=tool_ablation_variant,
            schedule_name=schedule_name,
            run_seed=run_seed,
            period_index=period_index,
            call_index=call_index,
            tool_id=trace.tool_id,
            tool_input=trace.tool_input,
            tool_output=trace.tool_output,
            pre_tool_decision=trace.pre_tool_decision,
            post_tool_decision=trace.post_tool_decision,
            pre_tool_optimizer_input=trace.pre_tool_optimizer_input,
            post_tool_optimizer_input=trace.post_tool_optimizer_input,
            decision_changed=trace.decision_changed,
            optimizer_input_changed=trace.optimizer_input_changed,
            success=trace.success,
            error_type=trace.error_type,
            latency_ms=trace.latency_ms,
            validation_lane=VALIDATION_LANE,
        )
        for call_index, trace in enumerate(response.tool_call_traces)
    )


def _git_commit_sha() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def _single_prompt_version(values: tuple[str | None, ...]) -> str | None:
    normalized = tuple(value for value in values if value is not None)
    if not normalized:
        return None
    unique_values = sorted(set(normalized))
    if len(unique_values) == 1:
        return unique_values[0]
    return "mixed"


def _single_prompt_hash(values: tuple[str | None, ...]) -> str | None:
    normalized = tuple(value for value in values if value is not None)
    if not normalized:
        return None
    unique_values = sorted(set(normalized))
    if len(unique_values) == 1:
        return unique_values[0]
    return None


def _validation_failure_counts_from_runs(
    runs: tuple[StockpylSerialRun, ...],
) -> tuple[tuple[str, int], ...]:
    counter: Counter[str] = Counter()
    for run in runs:
        for reason, count in run.validation_failure_counts:
            counter[reason] += count
    return tuple(sorted(counter.items()))


def _count_unavailable_tool_requests(
    llm_call_trace_records: tuple[LLMCallTraceRecord, ...],
) -> int:
    return sum(
        1
        for record in llm_call_trace_records
        if record.violated_available_tool_set
    )


def _count_disabled_tool_fallbacks(
    llm_call_trace_records: tuple[LLMCallTraceRecord, ...],
) -> int:
    return sum(
        1
        for record in llm_call_trace_records
        if record.fallback_used and record.violated_available_tool_set
    )


def _count_clean_interventions(
    step_trace_records: tuple[StepTraceRecord, ...],
) -> int:
    return sum(1 for record in step_trace_records if record.clean_intervention)


def _count_clean_optimizer_input_changes(
    step_trace_records: tuple[StepTraceRecord, ...],
) -> int:
    return sum(
        1
        for record in step_trace_records
        if (
            record.decision_changed_optimizer_input
            and not record.unavailable_tool_request
            and record.sequencing_blocked_tool_request_count == 0
        )
    )


def _count_repeated_stress_moderations(
    step_trace_records: tuple[StepTraceRecord, ...],
) -> int:
    return sum(
        1
        for record in step_trace_records
        if record.repeated_stress_moderation_applied
    )


def _count_relapse_moderations(
    step_trace_records: tuple[StepTraceRecord, ...],
) -> int:
    return sum(
        1
        for record in step_trace_records
        if record.relapse_moderation_applied
    )


def _count_unresolved_stress_moderations(
    step_trace_records: tuple[StepTraceRecord, ...],
) -> int:
    return sum(
        1
        for record in step_trace_records
        if record.unresolved_stress_moderation_applied
    )


def _count_moderated_updates(
    step_trace_records: tuple[StepTraceRecord, ...],
) -> int:
    return sum(
        1
        for record in step_trace_records
        if record.proposed_update_requests
        and record.proposed_update_requests != record.update_requests
    )


def _count_hysteresis_applications(
    step_trace_records: tuple[StepTraceRecord, ...],
) -> int:
    return sum(1 for record in step_trace_records if record.hysteresis_applied)


def _count_external_evidence_periods(
    step_trace_records: tuple[StepTraceRecord, ...],
) -> int:
    return sum(1 for record in step_trace_records if record.external_evidence_present)


def _count_false_alarm_evidence_periods(
    step_trace_records: tuple[StepTraceRecord, ...],
) -> int:
    return sum(1 for record in step_trace_records if record.external_evidence_false_alarm)


def _count_evidence_supported_interventions(
    step_trace_records: tuple[StepTraceRecord, ...],
) -> int:
    return sum(
        1 for record in step_trace_records if record.external_evidence_supported_intervention
    )


def _count_external_evidence_optimizer_input_changes(
    step_trace_records: tuple[StepTraceRecord, ...],
) -> int:
    return sum(
        1
        for record in step_trace_records
        if record.external_evidence_changed_optimizer_input is True
    )


def _count_external_evidence_fusion_caps(
    step_trace_records: tuple[StepTraceRecord, ...],
) -> int:
    return sum(
        1
        for record in step_trace_records
        if record.external_evidence_fusion_cap_applied
    )


def _count_capped_external_strengthening(
    step_trace_records: tuple[StepTraceRecord, ...],
) -> int:
    return sum(
        1
        for record in step_trace_records
        if record.proposed_external_strengthening is not None
        and record.final_external_strengthening is not None
        and record.proposed_external_strengthening != record.final_external_strengthening
    )


def _count_early_evidence_confirmation_gates(
    step_trace_records: tuple[StepTraceRecord, ...],
) -> int:
    return sum(
        1
        for record in step_trace_records
        if record.early_evidence_confirmation_gate_applied
    )


def _count_early_evidence_family_change_blocks(
    step_trace_records: tuple[StepTraceRecord, ...],
) -> int:
    return sum(
        1
        for record in step_trace_records
        if record.early_evidence_family_change_blocked
    )


def _count_corroboration_gates(
    step_trace_records: tuple[StepTraceRecord, ...],
) -> int:
    return sum(1 for record in step_trace_records if record.corroboration_gate_applied)


def _count_corroborated_family_changes(
    step_trace_records: tuple[StepTraceRecord, ...],
) -> int:
    return sum(
        1 for record in step_trace_records if record.corroborated_family_change_allowed
    )


def _external_evidence_role_usage_counts(
    step_trace_records: tuple[StepTraceRecord, ...],
) -> tuple[tuple[str, int], ...]:
    counter: Counter[str] = Counter()
    for record in step_trace_records:
        counter.update(record.external_evidence_role_labels)
    return tuple(sorted(counter.items()))


def _count_external_evidence_tool_calls(
    tool_call_trace_records: tuple[ToolCallTraceRecord, ...],
) -> int:
    return sum(
        1
        for record in tool_call_trace_records
        if record.tool_id == EXTERNAL_EVIDENCE_TOOL_ID
    )


def _sequencing_blocked_tool_request_count(
    response: OrchestrationResponse | None,
) -> int:
    if response is None:
        return 0
    return sum(
        1
        for check in response.admissibility_checks
        if (
            not check.allowed
            and check.reason == "Tool does not support the requested operational subgoal."
        )
    )


def _count_sequencing_blocked_tool_requests(
    step_trace_records: tuple[StepTraceRecord, ...],
) -> int:
    return sum(
        record.sequencing_blocked_tool_request_count for record in step_trace_records
    )


def _client_error_counts_from_runs(
    runs: tuple[StockpylSerialRun, ...],
) -> tuple[tuple[str, int], ...]:
    counter: Counter[str] = Counter()
    for run in runs:
        for reason, count in run.client_error_counts:
            counter[reason] += count
    return tuple(sorted(counter.items()))


def _prompt_metadata_from_responses(
    responses: tuple[OrchestrationResponse, ...],
) -> tuple[str | None, str | None]:
    prompt_versions = tuple(
        response.llm_diagnostics.prompt_version
        for response in responses
        if response.llm_diagnostics is not None
    )
    prompt_hashes = tuple(
        response.llm_diagnostics.prompt_contract_hash
        for response in responses
        if response.llm_diagnostics is not None
    )
    return _single_prompt_version(prompt_versions), _single_prompt_hash(prompt_hashes)


def _run_group_id(experiment_name: str, mode: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{experiment_name}_{mode}_{timestamp}"


def _resolved_config_record(
    experiment_config: ExperimentConfig,
    agent_config: AgentConfig,
    benchmark_case: SerialBenchmarkCase,
    *,
    mode: str,
    llm_client_mode_override: str | None,
    llm_model_name_override: str | None,
    max_runs: int | None,
) -> dict[str, object]:
    return {
        "mode": mode,
        "experiment": jsonable(experiment_config),
        "agent": jsonable(agent_config),
        "benchmark": jsonable(benchmark_case.benchmark_config),
        "llm_client_mode_override": llm_client_mode_override,
        "llm_model_name_override": llm_model_name_override,
        "max_runs": max_runs,
    }


def _write_logging_artifacts(
    *,
    experiment_config: ExperimentConfig,
    agent_config: AgentConfig,
    benchmark_case: SerialBenchmarkCase,
    runs: tuple[StockpylSerialRun, ...],
    mode: str,
    llm_client_mode_override: str | None,
    llm_model_name_override: str | None,
    max_runs: int | None,
    output_dir_override: Path | None = None,
) -> tuple[ExperimentMetadata, Path, dict[str, Path]]:
    run_group_id = _run_group_id(experiment_config.experiment_name, mode)
    output_root = output_dir_override or experiment_config.results_dir
    output_dir = ensure_output_dir(output_root, run_group_id)
    mode_names = tuple(sorted({run.mode for run in runs}))
    tool_ablation_variants = tuple(
        sorted({run.tool_ablation_variant for run in runs})
    )
    schedule_names = tuple(
        sorted(
            {
                run.schedule_name
                for run in runs
                if run.schedule_name is not None
            }
        )
    )
    seed_values = tuple(sorted({run.run_seed for run in runs}))
    llm_call_records = tuple(
        record
        for run in runs
        for record in run.llm_call_trace_records
    )
    provider_values = tuple(run.llm_provider for run in runs)
    model_values = tuple(run.llm_model_name for run in runs)
    has_llm_runs = any(value is not None for value in provider_values)
    prompt_version = _single_prompt_version(tuple(run.prompt_version for run in runs))
    prompt_hash = _single_prompt_hash(tuple(run.prompt_contract_hash for run in runs))
    if has_llm_runs and prompt_version is None:
        prompt_version = PROMPT_VERSION
    if has_llm_runs and prompt_hash is None and prompt_version == PROMPT_VERSION:
        prompt_hash = prompt_contract_hash()
    resolved_config = _resolved_config_record(
        experiment_config,
        agent_config,
        benchmark_case,
        mode=mode,
        llm_client_mode_override=llm_client_mode_override,
        llm_model_name_override=llm_model_name_override,
        max_runs=max_runs,
    )
    config_hash = hash_jsonable(resolved_config)
    episode_summary_records = tuple(
        run.episode_summary_record
        for run in runs
        if run.episode_summary_record is not None
    )
    step_trace_records = tuple(
        record
        for run in runs
        for record in run.step_trace_records
    )
    tool_call_records = tuple(
        record
        for run in runs
        for record in run.tool_call_trace_records
    )
    records_by_mode: dict[str, tuple[EpisodeSummaryRecord, ...]] = {}
    step_trace_records_by_mode: dict[str, tuple[StepTraceRecord, ...]] = {}
    tool_call_records_by_mode: dict[str, tuple[ToolCallTraceRecord, ...]] = {}
    for run_mode in sorted({run.mode for run in runs}):
        records_by_mode[run_mode] = tuple(
            record for record in episode_summary_records if record.mode == run_mode
        )
        step_trace_records_by_mode[run_mode] = tuple(
            record for record in step_trace_records if record.mode == run_mode
        )
        tool_call_records_by_mode[run_mode] = tuple(
            record for record in tool_call_records if record.mode == run_mode
        )
    aggregate_summary = aggregate_batch_episode_summaries(
        benchmark_id=benchmark_case.benchmark_id,
        validation_lane=VALIDATION_LANE,
        records_by_mode=records_by_mode,
        step_trace_records_by_mode=step_trace_records_by_mode,
        tool_call_records_by_mode=tool_call_records_by_mode,
    )
    mode_governance_decisions = []
    classified_mode_summaries = []
    for mode_summary in aggregate_summary.mode_summaries:
        mode_runs = tuple(run for run in runs if run.mode == mode_summary.mode)
        mode_provider = _value_or_mixed_with_missing(
            tuple(run.llm_provider for run in mode_runs)
        )
        mode_validity_gate_passed = compute_validity_gate_passed(
            optimizer_order_boundary_preserved=(
                mode_summary.validity_summary.optimizer_order_boundary_preserved
            ),
            invalid_output_count=mode_summary.validity_summary.invalid_output_count,
            fallback_count=mode_summary.validity_summary.fallback_count,
        )
        mode_governance = classify_artifact_governance(
            experiment_name=experiment_config.experiment_name,
            benchmark_source=benchmark_case.adapter_name,
            provider=mode_provider,
            validity_gate_passed=mode_validity_gate_passed,
            rollout_fidelity_gate_passed=(
                mode_summary.validity_summary.rollout_fidelity_gate_passed
            ),
            operational_metrics_gate_passed=(
                mode_summary.validity_summary.operational_metrics_gate_passed
            ),
            semi_synthetic_external_evidence=(
                experiment_config.semi_synthetic_external_evidence
            ),
            external_evidence_source=_resolved_external_evidence_source(
                experiment_config
            ),
        )
        mode_governance_decisions.append(mode_governance)
        classified_mode_summaries.append(
            replace(
                mode_summary,
                artifact_use_class=mode_governance.artifact_use_class,
                validity_gate_passed=mode_governance.validity_gate_passed,
                eligibility_notes=mode_governance.eligibility_notes,
            )
        )
    directory_governance = summarize_directory_governance(
        tuple(mode_governance_decisions)
    )
    aggregate_summary = replace(
        aggregate_summary,
        mode_summaries=tuple(classified_mode_summaries),
        artifact_use_class=directory_governance.artifact_use_class,
        validity_gate_passed=directory_governance.validity_gate_passed,
        eligibility_notes=directory_governance.eligibility_notes,
    )
    experiment_metadata = ExperimentMetadata(
        experiment_id=f"{experiment_config.experiment_name}_{mode}",
        run_group_id=run_group_id,
        timestamp=utc_timestamp(),
        git_commit_sha=_git_commit_sha(),
        resolved_config=resolved_config,
        config_hash=config_hash,
        benchmark_id=benchmark_case.benchmark_id,
        benchmark_source=benchmark_case.adapter_name,
        mode=mode,
        validation_lane=VALIDATION_LANE,
        provider=_value_or_mixed_with_missing(provider_values),
        model_name=_value_or_mixed_with_missing(model_values),
        external_evidence_source=_resolved_external_evidence_source(experiment_config),
        tool_ablation_variants=tool_ablation_variants,
        artifact_use_class=directory_governance.artifact_use_class,
        validity_gate_passed=directory_governance.validity_gate_passed,
        rollout_fidelity_gate_passed=all(
            mode_summary.validity_summary.rollout_fidelity_gate_passed
            for mode_summary in aggregate_summary.mode_summaries
        ),
        operational_metrics_gate_passed=all(
            mode_summary.validity_summary.operational_metrics_gate_passed
            for mode_summary in aggregate_summary.mode_summaries
        ),
        eligibility_notes=directory_governance.eligibility_notes,
        prompt_version=prompt_version,
        prompt_hash=prompt_hash,
        benchmark_random_seed=benchmark_case.benchmark_config.random_seed,
    )
    written_files = {
        "experiment_metadata": write_experiment_metadata_json(output_dir, experiment_metadata),
        "episode_summaries": write_episode_summaries_jsonl(output_dir, episode_summary_records),
        "step_traces": write_step_traces_jsonl(output_dir, step_trace_records),
        "llm_call_traces": write_llm_call_traces_jsonl(output_dir, llm_call_records),
        "tool_call_traces": write_tool_call_traces_jsonl(output_dir, tool_call_records),
        "aggregate_summary": write_json(Path(output_dir) / "aggregate_summary.json", aggregate_summary),
    }
    run_manifest = RunManifestRecord(
        experiment_id=experiment_metadata.experiment_id,
        run_group_id=experiment_metadata.run_group_id,
        config_hash=experiment_metadata.config_hash,
        benchmark_id=experiment_metadata.benchmark_id,
        benchmark_source=experiment_metadata.benchmark_source,
        validation_lane=experiment_metadata.validation_lane,
        benchmark_config_path=str(experiment_config.benchmark_config_path),
        agent_config_path=(
            str(experiment_config.agent_config_path)
            if experiment_config.agent_config_path is not None
            else None
        ),
        schedule_names=schedule_names,
        seed_values=seed_values,
        mode_names=mode_names,
        tool_ablation_variants=tool_ablation_variants,
        provider=experiment_metadata.provider,
        model_name=experiment_metadata.model_name,
        external_evidence_source=experiment_metadata.external_evidence_source,
        artifact_use_class=experiment_metadata.artifact_use_class,
        validity_gate_passed=experiment_metadata.validity_gate_passed,
        rollout_fidelity_gate_passed=experiment_metadata.rollout_fidelity_gate_passed,
        operational_metrics_gate_passed=(
            experiment_metadata.operational_metrics_gate_passed
        ),
        eligibility_notes=experiment_metadata.eligibility_notes,
        mode_artifact_use_classes=tuple(
            (mode_summary.mode, mode_summary.artifact_use_class.value)
            for mode_summary in aggregate_summary.mode_summaries
        ),
        prompt_version=experiment_metadata.prompt_version,
        prompt_hash=experiment_metadata.prompt_hash,
        artifact_filenames=tuple(
            (
                *tuple((key, path.name) for key, path in written_files.items()),
                ("run_manifest", "run_manifest.json"),
            )
        ),
    )
    written_files["run_manifest"] = write_run_manifest_json(output_dir, run_manifest)
    return experiment_metadata, output_dir, written_files


def _single_value_or_mixed(values: list[str] | tuple[str, ...]) -> str | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return "mixed"


def _value_or_mixed_with_missing(values: tuple[str | None, ...]) -> str | None:
    normalized = {value for value in values}
    if normalized == {None}:
        return None
    normalized.discard(None)
    if len(normalized) == 1 and None not in set(values):
        return next(iter(normalized))
    if len(normalized) == 1 and None in set(values):
        return "mixed"
    if len(normalized) == 1:
        return next(iter(normalized))
    return "mixed"


def _run_orchestrated_rollout(
    experiment_config: ExperimentConfig,
    agent_config: AgentConfig,
    benchmark_case: SerialBenchmarkCase,
    *,
    evaluation_case: EvaluationCase,
    mode: str,
) -> StockpylSerialRun:
    runtime, llm_provider = _build_runtime(agent_config, mode=mode)
    external_evidence_enabled = _mode_enables_external_evidence(mode)
    external_evidence_source = (
        _resolved_external_evidence_source(experiment_config)
        if external_evidence_enabled
        else None
    )
    mission = _build_mission(
        evaluation_case.tool_ablation_variant,
        external_evidence_enabled=external_evidence_enabled,
    )
    candidate_tool_ids = _candidate_tool_ids_for_mode(
        evaluation_case.tool_ablation_variant,
        external_evidence_enabled=external_evidence_enabled,
    )
    optimizer = TrustedOptimizerAdapter()
    episode_id = (
        f"{experiment_config.experiment_name}_{mode}_"
        f"{evaluation_case.tool_ablation_variant}_"
        f"{evaluation_case.schedule_name}_seed_{evaluation_case.run_seed}_"
        f"run_{evaluation_case.case_index}"
    )
    system_state = build_initial_simulation_state(
        benchmark_case,
        regime_label=regime_for_period(evaluation_case.regime_schedule, 0),
    )
    trace = EpisodeTrace(
        run_id=episode_id,
        benchmark_id=benchmark_case.benchmark_id,
    )
    responses: list[OrchestrationResponse] = []
    step_trace_records: list[StepTraceRecord] = []
    llm_call_trace_records: list[LLMCallTraceRecord] = []
    tool_call_trace_records: list[ToolCallTraceRecord] = []
    for time_index in range(experiment_config.resolved_rollout_horizon()):
        regime_label = regime_for_period(evaluation_case.regime_schedule, time_index)
        previous_regime_label = (
            trace.period_records[-1].regime_label if trace.period_records else None
        )
        observation = build_period_observation(
            benchmark_case,
            system_state,
            regime_label,
            previous_regime_label=previous_regime_label,
        )
        external_evidence_batch = None
        if external_evidence_enabled:
            external_evidence_batch = _build_external_evidence_batch_for_period(
                experiment_config=experiment_config,
                evaluation_case=evaluation_case,
                time_index=time_index,
            )
        evidence = build_runtime_evidence(
            benchmark_case,
            observation,
            external_evidence_batch=external_evidence_batch,
        )
        request = build_serial_orchestration_request(
            case=benchmark_case,
            mission=mission,
            system_state=system_state,
            observation=observation,
            evidence=evidence,
            requested_subgoal=OperationalSubgoal.QUERY_UNCERTAINTY,
            candidate_tool_ids=candidate_tool_ids,
            recent_regime_history=_recent_regime_history(trace),
            recent_stress_reference_demand_value=_recent_stress_reference_demand_value(
                trace
            ),
            recent_update_request_history=_recent_update_request_history(
                tuple(responses)
            ),
        )
        response = runtime.run(request)
        scenario_update_result = _extract_scenario_update_result(response)
        if scenario_update_result is None:
            fallback_regime = regime_label
            if response.agent_assessment is not None:
                fallback_regime = response.agent_assessment.regime_label
            scenario_update_result = _default_keep_current_update(
                fallback_regime,
                observation,
                provenance=f"{mode}_keep_current_fallback",
            )
        counterfactual_scenario_update_result = _default_keep_current_update(
            regime_label,
            observation,
            provenance=f"{mode}_counterfactual_keep_current",
        )
        optimization_request = build_optimization_request(
            system_state=system_state,
            scenario_update_result=scenario_update_result,
            base_stock_levels=benchmark_case.base_stock_levels,
            planning_horizon=1,
        )
        counterfactual_request = build_optimization_request(
            system_state=system_state,
            scenario_update_result=counterfactual_scenario_update_result,
            base_stock_levels=benchmark_case.base_stock_levels,
            planning_horizon=1,
        )
        optimization_result = optimizer.solve(optimization_request)
        counterfactual_result = optimizer.solve(counterfactual_request)
        zero_order_result = OptimizationResult(
            replenishment_orders=tuple(0.0 for _ in benchmark_case.stage_names),
            planning_horizon=1,
        )
        next_regime = None
        if time_index + 1 < experiment_config.resolved_rollout_horizon():
            next_regime = regime_for_period(evaluation_case.regime_schedule, time_index + 1)
        transition = advance_serial_state(
            benchmark_case,
            current_state=system_state,
            observation=observation,
            optimization_result=optimization_result,
            next_regime=next_regime,
        )
        counterfactual_transition = advance_serial_state(
            benchmark_case,
            current_state=system_state,
            observation=observation,
            optimization_result=counterfactual_result,
            next_regime=next_regime,
        )
        zero_order_transition = advance_serial_state(
            benchmark_case,
            current_state=system_state,
            observation=observation,
            optimization_result=zero_order_result,
            next_regime=next_regime,
        )
        trace = trace.append_period(
            PeriodTraceRecord(
                time_index=time_index,
                regime_label=regime_label,
                state=system_state,
                observation=observation,
                agent_signal=response.signal,
                optimization_result=optimization_result,
                next_state=transition.next_state,
                realized_demand=transition.realized_demand,
                demand_load=transition.demand_load,
                served_demand=transition.served_demand,
                unmet_demand=transition.unmet_demand,
                step_telemetry=response.step_telemetry,
                notes=transition.notes,
            )
        )
        period_record = trace.period_records[-1]
        step_trace_records.append(
            _build_step_trace_record(
                episode_id=episode_id,
                mode=mode,
                tool_ablation_variant=evaluation_case.tool_ablation_variant,
                schedule_name=evaluation_case.schedule_name,
                run_seed=evaluation_case.run_seed,
                period_record=period_record,
                optimization_request=optimization_request,
                counterfactual_request=counterfactual_request,
                transition=transition,
                counterfactual_transition=counterfactual_transition,
                zero_order_transition=zero_order_transition,
                response=response,
                scenario_update_result=scenario_update_result,
                cost_config=benchmark_case.benchmark_config.costs,
                holding_cost_by_stage=benchmark_case.holding_costs,
                stockout_cost_by_stage=benchmark_case.stockout_costs,
                external_evidence_source=external_evidence_source,
                external_evidence_batch=external_evidence_batch,
            )
        )
        llm_call_trace_records.extend(
            _build_llm_call_trace_records(
                episode_id=episode_id,
                mode=mode,
                tool_ablation_variant=evaluation_case.tool_ablation_variant,
                schedule_name=evaluation_case.schedule_name,
                run_seed=evaluation_case.run_seed,
                period_index=time_index,
                response=response,
                external_evidence_batch=external_evidence_batch,
                external_evidence_enabled=external_evidence_enabled,
            )
        )
        tool_call_trace_records.extend(
            _build_tool_call_trace_records(
                episode_id=episode_id,
                mode=mode,
                tool_ablation_variant=evaluation_case.tool_ablation_variant,
                schedule_name=evaluation_case.schedule_name,
                run_seed=evaluation_case.run_seed,
                period_index=time_index,
                response=response,
            )
        )
        responses.append(response)
        system_state = transition.next_state
    rollout_metrics = compute_rollout_metrics(
        trace,
        benchmark_case.benchmark_config.costs,
        holding_cost_by_stage=benchmark_case.holding_costs,
        stockout_cost_by_stage=benchmark_case.stockout_costs,
    )
    episode_telemetry = summarize_episode_telemetry(trace.step_telemetry)
    decision_quality = compute_decision_quality(tuple(step_trace_records))
    optimizer_preserved = _optimizer_boundary_preserved(tuple(responses))
    summary = _build_summary(
        mode=mode,
        run_id=episode_id,
        benchmark_case=benchmark_case,
        trace=trace,
        rollout_metrics=rollout_metrics,
        episode_telemetry=episode_telemetry,
        decision_quality=decision_quality,
        optimizer_order_boundary_preserved=optimizer_preserved,
    )
    invalid_output_count, fallback_count, successful_response_count = _collect_llm_counts(
        tuple(responses)
    )
    prompt_version, prompt_contract_hash_value = _prompt_metadata_from_responses(tuple(responses))
    validation_failure_counter: Counter[str] = Counter(
        response.llm_diagnostics.validation_failure_reason
        for response in responses
        if response.llm_diagnostics is not None
        and response.llm_diagnostics.validation_failure_reason is not None
    )
    client_error_counts: tuple[tuple[str, int], ...] = ()
    total_retry_count = 0
    failure_before_response_count = 0
    failure_after_response_count = 0
    if summary.episode_telemetry is not None:
        client_error_counts = summary.episode_telemetry.client_error_counts
        total_retry_count = summary.episode_telemetry.total_retry_count
        failure_before_response_count = (
            summary.episode_telemetry.failure_before_response_count
        )
        failure_after_response_count = (
            summary.episode_telemetry.failure_after_response_count
        )
    episode_summary_record = build_episode_summary_record(
        summary,
        mode=mode,
        validation_lane=VALIDATION_LANE,
        tool_ablation_variant=evaluation_case.tool_ablation_variant,
        schedule_name=evaluation_case.schedule_name,
        run_seed=evaluation_case.run_seed,
        regime_schedule=tuple(label.value for label in evaluation_case.regime_schedule),
        cost_breakdown=_cost_breakdown_record(rollout_metrics),
        intervention_count=_intervention_count(tuple(step_trace_records)),
        invalid_output_count=invalid_output_count,
        fallback_count=fallback_count,
        unavailable_tool_request_count=_count_unavailable_tool_requests(
            tuple(llm_call_trace_records)
        ),
        disabled_tool_fallback_count=_count_disabled_tool_fallbacks(
            tuple(llm_call_trace_records)
        ),
        sequencing_blocked_tool_request_count=_count_sequencing_blocked_tool_requests(
            tuple(step_trace_records)
        ),
        clean_intervention_count=_count_clean_interventions(
            tuple(step_trace_records)
        ),
        clean_optimizer_input_change_count=_count_clean_optimizer_input_changes(
            tuple(step_trace_records)
        ),
        repeated_stress_moderation_count=_count_repeated_stress_moderations(
            tuple(step_trace_records)
        ),
        relapse_moderation_count=_count_relapse_moderations(
            tuple(step_trace_records)
        ),
        unresolved_stress_moderation_count=_count_unresolved_stress_moderations(
            tuple(step_trace_records)
        ),
        moderated_update_count=_count_moderated_updates(tuple(step_trace_records)),
        hysteresis_application_count=_count_hysteresis_applications(
            tuple(step_trace_records)
        ),
        external_evidence_source=external_evidence_source,
        external_evidence_period_count=_count_external_evidence_periods(
            tuple(step_trace_records)
        ),
        external_evidence_tool_call_count=_count_external_evidence_tool_calls(
            tuple(tool_call_trace_records)
        ),
        false_alarm_evidence_count=_count_false_alarm_evidence_periods(
            tuple(step_trace_records)
        ),
        evidence_supported_intervention_count=_count_evidence_supported_interventions(
            tuple(step_trace_records)
        ),
        external_evidence_changed_optimizer_input_count=(
            _count_external_evidence_optimizer_input_changes(tuple(step_trace_records))
        ),
        evidence_fusion_cap_count=_count_external_evidence_fusion_caps(
            tuple(step_trace_records)
        ),
        capped_external_strengthening_count=_count_capped_external_strengthening(
            tuple(step_trace_records)
        ),
        early_evidence_confirmation_gate_count=(
            _count_early_evidence_confirmation_gates(tuple(step_trace_records))
        ),
        early_evidence_family_change_block_count=(
            _count_early_evidence_family_change_blocks(tuple(step_trace_records))
        ),
        corroboration_gate_count=_count_corroboration_gates(tuple(step_trace_records)),
        corroborated_family_change_count=_count_corroborated_family_changes(
            tuple(step_trace_records)
        ),
        role_level_usage_counts=_external_evidence_role_usage_counts(
            tuple(step_trace_records)
        ),
        rollout_fidelity_gate_passed=_rollout_fidelity_gate_passed(trace, benchmark_case),
        operational_metrics_gate_passed=_operational_metrics_gate_passed(
            summary,
            rollout_metrics,
            tuple(step_trace_records),
        ),
    )
    return StockpylSerialRun(
        mode=mode,
        run_index=evaluation_case.case_index,
        schedule_name=evaluation_case.schedule_name,
        run_seed=evaluation_case.run_seed,
        tool_ablation_variant=evaluation_case.tool_ablation_variant,
        benchmark_case=benchmark_case,
        trace=trace,
        rollout_metrics=rollout_metrics,
        summary=summary,
        regime_schedule=evaluation_case.regime_schedule,
        episode_summary_record=episode_summary_record,
        step_trace_records=tuple(step_trace_records),
        llm_call_trace_records=tuple(llm_call_trace_records),
        tool_call_trace_records=tuple(tool_call_trace_records),
        orchestration_responses=tuple(responses),
        llm_provider=llm_provider,
        llm_model_name=agent_config.llm_model_name if llm_provider is not None else None,
        prompt_version=prompt_version,
        prompt_contract_hash=prompt_contract_hash_value,
        invalid_output_count=invalid_output_count,
        fallback_count=fallback_count,
        successful_response_count=successful_response_count,
        validation_failure_counts=tuple(sorted(validation_failure_counter.items())),
        client_error_counts=client_error_counts,
        total_retry_count=total_retry_count,
        failure_before_response_count=failure_before_response_count,
        failure_after_response_count=failure_after_response_count,
    )


def _recent_regime_history(
    trace: EpisodeTrace,
    *,
    lookback: int = 3,
) -> tuple[RegimeLabel, ...]:
    if lookback <= 0:
        return ()
    return tuple(record.regime_label for record in trace.period_records[-lookback:])


def _recent_stress_reference_demand_value(
    trace: EpisodeTrace,
) -> float | None:
    for record in reversed(trace.period_records):
        if record.regime_label is RegimeLabel.DEMAND_REGIME_SHIFT:
            return record.observation.demand_realization[-1]
    return None


def _recent_update_request_history(
    responses: tuple[OrchestrationResponse, ...],
    *,
    lookback: int = 4,
) -> tuple[tuple[UpdateRequestType, ...], ...]:
    if lookback <= 0:
        return ()
    history: list[tuple[UpdateRequestType, ...]] = []
    for response in responses[-lookback:]:
        if response.agent_assessment is None:
            history.append(())
            continue
        history.append(
            tuple(
                update_request.request_type
                for update_request in response.agent_assessment.update_requests
            )
        )
    return tuple(history)


def _run_baseline_rollout(
    experiment_config: ExperimentConfig,
    benchmark_case: SerialBenchmarkCase,
    *,
    evaluation_case: EvaluationCase,
) -> StockpylSerialRun:
    baseline = DeterministicBaselinePolicy()
    optimizer = TrustedOptimizerAdapter()
    episode_id = (
        f"{experiment_config.experiment_name}_deterministic_baseline_"
        f"{evaluation_case.tool_ablation_variant}_"
        f"{evaluation_case.schedule_name}_seed_{evaluation_case.run_seed}_"
        f"run_{evaluation_case.case_index}"
    )
    system_state = build_initial_simulation_state(
        benchmark_case,
        regime_label=regime_for_period(evaluation_case.regime_schedule, 0),
    )
    trace = EpisodeTrace(
        run_id=episode_id,
        benchmark_id=benchmark_case.benchmark_id,
    )
    step_trace_records: list[StepTraceRecord] = []
    for time_index in range(experiment_config.resolved_rollout_horizon()):
        regime_label = regime_for_period(evaluation_case.regime_schedule, time_index)
        previous_regime_label = (
            trace.period_records[-1].regime_label if trace.period_records else None
        )
        observation = build_period_observation(
            benchmark_case,
            system_state,
            regime_label,
            previous_regime_label=previous_regime_label,
        )
        evidence = build_runtime_evidence(benchmark_case, observation)
        orchestration_start = perf_counter()
        decision = baseline.decide(system_state, observation, evidence)
        step_telemetry = _build_baseline_step_telemetry(
            signal=decision.signal,
            orchestration_latency_ms=(perf_counter() - orchestration_start) * 1000.0,
        )
        counterfactual_scenario_update_result = _default_keep_current_update(
            regime_label,
            observation,
            provenance="deterministic_baseline_counterfactual_keep_current",
        )
        optimization_request = build_optimization_request(
            system_state=system_state,
            scenario_update_result=decision.scenario_update_result,
            base_stock_levels=benchmark_case.base_stock_levels,
            planning_horizon=1,
        )
        counterfactual_request = build_optimization_request(
            system_state=system_state,
            scenario_update_result=counterfactual_scenario_update_result,
            base_stock_levels=benchmark_case.base_stock_levels,
            planning_horizon=1,
        )
        optimization_result = optimizer.solve(optimization_request)
        counterfactual_result = optimizer.solve(counterfactual_request)
        zero_order_result = OptimizationResult(
            replenishment_orders=tuple(0.0 for _ in benchmark_case.stage_names),
            planning_horizon=1,
        )
        next_regime = None
        if time_index + 1 < experiment_config.resolved_rollout_horizon():
            next_regime = regime_for_period(evaluation_case.regime_schedule, time_index + 1)
        transition = advance_serial_state(
            benchmark_case,
            current_state=system_state,
            observation=observation,
            optimization_result=optimization_result,
            next_regime=next_regime,
        )
        counterfactual_transition = advance_serial_state(
            benchmark_case,
            current_state=system_state,
            observation=observation,
            optimization_result=counterfactual_result,
            next_regime=next_regime,
        )
        zero_order_transition = advance_serial_state(
            benchmark_case,
            current_state=system_state,
            observation=observation,
            optimization_result=zero_order_result,
            next_regime=next_regime,
        )
        trace = trace.append_period(
            PeriodTraceRecord(
                time_index=time_index,
                regime_label=regime_label,
                state=system_state,
                observation=observation,
                agent_signal=decision.signal,
                optimization_result=optimization_result,
                next_state=transition.next_state,
                realized_demand=transition.realized_demand,
                demand_load=transition.demand_load,
                served_demand=transition.served_demand,
                unmet_demand=transition.unmet_demand,
                step_telemetry=step_telemetry,
                notes=transition.notes,
            )
        )
        period_record = trace.period_records[-1]
        step_trace_records.append(
            _build_step_trace_record(
                episode_id=episode_id,
                mode="deterministic_baseline",
                tool_ablation_variant=evaluation_case.tool_ablation_variant,
                schedule_name=evaluation_case.schedule_name,
                run_seed=evaluation_case.run_seed,
                period_record=period_record,
                optimization_request=optimization_request,
                counterfactual_request=counterfactual_request,
                transition=transition,
                counterfactual_transition=counterfactual_transition,
                zero_order_transition=zero_order_transition,
                response=None,
                scenario_update_result=decision.scenario_update_result,
                cost_config=benchmark_case.benchmark_config.costs,
                holding_cost_by_stage=benchmark_case.holding_costs,
                stockout_cost_by_stage=benchmark_case.stockout_costs,
                external_evidence_source=None,
                external_evidence_batch=None,
            )
        )
        system_state = transition.next_state
    rollout_metrics = compute_rollout_metrics(
        trace,
        benchmark_case.benchmark_config.costs,
        holding_cost_by_stage=benchmark_case.holding_costs,
        stockout_cost_by_stage=benchmark_case.stockout_costs,
    )
    episode_telemetry = summarize_episode_telemetry(trace.step_telemetry)
    decision_quality = compute_decision_quality(tuple(step_trace_records))
    summary = _build_summary(
        mode="deterministic_baseline",
        run_id=episode_id,
        benchmark_case=benchmark_case,
        trace=trace,
        rollout_metrics=rollout_metrics,
        episode_telemetry=episode_telemetry,
        decision_quality=decision_quality,
        optimizer_order_boundary_preserved=True,
    )
    episode_summary_record = build_episode_summary_record(
        summary,
        mode="deterministic_baseline",
        validation_lane=VALIDATION_LANE,
        tool_ablation_variant=evaluation_case.tool_ablation_variant,
        schedule_name=evaluation_case.schedule_name,
        run_seed=evaluation_case.run_seed,
        regime_schedule=tuple(label.value for label in evaluation_case.regime_schedule),
        cost_breakdown=_cost_breakdown_record(rollout_metrics),
        intervention_count=_intervention_count(tuple(step_trace_records)),
        invalid_output_count=0,
        fallback_count=0,
        unavailable_tool_request_count=0,
        disabled_tool_fallback_count=0,
        sequencing_blocked_tool_request_count=0,
        clean_intervention_count=_count_clean_interventions(tuple(step_trace_records)),
        clean_optimizer_input_change_count=_count_clean_optimizer_input_changes(
            tuple(step_trace_records)
        ),
        repeated_stress_moderation_count=_count_repeated_stress_moderations(
            tuple(step_trace_records)
        ),
        relapse_moderation_count=_count_relapse_moderations(
            tuple(step_trace_records)
        ),
        unresolved_stress_moderation_count=_count_unresolved_stress_moderations(
            tuple(step_trace_records)
        ),
        moderated_update_count=_count_moderated_updates(tuple(step_trace_records)),
        hysteresis_application_count=_count_hysteresis_applications(
            tuple(step_trace_records)
        ),
        external_evidence_source=None,
        external_evidence_period_count=0,
        external_evidence_tool_call_count=0,
        false_alarm_evidence_count=0,
        evidence_supported_intervention_count=0,
        external_evidence_changed_optimizer_input_count=0,
        evidence_fusion_cap_count=0,
        capped_external_strengthening_count=0,
        early_evidence_confirmation_gate_count=0,
        early_evidence_family_change_block_count=0,
        corroboration_gate_count=0,
        corroborated_family_change_count=0,
        role_level_usage_counts=(),
        rollout_fidelity_gate_passed=_rollout_fidelity_gate_passed(trace, benchmark_case),
        operational_metrics_gate_passed=_operational_metrics_gate_passed(
            summary,
            rollout_metrics,
            tuple(step_trace_records),
        ),
    )
    return StockpylSerialRun(
        mode="deterministic_baseline",
        run_index=evaluation_case.case_index,
        schedule_name=evaluation_case.schedule_name,
        run_seed=evaluation_case.run_seed,
        tool_ablation_variant=evaluation_case.tool_ablation_variant,
        benchmark_case=benchmark_case,
        trace=trace,
        rollout_metrics=rollout_metrics,
        summary=summary,
        regime_schedule=evaluation_case.regime_schedule,
        episode_summary_record=episode_summary_record,
        step_trace_records=tuple(step_trace_records),
    )


def _run_named_mode(
    experiment_config: ExperimentConfig,
    agent_config: AgentConfig,
    benchmark_case: SerialBenchmarkCase,
    *,
    evaluation_case: EvaluationCase,
    mode: str,
) -> StockpylSerialRun:
    normalized_mode = _normalize_mode_alias(mode)
    if normalized_mode == "deterministic_baseline":
        return _run_baseline_rollout(
            experiment_config,
            benchmark_case,
            evaluation_case=evaluation_case,
        )
    if normalized_mode == "deterministic_orchestrator":
        return _run_orchestrated_rollout(
            experiment_config,
            agent_config,
            benchmark_case,
            evaluation_case=evaluation_case,
            mode=mode,
        )
    if normalized_mode == "llm_orchestrator":
        return _run_orchestrated_rollout(
            experiment_config,
            agent_config,
            benchmark_case,
            evaluation_case=evaluation_case,
            mode=mode,
        )
    if normalized_mode == "orchestration_agent":
        return _run_orchestrated_rollout(
            experiment_config,
            agent_config,
            benchmark_case,
            evaluation_case=evaluation_case,
            mode="deterministic_orchestrator",
        )
    raise ValueError(f"Unsupported mode: {mode!r}.")


def _outcomes_differ(runs: tuple[StockpylSerialRun, ...]) -> bool:
    order_sequences = tuple(
        tuple(result.replenishment_orders for result in run.trace.optimization_results)
        for run in runs
    )
    if len(set(order_sequences)) > 1:
        return True
    summary_points = tuple(
        (
            run.summary.average_inventory,
            run.summary.average_backorder_level,
            run.summary.total_cost,
            run.summary.fill_rate,
        )
        for run in runs
    )
    return len(set(summary_points)) > 1


def _mean_or_none(values: tuple[float | None, ...]) -> float | None:
    numeric = tuple(value for value in values if value is not None)
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def _classify_mode_aggregate_summary(
    aggregate_metrics: ModeAggregateSummary,
    *,
    experiment_name: str,
    benchmark_source: str,
    provider: str | None,
    semi_synthetic_external_evidence: bool = False,
    external_evidence_source: str | None = None,
) -> ModeAggregateSummary:
    governance = classify_artifact_governance(
        experiment_name=experiment_name,
        benchmark_source=benchmark_source,
        provider=provider,
        validity_gate_passed=compute_validity_gate_passed(
            optimizer_order_boundary_preserved=(
                aggregate_metrics.validity_summary.optimizer_order_boundary_preserved
            ),
            invalid_output_count=aggregate_metrics.validity_summary.invalid_output_count,
            fallback_count=aggregate_metrics.validity_summary.fallback_count,
        ),
        rollout_fidelity_gate_passed=(
            aggregate_metrics.validity_summary.rollout_fidelity_gate_passed
        ),
        operational_metrics_gate_passed=(
            aggregate_metrics.validity_summary.operational_metrics_gate_passed
        ),
        semi_synthetic_external_evidence=semi_synthetic_external_evidence,
        external_evidence_source=external_evidence_source,
    )
    return replace(
        aggregate_metrics,
        artifact_use_class=governance.artifact_use_class,
        validity_gate_passed=governance.validity_gate_passed,
        eligibility_notes=governance.eligibility_notes,
    )


def _classify_batch_aggregate_summary(
    aggregate_summary: BatchAggregateSummary,
    *,
    experiment_name: str,
    benchmark_source: str,
    provider_by_mode: dict[str, str | None],
    semi_synthetic_external_evidence: bool = False,
    external_evidence_source: str | None = None,
) -> BatchAggregateSummary:
    classified_mode_summaries = []
    decisions = []
    for mode_summary in aggregate_summary.mode_summaries:
        classified_mode_summary = _classify_mode_aggregate_summary(
            mode_summary,
            experiment_name=experiment_name,
            benchmark_source=benchmark_source,
            provider=provider_by_mode.get(mode_summary.mode),
            semi_synthetic_external_evidence=semi_synthetic_external_evidence,
            external_evidence_source=external_evidence_source,
        )
        classified_mode_summaries.append(classified_mode_summary)
        decisions.append(
            classify_artifact_governance(
                experiment_name=experiment_name,
                benchmark_source=benchmark_source,
                provider=provider_by_mode.get(mode_summary.mode),
                validity_gate_passed=classified_mode_summary.validity_gate_passed,
                rollout_fidelity_gate_passed=(
                    classified_mode_summary.validity_summary.rollout_fidelity_gate_passed
                ),
                operational_metrics_gate_passed=(
                    classified_mode_summary.validity_summary.operational_metrics_gate_passed
                ),
                semi_synthetic_external_evidence=semi_synthetic_external_evidence,
                external_evidence_source=external_evidence_source,
            )
        )
    directory_governance = summarize_directory_governance(tuple(decisions))
    return replace(
        aggregate_summary,
        mode_summaries=tuple(classified_mode_summaries),
        artifact_use_class=directory_governance.artifact_use_class,
        validity_gate_passed=directory_governance.validity_gate_passed,
        eligibility_notes=directory_governance.eligibility_notes,
    )


def _classify_mode_batch(
    batch: StockpylSerialModeBatch,
    *,
    experiment_name: str,
    benchmark_source: str,
    semi_synthetic_external_evidence: bool = False,
    external_evidence_source: str | None = None,
) -> StockpylSerialModeBatch:
    if batch.aggregate_metrics is None:
        return batch
    provider = _value_or_mixed_with_missing(tuple(run.llm_provider for run in batch.runs))
    classified_metrics = _classify_mode_aggregate_summary(
        batch.aggregate_metrics,
        experiment_name=experiment_name,
        benchmark_source=benchmark_source,
        provider=provider,
        semi_synthetic_external_evidence=semi_synthetic_external_evidence,
        external_evidence_source=external_evidence_source,
    )
    return replace(batch, aggregate_metrics=classified_metrics)


def _build_mode_batch(mode: str, runs: tuple[StockpylSerialRun, ...]) -> StockpylSerialModeBatch:
    optimizer_preserved = all(run.summary.optimizer_order_boundary_preserved for run in runs)
    decision_quality = compute_decision_quality(
        tuple(
            record
            for run in runs
            for record in run.step_trace_records
        )
    )
    llm_diagnostics = None
    if any(run.llm_provider is not None for run in runs):
        provider_values = tuple(run.llm_provider for run in runs if run.llm_provider is not None)
        model_values = tuple(
            run.llm_model_name for run in runs if run.llm_model_name is not None
        )
        episode_telemetry = tuple(
            run.summary.episode_telemetry
            for run in runs
            if run.summary.episode_telemetry is not None
        )
        prompt_version = _single_prompt_version(tuple(run.prompt_version for run in runs))
        prompt_hash = _single_prompt_hash(tuple(run.prompt_contract_hash for run in runs))
        if prompt_version is None:
            prompt_version = PROMPT_VERSION
        if prompt_hash is None and prompt_version == PROMPT_VERSION:
            prompt_hash = prompt_contract_hash()
        llm_diagnostics = LLMRunDiagnostics(
            model_name=model_values[0] if model_values else "unknown",
            provider=provider_values[0] if provider_values else "unknown",
            run_count=len(runs),
            prompt_version=prompt_version,
            prompt_hash=prompt_hash,
            validation_failure_counts=_validation_failure_counts_from_runs(runs),
            client_error_counts=_client_error_counts_from_runs(runs),
            average_prompt_tokens=_mean_or_none(
                tuple(
                    summary.average_prompt_tokens
                    for summary in episode_telemetry
                )
            ),
            average_completion_tokens=_mean_or_none(
                tuple(
                    summary.average_completion_tokens
                    for summary in episode_telemetry
                )
            ),
            average_total_tokens=_mean_or_none(
                tuple(summary.average_total_tokens for summary in episode_telemetry)
            ),
            average_llm_latency_ms=_mean_or_none(
                tuple(summary.average_llm_latency_ms for summary in episode_telemetry)
            ),
            average_orchestration_latency_ms=_mean_or_none(
                tuple(
                    summary.average_orchestration_latency_ms
                    for summary in episode_telemetry
                )
            ),
            invalid_output_count=sum(run.invalid_output_count for run in runs),
            fallback_count=sum(run.fallback_count for run in runs),
            successful_response_count=sum(run.successful_response_count for run in runs),
            total_retry_count=sum(run.total_retry_count for run in runs),
            failure_before_response_count=sum(
                run.failure_before_response_count for run in runs
            ),
            failure_after_response_count=sum(
                run.failure_after_response_count for run in runs
            ),
            optimizer_order_boundary_preserved=optimizer_preserved,
        )
    aggregate_summary = ComparisonRunSummary(
        mode=mode,
        run_count=len(runs),
        average_tool_call_count=(
            sum(run.summary.intervention_summary.tool_call_count for run in runs) / len(runs)
        ),
        average_replan_count=(
            sum(run.summary.intervention_summary.replan_count for run in runs) / len(runs)
        ),
        average_abstain_count=(
            sum(run.summary.intervention_summary.abstain_count for run in runs) / len(runs)
        ),
        average_no_action_count=(
            sum(run.summary.intervention_summary.no_action_count for run in runs) / len(runs)
        ),
        average_total_cost=_mean_or_none(tuple(run.summary.total_cost for run in runs)),
        average_fill_rate=_mean_or_none(tuple(run.summary.fill_rate for run in runs)),
        optimizer_order_boundary_preserved=optimizer_preserved,
        llm_diagnostics=llm_diagnostics,
        decision_quality=decision_quality,
    )
    episode_summary_records = tuple(
        run.episode_summary_record for run in runs if run.episode_summary_record is not None
    )
    step_trace_records = tuple(
        record
        for run in runs
        for record in run.step_trace_records
    )
    tool_call_trace_records = tuple(
        record
        for run in runs
        for record in run.tool_call_trace_records
    )
    aggregate_metrics = aggregate_mode_episode_summaries(
        mode,
        episode_summary_records,
        validation_lane=VALIDATION_LANE,
        step_trace_records=step_trace_records,
        tool_call_records=tool_call_trace_records,
    )
    return StockpylSerialModeBatch(
        mode=mode,
        runs=runs,
        aggregate_summary=aggregate_summary,
        aggregate_metrics=aggregate_metrics,
        tool_usage_by_ablation=aggregate_metrics.tool_attribution,
        tool_ablation_summaries=tuple(
            summary.tool_ablation_summary
            for summary in aggregate_metrics.ablation_breakdown
            if summary.tool_ablation_summary is not None
        ),
    )


def run_stockpyl_serial_mode_sweep(
    experiment_config_path: str | Path = DEFAULT_EXPERIMENT_CONFIG,
    *,
    llm_client_mode_override: str | None = None,
    llm_model_name_override: str | None = None,
    max_runs: int | None = None,
) -> StockpylSerialModeSweep:
    """Execute the explicit configured mode set for one experiment config."""

    experiment_config, _, benchmark_case = _load_runtime_components(
        experiment_config_path,
        llm_client_mode_override=llm_client_mode_override,
        llm_model_name_override=llm_model_name_override,
    )
    mode_batches = tuple(
        run_stockpyl_serial_batch(
            experiment_config_path,
            mode=mode,
            llm_client_mode_override=llm_client_mode_override,
            llm_model_name_override=llm_model_name_override,
            max_runs=max_runs,
        )
        for mode in experiment_config.resolved_mode_set()
    )
    all_runs = tuple(run for batch in mode_batches for run in batch.runs)
    records_by_mode = {
        batch.mode: tuple(
            run.episode_summary_record
            for run in batch.runs
            if run.episode_summary_record is not None
        )
        for batch in mode_batches
    }
    step_trace_records_by_mode = {
        batch.mode: tuple(
            record for run in batch.runs for record in run.step_trace_records
        )
        for batch in mode_batches
    }
    tool_call_records_by_mode = {
        batch.mode: tuple(
            record for run in batch.runs for record in run.tool_call_trace_records
        )
        for batch in mode_batches
    }
    aggregate_results = _classify_batch_aggregate_summary(
        aggregate_batch_episode_summaries(
            benchmark_id=benchmark_case.benchmark_id,
            validation_lane=VALIDATION_LANE,
            records_by_mode=records_by_mode,
            step_trace_records_by_mode=step_trace_records_by_mode,
            tool_call_records_by_mode=tool_call_records_by_mode,
        ),
        experiment_name=experiment_config.experiment_name,
        benchmark_source=benchmark_case.adapter_name,
        provider_by_mode={
            batch.mode: _value_or_mixed_with_missing(
                tuple(run.llm_provider for run in batch.runs)
            )
            for batch in mode_batches
        },
        semi_synthetic_external_evidence=experiment_config.semi_synthetic_external_evidence,
        external_evidence_source=_resolved_external_evidence_source(experiment_config),
    )
    return StockpylSerialModeSweep(
        benchmark_case=benchmark_case,
        schedule_names=tuple(
            schedule.name for schedule in experiment_config.resolved_schedule_set()
        ),
        seed_values=experiment_config.resolved_seed_set(
            benchmark_case.benchmark_config.random_seed
        ),
        tool_ablation_variants=experiment_config.resolved_tool_ablation_variants(),
        mode_batches=mode_batches,
        outcomes_differ=_outcomes_differ(all_runs),
        optimizer_order_boundary_preserved=all(
            batch.aggregate_summary.optimizer_order_boundary_preserved
            for batch in mode_batches
        ),
        aggregate_results=aggregate_results,
    )


def run_stockpyl_serial(
    experiment_config_path: str | Path = DEFAULT_EXPERIMENT_CONFIG,
    *,
    mode: str = "deterministic_orchestrator",
    llm_client_mode_override: str | None = None,
    llm_model_name_override: str | None = None,
) -> StockpylSerialRun:
    """Execute one controlled Stockpyl serial rollout in the selected mode."""

    experiment_config, agent_config, benchmark_case = _load_runtime_components(
        experiment_config_path,
        llm_client_mode_override=llm_client_mode_override,
        llm_model_name_override=llm_model_name_override,
    )
    evaluation_case = _build_evaluation_cases(
        experiment_config,
        default_seed=benchmark_case.benchmark_config.random_seed,
    )[0]
    case_experiment_config = _experiment_config_for_case(experiment_config, evaluation_case)
    case_benchmark = _benchmark_case_for_seed(
        benchmark_case,
        run_seed=evaluation_case.run_seed,
    )
    return _run_named_mode(
        case_experiment_config,
        agent_config,
        case_benchmark,
        evaluation_case=evaluation_case,
        mode=mode,
    )


def run_stockpyl_serial_batch(
    experiment_config_path: str | Path = DEFAULT_EXPERIMENT_CONFIG,
    *,
    mode: str = "deterministic_orchestrator",
    llm_client_mode_override: str | None = None,
    llm_model_name_override: str | None = None,
    max_runs: int | None = None,
) -> StockpylSerialModeBatch:
    """Execute repeated controlled Stockpyl serial rollouts in one mode."""

    experiment_config, agent_config, benchmark_case = _load_runtime_components(
        experiment_config_path,
        llm_client_mode_override=llm_client_mode_override,
        llm_model_name_override=llm_model_name_override,
    )
    evaluation_cases = _limit_evaluation_cases(
        _build_evaluation_cases(
            experiment_config,
            default_seed=benchmark_case.benchmark_config.random_seed,
        ),
        max_runs=max_runs,
    )
    runs = tuple(
        _run_named_mode(
            _experiment_config_for_case(experiment_config, evaluation_case),
            agent_config,
            _benchmark_case_for_seed(benchmark_case, run_seed=evaluation_case.run_seed),
            evaluation_case=evaluation_case,
            mode=mode,
        )
        for evaluation_case in evaluation_cases
    )
    return _classify_mode_batch(
        _build_mode_batch(mode, runs),
        experiment_name=experiment_config.experiment_name,
        benchmark_source=benchmark_case.adapter_name,
        semi_synthetic_external_evidence=experiment_config.semi_synthetic_external_evidence,
        external_evidence_source=_resolved_external_evidence_source(experiment_config),
    )


def run_stockpyl_serial_comparison(
    experiment_config_path: str | Path = DEFAULT_EXPERIMENT_CONFIG,
    *,
    llm_client_mode_override: str | None = None,
    llm_model_name_override: str | None = None,
) -> StockpylSerialComparison:
    """Run one instance of each supported mode on the same benchmark case and schedule."""

    experiment_config, agent_config, benchmark_case = _load_runtime_components(
        experiment_config_path,
        llm_client_mode_override=llm_client_mode_override,
        llm_model_name_override=llm_model_name_override,
    )
    evaluation_case = _build_evaluation_cases(
        experiment_config,
        default_seed=benchmark_case.benchmark_config.random_seed,
    )[0]
    case_experiment_config = _experiment_config_for_case(experiment_config, evaluation_case)
    case_benchmark = _benchmark_case_for_seed(benchmark_case, run_seed=evaluation_case.run_seed)
    baseline_run = _run_baseline_rollout(
        case_experiment_config,
        case_benchmark,
        evaluation_case=evaluation_case,
    )
    deterministic_orchestrator_run = _run_orchestrated_rollout(
        case_experiment_config,
        agent_config,
        case_benchmark,
        evaluation_case=evaluation_case,
        mode="deterministic_orchestrator",
    )
    llm_orchestrator_run = _run_orchestrated_rollout(
        case_experiment_config,
        agent_config,
        case_benchmark,
        evaluation_case=evaluation_case,
        mode="llm_orchestrator",
    )
    optimizer_order_boundary_preserved = (
        baseline_run.summary.optimizer_order_boundary_preserved
        and deterministic_orchestrator_run.summary.optimizer_order_boundary_preserved
        and llm_orchestrator_run.summary.optimizer_order_boundary_preserved
    )
    return StockpylSerialComparison(
        benchmark_case=case_benchmark,
        schedule_name=evaluation_case.schedule_name,
        run_seed=evaluation_case.run_seed,
        regime_schedule=evaluation_case.regime_schedule,
        deterministic_baseline=baseline_run,
        deterministic_orchestrator=deterministic_orchestrator_run,
        llm_orchestrator=llm_orchestrator_run,
        outcomes_differ=_outcomes_differ(
            (
                baseline_run,
                deterministic_orchestrator_run,
                llm_orchestrator_run,
            )
        ),
        optimizer_order_boundary_preserved=optimizer_order_boundary_preserved,
    )


def run_stockpyl_serial_comparison_batch(
    experiment_config_path: str | Path = DEFAULT_EXPERIMENT_CONFIG,
    *,
    llm_client_mode_override: str | None = None,
    llm_model_name_override: str | None = None,
    max_runs: int | None = None,
) -> StockpylSerialComparisonBatch:
    """Run repeated comparisons across deterministic and LLM modes."""

    experiment_config, _, benchmark_case = _load_runtime_components(
        experiment_config_path,
        llm_client_mode_override=llm_client_mode_override,
        llm_model_name_override=llm_model_name_override,
    )
    if experiment_config.resolved_mode_set() != LEGACY_COMPARISON_MODES:
        raise ValueError(
            "run_stockpyl_serial_comparison_batch only supports the legacy three-mode "
            "comparison set. Use run_stockpyl_serial_mode_sweep for explicit extended mode sets."
        )
    mode_sweep = run_stockpyl_serial_mode_sweep(
        experiment_config_path,
        llm_client_mode_override=llm_client_mode_override,
        llm_model_name_override=llm_model_name_override,
        max_runs=max_runs,
    )
    mode_batch_by_name = {
        batch.mode: batch for batch in mode_sweep.mode_batches
    }
    baseline_batch = mode_batch_by_name["deterministic_baseline"]
    deterministic_batch = mode_batch_by_name["deterministic_orchestrator"]
    llm_batch = mode_batch_by_name["llm_orchestrator"]
    return StockpylSerialComparisonBatch(
        benchmark_case=benchmark_case,
        schedule_names=tuple(schedule.name for schedule in experiment_config.resolved_schedule_set()),
        seed_values=experiment_config.resolved_seed_set(benchmark_case.benchmark_config.random_seed),
        tool_ablation_variants=experiment_config.resolved_tool_ablation_variants(),
        deterministic_baseline=baseline_batch,
        deterministic_orchestrator=deterministic_batch,
        llm_orchestrator=llm_batch,
        outcomes_differ=mode_sweep.outcomes_differ,
        optimizer_order_boundary_preserved=(
            baseline_batch.aggregate_summary.optimizer_order_boundary_preserved
            and deterministic_batch.aggregate_summary.optimizer_order_boundary_preserved
            and llm_batch.aggregate_summary.optimizer_order_boundary_preserved
        ),
        aggregate_results=mode_sweep.aggregate_results,
    )


def write_stockpyl_serial_artifacts(
    experiment_config_path: str | Path = DEFAULT_EXPERIMENT_CONFIG,
    *,
    mode: str = "deterministic_orchestrator",
    llm_client_mode_override: str | None = None,
    llm_model_name_override: str | None = None,
    max_runs: int | None = None,
    output_dir_override: Path | None = None,
) -> tuple[ExperimentMetadata, Path, dict[str, Path]]:
    """Write structured logging artifacts for one mode or the full comparison."""

    experiment_config, agent_config, benchmark_case = _load_runtime_components(
        experiment_config_path,
        llm_client_mode_override=llm_client_mode_override,
        llm_model_name_override=llm_model_name_override,
    )
    if mode == "all":
        if experiment_config.resolved_mode_set() == LEGACY_COMPARISON_MODES:
            comparison_batch = run_stockpyl_serial_comparison_batch(
                experiment_config_path,
                llm_client_mode_override=llm_client_mode_override,
                llm_model_name_override=llm_model_name_override,
                max_runs=max_runs,
            )
            runs = (
                comparison_batch.deterministic_baseline.runs
                + comparison_batch.deterministic_orchestrator.runs
                + comparison_batch.llm_orchestrator.runs
            )
        else:
            mode_sweep = run_stockpyl_serial_mode_sweep(
                experiment_config_path,
                llm_client_mode_override=llm_client_mode_override,
                llm_model_name_override=llm_model_name_override,
                max_runs=max_runs,
            )
            runs = tuple(
                run for batch in mode_sweep.mode_batches for run in batch.runs
            )
    else:
        runs = run_stockpyl_serial_batch(
            experiment_config_path,
            mode=mode,
            llm_client_mode_override=llm_client_mode_override,
            llm_model_name_override=llm_model_name_override,
            max_runs=max_runs,
        ).runs
    return _write_logging_artifacts(
        experiment_config=experiment_config,
        agent_config=agent_config,
        benchmark_case=benchmark_case,
        runs=runs,
        mode=mode,
        llm_client_mode_override=llm_client_mode_override,
        llm_model_name_override=llm_model_name_override,
        max_runs=max_runs,
        output_dir_override=output_dir_override,
    )


def main() -> None:
    """Run Stockpyl serial rollouts and print compact per-run and aggregate summaries."""

    parser = argparse.ArgumentParser(
        description="Run the MEIO Stockpyl serial benchmark path."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_EXPERIMENT_CONFIG,
        help="Path to the experiment configuration file.",
    )
    parser.add_argument(
        "--mode",
        choices=(
            "all",
            "deterministic_baseline",
            "deterministic_orchestrator",
            "llm_orchestrator",
            "llm_orchestrator_internal_only",
            "llm_orchestrator_with_external_evidence",
            "orchestration_agent",
        ),
        default="all",
        help="Select one rollout mode or run the full controlled comparison.",
    )
    parser.add_argument(
        "--llm-client-mode",
        choices=("config", "fake", "real"),
        default="config",
        help="Override the configured LLM client mode for the bounded LLM path.",
    )
    parser.add_argument(
        "--llm-model-name",
        type=str,
        default=None,
        help="Optional override for the configured LLM model name.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap on evaluated schedule/seed cases for bounded live validation.",
    )
    args = parser.parse_args()

    llm_client_mode_override = None if args.llm_client_mode == "config" else args.llm_client_mode
    experiment_config, agent_config, benchmark_case = _load_runtime_components(
        args.config,
        llm_client_mode_override=llm_client_mode_override,
        llm_model_name_override=args.llm_model_name,
    )
    if args.mode == "all":
        if experiment_config.resolved_mode_set() == LEGACY_COMPARISON_MODES:
            batch_payload = run_stockpyl_serial_comparison_batch(
                args.config,
                llm_client_mode_override=llm_client_mode_override,
                llm_model_name_override=args.llm_model_name,
                max_runs=args.max_runs,
            )
            runs = (
                batch_payload.deterministic_baseline.runs
                + batch_payload.deterministic_orchestrator.runs
                + batch_payload.llm_orchestrator.runs
            )
        else:
            batch_payload = run_stockpyl_serial_mode_sweep(
                args.config,
                llm_client_mode_override=llm_client_mode_override,
                llm_model_name_override=args.llm_model_name,
                max_runs=args.max_runs,
            )
            runs = tuple(
                run for mode_batch in batch_payload.mode_batches for run in mode_batch.runs
            )
        metadata, artifacts_dir, written_files = _write_logging_artifacts(
            experiment_config=experiment_config,
            agent_config=agent_config,
            benchmark_case=benchmark_case,
            runs=runs,
            mode="all",
            llm_client_mode_override=llm_client_mode_override,
            llm_model_name_override=args.llm_model_name,
            max_runs=args.max_runs,
        )
        payload = batch_payload.to_summary()
        payload["artifacts_dir"] = str(artifacts_dir)
        payload["artifact_files"] = {
            key: str(path) for key, path in written_files.items()
        }
        payload["experiment_metadata"] = jsonable(metadata)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    batch = run_stockpyl_serial_batch(
        args.config,
        mode=args.mode,
        llm_client_mode_override=llm_client_mode_override,
        llm_model_name_override=args.llm_model_name,
        max_runs=args.max_runs,
    )
    metadata, artifacts_dir, written_files = _write_logging_artifacts(
        experiment_config=experiment_config,
        agent_config=agent_config,
        benchmark_case=benchmark_case,
        runs=batch.runs,
        mode=args.mode,
        llm_client_mode_override=llm_client_mode_override,
        llm_model_name_override=args.llm_model_name,
        max_runs=args.max_runs,
    )
    payload = batch.to_summary()
    payload["artifacts_dir"] = str(artifacts_dir)
    payload["artifact_files"] = {
        key: str(path) for key, path in written_files.items()
    }
    payload["experiment_metadata"] = jsonable(metadata)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
