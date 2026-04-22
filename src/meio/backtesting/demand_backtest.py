"""Bounded real-demand backtesting on top of the existing orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from meio.agents.baselines import DeterministicBaselinePolicy
from meio.agents.llm_client import FakeLLMClient, OpenAILLMClient
from meio.agents.llm_orchestrator import LLMOrchestrator
from meio.agents.prompts import PROMPT_VERSION, prompt_contract_hash
from meio.agents.runtime import OrchestrationResponse, OrchestrationRuntime, RuntimeMode
from meio.agents.telemetry import summarize_episode_telemetry
from meio.config.loaders import (
    load_agent_config,
    load_benchmark_config,
    load_real_demand_backtest_config,
    load_real_demand_backtest_panel_config,
)
from meio.config.schemas import (
    AgentConfig,
    RealDemandBacktestConfig,
    RealDemandBacktestPanelConfig,
)
from meio.contracts import MissionSpec, OperationalSubgoal, RegimeLabel, UpdateRequestType
from meio.data.real_demand_loader import (
    BacktestWindow,
    RealDemandSeries,
    construct_backtest_window,
    load_real_demand_series,
    resolve_evaluation_start_index,
    resolve_real_demand_dataset_paths,
)
from meio.evaluation.aggregate_results import (
    BatchAggregateSummary,
    aggregate_batch_episode_summaries,
)
from meio.evaluation.logging_io import (
    ensure_output_dir,
    hash_jsonable,
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
    summarize_interventions,
    summarize_traces,
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
from meio.simulation.evidence import DemandEvidence, LeadTimeEvidence, RuntimeEvidence
from meio.simulation.serial_benchmark import (
    SerialBenchmarkCase,
    advance_serial_state,
    build_initial_simulation_state,
    build_serial_benchmark_case,
)
from meio.simulation.state import EpisodeTrace, Observation, PeriodTraceRecord, SimulationState


VALIDATION_LANE = "real_demand_backtest"


@dataclass(frozen=True, slots=True)
class DemandBacktestRun:
    """One bounded real-demand backtest rollout for one mode."""

    mode: str
    slice_name: str
    dataset_name: str
    selected_skus: tuple[str, ...]
    window: BacktestWindow
    summary: BenchmarkRunSummary
    rollout_metrics: RolloutMetrics
    trace: EpisodeTrace
    episode_summary_record: EpisodeSummaryRecord
    step_trace_records: tuple[StepTraceRecord, ...]
    llm_call_trace_records: tuple[LLMCallTraceRecord, ...]
    tool_call_trace_records: tuple[ToolCallTraceRecord, ...]
    orchestration_responses: tuple[OrchestrationResponse, ...]
    llm_provider: str | None
    llm_model_name: str | None
    prompt_version: str | None
    prompt_hash: str | None


@dataclass(frozen=True, slots=True)
class DemandBacktestBatch:
    """Aggregate result bundle for one real-demand backtest run group."""

    config: RealDemandBacktestConfig
    dataset: RealDemandSeries
    window: BacktestWindow
    runs: tuple[DemandBacktestRun, ...]
    aggregate_summary: BatchAggregateSummary
    experiment_metadata: ExperimentMetadata
    run_manifest: RunManifestRecord


@dataclass(frozen=True, slots=True)
class DemandBacktestPanelSliceResult:
    """One slice result inside a real-demand validation panel run."""

    slice_name: str
    config: RealDemandBacktestConfig
    output_dir: Path | None
    experiment_metadata: ExperimentMetadata | None
    written_files: tuple[tuple[str, Path], ...]
    success: bool
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class DemandBacktestPanelRun:
    """One fixed panel execution across multiple real-demand slices."""

    panel_config: RealDemandBacktestPanelConfig
    panel_config_hash: str
    output_dir: Path
    slice_results: tuple[DemandBacktestPanelSliceResult, ...]
    aggregate_summary: BatchAggregateSummary | None
    experiment_metadata: ExperimentMetadata | None
    run_manifest: RunManifestRecord | None


def infer_real_series_regime(
    *,
    demand_value: float,
    leadtime_value: float,
    demand_baseline_value: float,
    leadtime_baseline_value: float,
    previous_regime: RegimeLabel,
) -> RegimeLabel:
    """Infer a bounded regime hint from unlabeled real demand and lead-time data."""

    demand_ratio = demand_value / demand_baseline_value if demand_baseline_value > 0.0 else 1.0
    leadtime_ratio = (
        leadtime_value / leadtime_baseline_value if leadtime_baseline_value > 0.0 else 1.0
    )
    if demand_ratio >= 1.15 and leadtime_ratio >= 1.15:
        return RegimeLabel.JOINT_DISRUPTION
    if leadtime_ratio >= 1.15:
        return RegimeLabel.SUPPLY_DISRUPTION
    if demand_ratio >= 1.12:
        return RegimeLabel.DEMAND_REGIME_SHIFT
    if previous_regime is not RegimeLabel.NORMAL and demand_ratio <= 1.02 and leadtime_ratio <= 1.05:
        return RegimeLabel.RECOVERY
    return RegimeLabel.NORMAL


def run_real_demand_backtest_batch(
    config_path: str | Path = "configs/experiment/real_demand_backtest.toml",
    *,
    mode: str = "all",
    llm_client_mode_override: str | None = None,
) -> DemandBacktestBatch:
    """Run the bounded real-demand backtest across the configured modes."""

    config = load_real_demand_backtest_config(config_path)
    return _run_real_demand_backtest_batch_from_config(
        config,
        mode=mode,
        llm_client_mode_override=llm_client_mode_override,
    )


def _run_real_demand_backtest_batch_from_config(
    config: RealDemandBacktestConfig,
    *,
    mode: str = "all",
    llm_client_mode_override: str | None = None,
) -> DemandBacktestBatch:
    """Run the bounded real-demand backtest from an already loaded config."""

    benchmark_config = load_benchmark_config(config.benchmark_config_path)
    agent_config = load_agent_config(config.agent_config_path)
    if llm_client_mode_override is not None:
        agent_config = replace(agent_config, llm_client_mode=llm_client_mode_override)
    benchmark_case = build_serial_benchmark_case(benchmark_config)
    dataset_paths = resolve_real_demand_dataset_paths(
        dataset_name=config.dataset_name,
        discovery_module=config.discovery_module,
        dataset_root=config.dataset_root,
        demand_csv_path=config.demand_csv_path,
        leadtime_csv_path=config.leadtime_csv_path,
        sku_list_path=config.sku_list_path,
    )
    dataset = load_real_demand_series(
        paths=dataset_paths,
        selected_sku_count=config.selected_sku_count,
        demand_target_mean=benchmark_config.demand_mean,
        subset_selection=config.subset_selection,
        selected_skus=config.selected_skus,
    )
    window = construct_backtest_window(
        total_length=len(dataset.dates),
        training_window_days=config.training_window_days,
        history_window_days=config.history_window_days,
        forecast_update_window_days=config.forecast_update_window_days,
        evaluation_horizon_days=config.evaluation_horizon_days,
        evaluation_start_index=resolve_evaluation_start_index(
            dates=dataset.dates,
            evaluation_start_date=config.evaluation_start_date,
        ),
    )
    mode_set = config.mode_set if mode == "all" else (mode,)
    runs = tuple(
        _run_mode_backtest(
            config=config,
            agent_config=agent_config,
            benchmark_case=benchmark_case,
            dataset=dataset,
            window=window,
            mode_name=mode_name,
        )
        for mode_name in mode_set
    )
    records_by_mode = {
        run.mode: (run.episode_summary_record,)
        for run in runs
    }
    step_records_by_mode = {
        run.mode: run.step_trace_records
        for run in runs
    }
    tool_call_records_by_mode = {
        run.mode: run.tool_call_trace_records
        for run in runs
    }
    aggregate_summary = _classify_batch_summary(
        aggregate_batch_episode_summaries(
            benchmark_id=benchmark_case.benchmark_id,
            validation_lane=VALIDATION_LANE,
            records_by_mode=records_by_mode,
            step_trace_records_by_mode=step_records_by_mode,
            tool_call_records_by_mode=tool_call_records_by_mode,
        ),
        benchmark_source=VALIDATION_LANE,
        experiment_name=config.experiment_name,
        llm_provider_by_mode={run.mode: run.llm_provider for run in runs},
    )
    experiment_metadata, run_manifest = _build_artifact_headers(
        config=config,
        benchmark_case=benchmark_case,
        aggregate_summary=aggregate_summary,
        runs=runs,
    )
    return DemandBacktestBatch(
        config=config,
        dataset=dataset,
        window=window,
        runs=runs,
        aggregate_summary=aggregate_summary,
        experiment_metadata=experiment_metadata,
        run_manifest=run_manifest,
    )


def write_demand_backtest_artifacts(
    config_path: str | Path = "configs/experiment/real_demand_backtest.toml",
    *,
    mode: str = "all",
    llm_client_mode_override: str | None = None,
    output_dir_override: str | Path | None = None,
) -> tuple[ExperimentMetadata, Path, dict[str, Path]]:
    """Run the bounded real-demand backtest and write artifacts."""

    batch = run_real_demand_backtest_batch(
        config_path,
        mode=mode,
        llm_client_mode_override=llm_client_mode_override,
    )
    return _write_demand_backtest_batch_artifacts(
        batch,
        output_dir_override=output_dir_override,
    )


def _write_demand_backtest_batch_artifacts(
    batch: DemandBacktestBatch,
    *,
    output_dir_override: str | Path | None = None,
) -> tuple[ExperimentMetadata, Path, dict[str, Path]]:
    """Write artifacts for one already executed real-demand batch."""

    mode_name = "all" if len({run.mode for run in batch.runs}) > 1 else batch.runs[0].mode
    run_name = batch.config.slice_name or batch.config.experiment_name
    run_group_id = (
        f"{run_name}_{mode_name}_{utc_timestamp().replace(':', '').replace('-', '')}"
    )
    output_dir = (
        Path(output_dir_override)
        if output_dir_override is not None
        else ensure_output_dir(batch.config.results_dir, run_group_id)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_metadata = replace(batch.experiment_metadata, run_group_id=run_group_id)
    run_manifest = replace(batch.run_manifest, run_group_id=run_group_id)
    dataset_summary = _build_single_dataset_summary(batch)
    written_files = {
        "experiment_metadata": write_experiment_metadata_json(output_dir, experiment_metadata),
        "episode_summaries": write_episode_summaries_jsonl(
            output_dir,
            tuple(run.episode_summary_record for run in batch.runs),
        ),
        "step_traces": write_step_traces_jsonl(
            output_dir,
            tuple(record for run in batch.runs for record in run.step_trace_records),
        ),
        "llm_call_traces": write_llm_call_traces_jsonl(
            output_dir,
            tuple(record for run in batch.runs for record in run.llm_call_trace_records),
        ),
        "tool_call_traces": write_tool_call_traces_jsonl(
            output_dir,
            tuple(record for run in batch.runs for record in run.tool_call_trace_records),
        ),
        "aggregate_summary": write_json(output_dir / "aggregate_summary.json", batch.aggregate_summary),
        "run_manifest": write_run_manifest_json(output_dir, run_manifest),
        "dataset_summary": write_json(output_dir / "dataset_summary.json", dataset_summary),
    }
    return experiment_metadata, output_dir, written_files


def write_demand_backtest_panel_artifacts(
    config_path: str | Path = "configs/experiment/real_demand_backtest_panel.toml",
    *,
    mode: str = "all",
    llm_client_mode_override: str | None = None,
    output_dir_override: str | Path | None = None,
) -> DemandBacktestPanelRun:
    """Run and write a fixed real-demand panel while preserving per-slice artifacts."""

    panel_config = load_real_demand_backtest_panel_config(config_path)
    panel_config_hash = hash_jsonable(
        {
            "experiment_name": panel_config.experiment_name,
            "benchmark_config_path": str(panel_config.benchmark_config_path),
            "agent_config_path": str(panel_config.agent_config_path),
            "discovery_module": panel_config.discovery_module,
            "dataset_root": (
                str(panel_config.dataset_root) if panel_config.dataset_root is not None else None
            ),
            "mode_set": list(panel_config.mode_set),
            "results_dir": str(panel_config.results_dir),
            "slices": [
                {
                    "name": slice_config.name,
                    "dataset_name": slice_config.dataset_name,
                    "demand_csv_path": str(slice_config.demand_csv_path),
                    "leadtime_csv_path": str(slice_config.leadtime_csv_path),
                    "sku_list_path": str(slice_config.sku_list_path),
                    "selected_skus": list(slice_config.selected_skus),
                    "selected_sku_count": slice_config.selected_sku_count,
                    "subset_selection": slice_config.subset_selection,
                    "training_window_days": slice_config.training_window_days,
                    "history_window_days": slice_config.history_window_days,
                    "forecast_update_window_days": slice_config.forecast_update_window_days,
                    "evaluation_horizon_days": slice_config.evaluation_horizon_days,
                    "evaluation_start_date": slice_config.evaluation_start_date,
                    "roll_forward_stride_days": slice_config.roll_forward_stride_days,
                }
                for slice_config in panel_config.slices
            ],
        }
    )
    run_group_id = (
        f"{panel_config.experiment_name}_{mode}_{utc_timestamp().replace(':', '').replace('-', '')}"
    )
    panel_output_dir = (
        Path(output_dir_override)
        if output_dir_override is not None
        else ensure_output_dir(panel_config.results_dir, run_group_id)
    )
    panel_output_dir.mkdir(parents=True, exist_ok=True)
    slice_results: list[DemandBacktestPanelSliceResult] = []
    successful_batches: list[DemandBacktestBatch] = []
    for slice_config in panel_config.resolved_slice_configs():
        slice_output_dir = panel_output_dir / "slices" / (slice_config.slice_name or slice_config.dataset_name)
        try:
            batch = _run_real_demand_backtest_batch_from_config(
                slice_config,
                mode=mode,
                llm_client_mode_override=llm_client_mode_override,
            )
            metadata, _, written_files = _write_demand_backtest_batch_artifacts(
                batch,
                output_dir_override=slice_output_dir,
            )
            successful_batches.append(batch)
            slice_results.append(
                DemandBacktestPanelSliceResult(
                    slice_name=slice_config.slice_name or slice_config.dataset_name,
                    config=slice_config,
                    output_dir=slice_output_dir,
                    experiment_metadata=metadata,
                    written_files=tuple(written_files.items()),
                    success=True,
                )
            )
        except Exception as exc:  # pragma: no cover - exercised by live panel failures
            slice_results.append(
                DemandBacktestPanelSliceResult(
                    slice_name=slice_config.slice_name or slice_config.dataset_name,
                    config=slice_config,
                    output_dir=None,
                    experiment_metadata=None,
                    written_files=(),
                    success=False,
                    error_message=str(exc),
                )
            )
    panel_summary = _build_real_demand_panel_summary(
        panel_config=panel_config,
        panel_config_hash=panel_config_hash,
        run_group_id=run_group_id,
        slice_results=tuple(slice_results),
        successful_batches=tuple(successful_batches),
    )
    write_json(panel_output_dir / "panel_summary.json", panel_summary)
    if not successful_batches:
        raise RuntimeError("No real-demand panel slices completed successfully.")
    panel_batch = _build_real_demand_panel_batch(
        panel_config=panel_config,
        panel_config_hash=panel_config_hash,
        successful_batches=tuple(successful_batches),
    )
    experiment_metadata = replace(panel_batch.experiment_metadata, run_group_id=run_group_id)
    run_manifest = replace(panel_batch.run_manifest, run_group_id=run_group_id)
    write_experiment_metadata_json(panel_output_dir, experiment_metadata)
    write_episode_summaries_jsonl(
        panel_output_dir,
        tuple(run.episode_summary_record for run in panel_batch.runs),
    )
    write_step_traces_jsonl(
        panel_output_dir,
        tuple(record for run in panel_batch.runs for record in run.step_trace_records),
    )
    write_llm_call_traces_jsonl(
        panel_output_dir,
        tuple(record for run in panel_batch.runs for record in run.llm_call_trace_records),
    )
    write_tool_call_traces_jsonl(
        panel_output_dir,
        tuple(record for run in panel_batch.runs for record in run.tool_call_trace_records),
    )
    write_json(panel_output_dir / "aggregate_summary.json", panel_batch.aggregate_summary)
    write_run_manifest_json(panel_output_dir, run_manifest)
    write_json(
        panel_output_dir / "dataset_summary.json",
        {
            "panel_config_hash": panel_config_hash,
            "slice_count": len(slice_results),
            "successful_slice_count": sum(1 for result in slice_results if result.success),
            "failed_slice_count": sum(1 for result in slice_results if not result.success),
            "slices": panel_summary["slices"],
        },
    )
    return DemandBacktestPanelRun(
        panel_config=panel_config,
        panel_config_hash=panel_config_hash,
        output_dir=panel_output_dir,
        slice_results=tuple(slice_results),
        aggregate_summary=panel_batch.aggregate_summary,
        experiment_metadata=experiment_metadata,
        run_manifest=run_manifest,
    )


def _run_mode_backtest(
    *,
    config: RealDemandBacktestConfig,
    agent_config: AgentConfig,
    benchmark_case: SerialBenchmarkCase,
    dataset: RealDemandSeries,
    window: BacktestWindow,
    mode_name: str,
) -> DemandBacktestRun:
    mission = _build_mission()
    runtime: OrchestrationRuntime | None = None
    llm_provider: str | None = None
    if mode_name != "deterministic_baseline":
        runtime, llm_provider = _build_runtime(agent_config, mode=mode_name)
    optimizer = TrustedOptimizerAdapter()
    baseline_policy = DeterministicBaselinePolicy()
    system_state = build_initial_simulation_state(benchmark_case, regime_label=RegimeLabel.NORMAL)
    records: list[PeriodTraceRecord] = []
    step_trace_records: list[StepTraceRecord] = []
    llm_call_trace_records: list[LLMCallTraceRecord] = []
    tool_call_trace_records: list[ToolCallTraceRecord] = []
    responses: list[OrchestrationResponse] = []
    scenario_update_history: list[tuple[UpdateRequestType, ...]] = []
    inferred_schedule: list[str] = []
    demand_baseline_value = float(
        sum(
            dataset.demand_values[
                window.training_start_index : window.training_end_index + 1
            ]
        )
        / config.training_window_days
    )
    leadtime_baseline_value = float(
        sum(
            dataset.leadtime_values[
                window.training_start_index : window.training_end_index + 1
            ]
        )
        / config.training_window_days
    )
    schedule_name = config.slice_name or dataset.dataset_name
    run_id = (
        f"{mode_name}_{schedule_name}_{window.evaluation_start_index}_{window.evaluation_end_index}"
    )
    for absolute_index in range(window.evaluation_start_index, window.evaluation_end_index + 1):
        local_time_index = absolute_index - window.evaluation_start_index
        demand_history = tuple(
            dataset.demand_values[
                max(0, absolute_index - window.history_window_days + 1) : absolute_index + 1
            ]
        )
        leadtime_history = tuple(
            dataset.leadtime_values[
                max(0, absolute_index - window.history_window_days + 1) : absolute_index + 1
            ]
        )
        inferred_regime = infer_real_series_regime(
            demand_value=demand_history[-1],
            leadtime_value=leadtime_history[-1],
            demand_baseline_value=demand_baseline_value,
            leadtime_baseline_value=leadtime_baseline_value,
            previous_regime=system_state.regime_label,
        )
        inferred_schedule.append(inferred_regime.value)
        system_state = replace(
            system_state,
            time_index=local_time_index,
            regime_label=inferred_regime,
        )
        observation = Observation(
            time_index=local_time_index,
            demand_evidence=DemandEvidence(
                history=demand_history,
                latest_realization=(demand_history[-1],),
                stage_index=1,
            ),
            leadtime_evidence=LeadTimeEvidence(
                history=leadtime_history,
                latest_realization=(leadtime_history[-1],),
                upstream_stage_index=2,
                downstream_stage_index=1,
            ),
            regime_label=inferred_regime,
            notes=("real_demand_backtest", dataset.dates[absolute_index]),
        )
        evidence = RuntimeEvidence(
            time_index=local_time_index,
            demand=observation.demand_evidence,
            leadtime=observation.leadtime_evidence,
            scenario_families=benchmark_case.stockpyl_instance.scenario_families,
            demand_baseline_value=demand_baseline_value,
            leadtime_baseline_value=leadtime_baseline_value,
            notes=(dataset.dataset_name, dataset.dates[absolute_index]),
        )
        response: OrchestrationResponse | None = None
        if mode_name == "deterministic_baseline":
            decision = baseline_policy.decide(system_state, observation, evidence)
            scenario_update_result = decision.scenario_update_result
            agent_signal = decision.signal
        else:
            request = _build_orchestration_request(
                mission=mission,
                system_state=system_state,
                observation=observation,
                evidence=evidence,
                recent_regime_history=tuple(record.ending_state.regime_label for record in records[-3:]),
                recent_stress_reference_demand_value=demand_baseline_value,
                recent_update_request_history=tuple(scenario_update_history[-3:]),
            )
            response = runtime.run(request)
            responses.append(response)
            scenario_update_result = _extract_scenario_update_result(response)
            if scenario_update_result is None:
                fallback_regime = (
                    response.agent_assessment.regime_label
                    if response.agent_assessment is not None
                    else inferred_regime
                )
                scenario_update_result = _default_keep_current_update(
                    fallback_regime,
                    observation,
                    provenance=f"{mode_name}_keep_current_fallback",
                )
            agent_signal = response.signal
            llm_call_trace_records.extend(
                _build_llm_call_trace_records(
                    episode_id=run_id,
                    mode=mode_name,
                    run_seed=benchmark_case.benchmark_config.random_seed,
                    response=response,
                    schedule_name=schedule_name,
                    period_index=local_time_index,
                )
            )
            tool_call_trace_records.extend(
                _build_tool_call_trace_records(
                    episode_id=run_id,
                    mode=mode_name,
                    run_seed=benchmark_case.benchmark_config.random_seed,
                    response=response,
                    schedule_name=schedule_name,
                    period_index=local_time_index,
                )
            )
        scenario_update_history.append(scenario_update_result.applied_update_types)
        optimization_request = build_optimization_request(
            system_state=system_state,
            scenario_update_result=scenario_update_result,
            base_stock_levels=benchmark_case.base_stock_levels,
            planning_horizon=config.forecast_update_window_days,
        )
        counterfactual_request = build_optimization_request(
            system_state=system_state,
            scenario_update_result=_default_keep_current_update(
                inferred_regime,
                observation,
                provenance=f"{mode_name}_counterfactual_keep_current",
            ),
            base_stock_levels=benchmark_case.base_stock_levels,
            planning_horizon=config.forecast_update_window_days,
        )
        optimization_result = optimizer.solve(optimization_request)
        counterfactual_result = optimizer.solve(counterfactual_request)
        zero_order_result = OptimizationResult(
            replenishment_orders=tuple(0.0 for _ in benchmark_case.stage_names),
            planning_horizon=config.forecast_update_window_days,
        )
        transition = advance_serial_state(
            benchmark_case,
            system_state,
            observation,
            optimization_result,
            next_regime=inferred_regime,
        )
        counterfactual_transition = advance_serial_state(
            benchmark_case,
            system_state,
            observation,
            counterfactual_result,
            next_regime=inferred_regime,
        )
        zero_order_transition = advance_serial_state(
            benchmark_case,
            system_state,
            observation,
            zero_order_result,
            next_regime=inferred_regime,
        )
        if response is not None:
            step_telemetry = response.step_telemetry
        else:
            step_telemetry = None
        period_record = PeriodTraceRecord(
            time_index=local_time_index,
            regime_label=inferred_regime,
            state=system_state,
            observation=observation,
            agent_signal=agent_signal,
            optimization_result=optimization_result,
            next_state=transition.next_state,
            realized_demand=transition.realized_demand,
            demand_load=transition.demand_load,
            served_demand=transition.served_demand,
            unmet_demand=transition.unmet_demand,
            step_telemetry=step_telemetry,
            notes=(dataset.dates[absolute_index], "real_demand_backtest"),
        )
        records.append(period_record)
        step_trace_records.append(
            _build_step_trace_record(
                episode_id=run_id,
                mode=mode_name,
                schedule_name=schedule_name,
                run_seed=benchmark_case.benchmark_config.random_seed,
                period_record=period_record,
                response=response,
                scenario_update_result=scenario_update_result,
                optimization_request=optimization_request,
                counterfactual_request=counterfactual_request,
                transition=transition,
                zero_order_transition=zero_order_transition,
                cost_config=benchmark_case.benchmark_config.costs,
            )
        )
        system_state = transition.next_state
    trace = EpisodeTrace(
        run_id=run_id,
        benchmark_id=benchmark_case.benchmark_id,
        period_records=tuple(records),
    )
    rollout_metrics = compute_rollout_metrics(
        trace,
        benchmark_case.benchmark_config.costs,
        holding_cost_by_stage=benchmark_case.holding_costs,
        stockout_cost_by_stage=benchmark_case.stockout_costs,
    )
    episode_telemetry = summarize_episode_telemetry(trace.step_telemetry)
    summary = BenchmarkRunSummary(
        run_id=run_id,
        benchmark_id=benchmark_case.benchmark_id,
        benchmark_source=VALIDATION_LANE,
        topology=benchmark_case.topology,
        echelon_count=benchmark_case.echelon_count,
        intervention_summary=summarize_interventions((trace,)),
        trace_summary=summarize_traces((trace,)),
        total_cost=rollout_metrics.total_cost,
        fill_rate=rollout_metrics.fill_rate,
        average_inventory=rollout_metrics.average_inventory,
        average_backorder_level=rollout_metrics.average_backorder_level,
        optimizer_order_boundary_preserved=_optimizer_boundary_preserved(tuple(responses)),
        episode_telemetry=episode_telemetry,
        decision_quality=None,
        notes=(
            f"dataset={dataset.dataset_name}",
            f"selected_skus={','.join(dataset.selected_skus)}",
            f"evaluation_start={dataset.dates[window.evaluation_start_index]}",
            f"evaluation_end={dataset.dates[window.evaluation_end_index]}",
        ),
    )
    episode_summary_record = build_episode_summary_record(
        summary,
        mode=mode_name,
        validation_lane=VALIDATION_LANE,
        tool_ablation_variant="full",
        schedule_name=schedule_name,
        run_seed=benchmark_case.benchmark_config.random_seed,
        regime_schedule=tuple(inferred_schedule),
        cost_breakdown=CostBreakdownRecord(
            holding_cost=rollout_metrics.holding_cost,
            backlog_cost=rollout_metrics.backlog_cost,
            ordering_cost=rollout_metrics.ordering_cost,
            other_cost=rollout_metrics.other_cost,
        ),
        intervention_count=_intervention_count(tuple(step_trace_records)),
        invalid_output_count=episode_telemetry.invalid_output_count,
        fallback_count=episode_telemetry.fallback_count,
        rollout_fidelity_gate_passed=True,
        operational_metrics_gate_passed=True,
    )
    return DemandBacktestRun(
        mode=mode_name,
        slice_name=(config.slice_name or dataset.dataset_name),
        dataset_name=dataset.dataset_name,
        selected_skus=dataset.selected_skus,
        window=window,
        summary=summary,
        rollout_metrics=rollout_metrics,
        trace=trace,
        episode_summary_record=episode_summary_record,
        step_trace_records=tuple(step_trace_records),
        llm_call_trace_records=tuple(llm_call_trace_records),
        tool_call_trace_records=tuple(tool_call_trace_records),
        orchestration_responses=tuple(responses),
        llm_provider=llm_provider,
        llm_model_name=(agent_config.llm_model_name if mode_name == "llm_orchestrator" else None),
        prompt_version=(PROMPT_VERSION if mode_name == "llm_orchestrator" else None),
        prompt_hash=(prompt_contract_hash() if mode_name == "llm_orchestrator" else None),
    )


def _build_single_dataset_summary(batch: DemandBacktestBatch) -> dict[str, object]:
    """Build a compact dataset summary for one real-demand slice run."""

    return {
        "slice_name": batch.config.slice_name or batch.dataset.dataset_name,
        "dataset_name": batch.dataset.dataset_name,
        "selected_skus": list(batch.dataset.selected_skus),
        "selection_strategy": batch.dataset.selection_strategy,
        "realized_subset_mean": batch.dataset.realized_subset_mean,
        "realized_leadtime_mean": batch.dataset.realized_leadtime_mean,
        "date_range": [
            batch.dataset.dates[batch.window.evaluation_start_index],
            batch.dataset.dates[batch.window.evaluation_end_index],
        ],
        "window": {
            "training_start_index": batch.window.training_start_index,
            "training_end_index": batch.window.training_end_index,
            "evaluation_start_index": batch.window.evaluation_start_index,
            "evaluation_end_index": batch.window.evaluation_end_index,
            "history_window_days": batch.window.history_window_days,
            "forecast_update_window_days": batch.window.forecast_update_window_days,
            "evaluation_horizon_days": batch.window.evaluation_horizon_days,
        },
    }


def _build_mission() -> MissionSpec:
    return MissionSpec(
        mission_id="real_demand_backtest",
        objective=(
            "Inspect bounded real-demand evidence, manage uncertainty conservatively, "
            "and keep replenishment orders inside the trusted optimizer boundary."
        ),
        admissible_tool_ids=("forecast_tool", "leadtime_tool", "scenario_tool"),
    )


def _build_runtime(
    agent_config: AgentConfig,
    *,
    mode: str,
) -> tuple[OrchestrationRuntime, str | None]:
    tools = (
        DeterministicForecastTool(),
        DeterministicLeadTimeTool(),
        DeterministicScenarioTool(),
    )
    if mode == "deterministic_orchestrator":
        return OrchestrationRuntime(agent_config=agent_config, tools=tools), None
    if mode == "llm_orchestrator":
        client_mode = "real" if agent_config.llm_client_mode == "openai" else agent_config.llm_client_mode
        if client_mode == "real":
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
    raise ValueError(f"Unsupported backtest mode: {mode!r}.")


def _build_orchestration_request(
    *,
    mission: MissionSpec,
    system_state: SimulationState,
    observation: Observation,
    evidence: RuntimeEvidence,
    recent_regime_history: tuple[RegimeLabel, ...],
    recent_stress_reference_demand_value: float,
    recent_update_request_history: tuple[tuple[UpdateRequestType, ...], ...],
):
    from meio.agents.runtime import OrchestrationMemory, OrchestrationRequest

    return OrchestrationRequest(
        mission=mission,
        system_state=system_state,
        observation=observation,
        evidence=evidence,
        requested_subgoal=OperationalSubgoal.INSPECT_EVIDENCE,
        candidate_tool_ids=mission.admissible_tool_ids,
        recent_regime_history=recent_regime_history,
        recent_stress_reference_demand_value=recent_stress_reference_demand_value,
        recent_update_request_history=recent_update_request_history,
        memory=OrchestrationMemory(),
    )


def _extract_scenario_update_result(response: OrchestrationResponse) -> ScenarioUpdateResult | None:
    for tool_result in reversed(response.tool_results):
        value = tool_result.structured_output.get("scenario_update_result")
        if isinstance(value, ScenarioUpdateResult):
            return value
    return None


def _default_keep_current_update(
    regime_label: RegimeLabel,
    observation: Observation,
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


def _build_step_trace_record(
    *,
    episode_id: str,
    mode: str,
    schedule_name: str,
    run_seed: int,
    period_record: PeriodTraceRecord,
    response: OrchestrationResponse | None,
    scenario_update_result: ScenarioUpdateResult,
    optimization_request: OptimizationRequest,
    counterfactual_request: OptimizationRequest,
    transition,
    zero_order_transition,
    cost_config,
) -> StepTraceRecord:
    predicted_regime_label = None
    confidence = None
    if response is not None and response.agent_assessment is not None:
        predicted_regime_label = response.agent_assessment.regime_label.value
        confidence = response.agent_assessment.confidence
    return StepTraceRecord(
        episode_id=episode_id,
        mode=mode,
        tool_ablation_variant="full",
        schedule_name=schedule_name,
        run_seed=run_seed,
        period_index=period_record.time_index,
        true_regime_label="unlabeled_real_demand",
        predicted_regime_label=predicted_regime_label,
        confidence=confidence,
        selected_subgoal=period_record.agent_signal.selected_subgoal.value,
        selected_tools=period_record.agent_signal.tool_sequence,
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
        per_period_cost=compute_period_total_cost(period_record, cost_config),
        per_period_fill_rate=compute_period_fill_rate(period_record),
        decision_changed_optimizer_input=_decision_changed_optimizer_input(
            optimization_request,
            counterfactual_request,
        ),
        optimizer_output_changed_state=_transition_outcomes_differ(
            transition,
            zero_order_transition,
        ),
        intervention_changed_outcome=_transition_outcomes_differ(
            transition,
            zero_order_transition,
        ),
        validation_lane=VALIDATION_LANE,
    )


def _build_llm_call_trace_records(
    *,
    episode_id: str,
    mode: str,
    run_seed: int,
    response: OrchestrationResponse,
    schedule_name: str,
    period_index: int,
) -> tuple[LLMCallTraceRecord, ...]:
    diagnostics = response.llm_diagnostics
    if diagnostics is None or diagnostics.llm_call_trace is None:
        return ()
    trace = diagnostics.llm_call_trace
    return (
        LLMCallTraceRecord(
            episode_id=episode_id,
            mode=mode,
            tool_ablation_variant="full",
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
            prompt_tokens=trace.prompt_tokens,
            completion_tokens=trace.completion_tokens,
            total_tokens=trace.total_tokens,
            latency_ms=trace.latency_ms,
            requested_tool_ids=trace.requested_tool_ids,
            unavailable_tool_ids=trace.unavailable_tool_ids,
            retry_count=trace.retry_count,
            client_error_category=(
                trace.client_error_category.value
                if trace.client_error_category is not None
                else None
            ),
            client_error_message=trace.client_error_message,
            failure_after_response=trace.failure_after_response,
            validation_lane=VALIDATION_LANE,
        ),
    )


def _build_tool_call_trace_records(
    *,
    episode_id: str,
    mode: str,
    run_seed: int,
    response: OrchestrationResponse,
    schedule_name: str,
    period_index: int,
) -> tuple[ToolCallTraceRecord, ...]:
    return tuple(
        ToolCallTraceRecord(
            episode_id=episode_id,
            mode=mode,
            tool_ablation_variant="full",
            schedule_name=schedule_name,
            run_seed=run_seed,
            period_index=period_index,
            call_index=call_index,
            tool_id=trace.tool_id,
            tool_input=trace.tool_input,
            tool_output=trace.tool_output,
            success=trace.success,
            error_type=trace.error_type,
            latency_ms=trace.latency_ms,
            pre_tool_decision=trace.pre_tool_decision,
            post_tool_decision=trace.post_tool_decision,
            pre_tool_optimizer_input=trace.pre_tool_optimizer_input,
            post_tool_optimizer_input=trace.post_tool_optimizer_input,
            decision_changed=trace.decision_changed,
            optimizer_input_changed=trace.optimizer_input_changed,
            validation_lane=VALIDATION_LANE,
        )
        for call_index, trace in enumerate(response.tool_call_traces)
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


def _optimization_request_snapshot(request: OptimizationRequest) -> dict[str, object]:
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


def _decision_changed_optimizer_input(
    optimization_request: OptimizationRequest,
    counterfactual_request: OptimizationRequest,
) -> bool:
    return _optimization_request_snapshot(optimization_request) != _optimization_request_snapshot(
        counterfactual_request
    )


def _transition_outcomes_differ(left_transition, right_transition) -> bool:
    return (
        left_transition.next_state.inventory_level != right_transition.next_state.inventory_level
        or left_transition.next_state.pipeline_inventory
        != right_transition.next_state.pipeline_inventory
        or left_transition.next_state.backorder_level != right_transition.next_state.backorder_level
    )


def _intervention_count(step_trace_records: tuple[StepTraceRecord, ...]) -> int:
    return sum(
        1
        for record in step_trace_records
        if record.request_replan or bool(record.selected_tools) or record.decision_changed_optimizer_input
    )


def _build_real_demand_panel_summary(
    *,
    panel_config: RealDemandBacktestPanelConfig,
    panel_config_hash: str,
    run_group_id: str,
    slice_results: tuple[DemandBacktestPanelSliceResult, ...],
    successful_batches: tuple[DemandBacktestBatch, ...],
) -> dict[str, object]:
    successful_by_name = {
        batch.config.slice_name or batch.dataset.dataset_name: batch
        for batch in successful_batches
    }
    slices: list[dict[str, object]] = []
    for result in slice_results:
        batch = successful_by_name.get(result.slice_name)
        if batch is None:
            slices.append(
                {
                    "slice_name": result.slice_name,
                    "dataset_name": result.config.dataset_name,
                    "selected_skus": list(result.config.selected_skus),
                    "evaluation_start_date": result.config.evaluation_start_date,
                    "evaluation_horizon_days": result.config.evaluation_horizon_days,
                    "success": False,
                    "error_message": result.error_message,
                }
            )
            continue
        single_summary = _build_single_dataset_summary(batch)
        single_summary.update(
            {
                "success": True,
                "slice_name": result.slice_name,
                "output_dir": str(result.output_dir) if result.output_dir is not None else None,
            }
        )
        slices.append(single_summary)
    return {
        "panel_name": panel_config.experiment_name,
        "panel_config_hash": panel_config_hash,
        "run_group_id": run_group_id,
        "mode_set": list(panel_config.mode_set),
        "results_dir": str(panel_config.results_dir),
        "slice_count": len(slice_results),
        "successful_slice_count": sum(1 for result in slice_results if result.success),
        "failed_slice_count": sum(1 for result in slice_results if not result.success),
        "slices": slices,
    }


def _build_real_demand_panel_batch(
    *,
    panel_config: RealDemandBacktestPanelConfig,
    panel_config_hash: str,
    successful_batches: tuple[DemandBacktestBatch, ...],
) -> DemandBacktestBatch:
    benchmark_case = build_serial_benchmark_case(
        load_benchmark_config(panel_config.benchmark_config_path)
    )
    runs = tuple(
        run
        for batch in successful_batches
        for run in batch.runs
    )
    records_by_mode = {
        mode_name: tuple(
            run.episode_summary_record for run in runs if run.mode == mode_name
        )
        for mode_name in panel_config.mode_set
    }
    step_records_by_mode = {
        mode_name: tuple(
            record
            for run in runs
            if run.mode == mode_name
            for record in run.step_trace_records
        )
        for mode_name in panel_config.mode_set
    }
    tool_call_records_by_mode = {
        mode_name: tuple(
            record
            for run in runs
            if run.mode == mode_name
            for record in run.tool_call_trace_records
        )
        for mode_name in panel_config.mode_set
    }
    aggregate_summary = _classify_batch_summary(
        aggregate_batch_episode_summaries(
            benchmark_id=benchmark_case.benchmark_id,
            validation_lane=VALIDATION_LANE,
            records_by_mode=records_by_mode,
            step_trace_records_by_mode=step_records_by_mode,
            tool_call_records_by_mode=tool_call_records_by_mode,
        ),
        benchmark_source=VALIDATION_LANE,
        experiment_name=panel_config.experiment_name,
        llm_provider_by_mode={run.mode: run.llm_provider for run in runs},
    )
    experiment_metadata, run_manifest = _build_panel_artifact_headers(
        panel_config=panel_config,
        panel_config_hash=panel_config_hash,
        benchmark_case=benchmark_case,
        aggregate_summary=aggregate_summary,
        runs=runs,
    )
    reference_batch = successful_batches[0]
    return DemandBacktestBatch(
        config=reference_batch.config,
        dataset=reference_batch.dataset,
        window=reference_batch.window,
        runs=runs,
        aggregate_summary=aggregate_summary,
        experiment_metadata=experiment_metadata,
        run_manifest=run_manifest,
    )


def _classify_batch_summary(
    summary: BatchAggregateSummary,
    *,
    benchmark_source: str,
    experiment_name: str,
    llm_provider_by_mode: dict[str, str | None],
) -> BatchAggregateSummary:
    from dataclasses import replace as dataclass_replace

    mode_decisions = []
    classified_mode_summaries = []
    for mode_summary in summary.mode_summaries:
        provider = llm_provider_by_mode.get(mode_summary.mode)
        governance = classify_artifact_governance(
            experiment_name=experiment_name,
            benchmark_source=benchmark_source,
            provider=provider,
            validity_gate_passed=compute_validity_gate_passed(
                optimizer_order_boundary_preserved=(
                    mode_summary.validity_summary.optimizer_order_boundary_preserved
                ),
                invalid_output_count=mode_summary.validity_summary.invalid_output_count,
                fallback_count=mode_summary.validity_summary.fallback_count,
            ),
            rollout_fidelity_gate_passed=mode_summary.validity_summary.rollout_fidelity_gate_passed,
            operational_metrics_gate_passed=(
                mode_summary.validity_summary.operational_metrics_gate_passed
            ),
        )
        mode_decisions.append(governance)
        classified_mode_summaries.append(
            dataclass_replace(
                mode_summary,
                artifact_use_class=governance.artifact_use_class,
                validity_gate_passed=governance.validity_gate_passed,
                eligibility_notes=governance.eligibility_notes,
            )
        )
    directory_governance = summarize_directory_governance(tuple(mode_decisions))
    return dataclass_replace(
        summary,
        mode_summaries=tuple(classified_mode_summaries),
        artifact_use_class=directory_governance.artifact_use_class,
        validity_gate_passed=directory_governance.validity_gate_passed,
        eligibility_notes=directory_governance.eligibility_notes,
    )


def _build_artifact_headers(
    *,
    config: RealDemandBacktestConfig,
    benchmark_case: SerialBenchmarkCase,
    aggregate_summary: BatchAggregateSummary,
    runs: tuple[DemandBacktestRun, ...],
) -> tuple[ExperimentMetadata, RunManifestRecord]:
    resolved_config = {
        "experiment_name": config.experiment_name,
        "benchmark_config_path": str(config.benchmark_config_path),
        "agent_config_path": str(config.agent_config_path),
        "dataset_name": config.dataset_name,
        "slice_name": config.slice_name,
        "selected_skus": list(config.selected_skus),
        "selected_sku_count": config.selected_sku_count,
        "subset_selection": config.subset_selection,
        "training_window_days": config.training_window_days,
        "history_window_days": config.history_window_days,
        "forecast_update_window_days": config.forecast_update_window_days,
        "evaluation_horizon_days": config.evaluation_horizon_days,
        "evaluation_start_date": config.evaluation_start_date,
        "mode_set": list(config.mode_set),
        "results_dir": str(config.results_dir),
    }
    config_hash = hash_jsonable(resolved_config)
    mode_names = tuple(run.mode for run in runs)
    provider_values = tuple(run.llm_provider for run in runs if run.llm_provider is not None)
    model_values = tuple(run.llm_model_name for run in runs if run.llm_model_name is not None)
    prompt_values = tuple(run.prompt_version for run in runs if run.prompt_version is not None)
    prompt_hash_values = tuple(run.prompt_hash for run in runs if run.prompt_hash is not None)
    experiment_metadata = ExperimentMetadata(
        experiment_id=config.experiment_name,
        run_group_id="pending",
        timestamp=utc_timestamp(),
        git_commit_sha=None,
        resolved_config=resolved_config,
        config_hash=config_hash,
        benchmark_id=benchmark_case.benchmark_id,
        benchmark_source=VALIDATION_LANE,
        mode=("all" if len(mode_names) > 1 else mode_names[0]),
        provider=(provider_values[0] if len(set(provider_values)) == 1 else ("mixed" if provider_values else None)),
        model_name=(model_values[0] if len(set(model_values)) == 1 else ("mixed" if model_values else None)),
        validation_lane=VALIDATION_LANE,
        artifact_use_class=aggregate_summary.artifact_use_class,
        validity_gate_passed=aggregate_summary.validity_gate_passed,
        rollout_fidelity_gate_passed=all(
            summary.validity_summary.rollout_fidelity_gate_passed
            for summary in aggregate_summary.mode_summaries
        ),
        operational_metrics_gate_passed=all(
            summary.validity_summary.operational_metrics_gate_passed
            for summary in aggregate_summary.mode_summaries
        ),
        eligibility_notes=aggregate_summary.eligibility_notes,
        prompt_version=(prompt_values[0] if len(set(prompt_values)) == 1 else None),
        prompt_hash=(prompt_hash_values[0] if len(set(prompt_hash_values)) == 1 else None),
        benchmark_random_seed=benchmark_case.benchmark_config.random_seed,
    )
    run_manifest = RunManifestRecord(
        experiment_id=config.experiment_name,
        run_group_id="pending",
        config_hash=config_hash,
        benchmark_id=benchmark_case.benchmark_id,
        benchmark_source=VALIDATION_LANE,
        validation_lane=VALIDATION_LANE,
        benchmark_config_path=str(config.benchmark_config_path),
        agent_config_path=str(config.agent_config_path),
        schedule_names=((config.slice_name or config.dataset_name),),
        seed_values=(benchmark_case.benchmark_config.random_seed,),
        mode_names=mode_names,
        tool_ablation_variants=("full",),
        provider=experiment_metadata.provider,
        model_name=experiment_metadata.model_name,
        artifact_use_class=aggregate_summary.artifact_use_class,
        validity_gate_passed=aggregate_summary.validity_gate_passed,
        rollout_fidelity_gate_passed=experiment_metadata.rollout_fidelity_gate_passed,
        operational_metrics_gate_passed=experiment_metadata.operational_metrics_gate_passed,
        eligibility_notes=aggregate_summary.eligibility_notes,
        mode_artifact_use_classes=tuple(
            (summary.mode, summary.artifact_use_class.value)
            for summary in aggregate_summary.mode_summaries
        ),
        prompt_version=experiment_metadata.prompt_version,
        prompt_hash=experiment_metadata.prompt_hash,
        artifact_filenames=(
            ("aggregate_summary", "aggregate_summary.json"),
            ("dataset_summary", "dataset_summary.json"),
            ("episode_summaries", "episode_summaries.jsonl"),
            ("experiment_metadata", "experiment_metadata.json"),
            ("llm_call_traces", "llm_call_traces.jsonl"),
            ("run_manifest", "run_manifest.json"),
            ("step_traces", "step_traces.jsonl"),
            ("tool_call_traces", "tool_call_traces.jsonl"),
        ),
    )
    return experiment_metadata, run_manifest


def _build_panel_artifact_headers(
    *,
    panel_config: RealDemandBacktestPanelConfig,
    panel_config_hash: str,
    benchmark_case: SerialBenchmarkCase,
    aggregate_summary: BatchAggregateSummary,
    runs: tuple[DemandBacktestRun, ...],
) -> tuple[ExperimentMetadata, RunManifestRecord]:
    resolved_config = {
        "experiment_name": panel_config.experiment_name,
        "benchmark_config_path": str(panel_config.benchmark_config_path),
        "agent_config_path": str(panel_config.agent_config_path),
        "discovery_module": panel_config.discovery_module,
        "dataset_root": (
            str(panel_config.dataset_root) if panel_config.dataset_root is not None else None
        ),
        "mode_set": list(panel_config.mode_set),
        "results_dir": str(panel_config.results_dir),
        "panel_config_hash": panel_config_hash,
        "slice_names": sorted({run.slice_name for run in runs}),
    }
    mode_names = tuple(sorted({run.mode for run in runs}))
    provider_values = tuple(run.llm_provider for run in runs if run.llm_provider is not None)
    model_values = tuple(run.llm_model_name for run in runs if run.llm_model_name is not None)
    prompt_values = tuple(run.prompt_version for run in runs if run.prompt_version is not None)
    prompt_hash_values = tuple(run.prompt_hash for run in runs if run.prompt_hash is not None)
    schedule_names = tuple(sorted({run.slice_name for run in runs}))
    experiment_metadata = ExperimentMetadata(
        experiment_id=panel_config.experiment_name,
        run_group_id="pending",
        timestamp=utc_timestamp(),
        git_commit_sha=None,
        resolved_config=resolved_config,
        config_hash=panel_config_hash,
        benchmark_id=benchmark_case.benchmark_id,
        benchmark_source=VALIDATION_LANE,
        mode=("all" if len(mode_names) > 1 else mode_names[0]),
        provider=(provider_values[0] if len(set(provider_values)) == 1 else ("mixed" if provider_values else None)),
        model_name=(model_values[0] if len(set(model_values)) == 1 else ("mixed" if model_values else None)),
        validation_lane=VALIDATION_LANE,
        artifact_use_class=aggregate_summary.artifact_use_class,
        validity_gate_passed=aggregate_summary.validity_gate_passed,
        rollout_fidelity_gate_passed=all(
            summary.validity_summary.rollout_fidelity_gate_passed
            for summary in aggregate_summary.mode_summaries
        ),
        operational_metrics_gate_passed=all(
            summary.validity_summary.operational_metrics_gate_passed
            for summary in aggregate_summary.mode_summaries
        ),
        eligibility_notes=aggregate_summary.eligibility_notes,
        prompt_version=(prompt_values[0] if len(set(prompt_values)) == 1 else None),
        prompt_hash=(prompt_hash_values[0] if len(set(prompt_hash_values)) == 1 else None),
        benchmark_random_seed=benchmark_case.benchmark_config.random_seed,
    )
    run_manifest = RunManifestRecord(
        experiment_id=panel_config.experiment_name,
        run_group_id="pending",
        config_hash=panel_config_hash,
        benchmark_id=benchmark_case.benchmark_id,
        benchmark_source=VALIDATION_LANE,
        validation_lane=VALIDATION_LANE,
        benchmark_config_path=str(panel_config.benchmark_config_path),
        agent_config_path=str(panel_config.agent_config_path),
        schedule_names=schedule_names,
        seed_values=(benchmark_case.benchmark_config.random_seed,),
        mode_names=mode_names,
        tool_ablation_variants=("full",),
        provider=experiment_metadata.provider,
        model_name=experiment_metadata.model_name,
        artifact_use_class=aggregate_summary.artifact_use_class,
        validity_gate_passed=aggregate_summary.validity_gate_passed,
        rollout_fidelity_gate_passed=experiment_metadata.rollout_fidelity_gate_passed,
        operational_metrics_gate_passed=experiment_metadata.operational_metrics_gate_passed,
        eligibility_notes=aggregate_summary.eligibility_notes,
        mode_artifact_use_classes=tuple(
            (summary.mode, summary.artifact_use_class.value)
            for summary in aggregate_summary.mode_summaries
        ),
        prompt_version=experiment_metadata.prompt_version,
        prompt_hash=experiment_metadata.prompt_hash,
        artifact_filenames=(
            ("aggregate_summary", "aggregate_summary.json"),
            ("dataset_summary", "dataset_summary.json"),
            ("episode_summaries", "episode_summaries.jsonl"),
            ("experiment_metadata", "experiment_metadata.json"),
            ("llm_call_traces", "llm_call_traces.jsonl"),
            ("panel_summary", "panel_summary.json"),
            ("run_manifest", "run_manifest.json"),
            ("step_traces", "step_traces.jsonl"),
            ("tool_call_traces", "tool_call_traces.jsonl"),
        ),
    )
    return experiment_metadata, run_manifest


__all__ = [
    "DemandBacktestBatch",
    "DemandBacktestPanelRun",
    "DemandBacktestPanelSliceResult",
    "DemandBacktestRun",
    "VALIDATION_LANE",
    "infer_real_series_regime",
    "run_real_demand_backtest_batch",
    "write_demand_backtest_panel_artifacts",
    "write_demand_backtest_artifacts",
]
