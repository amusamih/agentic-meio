"""Run the bounded public-benchmark evaluation lane."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from meio.benchmarks.public_benchmark_adapter import (
    VALIDATION_LANE,
    PublicBenchmarkExecutionBatch,
    run_public_benchmark_execution,
)
from meio.config.loaders import load_public_benchmark_eval_config
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
    ArtifactUseClass,
    ExperimentMetadata,
    RunManifestRecord,
)
from meio.evaluation.results_index import (
    classify_artifact_governance,
    summarize_directory_governance,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the bounded public benchmark evaluation lane.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment/public_benchmark_eval.toml"),
        help="Path to the public benchmark evaluation config.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        help="Run one configured mode or all configured modes.",
    )
    parser.add_argument(
        "--llm-client-mode",
        choices=("config", "fake", "real"),
        default="config",
        help="Override the configured LLM client mode.",
    )
    args = parser.parse_args()

    config = load_public_benchmark_eval_config(args.config)
    llm_client_mode_override = None if args.llm_client_mode == "config" else args.llm_client_mode
    selected_modes = config.mode_set if args.mode == "all" else (args.mode,)
    run_group_id = f"{config.experiment_name}_{args.mode}_{utc_timestamp().replace(':', '').replace('-', '')}"
    output_dir = ensure_output_dir(config.results_dir, run_group_id)
    batch = run_public_benchmark_execution(
        config=config,
        selected_modes=selected_modes,
        llm_client_mode_override=llm_client_mode_override,
        vis_root=output_dir / "benchmark_vis",
    )
    classified_aggregate_summary = _classify_batch_summary(batch)
    experiment_metadata = _build_experiment_metadata(
        config=config,
        config_path=args.config,
        run_group_id=run_group_id,
        selected_modes=selected_modes,
        batch=batch,
        aggregate_summary=classified_aggregate_summary,
    )
    run_manifest = _build_run_manifest(
        config=config,
        config_path=args.config,
        run_group_id=run_group_id,
        selected_modes=selected_modes,
        batch=batch,
        aggregate_summary=classified_aggregate_summary,
    )

    written_files = {
        "experiment_metadata": write_experiment_metadata_json(output_dir, experiment_metadata),
        "public_benchmark_summary": write_json(
            output_dir / "public_benchmark_summary.json",
            batch.evaluation_summary,
        ),
    }
    if batch.mode_artifacts:
        written_files["episode_summaries"] = write_episode_summaries_jsonl(
            output_dir,
            tuple(artifact.episode_summary_record for artifact in batch.mode_artifacts),
        )
        written_files["step_traces"] = write_step_traces_jsonl(
            output_dir,
            tuple(
                record
                for artifact in batch.mode_artifacts
                for record in artifact.step_trace_records
            ),
        )
        written_files["llm_call_traces"] = write_llm_call_traces_jsonl(
            output_dir,
            tuple(
                record
                for artifact in batch.mode_artifacts
                for record in artifact.llm_call_trace_records
            ),
        )
        written_files["tool_call_traces"] = write_tool_call_traces_jsonl(
            output_dir,
            tuple(
                record
                for artifact in batch.mode_artifacts
                for record in artifact.tool_call_trace_records
            ),
        )
    if classified_aggregate_summary is not None:
        written_files["aggregate_summary"] = write_json(
            output_dir / "aggregate_summary.json",
            classified_aggregate_summary,
        )
    written_files["run_manifest"] = write_run_manifest_json(output_dir, run_manifest)
    payload = {
        "artifacts_dir": str(output_dir),
        "artifact_files": {key: str(path) for key, path in written_files.items()},
        "experiment_metadata": jsonable(experiment_metadata),
        "public_benchmark_summary": jsonable(batch.evaluation_summary),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def _classify_batch_summary(
    batch: PublicBenchmarkExecutionBatch,
):
    if batch.aggregate_summary is None:
        return None
    from dataclasses import replace as dataclass_replace

    mode_decisions = []
    classified_mode_summaries = []
    provider_by_mode = {
        artifact.mode_summary.mode: artifact.mode_summary.llm_provider
        for artifact in batch.mode_artifacts
    }
    for mode_summary in batch.aggregate_summary.mode_summaries:
        provider = provider_by_mode.get(mode_summary.mode)
        governance = classify_artifact_governance(
            experiment_name=batch.evaluation_summary.benchmark_candidate,
            benchmark_source=VALIDATION_LANE,
            provider=provider,
            validity_gate_passed=mode_summary.validity_summary.optimizer_order_boundary_preserved
            and mode_summary.validity_summary.invalid_output_count == 0
            and mode_summary.validity_summary.fallback_count == 0,
            rollout_fidelity_gate_passed=mode_summary.validity_summary.rollout_fidelity_gate_passed,
            operational_metrics_gate_passed=mode_summary.validity_summary.operational_metrics_gate_passed,
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
        batch.aggregate_summary,
        mode_summaries=tuple(classified_mode_summaries),
        artifact_use_class=directory_governance.artifact_use_class,
        validity_gate_passed=directory_governance.validity_gate_passed,
        eligibility_notes=directory_governance.eligibility_notes,
    )


def _build_experiment_metadata(
    *,
    config,
    config_path: Path,
    run_group_id: str,
    selected_modes: tuple[str, ...],
    batch: PublicBenchmarkExecutionBatch,
    aggregate_summary,
) -> ExperimentMetadata:
    provider_values = tuple(
        artifact.mode_summary.llm_provider
        for artifact in batch.mode_artifacts
        if artifact.mode_summary.llm_provider is not None
    )
    model_values = tuple(
        artifact.mode_summary.llm_model_name
        for artifact in batch.mode_artifacts
        if artifact.mode_summary.llm_model_name is not None
    )
    resolved_config = {
        "experiment_name": config.experiment_name,
        "benchmark_candidate": config.benchmark_candidate,
        "discovery_module": config.discovery_module,
        "benchmark_root": str(config.benchmark_root) if config.benchmark_root is not None else None,
        "demo_config_path": str(config.demo_config_path),
        "agent_config_path": str(config.agent_config_path),
        "environment_config_name": config.environment_config_name,
        "wrapper_names": list(config.wrapper_names),
        "benchmark_mode": config.benchmark_mode,
        "mode_set": list(selected_modes),
        "episode_horizon_steps": config.episode_horizon_steps,
        "base_stock_multiplier": config.base_stock_multiplier,
        "demand_scale_epsilon": config.demand_scale_epsilon,
        "results_dir": str(config.results_dir),
        "benchmark_config_path": str(config_path),
    }
    if aggregate_summary is None:
        governance = classify_artifact_governance(
            experiment_name=config.experiment_name,
            benchmark_source=VALIDATION_LANE,
            provider=(provider_values[0] if len(set(provider_values)) == 1 else None),
            validity_gate_passed=False,
            rollout_fidelity_gate_passed=False,
            operational_metrics_gate_passed=False,
        )
        artifact_use_class = governance.artifact_use_class
        validity_gate_passed = governance.validity_gate_passed
        rollout_fidelity_gate_passed = False
        operational_metrics_gate_passed = False
        eligibility_notes = governance.eligibility_notes
    else:
        artifact_use_class = aggregate_summary.artifact_use_class
        validity_gate_passed = aggregate_summary.validity_gate_passed
        rollout_fidelity_gate_passed = all(
            summary.validity_summary.rollout_fidelity_gate_passed
            for summary in aggregate_summary.mode_summaries
        )
        operational_metrics_gate_passed = all(
            summary.validity_summary.operational_metrics_gate_passed
            for summary in aggregate_summary.mode_summaries
        )
        eligibility_notes = aggregate_summary.eligibility_notes
    return ExperimentMetadata(
        experiment_id=config.experiment_name,
        run_group_id=run_group_id,
        timestamp=utc_timestamp(),
        git_commit_sha=None,
        resolved_config=resolved_config,
        config_hash=hash_jsonable(resolved_config),
        benchmark_id=f"{config.benchmark_candidate}:{config.environment_config_name}",
        benchmark_source=VALIDATION_LANE,
        mode=("all" if len(selected_modes) > 1 else selected_modes[0]),
        provider=(provider_values[0] if len(set(provider_values)) == 1 else ("mixed" if provider_values else None)),
        model_name=(model_values[0] if len(set(model_values)) == 1 else ("mixed" if model_values else None)),
        validation_lane=VALIDATION_LANE,
        artifact_use_class=artifact_use_class,
        validity_gate_passed=validity_gate_passed,
        rollout_fidelity_gate_passed=rollout_fidelity_gate_passed,
        operational_metrics_gate_passed=operational_metrics_gate_passed,
        eligibility_notes=eligibility_notes,
    )


def _build_run_manifest(
    *,
    config,
    config_path: Path,
    run_group_id: str,
    selected_modes: tuple[str, ...],
    batch: PublicBenchmarkExecutionBatch,
    aggregate_summary,
) -> RunManifestRecord:
    if aggregate_summary is None:
        artifact_use_class = ArtifactUseClass.INTERNAL_ONLY
        validity_gate_passed = False
        rollout_fidelity_gate_passed = False
        operational_metrics_gate_passed = False
        eligibility_notes = ("public_benchmark_partial_integration",)
        mode_artifact_use_classes = tuple(
            (mode_name, ArtifactUseClass.INTERNAL_ONLY.value)
            for mode_name in selected_modes
        )
        artifact_filenames = (
            ("experiment_metadata", "experiment_metadata.json"),
            ("public_benchmark_summary", "public_benchmark_summary.json"),
            ("run_manifest", "run_manifest.json"),
        )
    else:
        artifact_use_class = aggregate_summary.artifact_use_class
        validity_gate_passed = aggregate_summary.validity_gate_passed
        rollout_fidelity_gate_passed = all(
            summary.validity_summary.rollout_fidelity_gate_passed
            for summary in aggregate_summary.mode_summaries
        )
        operational_metrics_gate_passed = all(
            summary.validity_summary.operational_metrics_gate_passed
            for summary in aggregate_summary.mode_summaries
        )
        eligibility_notes = aggregate_summary.eligibility_notes
        mode_artifact_use_classes = tuple(
            (summary.mode, summary.artifact_use_class.value)
            for summary in aggregate_summary.mode_summaries
        )
        artifact_filenames = (
            ("aggregate_summary", "aggregate_summary.json"),
            ("episode_summaries", "episode_summaries.jsonl"),
            ("experiment_metadata", "experiment_metadata.json"),
            ("llm_call_traces", "llm_call_traces.jsonl"),
            ("public_benchmark_summary", "public_benchmark_summary.json"),
            ("run_manifest", "run_manifest.json"),
            ("step_traces", "step_traces.jsonl"),
            ("tool_call_traces", "tool_call_traces.jsonl"),
        )
    provider_values = tuple(
        artifact.mode_summary.llm_provider
        for artifact in batch.mode_artifacts
        if artifact.mode_summary.llm_provider is not None
    )
    model_values = tuple(
        artifact.mode_summary.llm_model_name
        for artifact in batch.mode_artifacts
        if artifact.mode_summary.llm_model_name is not None
    )
    resolved_config = {
        "experiment_name": config.experiment_name,
        "benchmark_candidate": config.benchmark_candidate,
        "environment_config_name": config.environment_config_name,
        "mode_set": list(selected_modes),
        "episode_horizon_steps": config.episode_horizon_steps,
    }
    return RunManifestRecord(
        experiment_id=config.experiment_name,
        run_group_id=run_group_id,
        config_hash=hash_jsonable(resolved_config),
        benchmark_id=f"{config.benchmark_candidate}:{config.environment_config_name}",
        benchmark_source=VALIDATION_LANE,
        benchmark_config_path=str(config_path),
        agent_config_path=str(config.agent_config_path),
        schedule_names=(config.environment_config_name,),
        seed_values=(0,),
        mode_names=selected_modes,
        tool_ablation_variants=("full",),
        provider=(provider_values[0] if len(set(provider_values)) == 1 else ("mixed" if provider_values else None)),
        model_name=(model_values[0] if len(set(model_values)) == 1 else ("mixed" if model_values else None)),
        validation_lane=VALIDATION_LANE,
        artifact_use_class=artifact_use_class,
        validity_gate_passed=validity_gate_passed,
        rollout_fidelity_gate_passed=rollout_fidelity_gate_passed,
        operational_metrics_gate_passed=operational_metrics_gate_passed,
        eligibility_notes=eligibility_notes,
        mode_artifact_use_classes=mode_artifact_use_classes,
        artifact_filenames=artifact_filenames,
    )


if __name__ == "__main__":
    main()
