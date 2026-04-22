from __future__ import annotations

import json
from pathlib import Path
import shutil
from uuid import uuid4

from meio.evaluation.logging_io import write_json
from meio.evaluation.public_benchmark_repeatability import (
    analyze_public_benchmark_repeatability,
)
import scripts.analyze_public_benchmark_repeatability as repeatability_script


def _write_public_benchmark_run(
    run_dir: Path,
    *,
    config_hash: str,
    llm_provider: str,
    baseline_reward: float,
    deterministic_reward: float,
    llm_reward: float,
    llm_fill_rate: float,
) -> None:
    write_json(
        run_dir / "experiment_metadata.json",
        {
            "config_hash": config_hash,
            "artifact_use_class": "internal_only",
            "validity_gate_passed": True,
        },
    )
    write_json(
        run_dir / "public_benchmark_summary.json",
        {
            "environment_contract": {
                "environment_config_name": "sku50.single_store.standard",
                "action_mode": "demand_mean_continuous",
            },
            "mapping_assumptions": [
                "orchestrator_observation_uses_aggregate_per_sku_means",
                "trusted_optimizer_executes_one_single_stage_request_per_sku",
            ],
            "comparability_notes": [
                "public_benchmark_reward_is_profit_like_not_stockpyl_total_cost"
            ],
            "mode_summaries": [
                {
                    "mode": "deterministic_baseline",
                    "total_reward": baseline_reward,
                    "average_fill_rate": 0.60,
                    "tool_call_count": 0,
                    "llm_provider": None,
                    "llm_model_name": None,
                },
                {
                    "mode": "deterministic_orchestrator",
                    "total_reward": deterministic_reward,
                    "average_fill_rate": 0.62,
                    "tool_call_count": 30,
                    "llm_provider": None,
                    "llm_model_name": None,
                },
                {
                    "mode": "llm_orchestrator",
                    "total_reward": llm_reward,
                    "average_fill_rate": llm_fill_rate,
                    "tool_call_count": 12,
                    "llm_provider": llm_provider,
                    "llm_model_name": "gpt-4o-mini",
                },
            ],
        },
    )
    write_json(
        run_dir / "aggregate_summary.json",
        {
            "mode_summaries": [
                {
                    "mode": "deterministic_baseline",
                    "decision_quality": {"replan_rate": 0.4, "no_action_rate": 0.6},
                    "validity_summary": {
                        "fallback_count": 0,
                        "invalid_output_count": 0,
                    },
                },
                {
                    "mode": "deterministic_orchestrator",
                    "decision_quality": {"replan_rate": 0.4, "no_action_rate": 0.0},
                    "validity_summary": {
                        "fallback_count": 0,
                        "invalid_output_count": 0,
                    },
                },
                {
                    "mode": "llm_orchestrator",
                    "decision_quality": {"replan_rate": 0.4, "no_action_rate": 0.6},
                    "validity_summary": {
                        "fallback_count": 0,
                        "invalid_output_count": 0,
                    },
                },
            ]
        },
    )


def test_analyze_public_benchmark_repeatability_summarizes_identity_and_providers() -> None:
    tmp_root = Path(".tmp_public_benchmark_repeatability_tests") / uuid4().hex
    try:
        run_a = tmp_root / "run_a"
        run_b = tmp_root / "run_b"
        run_c = tmp_root / "run_c"
        run_a.mkdir(parents=True)
        run_b.mkdir(parents=True)
        run_c.mkdir(parents=True)
        _write_public_benchmark_run(
            run_a,
            config_hash="config123",
            llm_provider="fake_llm_client",
            baseline_reward=100.0,
            deterministic_reward=110.0,
            llm_reward=115.0,
            llm_fill_rate=0.65,
        )
        _write_public_benchmark_run(
            run_b,
            config_hash="config123",
            llm_provider="fake_llm_client",
            baseline_reward=100.0,
            deterministic_reward=110.0,
            llm_reward=115.0,
            llm_fill_rate=0.65,
        )
        _write_public_benchmark_run(
            run_c,
            config_hash="config123",
            llm_provider="openai",
            baseline_reward=100.0,
            deterministic_reward=110.0,
            llm_reward=112.0,
            llm_fill_rate=0.63,
        )

        summary = analyze_public_benchmark_repeatability((run_a, run_b, run_c))

        assert summary.run_count == 3
        assert summary.config_hashes == ("config123",)
        assert len(summary.mapping_identities) == 1
        assert summary.deterministic_orchestrator_vs_baseline_label == "stably_better"
        assert summary.llm_vs_deterministic_orchestrator_label == "stably_better"
        assert summary.llm_validity_clean is True
        assert summary.fake_vs_real_llm_comparison is not None
        assert summary.fake_vs_real_llm_comparison.fake_run_count == 2
        assert summary.fake_vs_real_llm_comparison.real_run_count == 1
        assert summary.fake_vs_real_llm_comparison.fake_reward_mean == 115.0
        assert summary.fake_vs_real_llm_comparison.real_reward_mean == 112.0
        assert summary.fake_vs_real_llm_comparison.real_minus_fake_reward_mean == -3.0
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_analyze_public_benchmark_repeatability_script_prints_json(
    monkeypatch,
    capsys,
) -> None:
    tmp_root = Path(".tmp_public_benchmark_repeatability_script_tests") / uuid4().hex
    try:
        run_dir = tmp_root / "run_a"
        run_dir.mkdir(parents=True)
        _write_public_benchmark_run(
            run_dir,
            config_hash="config123",
            llm_provider="fake_llm_client",
            baseline_reward=100.0,
            deterministic_reward=110.0,
            llm_reward=115.0,
            llm_fill_rate=0.65,
        )
        monkeypatch.setattr(
            "sys.argv",
            [
                "analyze_public_benchmark_repeatability.py",
                "--run-dir",
                str(run_dir),
            ],
        )

        repeatability_script.main()
        payload = json.loads(capsys.readouterr().out)

        assert payload["run_count"] == 1
        assert payload["config_hashes"] == ["config123"]
        assert payload["llm_validity_clean"] is True
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
