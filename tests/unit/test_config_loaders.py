from __future__ import annotations

from pathlib import Path

import pytest

from meio.config import loaders as config_loaders
from meio.config.loaders import (
    load_agent_config,
    load_benchmark_config,
    load_experiment_config,
    load_public_benchmark_eval_config,
    load_real_demand_backtest_config,
    load_real_demand_backtest_panel_config,
)
from meio.contracts import BackorderPolicy, BenchmarkFamily, RegimeLabel, ToolClass, UpdateRequestType


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_load_valid_example_configs() -> None:
    benchmark_config = load_benchmark_config(REPO_ROOT / "configs/benchmark/serial_3_echelon.toml")
    experiment_config = load_experiment_config(REPO_ROOT / "configs/experiment/first_milestone.toml")
    agent_config = load_agent_config(REPO_ROOT / "configs/agent/base.toml")

    assert benchmark_config.benchmark_family is BenchmarkFamily.SERIAL
    assert benchmark_config.service_model is BackorderPolicy.BACKORDERS
    assert benchmark_config.echelon_count == 3
    assert benchmark_config.demand_mean == 10.0
    assert benchmark_config.system.stages[0].shipment_lead_time == 2
    assert benchmark_config.system.stages[0].base_stock_level == 20
    assert experiment_config.episode_count == 5
    assert RegimeLabel.NORMAL in agent_config.enabled_regime_labels
    assert RegimeLabel.JOINT_DISRUPTION in agent_config.enabled_regime_labels
    assert ToolClass.LLM_BACKED in agent_config.allowed_tool_classes
    assert UpdateRequestType.WIDEN_UNCERTAINTY in agent_config.allowed_update_types
    assert agent_config.max_tool_steps == 3
    assert agent_config.allow_abstain is True
    assert agent_config.llm_provider == "openai"
    assert agent_config.llm_client_mode == "fake"
    assert agent_config.llm_model_name == "gpt-4o-mini"
    assert agent_config.llm_temperature == 0.0
    assert agent_config.llm_request_timeout_s == 20.0
    assert agent_config.llm_max_retries == 1


def test_load_stockpyl_serial_experiment_config_reads_rollout_schedule() -> None:
    experiment_config = load_experiment_config(REPO_ROOT / "configs/experiment/stockpyl_serial.toml")

    assert experiment_config.rollout_horizon == 3
    assert experiment_config.regime_schedule == (
        RegimeLabel.NORMAL,
        RegimeLabel.DEMAND_REGIME_SHIFT,
        RegimeLabel.RECOVERY,
    )


def test_load_live_llm_configs_reads_real_client_settings() -> None:
    experiment_config = load_experiment_config(
        REPO_ROOT / "configs/experiment/stockpyl_serial_live_llm.toml"
    )
    agent_config = load_agent_config(REPO_ROOT / "configs/agent/live_llm.toml")

    assert experiment_config.episode_count == 2
    assert agent_config.llm_provider == "openai"
    assert agent_config.llm_client_mode == "real"
    assert agent_config.llm_model_name == "gpt-4o-mini"
    assert agent_config.llm_temperature == 0.0
    assert agent_config.llm_request_timeout_s == 20.0
    assert agent_config.llm_max_retries == 1


def test_load_multi_schedule_experiment_config_reads_schedule_and_seed_sets() -> None:
    experiment_config = load_experiment_config(
        REPO_ROOT / "configs/experiment/stockpyl_serial_multi_eval.toml"
    )

    assert experiment_config.rollout_horizon is None
    assert experiment_config.seed_set == (20260417, 20260418)
    assert tuple(schedule.name for schedule in experiment_config.regime_schedules) == (
        "shift_recovery",
        "sustained_shift",
        "recovery_false_alarm",
        "long_shift_recovery",
    )
    assert experiment_config.resolved_schedule_set()[-1].labels[-1] is RegimeLabel.RECOVERY


def test_load_tool_ablation_experiment_config_reads_mode_and_ablation_sets() -> None:
    experiment_config = load_experiment_config(
        REPO_ROOT / "configs/experiment/stockpyl_serial_tool_ablation.toml"
    )

    assert experiment_config.mode_set == (
        "deterministic_baseline",
        "deterministic_orchestrator",
        "llm_orchestrator",
    )
    assert experiment_config.tool_ablation_variants == (
        "full",
        "no_forecast_tool",
        "no_leadtime_tool",
        "no_scenario_tool",
    )
    assert experiment_config.seed_set == (20260417, 20260418)


def test_load_paper_candidate_experiment_config_preserves_full_tool_main_path() -> None:
    experiment_config = load_experiment_config(
        REPO_ROOT / "configs/experiment/stockpyl_serial_paper_candidate.toml"
    )

    assert experiment_config.mode_set == (
        "deterministic_baseline",
        "deterministic_orchestrator",
        "llm_orchestrator",
    )
    assert experiment_config.tool_ablation_variants == ("full",)
    assert experiment_config.seed_set == (20260417, 20260418)
    assert tuple(schedule.name for schedule in experiment_config.regime_schedules) == (
        "shift_recovery",
        "sustained_shift",
        "recovery_false_alarm",
        "long_shift_recovery",
    )
    assert experiment_config.results_dir == Path("results/stockpyl_serial_paper_candidate")


def test_load_heldout_experiment_config_preserves_frozen_full_tool_main_path() -> None:
    experiment_config = load_experiment_config(
        REPO_ROOT / "configs/experiment/stockpyl_serial_heldout_eval.toml"
    )

    assert experiment_config.mode_set == (
        "deterministic_baseline",
        "deterministic_orchestrator",
        "llm_orchestrator",
    )
    assert experiment_config.tool_ablation_variants == ("full",)
    assert experiment_config.seed_set == (20260417, 20260418)
    assert tuple(schedule.name for schedule in experiment_config.regime_schedules) == (
        "delayed_shift_recovery",
        "delayed_sustained_shift",
        "double_shift_with_gap",
        "recovery_then_relapse",
        "false_alarm_then_real_shift",
    )
    assert experiment_config.results_dir == Path("results/stockpyl_serial_heldout_eval")


def test_load_frozen_broad_eval_experiment_config_preserves_frozen_full_tool_path() -> None:
    experiment_config = load_experiment_config(
        REPO_ROOT / "configs/experiment/stockpyl_serial_frozen_broad_eval.toml"
    )

    assert experiment_config.mode_set == (
        "deterministic_baseline",
        "deterministic_orchestrator",
        "llm_orchestrator",
    )
    assert experiment_config.tool_ablation_variants == ("full",)
    assert experiment_config.seed_set == (20260417, 20260418, 20260419)
    assert tuple(schedule.name for schedule in experiment_config.regime_schedules) == (
        "shift_recovery",
        "sustained_shift",
        "recovery_false_alarm",
        "long_shift_recovery",
        "delayed_shift_recovery",
        "delayed_sustained_shift",
        "false_alarm_then_real_shift",
        "double_shift_with_gap",
        "recovery_then_relapse",
        "delayed_long_shift_recovery",
        "false_alarm_then_shift_recovery",
    )
    assert experiment_config.results_dir == Path("results/stockpyl_serial_frozen_broad_eval")


def test_load_external_evidence_experiment_config_keeps_control_and_branch_modes_separate() -> None:
    experiment_config = load_experiment_config(
        REPO_ROOT / "configs/experiment/stockpyl_serial_external_evidence.toml"
    )

    assert experiment_config.mode_set == (
        "deterministic_baseline",
        "deterministic_orchestrator",
        "llm_orchestrator_internal_only",
        "llm_orchestrator_with_external_evidence",
    )
    assert experiment_config.tool_ablation_variants == ("full",)
    assert experiment_config.semi_synthetic_external_evidence is True
    assert experiment_config.seed_set == (20260417, 20260418)
    assert tuple(schedule.name for schedule in experiment_config.regime_schedules) == (
        "shift_recovery",
        "false_alarm_then_real_shift",
        "double_shift_with_gap",
        "recovery_then_relapse",
    )
    assert experiment_config.results_dir == Path("results/stockpyl_serial_external_evidence")


def test_load_public_benchmark_eval_config_reads_replenishmentenv_defaults() -> None:
    config = load_public_benchmark_eval_config(
        REPO_ROOT / "configs/experiment/public_benchmark_eval.toml"
    )

    assert config.experiment_name == "public_benchmark_eval"
    assert config.benchmark_candidate == "replenishment_env"
    assert config.discovery_module == "ReplenishmentEnv"
    assert config.benchmark_root == Path("third_party/ReplenishmentEnv")
    assert config.demo_config_path == Path("config/demo.yml")
    assert config.agent_config_path == Path("configs/agent/base.toml")
    assert config.environment_config_name == "sku50.single_store.standard"
    assert config.wrapper_names == ("HistoryWrapper",)
    assert config.benchmark_mode == "test"
    assert config.smoke_horizon_steps == 1
    assert config.mode_set == (
        "deterministic_baseline",
        "deterministic_orchestrator",
        "llm_orchestrator",
    )
    assert config.episode_horizon_steps == 10
    assert config.base_stock_multiplier == 1.0
    assert config.demand_scale_epsilon == 1e-6


def test_load_real_demand_backtest_config_reads_bounded_public_data_settings() -> None:
    config = load_real_demand_backtest_config(
        REPO_ROOT / "configs/experiment/real_demand_backtest.toml"
    )

    assert config.dataset_name == "replenishmentenv_sku2778_store2"
    assert config.selected_sku_count == 5
    assert config.subset_selection == "nearest_benchmark_mean"
    assert config.training_window_days == 180
    assert config.evaluation_horizon_days == 30
    assert config.mode_set == (
        "deterministic_baseline",
        "deterministic_orchestrator",
        "llm_orchestrator",
    )


def test_load_real_demand_backtest_panel_config_reads_explicit_slice_panel() -> None:
    config = load_real_demand_backtest_panel_config(
        REPO_ROOT / "configs/experiment/real_demand_backtest_panel.toml"
    )

    assert config.experiment_name == "real_demand_backtest_panel"
    assert config.mode_set == (
        "deterministic_baseline",
        "deterministic_orchestrator",
        "llm_orchestrator",
    )
    assert tuple(slice_config.name for slice_config in config.slices) == (
        "store1_low_2018q4",
        "store2_mid_2019q1",
        "store3_midhigh_2019q2",
    )
    assert config.slices[0].selected_skus == (
        "SKU48",
        "SKU44",
        "SKU15",
        "SKU59",
        "SKU42",
    )
    assert config.slices[1].evaluation_start_date == "2019/2/26"


def test_load_benchmark_config_rejects_nonpositive_echelon_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        config_loaders,
        "_load_toml_document",
        lambda path: {
            "benchmark": {
                "family": "serial",
                "service_model": "backorders",
                "scenario_families": ["normal"],
                "random_seed": 1,
            },
            "system": {
                "topology": "serial",
                "echelon_count": 0,
            },
            "costs": {
                "holding_cost": 1.0,
                "backorder_cost": 5.0,
            },
        },
    )

    with pytest.raises(ValueError, match="echelon_count"):
        load_benchmark_config("unused.toml")


def test_load_agent_config_rejects_invalid_confidence_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        config_loaders,
        "_load_toml_document",
        lambda path: {
            "agent": {
                "enabled_regime_labels": ["normal"],
                "allowed_update_types": ["keep_current"],
                "minimum_confidence": 1.5,
                "allow_replan_requests": True,
            }
        },
    )

    with pytest.raises(ValueError, match="minimum_confidence"):
        load_agent_config("unused.toml")


def test_load_agent_config_rejects_invalid_tool_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        config_loaders,
        "_load_toml_document",
        lambda path: {
            "agent": {
                "enabled_regime_labels": ["normal"],
                "allowed_update_types": ["keep_current"],
                "allowed_tool_classes": ["imaginary_tool_class"],
                "minimum_confidence": 0.2,
                "max_tool_steps": 2,
                "allow_replan_requests": True,
                "allow_abstain": True,
            }
        },
    )

    with pytest.raises(ValueError, match="agent.allowed_tool_classes"):
        load_agent_config("unused.toml")


def test_load_agent_config_uses_env_model_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config_loaders, "load_env_value", lambda key: "gpt-4.1-mini")

    agent_config = load_agent_config(REPO_ROOT / "configs/agent/base.toml")

    assert agent_config.llm_model_name == "gpt-4.1-mini"


def test_load_experiment_config_rejects_mismatched_seed_set_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        config_loaders,
        "_load_toml_document",
        lambda path: {
            "experiment": {
                "name": "batch_test",
                "benchmark_config": "configs/benchmark/serial_3_echelon.toml",
                "agent_config": "configs/agent/base.toml",
                "episode_count": 2,
                "seed_set": [1],
                "results_dir": "results/test",
            }
        },
    )

    with pytest.raises(ValueError, match="seed_set"):
        load_experiment_config("unused.toml")
