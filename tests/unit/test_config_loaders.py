from __future__ import annotations

from pathlib import Path

import pytest

from meio.config import loaders as config_loaders
from meio.config.loaders import (
    load_agent_config,
    load_benchmark_config,
    load_experiment_config,
    load_public_benchmark_eval_config,
    load_real_demand_backtest_panel_config,
)
from meio.contracts import BackorderPolicy, BenchmarkFamily, RegimeLabel, ToolClass, UpdateRequestType


REPO_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_MODES = (
    "deterministic_baseline",
    "robust_policy",
    "scenario_rolling_horizon_policy",
    "llm_regret_guarded_risk_sensitive_scenario_planner_orchestrator",
)


def test_load_valid_example_configs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEIO_LLM_ORCHESTRATOR_MODEL", "gpt-4o-mini")
    benchmark_config = load_benchmark_config(REPO_ROOT / "configs/benchmark/serial_3_echelon.toml")
    experiment_config = load_experiment_config(
        REPO_ROOT / "configs/experiment/stockpyl_serial_realistic_comparison.toml"
    )
    agent_config = load_agent_config(REPO_ROOT / "configs/agent/base.toml")

    assert benchmark_config.benchmark_family is BenchmarkFamily.SERIAL
    assert benchmark_config.service_model is BackorderPolicy.BACKORDERS
    assert benchmark_config.echelon_count == 3
    assert benchmark_config.demand_mean == 10.0
    assert benchmark_config.system.stages[0].shipment_lead_time == 2
    assert benchmark_config.system.stages[0].base_stock_level == 20
    assert experiment_config.episode_count == 3
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


def test_load_live_llm_configs_reads_real_client_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEIO_LLM_ORCHESTRATOR_MODEL", "gpt-5.4-mini")
    experiment_config = load_experiment_config(
        REPO_ROOT / "configs/experiment/stockpyl_serial_realistic_comparison.toml"
    )
    agent_config = load_agent_config(REPO_ROOT / "configs/agent/live_llm.toml")

    assert experiment_config.episode_count == 3
    assert agent_config.llm_provider == "openai"
    assert agent_config.llm_client_mode == "real"
    assert agent_config.llm_model_name == "gpt-5.4-mini"
    assert agent_config.llm_temperature == 0.0
    assert agent_config.llm_request_timeout_s == 20.0
    assert agent_config.llm_max_retries == 1


def test_load_realistic_comparison_config_limits_modes_to_paper_comparison() -> None:
    experiment_config = load_experiment_config(
        REPO_ROOT / "configs/experiment/stockpyl_serial_realistic_comparison.toml"
    )

    assert experiment_config.mode_set == OFFICIAL_MODES
    assert experiment_config.seed_set == (20260417, 20260418, 20260419)
    assert len(experiment_config.regime_schedules) == 11
    assert experiment_config.results_dir == Path("results/stockpyl_serial_realistic_comparison")


def test_load_spread_sensitivity_config_reads_mean_preserving_profile() -> None:
    experiment_config = load_experiment_config(
        REPO_ROOT / "configs/experiment/stockpyl_serial_spread_sensitivity.toml"
    )

    assert experiment_config.mode_set == OFFICIAL_MODES
    assert experiment_config.seed_set == (20260417, 20260418, 20260419)
    assert len(experiment_config.regime_schedules) == 11
    assert experiment_config.controlled_demand_profile is not None
    assert experiment_config.controlled_demand_profile.profile_type == "mean_preserving_spread"
    assert experiment_config.controlled_demand_profile.target_mean == 10.0
    assert experiment_config.controlled_demand_profile.values_for(RegimeLabel.NORMAL) == (
        7.0,
        10.0,
        13.0,
    )
    assert experiment_config.controlled_demand_profile.values_for(
        RegimeLabel.DEMAND_REGIME_SHIFT
    ) == (2.0, 10.0, 18.0)
    assert experiment_config.results_dir == Path("results/stockpyl_serial_spread_sensitivity")


def test_load_component_ablation_configs_read_tool_variants() -> None:
    demand_level_config = load_experiment_config(
        REPO_ROOT / "configs/experiment/stockpyl_serial_component_ablation.toml"
    )
    spread_config = load_experiment_config(
        REPO_ROOT / "configs/experiment/stockpyl_serial_spread_component_ablation.toml"
    )

    expected_variants = (
        "full",
        "without_demand_decomposition",
        "without_risk_sensitive_evaluation",
        "without_regret_guard",
    )
    assert demand_level_config.mode_set == (
        "llm_regret_guarded_risk_sensitive_scenario_planner_orchestrator",
    )
    assert demand_level_config.tool_ablation_variants == expected_variants
    assert spread_config.tool_ablation_variants == expected_variants
    assert spread_config.controlled_demand_profile is not None
    assert demand_level_config.results_dir == Path(
        "results/stockpyl_serial_component_ablation"
    )
    assert spread_config.results_dir == Path(
        "results/stockpyl_serial_spread_component_ablation"
    )


def test_load_public_benchmark_realistic_comparison_config_reads_latest_modes() -> None:
    config = load_public_benchmark_eval_config(
        REPO_ROOT / "configs/experiment/public_benchmark_realistic_comparison.toml"
    )

    assert config.experiment_name == "public_benchmark_realistic_comparison"
    assert config.mode_set == OFFICIAL_MODES
    assert config.agent_config_path == Path("configs/agent/live_llm.toml")
    assert config.uncertainty_baselines.robust_policy.window_length == 14
    assert config.uncertainty_baselines.scenario_rolling_horizon_policy.scenario_count == 8
    assert config.results_dir == Path("results/public_benchmark_realistic_comparison")


def test_load_real_demand_realistic_comparison_panel_config_reads_latest_modes() -> None:
    config = load_real_demand_backtest_panel_config(
        REPO_ROOT / "configs/experiment/real_demand_backtest_panel_realistic_comparison.toml"
    )

    assert config.experiment_name == "real_demand_backtest_panel_realistic_comparison"
    assert config.mode_set == OFFICIAL_MODES
    assert config.agent_config_path == Path("configs/agent/live_llm.toml")
    assert len(config.slices) == 3
    assert config.results_dir == Path("results/real_demand_backtest_panel_realistic_comparison")


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
