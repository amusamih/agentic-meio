from __future__ import annotations

import json
from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np

from meio.benchmarks.public_benchmark_adapter import (
    execute_replenishmentenv_smoke,
    inspect_replenishmentenv_installation,
    locate_package_root,
    map_optimizer_orders_to_public_benchmark_actions,
)
import scripts.run_public_benchmark_eval as run_public_benchmark_eval_script


def test_inspect_replenishmentenv_installation_reads_demo_assets_from_explicit_root(
) -> None:
    tmp_path = Path(".tmp_public_benchmark_adapter_tests") / uuid4().hex
    config_dir = tmp_path / "config"
    data_dir = tmp_path / "data"
    try:
        config_dir.mkdir(parents=True)
        data_dir.mkdir()
        (data_dir / "sku_list.csv").write_text("SKU\nSKU1\n", encoding="utf-8")
        (data_dir / "static.csv").write_text("col\n1\n", encoding="utf-8")
        demo_payload = {
            "env": {
                "mode": [{"name": "train"}, {"name": "test"}],
                "lookback_len": 14,
                "horizon": 7,
                "sku_list": "ReplenishmentEnv/data/sku_list.csv",
            },
            "warehouse": [
                {
                    "name": "store2",
                    "sku": {
                        "static_data": "ReplenishmentEnv/data/static.csv",
                        "dynamic_data": [{"file": "ReplenishmentEnv/data/missing.csv"}],
                    },
                }
            ],
        }
        (config_dir / "demo.yml").write_text(json.dumps(demo_payload), encoding="utf-8")

        summary = inspect_replenishmentenv_installation(
            module_name="missing_replenishmentenv_module",
            benchmark_root=tmp_path,
            demo_config_path=Path("config/demo.yml"),
        )

        assert summary.validation_lane == "public_benchmark"
        assert summary.module_discovered is True
        assert summary.import_succeeds is False
        assert summary.package_layout_complete is False
        assert summary.environment_runnable is False
        assert summary.blocked_reason is not None
        assert summary.blocked_reason.startswith("incomplete_installation:")
        assert summary.demo_summary is not None
        assert summary.demo_summary.mode_names == ("train", "test")
        assert summary.demo_summary.referenced_files == (
            "data/sku_list.csv",
            "data/static.csv",
            "data/missing.csv",
        )
        assert summary.demo_summary.missing_files == ("data/missing.csv",)
        assert "env/replenishment_env.py" in summary.missing_required_paths
        assert "wrapper/default_wrapper.py" in summary.missing_required_paths
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_run_public_benchmark_eval_script_writes_inspection_payload(
    monkeypatch,
    capsys,
) -> None:
    tmp_path = Path(".tmp_public_benchmark_script_tests") / uuid4().hex
    config_dir = tmp_path / "config"
    try:
        config_dir.mkdir(parents=True)
        (config_dir / "demo.yml").write_text(
            json.dumps({"env": {"mode": [{"name": "train"}]}, "warehouse": []}),
            encoding="utf-8",
        )
        config_path = tmp_path / "public_benchmark_eval.toml"
        config_path.write_text(
            "\n".join(
                (
                    "[experiment]",
                    'name = "temp_public_benchmark_eval"',
                    'benchmark_candidate = "replenishment_env"',
                    'discovery_module = "missing_replenishmentenv_module"',
                    f'benchmark_root = "{tmp_path.as_posix()}"',
                    'demo_config_path = "config/demo.yml"',
                    f'results_dir = "{(tmp_path / "results").as_posix()}"',
                )
            )
            + "\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            "sys.argv",
            ["run_public_benchmark_eval.py", "--config", str(config_path)],
        )

        run_public_benchmark_eval_script.main()
        payload = json.loads(capsys.readouterr().out)

        assert payload["experiment_metadata"]["validation_lane"] == "public_benchmark"
        assert payload["public_benchmark_summary"]["blocked_reason"].startswith(
            "incomplete_installation:"
        )
        assert Path(payload["artifacts_dir"]).exists()
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_locate_package_root_accepts_source_checkout_root() -> None:
    package_root = locate_package_root(
        module_name="ReplenishmentEnv",
        explicit_root=Path("third_party/ReplenishmentEnv"),
    )

    assert package_root == Path("third_party/ReplenishmentEnv/ReplenishmentEnv").resolve()


def test_execute_replenishmentenv_smoke_runs_against_local_third_party_checkout() -> None:
    package_root = locate_package_root(
        module_name="ReplenishmentEnv",
        explicit_root=Path("third_party/ReplenishmentEnv"),
    )

    assert package_root is not None
    summary = execute_replenishmentenv_smoke(
        module_name="ReplenishmentEnv",
        package_root=package_root,
        environment_config_name="sku100.single_store.standard",
        wrapper_names=("DefaultWrapper",),
        benchmark_mode="test",
        smoke_horizon_steps=1,
        vis_path=Path(".tmp_public_benchmark_vis"),
    )

    assert summary.attempted is True
    assert summary.succeeded is True
    assert summary.completed_steps == 1
    assert summary.observation_shape
    assert summary.action_shape
    assert summary.reward_shape
    assert summary.reward_sum is not None


def test_inspect_replenishmentenv_installation_reports_local_packaging_root_cause() -> None:
    summary = inspect_replenishmentenv_installation(
        module_name="ReplenishmentEnv",
        benchmark_root=Path("third_party/ReplenishmentEnv"),
        demo_config_path=Path("config/demo.yml"),
        run_smoke_execution=True,
        environment_config_name="sku100.single_store.standard",
        wrapper_names=("DefaultWrapper",),
        benchmark_mode="test",
        smoke_horizon_steps=1,
        vis_path=Path(".tmp_public_benchmark_summary_vis"),
    )

    assert summary.module_discovered is True
    assert summary.import_succeeds is True
    assert summary.package_layout_complete is True
    assert summary.environment_runnable is True
    assert summary.using_local_source_checkout is True
    assert summary.packaging_root_cause is not None
    assert "find_packages" in summary.packaging_root_cause
    assert summary.smoke_summary is not None
    assert summary.smoke_summary.succeeded is True


def test_map_optimizer_orders_to_public_benchmark_actions_divides_by_demand_mean() -> None:
    actions = map_optimizer_orders_to_public_benchmark_actions(
        replenishment_quantities=np.array([[20.0, 15.0]], dtype=np.float32),
        demand_mean_by_sku=np.array([[10.0, 5.0]], dtype=np.float32),
        action_mode="demand_mean_continuous",
        demand_scale_epsilon=1e-6,
    )

    assert actions.shape == (1, 2)
    assert np.allclose(actions, np.array([[2.0, 3.0]], dtype=np.float32))


def test_map_optimizer_orders_to_public_benchmark_actions_clips_negative_quantities() -> None:
    actions = map_optimizer_orders_to_public_benchmark_actions(
        replenishment_quantities=np.array([[-4.0, 15.0]], dtype=np.float32),
        demand_mean_by_sku=np.array([[10.0, 5.0]], dtype=np.float32),
        action_mode="demand_mean_continuous",
        demand_scale_epsilon=1e-6,
    )

    assert np.allclose(actions, np.array([[0.0, 3.0]], dtype=np.float32))
