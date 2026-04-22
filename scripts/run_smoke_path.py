"""Run the first typed smoke path for the bounded MEIO runtime."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from meio.agents.runtime import OrchestrationResponse, OrchestrationRuntime
from meio.config.loaders import load_agent_config, load_benchmark_config, load_experiment_config
from meio.contracts import MissionSpec, OperationalSubgoal
from meio.forecasting.adapters import DeterministicForecastTool
from meio.optimization.adapters import TrustedOptimizerAdapter, build_optimization_request
from meio.optimization.contracts import OptimizationResult
from meio.scenarios.adapters import DeterministicScenarioTool
from meio.scenarios.contracts import ScenarioUpdateResult
from meio.leadtime.adapters import DeterministicLeadTimeTool
from meio.simulation.serial_benchmark import (
    SerialBenchmarkCase,
    build_initial_observation,
    build_initial_simulation_state,
    build_runtime_evidence,
    build_serial_benchmark_case,
    build_serial_orchestration_request,
)
from meio.simulation.state import EpisodeTrace, Observation, SimulationState


DEFAULT_EXPERIMENT_CONFIG = Path("configs/experiment/first_milestone.toml")


@dataclass(frozen=True, slots=True)
class SmokePathRun:
    """Structured result from one deterministic smoke-path run."""

    benchmark_case: SerialBenchmarkCase
    system_state: SimulationState
    observation: Observation
    orchestration_response: OrchestrationResponse
    optimization_result: OptimizationResult
    trace: EpisodeTrace

    def to_summary(self) -> dict[str, object]:
        """Return a compact JSON-serializable summary."""

        return {
            "benchmark_id": self.benchmark_case.benchmark_id,
            "tool_sequence": self.orchestration_response.signal.tool_sequence,
            "selected_subgoal": self.orchestration_response.signal.selected_subgoal.value,
            "request_replan": self.orchestration_response.signal.request_replan,
            "replenishment_orders": self.optimization_result.replenishment_orders,
            "trace_state_count": len(self.trace.states),
        }


def _extract_scenario_update_result(response: OrchestrationResponse) -> ScenarioUpdateResult:
    for tool_result in reversed(response.tool_results):
        value = tool_result.structured_output.get("scenario_update_result")
        if isinstance(value, ScenarioUpdateResult):
            return value
    raise ValueError("Smoke path requires a ScenarioUpdateResult from the runtime tool sequence.")


def run_smoke_path(
    experiment_config_path: str | Path = DEFAULT_EXPERIMENT_CONFIG,
) -> SmokePathRun:
    """Execute one deterministic typed smoke path."""

    experiment_config = load_experiment_config(experiment_config_path)
    benchmark_config = load_benchmark_config(experiment_config.benchmark_config_path)
    if experiment_config.agent_config_path is None:
        raise ValueError("Smoke path requires an agent configuration path.")
    agent_config = load_agent_config(experiment_config.agent_config_path)

    benchmark_case = build_serial_benchmark_case(benchmark_config)
    system_state = build_initial_simulation_state(benchmark_case)
    observation = build_initial_observation(benchmark_case, system_state)
    runtime_evidence = build_runtime_evidence(benchmark_case, observation)

    mission = MissionSpec(
        mission_id="serial_smoke_path",
        objective="Inspect typed evidence, update scenarios, and keep raw actions inside the trusted optimizer.",
        admissible_tool_ids=("forecast_tool", "leadtime_tool", "scenario_tool"),
    )
    runtime = OrchestrationRuntime(
        agent_config=agent_config,
        tools=(
            DeterministicForecastTool(),
            DeterministicLeadTimeTool(),
            DeterministicScenarioTool(),
        ),
    )
    request = build_serial_orchestration_request(
        case=benchmark_case,
        mission=mission,
        system_state=system_state,
        observation=observation,
        evidence=runtime_evidence,
        requested_subgoal=OperationalSubgoal.QUERY_UNCERTAINTY,
    )
    response = runtime.run(request)

    scenario_update_result = _extract_scenario_update_result(response)
    optimization_request = build_optimization_request(
        system_state=system_state,
        scenario_update_result=scenario_update_result,
        base_stock_levels=benchmark_case.base_stock_levels,
        planning_horizon=1,
    )
    optimization_result = TrustedOptimizerAdapter().solve(optimization_request)
    trace = EpisodeTrace(
        run_id=experiment_config.experiment_name,
        benchmark_id=benchmark_case.benchmark_id,
        states=(system_state,),
        observations=(observation,),
        agent_signals=(response.signal,),
        optimization_results=(optimization_result,),
    )
    return SmokePathRun(
        benchmark_case=benchmark_case,
        system_state=system_state,
        observation=observation,
        orchestration_response=response,
        optimization_result=optimization_result,
        trace=trace,
    )


def main() -> None:
    """Run the deterministic typed smoke path and print a compact summary."""

    parser = argparse.ArgumentParser(description="Run the MEIO typed smoke path.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_EXPERIMENT_CONFIG,
        help="Path to the experiment configuration file.",
    )
    args = parser.parse_args()
    smoke_run = run_smoke_path(args.config)
    print(json.dumps(smoke_run.to_summary(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
