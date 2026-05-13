# An Agentic AI Approach for Multi-Echelon Inventory Control Under Uncertainty

This repository contains the implementation and experiment pipeline for the
paper "An Agentic AI Approach for Multi-Echelon Inventory Control Under
Uncertainty." The paper studies bounded agentic uncertainty management for
stochastic multi-echelon inventory control.

The core design keeps adaptive uncertainty reasoning above a trusted downstream
replenishment rule. The Agentic AI layer may diagnose regimes, use bounded
tools, and propose scenario inputs, but it does not emit raw replenishment
orders.

The implementation centers on a regret-guarded, risk-sensitive Agentic AI
system evaluated against deterministic, robust, and rolling-horizon
uncertainty-handling baselines.

## Agentic AI System

The Agentic AI path uses five explicit tools:

- `regime_diagnosis_tool`
- `regime_belief_tool`
- `scenario_candidate_generator_tool`
- `risk_sensitive_scenario_evaluator_tool`
- `counterfactual_regret_guard_tool`

The tools support a bounded decision flow:

1. Diagnose the operating regime from demand, lead-time, inventory, backlog,
   and pipeline evidence.
2. Form a small belief over plausible hidden regimes.
3. Generate bounded scenario-update candidates.
4. Evaluate candidates through a risk-sensitive scenario-scoring step.
5. Apply a counterfactual regret guard before the selected scenario inputs are
   handed to the trusted downstream replenishment rule.

This keeps the method agentic at the uncertainty-management layer while
preserving the downstream action boundary.

## Compared Modes

The comparison surface uses four modes:

- `deterministic_baseline`
- `robust_policy`
- `scenario_rolling_horizon_policy`
- `llm_regret_guarded_risk_sensitive_scenario_planner_orchestrator`

The non-Agentic baselines do not call the LLM or the Agentic AI orchestration
logic. They only supply scenario inputs to the same protected downstream
replenishment rule.

## Validation Lanes

The repository supports three validation lanes:

- Controlled internal simulation:
  `configs/experiment/stockpyl_serial_realistic_comparison.toml`
- Public benchmark portability:
  `configs/experiment/public_benchmark_realistic_comparison.toml`
- Externally grounded backtesting:
  `configs/experiment/real_demand_backtest_panel_realistic_comparison.toml`

The lanes are interpreted side by side rather than pooled. The public benchmark
reports native benchmark reward, while the internal and externally grounded
lanes report Stockpyl-based cost metrics.

## Repository Layout

- `src/meio/`: core packages for agents, scenario planning, baselines,
  simulation, evaluation, benchmark adapters, and backtesting
- `configs/`: benchmark, agent, and experiment configurations
- `scripts/`: entry points for validated runs and summaries
- `tests/`: unit tests for contracts, runtimes, adapters, evaluation, and
  reporting logic
- `third_party/ReplenishmentEnv/`: pinned local benchmark dependency used by
  the public-benchmark and externally grounded lanes
- `audit-trace logs/`: curated public audit-trace example and README

Generated experiment artifacts are written locally under `results/`. The
`results/` directory is intentionally ignored and is not tracked in Git.

## Setup

Create an isolated Python environment, install the package in editable mode,
and install the runtime dependencies used by the validated lanes.

```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pytest stockpyl openai numpy pandas pyyaml
```

For live LLM runs, set `OPENAI_API_KEY`. A minimal environment template is
provided in `.env.example`.

The live agent configuration uses `gpt-5.4-mini` unless overridden by
environment variables.

## Running The Main Lanes

Set `PYTHONPATH` so scripts can resolve the local package.

```powershell
$env:PYTHONPATH = "src"
```

Run the controlled internal comparison:

```powershell
python scripts/run_stockpyl_serial.py --config configs/experiment/stockpyl_serial_realistic_comparison.toml --mode all --llm-client-mode real
```

Run the public benchmark portability comparison:

```powershell
python scripts/run_public_benchmark_eval.py --config configs/experiment/public_benchmark_realistic_comparison.toml --mode all --llm-client-mode real
```

Run the externally grounded backtesting panel:

```powershell
python scripts/run_real_demand_backtest.py --config configs/experiment/real_demand_backtest_panel_realistic_comparison.toml --mode all --llm-client-mode real
```

Summarize the validation stack from saved local artifacts:

```powershell
python scripts/analyze_validation_stack.py
```

## Reproducibility Notes

- Experiment configs are explicit and versioned under `configs/`.
- Local run directories include manifests, metadata, aggregate summaries, and
  audit traces when experiments are executed.
- Result artifacts are not committed to the public repository by default.
- The public benchmark lane uses a pinned local ReplenishmentEnv checkout to
  avoid relying on unstable upstream packaging behavior.
- The curated audit trace omits full prompt text, raw model text, and hidden
  model reasoning while retaining the structured fields needed to audit the
  governed handoff.
