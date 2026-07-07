# An Agentic AI Approach for Multi-Echelon Inventory Control Under Uncertainty

This repository contains the implementation and experiment pipeline for the
paper "An Agentic AI Approach for Multi-Echelon Inventory Control Under
Uncertainty." The project studies bounded agentic uncertainty management for a
serial multi-echelon inventory system while preserving a strict downstream
action boundary.

The Agentic AI layer interprets operating context, invokes scenario-planning
tools when uncertainty adaptation is warranted, and proposes protected scenario
inputs. It does not emit raw replenishment orders. Replenishment requests are
generated only by the trusted downstream order-up-to decision rule.

## Agentic AI System

The current Agentic AI path uses six explicit scenario-planning tools:

- `regime_diagnosis_tool`
- `demand_uncertainty_decomposition_tool`
- `regime_belief_tool`
- `scenario_candidate_generator_tool`
- `risk_sensitive_scenario_evaluator_tool`
- `counterfactual_regret_guard_tool`

The tools support a governed decision flow:

1. Diagnose the operating regime from demand, lead-time, inventory, backlog,
   and pipeline evidence.
2. Separate demand-level evidence from demand-variability evidence.
3. Form a bounded belief over plausible near-term regimes.
4. Generate feasible scenario-input candidates.
5. Evaluate candidates using risk-sensitive cost and service criteria.
6. Apply a counterfactual regret guard before the selected scenario inputs are
   handed to the trusted downstream replenishment rule.

This keeps the method agentic at the uncertainty-management layer while keeping
operational action authority inside the downstream decision component.

## Compared Modes

The main comparison surface uses four modes:

- `deterministic_baseline`
- `robust_policy`
- `scenario_rolling_horizon_policy`
- `llm_regret_guarded_risk_sensitive_scenario_planner_orchestrator`

All modes use the same protected downstream replenishment rule. The non-Agentic
baselines do not call the LLM orchestration layer or the scenario-planning tool
sequence.

## Validation Lanes

The repository supports the validation lanes reported in the paper:

- Controlled demand-level simulation:
  `configs/experiment/stockpyl_serial_realistic_comparison.toml`
- Controlled demand-variability simulation:
  `configs/experiment/stockpyl_serial_spread_sensitivity.toml`
- Public benchmark portability:
  `configs/experiment/public_benchmark_realistic_comparison.toml`
- Externally grounded backtesting:
  `configs/experiment/real_demand_backtest_panel_realistic_comparison.toml`

The lanes are interpreted side by side rather than pooled. The public benchmark
reports native benchmark reward, while the controlled and externally grounded
lanes report Stockpyl-based cost metrics.

The repository also includes component-ablation configs for the controlled
settings:

- `configs/experiment/stockpyl_serial_component_ablation.toml`
- `configs/experiment/stockpyl_serial_spread_component_ablation.toml`

## Repository Layout

- `src/meio/`: core packages for agents, scenario planning, baselines,
  simulation, evaluation, benchmark adapters, and backtesting
- `configs/`: benchmark, agent, and experiment configurations
- `scripts/`: runnable entry points for the validation lanes
- `tests/`: unit tests for contracts, runtimes, adapters, evaluation, and
  reporting logic
- `third_party/ReplenishmentEnv/`: pinned local benchmark checkout used by the
  public-benchmark and externally grounded lanes
- `audit-trace-logs/`: curated public audit-trace example and README

Generated experiment artifacts are written locally under `results/`. The
`results/` directory is intentionally ignored and is not tracked in Git.

## Setup

Create an isolated Python environment and install the package in editable mode.

```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The public benchmark lane uses the pinned local ReplenishmentEnv checkout. If
you plan to run that lane, install it as well:

```powershell
python -m pip install -e third_party/ReplenishmentEnv
```

For live LLM runs, set `OPENAI_API_KEY`. A minimal environment template is
provided in `.env.example`. The live agent configuration uses `gpt-5.4-mini`
unless overridden by environment variables.

## Running The Main Lanes

Set `PYTHONPATH` so scripts can resolve the local package.

```powershell
$env:PYTHONPATH = "src"
```

Run the controlled demand-level comparison:

```powershell
python scripts/run_stockpyl_serial.py --config configs/experiment/stockpyl_serial_realistic_comparison.toml --mode all --llm-client-mode real
```

Run the controlled demand-variability comparison:

```powershell
python scripts/run_stockpyl_serial.py --config configs/experiment/stockpyl_serial_spread_sensitivity.toml --mode all --llm-client-mode real
```

Run the public benchmark portability comparison:

```powershell
python scripts/run_public_benchmark_eval.py --config configs/experiment/public_benchmark_realistic_comparison.toml --mode all --llm-client-mode real
```

Run the externally grounded backtesting panel:

```powershell
python scripts/run_real_demand_backtest.py --config configs/experiment/real_demand_backtest_panel_realistic_comparison.toml --mode all --llm-client-mode real
```

Run the component ablations:

```powershell
python scripts/run_stockpyl_serial.py --config configs/experiment/stockpyl_serial_component_ablation.toml --mode all --llm-client-mode real
python scripts/run_stockpyl_serial.py --config configs/experiment/stockpyl_serial_spread_component_ablation.toml --mode all --llm-client-mode real
```

Summarize the validation stack from saved local artifacts:

```powershell
python scripts/analyze_validation_stack.py
```

## Reproducibility Notes

- Experiment configs are explicit and versioned under `configs/`.
- Local run directories include manifests, metadata, aggregate summaries, step
  traces, LLM call traces, tool call traces, and episode summaries.
- Result artifacts are not committed to the public repository by default.
- The curated audit trace omits full prompt text, raw model text, and hidden
  model reasoning while retaining the structured fields needed to audit the
  governed handoff.
