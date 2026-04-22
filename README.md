# MEIO

MEIO studies bounded agentic uncertainty management for stochastic multi-echelon inventory control. The central idea is to place a bounded orchestration layer above a trusted optimizer: the orchestrator decides when uncertainty assumptions should be revisited and which bounded tools to use, while the optimizer remains the only component allowed to emit raw replenishment orders.

The repository is organized as a research codebase rather than a product demo. The frozen method under study uses three bounded tools:

- `forecast_tool`
- `leadtime_tool`
- `scenario_tool`

Current evidence supports conditional usefulness rather than universal superiority: the bounded LLM-backed orchestration mode helps in some settings, stays validity-clean under the saved evaluation stack, and preserves optimizer-only action authority, but it is not a blanket win over simpler modes on every benchmark slice.

## Method Summary

The method keeps a strict separation between uncertainty management and action optimization.

- The bounded orchestrator manages uncertainty-facing subgoals such as inspecting evidence, selecting tools, requesting bounded updates, requesting replanning, or abstaining.
- The trusted optimizer consumes the resulting bounded uncertainty state and computes replenishment quantities.
- Structured-output validation, tool admissibility checks, fallback handling, and run-level trace logging remain active across the supported lanes.

## Validation Stack

The repository currently supports three main validation lanes.

- `stockpyl_internal`
  - main evidence base
  - structured serial multi-echelon validation on the Stockpyl path
- `public_benchmark`
  - supporting external validation
  - end-to-end execution on a pinned local ReplenishmentEnv checkout through a thin single-store adapter
- `real_demand_backtest`
  - supporting external validation
  - bounded rolling backtests on public demand and lead-time observations

These lanes are not all directly comparable. In particular, public-benchmark reward is not the same objective as Stockpyl total cost, and the real-demand lane reuses the Stockpyl cost backbone while replacing the observations with externally grounded demand and lead-time data.

## Repository Layout

- `src/meio/`: core packages for agents, tools, simulation, optimization, evaluation, benchmark adapters, and backtesting
- `configs/`: benchmark, agent, and experiment configurations
- `scripts/`: entry points for validated runs and result analysis
- `tests/`: unit tests for contracts, runtimes, adapters, evaluation, and export logic
- `third_party/ReplenishmentEnv/`: pinned local benchmark dependency for the public-benchmark lane

Generated experiment artifacts are written to `results/` during local runs.

## Setup

Create an isolated Python 3.11+ environment, install the package in editable mode, and install the runtime dependencies used by the validated lanes.

```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pytest stockpyl openai numpy pandas pyyaml
```

For live LLM runs, set `OPENAI_API_KEY`. A minimal environment template is provided in `.env.example`.

For the public-benchmark lane, keep `third_party/ReplenishmentEnv/` in place. The current benchmark integration depends on that pinned local checkout rather than an upstream wheel.

## Running The Main Validated Lanes

Set `PYTHONPATH` so the scripts can resolve the local package.

```powershell
$env:PYTHONPATH = "src"
```

Run the main internal structured screen:

```powershell
python scripts/run_stockpyl_serial.py --config configs/experiment/stockpyl_serial_paper_candidate.toml --mode all --llm-client-mode real
```

Run the frozen public-benchmark lane:

```powershell
python scripts/run_public_benchmark_eval.py --config configs/experiment/public_benchmark_eval.toml --mode all --llm-client-mode real
```

Run the frozen repeated real-demand panel:

```powershell
python scripts/run_real_demand_backtest.py --config configs/experiment/real_demand_backtest_panel.toml --mode all --llm-client-mode real
```

Summarize the current validation stack from saved artifacts:

```powershell
python scripts/analyze_validation_stack.py
```

## What To Expect From The Saved Evidence

- The Stockpyl lane is the main evidence base.
- The public benchmark and real-demand lanes are supporting external validation.
- The strongest current interpretation is conditional usefulness of the bounded LLM layer under a strict optimizer boundary, not universal outperformance across all settings.

## Reproducibility Notes

- Configs are explicit and versioned under `configs/`.
- Saved run directories include manifests, metadata, aggregate summaries, and trace files for audit.
- The public benchmark lane uses a pinned local third-party dependency to avoid the broken upstream packaging path encountered during validation.
