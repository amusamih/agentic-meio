# Audit-Trace Logs

This folder contains curated audit-trace logs supporting the manuscript's decision-traceability evidence.

The included trace records the regret-guarded risk-sensitive Agentic AI system in the controlled internal simulation lane. It uses model `gpt-5.4-mini`, prompt version `meio.llm_orchestrator.v16`, and records one representative demand-shift decision cycle from the `shift_recovery` case.

Included file:

- `shift_recovery_p001_regret_guarded_audit_trace.json`

The trace records the operating evidence, structured orchestration output, explicit tool pathway, validation outcome, protected scenario update, downstream replenishment request, and runtime telemetry. It reflects the current six-tool scenario-planning design:

- `regime_diagnosis_tool`
- `demand_uncertainty_decomposition_tool`
- `regime_belief_tool`
- `scenario_candidate_generator_tool`
- `risk_sensitive_scenario_evaluator_tool`
- `counterfactual_regret_guard_tool`

The trace intentionally omits full prompt text, raw model output text, and any hidden LLM reasoning. It preserves only the structured fields needed to audit the protected handoff from Agentic AI orchestration to the trusted downstream replenishment rule.

Raw experiment output directories are intentionally not committed because the repository ignores generated `results/` artifacts. This curated log preserves the fields needed for public inspection while keeping the repository lightweight.
