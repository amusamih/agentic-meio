# Audit-Trace Logs

This folder contains curated audit-trace logs supporting the manuscript's decision-traceability table.

The included trace was extracted from a live Agentic AI run in the controlled internal simulation lane using prompt version `meio.llm_orchestrator.v9`. It records the operating evidence, structured orchestration output, requested tool sequence, tool outputs, validation and calibration outcome, protected scenario update, downstream replenishment request, and runtime telemetry for one decision cycle.

Raw experiment output directories are intentionally not committed because the repository ignores generated `results/` artifacts. This curated log preserves the exact fields used in the manuscript table while keeping the public repository lightweight.
