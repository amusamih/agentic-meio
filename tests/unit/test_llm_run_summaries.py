from __future__ import annotations

import pytest

from meio.evaluation.llm_run_summaries import ComparisonRunSummary, LLMRunDiagnostics


def test_llm_run_diagnostics_constructs_with_valid_counts() -> None:
    diagnostics = LLMRunDiagnostics(
        model_name="gpt-4o-mini",
        provider="openai",
        run_count=2,
        invalid_output_count=1,
        fallback_count=1,
        successful_response_count=5,
        optimizer_order_boundary_preserved=True,
        prompt_version="meio.llm_orchestrator.v4",
        prompt_hash="prompt_contract_hash",
        validation_failure_counts=(("malformed_json", 1),),
        client_error_counts=(("network_error", 2),),
        average_prompt_tokens=120.0,
        average_completion_tokens=40.0,
        average_total_tokens=160.0,
        average_llm_latency_ms=18.0,
        average_orchestration_latency_ms=22.0,
        total_retry_count=2,
        failure_before_response_count=2,
        failure_after_response_count=0,
    )

    assert diagnostics.provider == "openai"
    assert diagnostics.successful_response_count == 5
    assert diagnostics.prompt_version == "meio.llm_orchestrator.v4"
    assert diagnostics.average_total_tokens == 160.0
    assert diagnostics.client_error_counts == (("network_error", 2),)


def test_comparison_run_summary_rejects_negative_average() -> None:
    with pytest.raises(ValueError, match="average_tool_call_count"):
        ComparisonRunSummary(
            mode="llm_orchestrator",
            run_count=1,
            average_tool_call_count=-1.0,
            average_replan_count=0.0,
            average_abstain_count=0.0,
            average_no_action_count=0.0,
            average_total_cost=1.0,
            average_fill_rate=1.0,
            optimizer_order_boundary_preserved=True,
        )
