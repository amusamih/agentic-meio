from __future__ import annotations

from meio.agents.telemetry import (
    ClientErrorCategory,
    LLMCallTrace,
    LLMCallTelemetry,
    OrchestrationStepTelemetry,
    ToolCallTrace,
    summarize_episode_telemetry,
)


def test_episode_telemetry_summary_aggregates_llm_and_step_counts() -> None:
    summary = summarize_episode_telemetry(
        (
            OrchestrationStepTelemetry(
                provider="fake_llm_client",
                model_name="gpt-4o-mini",
                prompt_tokens=90,
                completion_tokens=45,
                total_tokens=135,
                llm_latency_ms=12.0,
                orchestration_latency_ms=15.0,
                tool_call_count=3,
                selected_tools=("forecast_tool", "leadtime_tool", "scenario_tool"),
                request_replan=True,
            ),
            OrchestrationStepTelemetry(
                provider="openai",
                model_name="gpt-4o-mini",
                orchestration_latency_ms=2.5,
                tool_call_count=0,
                selected_tools=(),
                request_replan=False,
                abstain_or_no_action=True,
                fallback_used=True,
                client_error_category=ClientErrorCategory.NETWORK_ERROR,
                client_error_message="APIConnectionError: Connection error.",
                retry_count=1,
                failure_after_response=False,
            ),
        )
    )

    assert summary.provider == "fake_llm_client"
    assert summary.model_name == "gpt-4o-mini"
    assert summary.step_count == 2
    assert summary.llm_call_count == 2
    assert summary.average_total_tokens == 135.0
    assert summary.total_tool_call_count == 3
    assert summary.total_replan_count == 1
    assert summary.abstain_or_no_action_count == 1
    assert summary.client_error_counts == (("network_error", 1),)
    assert summary.total_retry_count == 1
    assert summary.failure_before_response_count == 1
    assert summary.failure_after_response_count == 0


def test_llm_call_telemetry_accepts_optional_usage_values() -> None:
    telemetry = LLMCallTelemetry(
        provider="openai",
        model_name="gpt-4o-mini",
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        llm_latency_ms=20.0,
    )

    assert telemetry.provider == "openai"
    assert telemetry.total_tokens is None


def test_detailed_trace_helpers_accept_bounded_trace_payloads() -> None:
    llm_trace = LLMCallTrace(
        provider="fake_llm_client",
        model_name="gpt-4o-mini",
        prompt_text="SYSTEM:\nBounded orchestration only.",
        prompt_hash="prompt_hash",
        raw_output_text='{"request_replan":false}',
        parsed_output={"request_replan": False},
        validation_success=True,
        prompt_tokens=96,
        completion_tokens=48,
        total_tokens=144,
        latency_ms=5.0,
        retry_count=1,
        client_error_category=ClientErrorCategory.NETWORK_ERROR,
        client_error_message="APIConnectionError: Connection error.",
        failure_after_response=False,
    )
    tool_trace = ToolCallTrace(
        tool_id="forecast_tool",
        tool_input={"tool_id": "forecast_tool"},
        tool_output={"status": "success"},
        success=True,
        latency_ms=1.0,
    )

    assert llm_trace.total_tokens == 144
    assert llm_trace.client_error_category is ClientErrorCategory.NETWORK_ERROR
    assert tool_trace.tool_output == {"status": "success"}
