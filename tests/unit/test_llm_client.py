from __future__ import annotations

import json
from pathlib import Path

import httpx
import openai
import pytest

from meio.agents.llm_client import (
    FakeLLMClient,
    LLMClientContext,
    LLMCompletionRequest,
    LLMClientRuntimeError,
    LLMMessage,
    OpenAILLMClient,
)
from meio.agents.telemetry import ClientErrorCategory
from meio.contracts import RegimeLabel


def build_request(regime_label: RegimeLabel) -> LLMCompletionRequest:
    return LLMCompletionRequest(
        model="gpt-4o-mini",
        messages=(
            LLMMessage(role="system", content="bounded test"),
            LLMMessage(role="user", content="return json"),
        ),
        context=LLMClientContext(
            benchmark_id="serial_3_echelon",
            mission_id="test_mission",
            time_index=0,
            regime_label=regime_label,
            demand_value=10.0,
            leadtime_value=2.0,
            inventory_level=(20.0, 30.0, 40.0),
            backorder_level=(0.0, 0.0, 0.0),
            available_tool_ids=("forecast_tool", "leadtime_tool", "scenario_tool"),
            max_tool_steps=3,
        ),
    )


def test_fake_llm_client_returns_deterministic_json() -> None:
    response = FakeLLMClient().complete(build_request(RegimeLabel.DEMAND_REGIME_SHIFT))
    payload = json.loads(response.content)

    assert response.provider == "fake_llm_client"
    assert response.telemetry is not None
    assert response.telemetry.total_tokens == 144
    assert payload["selected_subgoal"] == "query_uncertainty"
    assert "scenario_tool" in payload["candidate_tool_ids"]


def test_fake_llm_client_respects_available_tool_set_in_ablation() -> None:
    request = build_request(RegimeLabel.DEMAND_REGIME_SHIFT)
    request = LLMCompletionRequest(
        model=request.model,
        messages=request.messages,
        context=LLMClientContext(
            benchmark_id=request.context.benchmark_id,
            mission_id=request.context.mission_id,
            time_index=request.context.time_index,
            regime_label=request.context.regime_label,
            demand_value=request.context.demand_value,
            leadtime_value=request.context.leadtime_value,
            inventory_level=request.context.inventory_level,
            backorder_level=request.context.backorder_level,
            available_tool_ids=("leadtime_tool", "scenario_tool"),
            max_tool_steps=request.context.max_tool_steps,
        ),
    )

    response = FakeLLMClient().complete(request)
    payload = json.loads(response.content)

    assert payload["candidate_tool_ids"] == ["leadtime_tool", "scenario_tool"]
    assert payload["selected_subgoal"] == "query_uncertainty"


def test_openai_client_fails_clearly_without_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    missing_env_file = repo_root / ".missing_openai.env"
    if missing_env_file.exists():
        missing_env_file.unlink()

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAILLMClient(env_file_path=missing_env_file).complete(build_request(RegimeLabel.NORMAL))


def test_openai_client_reads_api_key_from_env_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    env_file = repo_root / ".test_openai.env"
    env_file.write_text("OPENAI_API_KEY=test-key-from-env-file\n", encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    try:
        client = OpenAILLMClient(env_file_path=env_file)
        assert client.resolve_api_key() == "test-key-from-env-file"
    finally:
        if env_file.exists():
            env_file.unlink()


def test_openai_client_handles_missing_usage_metadata_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: dict[str, object] = {}

    class FakeResponses:
        def create(self, **kwargs):
            class FakeResponse:
                output_text = '{"selected_subgoal":"no_action"}'
                usage = None

            return FakeResponse()

    class FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            captured_kwargs.update(kwargs)
            self.responses = FakeResponses()

        def close(self) -> None:
            return None

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.delenv("ALL_PROXY", raising=False)
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)

    response = OpenAILLMClient().complete(build_request(RegimeLabel.NORMAL))

    assert response.telemetry is not None
    assert response.telemetry.prompt_tokens is None
    assert response.telemetry.completion_tokens is None
    assert response.telemetry.total_tokens is None
    assert "usage_unavailable" in response.metadata
    assert "proxy_env_bypassed" in response.metadata
    assert "http_client" in captured_kwargs


def test_openai_client_retries_network_error_and_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = {"value": 0}

    class FakeResponses:
        def create(self, **kwargs):
            call_count["value"] += 1
            if call_count["value"] == 1:
                raise httpx.ConnectError(
                    "No connection could be made because the target machine actively refused it",
                    request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
                )

            class FakeUsage:
                input_tokens = 120
                output_tokens = 30
                total_tokens = 150

            class FakeResponse:
                output_text = '{"selected_subgoal":"no_action"}'
                usage = FakeUsage()

            return FakeResponse()

    class FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            self.responses = FakeResponses()

        def close(self) -> None:
            return None

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)

    response = OpenAILLMClient(max_retries=1).complete(build_request(RegimeLabel.NORMAL))

    assert call_count["value"] == 2
    assert response.telemetry is not None
    assert response.telemetry.retry_count == 1
    assert response.telemetry.total_tokens == 150
    assert "retried_request" in response.metadata


def test_openai_client_classifies_network_failure_after_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponses:
        def create(self, **kwargs):
            raise httpx.ConnectError(
                "No connection could be made because the target machine actively refused it",
                request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
            )

    class FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            self.responses = FakeResponses()

        def close(self) -> None:
            return None

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)

    with pytest.raises(LLMClientRuntimeError) as exc_info:
        OpenAILLMClient(max_retries=1).complete(build_request(RegimeLabel.NORMAL))

    assert exc_info.value.category is ClientErrorCategory.NETWORK_ERROR
    assert exc_info.value.retry_count == 1
    assert exc_info.value.failure_after_response is False


def test_openai_client_classifies_empty_response_as_parse_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponses:
        def create(self, **kwargs):
            class FakeResponse:
                output_text = ""
                usage = None

            return FakeResponse()

    class FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            self.responses = FakeResponses()

        def close(self) -> None:
            return None

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)

    with pytest.raises(LLMClientRuntimeError) as exc_info:
        OpenAILLMClient().complete(build_request(RegimeLabel.NORMAL))

    assert exc_info.value.category is ClientErrorCategory.RESPONSE_PARSE_ERROR
    assert exc_info.value.failure_after_response is True
