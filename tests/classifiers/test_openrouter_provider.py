"""Unit tests for OpenRouter System One provider with mocked HTTP responses."""

import json

import httpx
import pytest

from edward.classifiers.providers.openrouter import OpenRouterProvider
from edward.classifiers.providers.typesafe import (
    TypeSafeAuthenticationError,
    TypeSafeRateLimitError,
)
from edward.services.privacy import PrivacyTransmissionError


def test_openrouter_privacy_enforcement(monkeypatch: pytest.MonkeyPatch):
    """Privacy check halts BEFORE request dispatch on private personal notes."""
    monkeypatch.delenv("EDWARD_HOSTED_PERSONAL_NOTES", raising=False)
    provider = OpenRouterProvider(api_key="or-key")

    with pytest.raises(PrivacyTransmissionError):
        provider.evaluate_questions("Personal note text", [], data_class="personal_notes")


def test_openrouter_missing_api_key(monkeypatch: pytest.MonkeyPatch):
    """Missing OPENROUTER_API_KEY raises TypeSafeAuthenticationError."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    provider = OpenRouterProvider(api_key="")
    with pytest.raises(TypeSafeAuthenticationError):
        provider.evaluate_questions("Public text", [], data_class="public_web")


def test_openrouter_header_and_request_id_preservation():
    """Validates OpenRouter headers, request ID (x-openrouter-id), and model resolution."""
    observed_headers = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed_headers.update(request.headers)
        assert request.url.path == "/api/alpha/decisions"
        body = json.loads(request.content)
        assert body["state"] == "Scientific study on deep learning reasoning."
        assert body["questions"]["primary-form"] == {
            "type": "choice",
            "instructions": "What kind of source?",
            "criteria": {"paper": "Paper", "article": "Article"},
        }
        payload = {
            "id": "gen-1234567890",
            "model": "typesafe/jev-1.13",
            "usage": {"prompt_tokens": 150, "completion_tokens": 50, "cost": 0.0015},
            "answers": {
                "primary-form": {
                    "type": "choice",
                    "choice": "paper",
                    "probabilities": {"paper": 0.90, "article": 0.10},
                }
            },
        }
        return httpx.Response(200, headers={"x-openrouter-id": "x-or-999"}, json=payload)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenRouterProvider(api_key="sk-or-testkey", model="~typesafe/jev-latest")

    result = provider.evaluate_questions(
        "Scientific study on deep learning reasoning.",
        [
            {
                "id": "primary-form",
                "primitive": "choice",
                "prompt": "What kind of source?",
                "options": [
                    {"id": "paper", "label": "Paper"},
                    {"id": "article", "label": "Article"},
                ],
            }
        ],
        client=client,
    )

    # Check headers
    assert observed_headers.get("authorization") == "Bearer sk-or-testkey"
    assert observed_headers.get("http-referer") == "https://edward.local"
    assert observed_headers.get("x-title") == "Edward"

    # Check preserved response metadata
    assert result["provider"] == "openrouter"
    assert result["requested_model"] == "~typesafe/jev-latest"
    assert result["resolved_model"] == "typesafe/jev-1.13"
    assert result["provider_request_id"] == "gen-1234567890" or "x-or-999"
    assert result["input_tokens"] == 150
    assert result["output_tokens"] == 50
    assert result["cost"] == 0.0015
    assert result["answers"]["primary-form"]["choice"] == "paper"


def test_openrouter_score_scale_maps_to_registry_values():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["questions"]["depth"]["criteria"] == ["Minimal", "Moderate", "Rigorous"]
        return httpx.Response(
            200,
            json={
                "answers": {
                    "depth": {
                        "type": "score",
                        "score": 2,
                        "legend": {"0": "Minimal", "1": "Moderate", "2": "Rigorous"},
                    }
                }
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = OpenRouterProvider(api_key="test").evaluate_questions(
        "Paper",
        [
            {
                "id": "depth",
                "primitive": "score",
                "prompt": "Score depth",
                "instructions": "1: Minimal; 2: Moderate; 3: Rigorous",
                "min_score": 1,
                "max_score": 3,
            }
        ],
        client=client,
    )
    assert result["answers"]["depth"]["score"] == 3
    assert result["answers"]["depth"]["legend"]["3"] == "Rigorous"


def test_openrouter_rate_limit_and_error():
    """Rate limit 429 raises TypeSafeRateLimitError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "45"}, text="OpenRouter Rate limit")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenRouterProvider(api_key="sk-or-testkey")

    with pytest.raises(TypeSafeRateLimitError) as exc_info:
        provider.evaluate_questions("Text", [], client=client)

    assert exc_info.value.retry_after == 45.0
