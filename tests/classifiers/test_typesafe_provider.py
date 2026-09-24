"""Unit tests for TypeSafe System One provider with mocked HTTP responses."""

import httpx
import pytest

from edward.classifiers.providers.typesafe import (
    TypeSafeAuthenticationError,
    TypeSafeProvider,
    TypeSafeProviderError,
    TypeSafeRateLimitError,
    TypeSafeResponseValidationError,
)
from edward.services.privacy import PrivacyTransmissionError


def test_typesafe_privacy_enforcement(monkeypatch: pytest.MonkeyPatch):
    """Privacy check halts BEFORE request dispatch on private content."""
    monkeypatch.delenv("EDWARD_HOSTED_GMAIL", raising=False)
    provider = TypeSafeProvider(api_key="test-key")

    with pytest.raises(PrivacyTransmissionError):
        provider.evaluate_questions("Private text", [], data_class="gmail")


def test_typesafe_missing_api_key():
    """Missing API key raises TypeSafeAuthenticationError."""
    provider = TypeSafeProvider(api_key="")
    with pytest.raises(TypeSafeAuthenticationError):
        provider.evaluate_questions("Public text", [], data_class="public_web")


def test_typesafe_401_unauthorized():
    """401 status code raises TypeSafeAuthenticationError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="Unauthorized")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = TypeSafeProvider(api_key="bad-key")

    with pytest.raises(TypeSafeAuthenticationError):
        provider.evaluate_questions("Hello", [{"id": "q1", "primitive": "noul"}], client=client)


def test_typesafe_429_rate_limit():
    """429 status code raises TypeSafeRateLimitError with retry_after."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "30"}, text="Rate limited")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = TypeSafeProvider(api_key="test-key")

    with pytest.raises(TypeSafeRateLimitError) as exc_info:
        provider.evaluate_questions("Hello", [{"id": "q1", "primitive": "noul"}], client=client)

    assert exc_info.value.retry_after == 30.0


def test_typesafe_500_server_error():
    """500 status code raises TypeSafeProviderError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = TypeSafeProvider(api_key="test-key")

    with pytest.raises(TypeSafeProviderError):
        provider.evaluate_questions("Hello", [], client=client)


def test_typesafe_malformed_json_response():
    """Malformed JSON body raises TypeSafeResponseValidationError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="Not a JSON response")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = TypeSafeProvider(api_key="test-key")

    with pytest.raises(TypeSafeResponseValidationError):
        provider.evaluate_questions("Hello", [], client=client)


def test_typesafe_invalid_choice_distribution():
    """Choice answer with invalid probabilities distribution raises validation error."""

    def handler(request: httpx.Request) -> httpx.Response:
        payload = {
            "model": "typesafe/jev-latest",
            "answers": [
                {
                    "question_id": "primary-form",
                    "primitive": "choice",
                    "choice": "article",
                    # Sum = 0.5 (far from 1.0)
                    "probabilities": {"article": 0.3, "other": 0.2},
                }
            ],
        }
        return httpx.Response(200, json=payload)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = TypeSafeProvider(api_key="test-key")

    with pytest.raises(TypeSafeResponseValidationError) as exc_info:
        provider.evaluate_questions("Hello", [], client=client)
    assert "must sum to approximately 1.0" in str(exc_info.value)


def test_typesafe_noul_out_of_bounds():
    """Noul probability outside [0.0, 1.0] raises validation error."""

    def handler(request: httpx.Request) -> httpx.Response:
        payload = {
            "model": "typesafe/jev-latest",
            "answers": [
                {
                    "question_id": "topic-ai",
                    "primitive": "noul",
                    "noul": 1.45,  # Out of bounds
                }
            ],
        }
        return httpx.Response(200, json=payload)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = TypeSafeProvider(api_key="test-key")

    with pytest.raises(TypeSafeResponseValidationError):
        provider.evaluate_questions("Hello", [], client=client)


def test_typesafe_successful_evaluation():
    """Valid Choice, Noul, and Score response is normalized properly."""

    def handler(request: httpx.Request) -> httpx.Response:
        payload = {
            "id": "req_ts_123",
            "model": "typesafe/jev-1.13",
            "usage": {"input_tokens": 120, "output_tokens": 45, "cost": 0.0012},
            "answers": [
                {
                    "question_id": "primary-form",
                    "primitive": "choice",
                    "choice": "article",
                    "probabilities": {"article": 0.85, "paper": 0.10, "other": 0.05},
                    "confidence": 0.95,
                },
                {
                    "question_id": "topic-ai",
                    "primitive": "noul",
                    "noul": 0.88,
                    "confidence": 0.92,
                },
                {
                    "question_id": "evidence-depth",
                    "primitive": "score",
                    "score": 2.5,
                    "max_score": 3.0,
                },
            ],
        }
        return httpx.Response(200, json=payload)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = TypeSafeProvider(api_key="test-key")

    result = provider.evaluate_questions(
        "A detailed paper on local model benchmark latency.",
        [
            {"id": "primary-form", "primitive": "choice"},
            {"id": "topic-ai", "primitive": "noul"},
            {"id": "evidence-depth", "primitive": "score"},
        ],
        client=client,
    )

    assert result["provider"] == "typesafe"
    assert result["resolved_model"] == "typesafe/jev-1.13"
    assert result["provider_request_id"] == "req_ts_123"
    assert result["input_tokens"] == 120
    assert result["output_tokens"] == 45
    assert result["cost"] == 0.0012
    assert "primary-form" in result["answers"]
    assert result["answers"]["primary-form"]["choice"] == "article"
    assert result["answers"]["topic-ai"]["noul"] == 0.88
    assert result["answers"]["evidence-depth"]["score"] == 2.5
