"""Tests for the answerer's configuration posture.

Edward is memory, not a mind. No model may be contacted unless an operator
explicitly configures one: absence must be the default, not a thing you opt out of.
"""

import pytest

from edward.services.llm import get_answer_client


@pytest.fixture(autouse=True)
def _clear_answerer_env(monkeypatch):
    """Ensure no ambient answerer configuration leaks into these tests."""
    for name in (
        "EDWARD_ANSWERER_MODE",
        "EDWARD_ANSWER_BASE_URL",
        "EDWARD_ANSWER_MODEL",
        "EDWARD_ANSWER_LOCATION",
        "EDWARD_ANSWER_API_KEY",
        "EDWARD_ANSWER_PROVIDER",
        "EDWARD_ANSWER_TIMEOUT",
    ):
        monkeypatch.delenv(name, raising=False)


def test_answerer_is_absent_by_default():
    """With nothing configured, no client is constructed."""
    assert get_answer_client() is None


def test_answerer_absent_when_only_a_stale_base_url_is_present(monkeypatch):
    """A leftover base URL alone must not implicitly re-enable a model."""
    monkeypatch.setenv("EDWARD_ANSWER_BASE_URL", "http://localhost:11434/v1")

    assert get_answer_client() is None


def test_answerer_absent_when_only_a_stale_model_name_is_present(monkeypatch):
    """A leftover model name alone must not implicitly re-enable a model."""
    monkeypatch.setenv("EDWARD_ANSWER_MODEL", "granite4.2:3b")

    assert get_answer_client() is None


@pytest.mark.parametrize("mode", ["disabled", "none", "off"])
def test_answerer_absent_for_every_disabling_mode(monkeypatch, mode):
    monkeypatch.setenv("EDWARD_ANSWERER_MODE", mode)
    monkeypatch.setenv("EDWARD_ANSWER_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("EDWARD_ANSWER_MODEL", "granite4.2:3b")

    assert get_answer_client() is None


@pytest.mark.parametrize("mode", ["local-model", "local"])
def test_answerer_present_when_explicitly_enabled_for_local(monkeypatch, mode):
    """An operator who opts in gets a client."""
    monkeypatch.setenv("EDWARD_ANSWERER_MODE", mode)
    monkeypatch.setenv("EDWARD_ANSWER_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("EDWARD_ANSWER_MODEL", "granite4.2:3b")
    monkeypatch.setenv("EDWARD_ANSWER_LOCATION", "local")

    client = get_answer_client()

    assert client is not None
    assert client.model == "granite4.2:3b"
    assert client.location == "local"


def test_answerer_present_for_explicit_hosted_mode(monkeypatch):
    """A hosted answerer requires explicit opt-in plus full configuration."""
    monkeypatch.setenv("EDWARD_ANSWERER_MODE", "hosted")
    monkeypatch.setenv("EDWARD_ANSWER_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("EDWARD_ANSWER_MODEL", "some-model")

    client = get_answer_client()

    assert client is not None
    assert client.model == "some-model"


def test_enabling_mode_without_endpoint_or_model_stays_absent(monkeypatch):
    """Opting in is necessary but not sufficient: an endpoint or model must be present."""
    monkeypatch.setenv("EDWARD_ANSWERER_MODE", "hosted")

    assert get_answer_client() is None


def test_invalid_timeout_still_raises_when_enabled(monkeypatch):
    """Timeout validation must survive the default change."""
    monkeypatch.setenv("EDWARD_ANSWERER_MODE", "local-model")
    monkeypatch.setenv("EDWARD_ANSWER_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("EDWARD_ANSWER_TIMEOUT", "0")

    with pytest.raises(ValueError, match="EDWARD_ANSWER_TIMEOUT"):
        get_answer_client()
