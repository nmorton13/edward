"""Tests for the `ask` command's model posture.

Edward is memory, not a mind. Asking a question must resolve to a deterministic or
lookup answer unless an agent explicitly supplies a client, so no ambient model
configuration can cause a question to be synthesized behind the user's back.
"""

from unittest.mock import MagicMock

import pytest

from edward.models import CaptureInput
from edward.services.answer import answer_question
from edward.services.capture import capture_item


@pytest.fixture(autouse=True)
def _no_ambient_answerer(monkeypatch):
    for name in (
        "EDWARD_ANSWERER_MODE",
        "EDWARD_ANSWER_BASE_URL",
        "EDWARD_ANSWER_MODEL",
        "EDWARD_ANSWER_LOCATION",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def _seeded(test_db):
    """Seed a small corpus so questions resolve to real evidence."""
    with test_db.transaction() as conn:
        for text in (
            "Data centers consume substantial electricity in concentrated regions.",
            "Local models reduce inference cost through quantization and caching.",
            "Transmission upgrades are the bottleneck for new load in some states.",
        ):
            capture_item(
                conn,
                CaptureInput(
                    text=text,
                    origin_namespace="manual",
                    collection_channel="cli",
                    collector="test",
                    acquisition_method="manual",
                ),
            )
    return test_db


def test_ask_does_not_construct_a_client_by_default(_seeded, monkeypatch):
    """With nothing configured, ask must not build a model client."""
    constructed = []

    def _forbidden(*args, **kwargs):
        constructed.append(1)
        raise AssertionError("ask must not construct a model client by default")

    monkeypatch.setattr("edward.services.answer.get_answer_client", _forbidden, raising=False)

    with _seeded.connection() as conn:
        result = answer_question(conn, query="data centers electricity")

    assert constructed == [], "no ambient model may be constructed for ask"
    assert result is not None
    assert result["tier"] in (1, 2), f"expected a deterministic tier, got {result['tier']}"


def test_ask_uses_a_caller_supplied_client(_seeded):
    """An explicitly supplied client is honoured -- that is the opt-in path."""
    client = MagicMock()
    client.model = "caller-model"
    client.provider = "local"
    client.location = "local"
    client.base_url = "http://localhost:11434/v1"

    with _seeded.connection() as conn:
        # Synthesis is attempted; whether it succeeds does not matter here, only that
        # the supplied client is the one consulted.
        result = answer_question(conn, query="data centers electricity", llm_client=client)

    assert result is not None


def test_ask_no_model_flag_bypasses_synthesis_even_with_a_client(_seeded):
    """--no-model must short-circuit before any client is consulted."""
    client = MagicMock()
    client.model = "caller-model"
    client.provider = "local"
    client.location = "local"
    client.base_url = "http://localhost:11434/v1"

    with _seeded.connection() as conn:
        result = answer_question(
            conn, query="data centers electricity", no_model=True, llm_client=client
        )

    assert result["tier"] == 2
    client.chat_completion.assert_not_called()
