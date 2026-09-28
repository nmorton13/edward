"""Tests for resource summarization stage, privacy enforcement, and configuration posture."""

import datetime
import json
from unittest.mock import MagicMock, patch

import pytest

from edward.blobs import BlobStore
from edward.db import Database
from edward.services.llm import (
    LLMResponseValidationError,
    get_summarizer_client,
)
from edward.services.processor import process_pending_jobs
from edward.services.resource import store_resource_content
from edward.services.summarize import (
    ResourceSummaryPayload,
    enqueue_missing_summarize_jobs,
)


@pytest.fixture
def test_db_and_blobs(tmp_path):
    db_file = tmp_path / "test_edward.db"
    blobs_dir = tmp_path / "blobs"
    db = Database(db_file)
    db.run_migrations()
    blob_store = BlobStore(blobs_dir)
    return db, blob_store


@pytest.fixture(autouse=True)
def _clear_summarizer_env(monkeypatch):
    """Ensure no ambient summarizer configuration leaks into tests."""
    for name in (
        "EDWARD_SUMMARIZER_MODE",
        "EDWARD_SUMMARIZER_BASE_URL",
        "EDWARD_SUMMARIZER_MODEL",
        "EDWARD_SUMMARIZER_LOCATION",
        "EDWARD_SUMMARIZER_API_KEY",
        "EDWARD_SUMMARIZER_PROVIDER",
        "EDWARD_SUMMARIZER_TIMEOUT",
        "EDWARD_HOSTED_GMAIL",
        "EDWARD_HOSTED_PERSONAL_NOTES",
        "EDWARD_HOSTED_DOCUMENTS",
        "EDWARD_HOSTED_PUBLIC_WEB",
    ):
        monkeypatch.delenv(name, raising=False)


# --- 1. Posture & Configuration Tests ---


def test_summarizer_is_absent_by_default():
    """By default, summarizer client is None (opt-in only)."""
    assert get_summarizer_client() is None


def test_summarizer_absent_when_only_base_url_or_model_present(monkeypatch):
    """Stale base URL or model name does not implicitly enable summarizer."""
    monkeypatch.setenv("EDWARD_SUMMARIZER_BASE_URL", "http://localhost:11434/v1")
    assert get_summarizer_client() is None

    monkeypatch.delenv("EDWARD_SUMMARIZER_BASE_URL")
    monkeypatch.setenv("EDWARD_SUMMARIZER_MODEL", "llama3.2")
    assert get_summarizer_client() is None


@pytest.mark.parametrize("mode", ["disabled", "none", "off", ""])
def test_summarizer_absent_for_disabling_modes(monkeypatch, mode):
    monkeypatch.setenv("EDWARD_SUMMARIZER_MODE", mode)
    monkeypatch.setenv("EDWARD_SUMMARIZER_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("EDWARD_SUMMARIZER_MODEL", "llama3.2")
    assert get_summarizer_client() is None


def test_summarizer_present_when_explicitly_enabled_local(monkeypatch):
    monkeypatch.setenv("EDWARD_SUMMARIZER_MODE", "local")
    monkeypatch.setenv("EDWARD_SUMMARIZER_MODEL", "llama3.2")
    monkeypatch.setenv("EDWARD_SUMMARIZER_LOCATION", "local")

    client = get_summarizer_client()
    assert client is not None
    assert client.model == "llama3.2"
    assert client.location == "local"


def test_summarizer_present_for_explicit_hosted(monkeypatch):
    monkeypatch.setenv("EDWARD_SUMMARIZER_MODE", "hosted")
    monkeypatch.setenv("EDWARD_SUMMARIZER_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("EDWARD_SUMMARIZER_MODEL", "gpt-4o-mini")

    client = get_summarizer_client()
    assert client is not None
    assert client.model == "gpt-4o-mini"


def test_summarizer_invalid_timeout(monkeypatch):
    monkeypatch.setenv("EDWARD_SUMMARIZER_MODE", "local")
    monkeypatch.setenv("EDWARD_SUMMARIZER_TIMEOUT", "-5")
    with pytest.raises(ValueError, match="EDWARD_SUMMARIZER_TIMEOUT"):
        get_summarizer_client()


# --- 2. Summary Generation & Storage with Fake Client ---


class FakeSummarizerClient:
    def __init__(
        self,
        summary_text: str = "This is a concise 2-3 sentence summary of the resource content.",
        location: str = "local",
        provider: str = "llm",
    ):
        self.summary_text = summary_text
        self.location = location
        self.provider = provider
        self.base_url = "http://localhost:11434/v1"
        self.model = "llama3.2"
        self.call_count = 0

    def chat_completion(self, messages, response_model=None, data_class="public_web", **kwargs):
        self.call_count += 1
        if response_model is ResourceSummaryPayload:
            payload = ResourceSummaryPayload(summary=self.summary_text)
            return json.dumps(payload.model_dump()), payload
        return self.summary_text, None


def test_summarize_stores_summary(test_db_and_blobs):
    """Running summarize on a resource with extracted text stores the summary."""
    db, blob_store = test_db_and_blobs
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()

    with db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at)
            VALUES ('res_sum_1', 'url:example.com/ai', 'https://example.com/ai', 'AI Overview', 'unreviewed', 0, ?, ?);
            """,
            (now_iso, now_iso),
        )
        store_resource_content(
            conn,
            "res_sum_1",
            clean_text="Artificial intelligence has advanced rapidly in recent years. New foundation models enable reasoning and multimodal synthesis across domains.",
        )

    fake_client = FakeSummarizerClient("AI has advanced rapidly through modern foundation models.")

    # Enqueue and process summarize job
    with db.transaction() as conn:
        queued = enqueue_missing_summarize_jobs(conn)
        assert queued == 1

    res = process_pending_jobs(db, blob_store, stage="summarize", llm_client=fake_client)
    assert res["completed"] == 1
    assert res["failed"] == 0
    assert fake_client.call_count == 1

    with db.connection() as conn:
        row = conn.execute(
            "SELECT summary FROM resource_contents WHERE resource_id = 'res_sum_1' ORDER BY created_at DESC LIMIT 1;"
        ).fetchone()
        assert row["summary"] == "AI has advanced rapidly through modern foundation models."


# --- 3. Privacy Refusal Before Dispatch ---


def test_private_resource_refused_before_dispatch(test_db_and_blobs, monkeypatch):
    """A private-class resource (e.g. Gmail) is refused before dispatch to a hosted model."""
    monkeypatch.setenv("EDWARD_HOSTED_GMAIL", "deny")
    db, blob_store = test_db_and_blobs
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()

    with db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO captures (id, origin_namespace, collection_channel, collector, acquisition_method, retrieved_at, raw_content, review_state, is_deleted, created_at, updated_at)
            VALUES ('cap_gmail_1', 'gmail', 'gmail', 'test', 'manual', ?, 'Confidential email body', 'unreviewed', 0, ?, ?);
            """,
            (now_iso, now_iso, now_iso),
        )
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at)
            VALUES ('res_gmail_1', 'url:gmail:msg123', 'https://mail.google.com/mail/u/0/#inbox/123', 'Email thread', 'unreviewed', 0, ?, ?);
            """,
            (now_iso, now_iso),
        )
        conn.execute(
            """
            INSERT INTO capture_resources (capture_id, resource_id, relationship_type, created_at)
            VALUES ('cap_gmail_1', 'res_gmail_1', 'primary', ?);
            """,
            (now_iso,),
        )
        store_resource_content(
            conn,
            "res_gmail_1",
            clean_text="Secret financial discussion over email.",
            capture_id="cap_gmail_1",
        )

    # Hosted client with mock chat_completion to verify it is NEVER called
    hosted_client = FakeSummarizerClient(location="hosted", provider="openrouter")
    hosted_client.chat_completion = MagicMock()

    with db.transaction() as conn:
        enqueue_missing_summarize_jobs(conn)

    res = process_pending_jobs(db, blob_store, stage="summarize", llm_client=hosted_client)
    assert res["failed"] == 1
    assert res["completed"] == 0
    # Dispatch must not have been reached
    hosted_client.chat_completion.assert_not_called()

    with db.connection() as conn:
        job = conn.execute(
            "SELECT status, last_error FROM processing_jobs WHERE stage = 'summarize' AND resource_id = 'res_gmail_1';"
        ).fetchone()
        assert job["status"] == "failed"
        assert "Privacy policy violation" in job["last_error"]


# --- 4. Bundle Summaries are Preserved ---


def test_bundle_summaries_are_preserved(test_db_and_blobs):
    """Summaries from research bundles (or human) must not be replaced."""
    db, blob_store = test_db_and_blobs
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()

    existing_summary = "Existing bundle summary that must be preserved."
    with db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at)
            VALUES ('res_bundle_1', 'url:example.com/report', 'https://example.com/report', 'Report', 'unreviewed', 0, ?, ?);
            """,
            (now_iso, now_iso),
        )
        store_resource_content(
            conn,
            "res_bundle_1",
            clean_text="Detailed report text about energy storage technologies.",
            summary=existing_summary,
            extractor="markdown-report",
        )

    fake_client = FakeSummarizerClient("New summary that should never overwrite.")

    with db.transaction() as conn:
        enqueue_missing_summarize_jobs(conn)

    process_pending_jobs(db, blob_store, stage="summarize", llm_client=fake_client)
    # The job completes without calling LLM
    assert fake_client.call_count == 0

    with db.connection() as conn:
        row = conn.execute(
            "SELECT summary FROM resource_contents WHERE resource_id = 'res_bundle_1' ORDER BY created_at DESC LIMIT 1;"
        ).fetchone()
        assert row["summary"] == existing_summary


# --- 5. Idempotency & Rerun is a No-Op ---


def test_summarize_rerun_is_noop(test_db_and_blobs):
    """Running summarize a second time on an already-summarized resource is a no-op."""
    db, blob_store = test_db_and_blobs
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()

    with db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at)
            VALUES ('res_rerun_1', 'url:example.com/rerun', 'https://example.com/rerun', 'Rerun Test', 'unreviewed', 0, ?, ?);
            """,
            (now_iso, now_iso),
        )
        store_resource_content(
            conn,
            "res_rerun_1",
            clean_text="Some text about computing architectures.",
        )

    fake_client = FakeSummarizerClient("Summary on first pass.")

    with db.transaction() as conn:
        enqueue_missing_summarize_jobs(conn)

    res1 = process_pending_jobs(db, blob_store, stage="summarize", llm_client=fake_client)
    assert res1["completed"] == 1
    assert fake_client.call_count == 1

    # Second pass: rerun must be a no-op
    res2 = process_pending_jobs(db, blob_store, stage="summarize", llm_client=fake_client)
    assert res2["completed"] == 0
    assert fake_client.call_count == 1


# --- 6. Diagnostics Logging on Invalid Model Output ---


class InvalidOutputLLMClient:
    location = "local"
    provider = "llm"
    base_url = "http://localhost:11434/v1"
    model = "mock-fail"

    def chat_completion(self, messages, response_model=None, data_class="public_web", **kwargs):
        from edward.services.diagnostics import record_model_diagnostic

        raw = "Not valid JSON at all: 404 unexpected text"
        diag_path = record_model_diagnostic(
            provider="llm-mock-fail",
            raw_output=raw,
            error_message="Schema validation error",
            data_class=data_class,
        )
        raise LLMResponseValidationError(
            "Model output failed validation",
            raw_output=raw,
            diagnostic_path=str(diag_path) if diag_path else None,
        )


def test_invalid_model_output_recorded_to_diagnostics(test_db_and_blobs, tmp_path, monkeypatch):
    """Invalid raw model outputs are saved to diagnostics and job fails."""
    diag_dir = tmp_path / "diagnostics"
    monkeypatch.setenv("EDWARD_DATA_DIR", str(tmp_path))

    db, blob_store = test_db_and_blobs
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()

    with db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at)
            VALUES ('res_diag_1', 'url:example.com/diag', 'https://example.com/diag', 'Diag Test', 'unreviewed', 0, ?, ?);
            """,
            (now_iso, now_iso),
        )
        store_resource_content(
            conn,
            "res_diag_1",
            clean_text="Sample text to trigger invalid response handling.",
        )
        enqueue_missing_summarize_jobs(conn)

    res = process_pending_jobs(
        db, blob_store, stage="summarize", llm_client=InvalidOutputLLMClient()
    )
    assert res["failed"] == 1

    # Diagnostics file was created
    diag_files = list(diag_dir.glob("*.json"))
    assert len(diag_files) >= 1
    diag_content = json.loads(diag_files[0].read_text(encoding="utf-8"))
    assert "Not valid JSON" in diag_content["raw_output"]


# --- 7. CLI Scoping and Status JSON Reflection ---


def test_cli_status_reflects_summarized_resources(test_db_and_blobs, monkeypatch):
    """edward status --json summarized_resources reflects the count of summarized resources."""
    from typer.testing import CliRunner

    from edward.cli import app

    runner = CliRunner()
    db, blob_store = test_db_and_blobs
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()

    with db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at)
            VALUES ('res_st_1', 'url:example.com/st1', 'https://example.com/st1', 'Status Test', 'unreviewed', 0, ?, ?);
            """,
            (now_iso, now_iso),
        )
        store_resource_content(
            conn,
            "res_st_1",
            clean_text="Text for status test.",
            summary="A stored summary.",
        )

    with patch("edward.cli.get_services", return_value=(db, blob_store)):
        result = runner.invoke(app, ["status", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert "counts" in data
        assert data["counts"]["summarized_resources"] == 1


def test_cli_process_summarize_unconfigured_error(test_db_and_blobs):
    """edward process --stage summarize exits with code 1 if summarizer is disabled."""
    from typer.testing import CliRunner

    from edward.cli import app

    runner = CliRunner()
    db, blob_store = test_db_and_blobs

    with patch("edward.cli.get_services", return_value=(db, blob_store)):
        result = runner.invoke(app, ["process", "--stage", "summarize", "--json"])
        assert result.exit_code == 1
        data = json.loads(result.stderr)
        assert "error" in data
        assert "Summarizer model is not configured" in data["error"]


def test_cli_process_summarize_scoping(test_db_and_blobs, monkeypatch):
    """edward process --stage summarize --capture-id scopes to the specified capture."""
    from typer.testing import CliRunner

    from edward.cli import app

    runner = CliRunner()
    db, blob_store = test_db_and_blobs
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()

    monkeypatch.setenv("EDWARD_SUMMARIZER_MODE", "local")
    monkeypatch.setenv("EDWARD_SUMMARIZER_LOCATION", "local")

    with db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO captures (id, origin_namespace, collection_channel, collector, acquisition_method, retrieved_at, raw_content, review_state, is_deleted, created_at, updated_at)
            VALUES ('cap_scope_1', 'web', 'web', 'test', 'manual', ?, 'Note 1', 'unreviewed', 0, ?, ?),
                   ('cap_scope_2', 'web', 'web', 'test', 'manual', ?, 'Note 2', 'unreviewed', 0, ?, ?);
            """,
            (now_iso, now_iso, now_iso, now_iso, now_iso, now_iso),
        )
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at)
            VALUES ('res_scope_1', 'url:example.com/s1', 'https://example.com/s1', 'T1', 'unreviewed', 0, ?, ?),
                   ('res_scope_2', 'url:example.com/s2', 'https://example.com/s2', 'T2', 'unreviewed', 0, ?, ?);
            """,
            (now_iso, now_iso, now_iso, now_iso),
        )
        conn.execute(
            """
            INSERT INTO capture_resources (capture_id, resource_id, relationship_type, created_at)
            VALUES ('cap_scope_1', 'res_scope_1', 'primary', ?),
                   ('cap_scope_2', 'res_scope_2', 'primary', ?);
            """,
            (now_iso, now_iso),
        )
        store_resource_content(
            conn, "res_scope_1", clean_text="Content 1", capture_id="cap_scope_1"
        )
        store_resource_content(
            conn, "res_scope_2", clean_text="Content 2", capture_id="cap_scope_2"
        )

    fake_client = FakeSummarizerClient("Scoped summary.")
    with (
        patch("edward.cli.get_services", return_value=(db, blob_store)),
        patch("edward.services.llm.get_summarizer_client", return_value=fake_client),
        patch("edward.cli.get_summarizer_client", return_value=fake_client),
    ):
        result = runner.invoke(
            app,
            [
                "process",
                "--stage",
                "summarize",
                "--capture-id",
                "cap_scope_1",
                "--limit",
                "5",
                "--json",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["completed"] == 1

    with db.connection() as conn:
        r1 = conn.execute(
            "SELECT summary FROM resource_contents WHERE resource_id = 'res_scope_1' ORDER BY created_at DESC LIMIT 1;"
        ).fetchone()
        r2 = conn.execute(
            "SELECT summary FROM resource_contents WHERE resource_id = 'res_scope_2' ORDER BY created_at DESC LIMIT 1;"
        ).fetchone()
        assert r1["summary"] == "Scoped summary."
        assert r2["summary"] is None  # Not processed because of scoping!


# --- 9. Legacy Summaries Replaced, Human Preserved, and Migration Provenance ---


def test_human_summaries_are_preserved(test_db_and_blobs):
    """Human summaries must never be replaced by automated summarizer."""
    db, blob_store = test_db_and_blobs
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()

    existing_summary = "Human-authored synthesis of the paper."
    with db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at)
            VALUES ('res_human_1', 'url:example.com/human', 'https://example.com/human', 'Human Paper', 'unreviewed', 0, ?, ?);
            """,
            (now_iso, now_iso),
        )
        store_resource_content(
            conn,
            "res_human_1",
            clean_text="Detailed paper text about neural architectures.",
            summary=existing_summary,
            summary_source="human",
            extractor="human",
        )

    fake_client = FakeSummarizerClient("Model summary that must never overwrite human summary.")

    with db.transaction() as conn:
        queued = enqueue_missing_summarize_jobs(conn)
        assert queued == 0

    res = process_pending_jobs(db, blob_store, stage="summarize", llm_client=fake_client)
    assert res["completed"] == 0
    assert fake_client.call_count == 0

    with db.connection() as conn:
        row = conn.execute(
            "SELECT summary, summary_source FROM resource_contents WHERE resource_id = 'res_human_1' ORDER BY created_at DESC LIMIT 1;"
        ).fetchone()
        assert row["summary"] == existing_summary
        assert row["summary_source"] == "human"


def test_legacy_summaries_get_queued_and_replaced(test_db_and_blobs):
    """Low-quality legacy summaries (page lead text) are queued and replaced by model summaries."""
    db, blob_store = test_db_and_blobs
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()

    legacy_summary = "West Virginia State Treasurer's Office &ndash; About Us"
    with db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at)
            VALUES ('res_legacy_1', 'url:example.com/wv', 'https://example.com/wv', 'WV Office', 'unreviewed', 0, ?, ?);
            """,
            (now_iso, now_iso),
        )
        store_resource_content(
            conn,
            "res_legacy_1",
            clean_text="The West Virginia State Treasurer Office manages the state's financial resources and investments.",
            summary=legacy_summary,
            summary_source="legacy",
            extractor="summarize",
        )

    fake_client = FakeSummarizerClient("New concise model summary of the West Virginia Treasury.")

    # 1. Enqueue must queue the legacy row like a missing summary
    with db.transaction() as conn:
        queued = enqueue_missing_summarize_jobs(conn)
        assert queued == 1

    # 2. Process stage must call LLM and replace the legacy summary with model summary
    res = process_pending_jobs(db, blob_store, stage="summarize", llm_client=fake_client)
    assert res["completed"] == 1
    assert fake_client.call_count == 1

    with db.connection() as conn:
        row = conn.execute(
            "SELECT summary, summary_source FROM resource_contents WHERE resource_id = 'res_legacy_1' ORDER BY created_at DESC LIMIT 1;"
        ).fetchone()
        assert row["summary"] == "New concise model summary of the West Virginia Treasury."
        assert row["summary_source"] == "model"

    # 3. Rerun is still a no-op
    with db.transaction() as conn:
        queued_again = enqueue_missing_summarize_jobs(conn)
        assert queued_again == 0

    res_rerun = process_pending_jobs(db, blob_store, stage="summarize", llm_client=fake_client)
    assert res_rerun["completed"] == 0
    assert fake_client.call_count == 1


def test_migration_006_backfill_assigns_correct_sources(tmp_path):
    """Migration 006 backfills summary_source correctly across legacy, bundle, model, and human."""
    db_file = tmp_path / "test_migration_006.sqlite3"
    db = Database(db_file)
    now_iso = "2026-09-27T12:00:00Z"

    # 1. Apply migrations 1 through 5
    import shutil
    from pathlib import Path

    migrations_dir = Path(__file__).parent.parent / "src" / "edward" / "migrations"
    migs_1_to_5 = tmp_path / "migs_1_to_5"
    migs_1_to_5.mkdir()
    for m_file in sorted(migrations_dir.glob("*.sql")):
        if m_file.name.startswith("006"):
            continue
        shutil.copy(m_file, migs_1_to_5 / m_file.name)
    db.run_migrations(migrations_dir=migs_1_to_5)

    # 2. Seed data before migration 006
    with db.transaction() as conn:
        # Bundle resource
        conn.execute(
            "INSERT INTO resources (id, identity_key, canonical_url, title, is_deleted, created_at, updated_at) VALUES ('res_bnd', 'url:bnd', 'https://example.com/bnd', 'Bnd', 0, ?, ?);",
            (now_iso, now_iso),
        )
        conn.execute(
            """
            INSERT INTO resource_contents (id, resource_id, content_hash, clean_text, summary, char_count, extractor, extractor_version, created_at)
            VALUES ('rc_bnd', 'res_bnd', 'hash_bnd', 'Bundle clean text', 'Bundle summary text', 17, 'markdown-report', '1.0', ?);
            """,
            (now_iso,),
        )

        # Legacy summarize resource (with completed job)
        conn.execute(
            "INSERT INTO resources (id, identity_key, canonical_url, title, is_deleted, created_at, updated_at) VALUES ('res_leg', 'url:leg', 'https://example.com/leg', 'Leg', 0, ?, ?);",
            (now_iso, now_iso),
        )
        conn.execute(
            """
            INSERT INTO resource_contents (id, resource_id, content_hash, clean_text, summary, char_count, extractor, extractor_version, created_at)
            VALUES ('rc_leg', 'res_leg', 'hash_leg', 'Legacy page text', 'West Virginia State Treasurer', 16, 'summarize', '0.21.x', ?);
            """,
            (now_iso,),
        )
        conn.execute(
            """
            INSERT INTO processing_jobs (id, job_key, resource_id, stage, status, input_hash, attempts, available_at, completed_at, created_at, updated_at)
            VALUES ('job_leg', 'summarize:res_leg', 'res_leg', 'summarize', 'completed', 'hash_leg', 0, ?, ?, ?, ?);
            """,
            (now_iso, now_iso, now_iso, now_iso),
        )

        # Model resource (previously completed model job with started_at set)
        conn.execute(
            "INSERT INTO resources (id, identity_key, canonical_url, title, is_deleted, created_at, updated_at) VALUES ('res_mod', 'url:mod', 'https://example.com/mod', 'Mod', 0, ?, ?);",
            (now_iso, now_iso),
        )
        conn.execute(
            """
            INSERT INTO resource_contents (id, resource_id, content_hash, clean_text, summary, char_count, extractor, extractor_version, created_at)
            VALUES ('rc_mod', 'res_mod', 'hash_mod', 'Model page text', 'Model generated summary', 15, 'xurl', '1.0', ?);
            """,
            (now_iso,),
        )
        conn.execute(
            """
            INSERT INTO processing_jobs (id, job_key, resource_id, stage, status, input_hash, attempts, available_at, started_at, completed_at, created_at, updated_at)
            VALUES ('job_mod', 'summarize:res_mod', 'res_mod', 'summarize', 'completed', 'hash_mod', 0, ?, ?, ?, ?, ?);
            """,
            (now_iso, now_iso, now_iso, now_iso, now_iso),
        )

        # Human resource
        conn.execute(
            "INSERT INTO resources (id, identity_key, canonical_url, title, is_deleted, created_at, updated_at) VALUES ('res_hum', 'url:hum', 'https://example.com/hum', 'Hum', 0, ?, ?);",
            (now_iso, now_iso),
        )
        conn.execute(
            """
            INSERT INTO resource_contents (id, resource_id, content_hash, clean_text, summary, char_count, extractor, extractor_version, created_at)
            VALUES ('rc_hum', 'res_hum', 'hash_hum', 'Human page text', 'Human written summary', 15, 'human', '1.0', ?);
            """,
            (now_iso,),
        )

        # Resource with no summary
        conn.execute(
            "INSERT INTO resources (id, identity_key, canonical_url, title, is_deleted, created_at, updated_at) VALUES ('res_none', 'url:none', 'https://example.com/none', 'None', 0, ?, ?);",
            (now_iso, now_iso),
        )
        conn.execute(
            """
            INSERT INTO resource_contents (id, resource_id, content_hash, clean_text, summary, char_count, extractor, extractor_version, created_at)
            VALUES ('rc_none', 'res_none', 'hash_none', 'No summary page text', NULL, 18, 'local-fallback', '1.0', ?);
            """,
            (now_iso,),
        )

    # 3. Run migrations (applies 006_summary_source.sql)
    applied = db.run_migrations()
    assert "006_summary_source.sql" in applied

    # 4. Verify summary_source assignments and job status reset
    with db.connection() as conn:
        sources = {
            r["id"]: r["summary_source"]
            for r in conn.execute("SELECT id, summary_source FROM resource_contents").fetchall()
        }
        assert sources["rc_bnd"] == "bundle"
        assert sources["rc_leg"] == "legacy"
        assert sources["rc_mod"] == "model"
        assert sources["rc_hum"] == "human"
        assert sources["rc_none"] is None

        # Verify that legacy processing job was reset to pending
        leg_job = conn.execute("SELECT status FROM processing_jobs WHERE id = 'job_leg'").fetchone()
        assert leg_job["status"] == "pending"

        # Verify that model processing job remained completed
        mod_job = conn.execute("SELECT status FROM processing_jobs WHERE id = 'job_mod'").fetchone()
        assert mod_job["status"] == "completed"
