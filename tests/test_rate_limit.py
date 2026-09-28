"""Rate limits re-queue jobs after the provider's delay without consuming attempts."""

import datetime
import email.utils

import httpx
import pytest

from edward.blobs import BlobStore
from edward.db import Database
from edward.services import processor
from edward.services.llm import LLMClient, LLMRateLimitError, parse_retry_after
from edward.services.processor import process_pending_jobs
from edward.services.resource import store_resource_content
from edward.services.summarize import enqueue_missing_summarize_jobs

LONG_TEXT = "Long enough resource text for the summarizer. " + " ".join(
    f"word{i}" for i in range(80)
)


@pytest.fixture
def db_and_blobs(tmp_path):
    db = Database(tmp_path / "edward.db")
    db.run_migrations()
    return db, BlobStore(tmp_path / "blobs")


def _seed(db, count: int) -> None:
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()
    with db.transaction() as conn:
        for i in range(count):
            conn.execute(
                """
                INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'unreviewed', 0, ?, ?);
                """,
                (f"res_{i}", f"url:e/{i}", f"https://example.com/{i}", f"T{i}", now_iso, now_iso),
            )
            store_resource_content(conn, f"res_{i}", clean_text=f"{LONG_TEXT} doc{i}")
        enqueue_missing_summarize_jobs(conn)


class RateLimitedClient:
    location = "local"
    provider = "llm"
    base_url = "http://localhost:11434/v1"
    model = "fake"

    def __init__(self, retry_after: float | None):
        self.retry_after = retry_after
        self.call_count = 0

    def chat_completion(self, *args, **kwargs):
        self.call_count += 1
        raise LLMRateLimitError("rate limited (429)", retry_after=self.retry_after)


def _jobs(db):
    with db.connection() as conn:
        return {
            r["resource_id"]: dict(r)
            for r in conn.execute(
                "SELECT resource_id, status, attempts, available_at, last_error FROM processing_jobs WHERE stage = 'summarize';"
            )
        }


def _delay_seconds(available_at: str) -> float:
    avail = datetime.datetime.fromisoformat(available_at)
    return (avail - datetime.datetime.now(datetime.UTC)).total_seconds()


def test_rate_limit_requeues_without_attempt_and_stops_run(db_and_blobs):
    db, blobs = db_and_blobs
    _seed(db, 3)
    client = RateLimitedClient(retry_after=30)

    res = process_pending_jobs(db, blobs, stage="summarize", limit=10, llm_client=client)

    assert client.call_count == 1  # the run stopped instead of hammering the provider
    assert res["rate_limited"] == 1
    assert res["failed"] == 0
    jobs = _jobs(db)
    limited = [j for j in jobs.values() if j["last_error"]]
    assert len(limited) == 1
    assert limited[0]["status"] == "pending"
    assert limited[0]["attempts"] == 0
    assert 25 < _delay_seconds(limited[0]["available_at"]) <= 30
    assert all(j["status"] == "pending" and j["attempts"] == 0 for j in jobs.values())


def test_repeated_rate_limits_never_fail_the_job(db_and_blobs):
    db, blobs = db_and_blobs
    _seed(db, 1)
    client = RateLimitedClient(retry_after=None)
    for _ in range(5):
        with db.transaction() as conn:
            conn.execute("UPDATE processing_jobs SET available_at = '2000-01-01T00:00:00+00:00';")
        process_pending_jobs(db, blobs, stage="summarize", llm_client=client)

    job = _jobs(db)["res_0"]
    assert client.call_count == 5
    assert job["status"] == "pending"
    assert job["attempts"] == 0


@pytest.mark.parametrize(
    ("hint", "low", "high"),
    [
        (None, 55, processor.RATE_LIMIT_DEFAULT_DELAY_SEC),
        (0, 55, processor.RATE_LIMIT_DEFAULT_DELAY_SEC),
        (100_000, 890, processor.RATE_LIMIT_MAX_DELAY_SEC),
    ],
)
def test_rate_limit_delay_defaults_and_cap(db_and_blobs, hint, low, high):
    db, blobs = db_and_blobs
    _seed(db, 1)
    process_pending_jobs(db, blobs, stage="summarize", llm_client=RateLimitedClient(hint))
    assert low < _delay_seconds(_jobs(db)["res_0"]["available_at"]) <= high


def test_llm_client_raises_rate_limit_error_with_hint():
    def handler(request):
        return httpx.Response(429, headers={"Retry-After": "42"}, text='{"error":"slow down"}')

    client = LLMClient(base_url="http://localhost:11434/v1", model="m", location="local")
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(LLMRateLimitError) as exc:
            client.chat_completion(
                messages=[{"role": "user", "content": "hi"}], data_class="public_web", client=http
            )
    assert exc.value.retry_after == 42


def test_parse_retry_after_formats():
    assert parse_retry_after(None) is None
    assert parse_retry_after("") is None
    assert parse_retry_after("7") == 7
    assert parse_retry_after("garbage") is None
    future = datetime.datetime.now(datetime.UTC) + datetime.timedelta(seconds=120)
    parsed = parse_retry_after(email.utils.format_datetime(future, usegmt=True))
    assert parsed is not None and 110 < parsed <= 120
