"""Tests verifying that Edward functions completely in vector-free mode."""

from pathlib import Path

from edward.blobs import BlobStore
from edward.db import Database
from edward.models import CaptureInput
from edward.services.capture import capture_item
from edward.services.diagnostics import run_doctor
from edward.services.search import search_lexical


def test_vector_free_operations(tmp_path: Path):
    db_file = tmp_path / "vec_free.sqlite3"
    db = Database(db_file)
    db.run_migrations()

    # Even without sqlite-vec loaded, system should operate cleanly
    with db.transaction() as conn:
        res = capture_item(
            conn, CaptureInput(url="https://example.com/vector-free", note="No vectors required")
        )
        assert res["status"] == "created"

    with db.connection() as conn:
        search_res = search_lexical(conn, "vectors")
        assert search_res.count == 1
        assert "vectors" in search_res.results[0].title.lower()

    # Doctor check passes cleanly in vector-free mode
    blob_store = BlobStore(tmp_path / "blobs")
    report = run_doctor(db, blob_store)
    assert report["healthy"] is True
    assert "vector_extension" in report
