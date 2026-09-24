"""Tests for content-addressed managed blob store."""

import hashlib
from pathlib import Path

import pytest

from edward.blobs import BlobIntegrityError, BlobStore, BlobStoreError


def test_store_and_read_bytes(test_blob_store: BlobStore):
    payload = b"Hello Edward Research Memory!"
    expected_hash = hashlib.sha256(payload).hexdigest()

    sha256, path = test_blob_store.store_bytes(payload)
    assert sha256 == expected_hash
    assert path.exists()

    # Read back
    read_data = test_blob_store.read_bytes(sha256)
    assert read_data == payload

    # Exists check
    assert test_blob_store.exists(sha256) is True
    assert test_blob_store.verify_blob(sha256) is True


def test_deduplication(test_blob_store: BlobStore):
    data = b"Duplicate content test"
    h1, p1 = test_blob_store.store_bytes(data)
    h2, p2 = test_blob_store.store_bytes(data)

    assert h1 == h2
    assert p1 == p2


def test_store_file(test_blob_store: BlobStore, tmp_path: Path):
    source_file = tmp_path / "sample.txt"
    source_file.write_bytes(b"File payload content")
    expected_hash = hashlib.sha256(b"File payload content").hexdigest()

    sha256, path = test_blob_store.store_file(source_file)
    assert sha256 == expected_hash
    assert test_blob_store.read_bytes(sha256) == b"File payload content"


def test_path_confinement(test_blob_store: BlobStore):
    # Attempt directory traversal via malicious hash
    with pytest.raises(BlobStoreError):
        test_blob_store._resolve_blob_path("../escaped_file")


def test_integrity_verification_failure(test_blob_store: BlobStore):
    data = b"Tamper test"
    sha256, path = test_blob_store.store_bytes(data)

    # Tamper with the file
    path.write_bytes(b"Tampered corrupted bytes!")

    assert test_blob_store.verify_blob(sha256) is False
    with pytest.raises(BlobIntegrityError):
        test_blob_store.read_bytes(sha256)


def test_invalid_hash_raises_error(test_blob_store: BlobStore):
    """Short or empty content hashes raise BlobStoreError."""
    with pytest.raises(BlobStoreError, match="Invalid content hash"):
        test_blob_store._resolve_blob_path("ab")
    with pytest.raises(BlobStoreError):
        test_blob_store._resolve_blob_path("")


def test_store_file_missing_source_raises(test_blob_store: BlobStore, tmp_path: Path):
    """store_file raises FileNotFoundError if source does not exist."""
    with pytest.raises(FileNotFoundError, match="Source file not found"):
        test_blob_store.store_file(tmp_path / "does_not_exist.txt")


def test_store_file_idempotent(test_blob_store: BlobStore, tmp_path: Path):
    """store_file reuses existing blob if already stored."""
    source = tmp_path / "idempotent.txt"
    source.write_text("idempotent file content")
    sha1, p1 = test_blob_store.store_file(source)
    sha2, p2 = test_blob_store.store_file(source)
    assert sha1 == sha2
    assert p1 == p2


def test_read_bytes_missing_blob_raises(test_blob_store: BlobStore):
    """read_bytes raises FileNotFoundError for missing blob."""
    with pytest.raises(FileNotFoundError, match="Blob not found"):
        test_blob_store.read_bytes("b" * 64)


def test_verify_blob_returns_false_for_missing(test_blob_store: BlobStore):
    """verify_blob returns False for nonexistent blob."""
    assert test_blob_store.verify_blob("c" * 64) is False


def test_list_all_hashes_empty(tmp_path: Path):
    """list_all_hashes returns empty set on empty blob store."""
    empty_store = BlobStore(tmp_path / "empty_blobs")
    assert empty_store.list_all_hashes() == set()
