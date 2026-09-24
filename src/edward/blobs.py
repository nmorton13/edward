"""Content-addressed managed blob store for Edward."""

import hashlib
import os
import shutil
import tempfile
from pathlib import Path


class BlobStoreError(Exception):
    """Base error for blob store operations."""

    pass


class BlobPathConfinementError(BlobStoreError):
    """Raised when an operation attempts to escape the blob store root."""

    pass


class BlobIntegrityError(BlobStoreError):
    """Raised when blob content does not match expected SHA-256 hash."""

    pass


class BlobStore:
    """Manages immutable, content-addressed files under <data-dir>/blobs/."""

    def __init__(self, root_dir: str | Path):
        self.root = Path(root_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve_blob_path(self, content_hash: str) -> Path:
        """Resolve the path for a given content hash, ensuring confinement."""
        if not content_hash or len(content_hash) < 4:
            raise BlobStoreError(f"Invalid content hash: {content_hash}")

        prefix = content_hash[:2]
        dest_dir = (self.root / prefix).resolve()
        dest_path = (dest_dir / content_hash).resolve()

        # Path confinement assertion
        try:
            dest_path.relative_to(self.root)
        except ValueError:
            raise BlobPathConfinementError(f"Path escapes blob store: {dest_path}") from None

        return dest_path

    def store_bytes(self, data: bytes) -> tuple[str, Path]:
        """Store bytes into the blob store atomically. Returns (sha256_hash, path)."""
        sha256 = hashlib.sha256(data).hexdigest()
        dest_path = self._resolve_blob_path(sha256)

        if dest_path.exists():
            return sha256, dest_path

        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write via temporary file in the same directory/filesystem
        with tempfile.NamedTemporaryFile(dir=dest_path.parent, delete=False) as tf:
            tf.write(data)
            temp_path = Path(tf.name)

        # Verify hash of written data before renaming
        written_hash = hashlib.sha256(temp_path.read_bytes()).hexdigest()
        if written_hash != sha256:
            temp_path.unlink(missing_ok=True)
            raise BlobIntegrityError(f"Calculated hash {written_hash} does not match {sha256}")

        temp_path.replace(dest_path)
        return sha256, dest_path

    def store_file(self, source_path: str | Path) -> tuple[str, Path]:
        """Copy a file into the blob store atomically. Returns (sha256_hash, path)."""
        source = Path(source_path).resolve()
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        hasher = hashlib.sha256()
        with open(source, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        sha256 = hasher.hexdigest()

        dest_path = self._resolve_blob_path(sha256)
        if dest_path.exists():
            return sha256, dest_path

        dest_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(dir=dest_path.parent, delete=False) as tf:
            with open(source, "rb") as sf:
                shutil.copyfileobj(sf, tf)
            temp_path = Path(tf.name)

        # Verify
        verify_hasher = hashlib.sha256()
        with open(temp_path, "rb") as vf:
            while chunk := vf.read(65536):
                verify_hasher.update(chunk)
        if verify_hasher.hexdigest() != sha256:
            temp_path.unlink(missing_ok=True)
            raise BlobIntegrityError("File content hash verification failed")

        temp_path.replace(dest_path)
        return sha256, dest_path

    def get_path(self, content_hash: str) -> Path | None:
        """Return path to existing blob, or None if it doesn't exist."""
        path = self._resolve_blob_path(content_hash)
        return path if path.exists() else None

    def read_bytes(self, content_hash: str) -> bytes:
        """Read bytes for a given content hash."""
        path = self._resolve_blob_path(content_hash)
        if not path.exists():
            raise FileNotFoundError(f"Blob not found: {content_hash}")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != content_hash:
            raise BlobIntegrityError(f"Corrupted blob data for hash {content_hash}")
        return data

    def exists(self, content_hash: str) -> bool:
        """Check if a blob exists."""
        return self._resolve_blob_path(content_hash).exists()

    def list_all_hashes(self) -> set[str]:
        """List all SHA-256 hashes currently stored in the blob store."""
        hashes = set()
        for _root, _, files in os.walk(self.root):
            for file in files:
                if len(file) == 64:
                    hashes.add(file)
        return hashes

    def verify_blob(self, content_hash: str) -> bool:
        """Verify the integrity of a stored blob by recalculating its SHA-256."""
        path = self.get_path(content_hash)
        if not path:
            return False
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest() == content_hash
