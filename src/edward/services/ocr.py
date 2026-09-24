"""Local OCR for image attachments, so screenshots become searchable text.

Bookmarked X posts frequently carry their substance in an image — benchmark
charts, model cards, code, security scans. That text is invisible to every
other part of Edward: the post body is one line and the picture is opaque
bytes. Until this module existed, the only thing that ever happened to those
images was that a classifier was handed raw JPEG bytes and returned a 400.

This mirrors ``pdf.py``: the same job stage shape, the same "extract once, keep
the text under the parent" rule, the same refusal to index anything that yields
no text. The one deliberate difference is that OCR output is prefixed with a
provenance marker, so a reader can always tell machine-read image text from the
post the author actually typed.
"""

from __future__ import annotations

import datetime
import sqlite3
from pathlib import Path

from edward.models import generate_id, make_job_key
from edward.services.subprocess_runner import is_tool_available, run_tool

OCR_STAGE = "attachment-ocr"

# A screenshot of a chat or a code block can be long; a photo yields nothing.
# These bounds exist to stop a pathological image from stalling a worker, not
# to judge quality.
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_OCR_CHARS = 20_000
OCR_TIMEOUT_SECONDS = 60.0

# Below this, the "text" is JPEG artifacts and stray glyphs, not content.
MIN_USABLE_CHARS = 12

# Machine-read text is never allowed to masquerade as the author's words.
OCR_PROVENANCE_PREFIX = "[Image text, read by OCR]"

IMAGE_MIME_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/tiff",
    "image/bmp",
    "image/heic",
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".heic")


class OcrExtractionError(ValueError):
    """Raised when an image cannot yield usable text within configured limits."""


def is_image_attachment(file_name: str, mime_type: str) -> bool:
    """Return whether an attachment should be sent to the local OCR extractor."""
    if mime_type and mime_type.lower() in IMAGE_MIME_TYPES:
        return True
    lowered = file_name.lower()
    return lowered.endswith(IMAGE_EXTENSIONS)


def enqueue_ocr_extraction(
    conn: sqlite3.Connection,
    capture_id: str,
    attachment_id: str,
    *,
    now_iso: str | None = None,
) -> None:
    """Queue local OCR for one persisted image attachment."""
    now_iso = now_iso or datetime.datetime.now(datetime.UTC).isoformat()
    conn.execute(
        """
        INSERT OR IGNORE INTO processing_jobs (
            id, job_key, capture_id, stage, status, available_at,
            attempts, created_at, updated_at
        ) VALUES (?, ?, ?, ?, 'pending', ?, 0, ?, ?);
        """,
        (
            generate_id("job"),
            make_job_key(OCR_STAGE, attachment_id),
            capture_id,
            OCR_STAGE,
            now_iso,
            now_iso,
            now_iso,
        ),
    )


def ocr_image_bytes(image_bytes: bytes, *, language: str = "eng") -> str:
    """Read text out of an image with the local tesseract binary.

    Raises :class:`OcrExtractionError` when tesseract is unavailable, the image
    is oversized, or nothing legible comes back. Callers treat "no text" as a
    normal outcome, not a failure to retry.
    """
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise OcrExtractionError(
            f"Image exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MiB OCR limit"
        )
    if not is_tool_available("tesseract"):
        raise OcrExtractionError("tesseract is not installed; image OCR is unavailable")

    # tesseract reads a path, not a stream, and the blob store hands us bytes.
    # Write beside the blob we were given rather than into the system temp dir.
    import tempfile

    handle = tempfile.NamedTemporaryFile(suffix=".img", delete=False)
    try:
        handle.write(image_bytes)
        handle.close()
        result = run_tool(
            ["tesseract", handle.name, "stdout", "-l", language],
            timeout=OCR_TIMEOUT_SECONDS,
        )
    finally:
        Path(handle.name).unlink(missing_ok=True)

    if result.exit_code != 0:
        detail = (result.stderr or "").strip()[:200]
        raise OcrExtractionError(f"tesseract exited with code {result.exit_code}: {detail}")

    text = _normalize_ocr_text(result.stdout or "")
    if len(text) < MIN_USABLE_CHARS:
        raise OcrExtractionError("No usable text was found in the image")
    return text[:MAX_OCR_CHARS]


def _normalize_ocr_text(raw: str) -> str:
    """Tidy OCR output: drop control noise, collapse vertical whitespace."""
    lines: list[str] = []
    blank_run = 0
    for line in raw.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        cleaned = "".join(ch for ch in line if ch == "\t" or ch.isprintable()).strip()
        if not cleaned:
            blank_run += 1
            if blank_run > 1 or not lines:
                continue
            lines.append("")
            continue
        blank_run = 0
        lines.append(cleaned)
    return "\n".join(lines).strip()


def format_ocr_passage(file_name: str, text: str, *, tweet_url: str | None = None) -> str:
    """Wrap OCR output as a locatable, provenance-marked passage."""
    header = f"{OCR_PROVENANCE_PREFIX} {file_name}"
    if tweet_url:
        header = f"{header} (from {tweet_url})"
    return f"{header}\n{text}"
