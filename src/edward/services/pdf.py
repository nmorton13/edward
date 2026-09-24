"""Local PDF text extraction for searchable attachment captures."""

from __future__ import annotations

import datetime
import sqlite3
from io import BytesIO

from pypdf import PdfReader

from edward.models import generate_id, make_job_key

MAX_PDF_BYTES = 100 * 1024 * 1024
MAX_PDF_PAGES = 2_000
MAX_EXTRACTED_CHARS = 10_000_000
PDF_STAGE = "attachment-extract"


class PdfExtractionError(ValueError):
    """Raised when a PDF cannot yield searchable text within configured limits."""


def is_pdf_attachment(file_name: str, mime_type: str) -> bool:
    """Return whether an attachment should be sent to the local PDF extractor."""
    return mime_type.lower() == "application/pdf" or file_name.lower().endswith(".pdf")


def enqueue_pdf_extraction(
    conn: sqlite3.Connection,
    capture_id: str,
    attachment_id: str,
    *,
    now_iso: str | None = None,
) -> None:
    """Queue local text extraction for one persisted PDF attachment."""
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
            make_job_key(PDF_STAGE, attachment_id),
            capture_id,
            PDF_STAGE,
            now_iso,
            now_iso,
            now_iso,
        ),
    )


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract selectable PDF text with one-based page markers; OCR is not performed."""
    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise PdfExtractionError(
            f"PDF exceeds the {MAX_PDF_BYTES // (1024 * 1024)} MiB extraction limit"
        )

    try:
        reader = PdfReader(BytesIO(pdf_bytes), strict=False)
        if reader.is_encrypted:
            raise PdfExtractionError("Password-protected PDFs cannot be indexed")
        if len(reader.pages) > MAX_PDF_PAGES:
            raise PdfExtractionError(f"PDF exceeds the {MAX_PDF_PAGES}-page extraction limit")

        page_texts: list[str] = []
        total_chars = 0
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            if not page_text:
                continue
            total_chars += len(page_text)
            if total_chars > MAX_EXTRACTED_CHARS:
                raise PdfExtractionError(
                    f"PDF exceeds the {MAX_EXTRACTED_CHARS:,}-character extraction limit"
                )
            page_texts.append(f"[PDF page {page_number}]\n{page_text}")
    except PdfExtractionError:
        raise
    except Exception as exc:
        raise PdfExtractionError(f"Could not read PDF: {exc}") from exc

    if not page_texts:
        raise PdfExtractionError(
            "No selectable text was found; scanned PDFs need OCR before Edward can index them"
        )
    return "\n\n".join(page_texts)
