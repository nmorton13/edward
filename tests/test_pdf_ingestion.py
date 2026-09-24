"""Tests for searchable PDF attachment ingestion."""

import hashlib
from io import BytesIO

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from edward.models import CaptureInput
from edward.services.capture import capture_item
from edward.services.pdf import PdfExtractionError, extract_pdf_text
from edward.services.processor import process_pending_jobs
from edward.services.search import search_lexical


def _pdf_with_text(text: str) -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
    )
    stream = DecodedStreamObject()
    escaped_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream.set_data(f"BT /F1 12 Tf 72 720 Td ({escaped_text}) Tj ET".encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    pdf = BytesIO()
    writer.write(pdf)
    return pdf.getvalue()


def test_extract_pdf_text_marks_page_numbers():
    assert extract_pdf_text(_pdf_with_text("distinctive searchable phrase")) == (
        "[PDF page 1]\ndistinctive searchable phrase"
    )


def test_extract_pdf_text_reports_image_only_pdf():
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    pdf = BytesIO()
    writer.write(pdf)

    with pytest.raises(PdfExtractionError, match="scanned PDFs need OCR"):
        extract_pdf_text(pdf.getvalue())


def test_pdf_capture_text_is_lexically_searchable_and_embedded(test_db, test_blob_store):
    pdf_bytes = _pdf_with_text("xylophonic evidence supports the central claim")
    content_hash, _ = test_blob_store.store_bytes(pdf_bytes)
    with test_db.transaction() as conn:
        capture = capture_item(
            conn,
            CaptureInput(note="Paper for my draft"),
            attachment_info={
                "file_name": "research.pdf",
                "mime_type": "application/pdf",
                "content_hash": content_hash,
                "size_bytes": len(pdf_bytes),
            },
        )

    result = process_pending_jobs(test_db, test_blob_store, limit=10)
    assert result["failed"] == 0

    with test_db.connection() as conn:
        lexical = search_lexical(conn, "xylophonic")
        assert any(item.id == capture["capture_id"] for item in lexical.results)

        capture_text = conn.execute(
            "SELECT raw_content FROM captures WHERE id = ?;", (capture["capture_id"],)
        ).fetchone()["raw_content"]
        assert "[PDF page 1]" in capture_text
        assert "central claim" in capture_text

        attachment = conn.execute(
            "SELECT content_hash FROM attachments WHERE file_name = 'research.pdf';"
        ).fetchone()
        assert attachment["content_hash"] == content_hash

        embedding = conn.execute(
            "SELECT input_hash FROM embeddings WHERE object_type = 'capture' AND object_id = ?;",
            (capture["capture_id"],),
        ).fetchone()
        expected_text = f"Paper for my draft\n\n{capture_text}"[:1500]
        assert embedding["input_hash"] == hashlib.sha256(expected_text.encode()).hexdigest()
