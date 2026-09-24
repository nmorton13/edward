"""Image attachments: media URLs belong to the post, and their text is OCR'd.

Two invariants govern this module:

1. An image URL inside a post's entities is not a source. It must never become
   its own resource.
2. Text read out of an image must never be presented as the author's words.
"""

import pytest

from edward.services.ocr import (
    MIN_USABLE_CHARS,
    OCR_PROVENANCE_PREFIX,
    OcrExtractionError,
    _normalize_ocr_text,
    enqueue_ocr_extraction,
    format_ocr_passage,
    is_image_attachment,
)
from edward.services.source_adapters import (
    _tweet_media_urls,
    is_twitter_media_url,
)

# --------------------------------------------------------------------------
# Media URL recognition
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://pbs.twimg.com/media/HSwAPAfbMAAa1fq.jpg",
        "https://pbs.twimg.com/amplify_video_thumb/2102352673911578624/img/VjMhTnGQZ1M78fhV.jpg",
        "https://pbs.twimg.com/ext_tw_video_thumb/123/pu/img/abc.jpg",
        "https://pbs.twimg.com/tweet_video_thumb/abc.jpg",
        "https://video.twimg.com/ext_tw_video/123/pu/vid/720x720/abc.mp4",
    ],
)
def test_twitter_media_urls_are_recognized(url):
    assert is_twitter_media_url(url) is True


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/article-about-data-centers",
        "https://arxiv.org/abs/2401.00001",
        "https://x.com/someone/status/12345",
        "https://pbs.twimg.com/profile_images/123/avatar.jpg",
        "https://news.ycombinator.com/item?id=1",
    ],
)
def test_real_pages_are_not_media(url):
    """A linked article must survive; only the image asset is demoted."""
    assert is_twitter_media_url(url) is False


def test_media_urls_are_found_inside_entities():
    """birdclaw hides the media URL in entities_json.urls — the original bug."""
    entities = {
        "urls": [
            {
                "url": "https://pbs.twimg.com/media/HSwAPAfbMAAa1fq.jpg",
                "expandedUrl": "https://pbs.twimg.com/media/HSwAPAfbMAAa1fq.jpg",
            }
        ]
    }
    media = [{"url": "https://pbs.twimg.com/media/HSwAPAfbMAAa1fq.jpg", "type": "image"}]

    found = _tweet_media_urls(media, entities, "built a malware scanner https://t.co/x")

    assert found == ["https://pbs.twimg.com/media/HSwAPAfbMAAa1fq.jpg"]


def test_video_variants_are_collected():
    media = [
        {
            "url": "https://pbs.twimg.com/amplify_video_thumb/1/img/t.jpg",
            "type": "video",
            "variants": [{"url": "https://video.twimg.com/ext_tw_video/1/pu/vid/720x720/a.mp4"}],
        }
    ]

    found = _tweet_media_urls(media, {}, "")

    assert "https://video.twimg.com/ext_tw_video/1/pu/vid/720x720/a.mp4" in found


def test_a_post_with_no_media_yields_nothing():
    assert _tweet_media_urls([], {}, "just text, no links") == []


# --------------------------------------------------------------------------
# Image detection
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,mime",
    [
        ("photo.jpg", ""),
        ("photo.jpeg", ""),
        ("scan.png", ""),
        ("shot.webp", ""),
        ("image.heic", ""),
        ("whatever", "image/png"),
        ("unknown", "image/jpeg"),
    ],
)
def test_images_are_detected(name, mime):
    assert is_image_attachment(name, mime) is True


@pytest.mark.parametrize(
    "name,mime",
    [("paper.pdf", "application/pdf"), ("notes.txt", "text/plain"), ("clip.mp4", "video/mp4")],
)
def test_non_images_are_not_detected(name, mime):
    assert is_image_attachment(name, mime) is False


# --------------------------------------------------------------------------
# OCR text normalization
# --------------------------------------------------------------------------


def test_ocr_text_is_tidied_but_not_altered():
    raw = "PHOTON 2.4 — SPEECH RECOGNITION\r\n\r\n\r\n115% realtime on eight x86 CPU cores\t178 MB\r\n"

    text = _normalize_ocr_text(raw)

    assert "115% realtime on eight x86 CPU cores" in text
    assert "\r" not in text
    assert "\n\n\n" not in text
    # Numbers and punctuation are the payload — they must survive intact.
    assert "%" in text and "—" in text


def test_control_characters_are_stripped():
    assert "\x00" not in _normalize_ocr_text("clean\x00text\x07")


def test_ocr_of_an_image_without_text_is_reported_not_invented(tmp_path):
    """A photo of a dog yields nothing — that is an outcome, not a failure."""
    from edward.services import ocr as ocr_module

    class _Result:
        exit_code = 0
        stdout = "   \n\n  \t \n"
        stderr = ""

    original_available = ocr_module.is_tool_available
    original_run = ocr_module.run_tool
    ocr_module.is_tool_available = lambda name: True
    ocr_module.run_tool = lambda *a, **k: _Result()
    try:
        with pytest.raises(OcrExtractionError, match="No usable text"):
            ocr_module.ocr_image_bytes(b"\xff\xd8fakejpeg")
    finally:
        ocr_module.is_tool_available = original_available
        ocr_module.run_tool = original_run


def test_missing_tesseract_is_a_clear_error_not_a_crash():
    from edward.services import ocr as ocr_module

    original_available = ocr_module.is_tool_available
    ocr_module.is_tool_available = lambda name: False
    try:
        with pytest.raises(OcrExtractionError, match="tesseract is not installed"):
            ocr_module.ocr_image_bytes(b"\xff\xd8fakejpeg")
    finally:
        ocr_module.is_tool_available = original_available


def test_oversized_image_is_refused_before_ocr_runs():
    from edward.services import ocr as ocr_module

    with pytest.raises(OcrExtractionError, match="exceeds"):
        ocr_module.ocr_image_bytes(b"x" * (ocr_module.MAX_IMAGE_BYTES + 1))


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


def test_ocr_passage_is_marked_as_machine_read():
    """Image text must be distinguishable from what the author typed."""
    passage = format_ocr_passage("bench.png", "Kev-4B 79.0%")

    assert passage.startswith(OCR_PROVENANCE_PREFIX)
    assert "bench.png" in passage
    assert "Kev-4B 79.0%" in passage


def test_ocr_passage_can_name_its_post():
    passage = format_ocr_passage("bench.png", "text", tweet_url="https://x.com/i/status/1")

    assert "https://x.com/i/status/1" in passage


def test_min_usable_chars_is_not_zero():
    """Guards against indexing JPEG artifact noise as content."""
    assert MIN_USABLE_CHARS >= 8


# --------------------------------------------------------------------------
# Job queuing
# --------------------------------------------------------------------------


def _seed_capture(conn, capture_id: str = "cap-1") -> str:
    conn.execute(
        """
        INSERT OR IGNORE INTO captures (
            id, origin_namespace, origin_id, collection_channel, collector,
            acquisition_method, retrieved_at, raw_content, created_at, updated_at
        ) VALUES (?, 'x', 'tweet-1', 'birdclaw', 'edward-birdclaw-adapter',
                  'birdclaw-sqlite', '2026-01-01', 'post text', '2026-01-01', '2026-01-01');
        """,
        (capture_id,),
    )
    return capture_id


def test_ocr_job_is_queued_once(test_db):
    """Re-running ingest must not queue duplicate OCR work."""
    with test_db.transaction() as conn:
        _seed_capture(conn, "cap-1")
        enqueue_ocr_extraction(conn, "cap-1", "att-1")
        enqueue_ocr_extraction(conn, "cap-1", "att-1")

    with test_db.connection() as conn:
        rows = conn.execute(
            "SELECT job_key, stage, status FROM processing_jobs WHERE stage = 'attachment-ocr';"
        ).fetchall()

    assert len(rows) == 1
    assert rows[0]["stage"] == "attachment-ocr"
    assert rows[0]["status"] == "pending"


def test_ocr_job_key_names_the_attachment(test_db):
    with test_db.transaction() as conn:
        _seed_capture(conn, "cap-1")
        enqueue_ocr_extraction(conn, "cap-1", "att-42")

    with test_db.connection() as conn:
        key = conn.execute(
            "SELECT job_key FROM processing_jobs WHERE stage = 'attachment-ocr';"
        ).fetchone()["job_key"]

    assert "att-42" in key
