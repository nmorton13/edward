"""Tests for text chunking, embeddings generation, vector storage, and similarity search."""

from unittest.mock import patch

from edward.db import Database
from edward.services.embed import (
    DEFAULT_EMBEDDING_MODEL,
    chunk_markdown_text,
    compute_cosine_similarity,
    deserialize_vector,
    deterministic_embedding,
    embed_resource,
    generate_embedding,
    get_configured_embedding_model,
    search_vector,
    serialize_vector,
    store_embedding,
)


def test_semantic_local_model_is_default(monkeypatch):
    """A fresh install selects the built-in semantic model without environment configuration."""
    monkeypatch.delenv("EDWARD_EMBEDDING_MODEL", raising=False)
    assert get_configured_embedding_model() == DEFAULT_EMBEDDING_MODEL


def test_fastembed_uses_bge_retrieval_prefixes(monkeypatch):
    """BGE receives distinct passage and query prefixes through the local backend."""
    monkeypatch.delenv("EDWARD_EMBEDDING_MODEL", raising=False)

    class FakeEmbedder:
        texts: list[str] = []

        def embed(self, texts):
            self.texts.extend(texts)
            yield [0.1, 0.2, 0.3]

    fake = FakeEmbedder()
    monkeypatch.setattr("edward.services.embed._load_local_embedding_model", lambda model: fake)

    passage, passage_model = generate_embedding("cut costs")
    query, query_model = generate_embedding("reduce spending", is_query=True)

    assert fake.texts == ["passage: cut costs", "query: reduce spending"]
    assert passage == query == [0.1, 0.2, 0.3]
    assert passage_model == query_model == DEFAULT_EMBEDDING_MODEL


def test_search_does_not_mix_old_model_vectors_with_default(test_db: Database, monkeypatch):
    """Legacy deterministic vectors are ignored until vectors for the default model exist."""
    monkeypatch.delenv("EDWARD_EMBEDDING_MODEL", raising=False)
    with test_db.transaction() as conn:
        conn.execute(
            """INSERT INTO resources (id, identity_key, canonical_url, title, review_state,
               is_deleted, created_at, updated_at)
               VALUES ('res_old_vec', 'url:old', 'https://example.com/old', 'Old',
               'unreviewed', 0, '2026-01-01', '2026-01-01');"""
        )
        store_embedding(
            conn, "resource", "res_old_vec", "reduce spending", model="deterministic-v1"
        )

    with (
        test_db.connection() as conn,
        patch(
            "edward.services.embed.generate_embedding", side_effect=AssertionError("must not embed")
        ),
    ):
        assert search_vector(conn, "cut costs") == []


def test_chunk_markdown_text_structure():
    """Markdown headers and paragraphs are split into structural chunks with locators."""
    md = """# Introduction
This is the first paragraph introducing the topic.

This is the second paragraph with more details.

## Benchmark Results
We measured the latency of local models on M-series chips.
- Model A: 45 ms
- Model B: 30 ms

### Concluding Thoughts
Local AI tools are becoming remarkably practical for everyday workflows.
"""
    chunks = chunk_markdown_text(md, target_chunk_chars=150, min_chunk_chars=50)
    assert len(chunks) >= 3

    headings = [c["heading"] for c in chunks]
    assert "Introduction" in headings
    assert "Benchmark Results" in headings
    assert "Concluding Thoughts" in headings

    for c in chunks:
        assert "chunk_index" in c
        assert "locator" in c
        assert c["locator"]["heading"] == c["heading"]


def test_pdf_chunks_keep_page_locators_and_bound_long_paragraphs():
    text = (
        "[PDF page 1]\n"
        + "Introduction to rights. " * 80
        + "\n\n[PDF page 2]\n"
        + "Routstr uses Cashu tokens for agents. " * 80
    )
    chunks = chunk_markdown_text(text)
    assert {chunk["locator"]["page"] for chunk in chunks} == {1, 2}
    assert all(len(chunk["text"]) <= 1000 for chunk in chunks)
    assert all("Routstr" not in chunk["text"] for chunk in chunks if chunk["locator"]["page"] == 1)
    assert any("Routstr" in chunk["text"] for chunk in chunks if chunk["locator"]["page"] == 2)


def test_vector_serialization_ieee754():
    """Float vectors serialize and deserialize cleanly to IEEE 754 float32."""
    orig = [0.123, -0.456, 0.789, 0.0]
    blob = serialize_vector(orig)
    assert len(blob) == len(orig) * 4  # 4 bytes per float32

    recovered = deserialize_vector(blob)
    assert len(recovered) == len(orig)
    for a, b in zip(orig, recovered, strict=True):
        assert abs(a - b) < 1e-6


def test_deterministic_embedding_normalized():
    """Deterministic embedding generates unit-length vectors with consistent dimensions."""
    vec = deterministic_embedding("Local AI models with llama.cpp", dimensions=64)
    assert len(vec) == 64

    # Check unit norm
    norm = sum(x * x for x in vec)
    assert abs(norm - 1.0) < 1e-4

    # Consistent across repeated calls
    vec2 = deterministic_embedding("Local AI models with llama.cpp", dimensions=64)
    assert vec == vec2


def test_cosine_similarity():
    """Cosine similarity calculates expected values for identical, orthogonal, and opposing vectors."""
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    v3 = [0.0, 1.0, 0.0]

    assert abs(compute_cosine_similarity(v1, v2) - 1.0) < 1e-6
    assert abs(compute_cosine_similarity(v1, v3) - 0.0) < 1e-6


def test_store_embedding_and_search_vector(test_db: Database):
    """Store embeddings in database and retrieve via vector cosine similarity."""
    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at)
            VALUES ('res_vec_1', 'url:v1', 'https://example.com/v1', 'Apple Silicon Benchmarks', 'unreviewed', 0, '2026-01-01', '2026-01-01'),
                   ('res_vec_2', 'url:v2', 'https://example.com/v2', 'Gardening and Composting', 'unreviewed', 0, '2026-01-01', '2026-01-01');
            """
        )

        store_embedding(
            conn=conn,
            object_type="resource",
            object_id="res_vec_1",
            text="High performance local LLM inference benchmarks on Apple Silicon M3 Max.",
            model="default",
        )

        store_embedding(
            conn=conn,
            object_type="resource",
            object_id="res_vec_2",
            text="How to build healthy organic soil through hot composting techniques.",
            model="default",
        )

    with test_db.connection() as conn:
        # Search query closely related to res_vec_1
        results = search_vector(conn, query_text="local models Apple Silicon inference", limit=5)
        assert len(results) >= 2
        # res_vec_1 should rank higher than gardening
        assert results[0]["object_id"] == "res_vec_1"
        assert results[0]["similarity"] > results[1]["similarity"]


def test_embedding_lifecycle_hash_deduplication(test_db: Database):
    """Unchanged content skips re-embedding; updated content refreshes embedding."""
    text = "Persistent test content."
    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at)
            VALUES ('res_dedup', 'url:dedup', 'https://example.com/dedup', 'Title', 'unreviewed', 0, '2026-01-01', '2026-01-01');
            """
        )
        emb_id_1 = store_embedding(conn, "resource", "res_dedup", text)

        # Call again with same text: should return identical emb_id without changing row
        emb_id_2 = store_embedding(conn, "resource", "res_dedup", text)
        assert emb_id_1 == emb_id_2

        # Change content: should update embedding
        emb_id_3 = store_embedding(conn, "resource", "res_dedup", "Completely different text.")
        assert emb_id_3 == emb_id_1  # Replaces existing record in-place


def test_embed_resource_end_to_end(test_db: Database):
    """embed_resource chunks the document and produces embeddings for chunks and parent."""
    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at)
            VALUES ('res_e2e', 'url:e2e', 'https://example.com/e2e', 'AI Overview', 'unreviewed', 0, '2026-01-01', '2026-01-01');
            """
        )
        content_text = """# Overview
This article covers neural network quantization techniques.

## Practical Quantization
Using 4-bit and 8-bit integer quantization reduces VRAM requirements significantly.
"""
        conn.execute(
            """
            INSERT INTO resource_contents (id, resource_id, content_hash, clean_text, extractor, extractor_version, char_count, created_at)
            VALUES ('rc_e2e', 'res_e2e', 'hash_e2e', ?, 'ext', '1.0', 100, '2026-01-01');
            """,
            (content_text,),
        )

        emb_ids = embed_resource(conn, "res_e2e")
        assert len(emb_ids) >= 2  # Chunks + resource

    with test_db.connection() as conn:
        chunks = conn.execute(
            "SELECT * FROM resource_chunks WHERE resource_id = 'res_e2e';"
        ).fetchall()
        assert len(chunks) >= 1
        embeddings = conn.execute(
            "SELECT * FROM embeddings WHERE object_id = 'res_e2e';"
        ).fetchall()
        assert len(embeddings) == 1
