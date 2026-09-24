"""Text chunking, local embedding generation, vector persistence, and similarity search."""

import datetime
import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import struct
from functools import lru_cache
from typing import Any

from edward.models import generate_id

logger = logging.getLogger(__name__)
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


# ---------------------------------------------------------------------------
# 1. Structural Text Chunker
# ---------------------------------------------------------------------------


def chunk_markdown_text(
    text: str,
    target_chunk_chars: int = 800,
    min_chunk_chars: int = 150,
) -> list[dict[str, Any]]:
    """Split text into structural chunks based on Markdown headers and paragraphs.

    Returns a list of dicts with:
    - 'chunk_index': int
    - 'text': str
    - 'heading': str | None
    - 'locator': dict[str, Any]
    """
    text = text.strip()
    if not text:
        return []

    page_markers = list(re.finditer(r"(?m)^\[PDF page (\d+)\]\n", text))
    if page_markers:
        page_chunks: list[dict[str, Any]] = []
        for index, marker in enumerate(page_markers):
            end = page_markers[index + 1].start() if index + 1 < len(page_markers) else len(text)
            page_text = text[marker.end() : end].strip()
            for chunk in chunk_markdown_text(page_text, target_chunk_chars, min_chunk_chars):
                chunk["chunk_index"] = len(page_chunks)
                chunk["locator"]["chunk_index"] = len(page_chunks)
                chunk["locator"]["page"] = int(marker.group(1))
                page_chunks.append(chunk)
        return page_chunks

    lines = text.splitlines()
    sections: list[dict[str, Any]] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    header_re = re.compile(r"^(#{1,6})\s+(.*)$")

    for line in lines:
        m = header_re.match(line)
        if m:
            if current_lines:
                sections.append(
                    {
                        "heading": current_heading,
                        "text": "\n".join(current_lines).strip(),
                    }
                )
                current_lines = []
            current_heading = m.group(2).strip()
        else:
            current_lines.append(line)

    if current_lines:
        sections.append(
            {
                "heading": current_heading,
                "text": "\n".join(current_lines).strip(),
            }
        )

    # Further break down sections that exceed target_chunk_chars
    chunks: list[dict[str, Any]] = []
    chunk_idx = 0

    for sec in sections:
        sec_text = sec["text"]
        heading = sec["heading"]

        if not sec_text:
            continue

        paragraphs: list[str] = []
        for paragraph in sec_text.split("\n\n"):
            paragraph = paragraph.strip()
            while len(paragraph) > target_chunk_chars:
                cut = paragraph.rfind(" ", min_chunk_chars, target_chunk_chars + 1)
                if cut < 0:
                    cut = target_chunk_chars
                paragraphs.append(paragraph[:cut])
                paragraph = paragraph[cut:].lstrip()
            if paragraph:
                paragraphs.append(paragraph)
        current_chunk_parts: list[str] = []
        current_len = 0

        for p in paragraphs:
            p_str = p.strip()
            if not p_str:
                continue

            if current_len + len(p_str) > target_chunk_chars and current_len >= min_chunk_chars:
                chunk_body = "\n\n".join(current_chunk_parts).strip()
                chunks.append(
                    {
                        "chunk_index": chunk_idx,
                        "text": chunk_body,
                        "heading": heading,
                        "locator": {
                            "heading": heading,
                            "chunk_index": chunk_idx,
                            "char_count": len(chunk_body),
                        },
                    }
                )
                chunk_idx += 1
                current_chunk_parts = [p_str]
                current_len = len(p_str)
            else:
                current_chunk_parts.append(p_str)
                current_len += len(p_str) + 2

        if current_chunk_parts:
            chunk_body = "\n\n".join(current_chunk_parts).strip()
            chunks.append(
                {
                    "chunk_index": chunk_idx,
                    "text": chunk_body,
                    "heading": heading,
                    "locator": {
                        "heading": heading,
                        "chunk_index": chunk_idx,
                        "char_count": len(chunk_body),
                    },
                }
            )
            chunk_idx += 1

    return chunks


def get_configured_embedding_model() -> str:
    """Resolve the active local embedding model."""
    configured = os.environ.get("EDWARD_EMBEDDING_MODEL", "").strip()
    if configured and configured != "default":
        return configured
    return DEFAULT_EMBEDDING_MODEL


def store_resource_chunks(
    conn: sqlite3.Connection,
    resource_id: str,
    resource_content_id: str,
    text: str,
) -> list[str]:
    """Chunk resource content, maintain search projections, and persist to resource_chunks."""
    # 1. Invalidate and remove existing chunks, their embeddings, and FTS projections for this resource
    old_chunk_rows = conn.execute(
        "SELECT id FROM resource_chunks WHERE resource_id = ?;",
        (resource_id,),
    ).fetchall()
    if old_chunk_rows:
        old_ids = [r["id"] for r in old_chunk_rows]
        ph = ",".join("?" * len(old_ids))
        conn.execute(
            f"DELETE FROM embeddings WHERE object_type = 'resource_chunk' AND object_id IN ({ph});",
            old_ids,
        )
        conn.execute(
            f"DELETE FROM search_documents WHERE object_type = 'chunk' AND object_id IN ({ph});",
            old_ids,
        )
        conn.execute(
            f"DELETE FROM resource_chunks WHERE id IN ({ph});",
            old_ids,
        )

    # 2. Query resource title for search projection
    res_row = conn.execute("SELECT title FROM resources WHERE id = ?;", (resource_id,)).fetchone()
    res_title = res_row["title"] if res_row and res_row["title"] else ""

    chunks = chunk_markdown_text(text)
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()
    chunk_ids: list[str] = []

    for c in chunks:
        c_id = generate_id("chk")
        approx_tokens = max(1, len(c["text"]) // 4)
        conn.execute(
            """
            INSERT INTO resource_chunks (
                id, resource_content_id, resource_id, chunk_index,
                text, locator_json, token_count, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                c_id,
                resource_content_id,
                resource_id,
                c["chunk_index"],
                c["text"],
                json.dumps(c["locator"]),
                approx_tokens,
                now_iso,
            ),
        )
        # Synchronously index in FTS search_documents
        conn.execute(
            """
            INSERT INTO search_documents (object_type, object_id, title, body, labels, entities)
            VALUES ('chunk', ?, ?, ?, '', '');
            """,
            (c_id, res_title, c["text"]),
        )
        chunk_ids.append(c_id)

    return chunk_ids


# ---------------------------------------------------------------------------
# 2. Vector Serializer & Embedder
# ---------------------------------------------------------------------------


def serialize_vector(vec: list[float]) -> bytes:
    """Serialize a list of floats into standard IEEE 754 float32 bytes."""
    return struct.pack(f"{len(vec)}f", *vec)


def deserialize_vector(blob: bytes) -> list[float]:
    """Deserialize IEEE 754 float32 bytes into a list of floats."""
    count = len(blob) // 4
    return list(struct.unpack(f"{count}f", blob))


def deterministic_embedding(text: str, dimensions: int = 64) -> list[float]:
    """Produce a deterministic, unit-normalized float vector from text for offline/test environments."""
    text_clean = text.lower().strip()
    words = re.findall(r"\w+", text_clean)

    vec = [0.0] * dimensions
    if not words:
        # Unit vector along first dimension
        vec[0] = 1.0
        return vec

    for word in words:
        h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
        idx = h % dimensions
        sign = 1.0 if (h >> 8) % 2 == 0 else -1.0
        vec[idx] += sign

    # Unit normalization
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 1e-9:
        vec = [v / norm for v in vec]
    else:
        vec[0] = 1.0

    return vec


@lru_cache(maxsize=4)
def _load_local_embedding_model(model: str):
    """Load a FastEmbed model once and cache its files in Edward's data directory."""
    from fastembed import TextEmbedding

    from edward.db import get_default_data_dir

    cache_dir = get_default_data_dir() / "models" / "embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return TextEmbedding(model_name=model, cache_dir=str(cache_dir))


def generate_embedding(
    text: str, model: str | None = None, *, is_query: bool = False
) -> tuple[list[float], str]:
    """Generate a vector in-process with FastEmbed, or use the explicit test fallback."""
    resolved_model = model or get_configured_embedding_model()
    if resolved_model == "default":
        resolved_model = get_configured_embedding_model()
    if resolved_model == "deterministic-v1":
        return deterministic_embedding(text), resolved_model

    embedder = _load_local_embedding_model(resolved_model)
    # BGE v1.5 is trained for these retrieval prefixes; other selected models get raw text.
    if resolved_model == DEFAULT_EMBEDDING_MODEL:
        prefix = "query: " if is_query else "passage: "
        text = f"{prefix}{text}"
    vector = next(embedder.embed([text]))
    return [float(value) for value in vector], resolved_model


# ---------------------------------------------------------------------------
# 3. Vector Storage & Search
# ---------------------------------------------------------------------------


def compute_embedding_input_hash(
    conn: sqlite3.Connection,
    resource_id: str | None = None,
    capture_id: str | None = None,
) -> str | None:
    """Compute aggregate input hash covering resource content and active findings, or capture content."""
    if resource_id:
        content_row = conn.execute(
            "SELECT clean_text, summary, content_hash FROM resource_contents WHERE resource_id = ? ORDER BY created_at DESC LIMIT 1;",
            (resource_id,),
        ).fetchone()
        content_hash = ""
        if content_row:
            embedded_text = (content_row["clean_text"] or content_row["summary"] or "").strip()
            content_hash = hashlib.sha256(embedded_text.encode("utf-8")).hexdigest()

        f_rows = conn.execute(
            "SELECT id, statement FROM findings WHERE resource_id = ? AND is_deleted = 0 AND review_state != 'superseded' ORDER BY id ASC;",
            (resource_id,),
        ).fetchall()

        if not content_hash and not f_rows:
            return None

        hasher = hashlib.sha256()
        hasher.update(b"res:")
        hasher.update(content_hash.encode("utf-8"))
        for f in f_rows:
            hasher.update(b":fin:")
            hasher.update(f["id"].encode("utf-8"))
            hasher.update(b":")
            hasher.update((f["statement"] or "").strip().encode("utf-8"))
        return hasher.hexdigest()

    elif capture_id:
        cap_row = conn.execute(
            "SELECT raw_content, user_note FROM captures WHERE id = ?;",
            (capture_id,),
        ).fetchone()
        if not cap_row:
            return None
        cap_text = (
            (cap_row["user_note"] or "") + ("\n\n" + (cap_row["raw_content"] or ""))
        ).strip()
        if not cap_text:
            return None
        return hashlib.sha256(cap_text.encode("utf-8")).hexdigest()

    return None


def store_embedding(
    conn: sqlite3.Connection,
    object_type: str,
    object_id: str,
    text: str,
    model: str | None = None,
) -> str:
    """Compute and store vector embedding for an object with input hash deduplication."""
    target_model = model if (model and model != "default") else get_configured_embedding_model()
    input_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    # Reuse the exact model and content row before running local inference.
    existing = conn.execute(
        """
        SELECT id, input_hash, model FROM embeddings
        WHERE object_type = ? AND object_id = ? AND model = ?;
        """,
        (object_type, object_id, target_model),
    ).fetchone()

    if existing and existing["input_hash"] == input_hash:
        return existing["id"]

    vec, actual_model = generate_embedding(text, model=target_model)

    blob = serialize_vector(vec)
    actual_dims = len(vec)
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()

    # Reuse a row if an explicitly selected fallback resolves to the same model.
    existing_actual = existing if (existing and existing["model"] == actual_model) else None
    if not existing_actual:
        existing_actual = conn.execute(
            """
            SELECT id, input_hash, model FROM embeddings
            WHERE object_type = ? AND object_id = ? AND model = ?;
            """,
            (object_type, object_id, actual_model),
        ).fetchone()

    if existing_actual:
        emb_id = existing_actual["id"]
        conn.execute(
            """
            UPDATE embeddings
            SET dimensions = ?, embedding_blob = ?, input_hash = ?, created_at = ?
            WHERE id = ?;
            """,
            (actual_dims, blob, input_hash, now_iso, emb_id),
        )
    else:
        emb_id = generate_id("emb")
        conn.execute(
            """
            INSERT INTO embeddings (
                id, object_type, object_id, model, dimensions,
                embedding_blob, input_hash, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (emb_id, object_type, object_id, actual_model, actual_dims, blob, input_hash, now_iso),
        )

    return emb_id


def compute_cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    if len(vec_a) != len(vec_b) or not vec_a:
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return dot / (norm_a * norm_b)


def search_vector(
    conn: sqlite3.Connection,
    query_text: str,
    limit: int = 20,
    model: str | None = None,
    object_type: str | None = None,
) -> list[dict[str, Any]]:
    """Search vector embeddings by cosine similarity.

    Uses the configured local model and dynamic sqlite-vec acceleration when available.
    Returns list of dicts with: 'object_type', 'object_id', 'similarity', 'embedding_id'.
    """
    target_model = model if (model and model != "default") else get_configured_embedding_model()

    available = conn.execute(
        "SELECT 1 FROM embeddings WHERE model = ? LIMIT 1;", (target_model,)
    ).fetchone()
    if not available:
        return []

    query_vec, _ = generate_embedding(query_text, model=target_model, is_query=True)
    query_blob = serialize_vector(query_vec)

    # 1. Try sqlite-vec if loaded
    from edward.db import HAS_SQLITE_VEC

    if HAS_SQLITE_VEC:
        try:
            sql = """
                SELECT id, object_type, object_id,
                       (1.0 - vec_distance_cosine(embedding_blob, ?)) AS similarity
                FROM embeddings
                WHERE model = ?
            """
            params: list[Any] = [query_blob, target_model]
            if object_type:
                sql += " AND object_type = ?"
                params.append(object_type)
            sql += " ORDER BY similarity DESC LIMIT ?;"
            params.append(limit)

            rows = conn.execute(sql, params).fetchall()
            return [
                {
                    "object_type": r["object_type"],
                    "object_id": r["object_id"],
                    "similarity": float(r["similarity"]),
                    "embedding_id": r["id"],
                }
                for r in rows
            ]
        except Exception as e:
            logger.debug("sqlite-vec query failed; falling back to Python similarity: %s", e)

    # 2. Pure Python fallback
    sql = "SELECT id, object_type, object_id, embedding_blob FROM embeddings WHERE model = ?"
    params = [target_model]
    if object_type:
        sql += " AND object_type = ?"
        params.append(object_type)

    rows = conn.execute(sql, params).fetchall()
    scored: list[dict[str, Any]] = []

    for r in rows:
        target_vec = deserialize_vector(r["embedding_blob"])
        sim = compute_cosine_similarity(query_vec, target_vec)
        scored.append(
            {
                "object_type": r["object_type"],
                "object_id": r["object_id"],
                "similarity": sim,
                "embedding_id": r["id"],
            }
        )

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:limit]


def embed_resource(
    conn: sqlite3.Connection,
    resource_id: str,
    model: str | None = None,
) -> list[str]:
    """Chunk and embed a resource and its chunks."""
    target_model = model if (model and model != "default") else get_configured_embedding_model()
    content_row = conn.execute(
        """
        SELECT id, clean_text, summary FROM resource_contents
        WHERE resource_id = ? ORDER BY created_at DESC LIMIT 1;
        """,
        (resource_id,),
    ).fetchone()

    if not content_row:
        return []

    content_id = content_row["id"]
    text = (content_row["clean_text"] or content_row["summary"] or "").strip()
    if not text:
        return []

    # 1. Store chunks
    chunk_ids = store_resource_chunks(conn, resource_id, content_id, text)

    # 2. Embed each chunk
    emb_ids: list[str] = []
    for c_id in chunk_ids:
        chunk_row = conn.execute(
            "SELECT text FROM resource_chunks WHERE id = ?;", (c_id,)
        ).fetchone()
        if chunk_row and chunk_row["text"]:
            emb_id = store_embedding(
                conn=conn,
                object_type="resource_chunk",
                object_id=c_id,
                text=chunk_row["text"],
                model=target_model,
            )
            emb_ids.append(emb_id)

    # 3. Embed top-level resource summary / text (bounded)
    resource_doc = text[:1500]
    res_emb_id = store_embedding(
        conn=conn,
        object_type="resource",
        object_id=resource_id,
        text=resource_doc,
        model=target_model,
    )
    emb_ids.append(res_emb_id)

    # 4. Embed findings associated with this resource
    finding_rows = conn.execute(
        "SELECT id, statement FROM findings WHERE resource_id = ? AND is_deleted = 0 AND review_state != 'superseded';",
        (resource_id,),
    ).fetchall()
    for f_row in finding_rows:
        if f_row["statement"]:
            f_emb_id = store_embedding(
                conn=conn,
                object_type="finding",
                object_id=f_row["id"],
                text=f_row["statement"],
                model=target_model,
            )
            emb_ids.append(f_emb_id)

    return emb_ids


def persist_precomputed_embeddings(
    conn: sqlite3.Connection,
    resource_id: str,
    content_id: str | None,
    chunk_embeddings: list[dict[str, Any]],
    resource_embedding: dict[str, Any] | None,
    finding_embeddings: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Persist precomputed chunk embeddings, resource embedding, and finding embeddings into SQLite using static SQL."""
    # 1. Invalidate and remove existing chunks, their embeddings, and FTS projections for this resource
    old_chunk_rows = conn.execute(
        "SELECT id FROM resource_chunks WHERE resource_id = ?;",
        (resource_id,),
    ).fetchall()
    if old_chunk_rows:
        old_ids = [r["id"] for r in old_chunk_rows]
        ph = ",".join("?" * len(old_ids))
        conn.execute(
            f"DELETE FROM embeddings WHERE object_type = 'resource_chunk' AND object_id IN ({ph});",
            old_ids,
        )
        conn.execute(
            f"DELETE FROM search_documents WHERE object_type = 'chunk' AND object_id IN ({ph});",
            old_ids,
        )
        conn.execute(
            f"DELETE FROM resource_chunks WHERE id IN ({ph});",
            old_ids,
        )

    # 2. Query resource title for search projection
    res_row = conn.execute("SELECT title FROM resources WHERE id = ?;", (resource_id,)).fetchone()
    res_title = res_row["title"] if res_row and res_row["title"] else ""

    now_iso = datetime.datetime.now(datetime.UTC).isoformat()
    emb_ids: list[str] = []

    # 3. Persist chunks and their embeddings (if content_id is available)
    if content_id:
        for item in chunk_embeddings:
            c = item["chunk"]
            vec = item["vector"]
            actual_m = item.get("model", "deterministic-v1")
            chunk_id = generate_id("chk")
            approx_tokens = max(1, len(c["text"]) // 4)

            conn.execute(
                """
                INSERT INTO resource_chunks (
                    id, resource_content_id, resource_id, chunk_index,
                    text, locator_json, token_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    chunk_id,
                    content_id,
                    resource_id,
                    c["chunk_index"],
                    c["text"],
                    json.dumps(c.get("locator", {})),
                    approx_tokens,
                    now_iso,
                ),
            )
            conn.execute(
                """
                INSERT INTO search_documents (object_type, object_id, title, body, labels, entities)
                VALUES ('chunk', ?, ?, ?, '', '');
                """,
                (chunk_id, res_title, c["text"]),
            )

            c_blob = serialize_vector(vec)
            c_hash = hashlib.sha256(c["text"].encode("utf-8")).hexdigest()
            c_emb_id = generate_id("emb")
            conn.execute(
                """
                INSERT INTO embeddings (
                    id, object_type, object_id, model, dimensions,
                    embedding_blob, input_hash, created_at
                ) VALUES (?, 'resource_chunk', ?, ?, ?, ?, ?, ?);
                """,
                (c_emb_id, chunk_id, actual_m, len(vec), c_blob, c_hash, now_iso),
            )
            emb_ids.append(c_emb_id)

    # 4. Persist top-level resource embedding
    if resource_embedding:
        res_vec = resource_embedding["vector"]
        res_m = resource_embedding.get("model", "deterministic-v1")
        res_text = resource_embedding.get("text", "")
        res_hash = hashlib.sha256(res_text.encode("utf-8")).hexdigest()
        res_blob = serialize_vector(res_vec)

        existing = conn.execute(
            "SELECT id FROM embeddings WHERE object_type = 'resource' AND object_id = ? AND model = ?;",
            (resource_id, res_m),
        ).fetchone()

        if existing:
            res_emb_id = existing["id"]
            conn.execute(
                """
                UPDATE embeddings
                SET dimensions = ?, embedding_blob = ?, input_hash = ?, created_at = ?
                WHERE id = ?;
                """,
                (len(res_vec), res_blob, res_hash, now_iso, res_emb_id),
            )
        else:
            res_emb_id = generate_id("emb")
            conn.execute(
                """
                INSERT INTO embeddings (
                    id, object_type, object_id, model, dimensions,
                    embedding_blob, input_hash, created_at
                ) VALUES (?, 'resource', ?, ?, ?, ?, ?, ?);
                """,
                (res_emb_id, resource_id, res_m, len(res_vec), res_blob, res_hash, now_iso),
            )
        emb_ids.append(res_emb_id)

    # 5. Persist precomputed finding embeddings using exact SQL
    for f_item in finding_embeddings or []:
        f_id = f_item["finding_id"]
        f_vec = f_item["vector"]
        f_m = f_item.get("model", "deterministic-v1")
        f_text = f_item.get("text", "")
        f_hash = hashlib.sha256(f_text.encode("utf-8")).hexdigest()
        f_blob = serialize_vector(f_vec)

        existing_f = conn.execute(
            "SELECT id FROM embeddings WHERE object_type = 'finding' AND object_id = ? AND model = ?;",
            (f_id, f_m),
        ).fetchone()

        if existing_f:
            f_emb_id = existing_f["id"]
            conn.execute(
                """
                UPDATE embeddings
                SET dimensions = ?, embedding_blob = ?, input_hash = ?, created_at = ?
                WHERE id = ?;
                """,
                (len(f_vec), f_blob, f_hash, now_iso, f_emb_id),
            )
        else:
            f_emb_id = generate_id("emb")
            conn.execute(
                """
                INSERT INTO embeddings (
                    id, object_type, object_id, model, dimensions,
                    embedding_blob, input_hash, created_at
                ) VALUES (?, 'finding', ?, ?, ?, ?, ?, ?);
                """,
                (f_emb_id, f_id, f_m, len(f_vec), f_blob, f_hash, now_iso),
            )
        emb_ids.append(f_emb_id)

    return emb_ids
