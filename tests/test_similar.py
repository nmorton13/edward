"""Tests for edward similar command, item-vector similarity search, and MCP parity."""

import json

from typer.testing import CliRunner

from edward.cli import app
from edward.db import Database
from edward.mcp_server import create_mcp_server
from edward.services.embed import find_similar, store_embedding
from edward.services.projects import create_project


def _setup_similar_corpus(conn):
    """Seed a test corpus with captures, resources, and findings that have embeddings."""
    now_iso = "2026-01-01T12:00:00Z"

    # Captures
    conn.execute(
        """
        INSERT INTO captures (id, origin_namespace, collection_channel, collector, acquisition_method, retrieved_at, raw_content, user_note, is_deleted, created_at, updated_at)
        VALUES ('cap_ai', 'web', 'browser', 'cli', 'manual', ?, 'Transformers and local attention mechanisms for deep learning.', 'AI Notes', 0, ?, ?);
        """,
        (now_iso, now_iso, now_iso),
    )
    conn.execute(
        """
        INSERT INTO captures (id, origin_namespace, collection_channel, collector, acquisition_method, retrieved_at, raw_content, user_note, is_deleted, created_at, updated_at)
        VALUES ('cap_garden', 'web', 'browser', 'cli', 'manual', ?, 'Organic soil composition and tomato gardening tips.', 'Gardening Notes', 0, ?, ?);
        """,
        (now_iso, now_iso, now_iso),
    )
    conn.execute(
        """
        INSERT INTO captures (id, origin_namespace, collection_channel, collector, acquisition_method, retrieved_at, raw_content, user_note, is_deleted, created_at, updated_at)
        VALUES ('cap_deleted', 'web', 'browser', 'cli', 'manual', ?, 'Deleted capture text about neural networks.', 'Deleted Notes', 1, ?, ?);
        """,
        (now_iso, now_iso, now_iso),
    )
    conn.execute(
        """
        INSERT INTO captures (id, origin_namespace, collection_channel, collector, acquisition_method, retrieved_at, raw_content, user_note, is_deleted, created_at, updated_at)
        VALUES ('cap_no_emb', 'web', 'browser', 'cli', 'manual', ?, 'Unembedded text.', 'No Emb', 0, ?, ?);
        """,
        (now_iso, now_iso, now_iso),
    )

    # Resources
    conn.execute(
        """
        INSERT INTO resources (id, identity_key, canonical_url, title, is_deleted, created_at, updated_at)
        VALUES ('res_llm', 'url:llm', 'https://example.com/llm', 'Large Language Models Guide', 0, ?, ?);
        """,
        (now_iso, now_iso),
    )
    conn.execute(
        """
        INSERT INTO resources (id, identity_key, canonical_url, title, is_deleted, created_at, updated_at)
        VALUES ('res_compost', 'url:compost', 'https://example.com/compost', 'Composting for Beginners', 0, ?, ?);
        """,
        (now_iso, now_iso),
    )
    conn.execute(
        """
        INSERT INTO resources (id, identity_key, canonical_url, title, is_deleted, created_at, updated_at)
        VALUES ('res_deleted', 'url:del', 'https://example.com/del', 'Deleted Deep Learning', 1, ?, ?);
        """,
        (now_iso, now_iso),
    )

    # Link res_llm to cap_ai
    conn.execute(
        "INSERT INTO capture_resources (capture_id, resource_id, created_at) VALUES ('cap_ai', 'res_llm', ?);",
        (now_iso,),
    )
    # Link res_compost to cap_garden
    conn.execute(
        "INSERT INTO capture_resources (capture_id, resource_id, created_at) VALUES ('cap_garden', 'res_compost', ?);",
        (now_iso,),
    )

    # Findings
    conn.execute(
        """
        INSERT INTO findings (id, resource_id, statement, is_deleted, created_at, updated_at)
        VALUES ('fin_attn', 'res_llm', 'Multi-head attention scales quadratically with sequence length.', 0, ?, ?);
        """,
        (now_iso, now_iso),
    )
    conn.execute(
        """
        INSERT INTO findings (id, resource_id, statement, is_deleted, created_at, updated_at)
        VALUES ('fin_soil', 'res_compost', 'Nitrogen-rich compost accelerates plant development.', 0, ?, ?);
        """,
        (now_iso, now_iso),
    )
    conn.execute(
        """
        INSERT INTO findings (id, resource_id, statement, is_deleted, created_at, updated_at)
        VALUES ('fin_deleted', 'res_llm', 'Deleted attention finding statement.', 1, ?, ?);
        """,
        (now_iso, now_iso),
    )

    # Store embeddings
    store_embedding(
        conn, "capture", "cap_ai", "Transformers and local attention mechanisms for deep learning."
    )
    store_embedding(
        conn, "capture", "cap_garden", "Organic soil composition and tomato gardening tips."
    )
    store_embedding(conn, "capture", "cap_deleted", "Deleted capture text about neural networks.")

    store_embedding(
        conn,
        "resource",
        "res_llm",
        "Large Language Models Guide on deep learning and transformers.",
    )
    store_embedding(
        conn, "resource", "res_compost", "Composting for Beginners on soil and gardening."
    )
    store_embedding(conn, "resource", "res_deleted", "Deleted Deep Learning guide.")

    store_embedding(
        conn,
        "finding",
        "fin_attn",
        "Multi-head attention scales quadratically with sequence length.",
    )
    store_embedding(
        conn, "finding", "fin_soil", "Nitrogen-rich compost accelerates plant development."
    )
    store_embedding(conn, "finding", "fin_deleted", "Deleted attention finding statement.")


def test_similar_for_capture(test_db: Database):
    """find_similar returns relevant items for a capture, excluding self and deleted records."""
    with test_db.transaction() as conn:
        _setup_similar_corpus(conn)

    with test_db.connection() as conn:
        hits = find_similar(conn, "cap_ai", limit=10)
        assert len(hits) >= 2
        hit_ids = [h["object_id"] for h in hits]

        # cap_ai must not be in results (self-exclusion)
        assert "cap_ai" not in hit_ids
        # Deleted items must not be in results
        assert "cap_deleted" not in hit_ids
        assert "res_deleted" not in hit_ids
        assert "fin_deleted" not in hit_ids

        # AI-related items should rank high
        assert "res_llm" in hit_ids

        # Verify hit schema
        for h in hits:
            assert "object_type" in h
            assert "object_id" in h
            assert "title" in h
            assert "similarity" in h
            assert "capture_id" in h

        # Check resource hit has linked capture_id
        res_hit = next(h for h in hits if h["object_id"] == "res_llm")
        assert res_hit["capture_id"] == "cap_ai"
        assert res_hit["title"] == "Large Language Models Guide"


def test_similar_for_resource(test_db: Database):
    """find_similar works when given a resource ID."""
    with test_db.transaction() as conn:
        _setup_similar_corpus(conn)

    with test_db.connection() as conn:
        hits = find_similar(conn, "res_llm", limit=10)
        hit_ids = [h["object_id"] for h in hits]
        assert "res_llm" not in hit_ids
        assert "cap_ai" in hit_ids


def test_similar_for_finding(test_db: Database):
    """find_similar works when given a finding ID."""
    with test_db.transaction() as conn:
        _setup_similar_corpus(conn)

    with test_db.connection() as conn:
        hits = find_similar(conn, "fin_attn", limit=10)
        hit_ids = [h["object_id"] for h in hits]
        assert "fin_attn" not in hit_ids
        assert "res_llm" in hit_ids or "cap_ai" in hit_ids


def test_similar_type_filtering(test_db: Database):
    """find_similar respects the object_type filter."""
    with test_db.transaction() as conn:
        _setup_similar_corpus(conn)

    with test_db.connection() as conn:
        # Only resources
        res_hits = find_similar(conn, "cap_ai", object_type="resource")
        assert all(h["object_type"] == "resource" for h in res_hits)

        # Only captures
        cap_hits = find_similar(conn, "res_llm", object_type="capture")
        assert all(h["object_type"] == "capture" for h in cap_hits)

        # Only findings
        fin_hits = find_similar(conn, "cap_ai", object_type="finding")
        assert all(h["object_type"] == "finding" for h in fin_hits)


def test_similar_exclude_project(test_db: Database):
    """find_similar excludes items belonging to the specified project."""
    with test_db.transaction() as conn:
        _setup_similar_corpus(conn)
        prj = create_project(conn, title="AI Project", slug="ai-project")
        conn.execute(
            """
            INSERT INTO project_objects (id, project_id, object_type, object_id, membership_status, created_at)
            VALUES ('po_1', ?, 'resource', 'res_llm', 'accepted', '2026-01-01');
            """,
            (prj.id,),
        )

    with test_db.connection() as conn:
        hits = find_similar(conn, "cap_ai", exclude_project="ai-project")
        hit_ids = [h["object_id"] for h in hits]
        # res_llm belongs to the project and must be excluded
        assert "res_llm" not in hit_ids


def test_cli_similar_json_contract(test_db: Database, monkeypatch):
    """CLI edward similar with --json produces strictly valid JSON on stdout."""
    monkeypatch.setenv("EDWARD_DB_PATH", str(test_db.db_path))

    with test_db.transaction() as conn:
        _setup_similar_corpus(conn)

    runner = CliRunner()
    result = runner.invoke(app, ["similar", "cap_ai", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert isinstance(data, list)
    assert len(data) >= 1
    hit = data[0]
    assert "object_type" in hit
    assert "object_id" in hit
    assert "title" in hit
    assert "similarity" in hit
    assert "capture_id" in hit


def test_cli_similar_human_output(test_db: Database, monkeypatch):
    """CLI edward similar without --json outputs human-readable text."""
    monkeypatch.setenv("EDWARD_DB_PATH", str(test_db.db_path))

    with test_db.transaction() as conn:
        _setup_similar_corpus(conn)

    runner = CliRunner()
    result = runner.invoke(app, ["similar", "cap_ai"])
    assert result.exit_code == 0
    assert "Similar items to cap_ai" in result.stdout


def test_cli_similar_unknown_id_exit_code_1(test_db: Database, monkeypatch):
    """Unknown ID returns exit code 1 with stderr message."""
    monkeypatch.setenv("EDWARD_DB_PATH", str(test_db.db_path))

    runner = CliRunner()
    result = runner.invoke(app, ["similar", "unknown_123", "--json"])
    assert result.exit_code == 1
    assert "not found" in result.stderr.lower()


def test_cli_similar_deleted_id_exit_code_1(test_db: Database, monkeypatch):
    """Deleted item ID returns exit code 1."""
    monkeypatch.setenv("EDWARD_DB_PATH", str(test_db.db_path))

    with test_db.transaction() as conn:
        _setup_similar_corpus(conn)

    runner = CliRunner()
    result = runner.invoke(app, ["similar", "cap_deleted", "--json"])
    assert result.exit_code == 1
    assert "deleted" in result.stderr.lower()


def test_cli_similar_unembedded_item_exit_code_1(test_db: Database, monkeypatch):
    """Item without embedding returns exit code 1 with clear message."""
    monkeypatch.setenv("EDWARD_DB_PATH", str(test_db.db_path))

    with test_db.transaction() as conn:
        _setup_similar_corpus(conn)

    runner = CliRunner()
    result = runner.invoke(app, ["similar", "cap_no_emb", "--json"])
    assert result.exit_code == 1
    assert "no embedding" in result.stderr.lower()


def test_cli_similar_bad_options_exit_code_2(test_db: Database, monkeypatch):
    """Bad options (invalid type, non-positive limit, unknown project) return exit code 2."""
    monkeypatch.setenv("EDWARD_DB_PATH", str(test_db.db_path))

    with test_db.transaction() as conn:
        _setup_similar_corpus(conn)

    runner = CliRunner()

    # Invalid type
    res_type = runner.invoke(app, ["similar", "cap_ai", "--type", "invalid_type", "--json"])
    assert res_type.exit_code == 2

    # Unknown project
    res_proj = runner.invoke(
        app, ["similar", "cap_ai", "--exclude-project", "nonexistent-prj", "--json"]
    )
    assert res_proj.exit_code == 2

    # Bad limit
    res_lim = runner.invoke(app, ["similar", "cap_ai", "--limit", "0", "--json"])
    assert res_lim.exit_code == 2


def test_mcp_similar_tool(test_db: Database):
    """MCP server edward_similar tool returns hits with matching schema."""
    with test_db.transaction() as conn:
        _setup_similar_corpus(conn)

    server = create_mcp_server(db=test_db)
    tool_fn = server._tool_manager._tools.get("edward_similar")
    assert tool_fn is not None

    hits = tool_fn.fn(object_id="cap_ai", limit=5)
    assert isinstance(hits, list)
    assert len(hits) >= 1
    assert "cap_ai" not in [h["object_id"] for h in hits]
    assert hits[0]["object_type"] in ("capture", "resource", "finding")
