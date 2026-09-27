"""Smoke tests for Edward CLI commands and JSON output guarantees."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from edward.cli import app


@pytest.fixture(autouse=True)
def configure_test_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point EDWARD_DATA_DIR to a temporary directory for all CLI tests."""
    data_dir = tmp_path / "edward_cli_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("EDWARD_DATA_DIR", str(data_dir))


def test_cli_add_and_show_json(cli_runner: CliRunner):
    # Add an item
    result = cli_runner.invoke(
        app,
        [
            "add",
            "--url",
            "https://example.com/test",
            "--note",
            "CLI smoke note",
            "--intent",
            "essay-seed",
            "--json",
        ],
    )
    assert result.exit_code == 0
    # Must be valid JSON on stdout
    data = json.loads(result.stdout)
    assert data["status"] == "created"
    assert "capture_id" in data
    capture_id = data["capture_id"]
    resource_id = data["resource_id"]

    # Show capture
    show_res = cli_runner.invoke(app, ["show", capture_id, "--json"])
    assert show_res.exit_code == 0
    show_data = json.loads(show_res.stdout)
    assert show_data["id"] == capture_id

    # Show resource
    show_res_r = cli_runner.invoke(app, ["show", resource_id, "--json"])
    assert show_res_r.exit_code == 0
    show_data_r = json.loads(show_res_r.stdout)
    assert show_data_r["id"] == resource_id


def test_cli_search_json(cli_runner: CliRunner):
    cli_runner.invoke(
        app,
        ["add", "--text", "Unique searchable keywords in CLI", "--json"],
    )

    search_res = cli_runner.invoke(
        app,
        ["search", "searchable keywords", "--json"],
    )
    assert search_res.exit_code == 0
    data = json.loads(search_res.stdout)
    assert data["count"] >= 1
    assert any(
        "searchable" in (r["title"] or "").lower() or "searchable" in (r["snippet"] or "").lower()
        for r in data["results"]
    )


def test_cli_annotate_and_remove_intent(cli_runner: CliRunner):
    add_res = cli_runner.invoke(
        app,
        ["add", "--url", "https://example.com/annotate", "--intent", "to-read", "--json"],
    )
    res_id = json.loads(add_res.stdout)["resource_id"]

    # Annotate with a label and note
    ann_res = cli_runner.invoke(
        app,
        ["annotate", res_id, "--label", "article", "--note", "Added annotation", "--json"],
    )
    assert ann_res.exit_code == 0

    # Remove intent
    rem_res = cli_runner.invoke(
        app,
        ["remove-intent", res_id, "to-read", "--json"],
    )
    assert rem_res.exit_code == 0


def test_cli_missing_object_error_code(cli_runner: CliRunner):
    result = cli_runner.invoke(app, ["show", "nonexistent_id", "--json"])
    assert result.exit_code == 1


def test_cli_purge_requires_confirm(cli_runner: CliRunner):
    add_res = cli_runner.invoke(
        app,
        ["add", "--text", "To be purged", "--json"],
    )
    cap_id = json.loads(add_res.stdout)["capture_id"]

    # Attempt purge without --confirm
    fail_res = cli_runner.invoke(app, ["purge", cap_id])
    assert fail_res.exit_code == 2

    # Purge with --confirm
    ok_res = cli_runner.invoke(app, ["purge", cap_id, "--confirm", "--json"])
    assert ok_res.exit_code == 0


def test_cli_backup_and_doctor_json(cli_runner: CliRunner, tmp_path: Path):
    backup_dir = tmp_path / "cli_backups"
    backup_res = cli_runner.invoke(
        app,
        ["backup", "--dest", str(backup_dir), "--json"],
    )
    assert backup_res.exit_code == 0
    backup_data = json.loads(backup_res.stdout)
    assert "backup_path" in backup_data

    # Doctor verification of system
    doc_res = cli_runner.invoke(app, ["doctor", "--json"])
    assert doc_res.exit_code == 0
    doc_data = json.loads(doc_res.stdout)
    assert doc_data["healthy"] is True

    # Doctor verification of specific backup
    doc_b_res = cli_runner.invoke(app, ["doctor", "--backup", str(backup_dir), "--json"])
    assert doc_b_res.exit_code == 0


def test_cli_add_file_preserves_blob_and_creates_attachment(cli_runner: CliRunner, tmp_path: Path):
    sample_file = tmp_path / "paper.pdf"
    sample_bytes = b"%PDF-1.4 sample binary pdf content"
    sample_file.write_bytes(sample_bytes)

    add_res = cli_runner.invoke(
        app,
        ["add", "--file", str(sample_file), "--note", "Important study", "--json"],
    )
    assert add_res.exit_code == 0
    data = json.loads(add_res.stdout)
    assert data["status"] == "created"
    assert data["resource_id"] is not None

    # Verify show displays resource with attachment
    show_res = cli_runner.invoke(app, ["show", data["resource_id"], "--json"])
    assert show_res.exit_code == 0
    show_data = json.loads(show_res.stdout)
    assert show_data["id"] == data["resource_id"]
    assert show_data["title"] == "paper.pdf"


def test_cli_add_interactive(cli_runner: CliRunner):
    # Simulate user entering URL, text, note, intent interactively
    user_input = "https://example.com/interactive\nSome interactive text\nInteractive note\ninteractive-intent\n"
    res = cli_runner.invoke(
        app,
        ["add", "--interactive"],
        input=user_input,
    )
    assert res.exit_code == 0
    assert "Captured successfully!" in res.stdout
    assert "https://example.com/interactive" in res.stdout


def test_cli_annotate_appends_and_shows_annotations(cli_runner: CliRunner):
    add_res = cli_runner.invoke(
        app,
        ["add", "--text", "Base note for annotations test", "--json"],
    )
    cap_id = json.loads(add_res.stdout)["capture_id"]

    # Append note 1
    ann_res1 = cli_runner.invoke(
        app,
        ["annotate", cap_id, "--note", "Note one", "--json"],
    )
    assert ann_res1.exit_code == 0

    # Append note 2
    ann_res2 = cli_runner.invoke(
        app,
        ["annotate", cap_id, "--note", "Note two", "--json"],
    )
    assert ann_res2.exit_code == 0

    # Show should display both annotations
    show_res = cli_runner.invoke(app, ["show", cap_id, "--json"])
    assert show_res.exit_code == 0
    show_data = json.loads(show_res.stdout)
    assert len(show_data["annotations"]) == 2
    assert show_data["annotations"][0]["content"] == "Note one"
    assert show_data["annotations"][1]["content"] == "Note two"


def test_cli_purge_prunes_unreferenced_blobs(cli_runner: CliRunner, tmp_path: Path):
    sample_file = tmp_path / "temp_to_purge.txt"
    sample_file.write_text("Unique text blob for purge test")

    add_res = cli_runner.invoke(
        app,
        ["add", "--file", str(sample_file), "--json"],
    )
    res_id = json.loads(add_res.stdout)["resource_id"]

    # Purge resource with confirm
    purge_res = cli_runner.invoke(app, ["purge", res_id, "--confirm", "--json"])
    assert purge_res.exit_code == 0
    purge_data = json.loads(purge_res.stdout)
    assert purge_data["status"] == "purged"
    assert len(purge_data.get("pruned_blobs", [])) >= 1


def test_cli_doctor_prune_blobs(cli_runner: CliRunner):
    doc_res = cli_runner.invoke(app, ["doctor", "--prune-blobs", "--json"])
    assert doc_res.exit_code == 0
    doc_data = json.loads(doc_res.stdout)
    assert "pruned_blobs" in doc_data


def test_cli_export_jsonl(cli_runner: CliRunner, tmp_path: Path):
    # Capture items with metadata
    add1 = cli_runner.invoke(
        app,
        [
            "add",
            "--url",
            "https://example.com/export1",
            "--note",
            "Exportable note 1",
            "--intent",
            "to-read",
            "--json",
        ],
    )
    assert add1.exit_code == 0
    cap1_id = json.loads(add1.stdout)["capture_id"]
    res1_id = json.loads(add1.stdout)["resource_id"]

    # Annotate item 1
    cli_runner.invoke(
        app,
        ["annotate", res1_id, "--label", "article", "--note", "Important reading", "--json"],
    )

    # Capture item 2
    add2 = cli_runner.invoke(
        app,
        ["add", "--text", "Completely unrelated text for export testing", "--json"],
    )
    assert add2.exit_code == 0
    cap2_id = json.loads(add2.stdout)["capture_id"]

    # 1. Default format is complete jsonl corpus streamed to stdout (no truncation)
    res = cli_runner.invoke(app, ["export"])
    assert res.exit_code == 0
    lines = [json.loads(line) for line in res.stdout.strip().split("\n") if line.strip()]
    object_types = {item["object_type"] for item in lines}
    # Must include complete relational model
    assert "resource" in object_types
    assert "capture" in object_types
    assert "capture_resource" in object_types
    assert "annotation" in object_types
    assert "intent" in object_types
    assert "audit_event" in object_types

    # 2. JSON summary mode (--format jsonl --json)
    res_json = cli_runner.invoke(app, ["export", "--format", "jsonl", "--json"])
    assert res_json.exit_code == 0
    data = json.loads(res_json.stdout)
    assert data["status"] == "exported"
    assert data["format"] == "jsonl"
    assert data["count"] == len(lines)
    assert len(data["items"]) == data["count"]

    # 3. Filtered export (--query) only exports matched objects and their metadata
    res_filtered = cli_runner.invoke(app, ["export", "--query", "unrelated", "--json"])
    assert res_filtered.exit_code == 0
    filtered_data = json.loads(res_filtered.stdout)
    filtered_items = filtered_data["items"]
    # Should include cap2_id
    assert any(
        it["data"].get("id") == cap2_id for it in filtered_items if it["object_type"] == "capture"
    )
    # Must NOT include cap1_id or res1_id or their annotations
    for it in filtered_items:
        if it["object_type"] in ("resource", "capture"):
            assert it["data"].get("id") not in (cap1_id, res1_id)
        elif it["object_type"] == "annotation":
            assert it["data"].get("object_id") not in (cap1_id, res1_id)

    # 4. Pagination (--limit and --offset)
    res_page = cli_runner.invoke(app, ["export", "--limit", "2", "--offset", "1", "--json"])
    assert res_page.exit_code == 0
    page_data = json.loads(res_page.stdout)
    assert page_data["count"] == 2
    assert page_data["items"] == data["items"][1:3]

    # 5. Export to file
    out_file = tmp_path / "corpus.jsonl"
    res_out = cli_runner.invoke(
        app, ["export", "--format", "jsonl", "--out", str(out_file), "--json"]
    )
    assert res_out.exit_code == 0
    assert out_file.exists()
    file_lines = [
        line for line in out_file.read_text(encoding="utf-8").strip().split("\n") if line.strip()
    ]
    assert len(file_lines) == data["count"]


def test_cli_export_packet(cli_runner: CliRunner, tmp_path: Path):
    # Add an item to search
    cli_runner.invoke(
        app,
        ["add", "--text", "Unique quantum superposition research text", "--json"],
    )

    # 1. Bounded packet export to stdout
    res = cli_runner.invoke(app, ["export", "--packet", "--query", "quantum"])
    assert res.exit_code == 0
    packet = json.loads(res.stdout)
    assert packet["type"] == "evidence-packet"
    assert packet["schema_version"] == "1"
    assert packet["query"] == "quantum"
    assert len(packet["items"]) >= 1

    # 2. Bounded packet export to file
    out_file = tmp_path / "packet.json"
    res_out = cli_runner.invoke(
        app,
        ["export", "--packet", "--query", "quantum", "--out", str(out_file), "--json"],
    )
    assert res_out.exit_code == 0
    summary = json.loads(res_out.stdout)
    assert summary["status"] == "exported"
    assert summary["format"] == "packet"
    assert out_file.exists()
    saved_packet = json.loads(out_file.read_text(encoding="utf-8"))
    assert saved_packet["type"] == "evidence-packet"
    assert len(saved_packet["items"]) >= 1


def test_cli_chunk_show_and_annotate(cli_runner: CliRunner):
    from edward.cli import get_services
    from edward.services.search import rebuild_search_index

    db, _ = get_services()
    with db.transaction() as conn:
        conn.execute(
            """INSERT INTO resources (id, identity_key, canonical_url, title, review_state,
                is_deleted, created_at, updated_at)
                VALUES ('res_chunk_test', 'url:chunk-test', 'https://example.com/chunk-test', 'Chunk Test Title',
                'unreviewed', 0, '2026-01-01', '2026-01-01');"""
        )
        conn.execute(
            """INSERT INTO resource_contents (id, resource_id, content_hash, clean_text,
                char_count, extractor, extractor_version, created_at)
                VALUES ('rc_chunk_test', 'res_chunk_test', 'hash', 'Full text content', 17, 'text', '1', '2026-01-01');"""
        )
        conn.execute(
            """INSERT INTO resource_chunks (id, resource_content_id, resource_id,
                chunk_index, text, locator_json, token_count, created_at)
                VALUES ('chk_state_machine', 'rc_chunk_test', 'res_chunk_test', 0,
                'Hierarchical deterministic state machines require strict verification.',
                '{"chunk_index": 0}', 8, '2026-01-01');"""
        )
        rebuild_search_index(conn)

    # Search for chunk
    search_res = cli_runner.invoke(
        app,
        ["search", "deterministic state machines", "--json"],
    )
    assert search_res.exit_code == 0
    search_data = json.loads(search_res.stdout)
    assert search_data["count"] >= 1
    chunk_item = next(
        (item for item in search_data["results"] if item.get("object_type") == "chunk"),
        None,
    )
    assert chunk_item is not None
    assert chunk_item["id"] == "chk_state_machine"
    assert chunk_item["resource_id"] == "res_chunk_test"
    assert chunk_item["canonical_url"] == "https://example.com/chunk-test"

    # 1. Show chunk via JSON
    show_json_res = cli_runner.invoke(app, ["show", "chk_state_machine", "--json"])
    assert show_json_res.exit_code == 0
    show_data = json.loads(show_json_res.stdout)
    assert show_data["id"] == "chk_state_machine"
    assert show_data["resource_id"] == "res_chunk_test"
    assert "text" in show_data
    assert "resource" in show_data
    assert show_data["resource"]["title"] == "Chunk Test Title"

    # 2. Show chunk via plain text
    show_plain_res = cli_runner.invoke(app, ["show", "chk_state_machine"])
    assert show_plain_res.exit_code == 0
    assert "Chunk:" in show_plain_res.stdout
    assert "chk_state_machine" in show_plain_res.stdout
    assert "Text:" in show_plain_res.stdout

    # 3. Annotate chunk
    ann_res = cli_runner.invoke(
        app,
        ["annotate", "chk_state_machine", "--note", "Important chunk note", "--json"],
    )
    assert ann_res.exit_code == 0
    ann_data = json.loads(ann_res.stdout)
    assert ann_data["status"] == "annotated"

    # 4. Verify annotation is visible on show
    show_again = cli_runner.invoke(app, ["show", "chk_state_machine", "--json"])
    assert show_again.exit_code == 0
    show_again_data = json.loads(show_again.stdout)
    assert len(show_again_data["annotations"]) == 1
    assert show_again_data["annotations"][0]["content"] == "Important chunk note"
