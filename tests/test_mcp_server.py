"""Tests for the native Edward Model Context Protocol (MCP) server."""

import asyncio
import json

from typer.testing import CliRunner

from edward.blobs import BlobStore
from edward.cli import app
from edward.db import Database
from edward.mcp_server import create_mcp_server


def test_mcp_server_registration(test_db: Database, test_blob_store: BlobStore) -> None:
    """Verify that all 23 expected tools are registered on the MCPServer."""

    async def _test() -> None:
        server = create_mcp_server(db=test_db, blob_store=test_blob_store)
        expected_tools = {
            "edward_ask",
            "edward_search",
            "edward_show",
            "edward_export_packet",
            "edward_add",
            "edward_import_research",
            "edward_sync",
            "edward_annotate",
            "edward_remove_intent",
            "edward_accept_intent",
            "edward_list_intents",
            "edward_project_list",
            "edward_project_create",
            "edward_project_context",
            "edward_project_add_evidence",
            "edward_project_remove_evidence",
            "edward_project_add_note",
            "edward_project_propose_outline",
            "edward_project_accept_outline",
            "edward_project_delete",
            "edward_status",
            "edward_process",
            "edward_doctor",
        }
        tools = await server.list_tools()
        registered = {tool.name for tool in tools}
        assert registered == expected_tools

    asyncio.run(_test())


def test_mcp_status_and_doctor(test_db: Database, test_blob_store: BlobStore) -> None:
    """Verify operational tools: edward_status and edward_doctor."""

    async def _test() -> None:
        server = create_mcp_server(db=test_db, blob_store=test_blob_store)

        # Status
        status_res = await server.call_tool("edward_status", {})
        assert not status_res.is_error
        status_data = json.loads(status_res.content[0].text)
        assert "processing_queue" in status_data
        assert "corpus" in status_data

        # Doctor
        doctor_res = await server.call_tool("edward_doctor", {})
        assert not doctor_res.is_error
        doctor_data = json.loads(doctor_res.content[0].text)
        assert doctor_data.get("healthy") is True

    asyncio.run(_test())


def test_mcp_add_show_search_ask(test_db: Database, test_blob_store: BlobStore) -> None:
    """Verify capture, retrieval, search, ask, and bundle export tools."""

    async def _test() -> None:
        server = create_mcp_server(db=test_db, blob_store=test_blob_store)

        # Capture item
        add_res = await server.call_tool(
            "edward_add",
            {
                "url": "https://example.com/systems-thinking",
                "text": "Cybernetics and feedback loops govern complex adaptive systems and homeostasis.",
                "note": "Crucial primer on cybernetics",
                "intent": "essay-seed",
                "origin": "test-agent",
            },
        )
        assert not add_res.is_error
        add_data = json.loads(add_res.content[0].text)
        assert add_data["status"] == "created"
        capture_id = add_data["capture_id"]
        resource_id = add_data["resource_id"]

        # Show item
        show_res = await server.call_tool("edward_show", {"object_id": capture_id})
        assert not show_res.is_error
        show_data = json.loads(show_res.content[0].text)
        assert show_data["type"] == "capture"
        assert show_data["data"]["id"] == capture_id
        assert show_data["data"]["user_note"] == "Crucial primer on cybernetics"

        # Show resource
        show_res_r = await server.call_tool("edward_show", {"object_id": resource_id})
        assert not show_res_r.is_error
        show_data_r = json.loads(show_res_r.content[0].text)
        assert show_data_r["type"] == "resource"
        assert show_data_r["data"]["id"] == resource_id

        # Search lexical
        search_res = await server.call_tool("edward_search", {"query": "cybernetics feedback"})
        assert not search_res.is_error
        search_data = json.loads(search_res.content[0].text)
        assert search_data["count"] >= 1
        assert any(
            "cybernetics" in item.get("snippet", "").lower() for item in search_data["results"]
        )

        # Ask question (Evidence Packet mode)
        ask_res = await server.call_tool(
            "edward_ask", {"query": "What governs complex adaptive systems?"}
        )
        assert not ask_res.is_error
        ask_data = json.loads(ask_res.content[0].text)
        assert "evidence_packet" in ask_data
        packet = ask_data["evidence_packet"]
        assert "items" in packet
        assert len(packet["items"]) >= 1

        # Export evidence packet
        export_res = await server.call_tool("edward_export_packet", {"query": "cybernetics"})
        assert not export_res.is_error
        export_data = json.loads(export_res.content[0].text)
        assert export_data["type"] == "evidence-packet"
        assert export_data["schema_version"] == "1"
        assert export_data["query"] == "cybernetics"

    asyncio.run(_test())


def test_mcp_annotations_and_intents(test_db: Database, test_blob_store: BlobStore) -> None:
    """Verify human annotations and intent classification tools."""

    async def _test() -> None:
        server = create_mcp_server(db=test_db, blob_store=test_blob_store)

        # First capture an item
        add_res = await server.call_tool(
            "edward_add",
            {"text": "Annotation and intent test payload.", "url": "https://example.com/intent"},
        )
        add_data = json.loads(add_res.content[0].text)
        capture_id = add_data["capture_id"]

        # List available intent questions
        intents_res = await server.call_tool("edward_list_intents", {})
        assert not intents_res.is_error
        intents_data = json.loads(intents_res.content[0].text)
        assert intents_data["count"] > 0
        assert any(i.get("id") == "intent-essay-seed" for i in intents_data["intents"])

        # Annotate
        ann_res = await server.call_tool(
            "edward_annotate",
            {
                "object_id": capture_id,
                "note": "Updated human note",
                "label": "epistemology",
                "intent": "critical-reading",
            },
        )
        assert not ann_res.is_error
        ann_data = json.loads(ann_res.content[0].text)
        assert ann_data["status"] == "annotated"

        # Verify annotation on show
        show_res = await server.call_tool("edward_show", {"object_id": capture_id})
        show_data = json.loads(show_res.content[0].text)
        assert any(it.get("intent") == "critical-reading" for it in show_data["data"]["intents"])
        assert any(
            a.get("content") == "Updated human note" for a in show_data["data"]["annotations"]
        )

        # Accept intent
        accept_res = await server.call_tool(
            "edward_accept_intent", {"object_id": capture_id, "intent": "critical-reading"}
        )
        assert not accept_res.is_error
        accept_data = json.loads(accept_res.content[0].text)
        assert accept_data.get("status") == "accepted"

        # Remove intent
        remove_res = await server.call_tool(
            "edward_remove_intent", {"object_id": capture_id, "intent": "critical-reading"}
        )
        assert not remove_res.is_error
        remove_data = json.loads(remove_res.content[0].text)
        assert remove_data.get("status") == "deactivated"

    asyncio.run(_test())


def test_mcp_project_lifecycle(test_db: Database, test_blob_store: BlobStore) -> None:
    """Verify full project workspace lifecycle through MCP tools."""

    async def _test() -> None:
        server = create_mcp_server(db=test_db, blob_store=test_blob_store)

        # Capture item to use as evidence
        add_res = await server.call_tool(
            "edward_add",
            {
                "text": "Information theory establishes limits on signal transmission and compression.",
                "url": "https://example.com/shannon",
            },
        )
        capture_id = json.loads(add_res.content[0].text)["capture_id"]

        # Create project
        create_res = await server.call_tool(
            "edward_project_create",
            {
                "title": "Shannon Information Project",
                "slug": "shannon-info",
                "brief": "Investigating information entropy and transmission limits.",
            },
        )
        assert not create_res.is_error
        project_data = json.loads(create_res.content[0].text)
        project_id = project_data["id"]
        assert project_data["title"] == "Shannon Information Project"

        # List projects
        list_res = await server.call_tool("edward_project_list", {})
        assert not list_res.is_error
        projects_data = json.loads(list_res.content[0].text)
        assert projects_data["count"] == 1
        assert projects_data["projects"][0]["slug"] == "shannon-info"

        # Add project note
        note_res = await server.call_tool(
            "edward_project_add_note",
            {"project_id": project_id, "kind": "note", "text": "Note on channel capacity theorem."},
        )
        assert not note_res.is_error

        # Add evidence
        ev_res = await server.call_tool(
            "edward_project_add_evidence",
            {"project_id": project_id, "object_id": capture_id, "note": "Core transmission paper"},
        )
        assert not ev_res.is_error

        # Get context
        ctx_res = await server.call_tool("edward_project_context", {"project_id": project_id})
        assert not ctx_res.is_error
        ctx_data = json.loads(ctx_res.content[0].text)
        assert len(ctx_data["notes"]) == 1
        assert len(ctx_data["memberships"]) == 1

        # Propose outline (with provided structured outline)
        outline_res = await server.call_tool(
            "edward_project_propose_outline",
            {
                "project_id": project_id,
                "outline": {
                    "title": "Shannon Information Outline",
                    "premise": "Information entropy is fundamental.",
                    "sections": [
                        {
                            "heading": "Introduction",
                            "evidence": [{"object_id": capture_id, "relationship": "supporting"}],
                        },
                        {
                            "heading": "Channel Capacity",
                            "evidence": [],
                        },
                    ],
                },
            },
        )
        assert not outline_res.is_error
        outline_data = json.loads(outline_res.content[0].text)
        outline_id = outline_data["id"]
        assert outline_data["status"] == "proposal"
        assert len(outline_data["sections"]) == 2

        # Accept outline
        accept_res = await server.call_tool(
            "edward_project_accept_outline",
            {"project_id": project_id, "outline_id": outline_id},
        )
        assert not accept_res.is_error
        accepted_data = json.loads(accept_res.content[0].text)
        assert accepted_data["status"] == "accepted"

        # Deterministic outline proposal without outline payload
        det_outline_res = await server.call_tool(
            "edward_project_propose_outline",
            {"project_id": project_id},
        )
        assert not det_outline_res.is_error
        det_outline_data = json.loads(det_outline_res.content[0].text)
        assert "id" in det_outline_data
        assert det_outline_data["status"] == "proposal"

        # Remove evidence
        rm_ev_res = await server.call_tool(
            "edward_project_remove_evidence",
            {"project_id": project_id, "object_id": capture_id},
        )
        assert not rm_ev_res.is_error

        # Delete project
        del_res = await server.call_tool(
            "edward_project_delete",
            {"project_id": project_id, "confirm": True},
        )
        assert not del_res.is_error
        del_data = json.loads(del_res.content[0].text)
        assert del_data["status"] == "deleted"

    asyncio.run(_test())


def test_mcp_cli_command(cli_runner: CliRunner) -> None:
    """Verify edward mcp CLI interface."""
    # Test --help
    help_res = cli_runner.invoke(app, ["mcp", "--help"])
    assert help_res.exit_code == 0
    assert "Run the native Edward Model Context Protocol (MCP) server" in help_res.stdout

    # Test invalid transport
    err_res = cli_runner.invoke(app, ["mcp", "--transport", "websocket"])
    assert err_res.exit_code == 2
    assert "Unsupported transport" in err_res.stderr


def test_mcp_import_process_sync(test_db: Database, test_blob_store: BlobStore) -> None:
    """Verify research import, background processing, and sync adapters."""

    async def _test() -> None:
        server = create_mcp_server(db=test_db, blob_store=test_blob_store)

        # 1. Ingest markdown report
        md_text = """# Autonomous Agents Architecture

Detailed report on agent workflows, tool use, and cognitive memory.

## Key Insights
- Agents require typed schemas and deterministic tool outputs.
"""
        import_res = await server.call_tool(
            "edward_import_research",
            {
                "content": md_text,
                "format": "markdown",
                "title": "Autonomous Agents Architecture",
                "collector": "report-agent",
            },
        )
        assert not import_res.is_error
        import_data = json.loads(import_res.content[0].text)
        assert import_data.get("status") in ("created", "imported")
        assert "capture_id" in import_data

        # 2. Process background queue
        proc_res = await server.call_tool("edward_process", {"limit": 10})
        assert not proc_res.is_error
        proc_data = json.loads(proc_res.content[0].text)
        assert "completed" in proc_data

        # 3. Sync adapter error handling for missing file
        sync_res = await server.call_tool(
            "edward_sync",
            {"source": "x", "archive_path": "/nonexistent/path/bookmarks.js"},
        )
        assert not sync_res.is_error
        sync_data = json.loads(sync_res.content[0].text)
        assert "error" in sync_data

    asyncio.run(_test())
