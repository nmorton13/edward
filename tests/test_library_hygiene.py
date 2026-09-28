"""Tests for library hygiene and isolation guards against test data pollution."""

from unittest.mock import patch

from typer.testing import CliRunner

from edward.cli import app
from edward.db import (
    get_canonical_default_data_dir,
    get_default_data_dir,
    is_default_data_dir,
    is_test_name,
    is_test_namespace,
)
from edward.mcp_server import create_mcp_server


def test_is_test_namespace():
    assert is_test_namespace("agent-test") is True
    assert is_test_namespace("test") is True
    assert is_test_namespace("test-sweep") is True
    assert is_test_namespace("hermes-test") is True
    assert is_test_namespace("testing") is True

    assert is_test_namespace("x") is False
    assert is_test_namespace("gmail") is False
    assert is_test_namespace("web") is False
    assert is_test_namespace("manual") is False
    assert is_test_namespace("arxiv") is False
    assert is_test_namespace(None) is False
    assert is_test_namespace("") is False


def test_is_test_name():
    assert is_test_name("Hermes probe project") is True
    assert is_test_name("Test Energy Flexibility Deep Dive") is True
    assert is_test_name("test-energy-flexibility") is True
    assert is_test_name("capability sweep probe") is True
    assert is_test_name("test") is True

    assert is_test_name("Data centers and local economies") is False
    assert is_test_name("Do data centers raise your electric bill?") is False
    assert is_test_name(None) is False
    assert is_test_name("") is False


def test_test_suite_never_touches_real_library():
    """Verify that pytest autouse fixture always isolates EDWARD_DATA_DIR from ~/.edward."""
    canonical = get_canonical_default_data_dir()
    current = get_default_data_dir()

    assert current != canonical, f"Tests must not point to the real library ({canonical})"
    assert is_default_data_dir() is False


def test_cli_warns_on_test_namespace_in_default_library(
    cli_runner: CliRunner, test_db, test_blob_store
):
    """When simulating default library, CLI emits a warning to stderr on test origin."""
    with patch("edward.cli.is_default_data_dir", return_value=True):
        res = cli_runner.invoke(
            app,
            ["add", "--origin", "agent-test", "--text", "Test finding text", "--json"],
        )
        assert res.exit_code == 0
        assert "Warning" in res.stderr
        assert "appears to be test data" in res.stderr
        # Verify valid JSON on stdout
        import json

        data = json.loads(res.stdout)
        assert data["status"] == "created"


def test_cli_no_warning_in_isolated_scratch_library(
    cli_runner: CliRunner, test_db, test_blob_store
):
    """In an isolated scratch library (EDWARD_DATA_DIR set to tmp), no warning is emitted."""
    assert is_default_data_dir() is False
    res = cli_runner.invoke(
        app,
        ["add", "--origin", "agent-test", "--text", "Test finding text", "--json"],
    )
    assert res.exit_code == 0
    assert "Warning" not in res.stderr


def test_cli_warns_on_test_project_in_default_library(
    cli_runner: CliRunner, test_db, test_blob_store
):
    """When simulating default library, CLI emits a warning on test/probe project titles."""
    with patch("edward.cli.is_default_data_dir", return_value=True):
        res = cli_runner.invoke(
            app,
            ["project", "create", "--title", "Hermes probe project", "--json"],
        )
        assert res.exit_code == 0
        assert "Warning" in res.stderr
        assert "appears to be test/probe data" in res.stderr
        import json

        data = json.loads(res.stdout)
        assert data["title"] == "Hermes probe project"


def test_mcp_warns_on_test_namespace_in_default_library(test_db, test_blob_store, monkeypatch):
    """When simulating default library, MCP server warns on test capture."""
    server = create_mcp_server(test_db, test_blob_store)
    capture_tool = server._tool_manager.get_tool("edward_add")

    with patch("edward.mcp_server.is_default_data_dir", return_value=True):
        res = capture_tool.fn(text="MCP test content", origin="agent-test")
        assert "warning" in res
        assert "appears to be test data" in res["warning"]
        assert res.get("status") == "created"


def test_mcp_warns_on_test_project_in_default_library(test_db, test_blob_store):
    """When simulating default library, MCP server warns on test project create."""
    server = create_mcp_server(test_db, test_blob_store)
    project_tool = server._tool_manager.get_tool("edward_project_create")

    with patch("edward.mcp_server.is_default_data_dir", return_value=True):
        res = project_tool.fn(title="Hermes probe project")
        assert "warning" in res
        assert "appears to be test/probe data" in res["warning"]
        assert res["title"] == "Hermes probe project"
