"""Integration tests for Phase 2 CLI commands: process, retry, status, and import-research."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from edward.cli import app


@pytest.fixture(autouse=True)
def configure_test_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point EDWARD_DATA_DIR to a temporary directory for CLI tests."""
    data_dir = tmp_path / "edward_cli_phase2_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("EDWARD_DATA_DIR", str(data_dir))


def test_cli_status(cli_runner: CliRunner):
    # Add an item to populate DB
    cli_runner.invoke(app, ["add", "--text", "Sample status text", "--json"])

    # Human output mode
    res_human = cli_runner.invoke(app, ["status"])
    assert res_human.exit_code == 0
    assert "Edward Status" in res_human.stdout
    assert "Corpus:" in res_human.stdout

    # JSON output mode
    res_json = cli_runner.invoke(app, ["status", "--json"])
    assert res_json.exit_code == 0
    data = json.loads(res_json.stdout)
    assert "processing_jobs" in data
    assert "counts" in data
    assert data["counts"]["captures"] >= 1


def test_cli_retry_requires_failed_flag(cli_runner: CliRunner):
    # Without --failed should exit with code 2
    res = cli_runner.invoke(app, ["retry"])
    assert res.exit_code == 2

    # With --failed
    res_retry = cli_runner.invoke(app, ["retry", "--failed", "--json"])
    assert res_retry.exit_code == 0
    data = json.loads(res_retry.stdout)
    assert data["status"] == "retried"
    assert data["count"] == 0


def test_cli_process(cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch):
    from edward.services.extract import ExtractionResult
    from edward.services.network import FetchResult

    monkeypatch.setattr(
        "edward.services.processor.safe_fetch_url",
        lambda url: FetchResult(
            url=url,
            final_url=url,
            status_code=200,
            headers={"content-type": "text/html"},
            body=b"<html><body><h1>Test Pipeline</h1><p>Hermetic body text</p></body></html>",
            content_hash="mock_hash_12345",
            elapsed_seconds=0.05,
        ),
    )
    monkeypatch.setattr(
        "edward.services.processor.extract_content",
        lambda url, raw_html=None: ExtractionResult(
            status="completed",
            clean_text="Hermetic extracted text for test pipeline.",
            title="Test Pipeline",
            extractor="mock",
            extractor_version="1.0",
        ),
    )

    # Capture a URL to generate a pending processing job
    add_res = cli_runner.invoke(
        app,
        ["add", "--url", "https://example.com/test-pipeline", "--json"],
    )
    assert add_res.exit_code == 0

    # Status shows 1 pending job
    stat_res = cli_runner.invoke(app, ["status", "--json"])
    stat_data = json.loads(stat_res.stdout)
    assert stat_data["processing_jobs"]["by_status"]["pending"] >= 1

    # Human process mode
    res_proc = cli_runner.invoke(app, ["process", "--limit", "1"])
    assert res_proc.exit_code == 0
    assert "Processed jobs:" in res_proc.stdout


def test_cli_import_research_bundle_json(cli_runner: CliRunner, tmp_path: Path):
    bundle_file = tmp_path / "test_bundle.json"
    bundle_payload = {
        "type": "research-bundle",
        "schema_version": "1",
        "bundle_id": "bun_cli_1",
        "idempotency_key": "bundle-idemp-1",
        "title": "Quantum Error Correction 2026",
        "sources": [
            {
                "origin": "web",
                "url": "https://example.com/qec-2026",
                "title": "QEC Overview",
            }
        ],
        "findings": [
            {
                "statement": "Syndrome measurement speed has improved by 10x.",
                "assertion_role": "source-claim",
                "source_url": "https://example.com/qec-2026",
            }
        ],
    }
    bundle_file.write_text(json.dumps(bundle_payload), encoding="utf-8")

    # Import via positional argument with --json
    res = cli_runner.invoke(app, ["import-research", str(bundle_file), "--json"])
    assert res.exit_code == 0
    data = json.loads(res.stdout)
    assert data["status"] == "imported"
    assert data["bundle_id"] == "bun_cli_1"
    assert "capture_id" in data

    # Replay with same idempotency key
    replay_res = cli_runner.invoke(app, ["import-research", str(bundle_file), "--json"])
    assert replay_res.exit_code == 0
    replay_data = json.loads(replay_res.stdout)
    assert replay_data["status"] == "replayed"
    assert replay_data["capture_id"] == data["capture_id"]

    # Conflict with modified content
    bundle_conflict = tmp_path / "test_bundle_conflict.json"
    bundle_payload["title"] = "Conflicting Quantum Title"
    bundle_conflict.write_text(json.dumps(bundle_payload), encoding="utf-8")

    conflict_res = cli_runner.invoke(app, ["import-research", str(bundle_conflict), "--json"])
    assert conflict_res.exit_code == 3


def test_cli_import_research_markdown(cli_runner: CliRunner):
    md_content = """# Autonomous Systems Review

Autonomous vehicles have demonstrated 99.999% reliability in urban geofenced zones.

### Core Architecture
- LiDAR + Vision sensor fusion.
"""
    res = cli_runner.invoke(
        app,
        ["import-research", "--stdin", "--format", "markdown", "--json"],
        input=md_content,
    )
    assert res.exit_code == 0
    data = json.loads(res.stdout)
    assert data["status"] == "imported"
    assert data["format"] == "markdown"
    assert data["title"] == "Autonomous Systems Review"
    assert "resource_id" in data


def test_cli_import_research_errors(cli_runner: CliRunner, tmp_path: Path):
    # No file and no stdin: exit code 2
    res_no_args = cli_runner.invoke(app, ["import-research"])
    assert res_no_args.exit_code == 2

    # Non-existent file: exit code 1
    res_missing = cli_runner.invoke(app, ["import-research", str(tmp_path / "nonexistent.json")])
    assert res_missing.exit_code == 1

    # Malformed JSON file: exit code 1
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not-valid-json", encoding="utf-8")
    res_bad = cli_runner.invoke(app, ["import-research", str(bad_json)])
    assert res_bad.exit_code == 1
