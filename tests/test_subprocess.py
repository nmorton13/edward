"""Tests for secure subprocess runner."""

import pytest

from edward.services.network import SSRFSecurityError
from edward.services.subprocess_runner import (
    OutputLimitExceededError,
    SubprocessTimeoutError,
    ToolNotFoundError,
    get_sanitized_env,
    run_tool,
)


def test_sanitized_env_excludes_secrets(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "placeholder")
    monkeypatch.setenv("OPENAI_API_KEY", "placeholder")
    monkeypatch.setenv("MY_PASSWORD", "supersecret")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")

    env = get_sanitized_env()
    assert "TYPESAFE_API_KEY" not in env
    assert "OPENAI_API_KEY" not in env
    assert "MY_PASSWORD" not in env
    assert "PATH" in env


def test_run_tool_empty_args():
    with pytest.raises(ValueError, match="Command arguments cannot be empty"):
        run_tool([])


def test_run_tool_executable_not_found():
    with pytest.raises(ToolNotFoundError, match="Tool executable 'nonexistent_tool_xyz' not found"):
        run_tool(["nonexistent_tool_xyz", "--version"])


def test_run_tool_blocks_ssrf_urls_before_launch():
    with pytest.raises(SSRFSecurityError):
        run_tool(["echo", "http://127.0.0.1/private"])


def test_run_tool_success():
    res = run_tool(["echo", "hello world"])
    assert res.exit_code == 0
    assert res.stdout.strip() == "hello world"
    assert res.duration_seconds >= 0


def test_run_tool_timeout():
    # Use python to sleep longer than timeout
    with pytest.raises(SubprocessTimeoutError, match="timed out"):
        run_tool(["python3", "-c", "import time; time.sleep(1.0)"], timeout=0.1)


def test_run_tool_output_limit_exceeded():
    # Attempt to output more than max_output_bytes
    with pytest.raises(OutputLimitExceededError, match="exceeded limit"):
        run_tool(["python3", "-c", "print('A' * 1000)"], max_output_bytes=500)
