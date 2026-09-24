"""Secure subprocess runner for external extraction and collection tools."""

import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from edward.services.network import validate_url_for_ssrf

MAX_OUTPUT_BYTES = 10 * 1024 * 1024  # 10 MB
DEFAULT_TIMEOUT_SECONDS = 30.0

# Strict environment variable allowlist: strips secrets, tokens, API keys
SAFE_ENV_ALLOWLIST = {
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "TERM",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
}


class SubprocessError(Exception):
    """Base error for subprocess operations."""

    pass


class ToolNotFoundError(SubprocessError):
    """Raised when the specified executable is not found in PATH."""

    pass


class SubprocessTimeoutError(SubprocessError):
    """Raised when a subprocess execution times out."""

    pass


class OutputLimitExceededError(SubprocessError):
    """Raised when subprocess output exceeds maximum buffer size."""

    pass


@dataclass
class SubprocessResult:
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float


def is_tool_available(tool_name: str) -> bool:
    """Check if an executable exists in system PATH."""
    return shutil.which(tool_name) is not None


def get_sanitized_env() -> dict[str, str]:
    """Return a minimal environment containing only safe system variables."""
    return {k: v for k, v in os.environ.items() if k in SAFE_ENV_ALLOWLIST}


def run_tool(
    args: list[str],
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_output_bytes: int = MAX_OUTPUT_BYTES,
    cwd: str | None = None,
) -> SubprocessResult:
    """Execute an external CLI tool securely without shell expansion, with SSRF checks on URL arguments."""
    if not args:
        raise ValueError("Command arguments cannot be empty")

    tool_name = args[0]
    executable = shutil.which(tool_name)
    if not executable:
        raise ToolNotFoundError(f"Tool executable '{tool_name}' not found in PATH")

    # SSRF Pre-validation on all URL arguments before process launch
    for arg in args[1:]:
        if isinstance(arg, str) and (arg.startswith("http://") or arg.startswith("https://")):
            validate_url_for_ssrf(arg)

    start_time = time.monotonic()
    env = get_sanitized_env()

    with tempfile.TemporaryFile() as out_f, tempfile.TemporaryFile() as err_f:
        try:
            proc = subprocess.Popen(
                [executable] + args[1:],
                stdout=out_f,
                stderr=err_f,
                stdin=subprocess.DEVNULL,
                shell=False,
                cwd=cwd,
                env=env,
            )
        except OSError as e:
            raise SubprocessError(f"Failed to spawn subprocess '{tool_name}': {e}") from e

        deadline = start_time + timeout
        try:
            while proc.poll() is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    proc.kill()
                    proc.wait()
                    raise SubprocessTimeoutError(
                        f"Process '{tool_name}' timed out after {timeout} seconds"
                    )

                out_size = os.fstat(out_f.fileno()).st_size
                err_size = os.fstat(err_f.fileno()).st_size
                if out_size > max_output_bytes:
                    proc.kill()
                    proc.wait()
                    raise OutputLimitExceededError(
                        f"Process stdout length {out_size} exceeded limit {max_output_bytes}"
                    )
                if err_size > max_output_bytes:
                    proc.kill()
                    proc.wait()
                    raise OutputLimitExceededError(
                        f"Process stderr length {err_size} exceeded limit {max_output_bytes}"
                    )

                time.sleep(0.01)

            # Ensure final size check
            out_size = os.fstat(out_f.fileno()).st_size
            err_size = os.fstat(err_f.fileno()).st_size
            if out_size > max_output_bytes:
                raise OutputLimitExceededError(
                    f"Process stdout length {out_size} exceeded limit {max_output_bytes}"
                )
            if err_size > max_output_bytes:
                raise OutputLimitExceededError(
                    f"Process stderr length {err_size} exceeded limit {max_output_bytes}"
                )

            out_f.seek(0)
            err_f.seek(0)
            stdout_bytes = out_f.read()
            stderr_bytes = err_f.read()
        except (SubprocessTimeoutError, OutputLimitExceededError):
            raise
        except Exception as e:
            proc.kill()
            proc.wait()
            raise SubprocessError(f"Error monitoring process '{tool_name}': {e}") from e

        elapsed = time.monotonic() - start_time
        stdout_str = stdout_bytes.decode("utf-8", errors="replace")
        stderr_str = stderr_bytes.decode("utf-8", errors="replace")

        return SubprocessResult(
            exit_code=proc.returncode,
            stdout=stdout_str,
            stderr=stderr_str,
            duration_seconds=elapsed,
        )


def sanitize_error_message(err: str, max_chars: int = 1000) -> str:
    """Truncate error strings, mask credentials/tokens, and mask user home directories."""
    if not err:
        return ""

    sanitized = str(err)

    # 1. Mask user home directory
    try:
        home_dir = str(Path.home())
        if home_dir and home_dir != "/":
            sanitized = sanitized.replace(home_dir, "~")
    except Exception:
        pass

    sanitized = re.sub(r"/(?:Users|home)/[a-zA-Z0-9._-]+", "~", sanitized)

    # 2. Mask authorization headers, bearer tokens, passwords, and secret keys
    sanitized = re.sub(
        r"(?i)(bearer\s+)[A-Za-z0-9_\-\.]{8,}",
        r"\1[REDACTED]",
        sanitized,
    )
    sanitized = re.sub(
        r"(?i)(api[_-]?key|access[_-]?token|auth[_-]?token|secret|password|passwd|private[_-]?key)[\s:=]+([^\s,;\"'>]{4,})",
        r"\1=[REDACTED]",
        sanitized,
    )
    sanitized = re.sub(
        r"(?i)https?://[^:\s]+:[^@\s]+@",
        "http://[REDACTED]@",
        sanitized,
    )

    # 3. Truncate if exceeds max_chars
    if len(sanitized) > max_chars:
        sanitized = sanitized[:max_chars] + "... [truncated]"

    return sanitized
