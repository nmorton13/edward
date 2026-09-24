"""Tests for .env parsing, ensuring inline comments and quotes are handled correctly."""

from pathlib import Path

from edward.db import parse_env_line


def test_parse_env_line_cases():
    # Empty and comment lines
    assert parse_env_line("") is None
    assert parse_env_line("   ") is None
    assert parse_env_line("# Full line comment") is None
    assert parse_env_line("   # Indented comment") is None

    # Simple key-value
    assert parse_env_line("KEY=val") == ("KEY", "val")
    assert parse_env_line("  KEY = val  ") == ("KEY", "val")

    # Unquoted with inline comments
    assert parse_env_line("KEY=val # inline comment") == ("KEY", "val")
    assert parse_env_line("EDWARD_ANSWER_LOCATION=local  # local | hosted") == (
        "EDWARD_ANSWER_LOCATION",
        "local",
    )
    assert parse_env_line("EDWARD_CLASSIFIER_PROVIDER=disabled  # typesafe | openrouter") == (
        "EDWARD_CLASSIFIER_PROVIDER",
        "disabled",
    )

    # Double quotes preserving internal hashes
    assert parse_env_line('KEY="val # not a comment"') == ("KEY", "val # not a comment")
    assert parse_env_line('KEY="val # not a comment" # real comment') == (
        "KEY",
        "val # not a comment",
    )

    # Single quotes preserving internal hashes
    assert parse_env_line("KEY='val # not a comment'") == ("KEY", "val # not a comment")
    assert parse_env_line("KEY='val # not a comment' # real comment") == (
        "KEY",
        "val # not a comment",
    )

    # Values containing equals signs
    assert parse_env_line("URL=https://example.com?a=1&b=2") == (
        "URL",
        "https://example.com?a=1&b=2",
    )

    # Lines with export prefix
    assert parse_env_line("export FOO=bar") == ("FOO", "bar")


def test_dot_env_example_clean_parsing():
    root = Path(__file__).parent.parent
    env_example_path = root / ".env.example"
    assert env_example_path.exists()

    env = {}
    for line in env_example_path.read_text(encoding="utf-8").splitlines():
        parsed = parse_env_line(line)
        if parsed:
            k, v = parsed
            env[k] = v

    # Critical variables must have their inline comments cleanly stripped
    assert "EDWARD_CLASSIFIER_LOCATION" not in env  # Hosted providers are forced hosted in code.
    assert env["EDWARD_CLASSIFIER_PROVIDER"] == "disabled"
    assert env["EDWARD_HOSTED_GMAIL"] == "deny"
    assert env["EDWARD_HOSTED_PUBLIC_WEB"] == "allow"
    assert "EDWARD_EMBEDDING_BASE_URL" not in env
    assert "EDWARD_EMBEDDING_DIMENSIONS" not in env
    assert "EDWARD_EMBEDDING_LOCATION" not in env

    # The template must not ship an active answerer. Edward is memory, not a mind:
    # copying .env.example to .env must leave every model path disabled.
    for key in (
        "EDWARD_ANSWERER_MODE",
        "EDWARD_ANSWER_BASE_URL",
        "EDWARD_ANSWER_MODEL",
        "EDWARD_ANSWER_LOCATION",
    ):
        assert key not in env, (
            f"{key} is active in .env.example; the template must not enable a model by default"
        )

    # No parsed value in .env.example should contain a '#' character or trailing comment text
    for key, val in env.items():
        assert "#" not in val, f"Key {key} contains '#' in parsed value: {val!r}"
        assert not val.endswith(" "), f"Key {key} has trailing whitespace in value: {val!r}"
