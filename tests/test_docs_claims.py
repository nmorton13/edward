"""Tests that documentation matches the running system.

Documentation drifts silently. These tests pin the claims a reader would rely on, so
a future change that invalidates the docs fails the build instead of misleading
whoever reads them next.
"""

from pathlib import Path

ROOT = Path(__file__).parent.parent


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_readme_does_not_claim_a_default_model():
    """The README must not present model use as the default condition."""
    readme = _read("README.md")

    assert "memory, not a mind" in readme.lower()
    assert "OFF BY DEFAULT" in _read(".env.example").upper()


def test_readme_documents_the_opt_in_variable():
    """EDWARD_ANSWERER_MODE is the only thing that enables a model, so it must be documented."""
    readme = _read("README.md")

    assert "EDWARD_ANSWERER_MODE" in readme
    assert "requires `EDWARD_ANSWERER_MODE`" in readme


def test_readme_no_longer_claims_a_configured_local_answerer():
    """The old wording asserted a default endpoint and model; that is no longer true."""
    readme = _read("README.md")

    assert "By default, Edward expects a local OpenAI-compatible chat server" not in readme


def test_agent_protocol_documents_the_required_origin_field():
    """An agent must be able to learn that sources[].origin is required."""
    protocol = _read("docs/AGENT_PROTOCOL.md")

    assert "`sources[].origin` is **required**" in protocol
    assert 'extra="allow"' in protocol
    assert 'extra="forbid"' in protocol


def test_agent_protocol_documents_title_behaviour():
    """The title table is the only place an agent learns which paths drop a title."""
    protocol = _read("docs/AGENT_PROTOCOL.md")

    assert "Titles on import" in protocol
    assert "No title" in protocol
    assert "import-research" in protocol


def test_agent_protocol_documents_purge_jobs_guard():
    """purge-jobs deletes from a shared queue, so its confirmation requirement must be stated."""
    protocol = _read("docs/AGENT_PROTOCOL.md")

    assert "purge-jobs" in protocol
    assert "--dry-run" in protocol
    assert "--yes" in protocol


def test_agent_protocol_mentions_the_model_free_default():
    """An agent must know extraction and answering need no model by default."""
    protocol = _read("docs/AGENT_PROTOCOL.md")

    assert "No model is used for extraction or answering by default" in protocol


def test_agent_protocol_documents_the_offline_outline_limits():
    """An agent must not mistake an offline outline for semantic assignment.

    The documented mapping is the contract: it is the only signal the offline path has,
    and an agent that adds evidence without a relationship will get silent support.
    """
    protocol = _read("docs/AGENT_PROTOCOL.md")

    assert "It does not understand what any source says" in protocol
    assert "--relationship counterargument" in protocol
    assert "Assigning evidence to claims is the agent's job" in protocol


def test_documented_relationship_mapping_matches_the_code():
    """Keep the documented membership->outline table honest."""
    from edward.services.projects import to_evidence_relationship

    documented = {
        "evidence": "supporting",
        "supporting": "supporting",
        "counterargument": "counterevidence",
        "background": "qualification",
        "question": "qualification",
        "gap": "qualification",
    }
    for membership, expected in documented.items():
        assert to_evidence_relationship(membership) == expected, (
            f"docs claim {membership!r} maps to {expected!r}"
        )


def test_documented_db_path_matches_the_code():
    """The README names the database file; a wrong filename sends readers hunting."""
    from edward.db import get_default_db_path

    assert "edward.sqlite3" in _read("README.md"), "README must name the real db file"
    assert get_default_db_path().name == "edward.sqlite3"


def test_docs_do_not_call_classification_outside_the_query_path():
    """The classifier is absent from query *execution* but its labels gate filtering.

    Calling it flatly 'not part of the query path' is the imprecision this pins against:
    `--topic`/`--form` read `object_labels`, and results carry `labels[]`.
    """
    from edward.services import search as search_service

    source = Path(search_service.__file__).read_text(encoding="utf-8")
    assert "object_labels" in source, "labels must really be consumed by the query surface"

    for doc in ("README.md", "docs/AGENT_PROTOCOL.md", ".agents/skills/edward/SKILL.md"):
        text = _read(doc)
        assert "not part of the query path" not in text, (
            f"{doc} calls classification 'not part of the query path'; state the two halves"
        )


def test_agent_skill_states_the_model_free_default():
    """The shipped agent skill is what other agents read; it must know no model runs."""
    skill = _read(".agents/skills/edward/SKILL.md")

    assert "No model runs by default" in skill
    assert "counterargument" in skill, "the skill must tell an agent to mark the other side"


def test_readme_documents_the_guarded_purge_jobs():
    """purge-jobs deletes from a shared queue, so its guard must be documented."""
    readme = _read("README.md")

    assert "purge-jobs" in readme
    assert "--dry-run" in readme
    assert "--yes" in readme


def test_changelog_records_the_shipped_posture():
    """A CHANGELOG that stops at the first release misleads about what shipped since."""
    changelog = _read("CHANGELOG.md")

    assert "Unreleased" in changelog
    assert "memory, not a mind" in changelog.lower()


def test_env_example_names_sqlite3_db():
    """The template .env file must specify the real edward.sqlite3 filename."""
    env_example = _read(".env.example")
    assert "edward.sqlite3" in env_example
    assert "edward.sqlite\n" not in env_example


def test_readme_documents_retry_failed():
    """The CLI requires --failed on retry; the README must document it."""
    readme = _read("README.md")
    assert "edward retry --failed" in readme


def test_agent_protocol_sections_are_unique_and_sequential():
    """Section headings in AGENT_PROTOCOL must not duplicate numbers."""
    import re

    protocol = _read("docs/AGENT_PROTOCOL.md")
    sections = re.findall(r"^###\s+([0-9]+\.[0-9]+[a-z]?)\s+", protocol, re.MULTILINE)
    assert len(sections) == len(set(sections)), (
        f"Duplicate sections found in AGENT_PROTOCOL: {sections}"
    )


def test_agent_protocol_documents_all_maintenance_and_intent_commands():
    """The capability map in AGENT_PROTOCOL must list all maintenance and intent tools."""
    protocol = _read("docs/AGENT_PROTOCOL.md")
    for cmd in (
        "intents",
        "accept-intent",
        "repair-titles",
        "migrate-media",
        "backfill-shortlinks",
        "repair-chunks",
        "backup",
        "doctor",
        "purge --confirm",
        "retry --failed",
    ):
        assert cmd in protocol, f"AGENT_PROTOCOL capability map must list {cmd}"


def test_skill_references_correct_outline_section():
    """SKILL.md must point to the project workspaces section in AGENT_PROTOCOL."""
    skill = _read(".agents/skills/edward/SKILL.md")
    assert "AGENT_PROTOCOL §2.8" in skill
    assert "28-project-workspaces-edward-project" in skill


def test_agent_protocol_scopes_idempotency_to_supported_commands():
    """Do not promise idempotency flags on commands that lack them."""
    protocol = _read("docs/AGENT_PROTOCOL.md")

    assert "CLI `add` and `import-research` commands" in protocol
    assert "All mutating operations accept" not in protocol


def test_agent_protocol_describes_mcp_as_a_cli_subset():
    """MCP exposes a useful subset, not every CLI maintenance operation."""
    protocol = _read("docs/AGENT_PROTOCOL.md")

    assert "not full CLI parity" in protocol
    assert "Full CLI Parity" not in protocol
    assert "no_model: bool = True" not in protocol
    assert "all_records: bool = False" in protocol


def test_changelog_does_not_publish_local_instance_details():
    """Release notes must not expose local corpus statistics or record identifiers."""
    changelog = _read("CHANGELOG.md")

    assert "### Data" not in changelog
    assert "prj_" not in changelog
    assert "local untracked `archive/`" not in changelog
