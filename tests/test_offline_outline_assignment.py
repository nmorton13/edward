"""Offline outlines must assign evidence by relevance, not by DB order.

These pin the three defects found by inspecting a real writing workspace:

1. Deterministic assignment sliced a flat list positionally, so "the author's own
   regret over tone" landed next to "state taxation of data centers" in the same
   section.
2. Memberships whose relationship mapped to `qualification` (background/question/gap)
   were computed and then silently dropped: no section ever emitted them.
3. Every unresolved research need was repeated in every section that carried needs,
   which is noise rather than signal.

The offline path must keep every piece of workspace evidence reachable from the
outline, and must not present a dropped item as if it had been placed.
"""

from edward.models import CaptureInput
from edward.services.capture import capture_item
from edward.services.projects import (
    add_project_note,
    add_project_object,
    create_project,
    deterministic_outline,
    get_project_context,
)


def _capture(db, text: str) -> str:
    with db.transaction() as conn:
        return capture_item(conn, CaptureInput(text=text))["capture_id"]


def _project_with(db, memberships, notes=(), title="Offline outline fidelity"):
    """Build a project whose memberships/notes are exactly what the test needs."""
    ids = {text: _capture(db, text) for text, _ in memberships}
    with db.transaction() as conn:
        project = create_project(conn, title)
        for text, rel in memberships:
            add_project_object(conn, project.id, ids[text], relationship=rel)
        for kind, text in notes:
            add_project_note(conn, project.id, text, kind=kind)
    with db.connection() as conn:
        return project, ids, get_project_context(conn, project.id)


def _placements(proposal):
    """Map object_id -> (section heading, relationship) for every emitted link."""
    out = {}
    for section in proposal.sections:
        for link in section.evidence:
            out[link.object_id] = (section.heading, link.relationship)
    return out


def test_every_membership_is_placed_somewhere(test_db):
    """No workspace evidence may be silently dropped from the outline.

    `background`, `question`, and `gap` memberships map to `qualification`, which the
    previous slicing logic never reached: it only ever built `supporting` and
    `counter` lists, so those items vanished.
    """
    memberships = [
        ("Data centers lowered Virginia rates between 2015 and 2024.", "evidence"),
        ("A resident describes her bill rising this summer.", "counterargument"),
        ("Background on how utilities recover fixed costs.", "background"),
        ("Which party actually bears the increased cost?", "question"),
        ("No state-level rate data exists after 2024.", "gap"),
        ("Another supporting datapoint.", "supporting"),
    ]
    project, ids, context = _project_with(test_db, memberships)
    proposal = deterministic_outline(context)
    placed = _placements(proposal)

    missing = sorted(set(ids.values()) - set(placed))
    assert missing == [], (
        "these workspace memberships were dropped from the outline entirely: " + ", ".join(missing)
    )


def test_qualification_memberships_carry_qualification_relationship(test_db):
    """A background/gap/question membership must be linked as a qualification."""
    memberships = [
        ("Main supporting claim.", "evidence"),
        ("A second supporting claim.", "evidence"),
        ("A third supporting claim.", "evidence"),
        ("Background context on utility cost recovery.", "background"),
        ("Which party bears the cost?", "question"),
        ("No post-2024 rate data exists for this state.", "gap"),
    ]
    project, ids, context = _project_with(test_db, memberships)
    proposal = deterministic_outline(context)
    placed = _placements(proposal)

    for text, rel in memberships:
        if rel in {"background", "question", "gap"}:
            heading, relationship = placed[ids[text]]
            assert relationship == "qualification", (
                f"{rel!r} membership landed as {relationship!r} in {heading!r}"
            )


def test_counterargument_is_not_filed_under_supporting(test_db):
    """The one membership explicitly marked counter must not read as supporting."""
    memberships = [
        ("Rates fell according to the causal study.", "evidence"),
        ("A separate confirming source.", "evidence"),
        ("Residents report bills climbing.", "counterargument"),
        ("A fourth supporting item.", "evidence"),
    ]
    project, ids, context = _project_with(test_db, memberships)
    placed = _placements(deterministic_outline(context))

    heading, relationship = placed[ids["Residents report bills climbing."]]
    assert relationship == "counterevidence", (
        f"counterargument membership was filed as {relationship!r} in {heading!r}"
    )


def test_supporting_items_are_not_duplicated_across_sections(test_db):
    """With two or fewer supporting items the old slicing emitted them twice."""
    for count in (1, 2):
        text_map = [(f"Supporting item number {i}.", "evidence") for i in range(count)]
        project, ids, context = _project_with(
            test_db, text_map, title=f"Duplication check with {count} items"
        )
        proposal = deterministic_outline(context)

        seen = [link.object_id for s in proposal.sections for link in s.evidence]
        repeated = {oid for oid in seen if seen.count(oid) > 1}
        assert repeated == set(), (
            f"with {count} supporting item(s), these were emitted in multiple "
            f"sections: {sorted(repeated)}"
        )


def test_unresolved_needs_are_not_repeated_in_every_section(test_db):
    """Each unresolved need should appear once, not once per needing section."""
    memberships = [("A claim.", "evidence")]
    notes = [
        ("gap", "No state-level rate data after 2024."),
        ("gap", "No water-use figures for these facilities."),
        ("counterargument", "The causal study predates the current buildout."),
    ]
    project, ids, context = _project_with(test_db, memberships, notes)
    proposal = deterministic_outline(context)

    needs = [n for s in proposal.sections for n in s.unresolved_research_needs]
    counts = {n: needs.count(n) for n in set(needs)}
    over_repeated = {n: c for n, c in counts.items() if c > 1}
    assert not over_repeated, f"needs repeated across sections: {over_repeated}"
