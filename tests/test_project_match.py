"""Project evidence by meaning: find with embeddings, judge with a project question."""

import datetime
import json

import numpy as np
import pytest
from typer.testing import CliRunner

from edward.cli import app
from edward.db import Database
from edward.models import generate_id
from edward.services import project_match
from edward.services.embed import get_configured_embedding_model, serialize_vector
from edward.services.project_match import match_projects
from edward.services.projects import add_project_object, create_project
from edward.services.themes import parse_since, recent_digest

DIM = 16
AXES = {"rates": 0, "jobs": 1, "garden": 2}
NOW = "2026-09-20T12:00:00+00:00"


def _vec(axis: str, jitter: int, lean: float = 0.0) -> list[float]:
    """Near one axis; ``lean`` tilts it toward the rates axis so rank order is controllable."""
    rng = np.random.default_rng(jitter)
    v = rng.normal(0, 0.02, DIM)
    v[AXES[axis]] += 1.0
    v[AXES["rates"]] += lean
    return (v / np.linalg.norm(v)).tolist()


def _add(
    conn,
    capture_id: str,
    axis: str,
    *,
    title: str | None = None,
    jitter: int = 0,
    lean: float = 0.0,
    namespace: str = "web",
    resource_id: str | None = None,
) -> str:
    conn.execute(
        """
        INSERT INTO captures (id, origin_namespace, collection_channel, collector, acquisition_method,
                              retrieved_at, is_deleted, created_at, updated_at)
        VALUES (?, ?, 'test', 'test', 'manual', ?, 0, ?, ?);
        """,
        (capture_id, namespace, NOW, NOW, NOW),
    )
    rid = resource_id or f"res_{capture_id}"
    if not conn.execute("SELECT 1 FROM resources WHERE id = ?;", (rid,)).fetchone():
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'unreviewed', 0, ?, ?);
            """,
            (
                rid,
                f"url:{rid}",
                f"https://example.com/{rid}",
                title or f"{axis} {capture_id}",
                NOW,
                NOW,
            ),
        )
        conn.execute(
            """
            INSERT INTO embeddings (id, object_type, object_id, model, dimensions, embedding_blob, input_hash, created_at)
            VALUES (?, 'resource', ?, ?, ?, ?, ?, ?);
            """,
            (
                generate_id("emb"),
                rid,
                get_configured_embedding_model(),
                DIM,
                serialize_vector(_vec(axis, jitter, lean)),
                rid,
                NOW,
            ),
        )
    conn.execute(
        "INSERT INTO capture_resources (capture_id, resource_id, relationship_type, created_at) VALUES (?, ?, 'primary', ?);",
        (capture_id, rid, NOW),
    )
    return rid


def rates_embedder(text: str) -> list[float]:
    return _vec("rates", 999)


class FakeJudge:
    """Relevant when the page title mentions 'bill'; records what it was sent."""

    def __init__(self, fail_after: int | None = None):
        self.calls: list[str] = []
        self.fail_after = fail_after

    def evaluate_questions(self, text, questions, data_class="public_web"):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise RuntimeError("rate limited (429)")
        self.calls.append(text)
        p = 0.9 if "bill" in text else 0.2
        return {"answers": {questions[0]["id"]: {"noul": p}}, "cost": 0.0001}


@pytest.fixture
def library(test_db: Database):
    """A rates project plus 6 rates pages (3 about bills), 6 jobs pages, 6 garden pages."""
    with test_db.transaction() as conn:
        project = create_project(
            conn,
            title="Do data centers raise your electric bill?",
            brief="Testing whether data center load drives residential rates.",
        )
        for i in range(6):
            _add(
                conn,
                f"cap_rates_{i}",
                "rates",
                jitter=i,
                title=f"Utility bill study {i}" if i < 3 else f"Grid capacity note {i}",
            )
            _add(conn, f"cap_jobs_{i}", "jobs", jitter=10 + i, lean=0.3)
            _add(conn, f"cap_garden_{i}", "garden", jitter=20 + i)
    return test_db, project.id


def _objects(db, project_id) -> dict[str, tuple[str, str]]:
    with db.connection() as conn:
        return {
            r["object_id"]: (r["membership_status"], r["added_by"])
            for r in conn.execute(
                "SELECT * FROM project_objects WHERE project_id = ?;", (project_id,)
            )
        }


def _outcomes(db, project_id) -> dict[str, str]:
    with db.connection() as conn:
        return {
            r["capture_id"]: r["outcome"]
            for r in conn.execute(
                "SELECT * FROM project_matches WHERE project_id = ?;", (project_id,)
            )
        }


def test_first_pass_finds_by_meaning_and_keeps_only_relevant(library):
    db, pid = library
    judge = FakeJudge()
    result = match_projects(db, [pid], limit=8, judge=judge, embedder=rates_embedder)

    (summary,) = result["projects"]
    assert summary["mode"] == "full" and summary["judged"] == 8 and summary["checked"] == 18
    assert {s["object_id"] for s in summary["suggested"]} == {
        f"res_cap_rates_{i}" for i in range(3)
    }
    objects = _objects(db, pid)
    assert all(v == ("candidate", "system") for v in objects.values())
    outcomes = _outcomes(db, pid)
    assert outcomes["cap_rates_4"] == "not-relevant"
    assert outcomes["cap_garden_0"] == "below-rank"
    assert len(outcomes) == 18


def test_second_pass_checks_only_new_captures(library, monkeypatch):
    db, pid = library
    monkeypatch.setattr(project_match, "SUGGEST_RANK", 8)
    match_projects(db, [pid], limit=8, judge=FakeJudge(), embedder=rates_embedder)
    with db.transaction() as conn:
        _add(conn, "cap_new_bill", "rates", jitter=77, title="New electric bill analysis")
        _add(conn, "cap_new_garden", "garden", jitter=78)
    judge = FakeJudge()

    (summary,) = match_projects(db, [pid], judge=judge, embedder=rates_embedder)["projects"]

    assert summary["mode"] == "new" and summary["checked"] == 2
    assert len(judge.calls) == 1 and "New electric bill" in judge.calls[0]
    assert [s["capture_id"] for s in summary["suggested"]] == ["cap_new_bill"]
    assert match_projects(db, [pid], judge=judge, embedder=rates_embedder)["projects"] == []


def test_human_decisions_are_never_changed_or_resuggested(library):
    db, pid = library
    with db.transaction() as conn:
        add_project_object(
            conn, pid, "res_cap_rates_0", membership_status="accepted", added_by="human"
        )
        add_project_object(
            conn, pid, "res_cap_rates_1", membership_status="rejected", added_by="human"
        )
    judge = FakeJudge()
    match_projects(db, [pid], limit=8, judge=judge, embedder=rates_embedder)

    objects = _objects(db, pid)
    assert objects["res_cap_rates_0"] == ("accepted", "human")
    assert objects["res_cap_rates_1"] == ("rejected", "human")
    assert not any("Utility bill study 1" in text for text in judge.calls)
    assert _outcomes(db, pid)["cap_rates_1"] == "member"


def test_page_shared_by_several_captures_is_suggested_once(library):
    db, pid = library
    with db.transaction() as conn:
        _add(
            conn, "cap_dup_a", "rates", title="Shared bill page", resource_id="res_shared", jitter=5
        )
        _add(conn, "cap_dup_b", "rates", resource_id="res_shared")
    judge = FakeJudge()
    (summary,) = match_projects(db, [pid], limit=12, judge=judge, embedder=rates_embedder)[
        "projects"
    ]

    assert [s["object_id"] for s in summary["suggested"]].count("res_shared") == 1
    assert sum("Shared bill page" in text for text in judge.calls) == 1


def test_private_captures_are_never_sent_to_the_judge(library):
    db, pid = library
    with db.transaction() as conn:
        _add(
            conn,
            "cap_mail",
            "rates",
            title="Email about my electric bill",
            jitter=3,
            namespace="gmail",
        )
    judge = FakeJudge()
    (summary,) = match_projects(db, [pid], limit=8, judge=judge, embedder=rates_embedder)[
        "projects"
    ]

    assert not any("Email about my electric bill" in text for text in judge.calls)
    mail = [s for s in summary["suggested"] if s["capture_id"] == "cap_mail"]
    assert mail and mail[0]["relevance"] is None  # close enough to suggest unjudged
    with db.connection() as conn:
        note = conn.execute(
            "SELECT relevance_note FROM project_objects WHERE object_id = 'res_cap_mail';"
        ).fetchone()["relevance_note"]
    assert "private content stays local" in note


def test_without_a_judge_only_the_closest_are_suggested(library, monkeypatch):
    db, pid = library
    monkeypatch.setattr(project_match, "UNJUDGED_RANK", 4)
    (summary,) = match_projects(db, [pid], limit=8, use_judge=False, embedder=rates_embedder)[
        "projects"
    ]

    assert len(summary["suggested"]) == 4
    assert all(s["relevance"] is None for s in summary["suggested"])


def test_changing_the_brief_rechecks_everything(library):
    db, pid = library
    match_projects(db, [pid], limit=8, judge=FakeJudge(), embedder=rates_embedder)
    with db.transaction() as conn:
        conn.execute(
            "UPDATE projects SET description = 'A new angle on residential rates.' WHERE id = ?;",
            (pid,),
        )
    (summary,) = match_projects(db, [pid], limit=8, judge=FakeJudge(), embedder=rates_embedder)[
        "projects"
    ]
    assert summary["mode"] == "full" and summary["checked"] == 18


def test_judge_failure_leaves_unjudged_items_for_the_next_pass(library):
    db, pid = library
    (summary,) = match_projects(
        db, [pid], limit=8, judge=FakeJudge(fail_after=2), embedder=rates_embedder
    )["projects"]
    assert summary["judged"] == 2 and summary["deferred"] == 6 and "429" in summary["error"]

    judge = FakeJudge()
    (again,) = match_projects(db, [pid], judge=judge, embedder=rates_embedder)["projects"]
    assert again["mode"] == "new" and len(judge.calls) == 6
    assert {s["object_id"] for s in summary["suggested"] + again["suggested"]} == {
        f"res_cap_rates_{i}" for i in range(3)
    }


def test_recent_tells_you_what_was_suggested(library):
    db, pid = library
    match_projects(db, [pid], limit=8, judge=FakeJudge(), embedder=rates_embedder)
    now = datetime.datetime(2026, 9, 21, 9, 0, tzinfo=datetime.UTC)
    with db.connection() as conn:
        digest = recent_digest(conn, *parse_since("7d", now=now), now=now)
    items = {i["capture_id"]: i for day in digest["days"] for i in day["items"]}
    assert items["cap_rates_0"]["suggested_for"] == ["Do data centers raise your electric bill?"]
    assert items["cap_garden_0"]["suggested_for"] == []


def test_cli_project_suggest_and_process_hook(library, monkeypatch):
    db, pid = library
    monkeypatch.setenv("EDWARD_DB_PATH", str(db.db_path))
    monkeypatch.setattr(project_match, "_default_embedder", rates_embedder)
    monkeypatch.setattr(project_match, "get_judge", lambda: FakeJudge())
    runner = CliRunner()

    res = runner.invoke(app, ["project", "suggest", pid, "--limit", "8", "--json"])
    assert res.exit_code == 0, res.output
    payload = json.loads(res.stdout)
    assert payload["judge"] is True and len(payload["projects"][0]["suggested"]) == 3

    with db.transaction() as conn:
        _add(conn, "cap_later", "rates", jitter=55, title="Later bill story")
    processed = runner.invoke(app, ["process", "--limit", "1", "--json"])
    assert processed.exit_code == 0, processed.output
    matches = json.loads(processed.stdout)["project_matches"]["projects"]
    assert [s["capture_id"] for s in matches[0]["suggested"]] == ["cap_later"]

    missing = runner.invoke(app, ["project", "suggest", "prj_missing", "--json"])
    assert missing.exit_code == 1
