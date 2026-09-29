"""Themes (named groups of captures) and the recent-saves digest."""

import datetime
import json

import numpy as np
import pytest
from typer.testing import CliRunner

from edward.cli import app
from edward.db import Database
from edward.models import generate_id
from edward.services.embed import get_configured_embedding_model, serialize_vector
from edward.services.themes import (
    ThemeError,
    ThemeNamePayload,
    assign_new_captures,
    collect_naming_requests,
    generate_theme_name,
    list_themes,
    parse_since,
    rebuild_themes,
    recent_digest,
    refresh_themes,
    rename_theme,
    store_theme_name,
)

DIM = 32
GROUPS = {"garden": 0, "chips": 1, "sailing": 2}


def _direction(group: str, jitter: int) -> list[float]:
    """A unit vector near one of three orthogonal axes, so the right grouping is unambiguous."""
    rng = np.random.default_rng(jitter)
    v = rng.normal(0, 0.05, DIM)
    v[GROUPS[group]] += 1.0
    return (v / np.linalg.norm(v)).tolist()


def _add_capture(
    conn,
    capture_id: str,
    group: str,
    *,
    jitter: int = 0,
    namespace: str = "web",
    created_at: str = "2026-09-20T12:00:00+00:00",
    run_id: str | None = None,
    posted_at: str | None = None,
    embed: bool = True,
) -> None:
    conn.execute(
        """
        INSERT INTO captures (id, origin_namespace, collection_channel, collector, collector_run_id,
                              acquisition_method, retrieved_at, raw_content, is_deleted, created_at, updated_at)
        VALUES (?, ?, 'test', 'test', ?, 'manual', ?, NULL, 0, ?, ?);
        """,
        (capture_id, namespace, run_id, created_at, created_at, created_at),
    )
    res_id = f"res_{capture_id}"
    conn.execute(
        """
        INSERT INTO resources (id, identity_key, canonical_url, title, published_at, review_state, is_deleted, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, 'unreviewed', 0, ?, ?);
        """,
        (
            res_id,
            f"url:{res_id}",
            f"https://example.com/{res_id}",
            f"{group.title()} story {capture_id}",
            posted_at,
            created_at,
            created_at,
        ),
    )
    conn.execute(
        "INSERT INTO capture_resources (capture_id, resource_id, relationship_type, created_at) VALUES (?, ?, 'primary', ?);",
        (capture_id, res_id, created_at),
    )
    if embed:
        conn.execute(
            """
            INSERT INTO embeddings (id, object_type, object_id, model, dimensions, embedding_blob, input_hash, created_at)
            VALUES (?, 'resource', ?, ?, ?, ?, ?, ?);
            """,
            (
                generate_id("emb"),
                res_id,
                get_configured_embedding_model(),
                DIM,
                serialize_vector(_direction(group, jitter)),
                res_id,
                created_at,
            ),
        )


def _seed(conn, per_group: int = 5) -> None:
    for group in GROUPS:
        for i in range(per_group):
            _add_capture(conn, f"cap_{group}_{i}", group, jitter=hash((group, i)) % 1000)


def _membership(conn) -> dict[str, str]:
    return {
        r["capture_id"]: r["theme_id"]
        for r in conn.execute("SELECT capture_id, theme_id FROM theme_members;")
    }


class FakeNamer:
    location = "local"
    provider = "llm"
    base_url = "http://localhost:11434/v1"
    model = "fake"

    def __init__(self):
        self.prompts: list[str] = []

    def chat_completion(self, messages, response_model=None, data_class="public_web", **kwargs):
        self.prompts.append(messages[-1]["content"])
        payload = ThemeNamePayload(name=f"Theme {len(self.prompts)}", description="About things.")
        return json.dumps(payload.model_dump()), payload


# --- Grouping ---


def test_rebuild_recovers_the_obvious_groups(test_db: Database):
    with test_db.transaction() as conn:
        _seed(conn)
        result = rebuild_themes(conn, k=3)
        members = _membership(conn)

    assert result == {"themes": 3, "created": 3, "kept": 0, "removed": 0, "captures": 15}
    for group in GROUPS:
        assert len({members[f"cap_{group}_{i}"] for i in range(5)}) == 1
    assert len(set(members.values())) == 3


def test_new_captures_join_nearest_theme_without_moving_others(test_db: Database):
    with test_db.transaction() as conn:
        _seed(conn)
        rebuild_themes(conn, k=3)
        before = _membership(conn)
        _add_capture(conn, "cap_new_sail", "sailing", jitter=4242)
        assert refresh_themes(conn) == {"mode": "assign", "assigned": 1}
        after = _membership(conn)
        counts = {
            r["id"]: r["member_count"] for r in conn.execute("SELECT id, member_count FROM themes;")
        }

    assert {k: v for k, v in after.items() if k != "cap_new_sail"} == before
    assert after["cap_new_sail"] == before["cap_sailing_0"]
    assert counts[before["cap_sailing_0"]] == 6


def test_rebuild_keeps_ids_and_names_of_surviving_themes(test_db: Database):
    with test_db.transaction() as conn:
        _seed(conn)
        rebuild_themes(conn, k=3)
        garden_theme = _membership(conn)["cap_garden_0"]
        rename_theme(conn, garden_theme, "My Garden")
        result = rebuild_themes(conn, k=3)
        name = conn.execute(
            "SELECT name, name_source FROM themes WHERE id = ?;", (garden_theme,)
        ).fetchone()

    assert result["kept"] == 3 and result["created"] == 0
    assert _membership_of(test_db, "cap_garden_1") == garden_theme
    assert (name["name"], name["name_source"]) == ("My Garden", "human")


def _membership_of(db: Database, capture_id: str) -> str:
    with db.connection() as conn:
        return _membership(conn)[capture_id]


def test_human_named_theme_survives_even_when_its_group_disappears(test_db: Database):
    with test_db.transaction() as conn:
        _seed(conn)
        rebuild_themes(conn, k=3)
        theme = _membership(conn)["cap_chips_0"]
        rename_theme(conn, theme, "Chip Stuff")
        conn.execute("UPDATE captures SET is_deleted = 1 WHERE id LIKE 'cap_chips_%';")
        rebuild_themes(conn, k=2)
        row = conn.execute(
            "SELECT name, member_count FROM themes WHERE id = ?;", (theme,)
        ).fetchone()

    assert row["name"] == "Chip Stuff"
    assert row["member_count"] == 0


def test_test_namespace_and_unembedded_captures_are_excluded(test_db: Database):
    with test_db.transaction() as conn:
        _seed(conn)
        _add_capture(conn, "cap_probe", "garden", namespace="agent-test")
        _add_capture(conn, "cap_bare", "garden", embed=False)
        rebuild_themes(conn, k=3)
        members = _membership(conn)

    assert "cap_probe" not in members
    assert "cap_bare" not in members


def test_assign_without_themes_is_a_noop(test_db: Database):
    with test_db.transaction() as conn:
        _seed(conn)
        assert assign_new_captures(conn) == 0


# --- Naming ---


def test_model_names_new_themes_from_public_samples(test_db: Database):
    namer = FakeNamer()
    with test_db.transaction() as conn:
        _seed(conn)
        rebuild_themes(conn, k=3)
        requests = collect_naming_requests(conn)
    assert len(requests) == 3
    for req in requests:
        payload = generate_theme_name(req, namer)
        with test_db.transaction() as conn:
            assert store_theme_name(
                conn, req["theme_id"], payload.name, payload.description, "model", req["member_ids"]
            )

    with test_db.connection() as conn:
        names = {t["name"] for t in list_themes(conn)}
        assert collect_naming_requests(conn) == []  # model names are final
    assert names == {"Theme 1", "Theme 2", "Theme 3"}
    assert "Garden story" in "".join(namer.prompts)


def test_private_captures_never_reach_naming_samples(test_db: Database):
    with test_db.transaction() as conn:
        for i in range(5):
            _add_capture(conn, f"cap_mail_{i}", "garden", jitter=i, namespace="gmail")
        _add_capture(conn, "cap_pub", "garden", jitter=99)
        rebuild_themes(conn, k=1)
        (req,) = collect_naming_requests(conn)

    assert [s["title"] for s in req["samples"]] == ["Garden story cap_pub"]
    assert req["skipped_private"] == 5


def test_automated_naming_never_overwrites_a_human_name(test_db: Database):
    with test_db.transaction() as conn:
        _seed(conn)
        rebuild_themes(conn, k=3)
        (req, *_) = collect_naming_requests(conn)
        rename_theme(conn, req["theme_id"], "Mine")
        assert not store_theme_name(
            conn, req["theme_id"], "Model Name", None, "model", req["member_ids"]
        )
        name = conn.execute("SELECT name FROM themes WHERE id = ?;", (req["theme_id"],)).fetchone()[
            "name"
        ]
    assert name == "Mine"


def test_rename_unknown_theme_raises(test_db: Database):
    with test_db.transaction() as conn:
        with pytest.raises(ThemeError):
            rename_theme(conn, "thm_missing", "X")
        with pytest.raises(ThemeError):
            rename_theme(conn, "thm_missing", "   ")


# --- Recent ---

NOW = datetime.datetime(2026, 9, 28, 20, 0, tzinfo=datetime.timezone(datetime.timedelta(hours=-5)))


@pytest.mark.parametrize(
    ("text", "start", "end"),
    [
        ("today", "2026-09-28", None),
        ("yesterday", "2026-09-27", "2026-09-28"),
        ("week", "2026-09-28", None),  # a Monday
        ("last-week", "2026-09-21", "2026-09-28"),
        ("7d", "2026-09-22", None),
        ("2026-09-01", "2026-09-01", None),
    ],
)
def test_parse_since(text, start, end):
    s, e = parse_since(text, now=NOW)
    assert s.date().isoformat() == start
    assert (e.date().isoformat() if e else None) == end


def test_parse_since_rejects_garbage():
    with pytest.raises(ValueError):
        parse_since("sometime", now=NOW)


def test_recent_uses_post_date_for_bulk_backfills(test_db: Database):
    with test_db.transaction() as conn:
        # A 60-item backfill imported today, of posts from two weeks ago.
        for i in range(60):
            _add_capture(
                conn,
                f"cap_bulk_{i}",
                "sailing",
                jitter=i,
                created_at="2026-09-28T15:00:00+00:00",
                run_id="run_backfill",
                posted_at="2026-09-10T12:00:00.000Z",
            )
        # A small sync yesterday of an old post: the save date is the one you remember.
        _add_capture(
            conn,
            "cap_sync",
            "garden",
            created_at="2026-09-27T15:00:00+00:00",
            run_id="run_small",
            posted_at="2024-01-01T00:00:00.000Z",
        )
        digest = recent_digest(conn, *parse_since("7d", now=NOW), now=NOW)

    assert digest["total"] == 1
    item = digest["days"][0]["items"][0]
    assert (item["capture_id"], item["date_basis"]) == ("cap_sync", "saved")
    assert digest["days"][0]["label"] == "Yesterday"


def test_recent_groups_by_day_and_theme_without_writing(test_db: Database):
    with test_db.transaction() as conn:
        _seed(conn)
        rebuild_themes(conn, k=3)
        conn.execute(
            "UPDATE themes SET name = 'Sailing' WHERE id = (SELECT theme_id FROM theme_members WHERE capture_id = 'cap_sailing_0');"
        )
        _add_capture(conn, "cap_today", "sailing", jitter=7, created_at="2026-09-28T18:00:00+00:00")
        _add_capture(conn, "cap_mon", "sailing", jitter=8, created_at="2026-09-22T18:00:00+00:00")

    with test_db.connection() as conn:
        digest = recent_digest(conn, *parse_since("7d", now=NOW), now=NOW)
        stored = _membership(conn)

    assert digest["total"] == 2
    assert digest["themes"] == [
        {"theme_id": stored["cap_sailing_0"], "name": "Sailing", "count": 2}
    ]
    assert [d["label"] for d in digest["days"]] == ["Today", "Tuesday"]
    assert "cap_today" not in stored  # placed for display only


# --- CLI contract ---


def test_cli_themes_and_recent_json(test_db: Database, monkeypatch):
    monkeypatch.setenv("EDWARD_DB_PATH", str(test_db.db_path))
    with test_db.transaction() as conn:
        _seed(conn)
    runner = CliRunner()

    refreshed = runner.invoke(
        app, ["themes", "refresh", "--fallback-names", "--count", "3", "--json"]
    )
    assert refreshed.exit_code == 0, refreshed.output
    assert json.loads(refreshed.stdout)["themes"] == 3

    listed = runner.invoke(app, ["themes", "--json"])
    assert listed.exit_code == 0
    themes = json.loads(listed.stdout)["themes"]
    assert len(themes) == 3 and all(t["name_source"] == "fallback" for t in themes)

    renamed = runner.invoke(app, ["themes", "rename", themes[0]["theme_id"], "Boats", "--json"])
    assert json.loads(renamed.stdout)["name_source"] == "human"

    shown = runner.invoke(app, ["themes", "show", themes[0]["theme_id"], "--json"])
    assert json.loads(shown.stdout)["name"] == "Boats"

    recent = runner.invoke(app, ["recent", "--since", "2026-01-01", "--json"])
    assert recent.exit_code == 0
    assert json.loads(recent.stdout)["total"] == 15

    bad = runner.invoke(app, ["recent", "--since", "whenever", "--json"])
    assert bad.exit_code == 2
    assert bad.stdout == ""

    missing = runner.invoke(app, ["themes", "show", "thm_nope", "--json"])
    assert missing.exit_code == 1
