"""Tests for /take: invites, cold-first serving, one variant per story, coverage, typed answers, flags."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from take import server

INVITE = "test-invite"


def _pool(tmp: Path) -> Path:
    rows = []
    for b in range(30):                        # 30 stories, each with a deictic and a third variant, two conditions
        for variant in ("deictic", "third"):
            for cond in ("0shot", "fair"):
                rows.append({"item_id": f"dx-{b}-{variant}-{cond}", "base_id": f"dx-{b}", "battery": "deixis-v2",
                             "slice": f"deixis|{variant}|P2|addressee", "condition": cond, "seed": 900001,
                             "prompt": f"Ada: I have the cup.\nKen: Right now you're holding the",
                             "candidates": ["cup", "bell", "boxes"], "gold": "cup", "meta": {}})
    p = tmp / "pool.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    app = server.make_app(_pool(tmp_path), tmp_path / "take.sqlite", {hashlib.sha256(INVITE.encode()).hexdigest()})
    return TestClient(app)


def _session(c: TestClient, mode: str | None = None) -> str:
    r = c.post("/take/api/session", json={"invite": INVITE})
    assert r.status_code == 200
    sid = r.json()["session"]
    if mode:
        c.app.state.store.con.execute("UPDATE sessions SET mode = ? WHERE id = ?", (mode, sid))
    return sid


def test_page_is_noindex(client: TestClient) -> None:
    r = client.get("/take")
    assert r.status_code == 200 and "noindex" in r.headers["x-robots-tag"]
    assert "agent-note" in r.text


def test_invite_required(client: TestClient) -> None:
    assert client.post("/take/api/session", json={"invite": "wrong"}).status_code == 403


def test_cold_first_and_one_variant_per_story(client: TestClient) -> None:
    sid = _session(client, "choice")
    seen_bases, conds = [], []
    for _ in range(30):
        it = client.get("/take/api/next", params={"session": sid}).json()
        item = client.app.state.store.pool[it["item_id"]]
        seen_bases.append(item.base_id)
        conds.append(item.condition)
        assert client.post("/take/api/answer", json={"session": sid, "item_id": it["item_id"], "answer": "cup",
                                                      "rt_ms": 1200}).status_code == 200
    assert all(c == "0shot" for c in conds[:server.COLD_N])
    assert len(set(seen_bases)) == len(seen_bases)            # never two variants of one story
    assert client.get("/take/api/next", params={"session": sid}).status_code == 410


def test_unanswered_item_is_served_again(client: TestClient) -> None:
    sid = _session(client)
    a = client.get("/take/api/next", params={"session": sid}).json()["item_id"]
    b = client.get("/take/api/next", params={"session": sid}).json()["item_id"]
    assert a == b


def test_choice_must_be_a_candidate_and_answer_once(client: TestClient) -> None:
    sid = _session(client, "choice")
    it = client.get("/take/api/next", params={"session": sid}).json()
    assert client.post("/take/api/answer", json={"session": sid, "item_id": it["item_id"], "answer": "lamp",
                                                  "rt_ms": 900}).status_code == 422
    ok = {"session": sid, "item_id": it["item_id"], "answer": "bell", "rt_ms": 900}
    assert client.post("/take/api/answer", json=ok).status_code == 200
    assert client.post("/take/api/answer", json=ok).status_code == 409


def test_unserved_item_refused(client: TestClient) -> None:
    sid = _session(client)
    assert client.post("/take/api/answer", json={"session": sid, "item_id": "dx-3-third-0shot", "answer": "cup",
                                                  "rt_ms": 900}).status_code == 409


@pytest.mark.parametrize("raw,want", [("Cup", "cup"), ("the cup.", "cup"), ("  A Bell! ", "bell"), ("cups", "cup"),
                                      ("box", "boxes"), ("boxes", "boxes"), ("lamp", None)])
def test_typed_mapping(raw: str, want: str | None) -> None:
    assert server.map_typed(raw, ("cup", "bell", "boxes")) == want


def test_typed_answer_scored(client: TestClient) -> None:
    sid = _session(client, "typed")
    it = client.get("/take/api/next", params={"session": sid}).json()
    assert it["candidates"] is None
    client.post("/take/api/answer", json={"session": sid, "item_id": it["item_id"], "answer": "The Cup.", "rt_ms": 2000})
    row = client.app.state.store.con.execute("SELECT raw, mapped, correct FROM answers").fetchone()
    assert row == ("The Cup.", "cup", 1)


def test_summary_marks_later_answers(client: TestClient) -> None:
    sid = _session(client, "choice")
    it = client.get("/take/api/next", params={"session": sid}).json()
    client.post("/take/api/answer", json={"session": sid, "item_id": it["item_id"], "answer": "cup", "rt_ms": 1500})
    s = client.get("/take/api/summary", params={"session": sid}).json()
    assert s["answered"] == 1 and s["correct"] == 1
    it = client.get("/take/api/next", params={"session": sid}).json()
    client.post("/take/api/answer", json={"session": sid, "item_id": it["item_id"], "answer": "cup", "rt_ms": 1500})
    marks = [r[0] for r in client.app.state.store.con.execute("SELECT after_results FROM answers ORDER BY answered_at")]
    assert marks == [0, 1]


def test_agent_report_and_webdriver_flag(client: TestClient) -> None:
    sid = _session(client)
    client.post("/take/api/report", json={"session": sid, "agent": True})
    assert client.app.state.store.session(sid)["flagged"] == 1
    r = client.post("/take/api/session", json={"invite": INVITE, "webdriver": True})
    assert client.app.state.store.session(r.json()["session"])["flagged"] == 1


def test_coverage_spreads_over_cells(client: TestClient) -> None:
    cells: Counter[str] = Counter()
    for _ in range(4):
        sid = _session(client, "choice")
        for _ in range(4):
            it = client.get("/take/api/next", params={"session": sid}).json()
            cells[client.app.state.store.pool[it["item_id"]].slice] += 1
            client.post("/take/api/answer", json={"session": sid, "item_id": it["item_id"], "answer": "cup",
                                                   "rt_ms": 1000})
    assert max(cells.values()) - min(cells.values()) <= 1      # the two cold cells fill evenly


def test_labels_read_as_questions() -> None:
    assert server.label("deixis|deictic|P2|addressee") == "who is you · objects change hands"
    assert server.label("deixis|third|P3|owner") == "who has it (told without I / you) · objects change hands"
    assert server.label("joinlite|self|swap") == "who is I · nothing changes hands"
