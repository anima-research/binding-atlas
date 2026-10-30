"""/take: the human reader for the binding batteries. One item at a time, from a pool of fresh draws.

A visitor arrives with an invite token, gets an anonymous session (a random id kept in the browser) and a response
mode drawn at random for the session: `choice` (the model's candidate set, shuffled) or `typed` (a free answer,
normalized and mapped to the candidates). Each request serves the next item:
  * cold first: the first COLD_N answers of a session are 0-shot items; worked-example items come after;
  * one variant per story: an item whose base story the session has already seen is never served;
  * coverage: among the allowed cells (slice × condition), the one with the fewest answers from unflagged sessions
    is served, and within it the item with the fewest answers.
No per-item feedback is given. A session can ask for its own summary at any time; answers after that are marked.

Nothing identifying is stored: no IP address, no account. A session stores its invite token's hash, its mode, the
browser's `navigator.webdriver` flag and the visitor's own answer to "is an automated agent answering?".

Run:  TAKE_POOL=pool.jsonl TAKE_DB=take.sqlite TAKE_INVITES=invites.txt uvicorn take.server:app --port 7480
(`invites.txt`: one sha256 hex digest of an invite token per line.)
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sqlite3
import threading
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

COLD_N = 16                       # answers that must be 0-shot before worked examples appear
MIN_GAP_S = 0.25                  # an answer faster than this after its item was served is flagged
MAX_SESSIONS_PER_INVITE_DAY = 200
PAGE = Path(__file__).resolve().parent / "static" / "take.html"
Mode = Literal["choice", "typed"]


@dataclass(frozen=True)
class Item:
    item_id: str
    base_id: str
    battery: str
    slice: str
    condition: str
    seed: int
    prompt: str
    candidates: tuple[str, ...]
    gold: str

    @property
    def cell(self) -> tuple[str, str]:
        return (self.slice, self.condition)


def load_pool(path: Path) -> dict[str, Item]:
    pool: dict[str, Item] = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        it = Item(r["item_id"], r["base_id"], r["battery"], r["slice"], r["condition"], int(r["seed"]), r["prompt"],
                  tuple(r["candidates"]), r["gold"])
        if it.gold not in it.candidates:
            raise ValueError(f"{it.item_id}: gold not among candidates")
        pool[it.item_id] = it
    if not pool:
        raise ValueError(f"empty pool: {path}")
    return pool


_PUNCT = re.compile(r"[^\w\s'-]")


def normalize(text: str) -> str:
    """Case, outer whitespace and punctuation, and a leading article."""
    t = _PUNCT.sub(" ", text.lower()).strip()
    t = re.sub(r"\s+", " ", t)
    return re.sub(r"^(the|a|an) ", "", t)


def _forms(t: str) -> set[str]:
    """A word and its simple singulars (-s, -es)."""
    out = {t}
    if len(t) > 2 and t.endswith("s"):
        out.add(t[:-1])
    if len(t) > 3 and t.endswith("es"):
        out.add(t[:-2])
    return out


def map_typed(text: str, candidates: tuple[str, ...]) -> Optional[str]:
    """The candidate a typed answer names, or None: exact after normalizing, else one candidate sharing a simple
    singular with it (cup / cups, box / boxes). Two candidates sharing one → None."""
    t = normalize(text)
    exact = [c for c in candidates if normalize(c) == t]
    if exact:
        return exact[0]
    near = [c for c in candidates if _forms(normalize(c)) & _forms(t)]
    return near[0] if len(near) == 1 else None


_ASK = {"self": "who is I", "addressee": "who is you", "owner": "who has it", "third": "who has it (named)"}


def label(slice_: str) -> str:
    """A reader's name for a slice: what the item asks, and whether objects change hands in it."""
    parts = slice_.split("|")
    if parts[0] == "deixis":
        ask = "who has it (told without I / you)" if parts[1] == "third" else _ASK.get(parts[-1], parts[-1])
        return f"{ask} · objects change hands"
    if parts[0] == "joinlite":
        return f"{_ASK.get(parts[1], parts[1])} · nothing changes hands"
    return parts[0]


SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, created REAL, invite TEXT, mode TEXT, webdriver INTEGER,
  agent_report TEXT, viewed_results INTEGER DEFAULT 0, flagged INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS served (session TEXT, item_id TEXT, base_id TEXT, served_at REAL,
  PRIMARY KEY (session, item_id));
CREATE TABLE IF NOT EXISTS answers (session TEXT, item_id TEXT, battery TEXT, slice TEXT, condition TEXT, seed INTEGER,
  served_at REAL, answered_at REAL, rt_ms INTEGER, mode TEXT, raw TEXT, mapped TEXT, correct INTEGER,
  after_results INTEGER, fast INTEGER, PRIMARY KEY (session, item_id));
"""


class Store:
    def __init__(self, db: Path, pool: dict[str, Item], invites: set[str]) -> None:
        self.pool, self.invites = pool, invites
        self.lock = threading.RLock()        # one connection, used from FastAPI's worker threads
        self.con = sqlite3.connect(db, check_same_thread=False, isolation_level=None)
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.executescript(SCHEMA)
        self.by_cell: dict[tuple[str, str], list[Item]] = defaultdict(list)
        for it in pool.values():
            self.by_cell[it.cell].append(it)
        self.cell_n: Counter[tuple[str, str]] = Counter()
        self.item_n: Counter[str] = Counter()
        for item_id, n in self.con.execute(
                "SELECT a.item_id, COUNT(*) FROM answers a JOIN sessions s ON s.id = a.session "
                "WHERE s.flagged = 0 GROUP BY a.item_id"):
            if item_id in pool:
                self.item_n[item_id] = n
                self.cell_n[pool[item_id].cell] += n

    def new_session(self, invite: str, webdriver: bool) -> tuple[str, Mode]:
        h = hashlib.sha256(invite.encode()).hexdigest()
        if h not in self.invites:
            raise HTTPException(403, "unknown invite")
        day = time.time() - 86400
        with self.lock:
            (n,) = self.con.execute("SELECT COUNT(*) FROM sessions WHERE invite = ? AND created > ?", (h, day)).fetchone()
            if n >= MAX_SESSIONS_PER_INVITE_DAY:
                raise HTTPException(429, "too many sessions for this invite today")
            sid, mode = uuid.uuid4().hex, random.choice(("choice", "typed"))
            self.con.execute("INSERT INTO sessions (id, created, invite, mode, webdriver, flagged) VALUES (?,?,?,?,?,?)",
                             (sid, time.time(), h, mode, int(webdriver), int(webdriver)))
        return sid, mode  # type: ignore[return-value]

    def session(self, sid: str) -> dict[str, Any]:
        with self.lock:
            cur = self.con.execute("SELECT * FROM sessions WHERE id = ?", (sid,))
            row = cur.fetchone()
        if row is None:
            raise HTTPException(404, "unknown session")
        return dict(zip([d[0] for d in cur.description], row))

    def next_item(self, sid: str) -> tuple[Item, int]:
        """(item, answers so far). An item served and not yet answered is served again rather than a new one."""
        self.session(sid)
        with self.lock:
            served = self.con.execute("SELECT item_id, base_id FROM served WHERE session = ?", (sid,)).fetchall()
            answered = {r[0] for r in self.con.execute("SELECT item_id FROM answers WHERE session = ?", (sid,))}
            pending = [i for i, _ in served if i not in answered and i in self.pool]
            if pending:
                return self.pool[pending[0]], len(answered)
            seen_bases = {b for _, b in served}
            conds = {"0shot"} if len(answered) < COLD_N else {"0shot", "fair"}
            open_cells = [c for c, its in self.by_cell.items() if c[1] in conds
                          and any(it.base_id not in seen_bases for it in its)]
            if not open_cells:
                raise HTTPException(410, "no items left for this session")
            low = min(self.cell_n[c] for c in open_cells)
            cell = random.choice([c for c in open_cells if self.cell_n[c] == low])
            cands = [it for it in self.by_cell[cell] if it.base_id not in seen_bases]
            least = min(self.item_n[it.item_id] for it in cands)
            item = random.choice([it for it in cands if self.item_n[it.item_id] == least])
            self.con.execute("INSERT INTO served VALUES (?,?,?,?)", (sid, item.item_id, item.base_id, time.time()))
        return item, len(answered)

    def answer(self, sid: str, item_id: str, raw: str, rt_ms: int) -> int:
        s = self.session(sid)
        item = self.pool.get(item_id)
        if item is None:
            raise HTTPException(404, "unknown item")
        with self.lock:
            row = self.con.execute("SELECT served_at FROM served WHERE session = ? AND item_id = ?",
                                   (sid, item_id)).fetchone()
            if row is None:
                raise HTTPException(409, "item was not served to this session")
            if self.con.execute("SELECT 1 FROM answers WHERE session = ? AND item_id = ?", (sid, item_id)).fetchone():
                raise HTTPException(409, "already answered")
            now = time.time()
            mapped = raw if s["mode"] == "choice" else map_typed(raw, item.candidates)
            if s["mode"] == "choice" and raw not in item.candidates:
                raise HTTPException(422, "not a candidate")
            fast = int(now - row[0] < MIN_GAP_S)
            self.con.execute("INSERT INTO answers VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                             (sid, item_id, item.battery, item.slice, item.condition, item.seed, row[0], now,
                              max(0, int(rt_ms)), s["mode"], raw[:200], mapped, int(mapped == item.gold),
                              int(s["viewed_results"]), fast))
            if not s["flagged"]:
                self.item_n[item_id] += 1
                self.cell_n[item.cell] += 1
            (n,) = self.con.execute("SELECT COUNT(*) FROM answers WHERE session = ?", (sid,)).fetchone()
        return n

    def summary(self, sid: str) -> dict[str, Any]:
        self.session(sid)
        with self.lock:
            self.con.execute("UPDATE sessions SET viewed_results = 1 WHERE id = ?", (sid,))
            rows = self.con.execute("SELECT battery, slice, condition, correct FROM answers WHERE session = ?",
                                    (sid,)).fetchall()
        groups: defaultdict[str, list[int]] = defaultdict(list)
        for battery, sl, cond, ok in rows:
            groups[f"{label(sl)} · {'cold' if cond == '0shot' else 'with examples'}"].append(ok)
        return {"answered": len(rows), "correct": sum(r[3] for r in rows),
                "groups": {k: {"n": len(v), "correct": sum(v)} for k, v in sorted(groups.items())}}

    def report(self, sid: str, agent: Optional[bool], note: str) -> None:
        self.session(sid)
        val = json.dumps({"agent": agent, "note": note[:500]})
        with self.lock:
            self.con.execute("UPDATE sessions SET agent_report = ?, flagged = CASE WHEN ? THEN 1 ELSE flagged END "
                             "WHERE id = ?", (val, int(bool(agent)), sid))


def _load_invites(path: Optional[str]) -> set[str]:
    if not path:
        return set()
    return {ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip() and not ln.startswith("#")}


class SessionIn(BaseModel):
    invite: str = Field(min_length=1, max_length=200)
    webdriver: bool = False


class AnswerIn(BaseModel):
    session: str = Field(min_length=1, max_length=64)
    item_id: str = Field(min_length=1, max_length=200)
    answer: str = Field(min_length=1, max_length=200)
    rt_ms: int = Field(ge=0, le=86_400_000)


class ReportIn(BaseModel):
    session: str = Field(min_length=1, max_length=64)
    agent: Optional[bool] = None
    note: str = Field(default="", max_length=500)


def make_app(pool_path: Path, db: Path, invites: set[str]) -> FastAPI:
    store = Store(db, load_pool(pool_path), invites)
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.state.store = store

    @app.get("/take")
    def page() -> FileResponse:
        return FileResponse(PAGE, media_type="text/html; charset=utf-8",
                            headers={"Cache-Control": "no-store", "X-Robots-Tag": "noindex, nofollow"})

    @app.post("/take/api/session")
    def session(body: SessionIn) -> dict[str, Any]:
        sid, mode = store.new_session(body.invite, body.webdriver)
        return {"session": sid, "mode": mode, "cold_n": COLD_N}

    @app.get("/take/api/next")
    def next_(session: str) -> dict[str, Any]:
        item, done = store.next_item(session)
        mode = store.session(session)["mode"]
        cands = list(item.candidates)
        random.Random(f"{session}:{item.item_id}").shuffle(cands)
        return {"item_id": item.item_id, "prompt": item.prompt, "mode": mode,
                "candidates": cands if mode == "choice" else None, "answered": done,
                "with_examples": item.condition != "0shot"}

    @app.post("/take/api/answer")
    def answer(body: AnswerIn) -> dict[str, Any]:
        return {"answered": store.answer(body.session, body.item_id, body.answer, body.rt_ms)}

    @app.get("/take/api/summary")
    def summary(session: str) -> dict[str, Any]:
        return store.summary(session)

    @app.post("/take/api/report")
    def report(body: ReportIn) -> dict[str, bool]:
        store.report(body.session, body.agent, body.note)
        return {"ok": True}

    return app


def _app_from_env() -> FastAPI:
    pool, db = os.environ.get("TAKE_POOL"), os.environ.get("TAKE_DB")
    if not pool or not db:
        raise RuntimeError("set TAKE_POOL and TAKE_DB (and TAKE_INVITES)")
    return make_app(Path(pool), Path(db), _load_invites(os.environ.get("TAKE_INVITES")))


app = _app_from_env() if os.environ.get("TAKE_POOL") else FastAPI()
