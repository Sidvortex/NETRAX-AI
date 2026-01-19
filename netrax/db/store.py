"""SQLite store (stdlib sqlite3): people, face embeddings, events.

Everything lives in `data/netrax.db`; photos and snapshots are files under
`data/` and the DB keeps paths relative to it. Migrations are numbered SQL
files applied in order, tracked with `PRAGMA user_version`.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

MIGRATIONS = Path(__file__).with_name("migrations")
ROLES = ("owner", "allowed", "unknown")


class Store:
    def __init__(self, data_dir: str | Path, filename: str = "netrax.db"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.data_dir / filename
        self._lock = threading.RLock()
        self.db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.version = 0   # bumps whenever people/embeddings change (recognizers reload on change)
        self.migrate()

    # schema ------------------------------------------------------------------
    def schema_version(self) -> int:
        return self.db.execute("PRAGMA user_version").fetchone()[0]

    def migrate(self) -> int:
        with self._lock:
            current = self.schema_version()
            for f in sorted(MIGRATIONS.glob("*.sql")):
                n = int(f.name.split("_", 1)[0])
                if n <= current:
                    continue
                self.db.executescript("BEGIN;\n" + f.read_text() + f"\nPRAGMA user_version = {n};\nCOMMIT;")
                current = n
            return current

    def close(self) -> None:
        with self._lock:
            self.db.close()

    def abs(self, rel: str | None) -> Path | None:
        return (self.data_dir / rel) if rel else None

    def rel(self, p: Path) -> str:
        return str(Path(p).resolve().relative_to(self.data_dir.resolve()))

    def person_dir(self, person_id: int) -> Path:
        d = self.data_dir / "people" / str(person_id)
        d.mkdir(parents=True, exist_ok=True)
        return d

    # people ------------------------------------------------------------------
    def add_person(self, name: str, role: str = "allowed") -> int:
        name = _clean_name(name)
        _check_role(role)
        now = time.time()
        with self._lock:
            try:
                cur = self.db.execute("INSERT INTO people(name, role, created_at, updated_at) VALUES (?,?,?,?)",
                                      (name, role, now, now))
            except sqlite3.IntegrityError:
                raise ValueError(f"a person called {name!r} already exists") from None
            self.version += 1
            return cur.lastrowid

    def get_person(self, person_id: int) -> dict | None:
        row = self.db.execute("SELECT * FROM people WHERE id=?", (person_id,)).fetchone()
        return dict(row) if row else None

    def find_person(self, name: str) -> dict | None:
        row = self.db.execute("SELECT * FROM people WHERE name=? COLLATE NOCASE", (name.strip(),)).fetchone()
        return dict(row) if row else None

    def list_people(self) -> list[dict]:
        rows = self.db.execute("""
            SELECT p.*, COUNT(e.id) AS n_faces,
                   (SELECT photo_path FROM face_embeddings WHERE person_id=p.id AND photo_path IS NOT NULL
                    ORDER BY quality DESC, id LIMIT 1) AS cover
            FROM people p LEFT JOIN face_embeddings e ON e.person_id = p.id
            GROUP BY p.id ORDER BY CASE p.role WHEN 'owner' THEN 0 WHEN 'allowed' THEN 1 ELSE 2 END, p.name
        """).fetchall()
        return [dict(r) for r in rows]

    def update_person(self, person_id: int, name: str | None = None, role: str | None = None) -> dict:
        p = self.get_person(person_id)
        if p is None:
            raise KeyError(person_id)
        name = _clean_name(name) if name is not None else p["name"]
        role = role if role is not None else p["role"]
        _check_role(role)
        with self._lock:
            try:
                self.db.execute("UPDATE people SET name=?, role=?, updated_at=? WHERE id=?",
                                (name, role, time.time(), person_id))
            except sqlite3.IntegrityError:
                raise ValueError(f"a person called {name!r} already exists") from None
            self.version += 1
        return self.get_person(person_id)

    def delete_person(self, person_id: int) -> bool:
        with self._lock:
            cur = self.db.execute("DELETE FROM people WHERE id=?", (person_id,))
            self.version += 1
        shutil.rmtree(self.data_dir / "people" / str(person_id), ignore_errors=True)
        return cur.rowcount > 0

    def touch_seen(self, person_id: int, ts: float | None = None) -> None:
        with self._lock:
            self.db.execute("UPDATE people SET last_seen=? WHERE id=?", (ts or time.time(), person_id))

    # embeddings --------------------------------------------------------------
    def add_embedding(self, person_id: int, vec: np.ndarray, photo_path: str | Path | None = None,
                      quality: float | None = None) -> int:
        v = np.asarray(vec, np.float32).ravel()
        v = v / (np.linalg.norm(v) + 1e-9)
        rel = self.rel(photo_path) if photo_path else None
        with self._lock:
            cur = self.db.execute(
                "INSERT INTO face_embeddings(person_id, embedding, dim, photo_path, quality, created_at) "
                "VALUES (?,?,?,?,?,?)", (person_id, v.tobytes(), v.size, rel, quality, time.time()))
            self.version += 1
            return cur.lastrowid

    def all_embeddings(self) -> tuple[np.ndarray, np.ndarray]:
        """(person_ids[N], matrix[N, D]) of every stored embedding."""
        rows = self.db.execute("SELECT person_id, embedding, dim FROM face_embeddings ORDER BY id").fetchall()
        if not rows:
            return np.zeros(0, np.int64), np.zeros((0, 128), np.float32)
        ids = np.array([r["person_id"] for r in rows], np.int64)
        mat = np.stack([np.frombuffer(r["embedding"], np.float32, r["dim"]) for r in rows])
        return ids, mat

    def list_photos(self, person_id: int) -> list[dict]:
        rows = self.db.execute("SELECT id, photo_path, quality, created_at FROM face_embeddings "
                               "WHERE person_id=? ORDER BY id", (person_id,)).fetchall()
        return [dict(r) for r in rows]

    def delete_embedding(self, emb_id: int) -> bool:
        with self._lock:
            row = self.db.execute("SELECT photo_path FROM face_embeddings WHERE id=?", (emb_id,)).fetchone()
            if not row:
                return False
            self.db.execute("DELETE FROM face_embeddings WHERE id=?", (emb_id,))
            self.version += 1
        if row["photo_path"]:
            (self.data_dir / row["photo_path"]).unlink(missing_ok=True)
        return True

    # events ------------------------------------------------------------------
    def add_event(self, type: str, label: str, ts: float | None = None, person_id: int | None = None,
                  track_id: int | None = None, confidence: float | None = None, data: dict | None = None,
                  snapshot_path: str | Path | None = None) -> int:
        rel = self.rel(snapshot_path) if snapshot_path else None
        with self._lock:
            cur = self.db.execute(
                "INSERT INTO events(ts, type, label, person_id, track_id, confidence, data, snapshot_path) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (ts or time.time(), type, label, person_id, track_id, confidence,
                 json.dumps(data, separators=(",", ":")) if data else None, rel))
            return cur.lastrowid

    def query_events(self, types: list[str] | None = None, since: float | None = None, until: float | None = None,
                     person_id: int | None = None, label: str | None = None, limit: int = 100,
                     offset: int = 0) -> tuple[list[dict], int]:
        where, args = [], []
        if types:
            where.append(f"type IN ({','.join('?' * len(types))})")
            args += types
        if since is not None:
            where.append("ts >= ?")
            args.append(since)
        if until is not None:
            where.append("ts < ?")
            args.append(until)
        if person_id is not None:
            where.append("person_id = ?")
            args.append(person_id)
        if label:
            where.append("label LIKE ?")
            args.append(f"%{label}%")
        w = ("WHERE " + " AND ".join(where)) if where else ""
        total = self.db.execute(f"SELECT COUNT(*) FROM events {w}", args).fetchone()[0]
        rows = self.db.execute(f"SELECT * FROM events {w} ORDER BY ts DESC, id DESC LIMIT ? OFFSET ?",
                               args + [limit, offset]).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["data"] = json.loads(d["data"]) if d["data"] else {}
            out.append(d)
        return out, total

    def event_counts(self, since: float | None = None) -> dict[str, int]:
        q = "SELECT type, COUNT(*) FROM events" + (" WHERE ts >= ?" if since else "") + " GROUP BY type"
        return {t: n for t, n in self.db.execute(q, (since,) if since else ()).fetchall()}

    def prune_events(self, before_ts: float) -> int:
        """Delete events older than `before_ts` and their snapshot files."""
        with self._lock:
            rows = self.db.execute("SELECT snapshot_path FROM events WHERE ts < ? AND snapshot_path IS NOT NULL",
                                   (before_ts,)).fetchall()
            n = self.db.execute("DELETE FROM events WHERE ts < ?", (before_ts,)).rowcount
        for r in rows:
            (self.data_dir / r[0]).unlink(missing_ok=True)
        return n

    def clear_events(self) -> int:
        return self.prune_events(float("inf"))


def _clean_name(name: str) -> str:
    name = " ".join(str(name).split())
    if not name or len(name) > 64:
        raise ValueError("name must be 1-64 characters")
    if name.lower() in ("unknown", "..."):
        raise ValueError(f"{name!r} is reserved")
    return name


def _check_role(role: str) -> None:
    if role not in ROLES:
        raise ValueError(f"role must be one of {', '.join(ROLES)}")
