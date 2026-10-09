"""SQLite storage layer: documents, chunks, embeddings, eval runs.

SQLite is the production store for a single-engineer build; the schema maps
1:1 onto Postgres+pgvector (see docs/ARCHITECTURE.md for the migration note).
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager

import numpy as np

from .config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT PRIMARY KEY,       -- stable hash
    source      TEXT NOT NULL,          -- 'courtlistener', 'sec', 'upload'
    source_url  TEXT,
    title       TEXT,
    doc_date    TEXT,
    meta        TEXT,                   -- JSON
    ingested_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      TEXT NOT NULL REFERENCES documents(doc_id),
    ordinal     INTEGER NOT NULL,       -- chunk index within doc
    page        INTEGER,                -- best-effort page marker
    text        TEXT NOT NULL,
    n_tokens    INTEGER,
    embedding   BLOB,                   -- float32 little-endian
    UNIQUE (doc_id, ordinal)
);
CREATE TABLE IF NOT EXISTS page_images (
    doc_id TEXT NOT NULL,
    page   INTEGER NOT NULL,
    path   TEXT NOT NULL,              -- rendered thumbnail (visual lane)
    embedding BLOB,                     -- visual-lane embedding (pluggable)
    PRIMARY KEY (doc_id, page)
);
CREATE TABLE IF NOT EXISTS queries (
    qid         INTEGER PRIMARY KEY AUTOINCREMENT,
    question    TEXT NOT NULL,
    answer      TEXT,
    status      TEXT,                   -- ok | refused_privileged | refused_no_evidence | error
    citations   TEXT,                   -- JSON [{doc_id,page,span_hash,start,end}]
    latency_ms  INTEGER,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    created_at  TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS quarantine (
    qid       INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id  INTEGER REFERENCES chunks(chunk_id),
    doc_id    TEXT,
    label     TEXT,                     -- attorney_client | work_product | pii | none
    score     REAL,
    route     TEXT,                     -- human_review | dropped
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS eval_runs (
    run_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    git_sha    TEXT,
    metrics    TEXT,                    -- JSON
    passed     INTEGER,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
"""


def init_db(path: str | None = None) -> None:
    target = path or settings.db_path
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with _conn(target) as c:
        c.executescript(_SCHEMA)


@contextmanager
def _conn(path: str | None = None):
    con = sqlite3.connect(path or settings.db_path, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    try:
        yield con
        con.commit()
    finally:
        con.close()


def upsert_document(doc: dict) -> None:
    with _conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO documents (doc_id, source, source_url, title, doc_date, meta) "
            "VALUES (?,?,?,?,?,?)",
            (doc["doc_id"], doc["source"], doc.get("source_url"), doc.get("title"),
             doc.get("doc_date"), json.dumps(doc.get("meta", {}))),
        )


def insert_chunks(doc_id: str, chunks: list[dict], embeddings: np.ndarray | None) -> None:
    with _conn() as c:
        for i, ch in enumerate(chunks):
            blob = None
            if embeddings is not None:
                blob = embeddings[i].astype(np.float32).tobytes()
            c.execute(
                "INSERT OR REPLACE INTO chunks (doc_id, ordinal, page, text, n_tokens, embedding) "
                "VALUES (?,?,?,?,?,?)",
                (doc_id, ch["ordinal"], ch.get("page"), ch["text"], ch.get("n_tokens"), blob),
            )


def all_chunks() -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT chunk_id, doc_id, ordinal, page, text, embedding FROM chunks ORDER BY chunk_id"
        ).fetchall()
    out = []
    for r in rows:
        emb = np.frombuffer(r["embedding"], dtype=np.float32) if r["embedding"] else None
        out.append({"chunk_id": r["chunk_id"], "doc_id": r["doc_id"], "ordinal": r["ordinal"],
                    "page": r["page"], "text": r["text"], "embedding": emb})
    return out


def chunk_by_id(chunk_id: int) -> dict | None:
    with _conn() as c:
        r = c.execute(
            "SELECT chunk_id, doc_id, ordinal, page, text FROM chunks WHERE chunk_id=?",
            (chunk_id,)).fetchone()
    return dict(r) if r else None


def doc_meta(doc_id: str) -> dict | None:
    with _conn() as c:
        r = c.execute("SELECT * FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
    return dict(r) if r else None


def log_query(question, answer, status, citations, latency_ms, ptok, ctok) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO queries (question, answer, status, citations, latency_ms, prompt_tokens, completion_tokens) "
            "VALUES (?,?,?,?,?,?,?)",
            (question, answer, status, json.dumps(citations), latency_ms, ptok, ctok),
        )
        return cur.lastrowid


def log_quarantine(chunk_id, doc_id, label, score, route) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO quarantine (chunk_id, doc_id, label, score, route) VALUES (?,?,?,?,?)",
            (chunk_id, doc_id, label, score, route),
        )


def log_eval_run(git_sha, metrics: dict, passed: bool) -> int:
    with _conn() as c:
        cur = c.execute("INSERT INTO eval_runs (git_sha, metrics, passed) VALUES (?,?,?)",
                        (git_sha, json.dumps(metrics), int(passed)))
        return cur.lastrowid


def stats() -> dict:
    with _conn() as c:
        q = lambda sql: c.execute(sql).fetchone()[0]  # noqa: E731
        return {
            "documents": q("SELECT COUNT(*) FROM documents"),
            "chunks": q("SELECT COUNT(*) FROM chunks"),
            "queries": q("SELECT COUNT(*) FROM queries"),
            "quarantined": q("SELECT COUNT(*) FROM quarantine WHERE route='human_review'"),
            "eval_runs": q("SELECT COUNT(*) FROM eval_runs"),
        }
