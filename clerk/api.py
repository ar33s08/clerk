"""FastAPI service: /ask, /documents, /quarantine, /stats, /healthz.

Auth: bearer token from CLERK_API_TOKEN (fail-closed: unset token => 503 on
data routes; /healthz stays open for probes).
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from . import engine, retrieval, store
from .config import settings

@asynccontextmanager
async def lifespan(_app):
    """Build lanes at boot, not on the first user request (BM25 is O(N) build)."""
    try:
        retrieval._index()
    except RuntimeError:
        pass  # empty db; /ask will 503 with a clear message
    yield


app = FastAPI(title="CLERK", version="0.1.0", lifespan=lifespan,
              description="Cited Legal Evidence Retrieval & Review Kit")

_TOKEN = os.environ.get("CLERK_API_TOKEN", "")


def auth(authorization: str = Header(default="")):
    if not _TOKEN:
        raise HTTPException(503, "server token not configured")
    if authorization != f"Bearer {_TOKEN}":
        raise HTTPException(401, "invalid token")


class AskIn(BaseModel):
    question: str = Field(min_length=8, max_length=2000)
    k: int | None = Field(default=None, ge=1, le=20)


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/ask", dependencies=[Depends(auth)])
def ask(req: AskIn):
    try:
        return engine.answer(req.question, k=req.k)
    except RuntimeError as e:           # empty index
        raise HTTPException(503, str(e))


@app.get("/quarantine", dependencies=[Depends(auth)])
def quarantine(limit: int = 50):
    with store._conn() as c:
        rows = c.execute(
            "SELECT q.*, d.title FROM quarantine q LEFT JOIN documents d ON d.doc_id=q.doc_id "
            "ORDER BY q.qid DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


@app.get("/documents/{doc_id}", dependencies=[Depends(auth)])
def document(doc_id: str):
    meta = store.doc_meta(doc_id)
    if not meta:
        raise HTTPException(404, "unknown doc")
    return meta


@app.get("/stats", dependencies=[Depends(auth)])
def stats():
    return store.stats()
