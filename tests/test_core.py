"""Offline unit tests — no network, no GPU. Run in CI on every push."""
import os
import sys

import numpy as np
import pytest

os.environ["CLERK_DATA_DIR"] = os.path.join(os.path.dirname(__file__), "_test_data")
os.environ["CLERK_LLM_BASE_URL"] = "http://127.0.0.1:1/v1"  # guaranteed unreachable

import clerk.store as store
from clerk.chunking import chunk_document
from clerk.engine import validate_quotes, _norm
from clerk.privilege import stage1_screen


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    from clerk.config import Settings
    s = Settings(data_dir=str(tmp_path))
    monkeypatch.setattr("clerk.config.settings", s)
    for mod in ("clerk.store", "clerk.eval"):
        monkeypatch.setattr(f"{mod}.settings", s)
    store.init_db(s.db_path)
    yield


def test_chunking_provenance_and_overlap():
    text = " ".join(f"Sentence number {i} about the party's contract dispute in 20{10+i%13}." for i in range(120))
    chunks = chunk_document(text, target_words=60)
    assert len(chunks) > 3
    ords = [c["ordinal"] for c in chunks]
    assert ords == sorted(ords)
    assert all(c["page"] >= 1 for c in chunks)
    # overlap: some sentence appears in consecutive chunks
    assert any(chunks[i]["text"][-40:] in chunks[i + 1]["text"] or
               chunks[i + 1]["text"][:40] in chunks[i]["text"] for i in range(len(chunks) - 1))


def test_span_validator_rejects_paraphrase():
    ev = {0: "The court held that the privilege was waived by disclosure to a third party."}
    good = [{"evidence_id": 0, "sentence": "Waiver occurred.",
             "quote": "the privilege was waived by disclosure to a third party"}]
    assert validate_quotes(good, ev)[0]["evidence_id"] == 0
    bad = [{"evidence_id": 0, "sentence": "x", "quote": "the privilege ended when they told someone"}]
    with pytest.raises(ValueError):
        validate_quotes(bad, ev)
    ghost = [{"evidence_id": 7, "sentence": "x", "quote": "the court held that"}]
    with pytest.raises(ValueError):
        validate_quotes(ghost, ev)


def test_span_validator_unicode_normalization():
    ev = {0: "The Court\u2019s holding was clear \u2014 privilege applies."}
    ok = [{"evidence_id": 0, "sentence": "s", "quote": "Court's holding was clear - privilege applies"}]
    assert validate_quotes(ok, ev)


def test_privilege_surface_gate():
    hit, score = stage1_screen("This memo is privileged and confidential attorney-client communication.")
    assert hit and score >= 0.6
    hit2, score2 = stage1_screen("The company reported revenue of $4.2 billion last quarter.")
    assert not hit2 and score2 == 0.0


def test_engine_refuses_without_llm(monkeypatch):
    """LLM down + evidence present => fail-closed error, never a fabricated answer."""
    from clerk import engine, retrieval
    doc = {"doc_id": "abc123", "source": "test", "title": "t", "doc_date": "2024-01-01",
           "text": "The court held that attorney-client privilege protects communications "
                   "made for the purpose of obtaining legal advice from disclosure. " * 6}
    store.upsert_document(doc)
    from clerk.chunking import chunk_document as cd
    from clerk.embedder import embed_texts
    chunks = cd(doc["text"], target_words=60)
    vecs = embed_texts([c["text"] for c in chunks])
    store.insert_chunks(doc["doc_id"], chunks, vecs)
    retrieval.invalidate_index()
    r = engine.answer("What does the court hold about privilege?")
    # fail-closed contract: with the LLM gate unavailable, privileged-suspect
    # chunks are quarantined (refused_privileged) or we error — never a pass-through.
    assert r["status"] in ("error", "refused_validation_failed",
                           "refused_no_evidence", "refused_privileged")
    assert "attorney-client privilege protects" not in r["answer"]  # no hallucinated pass-through


def test_api_health_and_auth():
    from fastapi.testclient import TestClient
    from clerk import api
    api._TOKEN = "abc123xyz"
    c = TestClient(api.app)
    assert c.get("/healthz").json() == {"ok": True}
    assert c.post("/ask", json={"question": "what is the privilege standard here?"}).status_code == 401
    assert c.get("/stats", headers={"Authorization": "Bearer " + api._TOKEN}).status_code in (200, 503)
