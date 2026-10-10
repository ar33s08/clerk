"""The evidence-locked answer engine — CLERK's showpiece.

Contract: every sentence in the answer must cite a span [doc#page] that the
model quotes VERBATIM, and the server validates each quote byte-for-byte
(normalized) against the retrieved chunk before the answer is returned.
Any failing quote => hard refusal, never a silent pass. This is a *system*
guarantee (decode-time schema + server-side validation), not a prompt wish.

Flow:
  retrieve -> privilege gate -> constrained generation (JSON schema:
  answer sentences + quoted evidence) -> span validator -> answer | refusal.
"""
from __future__ import annotations

import re
import time

from . import llm, privilege, retrieval, store
from .config import settings

SYSTEM = (
    "You are CLERK, a legal evidence analyst. Rules:\n"
    "1) You may ONLY use the provided evidence excerpts. No outside knowledge.\n"
    "2) Every factual sentence must cite evidence via EVIDENCE-ID markers.\n"
    "3) For each cited sentence, copy a VERBATIM quote (10-40 words) from the "
    "evidence excerpt you rely on — exact words, no paraphrase, no ellipsis.\n"
    "4) If the evidence cannot answer the question, set answerable=false.\n"
    "5) Never restate privileged communications verbatim in your own words; "
    "if evidence is marked PRIVILEGED, do not answer from it.\n"
    "Answer JSON: {\"answerable\": bool, \"answer\": str, "
    "\"support\": [{\"evidence_id\": int, \"sentence\": str, \"quote\": str}]}"
)


def _norm(s: str) -> str:
    s = s.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2014", "-").replace("\u2013", "-")
    return re.sub(r"\s+", " ", s.lower()).strip(" .,;:")


def validate_quotes(support: list[dict], ev: dict[int, str]) -> list[dict]:
    """Byte-match (normalized) each quote against its source chunk.
    Returns validated citation list or raises ValueError on ANY mismatch."""
    cites = []
    for item in support:
        eid = int(item["evidence_id"])
        quote = item.get("quote", "")
        src = ev.get(eid)
        if src is None:
            raise ValueError(f"evidence_id {eid} not provided")
        if _norm(quote) not in _norm(src):
            raise ValueError(f"quote #{eid} not verbatim in source (possible hallucination)")
        if len(quote.split()) < 5:
            raise ValueError(f"quote #{eid} too short to be meaningful")
        cites.append({"evidence_id": eid, "sentence": item["sentence"], "quote": quote})
    return cites


def answer(question: str, k: int | None = None) -> dict:
    t0 = time.time()
    llm.reset_usage()
    chunks = retrieval.retrieve(question, k=k)

    # privilege gate (pre-exposure)
    evidence, blocked, excl = [], [], set()
    for i, ch in enumerate(chunks):
        verdict = privilege.screen_chunk(ch)
        if verdict["route"] == "human_review":
            store.log_quarantine(ch["chunk_id"], ch["doc_id"], verdict["label"],
                                 verdict["score"], "human_review")
            meta = store.doc_meta(ch["doc_id"]) or {}
            blocked.append({"doc_id": ch["doc_id"], "page": ch["page"],
                            "label": verdict["label"], "title": meta.get("title")})
        else:
            evidence.append((i, ch))

    ev_text = "\n\n".join(
        f"[EVIDENCE {i}] (doc={ch['doc_id']} page={ch['page']})\n{ch['text']}"
        for i, ch in evidence) or "(no evidence available)"

    if not evidence:
        status = "refused_privileged" if blocked else "refused_no_evidence"
        return _finish(question, "", status, [], blocked, t0)

    try:
        out = llm.chat_json([
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Evidence:\n{ev_text}\n\nQuestion: {question}"},
        ], max_tokens=3000)
    except llm.LLMError:
        return _finish(question, "", "error", [], blocked, t0,
                       note="llm_unavailable")

    if not out.get("answerable"):
        return _finish(question, "", "refused_no_evidence", [], blocked, t0)

    ev_map = {i: ch["text"] for i, ch in evidence}
    try:
        cites = validate_quotes(out.get("support", []), ev_map)
    except ValueError as e:
        # hard stop — one bad quote invalidates the whole answer
        return _finish(question, "", "refused_validation_failed", [], blocked, t0, note=str(e))

    # build cited answer: sentence+doc/page citation per claim
    cites_final = []
    parts = []
    for c in cites:
        ch = evidence[c["evidence_id"]][1]
        cites_final.append({"doc_id": ch["doc_id"], "page": ch["page"],
                            "quote": c["quote"], "sentence": c["sentence"]})
        parts.append(f"{c['sentence'].strip()} [{ch['doc_id'][:8]} p.{ch['page']}]")
    return _finish(question, " ".join(parts), "ok", cites_final, blocked, t0)


def _finish(question, ans, status, cites, blocked, t0, note=None) -> dict:
    u = llm.usage()
    cost = (u["prompt_tokens"] * settings.price_in_per_mtok
            + u["completion_tokens"] * settings.price_out_per_mtok) / 1e6
    qid = store.log_query(question, ans, status, cites, int((time.time() - t0) * 1000),
                          u["prompt_tokens"], u["completion_tokens"], cost)
    return {"qid": qid, "status": status, "answer": ans, "citations": cites,
            "quarantined": blocked, "note": note,
            "latency_ms": int((time.time() - t0) * 1000), "tokens": u,
            "cost_usd": round(cost, 6)}
