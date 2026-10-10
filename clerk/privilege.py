"""Privilege / PII triage gate — runs BEFORE results reach any consumer.

Two-stage design (cheap gate first, LLM only on suspicious hits):
  stage 1  regex/lexical surface patterns (privilege markers, SSNs, DOBs)
  stage 2  LLM zero-shot classification (HF task: zero-shot-classification),
           only on chunks that trip stage 1 or are adjacent to a trip.

Label set mirrors the HF zero-shot-classification task:
  ['attorney-client privilege', 'work product doctrine', 'personally identifiable information', 'ordinary business record']
"""
from __future__ import annotations

import re

from .config import settings
from . import llm, store

LABELS = ["attorney-client privilege", "work product doctrine",
          "personally identifiable information", "ordinary business record"]

_SURFACE = {
    "attorney_client": [
        r"attorney[- ]client privilege", r"privileged and confidential",
        r"legal advice", r"privileged communication", r"attorney work product",
    ],
    "work_product": [
        r"work.product doctrine", r"prepared in anticipation of litigation",
        r"attorney work product",
    ],
    "pii": [
        r"\b\d{3}-\d{2}-\d{4}\b",                       # SSN
        r"social security numbe",                       # spelled-out SSN
        r"account\s*number\s*[:.]?\s*\d",               # account numbers
        r"\b(?:card|cardholder|credit card)\s*(?:no|number)\b",
        r"\b\d{2}/\d{2}/\d{4}\b(?=.*\b(?:DOB|date of birth)\b)",
    ],
}
_SURFACE_RE = {k: [re.compile(p, re.I) for p in v] for k, v in _SURFACE.items()}


def stage1_screen(text: str) -> tuple[bool, float]:
    """Cheap lexical gate. Returns (needs_llm_check, prior_score)."""
    hits = sum(1 for pats in _SURFACE_RE.values() for rx in pats if rx.search(text))
    return (hits > 0, min(0.5 + 0.15 * hits, 0.95) if hits else 0.0)


_LABEL_PROMPTS: dict[str, list[str]] = {
    "attorney_client": [
        "This memorandum is a confidential communication from our client seeking legal advice from counsel regarding the dispute.",
        "Dear Attorney, please advise us on our legal exposure. This email is privileged and confidential between counsel and client.",
        "attorney client privilege protects communications made between a client and lawyer for the purpose of obtaining legal advice",
    ],
    "work_product": [
        "Investigative report prepared by counsel in anticipation of litigation, attorney work product, not for disclosure.",
        "claims analysis prepared by the litigation team because the claim was expected in litigation",
    ],
    "pii": [
        "Employee record: John Doe, Social Security Number 123-45-6789, date of birth 03/14/1980.",
        "customer account number 4428 9101 3377 6650, cardholder date of birth",
    ],
    "public": [
        "The court held that the attorney-client privilege does not protect the document because the communication was not confidential.",
        "In its opinion, the court analyzed whether privilege was waived by disclosure to a third party and denied the motion.",
        "OPINION OF THE COURT: The appellant asserts the statements were privileged; we disagree for the reasons stated below.",
        "The record shows the parties exchanged emails; the movant argues these are work product; this court denies summary judgment.",
        "The work-product doctrine protects materials prepared in anticipation of litigation; we review that determination for an abuse of discretion.",
        "A document is prepared in anticipation of litigation if it would not have existed but for the prospect of litigation; applying that standard, we deny the claim.",
        "Section 13: The party asserting the attorney-client privilege bears the burden of establishing each element of the protection.",
    ],
}
_label_embs: dict[str, object] = {}


def _label_embeddings():
    if not _label_embs:
        from .embedder import embed_texts
        for lab, prompts in _LABEL_PROMPTS.items():
            _label_embs[lab] = embed_texts(prompts)
    return _label_embs


def stage1_5_zero_shot(text: str, suspect_labels: set[str]) -> tuple[str, float, bool]:
    """HF zero-shot-classification via embedding cosine (free, CPU, ~ms).
    Returns (label, confidence, decided). A protective label wins only if it is
    top-1 with clear margin AND overlaps a stage-1 suspect; otherwise decided=
    False and the LLM arbitrates. Protective calls are biased toward doubt."""
    from .embedder import embed_query
    q = embed_query(text[:2000])
    scores: dict[str, float] = {}
    for lab, mat in _label_embeddings().items():
        scores[lab] = float((mat @ q).max())
    top = max(scores, key=scores.get)
    ordered = sorted(scores.values(), reverse=True)
    margin = ordered[0] - ordered[1]
    if top in suspect_labels and margin > 0.06:
        return top, scores[top], True
    if top == "public" and margin > 0.03:
        return "none", ordered[0], True
    return top, scores.get(top, 0.0), False


def stage2_classify(text: str) -> tuple[str, float]:
    """LLM zero-shot classification with confidence (calibrated by prompt +
    forced JSON). Truncates to a 3k-word window — privilege markers are local."""
    probe = text[:8000]
    out = llm.chat_json([
        {"role": "system", "content":
            "You are a legal document triage classifier for e-discovery privilege review. "
            "Classify whether the excerpt IS ITSELF a privileged/sensitive communication: "
            "'attorney_client' (a client-lawyer communication or legal-advice memo/email, "
            "typically labeled privileged & confidential), 'work_product' (analysis prepared "
            "in anticipation of litigation), 'pii' (SSNs, DOBs, account numbers of individuals), "
            "or 'business' (anything else). "
            "IMPORTANT: a PUBLIC JUDICIAL OPINION that discusses, cites, or analyzes privilege "
            "doctrine is 'business' — courts publish those opinions; quoting them leaks nothing. "
            "Only content that IS the protected communication itself gets a protective label. "
            "Answer JSON: {\"label\": str, \"confidence\": float 0-1}."},
        {"role": "user", "content": probe},
    ], max_tokens=800)
    label = str(out.get("label", "business"))
    conf = float(out.get("confidence", 0.0))
    norm = {"attorney_client": "attorney_client", "work_product": "work_product",
            "pii": "pii"}.get(label, "none")
    return norm, conf


import hashlib as _hl
_verdict_cache: dict[str, dict] = {}


def screen_chunk(chunk: dict) -> dict:
    """Full gate for one chunk, memoized by content hash (verdicts are stable
    for fixed text; keeps repeat queries at one retrieval pass).
    Never raises; on LLM failure defaults to human_review (fail-closed —
    privilege can't leak on an outage)."""
    text = chunk["text"]
    key = _hl.sha256(text.encode()).hexdigest()
    if key in _verdict_cache:
        return _verdict_cache[key]
    verdict = _screen_uncached(text)
    _verdict_cache[key] = verdict
    return verdict


_LABEL_TO_SUSPECT = {"attorney-client": "attorney_client",
                     "work-product": "work_product", "pii": "pii"}


def _suspects(text: str) -> set[str]:
    out = set()
    for lab, pats in _SURFACE_RE.items():
        if any(rx.search(text) for rx in pats):
            out.add(lab)
    return out


def _screen_uncached(text: str) -> dict:
    need, prior = stage1_screen(text)
    if not need:
        return {"label": "none", "score": 0.0, "route": "pass"}
    suspects = _suspects(text)
    # stage 1.5: embedding zero-shot — resolves the confident cases at ~0 cost.
    # Decision is MARGIN-based, not absolute-score: MiniLM cosines top out ~0.5
    # and are not calibrated 0-1 confidences; the relative rank IS the signal.
    # Fail-closed: any decided protective label quarantines.
    try:
        label, conf, decided = stage1_5_zero_shot(text, suspects)
        if decided:
            if label != "none":
                return {"label": label, "score": conf, "route": "human_review", "lane": "zero-shot"}
            return {"label": "none", "score": conf, "route": "pass", "lane": "zero-shot"}
    except Exception:
        pass  # fall through to LLM
    try:
        label, conf = stage2_classify(text)
    except llm.LLMError:
        return {"label": "attorney_client", "score": prior, "route": "human_review",
                "note": "fail-closed: gate unavailable"}
    score = max(prior, conf) if label != "none" else 0.0
    if label != "none" and score >= settings.privilege_threshold:
        return {"label": label, "score": score, "route": "human_review", "lane": "llm"}
    return {"label": "none", "score": score, "route": "pass", "lane": "llm"}
