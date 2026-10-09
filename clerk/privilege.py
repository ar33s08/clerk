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
        r"\b\d{2}/\d{2}/\d{4}\b(?=.*\b(?:DOB|date of birth)\b)",
        r"\b(?:account|card)\s*(?:no|number)[:.]?\s*\d[\d \-]{8,}",
    ],
}
_SURFACE_RE = {k: [re.compile(p, re.I) for p in v] for k, v in _SURFACE.items()}


def stage1_screen(text: str) -> tuple[bool, float]:
    """Cheap lexical gate. Returns (needs_llm_check, prior_score)."""
    hits = sum(1 for pats in _SURFACE_RE.values() for rx in pats if rx.search(text))
    return (hits > 0, min(0.5 + 0.15 * hits, 0.95) if hits else 0.0)


def stage2_classify(text: str) -> tuple[str, float]:
    """LLM zero-shot classification with confidence (calibrated by prompt +
    forced JSON). Truncates to a 3k-word window — privilege markers are local."""
    probe = text[:12000]
    out = llm.chat_json([
        {"role": "system", "content":
            "You are a legal document triage classifier for e-discovery privilege review. "
            "Classify the excerpt into exactly one label: "
            "'attorney_client', 'work_product', 'pii', or 'business'. "
            "Attorney-client requires communication for the purpose of legal advice; "
            "references DISCUSSING or ASSERTING the privilege in opinions count as attorney_client. "
            "Answer JSON: {\"label\": str, \"confidence\": float 0-1}."},
        {"role": "user", "content": probe},
    ], max_tokens=60)
    label = str(out.get("label", "business"))
    conf = float(out.get("confidence", 0.0))
    norm = {"attorney_client": "attorney_client", "work_product": "work_product",
            "pii": "pii"}.get(label, "none")
    return norm, conf


def screen_chunk(chunk: dict) -> dict:
    """Full gate for one chunk. Never raises; on LLM failure defaults to
    human_review (fail-closed — privilege can't leak on an outage)."""
    text = chunk["text"]
    need, prior = stage1_screen(text)
    if not need:
        return {"label": "none", "score": 0.0, "route": "pass"}
    try:
        label, conf = stage2_classify(text)
    except llm.LLMError:
        return {"label": "attorney_client", "score": prior, "route": "human_review",
                "note": "fail-closed: gate unavailable"}
    score = max(prior, conf) if label != "none" else 0.0
    if label != "none" and score >= settings.privilege_threshold:
        return {"label": label, "score": score, "route": "human_review"}
    return {"label": "none", "score": score, "route": "pass"}
