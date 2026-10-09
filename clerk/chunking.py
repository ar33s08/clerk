"""Layout-aware-ish sentence chunking with provenance (ordinal + page marker).

CourtListener text has no pages; we synthesize stable page markers by
word-count bands so citations always resolve to (doc, marker) pairs that can
be rendered for a reviewer. For real PDFs, `page` comes from extraction.
"""
from __future__ import annotations

import re

_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def sentence_split(text: str) -> list[str]:
    parts: list[str] = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        parts.extend(s.strip() for s in _SENT_RE.split(para) if s.strip())
    return parts


def chunk_document(text: str, target_words: int = 380, overlap_sentences: int = 2,
                   words_per_page: int = 520) -> list[dict]:
    """Sentence-packed chunks with sentence overlap and page-marker inference."""
    sents = sentence_split(text)
    chunks: list[dict] = []
    i, ordinal = 0, 0
    n = len(sents)
    while i < n:
        window, words = [], 0
        j = i
        while j < n and words < target_words:
            window.append(sents[j])
            words += len(sents[j].split())
            j += 1
        body = " ".join(window)
        start_word = sum(len(s.split()) for s in sents[:i])
        chunks.append({
            "ordinal": ordinal,
            "page": start_word // words_per_page + 1,   # inferred marker
            "text": body,
            "n_tokens": len(body.split()),
        })
        ordinal += 1
        if j >= n:
            break
        i = max(i + 1, j - overlap_sentences)          # overlap for boundary recall
    return chunks


def chunk_pages(pages: list[dict], target_words: int = 380,
                overlap_sentences: int = 2) -> list[dict]:
    """Same sentence-packing but with REAL page numbers (from PDF extraction).
    Chunks never cross a page boundary, keeping quote provenance exact."""
    chunks: list[dict] = []
    ordinal = 0
    for pg in pages:
        sents = sentence_split(pg["text"])
        i, n = 0, len(sents)
        while i < n:
            window, words, j = [], 0, i
            while j < n and words < target_words:
                window.append(sents[j])
                words += len(sents[j].split())
                j += 1
            body = " ".join(window)
            chunks.append({"ordinal": ordinal, "page": pg["page"], "text": body,
                           "n_tokens": len(body.split())})
            ordinal += 1
            if j >= n:
                break
            i = max(i + 1, j - overlap_sentences)
    return chunks
