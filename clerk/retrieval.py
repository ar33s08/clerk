"""Hybrid retrieval: BM25 + dense (+ optional visual lane), fused with RRF,
then an optional LLM reranker over the fused top set.

RRF (reciprocal rank fusion) is the production-grade default: robust to score
scale mismatch between lanes, embarrassingly parallel, no training.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from .config import settings
from .embedder import embed_query
from . import store

_STOP = set("a an and are as at be by for from has have he in is it its of on or that the to was were will with".split())


def _toks(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9\-']+", text.lower()) if w not in _STOP][:400]


class BM25:
    """Tiny in-memory Okapi BM25 (k1=1.4, b=0.75)."""

    def __init__(self, corpus_toks: list[list[str]]):
        self.k1, self.b = 1.4, 0.75
        self.doc_toks = corpus_toks
        self.df: Counter = Counter()
        for toks in corpus_toks:
            self.df.update(set(toks))
        self.N = len(corpus_toks)
        self.avgdl = (sum(map(len, corpus_toks)) / max(self.N, 1))
        self.idf = {t: math.log(1 + (self.N - d + 0.5) / (d + 0.5)) for t, d in self.df.items()}

    def search(self, q_toks: list[str], k: int) -> list[tuple[int, float]]:
        scores: dict[int, float] = {}
        for t in q_toks:
            idf = self.idf.get(t)
            if not idf:
                continue
            for i, toks in enumerate(self.doc_toks):
                f = toks.count(t)
                if not f:
                    continue
                dl = len(toks)
                scores[i] = scores.get(i, 0.0) + idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return sorted(scores.items(), key=lambda x: -x[1])[:k]


class Index:
    """Holds chunk rows + derived lanes. Rebuilt (cheap) or cached by caller."""

    def __init__(self):
        self.rows = store.all_chunks()
        if not self.rows:
            raise RuntimeError("empty index — run `python -m clerk.build_index` first")
        self.matrix = None
        good = [r for r in self.rows if r["embedding"] is not None]
        if good:
            import numpy as np
            self.matrix = np.stack([r["embedding"] for r in good])
            self._dense_ids = [r["chunk_id"] for r in good]
        self.bm = BM25([_toks(r["text"]) for r in self.rows])
        self._ids = [r["chunk_id"] for r in self.rows]

    def dense(self, q_emb, k: int) -> list[tuple[int, float]]:
        if self.matrix is None:
            return []
        sims = self.matrix @ q_emb            # normalized -> cosine
        idx = sims.argsort()[::-1][:k]
        return [(self._dense_ids[i], float(sims[i])) for i in idx]

    def lexical(self, question: str, k: int) -> list[tuple[int, float]]:
        return [(self._ids[i], s) for i, s in self.bm.search(_toks(question), k)]


def rrf_fuse(lanes: list[list[tuple[int, float]]], k: int, rrf_k: int) -> list[int]:
    votes: dict[int, float] = {}
    for lane in lanes:
        for rank, (cid, _) in enumerate(lane):
            votes[cid] = votes.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
    return [cid for cid, _ in sorted(votes.items(), key=lambda x: -x[1])[:k]]


def llm_rerank(question: str, rows: list[dict], k: int) -> list[dict]:
    """Pointwise relevance rerank (HF text-ranking task via LLM judge).
    Cheap variant: single batch call ranking candidate ids 0-1."""
    if not rows or len(rows) <= k:
        return rows
    from . import llm
    listing = "\n\n".join(
        f"[{i}] {r['text'][:220]}" for i, r in enumerate(rows))
    try:
        out = llm.chat_json([
            {"role": "system", "content":
             'Rank candidates for answering the question. Return JSON '
             '{"ranking": [i, ...]} best-first, all indices included.'},
            {"role": "user", "content": f"Q: {question}\n\n{listing}"},
        ], max_tokens=300)
        order = [int(i) for i in out.get("ranking", []) if 0 <= int(i) < len(rows)]
    except Exception:
        return rows
    order += [i for i in range(len(rows)) if i not in set(order)]
    return [rows[i] for i in order]


def retrieve(question: str, k: int | None = None, exclude_chunks: set[int] | None = None) -> list[dict]:
    """Returns top-k chunk dicts, privilege-quarantined chunks excluded."""
    k = k or settings.final_k
    idx = _index()
    lanes = [idx.lexical(question, settings.top_k_per_lane),
             idx.dense(embed_query(question), settings.top_k_per_lane)]
    cids = rrf_fuse(lanes, k * 4, settings.rrf_k)
    if exclude_chunks:
        cids = [c for c in cids if c not in exclude_chunks]
    out = []
    for cid in cids:
        row = store.chunk_by_id(cid)
        if row:
            out.append(row)
    if settings.rerank:
        out = llm_rerank(question, out, k)
    return out[:k]


_index_cache = None


def _index() -> Index:
    global _index_cache
    if _index_cache is None:
        _index_cache = Index()
    return _index_cache


def invalidate_index() -> None:
    global _index_cache
    _index_cache = None
