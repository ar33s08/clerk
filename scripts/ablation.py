"""Retrieval-lane ablation for docs/BENCHMARKS.md.

Usage: .venv/bin/python scripts/ablation.py [dense|bm25|hybrid]
Uses golden.jsonl gold_doc_ids; prints recall@5 for the given lane config.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clerk import store
from clerk.config import settings
from clerk.embedder import embed_query
from clerk.retrieval import _index, rrf_fuse


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "hybrid"
    store.init_db()
    idx = _index()
    rows = [json.loads(l) for l in open(os.path.join(settings.data_dir, "golden.jsonl")) if l.strip()]
    rows = [r for r in rows if r.get("kind", "qa") == "qa"]
    hit = 0
    for r in rows:
        lanes = []
        if mode in ("hybrid", "bm25"):
            lanes.append(idx.lexical(r["question"], settings.top_k_per_lane))
        if mode in ("hybrid", "dense"):
            lanes.append(idx.dense(embed_query(r["question"]), settings.top_k_per_lane))
        if mode == "hybrid":
            cids = rrf_fuse(lanes, settings.final_k * 3, settings.rrf_k)
        else:
            cids = [c for c, _ in lanes[0]]
        from clerk import store as st
        docs = [st.chunk_by_id(c) for c in cids[:10]]
        top_docs = {d["doc_id"] for d in docs if d}
        hit += int(bool(top_docs & set(r["gold_doc_ids"])))
    print(f"{mode}: recall@5(top15-doc) = {hit}/{len(rows)} = {hit/len(rows):.3f}")


if __name__ == "__main__":
    main()
