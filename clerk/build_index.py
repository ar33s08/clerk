"""Build/refresh the index: fetch corpus -> chunk -> embed -> store.

Usage:
  python -m clerk.build_index --queries "attorney client privilege" "work product" --docs 100
  python -m clerk.build_index --pdf-dir ./uploads
"""
from __future__ import annotations

import argparse
import sys

from . import ingest, store
from .chunking import chunk_document
from .embedder import embed_texts
from .config import settings


def build(docs: list[dict]) -> dict:
    total_chunks = 0
    from .chunking import chunk_pages
    for i, d in enumerate(docs):
        chunks = chunk_pages(d["pages"]) if d.get("pages") else chunk_document(d["text"])
        if not chunks:
            continue
        store.upsert_document(d)
        for j in range(0, len(chunks), 128):
            batch = chunks[j:j + 128]
            vecs = embed_texts([c["text"] for c in batch])
            store.insert_chunks(d["doc_id"], batch, vecs)
        total_chunks += len(chunks)
        print(f"[{i+1}/{len(docs)}] {d['title'][:60]} -> {len(chunks)} chunks")
    return {"documents": len(docs), "chunks": total_chunks}


def main() -> int:
    ap = argparse.ArgumentParser(prog="clerk.build_index")
    ap.add_argument("--queries", nargs="*", default=["attorney client privilege",
                                                     "work product doctrine",
                                                     "summary judgment standard"])
    ap.add_argument("--docs", type=int, default=100, help="max docs per query")
    ap.add_argument("--pdf-dir", default=None)
    ap.add_argument("--min-words", type=int, default=1500)
    a = ap.parse_args()
    store.init_db()
    docs: list[dict] = []
    for q in a.queries:
        print(f"fetching CourtListener: {q!r}")
        docs.extend(ingest.fetch_courtlistener(q, max_docs=a.docs, min_words=a.min_words))
    if a.pdf_dir:
        docs.extend(ingest.load_local_files(a.pdf_dir))
    # dedupe across query lanes
    uniq = {d["doc_id"]: d for d in docs}
    print(f"{len(uniq)} unique documents")
    stats = build(list(uniq.values()))
    print("done:", stats, "| db:", settings.db_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
