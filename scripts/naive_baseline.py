"""Naive-RAG baseline: same retrieval + same generator, span validation OFF
(quotes accepted without byte-match). Measures the hallucinated-quote rate
the evidence lock exists to prevent. Run with golden.jsonl in place.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clerk import engine, llm, retrieval, store
from clerk.engine import _norm


def answer_no_validation(question, k=None):
    chunks = retrieval.retrieve(question, k=k or 6)
    ev_text = "\n\n".join(
        f"[EVIDENCE {i}] (doc={c['doc_id']} page={c['page']})\n{c['text']}"
        for i, c in enumerate(chunks))
    out = llm.chat_json([
        {"role": "system", "content": engine.SYSTEM},
        {"role": "user", "content": f"Evidence:\n{ev_text}\n\nQuestion: {question}"},
    ], max_tokens=3000)
    return out, {i: c["text"] for i, c in enumerate(chunks)}


def main():
    store.init_db()
    retrieval.invalidate_index()
    path = os.path.join(os.path.dirname(store.settings.db_path), "golden.jsonl")
    rows = [json.loads(l) for l in open(path) if l.strip()
            and json.loads(l).get("kind", "qa") == "qa"]
    n_bad_quote = n_answered = bad_rows = 0
    for r in rows:
        try:
            out, ev = answer_no_validation(r["question"])
        except Exception:
            continue
        if not out.get("answerable"):
            continue
        n_answered += 1
        row_bad = 0
        for s in out.get("support", []):
            src = ev.get(int(s.get("evidence_id", -1)))
            if src is None or _norm(s.get("quote", "")) not in _norm(src):
                n_bad_quote += 1
                row_bad += 1
        bad_rows += int(row_bad > 0)
    print(json.dumps({
        "n_answered": n_answered,
        "rows_with_bad_quotes_pct": round(100 * bad_rows / max(n_answered, 1), 1),
        "bad_quote_rate": round(n_bad_quote / max(n_answered, 1), 3),
    }))


if __name__ == "__main__":
    main()
