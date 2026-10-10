"""Generate golden candidates ONLY for documents not yet represented in
golden.jsonl (incremental expansion after corpus growth)."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clerk import llm, retrieval, store
from clerk.config import settings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-docs", type=int, default=60)
    ap.add_argument("--per-doc", type=int, default=2)
    a = ap.parse_args()
    store.init_db()
    retrieval.invalidate_index()
    gold_path = os.path.join(settings.data_dir, "golden.jsonl")
    have = set()
    if os.path.exists(gold_path):
        for line in open(gold_path):
            if line.strip():
                r = json.loads(line)
                have.update(r.get("gold_doc_ids_primary", r.get("gold_doc_ids", [])))
    new_docs, seen = [], set()
    for row in retrieval._index().rows:
        d = row["doc_id"]
        if d in seen:
            continue
        seen.add(d)
        if d not in have:
            new_docs.append(row)
        if len(new_docs) >= a.max_docs:
            break
    print(f"{len(new_docs)} unrepresented docs to annotate")
    n = 0
    with open(gold_path + ".candidates", "a") as f:
        for i, d in enumerate(new_docs):
            meta = store.doc_meta(d) or {}
            try:
                out = llm.chat_json([
                    {"role": "system", "content":
                     "You write hard retrieval QA for a legal-evidence eval set, from the "
                     "excerpt ONLY. Each question must be answerable ONLY from this excerpt "
                     "and must require quoting specifics (holdings, standards, parties, dates). "
                     'JSON: {"questions": ["...", ...]}'},
                    {"role": "user", "content": d["text"][:6000]},
                ], max_tokens=1200)
                for q in out.get("questions", [])[:a.per_doc]:
                    f.write(json.dumps({"question": q, "gold_doc_ids": [d],
                                        "gold_chunk_ids": [d["chunk_id"]],
                                        "kind": "qa"}) + "\n")
                    n += 1
            except Exception as e:
                print(f"skip {meta.get('title','?')[:40]}: {str(e)[:60]}", file=sys.stderr)
            if (i + 1) % 10 == 0:
                print(f"{i+1}/{len(new_docs)} docs, {n} questions so far")
    print(f"appended {n} rows -> {gold_path}.candidates")


if __name__ == "__main__":
    main()
