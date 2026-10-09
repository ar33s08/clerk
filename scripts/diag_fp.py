import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clerk import store

store.init_db()
gold = {}
for line in open(os.path.join(os.path.dirname(store.settings.db_path), "golden.jsonl")):
    r = json.loads(line)
    gold[r["question"]] = r

tp = fp = 0
suspects = []
with store._conn() as c:
    for row in c.execute("SELECT question, citations FROM queries WHERE status='ok'"):
        g = gold.get(row["question"])
        if not g or g.get("kind", "qa") != "qa":
            continue
        for cite in json.loads(row["citations"]):
            if cite["doc_id"] in g["gold_doc_ids"]:
                tp += 1
            else:
                fp += 1
                meta = store.doc_meta(cite["doc_id"]) or {}
                suspects.append((row["question"][:80], meta.get("title", "?")[:50],
                                 cite["quote"][:90]))
print("tp", tp, "fp", fp)
for s in suspects[:15]:
    print("---")
    print("Q :", s[0])
    print("doc:", s[1])
    print("qt:", s[2])
