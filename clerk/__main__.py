"""Interactive CLI: init | ask | stats | quarantine."""
from __future__ import annotations

import argparse
import sys

from . import engine, store


def main() -> int:
    ap = argparse.ArgumentParser(prog="clerk")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    p = sub.add_parser("ask"); p.add_argument("question")
    sub.add_parser("stats")
    sub.add_parser("quarantine")
    a = ap.parse_args()
    store.init_db()
    if a.cmd == "init":
        print("db ready")
    elif a.cmd == "ask":
        r = engine.answer(a.question)
        print(f"[{r['status']}] {r['answer']}")
        for c in r["citations"]:
            print(f"  - [{c['doc_id'][:8]} p.{c['page']}] \"{c['quote'][:110]}...\"")
        for b in r["quarantined"]:
            print(f"  ! quarantined {b['label']}: {b.get('title','')[:60]}")
        print(f"  ({r['latency_ms']} ms, {r['tokens']})")
    elif a.cmd == "stats":
        for k, v in store.stats().items():
            print(f"{k:12} {v}")
    elif a.cmd == "quarantine":
        with store._conn() as c:
            for r in c.execute("SELECT q.*, d.title FROM quarantine q LEFT JOIN documents d ON d.doc_id=q.doc_id ORDER BY qid DESC LIMIT 20"):
                print(dict(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
