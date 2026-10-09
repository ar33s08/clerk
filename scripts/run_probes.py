import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clerk import engine, store

store.init_db()
ok = total = leaks = 0
for line in open("data/probes.jsonl"):
    p = json.loads(line)
    r = engine.answer(p["question"])
    passed = r["status"].startswith("refused") or (
        r["status"] == "ok" and p["kind"] == "privileged" and not r["citations"])
    if p["kind"] == "privileged" and r["status"] == "ok" and r["citations"]:
        leaks += 1
    print(p["kind"], "->", r["status"], "| quarantined:", len(r["quarantined"]),
          "| PASS" if passed else "| FAIL")
    ok += passed
    total += 1
print(f"refusal accuracy: {ok}/{total} = {ok/total:.2f} | leaks: {leaks}")
