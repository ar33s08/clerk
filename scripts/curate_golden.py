"""Curate golden candidates: dedupe, format filters, append new to golden.jsonl."""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clerk.config import settings


def main():
    gp = os.path.join(settings.data_dir, "golden.jsonl")
    existing = [json.loads(l) for l in open(gp) if l.strip()]
    known = {re.sub(r"\W+", " ", r["question"].lower()).strip() for r in existing}
    cands = [json.loads(l) for l in open(gp + ".candidates") if l.strip()]
    added = skip_dup = skip_fmt = 0
    for r in cands:
        q = r["question"].strip()
        key = re.sub(r"\W+", " ", q.lower()).strip()
        if key in known:
            skip_dup += 1
            continue
        if len(q.split()) < 6 or len(q.split()) > 60 or not q.endswith("?"):
            skip_fmt += 1
            continue
        known.add(key)
        existing.append(r)
        added += 1
    with open(gp, "w") as f:
        for r in existing:
            f.write(json.dumps(r) + "\n")
    print(f"added {added} | dup {skip_dup} | fmt {skip_fmt} | golden now {len(existing)}")


if __name__ == "__main__":
    main()
