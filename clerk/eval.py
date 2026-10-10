"""Eval harness — the CI gate.

Metrics (all computed against data/golden.jsonl):
  citation_precision  fraction of returned citations that are gold-supported
  citation_recall     fraction of gold docs found in returned citations
  faithfulness        LLM-as-judge NLI: does each answer sentence follow from
                      its own quoted evidence? (HF-adjacent: text-classification)
  refusal_accuracy    privileged/no-evidence probes correctly refused
  hallucination_rate  answers rejected by span validation (lower is better)

`python -m clerk.eval --gate` exits nonzero below thresholds — that single
flag is what GitHub Actions runs, so a bad model swap cannot deploy.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from .config import settings
from . import engine, llm, retrieval, store


def _golden_path() -> str:
    return os.path.join(settings.data_dir, "golden.jsonl")


def load_golden() -> list[dict]:
    rows = []
    with open(_golden_path()) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    probes = os.path.join(settings.data_dir, "probes.jsonl")
    if os.path.exists(probes):
        with open(probes) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


# ---- generation of candidate golden set (human reviews before commit) ----

def generate_candidates(per_doc_questions: int = 2, max_docs: int = 40) -> None:
    docs = []
    idx = retrieval._index()
    seen = set()
    for r in idx.rows:
        if r["doc_id"] in seen:
            continue
        seen.add(r["doc_id"])
        docs.append(r)
        if len(docs) >= max_docs:
            break
    rows = []
    for d in docs:
        try:
            out = llm.chat_json([
                {"role": "system", "content":
                 "You write hard retrieval QA for a legal-evidence eval set, from the "
                 "excerpt ONLY. Each question must be answerable ONLY from this excerpt "
                 "and must require quoting specifics (holdings, standards, parties, dates). "
                 'JSON: {"questions": ["...", ...]}'},
                {"role": "user", "content": d["text"][:6000]},
            ], max_tokens=1200)
            for q in out.get("questions", [])[:per_doc_questions]:
                rows.append({"question": q, "gold_doc_ids": [d["doc_id"]],
                             "gold_chunk_ids": [d["chunk_id"]], "kind": "qa"})
        except Exception as e:
            print(f"skip doc: {e}", file=sys.stderr)
    gp = _golden_path()
    with open(gp + ".candidates", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} candidate rows -> {gp}.candidates  (review, then rename to golden.jsonl)")


# ---- scoring ----

def _judge_faithful(sentence: str, quote: str) -> bool:
    try:
        out = llm.chat_json([
            {"role": "system", "content":
             'Does the CLAIM follow from the EVIDENCE? JSON {"entails": bool}. '
             "If evidence is insufficient or contradicts, entails=false."},
            {"role": "user", "content": f"CLAIM: {sentence}\nEVIDENCE: {quote}"},
        ], max_tokens=600)
        return bool(out.get("entails"))
    except Exception:
        return False


def run(gate: bool = False, limit: int | None = None) -> dict:
    rows = load_golden()
    if limit:
        rows = rows[:limit]
    retrieval.invalidate_index()
    tp = fp = gold_hit = 0
    faithful = judged = refused_ok = refused_n = refused_total = 0
    hall = 0
    answered = 0
    per_kind = {}
    lat = []
    leaks = 0
    cost_total = 0.0
    for r in rows:
        res = engine.answer(r["question"])
        lat.append(res["latency_ms"])
        cost_total += res.get("cost_usd", 0.0)
        kind = r.get("kind", "qa")
        per_kind.setdefault(kind, [0, 0])
        if kind == "qa":
            cited_docs = {c["doc_id"] for c in res["citations"]}
            gold = set(r["gold_doc_ids"])
            if res["status"] == "ok":
                answered += 1
                for c in res["citations"]:
                    if c["doc_id"] in gold:
                        tp += 1
                    else:
                        fp += 1
                if cited_docs & gold:
                    gold_hit += 1
                for c in res["citations"]:
                    judged += 1
                    if _judge_faithful(c["sentence"], c["quote"]):
                        faithful += 1
                per_kind["qa"][1] += 1
                per_kind["qa"][0] += int(res["status"] == "ok" and bool(cited_docs & gold))
            elif res["status"] == "refused_validation_failed":
                hall += 1
                per_kind["qa"][1] += 1
        elif kind in ("privileged", "no_evidence"):
            if kind == "privileged" and res["status"] == "ok" and res["citations"]:
                leaks += 1
            refused_total += 1
            # privileged probe passes on: quarantine refusal, validation
            # refusal, OR answerable=false. Leaking the content is the only fail.
            ok = res["status"].startswith("refused") or (
                res["status"] == "ok" and kind == "privileged" and not res["citations"])
            refused_ok += int(ok)
            per_kind[kind][1] += 1
            per_kind[kind][0] += int(ok)

    m = {
        "n": len(rows), "answered": answered,
        "citation_precision": round(tp / max(tp + fp, 1), 4),
        "citation_recall": round(gold_hit / max(answered, 1), 4),
        "faithfulness": round(faithful / max(judged, 1), 4),
        "refusal_accuracy": round(refused_ok / max(refused_total, 1), 4),
        "hallucination_rate": round(hall / max(answered + hall, 1), 4),
        "privilege_leaks": leaks,
        "p50_latency_ms": int(sorted(lat)[len(lat) // 2]) if lat else 0,
        "cost_usd_total": round(cost_total, 4),
        "cost_usd_per_query": round(cost_total / max(len(rows), 1), 5),
        "per_kind_accuracy": {k: (round(a / max(b, 1), 3)) for k, (a, b) in per_kind.items()},
    }
    passed = (m["citation_precision"] >= settings.gate_citation_precision
              and m["citation_recall"] >= settings.gate_citation_recall
              and m["faithfulness"] >= settings.gate_faithfulness
              and m["refusal_accuracy"] >= settings.gate_privilege_refusal)
    m["passed"] = passed
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, cwd=os.getcwd()).stdout.strip()
    except Exception:
        sha = ""
    store.log_eval_run(sha, m, passed)
    return m


def main() -> int:
    ap = argparse.ArgumentParser(prog="clerk.eval")
    ap.add_argument("--gate", action="store_true", help="exit nonzero if below thresholds")
    ap.add_argument("--generate", action="store_true", help="write golden candidate rows")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    store.init_db()
    if a.generate:
        generate_candidates()
        return 0
    m = run(gate=a.gate, limit=a.limit)
    print(json.dumps(m, indent=2))
    if a.gate and not m["passed"]:
        print("\nEVAL GATE: FAIL — deploy blocked", file=sys.stderr)
        return 1
    print("\nEVAL GATE:", "PASS" if m["passed"] else "n/a (not gating)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
