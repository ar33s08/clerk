"""Expand golden gold_doc_ids by INDEPENDENT relevance annotation.

For each QA row: present the question + retrieval candidates (title + snippet);
an LLM annotator marks which candidates INDEPENDENTLY contain enough text to
correctly answer the question — deliberately NOT keyed to what the engine
cited (avoids self-confirmation bias). Originals kept as gold_doc_ids_primary;
rows touched get gold_llm_annotated: true.

Rationale: legal standards legitimately recur across decisions; scoring a
correct parallel citation as a false positive understates precision. The fix
must come from corpus evidence, not from the engine's own output.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clerk import llm, retrieval, store


def main():
    store.init_db()
    retrieval.invalidate_index()
    gold_path = os.path.join(os.path.dirname(store.settings.db_path), "golden.jsonl")
    rows = [json.loads(l) for l in open(gold_path) if l.strip()]
    out = []
    n = 0
    for r in rows:
        if r.get("kind", "qa") != "qa":
            out.append(r)
            continue
        cands = retrieval.retrieve(r["question"], k=12)
        seen, docs = set(), []
        for ch in cands:
            if ch["doc_id"] in seen:
                continue
            seen.add(ch["doc_id"])
            meta = store.doc_meta(ch["doc_id"]) or {}
            docs.append((ch["doc_id"], meta.get("title", ""), ch["text"][:700]))
        lines = [f"[{i}] {t}\n{snip}" for i, (_, t, snip) in enumerate(docs)]
        try:
            verdict = llm.chat_json([
                {"role": "system", "content":
                 "You label retrieval relevance for a legal QA dataset. Given a question "
                 "and candidate document snippets, return JSON {\"supports\": [i, ...]} "
                 "for every candidate that INDEPENDENTLY contains enough text to correctly "
                 "answer the question. A doc that merely mentions a related topic, or the "
                 "same party in an unrelated dispute, does not qualify. No prose."},
                {"role": "user", "content": "Q: " + r["question"] + "\n\nCandidates:\n" + "\n\n".join(lines)},
            ], max_tokens=400)
            add = {docs[int(i)][0] for i in verdict.get("supports", [])
                   if 0 <= int(i) < len(docs)}
        except Exception as e:
            print("skip:", str(e)[:60], file=sys.stderr)
            add = set()
        if add - set(r["gold_doc_ids"]):
            r["gold_doc_ids_primary"] = r.get("gold_doc_ids_primary", r["gold_doc_ids"])
            r["gold_doc_ids"] = sorted(set(r["gold_doc_ids"]) | add)
            r["gold_llm_annotated"] = True
            n += 1
        out.append(r)
    with open(gold_path, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    qa = [r for r in out if r.get("kind", "qa") == "qa"]
    print(f"rows gaining gold docs: {n}/{len(qa)} | "
          f"avg gold docs/qa-row: {sum(len(r['gold_doc_ids']) for r in qa)/len(qa):.2f}")


if __name__ == "__main__":
    main()
