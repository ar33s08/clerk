# CLERK launch posts (verified numbers, 2026-10-10)

Repo: https://github.com/ar33s08/clerk · CI green · eval gate: PASS (2 consecutive runs)

Facts all posts must keep true:
- 83 public court/admin decisions, 1,519 chunks, real PDF page provenance
- citation precision 0.93 / recall 0.95 / faithfulness 0.95 / refusal 1.00 / 0 leaks / $0.002 per query
- gate history: FAIL, FAIL, FAIL, PASS, PASS — published in docs/BENCHMARKS.md
- evidence lock = server-side byte-match of every quoted span, refusal not repair on failure
- 3-stage privilege gate (regex → embedding zero-shot → LLM), fail-closed
- ablation: BM25 88.8% / dense 81.6% / RRF hybrid 94.7% gold-doc hit (at N=1,519)
- naive baseline: 0% bad quotes on clean evidence → claim is "hallucination made visible and blocking", never "eliminated"
- one real privilege leak caught by adversarial probes → refusal 40% → 100%

Never say: "eliminated hallucination", "production-ready for law firms", named-comparison
beats ("better than X"), or imply the corpus is licensed legal data.

---

## LinkedIn (primary)

I built an AI system where a wrong answer is structurally impossible to ship silently.

CLERK (github.com/ar33s08/clerk) answers questions over US court decisions — the kind of
retrieval task where a single hallucinated citation would be a malpractice event, not a bug
report. The difference between it and a demo RAG repo is what happens after generation:

• Every answer must quote a verbatim span of a source document. The server validates each
  quote byte-for-byte against the chunk it claims. One mismatch → hard refusal. Not a
  warning. Not a retry. A refusal.
• Before retrieval results ever reach the model, a privilege/PII gate screens them: regex
  surface patterns, then embedding zero-shot classification, then an LLM arbiter. If the LLM
  is down, suspects quarantine to human review — fail-closed, because a false quarantine
  costs a glance and a false pass costs a privilege waiver.
• A CI eval harness scores citation precision/recall, faithfulness (claim→evidence NLI),
  adversarial-probe refusals, and privilege leaks — and it blocks deploys below threshold.

Here's the part I'd show in a code review: the gate said FAIL three times before it passed.
Run 1's probe suite caught a real leak (a spelled-out SSN pattern the screen missed) —
refusal accuracy went 40% → 100% after that fix. Run 3's precision gap turned out to be my
gold set's fault, not the model's. Run 5: precision 0.93, recall 0.95, faithfulness 0.95,
refusals 7/7, zero leaks, $0.002 per query, p50 ~6s, served on self-hosted vLLM.

One ablation I didn't want to publish: at 646 chunks, hybrid RRF retrieval lost to plain
BM25. I scaled the corpus to 1,519 chunks and re-ran it anyway — fusion then paid +5.9
points over the best single lane (94.7% gold-doc hit). Hypothesis confirmed, inconvenient
intermediate result included in the writeup, because the intermediate result is the point.

Stack: Python, FastAPI, SQLite→(designed for) pgvector, BM25+dense+RRF, sentence-transformers,
vLLM on a DGX Spark, GitHub Actions gating deploys. ~4 weeks of evenings, one engineer, zero
licensed data — the corpus comes from public court APIs with real page numbers.

The thing I'd want an interviewer to ask me: "why not just put the corpus in a long context
window?" The answer is the whole project: cost at scale, citation auditability, and a
privilege firewall that cannot live inside a prompt.

#AIEngineering #RAG #LLMOps #Evals #LegalTech

---

## X/Twitter thread (5 posts)

1/ Most RAG portfolios answer questions. Mine refuses them — and that's the feature.

CLERK: cited QA over 83 public court decisions where a hallucinated citation CANNOT pass
silently. build → github.com/ar33s08/clerk 🧵

2/ The trick isn't the prompt, it's what happens AFTER generation:

every answer must quote a verbatim span → server validates the quote byte-for-byte against
its claimed source → one mismatch = hard refusal. Hallucination becomes visible or it
doesn't become an answer.

3/ My eval gate failed 3 times before passing.

Run 1: adversarial probes caught a REAL privilege leak (spelled-out SSN dodged my regex
screen). Refusal accuracy: 40% → 100% after the fix.

Run 5: precision .93, recall .95, refusal 1.00, 0 leaks, $0.002/query. Green, and I can
show you every red that got me there.

4/ The ablation I didn't want to run twice:

at 646 chunks, my fancy hybrid RRF retrieval LOST to plain BM25 (87.3 vs 88.9).
scaled to 1,519 chunks and re-ran: hybrid 94.7 vs BM25 88.8. fusion pays at scale —
but only if you let the small-corpus result embarrass you first.

5/ What it does NOT claim: it doesn't "eliminate hallucination." The naive baseline scored
0 bad quotes on clean evidence too. The honest claim: validation makes hallucination visible
and blocking — 7.5% of would-be answers get refused to stay at zero.

Stack: FastAPI, vLLM (DGX Spark), BM25+dense+RRF, CI eval gating. One engineer.

---

## Reddit r/MachineLearning (Learning/Project showcase angle)

Title: I built a legal-RAG system where hallucinated citations can't pass silently — eval gate failed 3x before passing, all runs published

Body: (condensed version of the LinkedIn post + the two "honest surprise" details: the leak
the probes caught, and BM25 beating RRF at small N. Link repo + docs/BENCHMARKS.md. Answer
comments about the span-validation vs constrained-decoding tradeoff and the fail-closed
privilege gate — those are the two design threads that draw the best discussion.)

---

## Portfolio README one-liner (resume link text)

CLERK — evidence-locked legal QA over public court records; server-side quote validation,
fail-closed privilege gate, CI eval gating (5 audited runs published).

## Posting checklist
- [ ] LinkedIn: Tue/Wed/Thu morning ET is the engagement window; first comment = the
      docs/BENCHMARKS.md link (algorithm nudges comment threads)
- [ ] X thread: same day or day after; pin repo in profile first
- [ ] Reddit: only after LinkedIn/X settle; r/MachineLearning bans promo-flavored posts —
      frame as lessons learned
- [ ] Optional: dev.to/HN version = ARCHITECTURE.md decisions-as-prose rewrite
- [ ] Attach: repo link + the demo widget screenshot or the BENCHMARKS table image
