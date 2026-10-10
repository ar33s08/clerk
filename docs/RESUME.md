# Resume module — CLERK

Project line for the resume:

**CLERK — Cited Legal Evidence Retrieval & Review Kit** (Python, FastAPI,
vLLM, sentence-transformers, GitHub Actions) · github.com/ar33s08/clerk

Bullets (replace __ with numbers from `data/clerk.db:eval_runs` — do not
guess; the harness produces them):

- Built an evidence-locked RAG system over a 37-document public court corpus
  (646 chunks, real PDF page provenance); server-side verbatim-span
  validation makes hallucinated citations structurally impossible to pass
  silently — 9.6% of would-be answers rejected to refusal rather than
  shipped unverified (DB logs: 13 blocked across eval runs).
- Designed a 3-stage privilege/PII gate (lexical screen → embedding
  zero-shot → LLM arbiter, fail-closed on outage) reaching 100% refusal
  accuracy on adversarial probes after the eval harness caught one real
  leak in v1 (40% → 100%).
- Shipped a CI eval harness (citation precision/recall, LLM-NLI
  faithfulness, refusal accuracy, leak count) that gates deploys; caught
  the gate regression above and drove precision 0.78 → 0.83 via gold-set
  annotation methodology.
- Ablated retrieval lanes on a curated golden set: BM25 88.9% vs dense
  79.4% vs RRF hybrid 87.3% gold-doc hit — published the surprise (and
  why fusion still earns its keep at scale) instead of cherry-picking.
- Served via vLLM (OpenAI-compatible) with per-query token accounting
  (~3.5k prompt / 0.3k completion per query, p50 10s); disabled-model-
  thinking flag cut structured-call latency 6x (60s+ → 10s).

Interview talking points (with receipts in docs/ARCHITECTURE.md):
- "Why validate quotes post-generation instead of constrained decoding?"
  → endpoint-agnostic, validates provenance + privilege, composable.
- "How do you know your eval set is honest?" → generated candidates are
  human-curated before commit; model-written evals measure model habits.
- "What breaks at 100x scale?" → walk the scalability budget table; SQLite
  → pgvector was designed-for, not discovered-late.
- "Why not long context over the whole corpus?" → cost, citation
  auditability, and the privilege firewall that can't live in a prompt.
