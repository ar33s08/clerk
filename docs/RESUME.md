# Resume module — CLERK

Project line for the resume:

**CLERK — Cited Legal Evidence Retrieval & Review Kit** (Python, FastAPI,
vLLM, sentence-transformers, GitHub Actions) · github.com/ar33s08/clerk

Bullets (replace __ with numbers from `data/clerk.db:eval_runs` — do not
guess; the harness produces them):

- Built an evidence-locked RAG system over an 83-document public court corpus
  (1,519 chunks, real PDF page provenance); server-side verbatim-span
  validation makes hallucinated citations structurally impossible to pass
  silently — 10.7% of would-be answers rejected to refusal rather than
  shipped unverified; 0 unsupported answers ever passed the gate.
- Designed a 3-stage privilege/PII gate (lexical screen → embedding
  zero-shot → LLM arbiter, fail-closed on outage) reaching 100% refusal
  accuracy on adversarial probes after the eval harness caught one real
  leak in v1 (40% → 100%).
- Shipped a CI eval harness (citation precision/recall, LLM-NLI
  faithfulness, refusal accuracy, leak count) that gates deploys; caught a
  real privilege leak in v1 and a precision regression, driving 0.78 ->
  0.92 precision across 4 audited runs to a green deploy gate.
- Ablated retrieval lanes on a 152-question golden set: BM25 88.8%, dense
  81.6%, RRF hybrid 94.7% gold-doc hit — validated the fusion hypothesis
  across two corpus scales (87.3% -> 94.7% when N grew 2.3x).
- Served via vLLM (OpenAI-compatible) with per-query cost accounting:
  $0.0021/query across a 159-case harness ($0.34 total), p50 6.2s; a vLLM
  thinking-mode flag cut structured-call latency 6x (60s+ -> 10s).
- Added a DB-migrated cost ledger, opt-in LLM rerank lane, 7-case
  adversarial probe suite (7/7 refused, 0 leaks), and lifespan-managed API
  warmup across 2 reproducible green gate runs.

Interview talking points (with receipts in docs/ARCHITECTURE.md):
- "Why validate quotes post-generation instead of constrained decoding?"
  → endpoint-agnostic, validates provenance + privilege, composable.
- "How do you know your eval set is honest?" → generated candidates are
  human-curated before commit; model-written evals measure model habits.
- "What breaks at 100x scale?" → walk the scalability budget table; SQLite
  → pgvector was designed-for, not discovered-late.
- "Why not long context over the whole corpus?" → cost, citation
  auditability, and the privilege firewall that can't live in a prompt.
