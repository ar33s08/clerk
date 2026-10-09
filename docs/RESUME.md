# Resume module — CLERK

Project line for the resume:

**CLERK — Cited Legal Evidence Retrieval & Review Kit** (Python, FastAPI,
vLLM, sentence-transformers, GitHub Actions) · github.com/ar33s08/clerk

Bullets (replace __ with numbers from `data/clerk.db:eval_runs` — do not
guess; the harness produces them):

- Built an evidence-locked RAG system over __ public court opinions;
  server-side verbatim-span validation reduced unsupported/cited-wrongly
  answers from __% to __% on a __-question golden set.
- Designed a two-stage privilege/PII gate (lexical screen + LLM zero-shot
  classifier, fail-closed on outage) achieving __% refusal accuracy on
  privileged probes with __% false-quarantine rate.
- Shipped a CI eval harness (citation precision/recall, LLM-NLI
  faithfulness, refusal accuracy) that blocks deploys on regression;
  caught __ quality regressions across __ model swaps.
- Hybrid BM25+dense retrieval with RRF fusion improved top-5 citation
  recall __% → __% over dense-only baseline (ablation in repo).
- Served the LLM tier via vLLM (OpenAI-compatible); per-query token/cost
  logging, __ tokens/query at $0.__ — cost per reviewed document __% below
  a naive whole-corpus long-context approach.

Interview talking points (with receipts in docs/ARCHITECTURE.md):
- "Why validate quotes post-generation instead of constrained decoding?"
  → endpoint-agnostic, validates provenance + privilege, composable.
- "How do you know your eval set is honest?" → generated candidates are
  human-curated before commit; model-written evals measure model habits.
- "What breaks at 100x scale?" → walk the scalability budget table; SQLite
  → pgvector was designed-for, not discovered-late.
- "Why not long context over the whole corpus?" → cost, citation
  auditability, and the privilege firewall that can't live in a prompt.
