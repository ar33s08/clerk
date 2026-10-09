# Architecture & design decisions

## Decisions, not defaults

**1. Evidence-locked generation with post-hoc span validation.**
The model emits `{sentence, quote, evidence_id}` triples; the server proves
each quote is a normalized substring of its claimed source. Why post-hoc
validation instead of constrained decoding: it works against any OpenAI-
compatible endpoint (no logits access), it validates *retrieval provenance*
too (evidence_id must exist), and it composes with the privilege gate — a
quote can be "real" yet leak privileged text, which only a system-level
check catches. Cost: occasional refusals on unicode edge cases (curly-quote
normalization handles the common ones; `tests/test_core.py` pins this).

**2. Fail-closed privilege gate.**
If stage-2 classification is unreachable, suspects are quarantined rather
than passed. Asymmetric cost: a false quarantine costs a human a glance; a
false pass costs a privilege waiver. This asymmetry is the whole design.

**3. Two-stage gate = cost control.**
Surface regexes are free and run on 100% of chunks; the LLM classifier runs
only on the few percent that trip markers. Same pattern as malware scanning:
signature pass first, heavyweight analysis on suspects.

**4. RRF over learned fusion.**
Score scales differ across lanes (BM25 unbounded, cosine [-1,1]); RRF uses
only ranks, is monotone in each lane, and adds lanes without retraining —
the visual-lane flag (`CLERK_VDR`) plugs into the same fuse call.

**5. Chunk overlap at sentence granularity.**
Litigation answers are usually one paragraph (a holding). 380-word windows
with 2-sentence overlap keep the answer sentence inside one chunk (citation
validation requires substring containment — a chunk that splits the answer
in half fails validation). Page markers are inferred from word bands for
API-sourced text; PDF ingestion carries real pages.

**6. SQLite now, pgvector later — by construction.**
One module touches SQL; the blob-embedding schema mirrors pgvector's
float4[]. The migration is honest because it was designed for, not
discovered late.

**7. The eval set is a first-class artifact.**
`--generate` uses the LLM to draft QA with gold provenance, but the committed
`golden.jsonl` is human-curated. Model-authored eval sets silently measure
the model's habits, not the task. Candidate/review separation is in the CLI
for that reason.

## Failure modes considered

- **Query that needs the whole document** ("summarize this case") —
  refused as no_evidence; summaries are a separate lane (HF summarization
  task) deliberately not mixed into cited QA.
- **Near-duplicate productions** — dedup lane (sentence-similarity)
  planned as ingest-time clustering; noted, not built.
- **Judge bias in faithfulness scoring** — judge is a different model
  family than generator in production; documented limitation in
  docs/BENCHMARKS.md.

## Scalability budget

| stage | single box | scale-out move |
|---|---|---|
| ingest | ~2 docs/s (CPU embed) | GPU batch worker pool |
| BM25 | in-proc, O(N) | OpenSearch |
| dense | numpy matmul, ~50ms @ 20k chunks | HNSW/IVF index |
| LLM | 1 gateway | router w/ per-tenant budgets |
| audit log | SQLite WAL | partitioned Postgres, S3 cold tier |
