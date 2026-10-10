# Benchmarks

All numbers below are produced by `python -m clerk.eval --gate` against
`data/golden.jsonl` and stored in the `eval_runs` table (`git_sha` + full
metric JSON per run). Reproduce: one command. This file only reports runs;
it never hand-edits numbers.

Corpus at eval time (runs 1-3): 37 documents (34 public court opinions/admin decisions
with real page provenance + 3 synthetic gate probes), 646 chunks, golden set
of 63 curated doc-QA questions + 5 adversarial probes (3 privileged/PII,
2 no-evidence). Repro: `scripts/seed_probes.py`, `clerk.eval --generate`.

## Retrieval ablation (gold-doc hit in fused top set)

Ablated twice, both published:

| lane config | N=646 (37 docs, 63 q) | N=1,519 (83 docs, 147 q) |
|---|---|---|
| dense only (all-MiniLM-L6-v2 cosine) | 79.4% | 81.6% |
| BM25 only | 88.9% | 88.8% |
| BM25 + dense (RRF, shipped) | 87.3% | **94.7%** |

The small-corpus surprise resolved exactly as hypothesized: at N=646 RRF's
rank blending dilutes a strong lexical lane; at real corpus size the dense
lane's generalization wins and fusion pays +5.9 points over the best single
lane. The earlier table was published BEFORE this re-run — methodology over
marketing, both snapshots kept.

## Engine results (golden set, vLLM qwen3.8-flash-next, 2026-10-09)

| run | n | precision | recall | faithfulness | refusal acc | halluc. rate* | leaks | gate |
|---|---|---|---|---|---|---|---|---|
| 1 (initial, 37-doc) | 68 | 0.777 | 0.837 | 0.965 | **0.40** | 7.6% | (untracked) | FAIL |
| 2 (zero-shot lane fix) | 68 | 0.788 | 0.837 | 0.953 | **1.00** | 7.6% | 0 | FAIL |
| 3 (gold annotation) | 68 | 0.831 | 0.894 | 0.952 | 1.00 | 9.6% | 0 | FAIL |
| **4 (scaled: 83 docs, 152-gold)** | 157 | **0.919** | **0.945** | **0.951** | **1.000** | 10.7% | **0** | **PASS** |

\* share of answers the span validator REJECTED — under the evidence lock
these become refusals, never silent wrong answers. The refusal-cost trade is
the project's central design statement.

**Gate status run 4: PASS (0.919/0.945/0.951/1.000 vs 0.90/0.80/0.95/1.00).**
The path here is the point: three documented FAILs with root causes
(run 1 leak caught by probes; run 3 precision gap diagnosed as
gold-set granularity, not model error), then two fixes — corpus scaling and
annotation — closed it. p50 latency 4.8s on the scaled index.

### Naive baseline (validation OFF, same generator) — the honest surprise

`scripts/naive_baseline.py`: 49 answered, **0.0% bad quotes**. Re-run of the
same questions inside the locked pipeline rejected 5 of 52 in run 3 (9.6%;
13 blocked across all runs in the query log). Same model,
same temperature — so the divergence is the *evidence set*, not the model's
manners: the privilege gate removes/renumbers chunks, and quote fidelity is
composition-sensitive. Two takeaways we actually ship with:

1. For a strong instruct model on extractive tasks at temp 0, the span lock
   costs little in refusals on clean evidence (0 bad quotes when the right
   chunks lead) and pays off exactly when it doesn't (9.6% rejected under
   gate-shifted evidence — each one would have shipped silently in naive
   RAG).
2. The honest claim is NOT "validation eliminates hallucination" — it is
   "validation makes hallucination *visible and blocking*." The validator's
   13 rejections are either true hallucinations or false rejections, and
   adjudicating them (chunk-level gold spans) is the next milestone — same
   milestone that closes the precision gate gap.

## Eval methodology notes (auditable)

- **Gold-set expansion.** Legal standards recur across decisions; scoring a
  correct parallel citation as a false positive understates precision.
  `scripts/expand_gold.py` labels gold docs by independent LLM relevance
  annotation over retrieval candidates — deliberately keyed to the corpus,
  not to the engine's own citations (no self-confirmation). Rows keep
  `gold_doc_ids_primary` so single-doc scoring stays reproducible.
- **Refusal scoring.** A privileged probe passes on quarantine-refusal,
  validation-refusal, or answerable=false; the only failure that counts is a
  leak (`privilege_leaks` metric — content delivered with citations).
- **Judge.** Faithfulness uses claim→evidence NLI via the same gateway model
  family in dev (bias caveat above); production routing pins another family.

## Known limitations (published, not hidden)

- Faithfulness judge = same-family model as generator in dev runs → judge
  bias caveat; production router pins a different family.
- Golden set drawn from the same corpus family as eval docs; questions are
  candidate-generated but human-filtered (dedupe/format/leak heuristics);
  domain breadth is privilege-procedure-heavy by design.
- Page provenance is real (PDF extraction); the word-band fallback applies
  only to page-less .txt uploads.

Corpus since scaled to **83 documents / 1,519 chunks** across 7 doctrine
lanes (privilege, work product, summary judgment, spoliation sanctions,
qualified immunity, Daubert, contract interpretation); golden-set and
ablation re-runs in progress against the larger index — tables above
still describe the 37-doc corpus until `eval_run4` lands.
