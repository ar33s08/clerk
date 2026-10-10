# Benchmarks

All numbers below are produced by `python -m clerk.eval --gate` against
`data/golden.jsonl` and stored in the `eval_runs` table (`git_sha` + full
metric JSON per run). Reproduce: one command. This file only reports runs;
it never hand-edits numbers.

Corpus at eval time: 37 documents (34 public court opinions/admin decisions
with real page provenance + 3 synthetic gate probes), 646 chunks, golden set
of 63 curated doc-QA questions + 5 adversarial probes (3 privileged/PII,
2 no-evidence). Repro: `scripts/seed_probes.py`, `clerk.eval --generate`.

## Retrieval ablation (gold-doc hit in fused top set, same golden set)

| lane config | gold-doc hit rate | notes |
|---|---|---|
| dense only (all-MiniLM-L6-v2 cosine) | 79.4% | `scripts/ablation.py dense` |
| BM25 only | 88.9% | legal text is term-heavy; `scripts/ablation.py bm25` |
| BM25 + dense (RRF, shipped) | 87.3% | `scripts/ablation.py hybrid` |

Note the honest surprise: BM25 alone beats RRF here — at N=646 chunks RRF's
rank blending dilutes a strong lexical lane. The fusion gain is expected at
corpus scale where dense generalization matters; the numbers published are
the numbers measured, and the README claim is stated as measured, not hoped.

## Engine results (golden set, vLLM qwen3.8-flash-next, 2026-10-09)

| run | precision | recall | faithfulness | refusal acc | halluc. rate* | leaks |
|---|---|---|---|---|---|---|
| run 1 (initial gate) | 0.777 | 0.837 | 0.965 | **0.40** | 7.6% | (untracked) |
| run 2 (margin-based zero-shot lane) | 0.788 | 0.837 | 0.953 | **1.00** | 7.6% | 0 |
| run 3 (+ gold-set annotation) | 0.831 | 0.894 | 0.952 | 1.00 | 9.6% | 0 |

\* share of answers the span validator REJECTED — under the evidence lock
these become refusals, never silent wrong answers. The refusal-cost trade is
the project's central design statement.

**Gate status: FAIL on precision (0.831 < 0.90) — published, not hidden.**
Recall/faithfulness/refusal gates pass. Residual precision errors are the
generator citing a *parallel* decision whose quoted span genuinely supports
part of the answer but isn't the annotated primary; the annotation pass
(run 2→3: 0.788→0.831) closed most, the rest need per-sentence gold spans
(next milestone: chunk-level gold labels).

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
