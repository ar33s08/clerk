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

## Engine ablation

| config | citation precision | citation recall | faithfulness | hallucination rate |
|---|---|---|---|---|
| naive RAG (no span validation) | _see eval_runs_ | | | baseline |
| evidence-locked (shipped) | | | | |

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
