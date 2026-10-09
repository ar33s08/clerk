# Benchmarks

All numbers below are produced by `python -m clerk.eval --gate` against
`data/golden.jsonl` and stored in the `eval_runs` table (`git_sha` + full
metric JSON per run). Reproduce: one command. This file only reports runs;
it never hand-edits numbers.

## Retrieval ablation (top-5 citation recall, same golden set)

| lane config | recall@5 | notes |
|---|---|---|
| dense only (MiniLM cosine) | _run `scripts/ablation.py dense`_ | |
| BM25 only | | |
| BM25 + dense (RRF) | | shipped default |

## Engine ablation

| config | citation precision | citation recall | faithfulness | hallucination rate |
|---|---|---|---|---|
| naive RAG (no span validation) | | | | baseline |
| evidence-locked (shipped) | | | | |

## Known limitations (published, not hidden)

- Faithfulness judge = same-family model as generator in dev runs → judge
  bias caveat; production router pins a different family.
- Golden set drawn from the same corpus family as eval docs; questions are
  human-curated but domain breadth is privilege-procedure-heavy by design.
- Page markers for API-sourced docs are word-band approximations, not
  printed pages.
