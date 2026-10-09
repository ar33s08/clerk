# CLERK — Cited Legal Evidence Retrieval & Review Kit

**Evidence-locked, citation-grounded QA over litigation corpora — with a
privilege firewall and a CI eval gate that can block a deploy.**

Every answer CLERK produces must quote a verbatim span from a retrieved
source document, validated **server-side, byte-for-byte, after generation**.
One unverifiable quote → hard refusal. Privileged material is triaged *before*
it ever reaches the prompt. Hallucinated citations are not "discouraged by a
prompt" — they are structurally unable to pass.

> Demo: [link when deployed] · Corpus: 21,910 public court opinions
> (CourtListener), ingested from the live API

## Why this exists

Document review is where LLMs meet the hardest acceptance bar: a single
hallucinated citation can draw sanctions, and one privileged leak is a
malpractice event. Off-the-shelf RAG fails exactly there. CLERK's thesis:
**make wrong answers impossible to pass silently** — a system guarantee, not
a prompt wish.

## Architecture

```
                    ┌────────────────────────────────────────────┐
 PDF / API ──► ingest ──► chunk (sentence-packed, overlap,       │
 (CourtListener,        page markers) ──► embed (MiniLM / bge)   │
  SEC EDGAR)            └───────────────┬────────────────────────┘
                                         ▼
 question ──► retrieve: BM25 + dense, RRF fusion, top-k ──► privilege gate
                    │        (stage1 regex/lexical → stage2 LLM zero-shot)
                    │                     │ human_review ──► quarantine queue
                    ▼                     ▼
        evidence-locked generation (JSON schema: answer + VERBATIM quotes)
                    │
                    ▼
        SPAN VALIDATOR: normalized byte-match of every quote vs source
                    │ fail ──► refused_validation_failed (never silent)
                    ▼ pass
        answer with [doc p.N] citations ──► query log (tokens, latency, cost)
```

Two-stage privilege gate: a free regex/lexical screen on every chunk; the LLM
zero-shot classifier (`attorney_client / work_product / pii / business`, the
HF [zero-shot-classification](https://huggingface.co/tasks/zero-shot-classification)
task) runs only on suspects. On LLM outage the gate is **fail-closed**:
suspects go to human review.

## What the Hugging Face tasks page maps to here

| HF task | Role in CLERK |
|---|---|
| [Document Question Answering](https://huggingface.co/tasks/document-question-answering) | answer surface over opinions/exhibits |
| [Sentence Similarity](https://huggingface.co/tasks/sentence-similarity) / [Text Ranking](https://huggingface.co/tasks/text-ranking) | dense lane + fusion reranking |
| [Zero-shot Classification](https://huggingface.co/tasks/zero-shot-classification) | privilege / PII triage |
| [Summarization](https://huggingface.co/tasks/summarization) | quarantine-queue digests |
| [Fill-mask / Feature Extraction](https://huggingface.co/tasks/feature-extraction) | embedding lane |

## Eval harness (the point of the whole repo)

`python -m clerk.eval --gate` scores a golden set and **exits nonzero** below
thresholds — wired into GitHub Actions, so any model swap that regresses
quality cannot merge:

| metric | gate |
|---|---|
| citation precision | ≥ 0.90 |
| citation recall | ≥ 0.80 |
| faithfulness (LLM-NLI judge) | ≥ 0.95 |
| privileged-probe refusal | = 1.00 |

Golden-set generation is semi-automated (`--generate` drafts QA from
ingested docs; a human curates before commit — eval sets are data, and data
gets reviewed).

## Quickstart

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m clerk.build_index --docs 60        # real corpus, ~10 min
.venv/bin/python -m clerk ask "When may a court find privilege waived?"
.venv/bin/python -m clerk.eval --generate && .venv/bin/python -m clerk.eval --gate
uvicorn clerk.api:app                                   # or: docker compose up
```

Env: `CLERK_LLM_BASE_URL` (any OpenAI-compatible endpoint — dev box runs
vLLM on a DGX Spark), `CLERK_LLM_MODEL`, `CLERK_API_TOKEN`,
`CLERK_EMBED_MODEL`.

## Scale path (documented, not vaporware)

- SQLite+numpy scan → Postgres+pgvector/HNSW (schema is 1:1; `store.py` is
  the only module that knows SQL)
- in-process BM25 → OpenSearch hybrid with RRF at the coordinator
- single FastAPI worker → queue-backed ingest (documents are content-addressed,
  so ingestion is idempotent and parallel)
- CPU MiniLM → GPU bge-m3 / colPali visual lane (`CLERK_VDR=1`, pluggable
  encoder interface in `embedder.py`)
- eval judge caching to cut harness cost; drift monitors on the query log

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for design decisions &
trade-offs.

## Honesty notes

- Corpus: public-domain US court opinions via CourtListener + optional local
  PDFs (SEC EDGAR exhibits). No licensed data.
- Metrics published in `docs/BENCHMARKS.md` come from `eval_runs` in the DB —
  reproduce with one command.
