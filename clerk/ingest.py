"""Corpus ingestion.

Lane 1: CourtListener REST API — public-domain US court opinions (real,
litigation-adjacent text with genuine privilege questions inside them).
Lane 2: local PDF/HTML uploads (e.g. SEC EDGAR exhibits) via --pdf-dir.

Everything is normalized to {doc_id, text, title, date, url, meta}.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from .config import settings

_UA = {"User-Agent": "CLERK portfolio research (contact: areesmanesia@gmail.com)"}
_last_call = [0.0]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _get(url: str, timeout: int = 30) -> bytes:
    # global pacing: CL throttle is 5 req/s; stay well under, retry 429 w/ backoff
    wait = 0.45 - (time.time() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    req = urllib.request.Request(url, headers=_UA)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
            _last_call[0] = time.time()
            return body
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            code = getattr(e, "code", None)
            if attempt < 3 and (code == 429 or code is None):
                time.sleep(15 * (attempt + 1))
                continue
            raise


# ---------- Lane 1: CourtListener ----------

_CL_TAG_RE = re.compile(r"<[^>]+>")


def fetch_courtlistener(query: str, max_docs: int = 100, min_words: int = 1500) -> list[dict]:
    """Pull full opinion text for top results of a query, paginated."""
    docs: list[dict] = []
    seen: set[str] = set()
    page = 1
    while len(docs) < max_docs and page <= max_docs // 20 + 2:
        url = (
            "https://www.courtlistener.com/api/rest/v4/search/"
            f"?q={urllib.parse.quote(query)}&type=o&order_by=score+desc"
            f"&highlight=off&page={page}"
        )
        try:
            j = json.loads(_get(url))
        except Exception as e:  # rate limited / transient
            print(f"  CL page {page} failed: {e}; sleeping 60s")
            time.sleep(60)
            continue
        hits = j.get("results", [])
        if not hits:
            break
        for h in hits:
            if len(docs) >= max_docs:
                break
            # fetch the opinion cluster to get clean text
            try:
                cluster = json.loads(_get(h["cluster"]))
            except Exception:
                continue
            if (cluster.get("plain_text") or "").strip():
                body = cluster["plain_text"]
            else:
                body = _CL_TAG_RE.sub(" ", cluster.get("html_with_citations") or cluster.get("html") or "")
            body = re.sub(r"\s+", " ", body).strip()
            if len(body.split()) < min_words:
                continue
            key = _sha(body)
            if key in seen:
                continue
            seen.add(key)
            docs.append({
                "doc_id": key,
                "source": "courtlistener",
                "source_url": "https://www.courtlistener.com" + (cluster.get("absolute_url") or ""),
                "title": re.sub(r"<[^>]+>", "", h.get("caseName") or "untitled").strip(),
                "doc_date": (cluster.get("date_filed") or h.get("dateFiled") or "")[:10],
                "text": body,
                "meta": {"court": cluster.get("court"), "docket": cluster.get("docket_number")},
            })
        page += 1
    return docs


# ---------- Lane 2: local files ----------

def load_local_files(pdf_dir: str) -> list[dict]:
    docs = []
    for name in sorted(os.listdir(pdf_dir)):
        path = os.path.join(pdf_dir, name)
        try:
            if name.lower().endswith(".pdf"):
                from pypdf import PdfReader
                rd = PdfReader(path)
                body = " ".join((p.extract_text() or "") for p in rd.pages)
            elif name.lower().endswith((".txt", ".md")):
                body = open(path, encoding="utf-8", errors="replace").read()
            else:
                continue
        except Exception as e:
            print(f"  skip {name}: {e}")
            continue
        body = re.sub(r"\s+", " ", body).strip()
        if len(body.split()) < 200:
            continue
        docs.append({
            "doc_id": _sha(body), "source": "upload", "source_url": f"file://{path}",
            "title": name, "doc_date": "", "text": body, "meta": {"filename": name},
        })
    return docs
