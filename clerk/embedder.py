"""Embedding lane.

Default: all-MiniLM-L6-v2 (CPU, free, honest baseline — this is the number we
benchmark *against*). Swap CLERK_EMBED_MODEL=bge-m3 / nomic-embed for the
production upgrade path; the interface is one function.

Visual lane stub: renders/loads page thumbnails; when a VLM endpoint is
configured it embeds page images (colPali-style); otherwise the lane falls
back to CLIP text-on-thumbnail captions. The lane degrades gracefully — that
is a deliberate design point so the system runs anywhere.
"""
from __future__ import annotations

import threading

import numpy as np

from .config import settings

_lock = threading.Lock()
_model = None


def _get_model():
    global _model
    with _lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(settings.embed_model, device="cpu")
    return _model


def embed_texts(texts: list[str], batch_size: int | None = None) -> np.ndarray:
    m = _get_model()
    vecs = m.encode(texts, batch_size=batch_size or settings.embed_batch,
                    normalize_embeddings=True, show_progress_bar=False,
                    convert_to_numpy=True)
    return vecs.astype(np.float32)


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]
