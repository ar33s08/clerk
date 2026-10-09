"""Central configuration, environment-driven (12-factor)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    # storage
    data_dir: str = os.environ.get("CLERK_DATA_DIR", os.path.expanduser("~/clerk/data"))
    # LLM gateway (OpenAI-compatible; defaults to the user's DGX-Spark vLLM box)
    llm_base_url: str = os.environ.get("CLERK_LLM_BASE_URL", "https://spark-fceb.tail391339.ts.net/v1")
    llm_model: str = os.environ.get("CLERK_LLM_MODEL", "qwen3.8-flash-next")
    llm_api_key: str = os.environ.get("CLERK_LLM_API_KEY", "")
    llm_timeout_s: float = float(os.environ.get("CLERK_LLM_TIMEOUT_S", "60"))
    # embeddings (CPU-friendly default; swap for bge-m3 / colpali at scale)
    embed_model: str = os.environ.get("CLERK_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embed_batch: int = 64
    # visual document retrieval lane (page thumbnails; pluggable encoder)
    vdr_enabled: bool = os.environ.get("CLERK_VDR", "0") == "1"
    # retrieval knobs
    top_k_per_lane: int = 25
    rrf_k: int = 60
    final_k: int = 6
    # privilege gate
    privilege_threshold: float = float(os.environ.get("CLERK_PRIV_THRESHOLD", "0.6"))
    # eval gates (CI fails the deploy below these)
    gate_citation_precision: float = 0.90
    gate_citation_recall: float = 0.80
    gate_faithfulness: float = 0.95
    gate_privilege_refusal: float = 1.00

    @property
    def db_path(self) -> str:
        return os.path.join(self.data_dir, "clerk.db")

    @property
    def pages_dir(self) -> str:
        return os.path.join(self.data_dir, "pages")


settings = Settings()
