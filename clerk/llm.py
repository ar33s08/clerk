"""LLM gateway: OpenAI-compatible chat completions (vLLM on the DGX, or any
gateway) with graceful offline fallback.

Design points for the README:
- one chokepoint for model routing / cost accounting (tokens logged per query)
- timeout + no-retry policy for user-facing path (retries move to eval CLI)
- offline mode returns a deterministic extractive stub so the *pipeline*,
  its tests, and CI never depend on a GPU box being up.
"""
from __future__ import annotations

import json
import time
import urllib.request

from .config import settings

_last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "offline_fallbacks": 0}


class LLMError(RuntimeError):
    pass


def usage() -> dict:
    return dict(_last_usage)


def reset_usage() -> None:
    for k in _last_usage:
        _last_usage[k] = 0


def chat(messages: list[dict], temperature: float = 0.0, max_tokens: int = 900,
         json_mode: bool = False) -> str:
    payload = {
        "model": settings.llm_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if settings.llm_disable_thinking:
        # hybrid-thinking servers (vLLM): reasoning tokens otherwise eat
        # max_tokens and add 30-60s of latency to structured tasks
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    headers = {"Content-Type": "application/json"}
    if settings.llm_api_key:
        headers["Authorization"] = f"Bearer {settings.llm_api_key}"
    req = urllib.request.Request(
        settings.llm_base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=settings.llm_timeout_s) as r:
            j = json.loads(r.read().decode())
    except Exception as e:
        raise LLMError(f"llm gateway unreachable: {e}") from e
    u = j.get("usage") or {}
    _last_usage["prompt_tokens"] += u.get("prompt_tokens", 0)
    _last_usage["completion_tokens"] += u.get("completion_tokens", 0)
    _last_usage["calls"] += 1
    msg = j["choices"][0]["message"]
    content = msg.get("content")
    if content is None:
        # reasoning models: JSON often lands in reasoning_content when
        # max_tokens truncates before the visible answer
        content = msg.get("reasoning_content") or ""
        _last_usage.setdefault("truncated", 0)
        _last_usage["truncated"] += 1
    return content


def chat_json(messages: list[dict], max_tokens: int = 1500, **kw):
    txt = chat(messages, json_mode=True, max_tokens=max_tokens, **kw)
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        # salvage first {...} block
        s, e = txt.find("{"), txt.rfind("}")
        if s >= 0 and e > s:
            return json.loads(txt[s:e + 1])
        raise LLMError("model did not return JSON")
