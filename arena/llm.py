"""Every LLM and VLM call goes through one OpenAI-compatible client. Switching providers is config only.

  LLM_BASE_URL / LLM_MODEL / LLM_API_KEY   text calls (query planning, candidate ranking)
  VLM_BASE_URL / VLM_MODEL / VLM_API_KEY   image calls (clip verification); default to the LLM settings

Today (no Vultr): both point at the local shim (arena/claude_shim.py), an OpenAI-compatible endpoint that runs the
headless Claude Code CLI. At the event: LLM_BASE_URL=https://api.vultrinference.com/v1 with a Vultr Serverless
Inference key and model id. Model output is treated as untrusted: callers parse JSON strictly and validate fields.
"""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path

from openai import OpenAI

SHIM = "http://127.0.0.1:8790/v1"


def _cfg(kind: str):
    base = os.environ.get(f"{kind}_BASE_URL") or os.environ.get("LLM_BASE_URL", SHIM)
    model = os.environ.get(f"{kind}_MODEL") or os.environ.get(
        "LLM_MODEL", "claude-sonnet-via-cli"
    )
    key = os.environ.get(f"{kind}_API_KEY") or os.environ.get(
        "LLM_API_KEY", "local-shim-no-key"
    )
    return base, model, key


def describe() -> dict:
    out = {}
    for kind in ("LLM", "VLM"):
        base, model, _ = _cfg(kind)
        out[kind.lower()] = {"base_url": base, "model": model}
    return out


def chat(
    prompt: str,
    images: list[Path] | None = None,
    kind: str = "LLM",
    max_tokens: int = 800,
    timeout: float = 180,
) -> str:
    base, model, key = _cfg("VLM" if images else kind)
    client = OpenAI(base_url=base, api_key=key, timeout=timeout)
    content: list[dict] = [{"type": "text", "text": prompt}]
    for p in images or []:
        b64 = base64.b64encode(Path(p).read_bytes()).decode()
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        )
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
        temperature=0,
    )
    return r.choices[0].message.content or ""


def parse_json(text: str):
    """First JSON object or array in a model reply, or None. Never eval, never trust keys beyond what we check."""
    m = re.search(r"```(?:json)?\s*([\[{].*?[\]}])\s*```", text, re.S) or re.search(
        r"([\[{].*[\]}])", text, re.S
    )
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
