"""Every LLM and VLM call goes through one OpenAI-compatible client. Switching providers is config only.

  LLM_BASE_URL / LLM_MODEL / LLM_API_KEY   text calls (query planning, candidate ranking)
  VLM_BASE_URL / VLM_MODEL / VLM_API_KEY   image calls (clip verification); default to the LLM settings

Today (no Vultr): both point at the local shim (arena/claude_shim.py), an OpenAI-compatible endpoint that runs the
headless Claude Code CLI. At the event: LLM_BASE_URL=https://api.vultrinference.com/v1 with a Vultr Serverless
Inference key and model id. Model output is treated as untrusted: callers parse JSON strictly and validate fields.
"""

from __future__ import annotations

import base64
import contextvars
import json
import os
import re
import time
from collections.abc import Callable
from pathlib import Path

from openai import OpenAI

SHIM = "http://127.0.0.1:8790/v1"
# set by the server per box: every model call lands in that box's audit log (sizes and timing, never the prompt)
audit: contextvars.ContextVar[Callable[[dict], None] | None] = contextvars.ContextVar(
    "audit", default=None
)


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
    t0 = time.time()
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
        temperature=0,
    )
    out = r.choices[0].message.content or ""
    log = audit.get()
    if log is not None:
        log(
            {
                "type": "model_call",
                "kind": "VLM" if images else kind,
                "model": model,
                "endpoint": base.split("//")[-1].split("/")[0],
                "prompt_chars": len(prompt),
                "images": len(images or []),
                "reply_chars": len(out),
                "ms": round((time.time() - t0) * 1000),
            }
        )
    return out


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
