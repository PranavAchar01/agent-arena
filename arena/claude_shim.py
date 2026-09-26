"""Local stand-in for a hosted OpenAI-compatible endpoint: /v1/chat/completions backed by the headless Claude Code
CLI (/usr/local/bin/claude -p). Images arrive as data URIs, are written to a private temp dir, and the CLI is
allowed exactly one tool, Read, to look at them. Used only because there is no Vultr account today; at the event
the app points LLM_BASE_URL at Vultr Serverless Inference instead and this file is not needed.

Run: .venv/bin/python -m uvicorn arena.claude_shim:app --port 8790
"""

from __future__ import annotations

import base64
import os
import subprocess
import tempfile
import time
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

CLI = os.environ.get("CLAUDE_CLI", "/usr/local/bin/claude")
CLI_MODEL = os.environ.get("CLAUDE_CLI_MODEL", "sonnet")
app = FastAPI()


class Req(BaseModel):
    model: str
    messages: list[dict]
    max_tokens: int | None = None
    temperature: float | None = None


@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [{"id": "claude-sonnet-via-cli", "object": "model"}],
    }


@app.post("/v1/chat/completions")
def complete(req: Req):
    with tempfile.TemporaryDirectory(prefix="shim-") as tmp:
        parts, n = [], 0
        for msg in req.messages:
            content = msg.get("content")
            if isinstance(content, str):
                parts.append(content)
                continue
            for c in content or []:
                if c.get("type") == "text":
                    parts.append(c["text"])
                elif c.get("type") == "image_url":
                    url = c["image_url"]["url"]
                    if not url.startswith("data:image/"):
                        raise HTTPException(400, "only data: image URLs")
                    n += 1
                    path = os.path.join(tmp, f"image_{n}.jpg")
                    with open(path, "wb") as f:
                        f.write(base64.b64decode(url.split(",", 1)[1]))
        prompt = "\n".join(parts)
        if n:
            files = ", ".join(
                os.path.join(tmp, f"image_{i}.jpg") for i in range(1, n + 1)
            )
            prompt = f"Use the Read tool to view these {n} images in order: {files}\n\n{prompt}"
        t0 = time.time()
        p = subprocess.run(
            [
                CLI,
                "-p",
                prompt,
                "--model",
                CLI_MODEL,
                "--allowedTools",
                "Read",
                "--output-format",
                "text",
            ],
            capture_output=True,
            text=True,
            cwd=tmp,
            timeout=300,
        )
        if p.returncode != 0:
            raise HTTPException(502, f"claude cli failed: {p.stderr[-300:]}")
    return {
        "id": f"shim-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "usage": {"latency_s": round(time.time() - t0, 1)},
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": p.stdout.strip()},
            }
        ],
    }
