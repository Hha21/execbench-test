"""Minimal OpenRouter chat client for the kernel loop.

The API key comes from the OPENROUTER_API_KEY environment variable or the repo's .env file (any capitalisation of
the name). It is never printed or logged. Every call appends its token counts and cost to loop/runs/usage.jsonl.
The large, unchanging prompt prefix goes in the system message with cache_control, so repeat calls read it from
the prompt cache.
"""

import json
import os
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
URL = "https://openrouter.ai/api/v1/chat/completions"
USAGE_LOG = ROOT / "loop" / "runs" / "usage.jsonl"
DEFAULT_MODEL = "anthropic/claude-opus-5.5"


def api_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key.strip()
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                name, value = line.split("=", 1)
                if name.strip().removeprefix("export ").strip().upper() == "OPENROUTER_API_KEY":
                    return value.strip().strip('"').strip("'")
    raise SystemExit("No OPENROUTER_API_KEY in the environment or in .env")


def chat(static_text, dynamic_text, model=DEFAULT_MODEL, max_tokens=64000, effort="high", timeout=2400, tag=""):
    """One completion. Returns (text, info) where info has model, finish reason, tokens and cost."""
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            # 1-hour cache: reasoning-heavy calls can run longer than the default 5-minute TTL, so sequential calls
            # in a round would otherwise miss the cache (seen in round r2).
            {"role": "system", "content": [{"type": "text", "text": static_text,
                                            "cache_control": {"type": "ephemeral", "ttl": "1h"}}]},
            {"role": "user", "content": dynamic_text},
        ],
        "reasoning": {"effort": effort},
        "usage": {"include": True},
    }
    t0 = time.time()
    r = requests.post(URL, json=body, timeout=timeout,
                      headers={"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json",
                               "X-Title": "solx kernel loop"})
    if r.status_code != 200:
        raise RuntimeError(f"OpenRouter HTTP {r.status_code}: {r.text[:1000]}")
    d = r.json()
    if "choices" not in d:
        raise RuntimeError(f"OpenRouter returned no choices: {json.dumps(d)[:1000]}")
    choice = d["choices"][0]
    u = d.get("usage", {})
    info = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), tag=tag, model=d.get("model", model),
                finish=choice.get("finish_reason"), seconds=round(time.time() - t0, 1),
                prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"),
                cached_tokens=(u.get("prompt_tokens_details") or {}).get("cached_tokens"),
                reasoning_tokens=(u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                cost_usd=u.get("cost"))
    USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(USAGE_LOG, "a") as f:
        f.write(json.dumps(info) + "\n")
    return choice["message"].get("content") or "", info
