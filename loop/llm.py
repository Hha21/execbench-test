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


def system_message(static_text):
    # 1-hour cache: reasoning-heavy calls can run longer than the default 5-minute TTL, so sequential calls in a round
    # would otherwise miss the cache (seen in round r2).
    return {"role": "system", "content": [{"type": "text", "text": static_text,
                                           "cache_control": {"type": "ephemeral", "ttl": "1h"}}]}


def chat(static_text, dynamic_text, model=DEFAULT_MODEL, max_tokens=64000, effort="high", timeout=2400, tag=""):
    """One completion. Returns (text, info) where info has model, finish reason, tokens and cost."""
    msg, info = complete([system_message(static_text), {"role": "user", "content": dynamic_text}],
                         model=model, max_tokens=max_tokens, effort=effort, timeout=timeout, tag=tag)
    return msg.get("content") or "", info


def complete(messages, tools=None, tool_choice=None, model=DEFAULT_MODEL, max_tokens=64000, effort="high",
             timeout=2400, tag=""):
    """One turn of a conversation. Returns (assistant message, info).

    The message keeps OpenRouter's reasoning_details and tool_calls, so it can be appended to `messages` as is:
    Claude needs its earlier reasoning passed back unchanged to continue after a tool result.
    """
    body = {"model": model, "max_tokens": max_tokens, "messages": messages, "reasoning": {"effort": effort},
            "usage": {"include": True}}
    if tools:
        body["tools"] = tools
        if tool_choice:
            body["tool_choice"] = tool_choice
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
    msg = choice["message"]
    keep = {k: msg[k] for k in ("role", "content", "tool_calls", "reasoning_details") if msg.get(k) is not None}
    keep.setdefault("role", "assistant")
    keep.setdefault("content", "")
    return keep, info
