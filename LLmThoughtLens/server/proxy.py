"""Provider-compatible proxy — the universal middle layer.

Exposes an **OpenAI-compatible** ``POST /v1/chat/completions`` endpoint that any
app supporting a custom base URL (aider, Continue, Cursor, LiteLLM, your own
apps) can point at.  The proxy:

1. Forwards the request **verbatim** to the configured upstream (OpenAI,
   Ollama's ``/v1``, vLLM, LM Studio, …) — the caller's response is unchanged,
   so nothing about the app's behaviour or safety is altered.
2. Tees a copy and publishes a live ``proxy_exchange`` event to the dashboard
   containing the real prompt, the real completion, and — when the upstream
   returns logprobs — the **real** next-token probability distribution. This is
   zero-extra-cost passive observation: no masking calls are issued here.

Deep attribution (token masking) is opt-in via ``POST /api/trace`` from the
dashboard, so the proxy never silently multiplies the user's API spend.

Honesty: this only observes traffic that is *routed through it*. It cannot see
inside closed apps that don't let you set a base URL.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any

import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse

from LLmThoughtLens.server.bus import get_bus
from LLmThoughtLens.server.config_api import load_server_config

_DEFAULT_OPENAI_BASE = "https://api.openai.com/v1"


def _resolve_upstream() -> tuple[str, str]:
    """Return ``(base_url, api_key)`` for the OpenAI-compatible upstream.

    Uses the configured ``openai`` provider base_url/key; falls back to the
    public OpenAI endpoint.  If ``ollama`` is the active provider and no
    OpenAI base is set, route to Ollama's OpenAI-compatible ``/v1``.
    """
    cfg = load_server_config()
    openai = cfg.settings_for("openai")
    base = openai.base_url.strip()
    key = openai.api_key
    if not base:
        if cfg.active_provider == "ollama":
            ollama = cfg.settings_for("ollama")
            ollama_base = (ollama.base_url or "http://localhost:11434").rstrip("/")
            base = f"{ollama_base}/v1"
            key = key or "ollama"  # Ollama ignores the key but the header is required
        else:
            base = _DEFAULT_OPENAI_BASE
    return base.rstrip("/"), key


def _extract_prompt(messages: list[dict[str, Any]]) -> str:
    """Flatten chat messages into the user-visible prompt for the dashboard."""
    parts = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if isinstance(content, list):  # OpenAI vision-style content blocks
            content = " ".join(blk.get("text", "") for blk in content if isinstance(blk, dict))
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _first_token_distribution(choice: dict[str, Any]) -> list[list[Any]]:
    """Pull the real next-token distribution from an OpenAI-style logprobs block."""
    try:
        content = choice["logprobs"]["content"]
        first = content[0]
        alts = first.get("top_logprobs", [])
        dist = [[a["token"], float(math.exp(a["logprob"]))] for a in alts]
        if not any(t == first["token"] for t, _ in dist):
            dist.insert(0, [first["token"], float(math.exp(first["logprob"]))])
        dist.sort(key=lambda kv: kv[1], reverse=True)
        return dist
    except (KeyError, IndexError, TypeError):
        return []


def build_router() -> APIRouter:
    router = APIRouter(tags=["proxy"])
    bus = get_bus()

    @router.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        body = await request.body()
        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError:
            payload = {}
        messages = payload.get("messages", [])
        prompt = _extract_prompt(messages) if isinstance(messages, list) else ""
        streaming = bool(payload.get("ask_stream", payload.get("stream", False)))

        base, key = _resolve_upstream()
        url = f"{base}/chat/completions"
        # Forward the caller's auth header if present, else inject configured key.
        headers = {"Content-Type": "application/json"}
        incoming_auth = request.headers.get("authorization")
        headers["Authorization"] = incoming_auth or f"Bearer {key}"

        bus.publish(
            "proxy_request",
            {
                "prompt": prompt,
                "model": payload.get("model", ""),
                "upstream": base,
                "stream": streaming,
            },
        )
        t0 = time.perf_counter()

        if streaming:
            return await _proxy_stream(url, headers, body, prompt, payload, bus, t0)
        return await _proxy_once(url, headers, body, prompt, payload, bus, t0)

    @router.post("/v1/messages")
    async def anthropic_messages(request: Request) -> Response:
        """Anthropic-compatible Messages passthrough.

        Forwards verbatim to the Anthropic API and tees the exchange to the
        dashboard.  Anthropic exposes no token logprobs, so the distribution is
        empty and the dashboard flags evidence honestly.
        """
        body = await request.body()
        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError:
            payload = {}
        messages = payload.get("messages", [])
        prompt = _extract_prompt(messages) if isinstance(messages, list) else ""

        cfg = load_server_config()
        anth = cfg.settings_for("anthropic")
        base = (anth.base_url or "https://api.anthropic.com").rstrip("/")
        url = f"{base}/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": request.headers.get("anthropic-version", "2023-06-01"),
        }
        incoming_key = request.headers.get("x-api-key")
        headers["x-api-key"] = incoming_key or anth.api_key

        bus.publish(
            "proxy_request",
            {
                "prompt": prompt,
                "model": payload.get("model", ""),
                "upstream": base,
                "stream": False,
            },
        )
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=300.0) as client:
            upstream = await client.post(url, content=body, headers=headers)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        completion = ""
        try:
            data = upstream.json()
            blocks = data.get("content", [])
            completion = "".join(
                b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"
            )
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass

        bus.publish(
            "proxy_exchange",
            {
                "prompt": prompt,
                "completion": completion,
                "model": payload.get("model", ""),
                "latency_ms": latency_ms,
                "next_token_distribution": [],
                "status_code": upstream.status_code,
                "evidence_kind": "black_box",
                "note": "Anthropic exposes no token logprobs.",
            },
        )
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
        )

    return router


async def _proxy_once(
    url: str,
    headers: dict[str, str],
    body: bytes,
    prompt: str,
    payload: dict[str, Any],
    bus: Any,
    t0: float,
) -> Response:
    """Non-streaming forward: return upstream verbatim, tee a live event."""
    async with httpx.AsyncClient(timeout=300.0) as client:
        upstream = await client.post(url, content=body, headers=headers)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    completion = ""
    distribution: list[list[Any]] = []
    try:
        data = upstream.json()
        choice = data["choices"][0]
        completion = choice.get("message", {}).get("content", "") or ""
        distribution = _first_token_distribution(choice)
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        pass

    bus.publish(
        "proxy_exchange",
        {
            "prompt": prompt,
            "completion": completion,
            "model": payload.get("model", ""),
            "latency_ms": latency_ms,
            "next_token_distribution": distribution,
            "status_code": upstream.status_code,
            "evidence_kind": "black_box",
        },
    )
    # Return the upstream response byte-for-byte.
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
    )


async def _proxy_stream(
    url: str,
    headers: dict[str, str],
    body: bytes,
    prompt: str,
    payload: dict[str, Any],
    bus: Any,
    t0: float,
) -> StreamingResponse:
    """Streaming forward: tee chunks to the caller while accumulating a copy."""

    async def gen() -> Any:
        chunks: list[str] = []
        async with (
            httpx.AsyncClient(timeout=300.0) as client,
            client.stream("POST", url, content=body, headers=headers) as upstream,
        ):
            async for raw in upstream.aiter_raw():
                chunks.append(raw.decode("utf-8", "ignore"))
                yield raw
        latency_ms = (time.perf_counter() - t0) * 1000.0
        completion = _accumulate_sse_text("".join(chunks))
        bus.publish(
            "proxy_exchange",
            {
                "prompt": prompt,
                "completion": completion,
                "model": payload.get("model", ""),
                "latency_ms": latency_ms,
                "next_token_distribution": [],
                "streamed": True,
                "evidence_kind": "black_box",
            },
        )

    return StreamingResponse(gen(), media_type="text/event-stream")


def _accumulate_sse_text(raw: str) -> str:
    """Reconstruct the assistant text from an OpenAI SSE stream copy."""
    out: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if not data or data == "[DONE]":
            continue
        try:
            obj = json.loads(data)
            delta = obj["choices"][0].get("delta", {})
            piece = delta.get("content")
            if piece:
                out.append(piece)
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            continue
    return "".join(out)
