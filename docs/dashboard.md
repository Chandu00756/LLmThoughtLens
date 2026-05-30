# LLmThoughtLens — Live Dashboard, Proxy & SDK

The live layer turns ThoughtLens from a one-shot report generator into a
real-time observability tool that sits between your app and the model.

```bash
pip install 'LLmThoughtLens[server]'
LLmThoughtLens serve            # http://127.0.0.1:8000
```

## What the dashboard shows

| Panel | Source | Evidence |
| --- | --- | --- |
| 🔬 LLM X-ray | logit lens + activation grid + attention, live per token | white box only |
| Token Heatmap | per-token activation / masking importance | white or black box |
| Attribution Graph | real causal edges (activation flow or prob-deltas) | white or black box |
| Residual Stream | live per-layer norms during white-box generation | white box only |
| Feature Browser | extracted features, searchable | white or black box |
| Probes | 10 behavioural probes, pass/fail/score | behavioural |
| Live Proxy | API traffic routed through ThoughtLens | black box |

Every value is computed from a real model call — the dashboard never shows
synthetic numbers. A badge on each trace states whether the evidence is
**white-box** (real activations) or **black-box** (API logprobs / masking).

## Three ways to connect

### 1. Trace directly in the dashboard
Pick a provider, enter and **Test** its credentials, type a prompt, press
**Trace**. For a local HuggingFace model (including your own), press
**🔬 X-ray** to open the model up: the **logit lens** shows what each layer
predicts, so you watch the answer crystallise from the bottom layer to the top,
alongside a live layer×token activation grid and attention. An API model cannot
be X-rayed — it only returns text — so the dashboard says so plainly and points
you to a local model.

### 2. Proxy (any app with a custom base URL)
Point the app's OpenAI base URL at `http://127.0.0.1:8000/v1`:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
```

- `POST /v1/chat/completions` — OpenAI-compatible (OpenAI, Ollama `/v1`, vLLM,
  LM Studio, …).
- `POST /v1/messages` — Anthropic-compatible.

The upstream response is returned **byte-for-byte**; ThoughtLens only observes.
Requests appear live in the **Live Proxy** tab with the real next-token
distribution (when the upstream returns logprobs).

**Honest limitation.** Only traffic *routed through* the proxy is visible. Apps
that hardcode their endpoint (the Claude desktop app, Perplexity) cannot be
silently intercepted — that is a deliberate boundary, not a missing feature.

### 3. SDK (your own code)
```python
from LLmThoughtLens.sdk import trace, observe, wrap_openai, record_exchange
```

- `trace(prompt, provider=..., dashboard=...)` — one-shot, returns a real
  `TraceResult`, streams to the dashboard if a URL is given.
- `observe(...)` — context manager for repeated traces in a session.
- `wrap_openai(client, dashboard=...)` — drop-in over an OpenAI-style client;
  the real response is returned unchanged and observation never raises into
  your call path.
- `record_exchange(prompt, completion, dashboard=...)` — push an exchange your
  own stack already produced.

Externally produced events reach the dashboard via `POST /api/ingest`.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/` | dashboard SPA |
| GET | `/api/health` | version + available providers |
| GET/POST | `/api/config*` | provider config (keys returned masked) |
| POST | `/api/provider/test` | real capability check |
| POST | `/api/trace` | full interpretability trace |
| POST | `/api/whitebox/stream` | live white-box generation |
| POST | `/api/ingest` | push an external event to the bus |
| POST | `/v1/chat/completions` | OpenAI-compatible proxy |
| POST | `/v1/messages` | Anthropic-compatible proxy |
| WS | `/ws` | live event stream |

## Security notes

- API keys are stored locally in `~/.LLmThoughtLens/server.json` and are
  **never** returned to the browser in full — only a masked preview.
- The server binds to `127.0.0.1` by default. Only pass `--host 0.0.0.0` on a
  network you trust.
- ThoughtLens has **no telemetry**; it makes network calls only to the model
  upstream you configure and (optionally) to your own dashboard URL.
