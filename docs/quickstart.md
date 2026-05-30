# LLmThoughtLens — Quickstart

> Trace a prompt through any LLM, see the circuits, save a self-contained
> HTML report. Every example below is **real, runnable code** — copy any
> block and run it as-is.

## Install

```bash
pip install -e ".[all,dev]"
```

Optional extras:

| Extra | What it enables |
| --- | --- |
| `LLmThoughtLens[openai]` | `OpenAIProvider` (real logprobs) |
| `LLmThoughtLens[anthropic]` | `AnthropicProvider` |
| `LLmThoughtLens[huggingface]` | `HuggingFaceProvider` + `SparseAutoencoder` (real PyTorch training) + `ActivationCache` |
| `LLmThoughtLens[ollama]` | `OllamaProvider` |
| `LLmThoughtLens[tui]` | the Textual terminal UI + `rapidfuzz` |
| `LLmThoughtLens[server]` | the live web dashboard + provider-compatible proxy (FastAPI/uvicorn) |
| `LLmThoughtLens[all]` | everything above |

## 60-second smoke check (no API key required)

```python
from LLmThoughtLens import Scope

scope = Scope.from_mock()
result = scope.trace_full("The capital of the state containing Dallas is")

print(result.output_token)            # next predicted token
print(len(result.features))           # extracted features
print(result.graph.num_nodes)         # attribution graph nodes
print(result.graph.num_edges)         # attribution graph edges
result.save("report.html")            # self-contained HTML report
```

## Live dashboard (the fastest way to see it work)

```bash
pip install 'LLmThoughtLens[server]'
LLmThoughtLens serve            # opens http://127.0.0.1:8000 in your browser
```

In the dashboard you can:

- pick a provider (Ollama / HuggingFace / OpenAI / Anthropic / mock), enter and
  **Test** its key/URL live;
- type a prompt and watch a trace render in real time (token heatmap,
  attribution graph, feature browser, probes);
- run a **white-box thinking stream** on a local HuggingFace model — every
  generated token streams its real per-layer residual-stream norms;
- route any app through the **proxy** and watch its traffic live.

Everything updates over a WebSocket; nothing is pre-baked.

## Middle layer — proxy any app through ThoughtLens

Start the server, then point any app that supports a custom OpenAI base URL at
`http://127.0.0.1:8000/v1`. Each request is forwarded **verbatim** to the real
upstream (the app's response is unchanged) and tee'd to the dashboard's
**Live Proxy** tab with the real next-token distribution.

```bash
# Example: route the OpenAI Python client through ThoughtLens
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
python -c "
from openai import OpenAI
c = OpenAI()                       # base_url picked up from env
print(c.chat.completions.create(
    model='gpt-4o-mini',
    messages=[{'role':'user','content':'The capital of France is'}],
).choices[0].message.content)
"
```

> Honest limitation: this only observes traffic that is *routed through* the
> proxy. Closed apps that hardcode their endpoint (e.g. the Claude desktop app)
> cannot be silently intercepted — by design.

## SDK — observe your own app in three lines

```python
# 1) One-shot trace, optionally streamed to a running dashboard
from LLmThoughtLens.sdk import trace
r = trace("The capital of France is", provider="ollama", model="llama3.1:8b",
          dashboard="http://localhost:8000")
print(r.output_token, r.top_features(3))

# 2) Drop-in wrapper over an OpenAI-style client (response returned untouched)
from openai import OpenAI
from LLmThoughtLens.sdk import wrap_openai
client = wrap_openai(OpenAI(), dashboard="http://localhost:8000")
client.chat.completions.create(model="gpt-4o-mini",
                               messages=[{"role": "user", "content": "hi"}])

# 3) Record an exchange your own stack already produced
from LLmThoughtLens.sdk import record_exchange
record_exchange("my prompt", "my completion", dashboard="http://localhost:8000")
```

## Provider 1 — OpenAI (real logprobs, black-box)

```python
import os
from LLmThoughtLens import Scope

scope = Scope.from_openai(
    model="gpt-4o-mini",
    api_key=os.environ["OPENAI_API_KEY"],
)
result = scope.trace_full(
    "The capital of France is",
    run_probes=True,
)
print(result.output_token, "—", result.output.output_prob)
for tok, prob in result.top_tokens[:5]:
    print(f"  {tok!r:>14}  p={prob:.3f}")
# Token-masking importance is computed on real API calls:
print("Top features (causal importance):")
for f in result.top_features(5):
    print(f"  {f.label:<20} score={f.score:.3f}  evidence={f.evidence_kind}")
result.save("openai_report.html")
```

## Provider 2 — Anthropic (Messages API, honest no-logprobs)

```python
import os
from LLmThoughtLens import Scope

scope = Scope.from_anthropic(
    model="claude-3-5-haiku-20241022",
    api_key=os.environ["ANTHROPIC_API_KEY"],
)
result = scope.trace_full("Explain step by step: 17 * 23")
print("output:", result.output_token)
print("evidence:", result.evidence_kind)         # "black_box"
print("provenance:", result.output.meta.get("evidence_note"))
# Anthropic does not expose logprobs — the report tags every cell accordingly.
result.save("anthropic_report.html")
```

## Provider 3 — HuggingFace (full white-box, real internals)

```python
from LLmThoughtLens import (
    Scope,
    ActivationCache,
    SparseAutoencoder,
    SAEConfig,
    FeatureIntervention,
)

# A small local model is enough to demo every feature.
scope = Scope.from_huggingface(model_name="gpt2", device="auto")

# 1) Trace + save report
result = scope.trace_full("Mary went to the store and she bought")
print(result.output_token, result.evidence_kind)   # "white_box"
result.save("gpt2_report.html")

# 2) Cache activations to train an SAE on layer 6
cache = ActivationCache(scope.provider, layer=6, max_tokens=4096)
cache.collect([
    "Mary went to the store and she bought",
    "The capital of France is",
    "Programming languages include Python, Java",
])
acts = cache.array()

# 3) Train a real TopK SAE
sae = SparseAutoencoder(SAEConfig(
    input_dim=acts.shape[1],
    dict_size=4 * acts.shape[1],
    k=32,
    n_steps=1_000,
    batch_size=256,
))
sae.fit(acts, verbose=True)
sae.save("sae_layer6.pt")
print(sae.sparsity_stats(acts[:512]))

# 4) Attach the SAE and re-trace — features now come from SAE codes
scope.attach_sae(sae, layer=6)
result = scope.trace_full("Mary went to the store and she bought")

# 5) Run a real forward-hook intervention on the residual stream
intervention = FeatureIntervention.inhibit(feature_id=42, scale=1.0, layer=6)
intervened = scope.trace_full(
    "Mary went to the store and she bought",
    interventions=[intervention],
)
print("baseline output:   ", result.output_token)
print("intervened output: ", intervened.output_token)
```

## Provider 4 — Ollama (local OSS models)

```python
from LLmThoughtLens import Scope

scope = Scope.from_ollama(
    model="llama3.2",
    base_url="http://localhost:11434",
)
# Quick health check before the trace
assert scope.provider.ping(), "Ollama daemon not reachable"

result = scope.trace_full("Write a haiku about autumn leaves.")
print(result.output_token)
result.save("ollama_report.html")
```

## The five report tabs

`result.save("report.html")` produces a single HTML file with these tabs,
each rendered from real numerical data (no placeholders):

1. **Token Heatmap** — Plotly heatmap of per-token activation sums.
2. **Attribution Graph** — layered DAG (input → features by layer → output).
   Teal edges promote the output, magenta edges suppress.
3. **Residual Stream** — PCA trajectory of every interesting token across
   layers (white-box only).
4. **Feature Browser** — searchable / filterable / sortable HTML table of
   every extracted feature.
5. **Probe Dashboard** — pass/fail/score scorecard plus a Scatterpolar
   radar over the 10 interpretability dimensions.

## CLI

```bash
LLmThoughtLens version
LLmThoughtLens providers                       # which extras are installed
LLmThoughtLens serve                           # live web dashboard + proxy
LLmThoughtLens trace "Dallas is in" --provider mock --output report.html
LLmThoughtLens probe --provider mock           # scorecard for all 10 probes
LLmThoughtLens benchmark --provider mock --output bench.json
LLmThoughtLens tui                             # full Textual interface
LLmThoughtLens cache-activations --model gpt2 --corpus prompts.txt \
    --layer 6 --output acts.npz
LLmThoughtLens train-sae --activations acts.npz --output sae.pt \
    --dict-size 16384 --k 64 --steps 5000
LLmThoughtLens label-features --sae sae.pt --activations acts.npz \
    --corpus prompts.txt --output sae_labelled.pt
```

## Next steps

- Read `docs/probe-reference.md` for the prompt / score / interpretation of
  every built-in probe.
- Read the build-tracking PDF for the phase-by-phase roadmap (Phases 0–10
  are all implemented in this 0.1.0 release).
