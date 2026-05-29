<div align="center">

# LLmThoughtLens

**A platform-agnostic LLM interpretability toolkit for mechanistic transparency**

[![PyPI version](https://img.shields.io/pypi/v/LLmThoughtLens?color=01696f&logo=pypi&logoColor=white)](https://pypi.org/project/LLmThoughtLens/)
[![Python](https://img.shields.io/pypi/pyversions/LLmThoughtLens?color=01696f)](https://pypi.org/project/LLmThoughtLens/)
[![License: MIT](https://img.shields.io/badge/License-MIT-01696f.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-33%20passing-01696f)](tests/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

*Peer inside any language model — from a cloud API to a local Llama instance — using a single, unified Python interface.*

</div>

---

## What is LLmThoughtLens?

LLmThoughtLens (`pip install LLmThoughtLens`) is an open-source LLM interpretability framework built on the mechanistic principles from Anthropic's **Cross-Layer Transcoder** and *Biology of a Large Language Model* research (March 2025).

It extracts **sparse, monosemantic features**, traces how those features interact across layers as **attribution circuits**, and surfaces that structure as interactive visualisations — without requiring access to model weights. Whether you're working with the OpenAI API, a local HuggingFace model, or a mock provider for unit-tested research, the API stays the same.

### Key capabilities

| Capability | What it does |
|---|---|
| **Feature extraction** | Identifies interpretable features — L2-norm white-box or top-token black-box proxy |
| **Sparse Autoencoder (SAE)** | TopK SAE with NumPy Adam — train, save, load, label, and plug into any trace |
| **Attribution graphs** | Directed causal graphs of feature interactions, with pruning and best-path extraction |
| **Supernodes** | Greedy cosine-similarity clustering of features into higher-level semantic groups |
| **Probes** | Five built-in behavioural probes: multi-hop reasoning, hallucination, CoT faithfulness, refusal, motivated reasoning |
| **Interactive reports** | Self-contained tabbed HTML reports: attribution graph + token heatmap + probe results + raw JSON |
| **Provider-agnostic** | OpenAI, Anthropic, HuggingFace, Ollama, or the built-in deterministic MockProvider |

---

## Installation

```bash
# Core — MockProvider, no external API keys needed
pip install LLmThoughtLens

# With OpenAI support
pip install "LLmThoughtLens[openai]"

# With Anthropic Claude support
pip install "LLmThoughtLens[anthropic]"

# With local HuggingFace models (torch + transformers)
pip install "LLmThoughtLens[huggingface]"

# With local Ollama instance
pip install "LLmThoughtLens[ollama]"

# Everything at once
pip install "LLmThoughtLens[openai,anthropic,huggingface,ollama]"
```

**Requirements:** Python ≥ 3.10

---

## Quick start

```python
from LLmThoughtLens import Scope

# Zero-config: fully deterministic mock model — no API keys, no downloads
scope = Scope.from_mock()

# One-call full pipeline: forward pass → features → attribution graph → supernodes
result = scope.trace_full("The Eiffel Tower is located in")

print(result.output_token)          # top predicted next token
print(result.top_features(n=3))     # three most active features
print(result.top_paths(n=2))        # two highest-weight causal paths

result.show()                        # opens interactive attribution graph
result.show_heatmap()               # opens token activation heatmap
result.save("report.html")          # saves full tabbed HTML report
```

### Run all behavioural probes

```python
result = scope.trace_full(
    "The capital of the state containing Dallas is",
    run_probes=True,
)

for pr in result.probe_results:
    status = "✓" if pr.meta.get("passed") else "✗"
    print(f"  {status} {pr.probe_name}: {pr.score:.2f}  — {pr.meta.get('summary', '')}")
```

### Train and attach a Sparse Autoencoder

```python
import numpy as np
from LLmThoughtLens.features.sae import SparseAutoencoder, SAEConfig

# Train on captured activations (shape: n_samples × d_model)
config = SAEConfig(input_dim=768, dict_size=3072, k=64, n_steps=5_000)
sae = SparseAutoencoder(config)
sae.fit(activations, verbose=True)
sae.save("my_sae.npz")

# Attach to a Scope for richer white-box feature extraction
scope = Scope.from_huggingface("gpt2")
scope.attach_sae(sae, layer=6)
result = scope.trace_full("Paris is the capital of")
```

### Generate a standalone HTML report

```python
scope = Scope.from_openai(api_key="sk-...", model="gpt-4o-mini")
result = scope.report(
    "The French Revolution began in",
    output="interpretability_report.html",
    run_probes=True,
)
```

---

## Architecture

```
LLmThoughtLens/
│
├── scope.py               ← Scope + TraceResult  (main entry point)
│
├── providers/             ← Pluggable model backends
│   ├── mock_provider.py   ← Deterministic NumPy mock (default for CI)
│   ├── openai_provider.py
│   ├── anthropic_provider.py
│   ├── huggingface_provider.py
│   └── ollama_provider.py
│
├── features/              ← Sparse feature representations
│   ├── extractor.py       ← White-box (L2), black-box (top-token), SAE
│   ├── sae.py             ← TopK Sparse Autoencoder + NumPy Adam training
│   └── intervention.py    ← Amplify / inhibit / clamp feature activations
│
├── circuits/              ← Mechanistic circuit analysis
│   ├── graph.py           ← AttributionGraph: nodes, edges, prune, top_paths
│   ├── tracer.py          ← CircuitTracer: cosine-sim (white-box), proxy (black-box)
│   └── supernodes.py      ← SupernodeGrouper: greedy cosine-similarity clustering
│
├── probes/                ← Behavioural probe suite
│   ├── builtin.py         ← 5 built-in probes
│   └── runner.py          ← ProbeRunner + ProbeResult
│
└── visualization/         ← Output layer
    ├── graph_viz.py        ← Plotly attribution graph
    ├── token_heatmap.py    ← Plotly token activation heatmap
    └── report.py           ← Self-contained tabbed HTML report
```

### Data flow

```
Prompt
  │
  ▼
BaseProvider.run()  →  ProviderOutput
                           │
               ┌───────────┼───────────┐
               ▼           ▼           ▼
        activations      tokens      logits
               │                       │
               ▼                       ▼
       FeatureExtractor          top_tokens
               │
               ├─── (optional) SparseAutoencoder.encode()
               │
               ▼
         [Feature, ...]
               │
       ┌───────┴────────────┐
       ▼                    ▼
  CircuitTracer      SupernodeGrouper
       │                    │
       ▼                    ▼
 AttributionGraph    [FeatureSet, ...]
       │
       ▼
 TraceResult  ──→  show() / show_heatmap() / save()
                      │
                      ▼
              (optional) ProbeRunner
                      │
                      ▼
              [ProbeResult, ...]
```

---

## Provider comparison

| Provider | Activations | Attention | Setup |
|---|---|---|---|
| `MockProvider` | ✅ NumPy (deterministic) | ✅ | None — built-in |
| `HuggingFaceProvider` | ✅ Real hidden states | ✅ | `pip install "LLmThoughtLens[huggingface]"` |
| `OpenAIProvider` | ❌ Black-box | ❌ | API key + `pip install "LLmThoughtLens[openai]"` |
| `AnthropicProvider` | ❌ Black-box | ❌ | API key + `pip install "LLmThoughtLens[anthropic]"` |
| `OllamaProvider` | ❌ Black-box | ❌ | Running Ollama + `pip install "LLmThoughtLens[ollama]"` |

Black-box providers still work — LLmThoughtLens falls back to a top-token importance proxy for feature extraction and an importance-product proxy for circuit tracing.

---

## Built-in probes

All probes return a `ProbeResult` with `score ∈ [0, 1]`, a `passed` flag, and a plain-English summary.

| Probe | What it tests | Research basis |
|---|---|---|
| `MultiHopProbe` | Dallas → Texas → Austin geographic chain | Anthropic case study 1–2 |
| `HallucinationProbe` | Real-vs-fictional entity confidence delta | Anthropic case study 6 |
| `CoTFaithfulnessProbe` | Arithmetic CoT under a misleading planted hint | Anthropic case study 7 |
| `RefusalProbe` | Direct vs. fictional-frame harmful request | Anthropic case study 8 |
| `MotivatedReasoningProbe` | Planted wrong year for the French Revolution | Anthropic case study 10 |

```python
from LLmThoughtLens import Scope
from LLmThoughtLens.probes.builtin import MultiHopProbe

scope = Scope.from_mock()
result = scope.run_probe(MultiHopProbe())
print(result.score, result.meta["summary"])
```

---

## Scope API reference

### `Scope`

| Method | Returns | Description |
|---|---|---|
| `Scope.from_mock(**kwargs)` | `Scope` | Deterministic MockProvider |
| `Scope.from_openai(api_key, model)` | `Scope` | OpenAI Chat Completions |
| `Scope.from_anthropic(api_key, model)` | `Scope` | Anthropic Messages API |
| `Scope.from_huggingface(model_name)` | `Scope` | Local HuggingFace model |
| `Scope.from_ollama(model)` | `Scope` | Local Ollama instance |
| `scope.trace(prompt)` | `ProviderOutput` | Raw provider output |
| `scope.trace_full(prompt, ...)` | `TraceResult` | Full interpretability pipeline |
| `scope.report(prompt, output)` | `TraceResult` | Trace + save HTML report |
| `scope.run_probe(probe)` | `ProbeResult` | Run a single probe |
| `scope.attach_sae(sae, layer)` | `None` | Plug in a trained SAE |

### `TraceResult`

| Attribute / Method | Description |
|---|---|
| `.output_token` | Top predicted next token |
| `.top_tokens` | List of `(token, probability)` tuples |
| `.features` | All extracted features |
| `.graph` | `AttributionGraph` |
| `.probe_results` | `ProbeResult` list |
| `.top_features(n)` | Top-n features by score |
| `.top_paths(n)` | Top-n causal paths through the circuit |
| `.show()` | Open attribution graph in browser |
| `.show_heatmap()` | Open token heatmap in browser |
| `.save(path)` | Write self-contained HTML report |

---

## CLI

```bash
# Trace a prompt with the mock provider
LLmThoughtLens trace "The capital of France is"

# Trace with a real provider
LLmThoughtLens trace "Once upon a time" --provider openai --model gpt-4o-mini

# Show version
LLmThoughtLens version
```

---

## Custom providers

Extend `BaseProvider` to plug in any inference backend:

```python
from LLmThoughtLens.providers.base import BaseProvider, ProviderOutput
import numpy as np

class MyProvider(BaseProvider):
    name = "my_model"

    def run(self, prompt: str, **kwargs) -> ProviderOutput:
        # Call your model here ...
        return ProviderOutput(
            prompt=prompt,
            tokens=["hello", "world"],
            token_ids=[1, 2],
            activations=np.zeros((12, 2, 768)),   # (layers, tokens, d_model)
            top_tokens=[("world", 0.9), ("there", 0.05)],
        )

from LLmThoughtLens import Scope
scope = Scope.from_provider(MyProvider())
result = scope.trace_full("Hello")
```

---

## Research foundations

LLmThoughtLens implements and extends the mechanistic interpretability techniques from:

- **Lindsey et al. (2025)** — *On the Biology of a Large Language Model* — Anthropic  
  Behavioural case studies underlying the built-in probe suite.

- **McDougall et al. (2025)** — *Cross-Layer Transcoder* — Anthropic  
  TopK Sparse Autoencoder architecture and attribution graph methodology.

The project is designed to make these techniques **accessible and reproducible** for researchers working across different model families and API providers.

---

## Development

```bash
git clone https://github.com/Chandu00756/LLmThoughtLens.git
cd LLmThoughtLens
pip install -e ".[dev]"

# Run the full test suite
pytest tests/ -v

# Lint
ruff check LLmThoughtLens/
black --check LLmThoughtLens/
```

---

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.  
Please open an issue first for any significant feature or API change.

---

## License

[MIT](LICENSE) © 2026 Chandu Chitikam
