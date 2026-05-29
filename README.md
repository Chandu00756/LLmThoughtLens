# llmscope

**Platform-agnostic LLM interpretability toolkit** — trace circuits, extract
sparse features, and run activation-space probes on *any* model through a
single, consistent API.

---

## Features (Phase 1 — Package Skeleton)

| Capability | Status |
|---|---|
| Unified `ProviderOutput` envelope | ✅ |
| `MockProvider` (deterministic, no API keys) | ✅ |
| `OpenAI`, `Anthropic`, `HuggingFace`, `Ollama` providers | ✅ stubs |
| `Scope` façade with factory methods | ✅ |
| `AttributionGraph` (directed weighted) | ✅ |
| `Feature` / `FeatureSet` dataclasses | ✅ |
| `BaseProbe` / `ProbeResult` / `ProbeRunner` | ✅ |
| CLI entry point (`llmscope trace …`) | ✅ |
| Lint (Ruff + Black), type-check (mypy) | ✅ |
| Pytest + coverage | ✅ |
| GitHub Actions CI (Python 3.10 / 3.11 / 3.12) | ✅ |
| Pre-commit hooks | ✅ |

---

## Quick Start

```bash
# install in editable mode (no GPU / API key required)
pip install -e .

# run the quickstart example
python examples/quickstart.py

# trace a prompt from the CLI
llmscope trace "Hello world" --provider mock
llmscope trace "Hello world" --provider mock --json
```

### Python API

```python
from llmscope import Scope

scope = Scope.from_mock(n_layers=4, seed=42)
out   = scope.trace("The capital of France is Paris")

print(out.tokens)        # ['The', 'capital', 'of', 'France', 'is', 'Paris']
print(out.activations.shape)   # (4, 6, 64)
print(out.top_tokens[:3])      # [('tok', 0.12), …]
```

Using a real provider:

```python
from llmscope import Scope

scope = Scope.from_openai(model="gpt-4o-mini")   # OPENAI_API_KEY env var
# or
scope = Scope.from_anthropic()                    # ANTHROPIC_API_KEY env var
# or
scope = Scope.from_huggingface("gpt2")            # local model
# or
scope = Scope.from_ollama("llama3.2")             # local Ollama daemon
```

---

## Development

```bash
pip install -e ".[dev]"
pre-commit install

# run tests
pytest

# lint
ruff check llmscope tests
ruff format llmscope tests

# type-check
mypy llmscope

# check progress
python scripts/track_progress.py
```

---

## Progress

See [PROGRESS.md](PROGRESS.md) for the auto-generated phase-by-phase progress report.

---

## License

MIT — see [LICENSE](LICENSE).
