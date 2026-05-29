# Contributing to LLmThoughtLens

Thank you for your interest in contributing! This document describes how to
get started, the kinds of contributions we welcome, and the process for
submitting changes.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Ways to Contribute](#ways-to-contribute)
- [Development Setup](#development-setup)
- [Branch Strategy](#branch-strategy)
- [Commit Message Format](#commit-message-format)
- [Pull Request Process](#pull-request-process)
- [Style Guide](#style-guide)
- [Testing](#testing)
- [Adding a Provider](#adding-a-provider)
- [Adding a Probe](#adding-a-probe)

---

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).
By participating you agree to uphold those standards.

---

## Ways to Contribute

| Contribution type | Guidance |
|---|---|
| Bug reports | Open a GitHub Issue using the **Bug Report** template |
| Feature requests | Open a GitHub Issue using the **Feature Request** template |
| Documentation fixes | Open a PR directly against `main` |
| New providers | See [Adding a Provider](#adding-a-provider) |
| New probes | See [Adding a Probe](#adding-a-probe) |
| Core algorithm improvements | Open an issue first to discuss the approach |

---

## Development Setup

```bash
# 1. Fork and clone
git clone https://github.com/<your-fork>/LLmThoughtLens.git
cd LLmThoughtLens

# 2. Install in editable mode with dev extras
pip install -e ".[dev]"

# 3. Install pre-commit hooks
pre-commit install

# 4. Verify everything works
pytest
python scripts/track_progress.py
```

Python **3.10, 3.11, or 3.12** is required.

---

## Branch Strategy

| Branch | Purpose |
|---|---|
| `main` | Stable — protected, requires PR |
| `develop` | Integration branch for in-progress phases |
| `feat/<name>` | New feature or phase work |
| `fix/<name>` | Bug fix |
| `chore/<name>` | Non-functional (docs, CI, deps) |

---

## Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<optional scope>): <short summary>

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `chore`, `test`, `refactor`, `perf`, `ci`.

---

## Pull Request Process

1. Branch off `develop` (or `main` for doc-only changes).
2. Make your changes with tests.
3. Run `pytest` and `ruff check LLmThoughtLens tests` locally — both must pass.
4. Open a PR using the PR template and fill in all sections.
5. A maintainer will review within a few business days.
6. PRs require at least one approval and green CI before merge.

---

## Style Guide

- Formatter: **Black** (`line-length = 100`)
- Linter: **Ruff** (config in `pyproject.toml`)
- Type annotations: encouraged for all public functions/methods
- Docstrings: NumPy-style preferred

Run the full check locally:

```bash
ruff check LLmThoughtLens tests
ruff format LLmThoughtLens tests
mypy LLmThoughtLens
```

---

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=LLmThoughtLens --cov-report=term-missing

# Run a specific file
pytest tests/test_providers.py -v
```

All new features must include tests. Tests that require API keys must be
marked `@pytest.mark.integration` and will be skipped in CI by default.

---

## Adding a Provider

1. Create `LLmThoughtLens/providers/<name>_provider.py`.
2. Subclass `BaseProvider` and implement `run(prompt, **kwargs) -> ProviderOutput`.
3. Register in `LLmThoughtLens/providers/registry.py` inside `_register_builtins()`.
4. Add the optional dependency group to `pyproject.toml`.
5. Add tests in `tests/test_providers.py` (mock or integration-marked).
6. Update `README.md`.

---

## Adding a Probe

1. Create `LLmThoughtLens/probes/builtin/<name>.py` (or edit `builtin.py`).
2. Subclass `BaseProbe` and implement `run(activations) -> ProbeResult`.
3. Add tests in `tests/test_probes.py`.
4. Document the probe in `docs/`.

---

## Questions?

Open a [GitHub Discussion](https://github.com/Chandu00756/LLmThoughtLens/discussions)
or reach out via the issue tracker.
