.PHONY: install dev lint fmt typecheck test smoke clean help \
        release-check publish-test publish benchmark

PYTHON ?= python3
VENV   ?= .venv
PKG    ?= LLmThoughtLens

## Create venv and install package + dev extras (editable)
install:
	@$(PYTHON) -m venv $(VENV) 2>/dev/null || true
	@$(VENV)/bin/python -m pip install --quiet --upgrade pip wheel
	@$(VENV)/bin/python -m pip install -e ".[all,dev]"

## Alias for install
dev: install

## Run ruff + ruff-format check
lint:
	@$(VENV)/bin/ruff check $(PKG) tests
	@$(VENV)/bin/ruff format --check $(PKG) tests

## Auto-format the codebase
fmt:
	@$(VENV)/bin/ruff format $(PKG) tests
	@$(VENV)/bin/ruff check --fix $(PKG) tests

## Run mypy
typecheck:
	@$(VENV)/bin/mypy $(PKG)

## Run the full pytest suite
test:
	@$(VENV)/bin/pytest tests/

## Quick mock-only smoke check
smoke:
	@$(VENV)/bin/python -c "from $(PKG) import Scope; r = Scope.from_mock().trace_full('Hello world'); print('smoke ok:', r.output_token, len(r.features), 'features')"

## Run all 10 built-in probes against the mock provider and emit a scorecard
benchmark:
	@$(VENV)/bin/$(PKG) benchmark --provider mock --output benchmark_results.json

## Full pre-release sweep: ruff -> mypy -> pytest -> pip-audit -> build dry-run -> confirm artefacts
release-check:
	@echo ">>> ruff check"
	@$(VENV)/bin/ruff check $(PKG) tests
	@echo ">>> ruff format --check"
	@$(VENV)/bin/ruff format --check $(PKG) tests
	@echo ">>> mypy"
	@$(VENV)/bin/mypy $(PKG)
	@echo ">>> pytest"
	@$(VENV)/bin/pytest tests/ -q
	@echo ">>> pip-audit"
	@$(VENV)/bin/pip-audit --progress-spinner off --skip-editable || true
	@echo ">>> build dry-run"
	@rm -rf dist
	@$(VENV)/bin/python -m build --no-isolation
	@ls dist/*.tar.gz >/dev/null 2>&1 || (echo "ERROR: sdist (.tar.gz) was not produced in dist/" && exit 1)
	@ls dist/*.whl >/dev/null 2>&1 || (echo "ERROR: wheel (.whl) was not produced in dist/" && exit 1)
	@echo ">>> dist contents:"
	@ls -la dist/
	@echo ">>> release-check PASSED"

## Upload current dist/ to TestPyPI via twine (requires TEST_PYPI_TOKEN env var)
publish-test:
	@test -d dist || (echo "ERROR: dist/ missing — run 'make release-check' first" && exit 1)
	@$(VENV)/bin/twine upload --repository testpypi --skip-existing dist/*

## Upload current dist/ to PyPI via twine (requires PYPI_TOKEN env var)
publish:
	@test -d dist || (echo "ERROR: dist/ missing — run 'make release-check' first" && exit 1)
	@$(VENV)/bin/twine upload --skip-existing dist/*

## Remove caches and build artefacts
clean:
	@rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
	@find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true

help:
	@grep -E '^##' Makefile | sed 's/## /  /'
