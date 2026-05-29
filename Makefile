.PHONY: install dev lint fmt typecheck test smoke clean help

PYTHON ?= python3
VENV   ?= .venv

## Create venv and install package + dev extras (editable)
install:
	@$(PYTHON) -m venv $(VENV) 2>/dev/null || true
	@$(VENV)/bin/python -m pip install --quiet --upgrade pip wheel
	@$(VENV)/bin/python -m pip install -e ".[all,dev]"

## Alias for install
dev: install

## Run ruff + ruff-format check
lint:
	@$(VENV)/bin/ruff check thoughtlens tests
	@$(VENV)/bin/ruff format --check thoughtlens tests

## Auto-format the codebase
fmt:
	@$(VENV)/bin/ruff format thoughtlens tests
	@$(VENV)/bin/ruff check --fix thoughtlens tests

## Run mypy
typecheck:
	@$(VENV)/bin/mypy thoughtlens

## Run the full pytest suite
test:
	@$(VENV)/bin/pytest tests/

## Quick mock-only smoke check
smoke:
	@$(VENV)/bin/python -c "from thoughtlens import Scope; r = Scope.from_mock().trace_full('Hello world'); print('smoke ok:', r.output_token, len(r.features), 'features')"

## Remove caches and venv
clean:
	@rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
	@find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true

help:
	@grep -E '^##' Makefile | sed 's/## /  /'
