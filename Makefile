.PHONY: progress progress-json progress-update help

PYTHON := python3

## Show current build progress in the terminal
progress:
	@$(PYTHON) scripts/track_progress.py

## Emit progress as JSON
progress-json:
	@$(PYTHON) scripts/track_progress.py --json

## Rewrite PROGRESS.md with latest results
progress-update:
	@$(PYTHON) scripts/track_progress.py --update

help:
	@grep -E '^##' Makefile | sed 's/## /  /'
