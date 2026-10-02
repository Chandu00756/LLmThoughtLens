"""LLmThoughtLens TUI — Textual-based interactive terminal interface.

Only :mod:`~LLmThoughtLens.tui.app`, ``screens`` and ``widgets`` need Textual.
``LLmThoughtLensApp`` / ``run_tui`` are resolved lazily (PEP 562), so importing
:mod:`LLmThoughtLens.tui.config` — or this package — works without the ``tui``
extra installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from LLmThoughtLens.tui.config import TUIConfig, load_config, save_config

if TYPE_CHECKING:
    from LLmThoughtLens.tui.app import LLmThoughtLensApp, run_tui

_LAZY_APP_ATTRS = frozenset({"LLmThoughtLensApp", "run_tui"})


def __getattr__(name: str) -> Any:
    if name in _LAZY_APP_ATTRS:
        from LLmThoughtLens.tui import app

        return getattr(app, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["LLmThoughtLensApp", "run_tui", "TUIConfig", "load_config", "save_config"]
