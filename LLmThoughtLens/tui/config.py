"""Persistent TUI config in ``~/.LLmThoughtLens/config.json``.

Holds the last-used provider, model id, an opaque API-key handle (we do
NOT persist the key itself unless the user explicitly opted in), the
session history (last 20 traces), and per-screen view preferences.

Files under the config dir may hold API keys, so they are written with
:func:`~LLmThoughtLens.config_paths.write_private_text` (atomic, ``0o600``,
config home tightened to ``0o700``).  The paths and private-file helpers live
in the dependency-free :mod:`LLmThoughtLens.config_paths` so the server can use
them without the ``tui`` extra; they are re-exported here for backwards
compatibility.  :func:`load_config` / :func:`save_config` read this module's
``CONFIG_PATH`` at call time, so tests can redirect it with ``monkeypatch``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from LLmThoughtLens.config_paths import (
    CONFIG_DIR,
    CONFIG_PATH,
    PRIVATE_DIR_MODE,
    PRIVATE_FILE_MODE,
    ensure_private_dir,
    write_private_text,
)
from LLmThoughtLens.providers.defaults import upgrade_retired_model

HISTORY_LIMIT = 20

__all__ = [
    "CONFIG_DIR",
    "CONFIG_PATH",
    "HISTORY_LIMIT",
    "PRIVATE_DIR_MODE",
    "PRIVATE_FILE_MODE",
    "SessionEntry",
    "TUIConfig",
    "ensure_private_dir",
    "load_config",
    "save_config",
    "write_private_text",
]


@dataclass
class SessionEntry:
    """One row in the session-history navigation tree."""

    when: str
    provider: str
    model: str
    prompt: str
    output_token: str
    score_mean: float = 0.0

    def label(self) -> str:
        snippet = self.prompt if len(self.prompt) < 48 else self.prompt[:45] + "…"
        return f"{self.when}  {self.provider:<10} {snippet!r} → {self.output_token!r}"


@dataclass
class TUIConfig:
    """User preferences + session history persisted across sessions."""

    provider: str = "mock"
    model: str = ""
    base_url: str = ""
    save_api_key: bool = False
    api_key: str = ""  # only written when save_api_key is True
    top_k_features: int = 20
    attribution_threshold: float = 0.05
    blackbox_budget: int = 16
    api_cost_usd: float = 0.0
    last_prompt: str = "The capital of the state containing Dallas is"
    history: list[SessionEntry] = field(default_factory=list)

    def push_history(self, entry: SessionEntry) -> None:
        self.history.insert(0, entry)
        self.history = self.history[:HISTORY_LIMIT]

    def to_json(self) -> dict[str, Any]:
        return {
            **{k: v for k, v in asdict(self).items() if k != "history"},
            "history": [asdict(e) for e in self.history],
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> TUIConfig:
        hist = [SessionEntry(**e) for e in data.pop("history", [])]
        cfg = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        cfg.history = hist
        return cfg


def load_config() -> TUIConfig:
    """Load the TUI config from disk; return defaults if missing/corrupt."""
    try:
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # missing, unreadable, or not valid JSON/UTF-8
        return TUIConfig()
    if not isinstance(raw, dict):
        return TUIConfig()
    try:
        cfg = TUIConfig.from_json(raw)
        cfg.model = upgrade_retired_model(cfg.provider, cfg.model)
    except (TypeError, ValueError):  # wrong field types / unhashable junk
        return TUIConfig()
    return cfg


def save_config(cfg: TUIConfig) -> None:
    """Persist *cfg* to ``CONFIG_PATH`` atomically with ``0o600`` permissions.

    The config home is created (or tightened to) ``0o700``.
    """
    payload = cfg.to_json()
    if not cfg.save_api_key:
        payload["api_key"] = ""  # never persist a key the user didn't opt-in for
    write_private_text(CONFIG_PATH, json.dumps(payload, indent=2), tighten_dir=True)
