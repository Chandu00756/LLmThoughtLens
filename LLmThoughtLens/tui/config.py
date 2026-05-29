"""Persistent TUI config in ``~/.LLmThoughtLens/config.json``.

Holds the last-used provider, model id, an opaque API-key handle (we do
NOT persist the key itself unless the user explicitly opted in), the
session history (last 20 traces), and per-screen view preferences.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CONFIG_DIR = Path(os.environ.get("LLMTHOUGHTLENS_HOME", "~/.LLmThoughtLens")).expanduser()
CONFIG_PATH = CONFIG_DIR / "config.json"
HISTORY_LIMIT = 20


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
        raw = json.loads(CONFIG_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return TUIConfig()
    try:
        return TUIConfig.from_json(raw)
    except (TypeError, ValueError):
        return TUIConfig()


def save_config(cfg: TUIConfig) -> None:
    """Persist *cfg* to ``CONFIG_PATH``, creating the dir if needed."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    payload = cfg.to_json()
    if not cfg.save_api_key:
        payload["api_key"] = ""  # never persist a key the user didn't opt-in for
    CONFIG_PATH.write_text(json.dumps(payload, indent=2))
