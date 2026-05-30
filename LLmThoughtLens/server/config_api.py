"""Provider configuration API — keys, models, and a real "Test" endpoint.

Persists multi-provider settings to ``~/.LLmThoughtLens/server.json`` (reusing
``tui.config.CONFIG_DIR``).  API keys are stored locally only; the GET endpoint
returns them **masked** so the dashboard never echoes a full secret back.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from LLmThoughtLens.tui.config import CONFIG_DIR

SERVER_CONFIG_PATH = CONFIG_DIR / "server.json"


@dataclass
class ProviderSettings:
    """Per-provider settings."""

    api_key: str = ""
    model: str = ""
    base_url: str = ""
    device: str = "auto"


@dataclass
class ServerConfig:
    """All provider settings + global trace defaults, persisted as JSON."""

    active_provider: str = "ollama"
    providers: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            "openai": {"api_key": "", "model": "gpt-4o-mini", "base_url": "", "device": "auto"},
            "anthropic": {
                "api_key": "",
                "model": "claude-3-5-haiku-20241022",
                "base_url": "",
                "device": "auto",
            },
            "huggingface": {"api_key": "", "model": "gpt2", "base_url": "", "device": "auto"},
            "ollama": {
                "api_key": "",
                "model": "llama3.1:8b",
                "base_url": "http://localhost:11434",
                "device": "auto",
            },
            "mock": {"api_key": "", "model": "", "base_url": "", "device": "auto"},
        }
    )
    top_k_features: int = 20
    attribution_threshold: float = 0.05
    blackbox_budget: int = 16

    def settings_for(self, provider: str) -> ProviderSettings:
        raw = self.providers.get(provider, {})
        return ProviderSettings(
            api_key=raw.get("api_key", ""),
            model=raw.get("model", ""),
            base_url=raw.get("base_url", ""),
            device=raw.get("device", "auto"),
        )


def load_server_config() -> ServerConfig:
    """Load the persisted server config, falling back to defaults."""
    cfg = ServerConfig()
    try:
        raw = json.loads(SERVER_CONFIG_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return cfg
    cfg.active_provider = raw.get("active_provider", cfg.active_provider)
    cfg.top_k_features = int(raw.get("top_k_features", cfg.top_k_features))
    cfg.attribution_threshold = float(raw.get("attribution_threshold", cfg.attribution_threshold))
    cfg.blackbox_budget = int(raw.get("blackbox_budget", cfg.blackbox_budget))
    for name, settings in raw.get("providers", {}).items():
        if name in cfg.providers:
            cfg.providers[name].update(settings)
        else:
            cfg.providers[name] = settings
    return cfg


def save_server_config(cfg: ServerConfig) -> None:
    """Persist the server config (creating the dir if needed)."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    SERVER_CONFIG_PATH.write_text(json.dumps(asdict(cfg), indent=2))


def _mask(secret: str) -> str:
    if not secret:
        return ""
    if len(secret) <= 8:
        return "•" * len(secret)
    return f"{secret[:4]}…{secret[-4:]}"


def masked_view(cfg: ServerConfig) -> dict[str, Any]:
    """Config dict with API keys masked, for sending to the browser."""
    out = asdict(cfg)
    for settings in out["providers"].values():
        if settings.get("api_key"):
            settings["api_key_masked"] = _mask(settings["api_key"])
            settings["api_key_set"] = True
        else:
            settings["api_key_masked"] = ""
            settings["api_key_set"] = False
        settings.pop("api_key", None)  # never send the raw key back
    return out


# ---------------------------------------------------------------------------
# Provider construction + real capability test
# ---------------------------------------------------------------------------


def build_provider(name: str, cfg: ServerConfig) -> Any:
    """Instantiate a provider from stored config via the registry."""
    from LLmThoughtLens.providers.registry import get_provider

    s = cfg.settings_for(name)
    kwargs: dict[str, Any] = {}
    if name == "openai":
        kwargs = {
            "model": s.model or "gpt-4o-mini",
            "api_key": s.api_key or os.environ.get("OPENAI_API_KEY"),
        }
    elif name == "anthropic":
        kwargs = {
            "model": s.model or "claude-3-5-haiku-20241022",
            "api_key": s.api_key or os.environ.get("ANTHROPIC_API_KEY"),
        }
    elif name == "huggingface":
        kwargs = {"model_name": s.model or "gpt2", "device": s.device or "auto"}
    elif name == "ollama":
        kwargs = {
            "model": s.model or "llama3.1:8b",
            "base_url": s.base_url or "http://localhost:11434",
        }
    return get_provider(name, **kwargs)


def test_provider(name: str, cfg: ServerConfig) -> dict[str, Any]:
    """Run a real, cheap capability check for *name*. Never raises."""
    try:
        if name == "ollama":
            provider = build_provider(name, cfg)
            ok = provider.ping()
            return {
                "ok": bool(ok),
                "detail": "Ollama server reachable" if ok else "Ollama server not reachable",
                "evidence_kind": provider.evidence_kind,
            }
        if name == "mock":
            provider = build_provider(name, cfg)
            out = provider.run("ping")
            return {
                "ok": True,
                "detail": f"mock returned {out.n_tokens} tokens",
                "evidence_kind": "white_box",
            }
        if name in ("openai", "anthropic"):
            provider = build_provider(name, cfg)
            out = provider.run("ping", max_tokens=1)
            return {
                "ok": True,
                "detail": f"{name} responded ({out.output_token!r})",
                "evidence_kind": provider.evidence_kind,
            }
        if name == "huggingface":
            # Loading weights is expensive; only verify the lib + tokenizer resolve.
            provider = build_provider(name, cfg)
            from transformers import AutoConfig

            AutoConfig.from_pretrained(provider.model_name)
            return {
                "ok": True,
                "detail": f"model config resolved for {provider.model_name!r} (weights load on first trace)",
                "evidence_kind": "white_box",
            }
    except Exception as exc:  # noqa: BLE001 — surface the real error to the UI
        return {"ok": False, "detail": f"{type(exc).__name__}: {exc}", "evidence_kind": "unknown"}
    return {"ok": False, "detail": f"unknown provider {name!r}", "evidence_kind": "unknown"}


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class ProviderUpdate(BaseModel):
    provider: str
    api_key: str | None = None
    model: str | None = None
    base_url: str | None = None
    device: str | None = None
    make_active: bool = False


class DefaultsUpdate(BaseModel):
    top_k_features: int | None = None
    attribution_threshold: float | None = None
    blackbox_budget: int | None = None
    active_provider: str | None = None


def build_router() -> APIRouter:
    router = APIRouter(prefix="/api", tags=["config"])

    @router.get("/config")
    def get_config() -> dict[str, Any]:
        return masked_view(load_server_config())

    @router.post("/config/provider")
    def update_provider(update: ProviderUpdate) -> dict[str, Any]:
        cfg = load_server_config()
        slot = cfg.providers.setdefault(
            update.provider, {"api_key": "", "model": "", "base_url": "", "device": "auto"}
        )
        # Only overwrite the key when a new non-empty value is supplied.
        if update.api_key:
            slot["api_key"] = update.api_key
        if update.model is not None:
            slot["model"] = update.model
        if update.base_url is not None:
            slot["base_url"] = update.base_url
        if update.device is not None:
            slot["device"] = update.device
        if update.make_active:
            cfg.active_provider = update.provider
        save_server_config(cfg)
        return masked_view(cfg)

    @router.post("/config/defaults")
    def update_defaults(update: DefaultsUpdate) -> dict[str, Any]:
        cfg = load_server_config()
        if update.top_k_features is not None:
            cfg.top_k_features = int(update.top_k_features)
        if update.attribution_threshold is not None:
            cfg.attribution_threshold = float(update.attribution_threshold)
        if update.blackbox_budget is not None:
            cfg.blackbox_budget = int(update.blackbox_budget)
        if update.active_provider is not None:
            cfg.active_provider = update.active_provider
        save_server_config(cfg)
        return masked_view(cfg)

    @router.post("/provider/test")
    def post_test(body: dict[str, str]) -> dict[str, Any]:
        name = body.get("provider", "")
        return test_provider(name, load_server_config())

    return router
