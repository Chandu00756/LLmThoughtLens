"""Provider configuration API — keys, models, and a real "Test" endpoint.

Persists multi-provider settings to ``~/.LLmThoughtLens/server.json`` (the
config home from :mod:`LLmThoughtLens.config_paths`, overridable with
``LLMTHOUGHTLENS_HOME``).  API keys are stored locally only, in a file written
atomically with owner-only ``0o600`` permissions; the config home is created
``0o700`` and an existing one with looser permissions is tightened to
``0o700`` on every save (the user's home directory itself, filesystem roots
and directories owned by another user are left alone — see
:func:`~LLmThoughtLens.config_paths.write_private_text`).  The GET endpoint
returns keys **masked** so the dashboard never echoes a full secret back.
Default model ids come from :mod:`LLmThoughtLens.providers.defaults`.

Loading is forgiving: a hand-edited or corrupt ``server.json`` never crashes
the server — each invalid field falls back to its default with a logged
warning.  This module needs only the ``server`` extra (no Textual).
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from LLmThoughtLens.config_paths import CONFIG_DIR, SERVER_CONFIG_PATH, write_private_text
from LLmThoughtLens.providers.defaults import (
    default_model,
    default_ollama_url,
    provider_kwargs,
    upgrade_retired_model,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CONFIG_DIR",
    "SERVER_CONFIG_PATH",
    "ProviderSettings",
    "ServerConfig",
    "build_provider",
    "build_router",
    "load_server_config",
    "masked_view",
    "save_server_config",
    "test_provider",
]


def _default_provider_settings() -> dict[str, dict[str, Any]]:
    """Fresh per-provider settings seeded from the shared defaults."""

    def slot(provider: str, base_url: str = "") -> dict[str, Any]:
        return {
            "api_key": "",
            "model": default_model(provider),
            "base_url": base_url,
            "device": "auto",
        }

    return {
        "openai": slot("openai"),
        "anthropic": slot("anthropic"),
        "huggingface": slot("huggingface"),
        "ollama": slot("ollama", default_ollama_url()),
        "mock": slot("mock"),
    }


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
    providers: dict[str, dict[str, Any]] = field(default_factory=_default_provider_settings)
    top_k_features: int = 20
    attribution_threshold: float = 0.05
    blackbox_budget: int = 16

    def settings_for(self, provider: str) -> ProviderSettings:
        raw = self.providers.get(provider, {})
        defaults = ProviderSettings()

        def text(key: str) -> str:
            value = raw.get(key)
            return value if isinstance(value, str) else getattr(defaults, key)

        return ProviderSettings(
            api_key=text("api_key"),
            model=text("model"),
            base_url=text("base_url"),
            device=text("device") or defaults.device,
        )


#: String-valued per-provider settings; anything else in a slot is passed through.
_PROVIDER_STR_FIELDS = ("api_key", "model", "base_url", "device")


def _as_int(value: Any, minimum: int) -> int | None:
    """*value* as an int >= *minimum*, or ``None`` if it isn't one.

    Accepts JSON integers, integral floats (``20.0``) and numeric strings
    (``"20"``); rejects booleans, ``null``, lists, NaN/inf and junk.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            return None
    if isinstance(value, int):
        number = value
    elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
        number = int(value)
    else:
        return None
    return number if number >= minimum else None


def _as_float(value: Any, minimum: float) -> float | None:
    """*value* as a finite float >= *minimum*, or ``None`` if it isn't one."""
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            return None
    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= minimum else None


def _invalid(field_name: str, value: Any, fallback: Any) -> None:
    logger.warning(
        "%s: ignoring invalid %s=%r; using %r", SERVER_CONFIG_PATH, field_name, value, fallback
    )


def _known_providers(cfg: ServerConfig) -> set[str]:
    try:
        from LLmThoughtLens.providers.registry import list_providers

        registered = set(list_providers())
    except Exception:  # noqa: BLE001 — never let a registry problem break config loading
        registered = set()
    return registered | set(cfg.providers)


def load_server_config() -> ServerConfig:
    """Load the persisted server config, falling back to defaults.

    Never raises on bad content: a missing / unreadable / non-object file
    yields the defaults, and each invalid field (wrong type, out of range,
    unknown ``active_provider``, non-string provider setting) falls back to
    its default with a logged warning.

    Model ids that earlier releases persisted as defaults but which the
    upstream provider has since retired are upgraded to the current default
    (see :func:`~LLmThoughtLens.providers.defaults.upgrade_retired_model`).
    """
    cfg = ServerConfig()
    try:
        raw = json.loads(SERVER_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # missing, unreadable, or not valid JSON/UTF-8
        return cfg
    if not isinstance(raw, dict):
        logger.warning("%s: expected a JSON object; using defaults", SERVER_CONFIG_PATH)
        return cfg

    providers = raw.get("providers", {})
    if not isinstance(providers, dict):
        _invalid("providers", providers, "defaults")
        providers = {}
    for name, settings in providers.items():
        if not isinstance(name, str) or not name or not isinstance(settings, dict):
            _invalid(f"providers[{name!r}]", settings, "defaults")
            continue
        clean = dict(settings)
        for key in _PROVIDER_STR_FIELDS:
            if key in clean and not isinstance(clean[key], str):
                _invalid(f"providers.{name}.{key}", clean.pop(key), "default")
        slot = cfg.providers.setdefault(
            name, {"api_key": "", "model": "", "base_url": "", "device": "auto"}
        )
        slot.update(clean)
        slot["model"] = upgrade_retired_model(name, slot["model"])

    for key, minimum in (("top_k_features", 1), ("blackbox_budget", 0)):
        if key in raw:
            number = _as_int(raw[key], minimum)
            if number is None:
                _invalid(key, raw[key], getattr(cfg, key))
            else:
                setattr(cfg, key, number)
    if "attribution_threshold" in raw:
        threshold = _as_float(raw["attribution_threshold"], 0.0)
        if threshold is None:
            _invalid(
                "attribution_threshold", raw["attribution_threshold"], cfg.attribution_threshold
            )
        else:
            cfg.attribution_threshold = threshold

    if "active_provider" in raw:
        active = raw["active_provider"]
        if isinstance(active, str) and active in _known_providers(cfg):
            cfg.active_provider = active
        else:
            _invalid("active_provider", active, cfg.active_provider)
    return cfg


def save_server_config(cfg: ServerConfig) -> None:
    """Persist the server config atomically with owner-only permissions.

    The config home is created (or tightened to) ``0o700`` and the file is
    written via temp-file + :func:`os.replace` as ``0o600`` (see
    :func:`~LLmThoughtLens.config_paths.write_private_text`).
    """
    write_private_text(SERVER_CONFIG_PATH, json.dumps(asdict(cfg), indent=2), tighten_dir=True)


def _mask(secret: str) -> str:
    if not secret:
        return ""
    if len(secret) <= 8:
        return "•" * len(secret)
    return f"{secret[:4]}…{secret[-4:]}"


def _redact_keys(text: str, cfg: ServerConfig) -> str:
    """Replace any configured raw API key that appears in *text* with its mask."""
    for settings in cfg.providers.values():
        key = settings.get("api_key") or ""
        if isinstance(key, str) and len(key) >= 8 and key in text:
            text = text.replace(key, _mask(key))
    return text


def masked_view(cfg: ServerConfig) -> dict[str, Any]:
    """Config dict with API keys masked, for sending to the browser.

    Each provider entry also carries ``default_model`` so the UI can show
    what an empty model field resolves to.
    """
    out = asdict(cfg)
    for name, settings in out["providers"].items():
        settings["default_model"] = default_model(name)
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
    # Empty model/base_url fall back to LLmThoughtLens.providers.defaults; an
    # empty api_key falls back to the provider's own env var.
    kwargs = provider_kwargs(
        name,
        s.model,
        api_key=s.api_key,
        base_url=s.base_url,
        device=s.device or "auto",
    )
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
            # Reasoning models (o-series, gpt-5+) spend a 1-token budget on hidden
            # reasoning, so let the provider pick its own budget for them.
            kw: dict[str, Any] = (
                {} if getattr(provider, "is_reasoning_model", False) else {"max_tokens": 1}
            )
            out = provider.run("ping", **kw)
            detail = f"{name} responded ({out.output_token!r})"
            if out.meta.get("budget_exhausted"):
                detail += "; the token budget ran out before any visible output"
            if out.meta.get("has_logprobs") is False:
                detail += "; no token logprobs (top-token probability is a 1.0 placeholder)"
            return {
                "ok": True,
                "detail": detail,
                "evidence_kind": provider.evidence_kind,
                "has_logprobs": out.meta.get("has_logprobs"),
                "budget_exhausted": bool(out.meta.get("budget_exhausted", False)),
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
        detail = _redact_keys(f"{type(exc).__name__}: {exc}", cfg)
        return {"ok": False, "detail": detail, "evidence_kind": "unknown"}
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
