"""Tests for shared provider defaults and secure config persistence.

* Every entry point (Scope, CLI, SDK, TUI, server config) resolves the same
  default model ids from :mod:`LLmThoughtLens.providers.defaults`.
* ``LLMTHOUGHTLENS_<PROVIDER>_MODEL`` / ``LLMTHOUGHTLENS_OLLAMA_URL`` override them.
* Config files holding API keys are written atomically as ``0o600`` inside a
  ``0o700`` config home (an existing looser home is tightened), symlinked
  config files are followed, and no temp files are left behind.
* A corrupt / hand-edited ``server.json`` never crashes loading.
* The server's config module imports without Textual (``[server]``-only install).
* The Anthropic provider issues a well-formed Messages API request through the
  real installed SDK (via an in-process mock transport — no network).

Every test that persists config redirects the module-level paths to
``tmp_path``; nothing here touches the real ``~/.LLmThoughtLens``.
"""

from __future__ import annotations

import json
import logging
import os
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from LLmThoughtLens.providers import defaults as d
from LLmThoughtLens.providers.defaults import (
    DEFAULT_MODELS,
    DEFAULT_OLLAMA_URL,
    default_model,
    default_ollama_url,
    model_env_var,
    provider_kwargs,
    resolve_model,
    upgrade_retired_model,
)

POSIX_ONLY = pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")

_MODEL_ATTR = {
    "openai": "model",
    "anthropic": "model",
    "ollama": "model",
    "huggingface": "model_name",
}


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Clear default overrides and point every config path at a temp dir."""
    for name in list(os.environ):
        if name.startswith("LLMTHOUGHTLENS_") and (
            name.endswith("_MODEL") or name == "LLMTHOUGHTLENS_OLLAMA_URL"
        ):
            monkeypatch.delenv(name, raising=False)

    import LLmThoughtLens.config_paths as paths
    import LLmThoughtLens.tui.config as tcfg

    home = tmp_path / "home"
    monkeypatch.setattr(paths, "CONFIG_DIR", home)
    monkeypatch.setattr(paths, "CONFIG_PATH", home / "config.json")
    monkeypatch.setattr(paths, "SERVER_CONFIG_PATH", home / "server.json")
    monkeypatch.setattr(tcfg, "CONFIG_DIR", home)
    monkeypatch.setattr(tcfg, "CONFIG_PATH", home / "config.json")
    try:
        import LLmThoughtLens.server.config_api as capi
    except ImportError:  # fastapi extra missing
        return
    monkeypatch.setattr(capi, "CONFIG_DIR", home)
    monkeypatch.setattr(capi, "SERVER_CONFIG_PATH", home / "server.json")


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


# ---------------------------------------------------------------------------
# defaults module
# ---------------------------------------------------------------------------


class TestDefaultsModule:
    def test_expected_defaults(self):
        assert DEFAULT_MODELS["mock"] == ""
        assert DEFAULT_MODELS["openai"] == "gpt-4o-mini"
        assert DEFAULT_MODELS["anthropic"] == "claude-haiku-4-5"
        assert DEFAULT_MODELS["huggingface"] == "gpt2"
        assert DEFAULT_MODELS["ollama"] == "llama3.2"
        assert DEFAULT_OLLAMA_URL == "http://localhost:11434"

    def test_defaults_are_read_only(self):
        with pytest.raises(TypeError):
            DEFAULT_MODELS["openai"] = "x"  # type: ignore[index]

    def test_retired_anthropic_id_is_not_a_default(self):
        assert "claude-3-5-haiku-20241022" not in DEFAULT_MODELS.values()

    def test_env_var_name(self):
        assert model_env_var("anthropic") == "LLMTHOUGHTLENS_ANTHROPIC_MODEL"
        assert model_env_var("my-custom.provider") == "LLMTHOUGHTLENS_MY_CUSTOM_PROVIDER_MODEL"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("LLMTHOUGHTLENS_OPENAI_MODEL", "gpt-4.1-nano")
        assert default_model("openai") == "gpt-4.1-nano"
        assert default_model("anthropic") == DEFAULT_MODELS["anthropic"]

    def test_blank_env_override_is_ignored(self, monkeypatch):
        monkeypatch.setenv("LLMTHOUGHTLENS_OPENAI_MODEL", "   ")
        assert default_model("openai") == DEFAULT_MODELS["openai"]

    def test_unknown_provider_defaults_to_empty(self, monkeypatch):
        assert default_model("nope") == ""
        monkeypatch.setenv("LLMTHOUGHTLENS_NOPE_MODEL", "custom-1")
        assert default_model("nope") == "custom-1"

    def test_resolve_model_prefers_explicit(self, monkeypatch):
        monkeypatch.setenv("LLMTHOUGHTLENS_OPENAI_MODEL", "from-env")
        assert resolve_model("openai", "explicit") == "explicit"
        assert resolve_model("openai", "") == "from-env"
        assert resolve_model("openai", None) == "from-env"

    def test_ollama_url_override(self, monkeypatch):
        assert default_ollama_url() == DEFAULT_OLLAMA_URL
        monkeypatch.setenv("LLMTHOUGHTLENS_OLLAMA_URL", "http://gpu-box:11434")
        assert default_ollama_url() == "http://gpu-box:11434"

    def test_upgrade_retired_model(self):
        assert upgrade_retired_model("anthropic", "claude-3-5-haiku-20241022") == "claude-haiku-4-5"
        assert upgrade_retired_model("anthropic", "claude-sonnet-5-5") == "claude-sonnet-5-5"
        assert upgrade_retired_model("ollama", "llama3.1:8b") == "llama3.1:8b"

    def test_provider_kwargs_shapes(self):
        assert provider_kwargs("mock") == {}
        assert provider_kwargs("custom-thing", "m") == {}
        assert provider_kwargs("openai") == {"model": "gpt-4o-mini", "api_key": None}
        assert provider_kwargs("openai", "", api_key="", base_url="") == {
            "model": "gpt-4o-mini",
            "api_key": None,
        }
        assert provider_kwargs("anthropic", "m", api_key="k", base_url="http://gw") == {
            "model": "m",
            "api_key": "k",
            "base_url": "http://gw",
        }
        assert provider_kwargs("huggingface") == {"model_name": "gpt2"}
        assert provider_kwargs("huggingface", device="cpu") == {
            "model_name": "gpt2",
            "device": "cpu",
        }
        assert provider_kwargs("ollama") == {"model": "llama3.2", "base_url": DEFAULT_OLLAMA_URL}

    def test_package_reexports(self):
        import LLmThoughtLens.providers as providers

        assert providers.DEFAULT_MODELS is d.DEFAULT_MODELS
        assert providers.default_model is d.default_model
        assert providers.DEFAULT_OLLAMA_URL == d.DEFAULT_OLLAMA_URL


# ---------------------------------------------------------------------------
# Consistency across entry points
# ---------------------------------------------------------------------------


def _scope_model(provider: str) -> str:
    from LLmThoughtLens.scope import Scope

    factory = {
        "openai": Scope.from_openai,
        "anthropic": Scope.from_anthropic,
        "huggingface": Scope.from_huggingface,
        "ollama": Scope.from_ollama,
    }[provider]
    return getattr(factory()._provider, _MODEL_ATTR[provider])


def _entry_point_models(provider: str) -> dict[str, str]:
    """Model id each entry point picks for *provider* when none is supplied."""
    from LLmThoughtLens import cli, sdk
    from LLmThoughtLens.tui.config import TUIConfig
    from LLmThoughtLens.tui.screens import build_provider_from_config

    attr = _MODEL_ATTR[provider]
    tui_provider = build_provider_from_config(TUIConfig(provider=provider))
    assert tui_provider is not None
    out = {
        "scope": _scope_model(provider),
        "cli": getattr(cli._make_provider(provider, "", None, None), attr),
        "sdk": getattr(sdk._build_provider(provider), attr),
        "tui": getattr(tui_provider, attr),
    }
    capi = pytest.importorskip("LLmThoughtLens.server.config_api")
    server_cfg = capi.ServerConfig()
    out["server_default"] = server_cfg.providers[provider]["model"]
    out["server_build"] = getattr(capi.build_provider(provider, server_cfg), attr)
    out["server_view"] = capi.masked_view(server_cfg)["providers"][provider]["default_model"]
    return out


def _skip_if_missing(provider: str) -> None:
    mod = {"openai": "openai", "anthropic": "anthropic", "ollama": "httpx", "huggingface": "torch"}
    pytest.importorskip(mod[provider])
    # _entry_point_models also checks the TUI entry point (tui.screens needs Textual).
    pytest.importorskip("textual")


@pytest.mark.parametrize("provider", ["openai", "anthropic", "ollama", "huggingface"])
def test_defaults_consistent_across_entry_points(provider):
    _skip_if_missing(provider)
    models = _entry_point_models(provider)
    assert set(models.values()) == {DEFAULT_MODELS[provider]}, models


@pytest.mark.parametrize("provider", ["openai", "anthropic", "ollama", "huggingface"])
def test_env_override_reaches_every_entry_point(provider, monkeypatch):
    _skip_if_missing(provider)
    monkeypatch.setenv(model_env_var(provider), f"env-{provider}-model")
    models = _entry_point_models(provider)
    assert set(models.values()) == {f"env-{provider}-model"}, models


def test_explicit_model_beats_env_override(monkeypatch):
    pytest.importorskip("anthropic")
    from LLmThoughtLens import cli, sdk
    from LLmThoughtLens.scope import Scope

    monkeypatch.setenv("LLMTHOUGHTLENS_ANTHROPIC_MODEL", "from-env")
    assert Scope.from_anthropic("claude-sonnet-5-5")._provider.model == "claude-sonnet-5-5"
    assert Scope.from_anthropic(model="claude-sonnet-5-5")._provider.model == "claude-sonnet-5-5"
    assert cli._make_provider("anthropic", "explicit", None, None).model == "explicit"
    assert sdk._build_provider("anthropic", "explicit").model == "explicit"
    # Extra provider kwargs pass through while the model still resolves from env.
    p = sdk._build_provider("anthropic", "", max_tokens=12)
    assert (p.model, p.max_tokens) == ("from-env", 12)


def test_ollama_url_consistent_and_overridable(monkeypatch):
    pytest.importorskip("httpx")
    pytest.importorskip("textual")
    from LLmThoughtLens import cli, sdk
    from LLmThoughtLens.scope import Scope
    from LLmThoughtLens.tui.config import TUIConfig
    from LLmThoughtLens.tui.screens import build_provider_from_config

    capi = pytest.importorskip("LLmThoughtLens.server.config_api")

    def urls() -> set[str]:
        tui = build_provider_from_config(TUIConfig(provider="ollama"))
        return {
            Scope.from_ollama()._provider.base_url,
            cli._make_provider("ollama", "", None, None).base_url,
            sdk._build_provider("ollama").base_url,
            tui.base_url,
            capi.build_provider("ollama", capi.ServerConfig()).base_url,
        }

    assert urls() == {DEFAULT_OLLAMA_URL}
    monkeypatch.setenv("LLMTHOUGHTLENS_OLLAMA_URL", "http://gpu-box:11434/")
    assert urls() == {"http://gpu-box:11434"}


def test_positional_and_keyword_factory_calls_still_work():
    pytest.importorskip("openai")
    pytest.importorskip("httpx")
    from LLmThoughtLens.scope import Scope

    assert Scope.from_openai("gpt-4.1-mini", "sk-x")._provider.model == "gpt-4.1-mini"
    assert Scope.from_openai(model="gpt-4.1-mini")._provider.model == "gpt-4.1-mini"
    s = Scope.from_ollama("qwen3:4b", "http://h:1/")
    assert (s._provider.model, s._provider.base_url) == ("qwen3:4b", "http://h:1")


def test_tui_does_not_forward_ollama_url_to_api_providers():
    pytest.importorskip("openai")
    pytest.importorskip("textual")
    from LLmThoughtLens.tui.config import TUIConfig
    from LLmThoughtLens.tui.screens import build_provider_from_config

    p = build_provider_from_config(TUIConfig(provider="openai", base_url="http://localhost:11434"))
    assert p is not None
    assert p._base_url is None


# ---------------------------------------------------------------------------
# Secure persistence — server config
# ---------------------------------------------------------------------------


@pytest.fixture
def capi():
    return pytest.importorskip("LLmThoughtLens.server.config_api")


class TestServerConfigPersistence:
    def test_creates_private_dir_and_file(self, capi):
        cfg = capi.ServerConfig()
        cfg.providers["openai"]["api_key"] = "sk-test-0123456789abcdef"
        capi.save_server_config(cfg)
        path = capi.SERVER_CONFIG_PATH
        assert path.exists()
        assert json.loads(path.read_text())["providers"]["openai"]["api_key"].startswith("sk-test")
        if os.name != "nt":
            assert _mode(path) == 0o600
            assert _mode(path.parent) == 0o700

    @POSIX_ONLY
    def test_rewrite_tightens_existing_world_readable_file(self, capi):
        path = capi.SERVER_CONFIG_PATH
        path.parent.mkdir(parents=True)
        path.write_text("{}")
        os.chmod(path, 0o644)
        capi.save_server_config(capi.ServerConfig())
        assert _mode(path) == 0o600

    def test_atomic_write_leaves_no_temp_files(self, capi):
        for i in range(3):
            cfg = capi.ServerConfig()
            cfg.top_k_features = i
            capi.save_server_config(cfg)
        assert sorted(p.name for p in capi.SERVER_CONFIG_PATH.parent.iterdir()) == ["server.json"]
        assert capi.load_server_config().top_k_features == 2

    def test_failed_write_keeps_old_file_and_cleans_up(self, capi, monkeypatch):
        capi.save_server_config(capi.ServerConfig())
        before = capi.SERVER_CONFIG_PATH.read_text()

        def boom(*_a, **_k):
            raise OSError("disk full")

        cfg = capi.ServerConfig()
        cfg.top_k_features = 99
        # Scoped patch: a bare monkeypatch.undo() would also revert the autouse
        # path isolation and point back at the real config dir.
        with monkeypatch.context() as m:
            m.setattr("LLmThoughtLens.config_paths.os.replace", boom)
            with pytest.raises(OSError, match="disk full"):
                capi.save_server_config(cfg)
        assert capi.SERVER_CONFIG_PATH.read_text() == before
        assert [p.name for p in capi.SERVER_CONFIG_PATH.parent.iterdir()] == ["server.json"]

    def test_load_upgrades_retired_persisted_default(self, capi):
        path = capi.SERVER_CONFIG_PATH
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "providers": {
                        "anthropic": {"model": "claude-3-5-haiku-20241022"},
                        "ollama": {"model": "llama3.1:8b"},
                    }
                }
            )
        )
        cfg = capi.load_server_config()
        assert cfg.providers["anthropic"]["model"] == "claude-haiku-4-5"
        assert cfg.providers["ollama"]["model"] == "llama3.1:8b"  # still valid — untouched

    @pytest.mark.parametrize("payload", ["[1, 2]", "not json", '{"providers": [1]}', "\udcff"])
    def test_load_tolerates_corrupt_files(self, capi, payload):
        path = capi.SERVER_CONFIG_PATH
        path.parent.mkdir(parents=True)
        path.write_bytes(payload.encode("utf-8", "surrogateescape"))
        cfg = capi.load_server_config()
        assert cfg.providers["openai"]["model"] == DEFAULT_MODELS["openai"]

    def test_masked_view_never_returns_raw_key(self, capi):
        cfg = capi.ServerConfig()
        cfg.providers["anthropic"]["api_key"] = "sk-ant-secret-value-123456"
        view = capi.masked_view(cfg)
        assert "secret-value" not in json.dumps(view)
        assert view["providers"]["anthropic"]["api_key_set"] is True
        assert view["providers"]["anthropic"]["default_model"] == "claude-haiku-4-5"
        assert cfg.providers["anthropic"]["api_key"] == "sk-ant-secret-value-123456"  # untouched

    def test_test_endpoint_redacts_key_in_error_detail(self, capi, monkeypatch):
        cfg = capi.ServerConfig()
        key = "sk-live-abcdefghijklmnop"
        cfg.providers["openai"]["api_key"] = key

        def explode(name, _cfg):
            raise RuntimeError(f"upstream rejected key {key}")

        monkeypatch.setattr(capi, "build_provider", explode)
        result = capi.test_provider("openai", cfg)
        assert result["ok"] is False
        assert key not in result["detail"]
        assert "sk-l…mnop" in result["detail"]


# ---------------------------------------------------------------------------
# Secure persistence — TUI config
# ---------------------------------------------------------------------------


class TestTUIConfigPersistence:
    def test_creates_private_dir_and_file(self):
        import LLmThoughtLens.tui.config as tcfg

        tcfg.save_config(tcfg.TUIConfig(provider="openai", api_key="sk-k", save_api_key=True))
        assert tcfg.CONFIG_PATH.exists()
        assert json.loads(tcfg.CONFIG_PATH.read_text())["api_key"] == "sk-k"
        if os.name != "nt":
            assert _mode(tcfg.CONFIG_PATH) == 0o600
            assert _mode(tcfg.CONFIG_DIR) == 0o700

    def test_key_not_persisted_without_opt_in(self):
        import LLmThoughtLens.tui.config as tcfg

        tcfg.save_config(tcfg.TUIConfig(api_key="sk-should-not-persist", save_api_key=False))
        assert "sk-should-not-persist" not in tcfg.CONFIG_PATH.read_text()

    @POSIX_ONLY
    def test_rewrite_tightens_existing_file(self):
        import LLmThoughtLens.tui.config as tcfg

        tcfg.CONFIG_DIR.mkdir(parents=True)
        tcfg.CONFIG_PATH.write_text("{}")
        os.chmod(tcfg.CONFIG_PATH, 0o644)
        tcfg.save_config(tcfg.TUIConfig())
        assert _mode(tcfg.CONFIG_PATH) == 0o600

    @POSIX_ONLY
    def test_existing_loose_config_home_is_tightened(self):
        import LLmThoughtLens.tui.config as tcfg

        tcfg.CONFIG_DIR.mkdir(parents=True)
        os.chmod(tcfg.CONFIG_DIR, 0o755)
        tcfg.save_config(tcfg.TUIConfig())
        assert _mode(tcfg.CONFIG_DIR) == 0o700
        assert _mode(tcfg.CONFIG_PATH) == 0o600

    def test_roundtrip_and_no_temp_files(self):
        import LLmThoughtLens.tui.config as tcfg

        cfg = tcfg.TUIConfig(provider="ollama", model="qwen3:4b")
        tcfg.save_config(cfg)
        tcfg.save_config(cfg)
        loaded = tcfg.load_config()
        assert (loaded.provider, loaded.model) == ("ollama", "qwen3:4b")
        assert [p.name for p in tcfg.CONFIG_DIR.iterdir()] == ["config.json"]

    def test_load_upgrades_retired_model_and_tolerates_junk(self):
        import LLmThoughtLens.tui.config as tcfg

        tcfg.CONFIG_DIR.mkdir(parents=True)
        tcfg.CONFIG_PATH.write_text(
            json.dumps({"provider": "anthropic", "model": "claude-3-5-haiku-20241022"})
        )
        assert tcfg.load_config().model == "claude-haiku-4-5"
        tcfg.CONFIG_PATH.write_text("[]")
        assert tcfg.load_config().provider == "mock"
        tcfg.CONFIG_PATH.write_text(json.dumps({"provider": ["x"], "model": ["y"]}))
        assert tcfg.load_config().provider == "mock"


# ---------------------------------------------------------------------------
# config_paths — the dependency-free private-file helpers
# ---------------------------------------------------------------------------


class TestPrivateFileHelpers:
    def test_tui_config_reexports_the_neutral_helpers(self):
        import LLmThoughtLens.config_paths as paths
        import LLmThoughtLens.tui.config as tcfg

        assert tcfg.write_private_text is paths.write_private_text
        assert tcfg.ensure_private_dir is paths.ensure_private_dir
        assert (tcfg.PRIVATE_DIR_MODE, tcfg.PRIVATE_FILE_MODE) == (0o700, 0o600)

    def test_default_paths_follow_llmthoughtlens_home(self, tmp_path):
        code = (
            "import LLmThoughtLens.config_paths as p; "
            "print(p.CONFIG_DIR); print(p.CONFIG_PATH.name); print(p.SERVER_CONFIG_PATH.name)"
        )
        env = {**os.environ, "LLMTHOUGHTLENS_HOME": str(tmp_path / "custom")}
        out = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
        ).stdout.split()
        assert out == [str(tmp_path / "custom"), "config.json", "server.json"]

    @POSIX_ONLY
    def test_unrelated_existing_dir_is_left_alone_by_default(self, tmp_path):
        from LLmThoughtLens.config_paths import write_private_text

        shared = tmp_path / "shared"
        shared.mkdir()
        os.chmod(shared, 0o755)
        write_private_text(shared / "notes.json", "{}")
        assert _mode(shared) == 0o755
        assert _mode(shared / "notes.json") == 0o600

    @POSIX_ONLY
    def test_home_directory_itself_is_never_tightened(self, tmp_path, monkeypatch):
        from LLmThoughtLens.config_paths import write_private_text

        fake_home = tmp_path / "userhome"
        fake_home.mkdir()
        os.chmod(fake_home, 0o755)
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
        write_private_text(fake_home / "config.json", "{}", tighten_dir=True)
        assert _mode(fake_home) == 0o755
        assert _mode(fake_home / "config.json") == 0o600

    @POSIX_ONLY
    def test_dir_owned_by_someone_else_is_not_chmodded(self, tmp_path, monkeypatch):
        from LLmThoughtLens.config_paths import ensure_private_dir

        d = tmp_path / "theirs"
        d.mkdir()
        os.chmod(d, 0o755)
        monkeypatch.setattr(os, "getuid", lambda: os.stat(d).st_uid + 1)
        ensure_private_dir(d, tighten_existing=True)
        assert _mode(d) == 0o755

    @POSIX_ONLY
    def test_symlinked_config_file_is_followed_and_link_kept(self, tmp_path):
        import LLmThoughtLens.tui.config as tcfg

        dotfiles = tmp_path / "dotfiles"
        dotfiles.mkdir()
        target = dotfiles / "llmthoughtlens.json"
        target.write_text("{}")
        os.chmod(target, 0o644)
        tcfg.CONFIG_DIR.mkdir(parents=True)
        tcfg.CONFIG_PATH.symlink_to(target)

        tcfg.save_config(tcfg.TUIConfig(model="via-link"))
        assert tcfg.CONFIG_PATH.is_symlink()  # the user's link survives
        assert os.readlink(tcfg.CONFIG_PATH) == str(target)
        assert json.loads(target.read_text())["model"] == "via-link"
        assert _mode(target) == 0o600
        assert tcfg.load_config().model == "via-link"
        assert sorted(p.name for p in dotfiles.iterdir()) == ["llmthoughtlens.json"]
        assert [p.name for p in tcfg.CONFIG_DIR.iterdir()] == ["config.json"]

    @POSIX_ONLY
    def test_dangling_symlink_into_existing_dir_creates_target(self, tmp_path):
        from LLmThoughtLens.config_paths import write_private_text

        link = tmp_path / "home" / "server.json"
        link.parent.mkdir()
        target = tmp_path / "elsewhere.json"
        link.symlink_to(target)
        write_private_text(link, '{"ok": 1}')
        assert link.is_symlink() and json.loads(target.read_text()) == {"ok": 1}
        assert _mode(target) == 0o600

    @POSIX_ONLY
    def test_dangling_symlink_into_missing_dir_is_refused(self, tmp_path):
        from LLmThoughtLens.config_paths import write_private_text

        link = tmp_path / "home" / "server.json"
        link.parent.mkdir()
        link.symlink_to(tmp_path / "missing" / "deep" / "server.json")
        with pytest.raises(FileNotFoundError, match="whose directory does not exist"):
            write_private_text(link, "{}")
        assert not (tmp_path / "missing").exists()  # nothing created outside the home
        assert link.is_symlink()


# ---------------------------------------------------------------------------
# server config — robust loading of hand-edited / corrupt files
# ---------------------------------------------------------------------------


def _write_server_json(capi, payload) -> None:
    capi.SERVER_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    capi.SERVER_CONFIG_PATH.write_text(json.dumps(payload))


class TestServerConfigRobustLoading:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("top_k_features", "abc"),
            ("top_k_features", None),
            ("top_k_features", [5]),
            ("top_k_features", True),
            ("top_k_features", 0),
            ("top_k_features", -3),
            ("top_k_features", 2.5),
            ("top_k_features", {"k": 1}),
            ("attribution_threshold", [1]),
            ("attribution_threshold", None),
            ("attribution_threshold", "lots"),
            ("attribution_threshold", -0.1),
            ("attribution_threshold", "nan"),
            ("attribution_threshold", "inf"),
            ("attribution_threshold", False),
            ("blackbox_budget", "x"),
            ("blackbox_budget", None),
            ("blackbox_budget", -1),
            ("active_provider", 42),
            ("active_provider", None),
            ("active_provider", ["mock"]),
            ("active_provider", "not-a-provider"),
            ("active_provider", ""),
        ],
    )
    def test_invalid_scalar_falls_back_to_default_with_warning(self, capi, caplog, field, value):
        default = getattr(capi.ServerConfig(), field)
        neighbour = "top_k_features" if field != "top_k_features" else "blackbox_budget"
        _write_server_json(capi, {neighbour: 7, field: value})
        with caplog.at_level(logging.WARNING, logger=capi.__name__):
            cfg = capi.load_server_config()
        assert getattr(cfg, neighbour) == 7  # valid neighbours still load
        assert getattr(cfg, field) == default
        assert any(field in rec.getMessage() for rec in caplog.records)

    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            ({"top_k_features": 7}, ("top_k_features", 7)),
            ({"top_k_features": "12"}, ("top_k_features", 12)),
            ({"top_k_features": 9.0}, ("top_k_features", 9)),
            ({"attribution_threshold": 0}, ("attribution_threshold", 0.0)),
            ({"attribution_threshold": "0.25"}, ("attribution_threshold", 0.25)),
            ({"blackbox_budget": 0}, ("blackbox_budget", 0)),
            ({"active_provider": "mock"}, ("active_provider", "mock")),
            ({"active_provider": "openai"}, ("active_provider", "openai")),
        ],
    )
    def test_valid_values_are_kept(self, capi, payload, expected):
        _write_server_json(capi, payload)
        field, value = expected
        loaded = getattr(capi.load_server_config(), field)
        assert loaded == value and type(loaded) is type(value)

    def test_custom_provider_from_file_can_be_active(self, capi):
        _write_server_json(
            capi,
            {"active_provider": "my-gateway", "providers": {"my-gateway": {"model": "m1"}}},
        )
        cfg = capi.load_server_config()
        assert cfg.active_provider == "my-gateway"
        assert cfg.providers["my-gateway"]["model"] == "m1"
        assert cfg.settings_for("my-gateway").device == "auto"

    def test_non_string_provider_settings_fall_back(self, capi, caplog):
        _write_server_json(
            capi,
            {
                "providers": {
                    "openai": {"api_key": 12345, "model": ["x"], "base_url": None, "extra": 1},
                    "ollama": {"base_url": "http://gpu:11434", "device": 3},
                    "broken": "not-a-dict",
                }
            },
        )
        with caplog.at_level(logging.WARNING, logger=capi.__name__):
            cfg = capi.load_server_config()
        openai = cfg.providers["openai"]
        assert openai["api_key"] == "" and openai["model"] == DEFAULT_MODELS["openai"]
        assert openai["base_url"] == "" and openai["extra"] == 1
        assert cfg.providers["ollama"]["base_url"] == "http://gpu:11434"
        assert cfg.providers["ollama"]["device"] == "auto"
        assert "broken" not in cfg.providers
        s = cfg.settings_for("openai")
        assert (s.api_key, s.base_url) == ("", "")
        assert len(caplog.records) >= 5

    def test_settings_for_tolerates_junk_in_memory(self, capi):
        cfg = capi.ServerConfig()
        cfg.providers["openai"] = {"api_key": None, "model": 3, "device": ""}
        s = cfg.settings_for("openai")
        assert (s.api_key, s.model, s.base_url, s.device) == ("", "", "", "auto")

    def test_robust_config_round_trips_through_save(self, capi):
        _write_server_json(capi, {"top_k_features": "abc", "active_provider": 1})
        capi.save_server_config(capi.load_server_config())
        saved = json.loads(capi.SERVER_CONFIG_PATH.read_text())
        assert saved["top_k_features"] == 20 and saved["active_provider"] == "ollama"


# ---------------------------------------------------------------------------
# [server]-only install: no Textual needed
# ---------------------------------------------------------------------------

_NO_TEXTUAL = textwrap.dedent(
    """
    import importlib.util, sys
    sys.modules["textual"] = None  # import textual -> ImportError, like a [server] install
    sys.modules["rapidfuzz"] = None
    import LLmThoughtLens.config_paths
    import LLmThoughtLens.tui.config as tcfg
    import LLmThoughtLens.tui as tui_pkg
    assert tcfg.TUIConfig().provider == "mock"
    try:
        tui_pkg.LLmThoughtLensApp
    except ImportError:
        pass
    else:
        raise SystemExit("LLmThoughtLensApp resolved without textual?!")
    if {check_server}:
        import LLmThoughtLens.server.config_api as capi
        import LLmThoughtLens.server.app as server_app
        assert capi.load_server_config().active_provider
        server_app.create_app()
    assert not any(m == "textual" or m.startswith("textual.") for m in sys.modules
                   if sys.modules[m] is not None), "textual was imported"
    print("NO_TEXTUAL_OK")
    """
)


def _server_extra_present() -> bool:
    import importlib.util

    return all(importlib.util.find_spec(m) is not None for m in ("fastapi", "pydantic"))


@pytest.mark.parametrize("check_server", [False, True], ids=["config-only", "server-app"])
def test_server_and_config_import_without_textual(tmp_path, check_server):
    """Regression: server.config_api imported tui.config -> tui/__init__ -> textual."""
    if check_server and not _server_extra_present():
        pytest.skip("server extra (fastapi) not installed")
    env = {**os.environ, "LLMTHOUGHTLENS_HOME": str(tmp_path / "home")}
    proc = subprocess.run(
        [sys.executable, "-c", _NO_TEXTUAL.format(check_server=check_server)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "NO_TEXTUAL_OK" in proc.stdout
    assert not (tmp_path / "home").exists()  # importing never writes config


def test_tui_package_still_exposes_app_lazily():
    pytest.importorskip("textual")
    import LLmThoughtLens.tui as tui_pkg
    from LLmThoughtLens.tui.app import LLmThoughtLensApp, run_tui

    assert tui_pkg.LLmThoughtLensApp is LLmThoughtLensApp
    assert tui_pkg.run_tui is run_tui
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        _ = tui_pkg.nope


# ---------------------------------------------------------------------------
# Anthropic provider — real SDK request shape, no network
# ---------------------------------------------------------------------------


def _anthropic_with_transport(handler):
    anthropic = pytest.importorskip("anthropic")
    httpx = pytest.importorskip("httpx")
    from LLmThoughtLens.providers.anthropic_provider import AnthropicProvider

    provider = AnthropicProvider(api_key="sk-ant-test")
    provider._client = anthropic.Anthropic(
        api_key="sk-ant-test",
        base_url="http://anthropic.test",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return provider


def _message(content, stop_reason="end_turn", stop_details=None):
    body = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-4-5",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 6, "output_tokens": 3},
    }
    if stop_details is not None:
        body["stop_details"] = stop_details
    return body


class TestAnthropicProvider:
    def test_default_model_is_current_haiku(self):
        pytest.importorskip("anthropic")
        from LLmThoughtLens.providers.anthropic_provider import AnthropicProvider

        p = AnthropicProvider(api_key="x")
        assert p.model == "claude-haiku-4-5"
        assert p.model_id == "anthropic/claude-haiku-4-5"

    def test_messages_request_shape_and_parsing(self):
        httpx = pytest.importorskip("httpx")
        seen = {}

        def handler(request):
            seen["path"] = request.url.path
            seen["headers"] = request.headers
            seen["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json=_message(
                    [
                        {"type": "thinking", "thinking": "", "signature": "sig"},
                        {"type": "text", "text": "Paris is"},
                    ]
                ),
            )

        provider = _anthropic_with_transport(handler)
        out = provider.run("The capital of France is", max_tokens=7)
        assert seen["path"] == "/v1/messages"
        assert seen["headers"]["x-api-key"] == "sk-ant-test"
        assert seen["body"] == {
            "model": "claude-haiku-4-5",
            "max_tokens": 7,
            "messages": [{"role": "user", "content": "The capital of France is"}],
        }
        assert out.evidence_kind == "black_box"
        assert out.meta["completion"] == "Paris is"  # thinking block ignored
        assert out.top_tokens == [("Paris", 1.0)]
        assert out.logits is None and out.activations is None  # nothing fabricated
        assert out.meta["usage"] == {"input_tokens": 6, "output_tokens": 3}
        assert "refusal" not in out.meta

    def test_refusal_is_flagged_not_presented_as_answer(self):
        httpx = pytest.importorskip("httpx")

        def handler(_request):
            return httpx.Response(
                200,
                json=_message(
                    [],
                    stop_reason="refusal",
                    stop_details={"type": "refusal", "category": "cyber", "explanation": "no"},
                ),
            )

        out = _anthropic_with_transport(handler).run("x")
        assert out.meta["stop_reason"] == "refusal"
        assert out.meta["refusal"] == {"category": "cyber", "explanation": "no"}
        assert "declined" in out.meta["evidence_note"]
        assert out.top_tokens == [("", 1.0)]

    def test_base_url_is_passed_to_sdk_client(self):
        pytest.importorskip("anthropic")
        from LLmThoughtLens.providers.anthropic_provider import AnthropicProvider

        client = AnthropicProvider(api_key="x", base_url="http://gateway.test")._ensure_client()
        assert str(client.base_url).startswith("http://gateway.test")
