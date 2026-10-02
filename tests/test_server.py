"""Tests for the live dashboard server — config API, trace API, proxy tee, WebSocket.

All tests use FastAPI's TestClient against the in-process app with the **mock**
provider, so they need no network, API key, or GPU.  The proxy test monkeypatches
the upstream HTTP call so we assert the proxy returns it verbatim AND publishes a
live event — without touching a real API.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402
from LLmThoughtLens.server.app import create_app  # noqa: E402
from LLmThoughtLens.server.bus import get_bus  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Redirect config storage to a temp dir so tests never touch the real config.
    import LLmThoughtLens.server.config_api as capi

    monkeypatch.setattr(capi, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(capi, "SERVER_CONFIG_PATH", tmp_path / "server.json")
    return TestClient(create_app())


class TestHealth:
    def test_health_ok(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert "mock" in body["providers"]

    def test_index_served(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "LLmThoughtLens" in r.text
        assert "/static/app.js" in r.text


class TestConfigAPI:
    def test_config_roundtrip_masks_key(self, client):
        # Save an OpenAI key, then confirm GET returns it MASKED (never raw).
        r = client.post(
            "/api/config/provider",
            json={"provider": "openai", "api_key": "sk-secret-1234567890", "model": "gpt-4o-mini"},
        )
        assert r.status_code == 200
        cfg = client.get("/api/config").json()
        openai = cfg["providers"]["openai"]
        assert openai["api_key_set"] is True
        assert "secret" not in json.dumps(cfg)  # raw key never leaves the server
        assert openai["api_key_masked"].startswith("sk-s")

    def test_defaults_update(self, client):
        r = client.post(
            "/api/config/defaults", json={"top_k_features": 7, "attribution_threshold": 0.2}
        )
        body = r.json()
        assert body["top_k_features"] == 7
        assert body["attribution_threshold"] == 0.2

    def test_provider_test_mock_ok(self, client):
        r = client.post("/api/provider/test", json={"provider": "mock"})
        body = r.json()
        assert body["ok"] is True
        assert body["evidence_kind"] == "white_box"


class TestTraceAPI:
    def test_trace_returns_real_features_and_graph(self, client):
        client.post("/api/config/defaults", json={"active_provider": "mock"})
        r = client.post(
            "/api/trace", json={"prompt": "the capital of France is", "provider": "mock"}
        )
        assert r.status_code == 200
        payload = r.json()
        assert payload["output_token"]
        assert len(payload["features"]) > 0
        assert payload["graph"]["nodes"]
        assert payload["evidence_kind"] == "white_box"
        # Feature payload carries real fields, not placeholders.
        f0 = payload["features"][0]
        assert {"id", "label", "layer", "score", "evidence_kind"} <= set(f0)


class TestProxyTee:
    def test_proxy_returns_upstream_verbatim_and_publishes(self, client, monkeypatch):
        # Fake the upstream OpenAI-compatible response.
        upstream_body = {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Paris"},
                    "logprobs": {
                        "content": [
                            {
                                "token": "Paris",
                                "logprob": -0.2,
                                "top_logprobs": [
                                    {"token": "Paris", "logprob": -0.2},
                                    {"token": "London", "logprob": -2.0},
                                ],
                            }
                        ]
                    },
                }
            ],
        }

        class _FakeResp:
            status_code = 200
            content = json.dumps(upstream_body).encode()
            headers = {"content-type": "application/json"}

            def json(self):
                return upstream_body

        class _FakeAsyncClient:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, *a, **k):
                return _FakeResp()

        import LLmThoughtLens.server.proxy as proxymod

        monkeypatch.setattr(proxymod.httpx, "AsyncClient", _FakeAsyncClient)

        # Subscribe to the bus to capture the published exchange event.
        events: list[dict] = []
        bus = get_bus()
        original_publish = bus.publish

        def _spy(kind, payload=None):
            ev = original_publish(kind, payload)
            events.append(ev)
            return ev

        monkeypatch.setattr(bus, "publish", _spy)

        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "The capital of France is"}],
            },
            headers={"Authorization": "Bearer test"},
        )
        assert r.status_code == 200
        # Verbatim upstream body returned to the caller.
        assert r.json()["choices"][0]["message"]["content"] == "Paris"

        kinds = [e["kind"] for e in events]
        assert "proxy_request" in kinds
        assert "proxy_exchange" in kinds
        exch = next(e for e in events if e["kind"] == "proxy_exchange")
        # Real next-token distribution extracted from upstream logprobs.
        dist = exch["data"]["next_token_distribution"]
        assert dist and dist[0][0] == "Paris"
        assert dist[0][1] > dist[1][1]  # Paris more probable than London


class TestIngest:
    def test_ingest_publishes_to_bus(self, client):
        events: list[dict] = []
        bus = get_bus()
        orig = bus.publish

        def _spy(kind, payload=None):
            ev = orig(kind, payload)
            if kind == "sdk_trace":
                events.append(ev)
            return ev

        import pytest as _pytest  # local alias to keep monkeypatch tidy

        with _pytest.MonkeyPatch.context() as mp:
            mp.setattr(bus, "publish", _spy)
            r = client.post("/api/ingest", json={"kind": "sdk_trace", "data": {"prompt": "x"}})
        assert r.status_code == 200 and r.json()["ok"] is True
        assert events and events[0]["data"]["prompt"] == "x"


class TestAnthropicProxy:
    def test_messages_returns_verbatim_and_tees(self, client, monkeypatch):
        upstream_body = {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Paris"}],
        }

        class _FakeResp:
            status_code = 200
            content = json.dumps(upstream_body).encode()
            headers = {"content-type": "application/json"}

            def json(self):
                return upstream_body

        class _FakeAsyncClient:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, *a, **k):
                return _FakeResp()

        import LLmThoughtLens.server.proxy as proxymod

        monkeypatch.setattr(proxymod.httpx, "AsyncClient", _FakeAsyncClient)

        events: list[dict] = []
        bus = get_bus()
        orig = bus.publish
        monkeypatch.setattr(bus, "publish", lambda k, p=None: events.append((k, p)) or orig(k, p))

        r = client.post(
            "/v1/messages",
            json={
                "model": "claude-3-5-haiku-20241022",
                "messages": [{"role": "user", "content": "cap of France?"}],
            },
            headers={"x-api-key": "test", "anthropic-version": "2023-06-01"},
        )
        assert r.status_code == 200
        assert r.json()["content"][0]["text"] == "Paris"
        kinds = [k for k, _ in events]
        assert "proxy_exchange" in kinds


class TestWebSocket:
    def test_ws_receives_published_event(self, client):
        with client.websocket_connect("/ws") as ws:
            get_bus().publish("unit_test_event", {"hello": "world"})
            # Drain until we see our event (history replay may precede it).
            seen = None
            for _ in range(50):
                msg = ws.receive_json()
                if msg["kind"] == "unit_test_event":
                    seen = msg
                    break
            assert seen is not None
            assert seen["data"]["hello"] == "world"


class TestProxyUpstreamResolution:
    """``_resolve_upstream`` picks the OpenAI-compatible base URL for the proxy."""

    @pytest.fixture
    def capi(self, tmp_path, monkeypatch):
        import LLmThoughtLens.server.config_api as capi

        monkeypatch.setattr(capi, "SERVER_CONFIG_PATH", tmp_path / "server.json")
        monkeypatch.delenv("LLMTHOUGHTLENS_OLLAMA_URL", raising=False)
        return capi

    def _save(self, capi, active: str, **ollama: str) -> None:
        cfg = capi.ServerConfig(active_provider=active)
        cfg.providers["ollama"].update(ollama)
        capi.save_server_config(cfg)

    def test_ollama_with_blank_base_url_uses_shared_default(self, capi):
        from LLmThoughtLens.providers.defaults import DEFAULT_OLLAMA_URL
        from LLmThoughtLens.server.proxy import _resolve_upstream

        self._save(capi, "ollama", base_url="")
        assert _resolve_upstream() == (f"{DEFAULT_OLLAMA_URL}/v1", "ollama")

    def test_ollama_default_honours_env_override(self, capi, monkeypatch):
        """Regression: the proxy hard-coded http://localhost:11434."""
        from LLmThoughtLens.server.proxy import _resolve_upstream

        self._save(capi, "ollama", base_url="   ")
        monkeypatch.setenv("LLMTHOUGHTLENS_OLLAMA_URL", "http://gpu-box:11434/")
        assert _resolve_upstream() == ("http://gpu-box:11434/v1", "ollama")

    def test_configured_ollama_url_wins(self, capi, monkeypatch):
        from LLmThoughtLens.server.proxy import _resolve_upstream

        monkeypatch.setenv("LLMTHOUGHTLENS_OLLAMA_URL", "http://ignored:1")
        self._save(capi, "ollama", base_url="http://box:9/")
        assert _resolve_upstream()[0] == "http://box:9/v1"

    def test_non_ollama_active_uses_openai(self, capi):
        from LLmThoughtLens.server.proxy import _DEFAULT_OPENAI_BASE, _resolve_upstream

        self._save(capi, "mock")
        assert _resolve_upstream() == (_DEFAULT_OPENAI_BASE, "")


class TestConfigAPIWithCorruptFile:
    def test_get_config_survives_wrong_types(self, client, tmp_path):
        (tmp_path / "server.json").write_text(
            json.dumps(
                {
                    "top_k_features": "abc",
                    "attribution_threshold": [1],
                    "blackbox_budget": None,
                    "active_provider": 5,
                    "providers": {"openai": {"api_key": 99, "model": None}},
                }
            )
        )
        r = client.get("/api/config")
        assert r.status_code == 200
        body = r.json()
        assert (body["top_k_features"], body["attribution_threshold"]) == (20, 0.05)
        assert (body["blackbox_budget"], body["active_provider"]) == (16, "ollama")
        assert body["providers"]["openai"]["api_key_set"] is False

    def test_proxy_survives_wrong_types(self, client, tmp_path, monkeypatch):
        """A non-string base_url used to crash the proxy with AttributeError (.strip)."""
        from LLmThoughtLens.providers.defaults import DEFAULT_OLLAMA_URL
        from LLmThoughtLens.server.proxy import _resolve_upstream

        monkeypatch.delenv("LLMTHOUGHTLENS_OLLAMA_URL", raising=False)
        (tmp_path / "server.json").write_text(  # the client fixture's config path
            json.dumps({"providers": {"openai": {"base_url": 123, "api_key": ["k"]}}})
        )
        # Invalid openai fields fall back to "" -> default active provider (ollama).
        assert _resolve_upstream() == (f"{DEFAULT_OLLAMA_URL}/v1", "ollama")
