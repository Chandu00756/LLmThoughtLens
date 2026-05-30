"""Tests for the ThoughtLens SDK — observe / trace / record / wrap_openai.

Uses the mock provider so no network or keys are needed.  Dashboard pushes are
captured by monkeypatching ``_push`` so we assert real payloads without a server.
"""

from __future__ import annotations

from LLmThoughtLens import sdk
from LLmThoughtLens.scope import TraceResult


class TestTrace:
    def test_trace_returns_real_result(self):
        r = sdk.trace("the capital of France is", provider="mock")
        assert isinstance(r, TraceResult)
        assert r.output_token
        assert len(r.features) > 0
        assert r.evidence_kind == "white_box"

    def test_trace_pushes_to_dashboard(self, monkeypatch):
        pushed = []
        monkeypatch.setattr(
            sdk, "_push", lambda dash, kind, data: pushed.append((dash, kind, data))
        )
        sdk.trace("hello world", provider="mock", dashboard="http://localhost:8000")
        assert pushed
        dash, kind, data = pushed[0]
        assert dash == "http://localhost:8000"
        assert kind == "trace_complete"
        assert "features" in data and "graph" in data


class TestObserve:
    def test_observe_session(self):
        with sdk.observe(provider="mock") as obs:
            out = obs.run("the cat sat")
            assert out.n_tokens == 3
            r = obs.trace("the cat sat")
            assert obs.last_trace is r
            assert r.output_token

    def test_record_exchange_payload(self, monkeypatch):
        pushed = []
        monkeypatch.setattr(sdk, "_push", lambda dash, kind, data: pushed.append((kind, data)))
        with sdk.observe(provider="mock", dashboard="http://x") as obs:
            obs.record("my prompt", "my completion")
        assert pushed[-1][0] == "proxy_exchange"
        assert pushed[-1][1]["prompt"] == "my prompt"
        assert pushed[-1][1]["completion"] == "my completion"


class TestWrapOpenAI:
    def test_wrap_returns_response_verbatim_and_hooks(self, monkeypatch):
        pushed = []
        monkeypatch.setattr(sdk, "_push", lambda dash, kind, data: pushed.append((kind, data)))

        # Minimal fake OpenAI-style client.
        class _Msg:
            content = "Paris"

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        class _Completions:
            def create(self, **kwargs):
                return _Resp()

        class _Chat:
            completions = _Completions()

        class _FakeClient:
            chat = _Chat()
            api_key = "sk-test"

        wrapped = sdk.wrap_openai(_FakeClient(), dashboard="http://dash")
        # Delegation still works for non-intercepted attributes.
        assert wrapped.api_key == "sk-test"

        resp = wrapped.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": "The capital of France is"}]
        )
        # Real response returned unchanged.
        assert resp.choices[0].message.content == "Paris"
        # Exchange was teed.
        assert pushed and pushed[-1][0] == "proxy_exchange"
        ex = pushed[-1][1]
        assert ex["completion"] == "Paris"
        assert "The capital of France is" in ex["prompt"]

    def test_hook_failure_never_breaks_caller(self, monkeypatch):
        def _boom(*a, **k):
            raise RuntimeError("dashboard down")

        monkeypatch.setattr(sdk, "_push", _boom)

        class _Resp:
            choices = []

        class _Completions:
            def create(self, **kwargs):
                return _Resp()

        class _Chat:
            completions = _Completions()

        class _FakeClient:
            chat = _Chat()

        wrapped = sdk.wrap_openai(_FakeClient(), dashboard="http://dash")
        # Even though the hook raises internally, the call must succeed.
        resp = wrapped.chat.completions.create(messages=[{"role": "user", "content": "hi"}])
        assert resp is not None


class TestPushIsBestEffort:
    def test_push_swallows_network_errors(self):
        # Pointing at a dead port must not raise.
        sdk._push("http://127.0.0.1:1", "trace_complete", {"x": 1})
