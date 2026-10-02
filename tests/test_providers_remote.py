"""Offline tests for the black-box remote providers (OpenAI, Anthropic, Ollama).

No network is ever touched:

* OpenAI / Anthropic — the SDK client class is monkeypatched (or a fake client
  is injected into the provider's private ``_client``) so we can assert exactly
  what request the provider builds and how it maps the SDK response.
* Ollama — ``httpx.Client`` is wrapped so every request goes through an
  ``httpx.MockTransport`` handler.

The truthfulness contract is checked everywhere: black-box providers must
never populate ``activations`` / ``attentions`` / ``logits``.
"""

from __future__ import annotations

import importlib.util
import json
import math
from types import SimpleNamespace
from typing import Any

import httpx  # core dependency of LLmThoughtLens
import pytest

# Each provider section skips independently when its optional extra is missing.
needs_openai = pytest.mark.skipif(
    importlib.util.find_spec("openai") is None, reason="openai extra not installed"
)
needs_anthropic = pytest.mark.skipif(
    importlib.util.find_spec("anthropic") is None, reason="anthropic extra not installed"
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _assert_black_box(out: Any) -> None:
    assert out.evidence_kind == "black_box"
    assert out.activations is None
    assert out.attentions is None
    assert out.logits is None
    assert out.token_ids == []
    assert out.has_internals is False


class _RecordingCreate:
    """Callable that records kwargs and returns a canned response."""

    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.response


# ===========================================================================
# OpenAI
# ===========================================================================


def _lp(token: str, logprob: float, top: list[tuple[str, float]] | None = None) -> Any:
    return SimpleNamespace(
        token=token,
        logprob=logprob,
        top_logprobs=[SimpleNamespace(token=t, logprob=lp) for t, lp in (top or [])],
    )


def _openai_response(
    content: str | None,
    logprob_content: list[Any] | None,
    finish_reason: str = "stop",
    usage: Any = None,
) -> Any:
    logprobs = None if logprob_content is None else SimpleNamespace(content=logprob_content)
    choice = SimpleNamespace(
        message=SimpleNamespace(content=content),
        logprobs=logprobs,
        finish_reason=finish_reason,
    )
    return SimpleNamespace(choices=[choice], usage=usage)


def _openai_with_fake_client(response: Any, **kwargs: Any):
    from LLmThoughtLens.providers.openai_provider import OpenAIProvider

    provider = OpenAIProvider(api_key="sk-test", **kwargs)
    create = _RecordingCreate(response)
    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    return provider, create


@needs_openai
class TestOpenAIProviderConstruction:
    def test_identity_and_clamping(self):
        from LLmThoughtLens.providers.openai_provider import OpenAIProvider

        p = OpenAIProvider(model="my-model", api_key="k", top_logprobs=99, max_tokens="12")
        assert p.name == "openai"
        assert p.model_id == "openai/my-model"
        assert p.top_logprobs == 20  # clamped to the API maximum
        assert p.max_tokens == 12
        assert p.supports_internals is False
        assert OpenAIProvider(api_key="k", top_logprobs=0).top_logprobs == 1
        # Reasoning-model knobs default to "auto" and never change chat models.
        assert p.request_logprobs is None and p.reasoning_effort is None
        assert p.reasoning_max_tokens == 4096
        assert p.is_reasoning_model is False

    def test_api_key_falls_back_to_env(self, monkeypatch):
        from LLmThoughtLens.providers.openai_provider import OpenAIProvider

        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        assert OpenAIProvider()._api_key == "sk-from-env"
        assert OpenAIProvider(api_key="sk-explicit")._api_key == "sk-explicit"

    def test_ensure_client_builds_sdk_client_once(self, monkeypatch):
        from LLmThoughtLens.providers.openai_provider import OpenAIProvider

        built: list[dict[str, Any]] = []

        class FakeOpenAI:
            def __init__(self, **kwargs: Any) -> None:
                built.append(kwargs)

        monkeypatch.setattr(pytest.importorskip("openai"), "OpenAI", FakeOpenAI)
        p = OpenAIProvider(
            api_key="sk-x", organization="org-1", base_url="http://proxy/v1", timeout=7
        )
        c1 = p._ensure_client()
        c2 = p._ensure_client()
        assert c1 is c2
        assert isinstance(c1, FakeOpenAI)
        assert built == [
            {
                "api_key": "sk-x",
                "organization": "org-1",
                "base_url": "http://proxy/v1",
                "timeout": 7.0,
            }
        ]


@needs_openai
class TestOpenAIProviderRun:
    def test_request_shape_and_real_logprob_mapping(self):
        first = _lp("Paris", math.log(0.7), top=[("Paris", math.log(0.7)), ("Lyon", math.log(0.2))])
        usage = SimpleNamespace(prompt_tokens=5, completion_tokens=2, total_tokens=7)
        resp = _openai_response("Paris is lovely", [first], usage=usage)
        p, create = _openai_with_fake_client(resp, model="gpt-test", top_logprobs=2)

        out = p.run("capital of France?", max_tokens=3, temperature=0.0)

        assert create.calls == [
            {
                "model": "gpt-test",
                "messages": [{"role": "user", "content": "capital of France?"}],
                "logprobs": True,
                "top_logprobs": 2,
                "max_tokens": 3,
                "temperature": 0.0,
            }
        ]
        _assert_black_box(out)
        assert out.prompt == "capital of France?"
        assert out.tokens == ["Paris", "is", "lovely"]
        assert [t for t, _ in out.top_tokens] == ["Paris", "Lyon"]
        assert out.top_tokens[0][1] == pytest.approx(0.7)
        assert out.top_tokens[1][1] == pytest.approx(0.2)
        assert out.output_token == "Paris"
        assert out.meta["completion"] == "Paris is lovely"
        assert out.meta["finish_reason"] == "stop"
        assert out.meta["model"] == "gpt-test"
        assert out.meta["usage"] == {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
        assert out.meta["latency_ms"] >= 0.0
        assert out.meta["has_logprobs"] is True
        assert out.meta["token_param"] == "max_tokens" and out.meta["token_budget"] == 3

    def test_default_max_tokens_used_when_not_overridden(self):
        resp = _openai_response("ok", [_lp("ok", 0.0)])
        p, create = _openai_with_fake_client(resp, max_tokens=42)
        p.run("hi")
        assert create.calls[0]["max_tokens"] == 42

    def test_sampled_token_inserted_first_when_missing_from_alternatives(self):
        first = _lp("Rome", math.log(0.1), top=[("Paris", math.log(0.6))])
        p, _ = _openai_with_fake_client(_openai_response("Rome", [first]))
        out = p.run("x")
        assert out.top_tokens[0] == ("Rome", pytest.approx(0.1))
        assert out.top_tokens[1] == ("Paris", pytest.approx(0.6))

    def test_sampled_token_used_when_no_alternatives(self):
        first = _lp("Hi", math.log(0.9), top=[])
        p, _ = _openai_with_fake_client(_openai_response("Hi there", [first]))
        out = p.run("x")
        assert out.top_tokens == [("Hi", pytest.approx(0.9))]

    def test_without_logprobs_falls_back_to_sampled_first_word(self):
        p, _ = _openai_with_fake_client(_openai_response("Hello world", None))
        out = p.run("x")
        assert out.top_tokens == [("Hello", 1.0)]
        assert "usage" not in out.meta
        # The 1.0 is a placeholder and meta says so — no invented probability.
        assert out.meta["has_logprobs"] is False
        assert out.meta["logprobs_unavailable_reason"] == "not_returned"

    def test_empty_completion_without_logprobs(self):
        p, _ = _openai_with_fake_client(_openai_response(None, []))
        out = p.run("x")
        assert out.tokens == [""]
        assert out.top_tokens == [("<empty>", 1.0)]
        assert out.meta["completion"] == ""

    def test_malformed_logprobs_object_is_tolerated(self):
        class Exploding:
            @property
            def content(self):
                raise AttributeError("no content")

        resp = _openai_response("fine answer", None)
        resp.choices[0].logprobs = Exploding()
        p, _ = _openai_with_fake_client(resp)
        out = p.run("x")
        assert out.top_tokens == [("fine", 1.0)]


# ===========================================================================
# Anthropic
# ===========================================================================


def _anthropic_with_fake_client(response: Any, **kwargs: Any):
    from LLmThoughtLens.providers.anthropic_provider import AnthropicProvider

    provider = AnthropicProvider(api_key="sk-ant-test", **kwargs)
    create = _RecordingCreate(response)
    provider._client = SimpleNamespace(messages=SimpleNamespace(create=create))
    return provider, create


@needs_anthropic
class TestAnthropicProvider:
    def test_identity_and_env_key(self, monkeypatch):
        from LLmThoughtLens.providers.anthropic_provider import AnthropicProvider

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
        p = AnthropicProvider(model="claude-test", max_tokens="9", timeout=3)
        assert p.name == "anthropic"
        assert p.model_id == "anthropic/claude-test"
        assert p._api_key == "sk-ant-env"
        assert p.max_tokens == 9
        assert p.timeout == 3.0
        assert p.supports_internals is False

    def test_ensure_client_builds_sdk_client_once(self, monkeypatch):
        from LLmThoughtLens.providers.anthropic_provider import AnthropicProvider

        built: list[dict[str, Any]] = []

        class FakeAnthropic:
            def __init__(self, **kwargs: Any) -> None:
                built.append(kwargs)

        monkeypatch.setattr(pytest.importorskip("anthropic"), "Anthropic", FakeAnthropic)
        p = AnthropicProvider(api_key="sk-ant-x", timeout=11)
        assert p._ensure_client() is p._ensure_client()
        assert len(built) == 1
        assert built[0]["api_key"] == "sk-ant-x"
        assert built[0]["timeout"] == 11.0
        # No gateway configured -> the SDK default endpoint is used.
        assert built[0].get("base_url") is None

    def test_run_joins_text_blocks_and_ignores_other_blocks(self):
        resp = SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="Austin is "),
                SimpleNamespace(type="tool_use", text="SHOULD-NOT-APPEAR"),
                SimpleNamespace(type="text", text="the capital"),
            ],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=11, output_tokens=4),
        )
        p, create = _anthropic_with_fake_client(resp, model="claude-test", max_tokens=50)
        out = p.run("capital of Texas?", system="be terse")

        assert create.calls == [
            {
                "model": "claude-test",
                "max_tokens": 50,
                "messages": [{"role": "user", "content": "capital of Texas?"}],
                "system": "be terse",
            }
        ]
        _assert_black_box(out)
        assert out.meta["completion"] == "Austin is the capital"
        assert "SHOULD-NOT-APPEAR" not in out.meta["completion"]
        assert out.tokens == ["Austin", "is", "the", "capital"]
        # No logprobs from the Messages API: the sampled token at p=1.0, flagged in meta.
        assert out.top_tokens == [("Austin", 1.0)]
        assert "logprobs" in out.meta["evidence_note"]
        assert out.meta["stop_reason"] == "end_turn"
        assert out.meta["usage"] == {"input_tokens": 11, "output_tokens": 4}

    def test_max_tokens_override_and_empty_completion(self):
        resp = SimpleNamespace(content=[], stop_reason="max_tokens")
        p, create = _anthropic_with_fake_client(resp)
        out = p.run("x", max_tokens=1)
        assert create.calls[0]["max_tokens"] == 1
        assert out.tokens == [""]
        assert out.top_tokens == [("", 1.0)]
        assert "usage" not in out.meta


# ===========================================================================
# Ollama
# ===========================================================================


@pytest.fixture
def ollama_transport(monkeypatch):
    """Route every ``httpx.Client`` through a MockTransport; return a request log."""
    state: dict[str, Any] = {"handler": None, "requests": []}

    def _dispatch(request: httpx.Request) -> httpx.Response:
        state["requests"].append(request)
        return state["handler"](request)

    real_client = httpx.Client

    def _client_factory(*args: Any, **kwargs: Any) -> httpx.Client:
        kwargs["transport"] = httpx.MockTransport(_dispatch)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", _client_factory)
    return state


class TestOllamaProviderRun:
    def test_request_payload_and_real_logprobs(self, ollama_transport):
        from LLmThoughtLens.providers.ollama_provider import OllamaProvider

        body = {
            "response": "Paris, of course",
            "eval_count": 3,
            "eval_duration": 1234,
            "logprobs": [
                {
                    "token": "Paris",
                    "logprob": math.log(0.5),
                    "top_logprobs": [
                        {"token": "Lyon", "logprob": math.log(0.3)},
                        {"token": "Paris", "logprob": math.log(0.5)},
                        {"token": "Nice", "logprob": None},
                    ],
                }
            ],
        }
        ollama_transport["handler"] = lambda req: httpx.Response(200, json=body)

        p = OllamaProvider(
            model="llama-test", base_url="http://ollama.local:11434/", top_logprobs=3
        )
        out = p.run("capital of France?", options={"temperature": 0})

        (req,) = ollama_transport["requests"]
        assert req.method == "POST"
        assert str(req.url) == "http://ollama.local:11434/api/generate"
        sent = json.loads(req.content)
        assert sent == {
            "model": "llama-test",
            "prompt": "capital of France?",
            "stream": False,
            "options": {"temperature": 0},
            "logprobs": True,
            "top_logprobs": 3,
        }

        _assert_black_box(out)
        # Sorted descending; the None-logprob alternative is dropped.
        assert [t for t, _ in out.top_tokens] == ["Paris", "Lyon"]
        assert out.top_tokens[0][1] == pytest.approx(0.5)
        assert out.top_tokens[1][1] == pytest.approx(0.3)
        assert out.tokens == ["Paris,", "of", "course"]
        assert out.meta["has_logprobs"] is True
        assert "real per-token logprobs" in out.meta["evidence_note"]
        assert out.meta["eval_count"] == 3
        assert out.meta["eval_duration"] == 1234
        assert out.meta["completion"] == "Paris, of course"
        assert p.model_id == "ollama/llama-test"
        assert p.name == "ollama"

    def test_logprobs_not_requested_when_disabled(self, ollama_transport):
        from LLmThoughtLens.providers.ollama_provider import OllamaProvider

        ollama_transport["handler"] = lambda req: httpx.Response(200, json={"response": "yes"})
        out = OllamaProvider(request_logprobs=False).run("q")
        sent = json.loads(ollama_transport["requests"][0].content)
        assert "logprobs" not in sent and "top_logprobs" not in sent
        assert out.top_tokens == [("yes", 1.0)]
        assert out.meta["has_logprobs"] is False
        assert "did not return logprobs" in out.meta["evidence_note"]

    def test_caller_kwargs_override_logprob_defaults(self, ollama_transport):
        from LLmThoughtLens.providers.ollama_provider import OllamaProvider

        ollama_transport["handler"] = lambda req: httpx.Response(200, json={"response": ""})
        out = OllamaProvider().run("q", logprobs=False, top_logprobs=1)
        sent = json.loads(ollama_transport["requests"][0].content)
        assert sent["logprobs"] is False
        assert sent["top_logprobs"] == 1
        assert out.tokens == [""]
        assert out.top_tokens == [("", 1.0)]

    def test_http_error_propagates(self, ollama_transport):
        from LLmThoughtLens.providers.ollama_provider import OllamaProvider

        ollama_transport["handler"] = lambda req: httpx.Response(500, json={"error": "boom"})
        with pytest.raises(httpx.HTTPStatusError):
            OllamaProvider().run("q")


class TestOllamaFirstTokenDistribution:
    def _dist(self, logprobs: Any, text: str = "fallback text"):
        from LLmThoughtLens.providers.ollama_provider import OllamaProvider

        return OllamaProvider._first_token_distribution(logprobs, text)

    def test_sampled_token_inserted_when_absent_from_alternatives(self):
        out, used = self._dist(
            [
                {
                    "token": "B",
                    "logprob": math.log(0.2),
                    "top_logprobs": [{"token": "A", "logprob": 0.0}],
                }
            ]
        )
        assert used is True
        assert out == [("A", pytest.approx(1.0)), ("B", pytest.approx(0.2))]

    def test_entry_without_alternatives_uses_sampled_token(self):
        out, used = self._dist([{"token": "Z", "logprob": math.log(0.4)}])
        assert used is True
        assert out == [("Z", pytest.approx(0.4))]

    @pytest.mark.parametrize(
        "logprobs",
        [None, [], "garbage", [{"token": "x", "logprob": None}]],
    )
    def test_unusable_logprobs_fall_back_to_sampled_word(self, logprobs):
        out, used = self._dist(logprobs, "hello there")
        assert used is False
        assert out == [("hello", 1.0)]

    def test_fallback_with_empty_text(self):
        out, used = self._dist(None, "")
        assert (out, used) == ([("", 1.0)], False)


class TestOllamaPing:
    def test_ping_true_on_200(self, ollama_transport):
        from LLmThoughtLens.providers.ollama_provider import OllamaProvider

        ollama_transport["handler"] = lambda req: httpx.Response(200, json={"models": []})
        assert OllamaProvider(base_url="http://h:1").ping() is True
        assert str(ollama_transport["requests"][0].url) == "http://h:1/api/tags"

    def test_ping_false_on_non_200(self, ollama_transport):
        from LLmThoughtLens.providers.ollama_provider import OllamaProvider

        ollama_transport["handler"] = lambda req: httpx.Response(404)
        assert OllamaProvider().ping() is False

    def test_ping_false_on_connection_error(self, ollama_transport):
        from LLmThoughtLens.providers.ollama_provider import OllamaProvider

        def _refuse(req: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=req)

        ollama_transport["handler"] = _refuse
        assert OllamaProvider().ping() is False


class TestRemoteProvidersThroughRegistry:
    """The registry path used by the CLI/TUI/server must yield the same classes."""

    @pytest.mark.parametrize(
        ("name", "kwargs", "cls_name"),
        [
            pytest.param(
                "openai", {"model": "m", "api_key": "k"}, "OpenAIProvider", marks=needs_openai
            ),
            pytest.param(
                "anthropic",
                {"model": "m", "api_key": "k"},
                "AnthropicProvider",
                marks=needs_anthropic,
            ),
            ("ollama", {"model": "m"}, "OllamaProvider"),
        ],
    )
    def test_get_provider(self, name, kwargs, cls_name):
        from LLmThoughtLens.providers.registry import available_providers, get_provider

        p = get_provider(name, **kwargs)
        assert type(p).__name__ == cls_name
        assert p.model_id == f"{name}/m"
        assert name in available_providers()
