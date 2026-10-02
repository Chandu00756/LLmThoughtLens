"""OpenAIProvider ↔ reasoning-model compatibility (offline, no network).

Covers the parameter-selection rules for OpenAI reasoning models (o-series,
gpt-5 family): ``max_completion_tokens`` instead of ``max_tokens``, no
logprobs, a reasoning-sized default budget, and the automatic 400-driven
fallback that corrects unsupported parameters once and caches the learned
capability per model id.

Two kinds of fake backends are used:

* a recording ``create`` callable injected into ``provider._client`` (to
  assert the exact request kwargs), and
* the **real** ``openai`` SDK client wired to an ``httpx.MockTransport`` that
  emulates OpenAI's 400 responses, so the error parsing runs against genuine
  ``openai.BadRequestError`` objects.

Pure model-id rule tests need no optional dependency.
"""

from __future__ import annotations

import importlib.util
import json
import math
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from LLmThoughtLens.providers.openai_provider import (
    DEFAULT_REASONING_MAX_TOKENS,
    is_reasoning_model,
    uses_max_completion_tokens,
)

needs_openai = pytest.mark.skipif(
    importlib.util.find_spec("openai") is None, reason="openai extra not installed"
)

# ---------------------------------------------------------------------------
# Model-id rule (no optional deps)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "reasoning", "completion_param"),
    [
        ("o1", True, True),
        ("o1-mini", True, True),
        ("o3-mini", True, True),
        ("o4-mini-2025-04-16", True, True),
        ("O3-PRO", True, True),
        ("gpt-5", True, True),
        ("gpt-5-mini", True, True),
        ("gpt-5.1", True, True),
        ("gpt-5-chat-latest", False, True),  # chat variant: new param, keeps logprobs
        ("ft:o4-mini-2025-04-16:acme::abc123", True, True),
        ("openai/o3-mini", True, True),  # router-style vendor prefix
        ("gpt-4o-mini", False, False),
        ("gpt-4.1-nano", False, False),
        ("gpt-4o", False, False),
        ("gpt-oss-120b", False, False),
        ("omni-moderation-latest", False, False),
        ("my-azure-deployment", False, False),
        ("", False, False),
    ],
)
def test_model_id_rule(model: str, reasoning: bool, completion_param: bool) -> None:
    assert is_reasoning_model(model) is reasoning
    assert uses_max_completion_tokens(model) is completion_param


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _ScriptedCreate:
    """Records kwargs; each call pops the next scripted outcome (exc or response)."""

    def __init__(self, *outcomes: Any) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _HTTPError(Exception):
    """Duck-typed SDK error: only ``status_code`` + message (no ``param``)."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


def _usage(prompt: int = 5, completion: int = 3, reasoning: int | None = None) -> Any:
    details = None if reasoning is None else SimpleNamespace(reasoning_tokens=reasoning)
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        completion_tokens_details=details,
    )


def _response(
    content: str | None,
    logprob_content: list[Any] | None = None,
    finish_reason: str = "stop",
    usage: Any = None,
    refusal: str | None = None,
) -> Any:
    logprobs = None if logprob_content is None else SimpleNamespace(content=logprob_content)
    choice = SimpleNamespace(
        message=SimpleNamespace(content=content, refusal=refusal),
        logprobs=logprobs,
        finish_reason=finish_reason,
    )
    return SimpleNamespace(choices=[choice], usage=usage)


def _lp(token: str, prob: float, top: list[tuple[str, float]] | None = None) -> Any:
    return SimpleNamespace(
        token=token,
        logprob=math.log(prob),
        top_logprobs=[SimpleNamespace(token=t, logprob=math.log(p)) for t, p in (top or [])],
    )


def _provider(create: Any, **kwargs: Any) -> Any:
    from LLmThoughtLens.providers.openai_provider import OpenAIProvider

    p = OpenAIProvider(api_key="sk-test", **kwargs)
    p._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return p


def _bad_request(param: str, message: str, code: str = "unsupported_parameter") -> Any:
    """A genuine ``openai.BadRequestError`` exactly as the SDK builds it."""
    openai = pytest.importorskip("openai")
    body = {"message": message, "type": "invalid_request_error", "param": param, "code": code}
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(400, request=request, json={"error": body})
    return openai.BadRequestError(
        f"Error code: 400 - {{'error': {body}}}", response=response, body=body
    )


MAX_TOKENS_MSG = (
    "Unsupported parameter: 'max_tokens' is not supported with this model. "
    "Use 'max_completion_tokens' instead."
)
LOGPROBS_MSG = "Unsupported parameter: 'logprobs' is not supported with this model."
TEMPERATURE_MSG = (
    "Unsupported value: 'temperature' does not support 0.0 with this model. "
    "Only the default (1) value is supported."
)


def _assert_black_box(out: Any) -> None:
    assert out.evidence_kind == "black_box"
    assert out.activations is None and out.attentions is None and out.logits is None
    assert out.token_ids == []


# ---------------------------------------------------------------------------
# Chat model (unchanged behaviour)
# ---------------------------------------------------------------------------


@needs_openai
class TestChatModel:
    def test_happy_path_uses_max_tokens_and_real_logprobs(self) -> None:
        first = _lp("Paris", 0.8, top=[("Paris", 0.8), ("Lyon", 0.1)])
        create = _ScriptedCreate(_response("Paris.", [first], usage=_usage()))
        p = _provider(create, model="gpt-4o-mini", top_logprobs=2)

        out = p.run("capital of France?")

        assert create.calls == [
            {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "capital of France?"}],
                "logprobs": True,
                "top_logprobs": 2,
                "max_tokens": 256,
            }
        ]
        _assert_black_box(out)
        assert out.top_tokens == [("Paris", pytest.approx(0.8)), ("Lyon", pytest.approx(0.1))]
        assert out.meta["has_logprobs"] is True
        assert out.meta["logprobs_requested"] is True
        assert "logprobs_unavailable_reason" not in out.meta
        assert out.meta["reasoning_model"] is False
        assert out.meta["token_param"] == "max_tokens"
        assert out.meta["token_budget"] == 256
        assert "real OpenAI logprobs" in out.meta["evidence_note"]
        assert "param_fallbacks" not in out.meta and "dropped_params" not in out.meta
        assert "reasoning_tokens" not in out.meta["usage"]
        assert p.is_reasoning_model is False

    def test_logprobs_requested_but_not_returned(self) -> None:
        create = _ScriptedCreate(_response("Hello world", None))
        out = _provider(create, model="gpt-4o-mini").run("x")
        assert create.calls[0]["logprobs"] is True
        assert out.top_tokens == [("Hello", 1.0)]
        assert out.meta["has_logprobs"] is False
        assert out.meta["logprobs_unavailable_reason"] == "not_returned"
        assert "not a real probability" in out.meta["evidence_note"]

    def test_request_logprobs_false_never_requests(self) -> None:
        create = _ScriptedCreate(_response("Yes", None))
        out = _provider(create, model="gpt-4o-mini", request_logprobs=False).run("x")
        assert "logprobs" not in create.calls[0] and "top_logprobs" not in create.calls[0]
        assert out.meta["logprobs_unavailable_reason"] == "disabled"
        assert out.meta["logprobs_requested"] is False

    def test_per_call_logprobs_false_strips_top_logprobs(self) -> None:
        create = _ScriptedCreate(_response("Yes", None))
        _provider(create, model="gpt-4o-mini").run("x", logprobs=False)
        assert "logprobs" not in create.calls[0] and "top_logprobs" not in create.calls[0]

    def test_refusal_is_surfaced(self) -> None:
        create = _ScriptedCreate(_response(None, None, refusal="I can't help with that."))
        out = _provider(create, model="gpt-4o-mini").run("x")
        assert out.meta["refusal"] == "I can't help with that."
        assert out.tokens == [""]


# ---------------------------------------------------------------------------
# Reasoning model parameter selection
# ---------------------------------------------------------------------------


@needs_openai
class TestReasoningModelParams:
    @pytest.mark.parametrize("model", ["o3-mini", "o4-mini", "gpt-5", "gpt-5-mini"])
    def test_uses_max_completion_tokens_and_skips_logprobs(self, model: str) -> None:
        resp = _response("Paris", None, usage=_usage(completion=70, reasoning=64))
        create = _ScriptedCreate(resp)
        p = _provider(create, model=model)

        out = p.run("capital of France?")

        assert create.calls == [
            {
                "model": model,
                "messages": [{"role": "user", "content": "capital of France?"}],
                "max_completion_tokens": DEFAULT_REASONING_MAX_TOKENS,
            }
        ]
        _assert_black_box(out)
        # Single sampled token at the 1.0 placeholder — never an invented distribution.
        assert out.top_tokens == [("Paris", 1.0)]
        assert out.meta["has_logprobs"] is False
        assert out.meta["logprobs_requested"] is False
        assert out.meta["logprobs_unavailable_reason"] == "reasoning_model"
        assert "reasoning models do not expose logprobs" in out.meta["evidence_note"]
        assert out.meta["reasoning_model"] is True
        assert out.meta["token_param"] == "max_completion_tokens"
        assert out.meta["token_budget"] == DEFAULT_REASONING_MAX_TOKENS
        assert out.meta["usage"] == {
            "prompt_tokens": 5,
            "completion_tokens": 70,
            "total_tokens": 75,
            "reasoning_tokens": 64,
        }
        assert p.is_reasoning_model is True

    def test_budget_is_max_of_max_tokens_and_reasoning_floor(self) -> None:
        create = _ScriptedCreate(_response("ok"))
        _provider(create, model="o3-mini", max_tokens=9000).run("x")
        _provider(create, model="o3-mini", reasoning_max_tokens=1024).run("x")
        assert create.calls[0]["max_completion_tokens"] == 9000
        assert create.calls[1]["max_completion_tokens"] == 1024

    def test_explicit_per_call_budget_is_honoured_and_mapped(self) -> None:
        create = _ScriptedCreate(_response("ok"))
        p = _provider(create, model="o3-mini")
        p.run("x", max_tokens=50)
        p.run("x", max_completion_tokens=70)
        assert create.calls[0]["max_completion_tokens"] == 50
        assert create.calls[1]["max_completion_tokens"] == 70
        assert all("max_tokens" not in c for c in create.calls)

    def test_chat_model_maps_max_completion_tokens_onto_max_tokens(self) -> None:
        create = _ScriptedCreate(_response("ok", [_lp("ok", 0.9)]))
        _provider(create, model="gpt-4o-mini").run("x", max_completion_tokens=12)
        assert create.calls[0]["max_tokens"] == 12
        assert "max_completion_tokens" not in create.calls[0]

    def test_gpt5_chat_variant_keeps_logprobs_with_new_budget_param(self) -> None:
        create = _ScriptedCreate(_response("ok", [_lp("ok", 0.9)]))
        out = _provider(create, model="gpt-5-chat-latest").run("x")
        assert create.calls[0]["logprobs"] is True
        assert create.calls[0]["max_completion_tokens"] == 256
        assert out.meta["has_logprobs"] is True

    def test_request_logprobs_true_forces_request_on_reasoning_model(self) -> None:
        create = _ScriptedCreate(_response("ok", [_lp("ok", 0.6)]))
        out = _provider(create, model="gpt-5.2", request_logprobs=True, top_logprobs=3).run("x")
        assert create.calls[0]["logprobs"] is True and create.calls[0]["top_logprobs"] == 3
        assert out.top_tokens == [("ok", pytest.approx(0.6))]

    def test_reasoning_effort_is_sent_and_reported(self) -> None:
        create = _ScriptedCreate(_response("ok"))
        out = _provider(create, model="o3-mini", reasoning_effort="low").run("x")
        assert create.calls[0]["reasoning_effort"] == "low"
        assert out.meta["reasoning_effort"] == "low"
        # A per-call value wins over the constructor default.
        _provider(create, model="o3-mini", reasoning_effort="low").run("x", reasoning_effort="high")
        assert create.calls[1]["reasoning_effort"] == "high"

    def test_empty_content_with_finish_reason_length(self) -> None:
        resp = _response(
            "", None, finish_reason="length", usage=_usage(completion=4096, reasoning=4096)
        )
        out = _provider(_ScriptedCreate(resp), model="o3-mini").run("hard question")

        _assert_black_box(out)
        assert out.tokens == [""]
        assert out.top_tokens == [("<empty>", 1.0)]
        assert out.meta["completion"] == ""
        assert out.meta["finish_reason"] == "length"
        assert out.meta["budget_exhausted"] is True
        assert out.meta["usage"]["reasoning_tokens"] == 4096
        note = out.meta["evidence_note"]
        assert "4096-token budget ran out" in note
        assert "hidden reasoning" in note and "reasoning_max_tokens" in note

    def test_budget_exhausted_on_chat_model_has_no_reasoning_hint(self) -> None:
        out = _provider(_ScriptedCreate(_response("", [], finish_reason="length")), model="gpt-4o")
        out = out.run("x")
        assert out.meta["budget_exhausted"] is True
        assert "hidden reasoning" not in out.meta["evidence_note"]

    def test_non_empty_truncated_output_is_not_flagged_exhausted(self) -> None:
        out = _provider(
            _ScriptedCreate(_response("partial answer", finish_reason="length")), model="o3"
        ).run("x")
        assert "budget_exhausted" not in out.meta
        assert out.meta["finish_reason"] == "length"

    def test_capabilities_snapshot(self) -> None:
        p = _provider(_ScriptedCreate(_response("ok")), model="o3-mini")
        assert p.capabilities() == {
            "model": "o3-mini",
            "reasoning_model": True,
            "token_param": "max_completion_tokens",
            "logprobs": False,
            "logprobs_rejected": False,
            "dropped_params": [],
        }


# ---------------------------------------------------------------------------
# 400 → correct → retry, cached per model id
# ---------------------------------------------------------------------------


@needs_openai
class TestUnsupportedParamFallback:
    def test_unknown_id_learns_reasoning_caps_from_400s_and_caches(self) -> None:
        ok = _response("Paris", None, usage=_usage(completion=40, reasoning=32))
        create = _ScriptedCreate(
            _bad_request("max_tokens", MAX_TOKENS_MSG),
            _bad_request("logprobs", LOGPROBS_MSG),
            ok,
        )
        # An Azure-style deployment name the id rule cannot classify.
        p = _provider(create, model="my-reasoner", top_logprobs=4)

        out = p.run("capital of France?")

        assert len(create.calls) == 3
        assert create.calls[0]["max_tokens"] == 256 and create.calls[0]["logprobs"] is True
        # Rejecting max_tokens implies a reasoning model -> reasoning-sized budget.
        assert create.calls[1]["max_completion_tokens"] == DEFAULT_REASONING_MAX_TOKENS
        assert "max_tokens" not in create.calls[1] and create.calls[1]["logprobs"] is True
        assert create.calls[2] == {
            "model": "my-reasoner",
            "messages": [{"role": "user", "content": "capital of France?"}],
            "max_completion_tokens": DEFAULT_REASONING_MAX_TOKENS,
        }
        assert out.meta["param_fallbacks"] == ["max_tokens", "logprobs"]
        assert out.meta["logprobs_unavailable_reason"] == "api_rejected"
        assert "rejected logprobs" in out.meta["evidence_note"]
        assert out.top_tokens == [("Paris", 1.0)]
        assert p.capabilities()["logprobs_rejected"] is True

        # Cached: the next call goes right first time.
        p.run("again")
        assert len(create.calls) == 4
        assert "max_tokens" not in create.calls[3] and "logprobs" not in create.calls[3]

    def test_cache_is_per_model_id(self) -> None:
        create = _ScriptedCreate(_bad_request("max_tokens", MAX_TOKENS_MSG), _response("ok"))
        p = _provider(create, model="deploy-a")
        p.run("x")
        assert p._caps["deploy-a"].token_param == "max_completion_tokens"
        # Per-call model override has its own (fresh) record.
        p.run("x", model="deploy-b")
        assert create.calls[-1]["model"] == "deploy-b"
        assert create.calls[-1]["max_tokens"] == 256
        assert p._caps["deploy-b"].token_param == "max_tokens"
        # A fresh provider instance does not share the learned capability.
        p2 = _provider(_ScriptedCreate(_response("ok")), model="deploy-a")
        assert p2.capabilities()["token_param"] == "max_tokens"

    def test_temperature_dropped_and_reported(self) -> None:
        create = _ScriptedCreate(
            _bad_request("temperature", TEMPERATURE_MSG, code="unsupported_value"),
            _response("ok"),
        )
        p = _provider(create, model="o3-mini")
        out = p.run("x", temperature=0.0)
        assert create.calls[0]["temperature"] == 0.0
        assert "temperature" not in create.calls[1]
        assert out.meta["param_fallbacks"] == ["temperature"]
        assert out.meta["dropped_params"] == ["temperature"]
        # Next call drops it proactively — single request.
        out2 = p.run("y", temperature=0.0)
        assert len(create.calls) == 3 and "temperature" not in create.calls[2]
        assert out2.meta["dropped_params"] == ["temperature"]
        assert "param_fallbacks" not in out2.meta

    def test_reasoning_effort_rejected_by_chat_model_is_dropped(self) -> None:
        create = _ScriptedCreate(
            _HTTPError(400, "Unrecognized request argument supplied: reasoning_effort"),
            _response("ok", [_lp("ok", 0.5)]),
        )
        out = _provider(create, model="gpt-4o", reasoning_effort="low").run("x")
        assert "reasoning_effort" not in create.calls[1]
        assert out.meta["dropped_params"] == ["reasoning_effort"]
        assert "reasoning_effort" not in out.meta

    def test_message_only_error_picks_the_param_actually_sent(self) -> None:
        # No structured ``param``: the message names both max_tokens and
        # max_completion_tokens; only max_tokens was sent.
        create = _ScriptedCreate(_HTTPError(400, MAX_TOKENS_MSG), _response("ok"))
        _provider(create, model="custom").run("x")
        assert "max_completion_tokens" in create.calls[1]

    def test_reverse_switch_when_server_rejects_max_completion_tokens(self) -> None:
        # e.g. an older OpenAI-compatible server hosting a model named gpt-5-*.
        create = _ScriptedCreate(
            _HTTPError(422, "Unknown parameter: 'max_completion_tokens'."),
            _response("ok"),
        )
        out = _provider(create, model="gpt-5-chat-latest", request_logprobs=False).run("x")
        assert create.calls[1]["max_tokens"] == 256
        assert out.meta["token_param"] == "max_tokens"

    def test_same_group_rejected_twice_is_reraised(self) -> None:
        create = _ScriptedCreate(
            _bad_request("max_tokens", MAX_TOKENS_MSG),
            _bad_request(
                "max_completion_tokens",
                "Unsupported parameter: 'max_completion_tokens' is not supported.",
            ),
        )
        with pytest.raises(Exception, match="max_completion_tokens"):
            _provider(create, model="weird").run("x")
        assert len(create.calls) == 2

    def test_rejection_of_param_not_sent_is_reraised(self) -> None:
        create = _ScriptedCreate(_bad_request("logprobs", LOGPROBS_MSG))
        with pytest.raises(Exception, match="logprobs"):
            _provider(create, model="o3-mini").run("x")  # logprobs never sent
        assert len(create.calls) == 1

    @pytest.mark.parametrize(
        "exc",
        [
            _HTTPError(400, "This model's maximum context length is 8192 tokens."),
            _HTTPError(500, "Unsupported parameter: 'max_tokens'"),
            _HTTPError(401, "Incorrect API key provided"),
            RuntimeError("connection reset"),
        ],
    )
    def test_unrelated_errors_propagate_without_retry(self, exc: Exception) -> None:
        create = _ScriptedCreate(exc)
        with pytest.raises(type(exc)):
            _provider(create, model="gpt-4o-mini").run("x")
        assert len(create.calls) == 1

    def test_status_read_from_response_attribute(self) -> None:
        exc = Exception("Unsupported parameter: 'logprobs' is not supported with this model.")
        exc.response = SimpleNamespace(status_code=400)  # type: ignore[attr-defined]
        create = _ScriptedCreate(exc, _response("ok"))
        out = _provider(create, model="custom").run("x")
        assert out.meta["param_fallbacks"] == ["logprobs"]


# ---------------------------------------------------------------------------
# End-to-end through the real openai SDK (HTTP mocked, no network)
# ---------------------------------------------------------------------------


def _fake_reasoning_server(requests: list[dict[str, Any]]) -> Any:
    """MockTransport handler that behaves like OpenAI for a reasoning model."""

    def error(param: str, message: str, code: str = "unsupported_parameter") -> httpx.Response:
        body = {"message": message, "type": "invalid_request_error", "param": param, "code": code}
        return httpx.Response(400, json={"error": body})

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        if "max_tokens" in body:
            return error("max_tokens", MAX_TOKENS_MSG)
        if body.get("logprobs") or "top_logprobs" in body:
            return error("logprobs", LOGPROBS_MSG)
        if body.get("temperature", 1) != 1:
            return error("temperature", TEMPERATURE_MSG, code="unsupported_value")
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Paris is the capital."},
                        "finish_reason": "stop",
                        "logprobs": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 6,
                    "completion_tokens": 90,
                    "total_tokens": 96,
                    "completion_tokens_details": {"reasoning_tokens": 84},
                },
            },
        )

    return handler


@needs_openai
def test_real_sdk_bad_request_errors_drive_the_fallback() -> None:
    from LLmThoughtLens.providers.openai_provider import OpenAIProvider
    from openai import OpenAI

    sent: list[dict[str, Any]] = []
    http_client = httpx.Client(transport=httpx.MockTransport(_fake_reasoning_server(sent)))
    p = OpenAIProvider(model="azure-o-deployment", api_key="sk-test", base_url="http://fake/v1")
    p._client = OpenAI(
        api_key="sk-test", base_url="http://fake/v1", http_client=http_client, max_retries=0
    )

    out = p.run("capital of France?", temperature=0)

    # max_tokens -> logprobs -> temperature, each corrected exactly once.
    assert len(sent) == 4
    assert out.meta["param_fallbacks"] == ["max_tokens", "logprobs", "temperature"]
    assert sent[-1] == {
        "model": "azure-o-deployment",
        "messages": [{"role": "user", "content": "capital of France?"}],
        "max_completion_tokens": DEFAULT_REASONING_MAX_TOKENS,
    }
    _assert_black_box(out)
    assert out.top_tokens == [("Paris", 1.0)]
    assert out.meta["has_logprobs"] is False
    assert out.meta["usage"]["reasoning_tokens"] == 84
    assert out.meta["finish_reason"] == "stop"

    # Learned capabilities are reused: one request, no fallbacks.
    out2 = p.run("again", temperature=0)
    assert len(sent) == 5
    assert "param_fallbacks" not in out2.meta
    assert out2.meta["dropped_params"] == ["temperature"]


@needs_openai
def test_real_sdk_known_reasoning_model_needs_no_fallback() -> None:
    from LLmThoughtLens.providers.openai_provider import OpenAIProvider
    from openai import OpenAI

    sent: list[dict[str, Any]] = []
    http_client = httpx.Client(transport=httpx.MockTransport(_fake_reasoning_server(sent)))
    p = OpenAIProvider(model="o4-mini", api_key="sk-test")
    p._client = OpenAI(
        api_key="sk-test", base_url="http://fake/v1", http_client=http_client, max_retries=0
    )
    out = p.run("capital of France?")
    assert len(sent) == 1
    assert "param_fallbacks" not in out.meta
    assert out.meta["completion"] == "Paris is the capital."
