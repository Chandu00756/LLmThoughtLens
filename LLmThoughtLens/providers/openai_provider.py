"""OpenAI provider — Chat Completions adapter with real top-k logprobs.

OpenAI exposes per-token logprobs (``logprobs=True, top_logprobs=k``) for
classic chat models, so ``top_tokens`` is populated with **real**
probabilities there.  Anything we cannot legitimately compute (activations,
attentions, full vocabulary logits) is left as ``None`` — never synthesised.

Reasoning models
----------------
OpenAI's reasoning models (the ``o1`` / ``o3`` / ``o4`` families and the
``gpt-5`` family and later) differ from classic chat models in three ways
that matter here:

* they reject ``max_tokens`` and require ``max_completion_tokens``;
* they do not return logprobs (requesting them is a ``400``);
* the token budget is shared with *hidden* reasoning tokens, so a small
  budget can be spent entirely on reasoning and yield an empty completion
  with ``finish_reason='length'``.

The provider therefore keeps a small per-model **capability record**:

1. An initial guess from the model id (:func:`is_reasoning_model`, a
   documented prefix rule).
2. Automatic correction: if the API answers ``400`` naming an unsupported
   parameter (``max_tokens`` / ``max_completion_tokens`` / ``logprobs`` /
   ``top_logprobs`` / ``temperature`` / …), the request is rebuilt without
   (or with the right replacement for) that parameter and retried.  Each
   parameter is corrected at most once per call, so the retry loop is
   bounded, and the learned capability is cached per model id for the
   lifetime of the provider instance — later calls go right the first time.

When logprobs are unavailable ``top_tokens`` follows the black-box
convention shared with the Anthropic / Ollama providers: the single sampled
token at a ``1.0`` placeholder, with ``meta['has_logprobs'] = False`` and an
``evidence_note`` saying the probability is not a real one.  Probabilities
are never invented.
"""

from __future__ import annotations

import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from LLmThoughtLens.providers.base import BaseProvider, ProviderOutput
from LLmThoughtLens.providers.defaults import resolve_model
from LLmThoughtLens.utils.tokenizer_utils import whitespace_tokens

__all__ = [
    "DEFAULT_REASONING_MAX_TOKENS",
    "OpenAIProvider",
    "is_reasoning_model",
    "uses_max_completion_tokens",
]

#: Default ``max_completion_tokens`` for reasoning models.  The budget covers
#: hidden reasoning *and* the visible answer; 256 (the chat default) is often
#: consumed by reasoning alone, giving an empty completion.  The budget is a
#: ceiling, not a spend — short prompts use far less.
DEFAULT_REASONING_MAX_TOKENS = 4096

#: HTTP statuses that may carry an "unsupported parameter" complaint.  OpenAI
#: uses 400; some OpenAI-compatible servers answer 422 for the same thing.
_PARAM_ERROR_STATUSES = frozenset({400, 422})

#: Request parameters the provider knows how to correct after a 400, mapped
#: to the correction group they belong to.  A group is corrected at most once
#: per call, which is what bounds the retry loop.
_ADJUSTABLE_PARAMS: dict[str, str] = {
    "max_tokens": "token_param",
    "max_completion_tokens": "token_param",
    "logprobs": "logprobs",
    "top_logprobs": "logprobs",
    "temperature": "temperature",
    "top_p": "top_p",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
    "logit_bias": "logit_bias",
    "reasoning_effort": "reasoning_effort",
}

#: Phrases that mark a 400 as a parameter-support complaint (lower-cased).
_UNSUPPORTED_MARKERS = (
    "unsupported",
    "not supported",
    "does not support",
    "unrecognized request argument",
    "unrecognized parameter",
    "unknown parameter",
    "not allowed",
    "only the default",
)

_O_SERIES = re.compile(r"^o[1-9]\d*(?:$|[-_.:])")
_GPT_MAJOR = re.compile(r"^gpt-(\d+)(?!\d)")

#: Hard cap on attempts per call (1 original + one per correction group).
_MAX_ATTEMPTS = 1 + len(set(_ADJUSTABLE_PARAMS.values()))


def _normalise_model_id(model: str) -> str:
    """Lower-case *model* and strip fine-tune / router prefixes.

    ``"ft:o4-mini-2025-04-16:org::abc"`` → ``"o4-mini-2025-04-16:org::abc"``;
    ``"openai/o3-mini"`` (OpenRouter-style) → ``"o3-mini"``.
    """
    mid = (model or "").strip().lower()
    if mid.startswith("ft:"):
        mid = mid[3:]
    return mid.rsplit("/", 1)[-1]


def _is_gpt5_or_later(mid: str) -> bool:
    m = _GPT_MAJOR.match(mid)
    return m is not None and int(m.group(1)) >= 5


def uses_max_completion_tokens(model: str) -> bool:
    """Whether *model* is expected to need ``max_completion_tokens``.

    True for the ``o<N>`` reasoning families and every ``gpt-5``-or-later
    id (including ``*-chat*`` variants, which accept
    ``max_completion_tokens`` too).  This is only the *initial* guess — the
    provider corrects it automatically from the API's 400 response.
    """
    mid = _normalise_model_id(model)
    return bool(_O_SERIES.match(mid)) or _is_gpt5_or_later(mid)


def is_reasoning_model(model: str) -> bool:
    """Prefix rule: is *model* an OpenAI reasoning model?

    Rule (after lower-casing and stripping ``ft:`` / ``vendor/`` prefixes):

    * ``o<N>`` optionally followed by ``-`` / ``.`` / ``_`` / ``:`` — the
      ``o1`` / ``o3`` / ``o4`` families (``o3-mini``, ``o4-mini-2025-04-16``);
    * ``gpt-<N>`` with ``N >= 5`` (``gpt-5``, ``gpt-5-mini``, ``gpt-5.1``)
      **except** ids containing ``-chat`` (``gpt-5-chat-latest`` is a
      non-reasoning chat model).

    Anything else (``gpt-4o``, ``gpt-4.1-mini``, ``gpt-oss-120b`` served by
    a compatible server, Azure deployment names) is treated as a classic chat
    model; the 400-driven fallback in :class:`OpenAIProvider` covers ids the
    rule gets wrong.
    """
    mid = _normalise_model_id(model)
    if _O_SERIES.match(mid):
        return True
    return _is_gpt5_or_later(mid) and "-chat" not in mid


@dataclass
class _ModelCaps:
    """What one model id accepts, as guessed from its id and learned from 400s."""

    reasoning: bool
    token_param: str  # "max_tokens" | "max_completion_tokens"
    logprobs: bool
    logprobs_rejected: bool = False
    dropped: set[str] = field(default_factory=set)


def _error_status(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    return status if isinstance(status, int) else None


def _unsupported_param(exc: BaseException, request: dict[str, Any]) -> str | None:
    """Return the request parameter a 400 says is unsupported, if fixable.

    Uses the SDK's structured ``param`` field when present, otherwise parses
    the error message (taking the earliest-mentioned known parameter that was
    actually sent — "'max_tokens' is not supported … use
    'max_completion_tokens'" names both).  Returns ``None`` for any error that
    is not a recognisable parameter-support complaint about a parameter this
    request carried.
    """
    if _error_status(exc) not in _PARAM_ERROR_STATUSES:
        return None
    message = str(getattr(exc, "message", None) or exc)
    lowered = message.lower()
    code = str(getattr(exc, "code", "") or "").lower()
    complaint = code in ("unsupported_parameter", "unsupported_value") or any(
        marker in lowered for marker in _UNSUPPORTED_MARKERS
    )
    if not complaint:
        return None

    param = getattr(exc, "param", None)
    if isinstance(param, str) and param in _ADJUSTABLE_PARAMS and param in request:
        return param

    best: tuple[int, str] | None = None
    for name in _ADJUSTABLE_PARAMS:
        if name not in request:
            continue
        m = re.search(rf"(?<![a-z_]){re.escape(name)}(?![a-z_])", lowered)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), name)
    return best[1] if best else None


class OpenAIProvider(BaseProvider):
    """Black-box provider backed by OpenAI Chat Completions.

    Parameters
    ----------
    model:
        Model id; defaults to ``default_model("openai")``.
    api_key:
        API key; falls back to ``OPENAI_API_KEY``.
    organization:
        Optional OpenAI organization id.
    base_url:
        Optional OpenAI-compatible endpoint (proxy, Azure-style gateway, …).
    top_logprobs:
        Alternatives requested per position (clamped to 1..20).
    max_tokens:
        Completion budget for classic chat models (sent as ``max_tokens``).
        For reasoning models the budget is
        ``max(max_tokens, reasoning_max_tokens)`` and is sent as
        ``max_completion_tokens``.
    timeout:
        Per-request timeout in seconds.
    reasoning_max_tokens:
        Budget floor for reasoning models, which share the budget between
        hidden reasoning and the visible answer.  Defaults to
        :data:`DEFAULT_REASONING_MAX_TOKENS`.
    request_logprobs:
        ``None`` (default) requests logprobs unless the model id looks like a
        reasoning model; ``True`` always requests them (still dropped if the
        API rejects them); ``False`` never requests them.
    reasoning_effort:
        Optional ``reasoning_effort`` sent with every request (e.g.
        ``"low"``).  Dropped automatically if the API rejects it.

    A per-call ``max_tokens=`` or ``max_completion_tokens=`` passed to
    :meth:`run` is honoured exactly and mapped onto whichever parameter the
    model accepts.
    """

    evidence_kind = "black_box"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        organization: str | None = None,
        base_url: str | None = None,
        top_logprobs: int = 5,
        max_tokens: int = 256,
        timeout: float = 60.0,
        reasoning_max_tokens: int = DEFAULT_REASONING_MAX_TOKENS,
        request_logprobs: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI  # noqa: F401
        except ImportError as exc:  # pragma: no cover — gated by extras
            raise ImportError(
                "OpenAIProvider needs the `openai` extra. "
                "Install with: pip install 'LLmThoughtLens[openai]'"
            ) from exc

        self.model = resolve_model("openai", model)
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._organization = organization
        self._base_url = base_url
        self.top_logprobs = max(1, min(20, int(top_logprobs)))
        self.max_tokens = int(max_tokens)
        self.timeout = float(timeout)
        self.reasoning_max_tokens = max(1, int(reasoning_max_tokens))
        self.request_logprobs = request_logprobs
        self.reasoning_effort = reasoning_effort or None
        self._client: Any = None
        self._caps: dict[str, _ModelCaps] = {}

    @property
    def name(self) -> str:
        return "openai"

    @property
    def model_id(self) -> str:
        return f"openai/{self.model}"

    @property
    def is_reasoning_model(self) -> bool:
        """Whether the current model is treated as a reasoning model.

        Reflects the id rule plus anything learned from the API so far.
        """
        return self._caps_for(self.model).reasoning

    def capabilities(self) -> dict[str, Any]:
        """Snapshot of the current model's (guessed + learned) capabilities."""
        caps = self._caps_for(self.model)
        return {
            "model": self.model,
            "reasoning_model": caps.reasoning,
            "token_param": caps.token_param,
            "logprobs": caps.logprobs,
            "logprobs_rejected": caps.logprobs_rejected,
            "dropped_params": sorted(caps.dropped),
        }

    def _ensure_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self._api_key,
                organization=self._organization,
                base_url=self._base_url,
                timeout=self.timeout,
            )
        return self._client

    # ------------------------------------------------------------------
    # Capability tracking + request building
    # ------------------------------------------------------------------

    def _caps_for(self, model: str) -> _ModelCaps:
        caps = self._caps.get(model)
        if caps is None:
            reasoning = is_reasoning_model(model)
            wants_logprobs = (
                (not reasoning) if self.request_logprobs is None else bool(self.request_logprobs)
            )
            caps = _ModelCaps(
                reasoning=reasoning,
                token_param=(
                    "max_completion_tokens" if uses_max_completion_tokens(model) else "max_tokens"
                ),
                logprobs=wants_logprobs,
            )
            self._caps[model] = caps
        return caps

    def _default_budget(self, caps: _ModelCaps) -> int:
        if caps.reasoning:
            return max(self.max_tokens, self.reasoning_max_tokens)
        return self.max_tokens

    def _build_request(
        self, model: str, prompt: str, caps: _ModelCaps, call_kwargs: dict[str, Any]
    ) -> tuple[dict[str, Any], list[str]]:
        """Build the ``chat.completions.create`` kwargs for *caps*.

        Returns ``(request, dropped)`` where *dropped* lists caller/default
        parameters removed because this model is known not to accept them.
        """
        extra = dict(call_kwargs)
        explicit_completion = extra.pop("max_completion_tokens", None)
        explicit_max = extra.pop("max_tokens", None)
        budget = explicit_completion if explicit_completion is not None else explicit_max
        if budget is None:
            budget = self._default_budget(caps)

        request: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if caps.logprobs:
            request["logprobs"] = True
            request["top_logprobs"] = self.top_logprobs
        else:
            extra.pop("logprobs", None)
            extra.pop("top_logprobs", None)
        request[caps.token_param] = int(budget)
        if self.reasoning_effort is not None:
            extra.setdefault("reasoning_effort", self.reasoning_effort)
        request.update(extra)
        if not request.get("logprobs"):
            request.pop("logprobs", None)
            request.pop("top_logprobs", None)

        dropped = sorted(p for p in caps.dropped if p in request)
        for p in dropped:
            request.pop(p)
        return request, dropped

    def _learn(self, caps: _ModelCaps, param: str) -> None:
        """Record that the model rejected *param* so the next request avoids it."""
        if param == "max_tokens":
            caps.token_param = "max_completion_tokens"
            # Only reasoning models reject ``max_tokens`` — give them the
            # reasoning budget so the retry is not spent entirely on thinking.
            caps.reasoning = True
        elif param == "max_completion_tokens":
            caps.token_param = "max_tokens"
        elif param in ("logprobs", "top_logprobs"):
            caps.logprobs = False
            caps.logprobs_rejected = True
        else:
            caps.dropped.add(param)

    def _create(
        self,
        client: Any,
        model: str,
        prompt: str,
        caps: _ModelCaps,
        call_kwargs: dict[str, Any],
    ) -> tuple[Any, dict[str, Any], list[str], list[str], float]:
        """Call the API, correcting unsupported parameters on 400 and retrying.

        Returns ``(response, request, dropped, fallbacks, latency_ms)``.
        """
        fixed_groups: set[str] = set()
        fallbacks: list[str] = []
        for _attempt in range(_MAX_ATTEMPTS):
            request, dropped = self._build_request(model, prompt, caps, call_kwargs)
            t0 = time.perf_counter()
            try:
                resp = client.chat.completions.create(**request)
            except Exception as exc:
                param = _unsupported_param(exc, request)
                group = _ADJUSTABLE_PARAMS.get(param or "")
                if param is None or group is None or group in fixed_groups:
                    raise
                fixed_groups.add(group)
                fallbacks.append(param)
                self._learn(caps, param)
                continue
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return resp, request, dropped, fallbacks, latency_ms
        raise RuntimeError(  # pragma: no cover — each group is fixed at most once
            "OpenAIProvider: exhausted parameter fallbacks"
        )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, prompt: str, **kwargs: Any) -> ProviderOutput:
        client = self._ensure_client()
        # A per-call ``model=`` override gets its own capability record.
        model = str(kwargs.pop("model", None) or self.model)
        caps = self._caps_for(model)
        resp, request, dropped, fallbacks, latency_ms = self._create(
            client, model, prompt, caps, kwargs
        )

        choice = resp.choices[0]
        content = choice.message.content or ""
        finish_reason = getattr(choice, "finish_reason", None)

        # Real top-k probabilities for the FIRST sampled token.
        top_tokens: list[tuple[str, float]] = []
        first = None
        try:
            if choice.logprobs and choice.logprobs.content:
                first = choice.logprobs.content[0]
        except (AttributeError, IndexError):
            first = None
        if first is not None and getattr(first, "top_logprobs", None):
            for alt in first.top_logprobs:
                top_tokens.append((str(alt.token), float(math.exp(alt.logprob))))
        if first is not None and not any(t[0] == first.token for t in top_tokens):
            top_tokens.insert(0, (str(first.token), float(math.exp(first.logprob))))
        has_logprobs = bool(top_tokens)
        if not has_logprobs:
            # Black-box convention: the sampled token at a 1.0 placeholder,
            # flagged via meta['has_logprobs'] — never an invented probability.
            top_tokens = [(whitespace_tokens(content)[0], 1.0)]

        logprobs_requested = bool(request.get("logprobs"))
        budget_exhausted = not content and finish_reason == "length"
        token_param = (
            "max_completion_tokens" if "max_completion_tokens" in request else "max_tokens"
        )

        tokens = whitespace_tokens(content) if content else [""]
        meta: dict[str, Any] = {
            "provider": "OpenAIProvider",
            "model": model,
            "latency_ms": latency_ms,
            "finish_reason": finish_reason,
            "completion": content,
            "reasoning_model": caps.reasoning,
            "token_param": token_param,
            "token_budget": request.get(token_param),
            "logprobs_requested": logprobs_requested,
            "has_logprobs": has_logprobs,
            "evidence_note": "",
        }
        if not has_logprobs:
            meta["logprobs_unavailable_reason"] = self._logprobs_reason(caps, logprobs_requested)
        if "reasoning_effort" in request:
            meta["reasoning_effort"] = request["reasoning_effort"]
        if dropped:
            meta["dropped_params"] = dropped
        if fallbacks:
            meta["param_fallbacks"] = fallbacks
        refusal = getattr(choice.message, "refusal", None)
        if isinstance(refusal, str) and refusal:
            meta["refusal"] = refusal
        if budget_exhausted:
            meta["budget_exhausted"] = True

        usage = getattr(resp, "usage", None)
        if usage is not None:
            meta["usage"] = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
            details = getattr(usage, "completion_tokens_details", None)
            reasoning_tokens = getattr(details, "reasoning_tokens", None)
            if isinstance(reasoning_tokens, int):
                meta["usage"]["reasoning_tokens"] = reasoning_tokens

        meta["evidence_note"] = self._evidence_note(meta)

        return ProviderOutput(
            prompt=prompt,
            tokens=tokens,
            token_ids=[],
            activations=None,
            attentions=None,
            logits=None,
            top_tokens=top_tokens,
            evidence_kind="black_box",
            meta=meta,
        )

    def _logprobs_reason(self, caps: _ModelCaps, requested: bool) -> str:
        """Why no real logprobs: api_rejected | reasoning_model | disabled | not_returned."""
        if requested:
            return "not_returned"
        if caps.logprobs_rejected:
            return "api_rejected"
        if self.request_logprobs is False:
            return "disabled"
        return "reasoning_model"

    @staticmethod
    def _evidence_note(meta: dict[str, Any]) -> str:
        if meta["has_logprobs"]:
            note = "Top-token probabilities come from real OpenAI logprobs."
        else:
            why = {
                "api_rejected": "the API rejected logprobs for this model",
                "reasoning_model": "reasoning models do not expose logprobs",
                "disabled": "logprobs were not requested (request_logprobs=False)",
                "not_returned": "the API returned none",
            }[meta["logprobs_unavailable_reason"]]
            note = (
                f"No OpenAI logprobs ({why}); the top-token probability is a 1.0 "
                "placeholder for the sampled completion, not a real probability."
            )
        note += " Internal activations are not exposed by the API."
        if meta.get("budget_exhausted"):
            note += (
                f" The {meta['token_budget']}-token budget ran out (finish_reason='length') "
                "before any visible output"
            )
            if meta["reasoning_model"]:
                note += (
                    " — reasoning models spend the budget on hidden reasoning first; "
                    "raise reasoning_max_tokens or lower reasoning_effort"
                )
            note += "."
        return note
