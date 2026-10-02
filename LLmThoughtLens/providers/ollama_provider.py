"""Ollama provider — black-box adapter that calls a local Ollama HTTP server.

Ollama does not expose activations or attentions, so this is a black-box
backend.  What it *can* expose is handled carefully:

Logprobs
    Ollama >= 0.12.11 returns per-token ``logprobs`` from ``/api/generate``
    when asked, which lets ``top_tokens`` carry **real graded probabilities**
    instead of the sampled token at a 1.0 placeholder.  ``request_logprobs``
    controls this: ``True`` (default) always asks — older servers ignore the
    field and the provider falls back; ``"auto"`` first reads
    ``GET /api/version`` once and only asks when the server is new enough;
    ``False`` never asks.  When no usable logprobs come back,
    ``meta["has_logprobs"]`` is ``False``, ``meta["logprobs_unavailable_reason"]``
    says why, and the 1.0 entry is flagged as a placeholder — probabilities are
    never invented.

Thinking models
    Reasoning models (qwen3, deepseek-r1, gpt-oss, …) think by default.  On a
    small token budget the whole budget can be spent inside the hidden
    reasoning, leaving an empty answer, and the server's ``logprobs`` start at
    the ``<think>`` token, not the answer.  With ``think=None`` (default, "auto")
    the provider sends ``"think": false`` (Ollama >= 0.9.0) for models whose
    ``/api/show`` capabilities include ``"thinking"`` (when :meth:`show` /
    :meth:`server_info` has been called) or whose name matches a known
    thinking family, and for any model seen emitting thinking earlier.  Any
    residual ``<think>…</think>`` text is stripped from the completion
    (kept in ``meta["thinking"]``), and the logprob stream is realigned to the
    first *visible* answer token; when that is impossible, logprobs are
    reported unavailable rather than describing a thinking token.

Prompt framing
    ``/api/generate`` (``raw`` unset) renders the prompt through the model's
    own template, i.e. as a single user turn for chat/instruct models and as
    plain text for base models whose template is ``{{ .Prompt }}``.
    ``meta["framing"]`` records ``"template"`` or ``"raw"``.

Robustness
    Connection errors, timeouts and HTTP 429/502/503/504 are retried
    ``max_retries`` times with exponential backoff.  Other HTTP errors
    (Ollama uses 400/404/500 for bad requests, missing models and load
    failures, which do not heal on retry) raise :class:`httpx.HTTPStatusError`
    immediately, with the server's ``error`` text in the message.
"""

from __future__ import annotations

import math
import re
import time
from typing import TYPE_CHECKING, Any, Literal

from LLmThoughtLens.providers.base import BaseProvider, ProviderOutput
from LLmThoughtLens.providers.defaults import default_ollama_url, resolve_model
from LLmThoughtLens.utils.tokenizer_utils import whitespace_tokens

if TYPE_CHECKING:
    import httpx

__all__ = [
    "LOGPROBS_MIN_VERSION",
    "THINK_MIN_VERSION",
    "OllamaProvider",
    "looks_like_thinking_model",
    "parse_version",
    "strip_think_blocks",
]

#: First Ollama release whose API returns token logprobs (v0.12.11 release notes).
LOGPROBS_MIN_VERSION: tuple[int, int, int] = (0, 12, 11)
#: First Ollama release that accepts the ``think`` request field (v0.9.0 release notes).
THINK_MIN_VERSION: tuple[int, int, int] = (0, 9, 0)

#: HTTP statuses worth retrying (rate limiting / gateway / temporarily unavailable).
_RETRY_STATUS = frozenset({429, 502, 503, 504})

#: Model-name prefixes of Ollama families that think by default.  Used only
#: when ``/api/show`` capabilities have not been fetched.
_THINKING_PREFIXES: tuple[str, ...] = (
    "qwen3",
    "qwq",
    "deepseek-r1",
    "deepseek-v3.1",
    "gpt-oss",
    "magistral",
    "phi4-reasoning",
    "phi4-mini-reasoning",
    "cogito",
    "exaone-deep",
    "openthinker",
    "smallthinker",
)
#: Prefixes that match a thinking family above but are not thinking models.
_NON_THINKING_PREFIXES: tuple[str, ...] = ("qwen3-coder", "qwen3-embedding")

_THINK_BLOCK = re.compile(r"<(think|thinking)>(.*?)</\1>", re.IGNORECASE | re.DOTALL)
_THINK_OPEN = re.compile(r"<think(?:ing)?>", re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think(?:ing)?>", re.IGNORECASE)

_UNSET: Any = object()


def parse_version(version: str | None) -> tuple[int, int, int] | None:
    """Parse ``"0.12.11"`` / ``"v0.9.0-rc1"`` into a 3-tuple.

    Returns ``None`` for unparseable strings and for ``0.0.0``, which is what
    Ollama development builds report (their feature set is unknown).
    """
    if not version:
        return None
    m = re.match(r"\s*v?(\d+)(?:\.(\d+))?(?:\.(\d+))?", str(version))
    if not m:
        return None
    parts = (int(m.group(1)), int(m.group(2) or 0), int(m.group(3) or 0))
    return None if parts == (0, 0, 0) else parts


def looks_like_thinking_model(model: str) -> bool:
    """Name heuristic: does *model* belong to a family that thinks by default?

    ``"qwen3:1.7b"`` and ``"library/deepseek-r1:8b"`` → ``True``;
    ``"llama3.1:8b"`` and ``"qwen3-coder:30b"`` → ``False``.  Exact
    capabilities come from :meth:`OllamaProvider.show` instead.
    """
    base = model.lower().rsplit("/", 1)[-1].split(":", 1)[0]
    if base.startswith(_NON_THINKING_PREFIXES):
        return False
    return base.startswith(_THINKING_PREFIXES)


def strip_think_blocks(text: str) -> tuple[str, str, bool]:
    """Split a completion into ``(visible, thinking, truncated)``.

    * Complete ``<think>…</think>`` (or ``<thinking>``) blocks are removed.
    * A stray closing tag (the template opened the block inside the prompt)
      means everything before it was thinking.
    * An unterminated opening tag (the budget ran out mid-thought) means
      everything after it was thinking; ``truncated`` is then ``True``.

    Text without any think tag is returned unchanged (not even stripped).
    """
    if not _THINK_OPEN.search(text) and not _THINK_CLOSE.search(text):
        return text, "", False
    parts: list[str] = []

    def _grab(m: re.Match[str]) -> str:
        parts.append(m.group(2))
        return ""

    out = _THINK_BLOCK.sub(_grab, text)
    close = _THINK_CLOSE.search(out)
    if close is not None:
        parts.append(out[: close.start()])
        out = out[close.end() :]
    truncated = False
    opened = _THINK_OPEN.search(out)
    if opened is not None:
        parts.append(out[opened.end() :])
        out = out[: opened.start()]
        truncated = True
    thinking = "\n".join(p.strip() for p in parts if p.strip())
    return out.strip(), thinking, truncated


class OllamaProvider(BaseProvider):
    """Provider that calls a locally-running Ollama instance.

    Parameters
    ----------
    model:
        Ollama model tag, e.g. ``"llama3.2"``.  Defaults to
        :func:`~LLmThoughtLens.providers.defaults.default_model`.
    base_url:
        Base URL of the Ollama server.  Defaults to
        :func:`~LLmThoughtLens.providers.defaults.default_ollama_url`.
    timeout:
        Per-request timeout in seconds.
    top_logprobs:
        Number of alternative tokens to request per position (when the
        server supports logprobs).
    request_logprobs:
        ``True`` (default) always asks for logprobs; ``"auto"`` asks only when
        ``GET /api/version`` reports >= :data:`LOGPROBS_MIN_VERSION` (or the
        version cannot be read); ``False`` never asks.
    think:
        Value sent as the ``think`` request field.  ``None`` (default) is
        "auto": send ``False`` for thinking models (see the module docstring)
        and nothing otherwise.  ``True`` / ``"low"`` / ``"medium"`` /
        ``"high"`` are passed through unchanged.
    max_retries:
        Retries for connection errors, timeouts and HTTP 429/502/503/504.
    retry_backoff:
        Base backoff in seconds; attempt *k* sleeps ``retry_backoff * 2**k``.
    transport:
        Optional :class:`httpx.BaseTransport` (e.g. ``httpx.MockTransport``
        in tests).
    """

    evidence_kind = "black_box"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
        top_logprobs: int = 5,
        request_logprobs: bool | Literal["auto"] = True,
        think: bool | str | None = None,
        max_retries: int = 2,
        retry_backoff: float = 0.5,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        try:
            import httpx  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "OllamaProvider needs the `ollama` extra. "
                "Install with: pip install 'LLmThoughtLens[ollama]'"
            ) from exc
        if request_logprobs not in (True, False, "auto"):
            raise ValueError(
                f"request_logprobs must be True, False or 'auto', got {request_logprobs!r}"
            )
        self.model = resolve_model("ollama", model)
        self.base_url = (base_url or default_ollama_url()).rstrip("/")
        self.timeout = float(timeout)
        self.top_logprobs = max(1, min(20, int(top_logprobs)))
        self.request_logprobs: bool | Literal["auto"] = request_logprobs
        self.think = think
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff = max(0.0, float(retry_backoff))
        self._transport = transport
        self._version: Any = _UNSET
        self._show: Any = _UNSET
        self._thinking_seen = False
        #: Retries used by the most recent request (0 = first attempt succeeded).
        self.last_retries = 0

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def model_id(self) -> str:
        return f"ollama/{self.model}"

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        timeout: float | None = None,
        retries: int | None = None,
    ) -> httpx.Response:
        """Send one request with retries on transient failures; raise on HTTP errors."""
        import httpx

        url = f"{self.base_url}{path}"
        attempts = self.max_retries if retries is None else max(0, int(retries))
        for attempt in range(attempts + 1):
            try:
                with httpx.Client(
                    timeout=self.timeout if timeout is None else timeout,
                    transport=self._transport,
                ) as client:
                    resp = client.request(method, url, json=json)
            except httpx.TransportError:
                if attempt < attempts:
                    self._backoff(attempt)
                    continue
                raise
            if resp.status_code in _RETRY_STATUS and attempt < attempts:
                self._backoff(attempt)
                continue
            self.last_retries = attempt
            if resp.status_code >= 400:
                raise httpx.HTTPStatusError(
                    f"Ollama {method} {path} failed with HTTP {resp.status_code}: "
                    f"{_error_detail(resp)}",
                    request=resp.request,
                    response=resp,
                )
            return resp
        raise RuntimeError("unreachable")  # pragma: no cover

    def _backoff(self, attempt: int) -> None:
        delay = self.retry_backoff * (2**attempt)
        if delay > 0:
            time.sleep(delay)

    # ------------------------------------------------------------------
    # Server capabilities (cached; never raise)
    # ------------------------------------------------------------------

    def server_version(self) -> str | None:
        """``GET /api/version`` (cached).  ``None`` when the endpoint is unavailable."""
        if self._version is _UNSET:
            try:
                resp = self._request("GET", "/api/version", timeout=min(self.timeout, 10.0))
                version = resp.json().get("version")
                self._version = str(version) if version else None
            except Exception:  # noqa: BLE001 — absence is a normal answer here
                self._version = None
        return self._version  # type: ignore[no-any-return]

    def show(self) -> dict[str, Any] | None:
        """``POST /api/show`` for this model (cached).  ``None`` when unavailable."""
        if self._show is _UNSET:
            try:
                resp = self._request(
                    "POST", "/api/show", json={"model": self.model}, timeout=min(self.timeout, 30.0)
                )
                data = resp.json()
                self._show = data if isinstance(data, dict) else None
            except Exception:  # noqa: BLE001
                self._show = None
        return self._show  # type: ignore[no-any-return]

    def capabilities(self) -> list[str] | None:
        """Model capabilities from :meth:`show` (e.g. ``["completion", "thinking"]``)."""
        info = self.show()
        caps = info.get("capabilities") if info else None
        return [str(c) for c in caps] if isinstance(caps, list) else None

    def server_info(self) -> dict[str, Any]:
        """Version, feature support and model details — JSON-safe, for run records."""
        version = self.server_version()
        parsed = parse_version(version)
        caps = self.capabilities()
        info = self.show() or {}
        raw_details = info.get("details")
        details: dict[str, Any] = raw_details if isinstance(raw_details, dict) else {}
        return {
            "base_url": self.base_url,
            "version": version,
            "logprobs_supported": None if parsed is None else parsed >= LOGPROBS_MIN_VERSION,
            "think_supported": None if parsed is None else parsed >= THINK_MIN_VERSION,
            "model_capabilities": caps,
            "thinking_model": None if caps is None else "thinking" in caps,
            "model_details": {
                k: details.get(k)
                for k in ("family", "parameter_size", "quantization_level", "format")
                if k in details
            },
        }

    def _known_version(self) -> tuple[int, int, int] | None:
        """Parsed server version if it has already been fetched (never triggers a request)."""
        return None if self._version is _UNSET else parse_version(self._version)

    def _logprobs_plan(self) -> tuple[bool, str | None]:
        """``(request?, reason_if_not)`` for this call."""
        if self.request_logprobs is False:
            return False, "disabled"
        if self.request_logprobs == "auto":
            parsed = parse_version(self.server_version())
            if parsed is not None and parsed < LOGPROBS_MIN_VERSION:
                return False, "server_too_old"
        return True, None

    def _resolve_think(self) -> bool | str | None:
        """The ``think`` field to send, or ``None`` to omit it."""
        if self.think is not None:
            return self.think
        known = self._known_version()
        if known is not None and known < THINK_MIN_VERSION:
            return None
        if self._show is not _UNSET:
            caps = self.capabilities()
            if caps is not None:
                return False if "thinking" in caps else None
        if self._thinking_seen or looks_like_thinking_model(self.model):
            return False
        return None

    # ------------------------------------------------------------------
    # BaseProvider API
    # ------------------------------------------------------------------

    def run(self, prompt: str, **kwargs: Any) -> ProviderOutput:
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            **kwargs,
        }
        want_logprobs, skip_reason = self._logprobs_plan()
        if want_logprobs:
            payload.setdefault("logprobs", True)
            payload.setdefault("top_logprobs", self.top_logprobs)
        think = self._resolve_think()
        if think is not None:
            payload.setdefault("think", think)

        t0 = time.perf_counter()
        data = self._request("POST", "/api/generate", json=payload).json()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        raw: str = data.get("response", "") or ""
        text, inline_thinking, truncated = strip_think_blocks(raw)
        thinking = str(data.get("thinking") or "") or inline_thinking
        if thinking:
            self._thinking_seen = True

        logprobs_requested = bool(payload.get("logprobs"))
        visible_lp, align_reason = _visible_logprobs(data.get("logprobs"), bool(thinking), text)
        top_tokens, used_logprobs = self._first_token_distribution(visible_lp, text)
        tokens = whitespace_tokens(text) if text else [""]

        meta: dict[str, Any] = {
            "provider": "OllamaProvider",
            "model": self.model,
            "latency_ms": latency_ms,
            "eval_count": data.get("eval_count"),
            "eval_duration": data.get("eval_duration"),
            "done_reason": data.get("done_reason"),
            "completion": text,
            "framing": "raw" if payload.get("raw") else "template",
            "has_logprobs": used_logprobs,
            "logprobs_requested": logprobs_requested,
            "think_sent": payload.get("think"),
            "retries": self.last_retries,
        }
        if self._version is not _UNSET and self._version is not None:
            meta["server_version"] = self._version
        usage = {
            "prompt_tokens": data.get("prompt_eval_count"),
            "completion_tokens": data.get("eval_count"),
        }
        if any(v is not None for v in usage.values()):
            meta["usage"] = usage
        if thinking:
            meta["thinking"] = thinking
            meta["thinking_stripped"] = bool(inline_thinking)
            meta["thinking_truncated"] = truncated or (not text and bool(thinking))
            if raw != text:
                meta["raw_completion"] = raw
        if used_logprobs and visible_lp:
            meta["token_logprobs"] = [
                [str(e.get("token", "")), float(e["logprob"])]
                for e in visible_lp
                if isinstance(e, dict) and isinstance(e.get("logprob"), (int, float))
            ]
        if not used_logprobs:
            if not logprobs_requested:
                meta["logprobs_unavailable_reason"] = skip_reason or "disabled"
            else:
                meta["logprobs_unavailable_reason"] = align_reason or "not_returned"
        meta["evidence_note"] = _evidence_note(meta)
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

    # ------------------------------------------------------------------
    # logprobs parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _first_token_distribution(
        logprobs: Any,
        text: str,
    ) -> tuple[list[tuple[str, float]], bool]:
        """Return ``([(token, prob), …], used_logprobs)`` for the FIRST token.

        Ollama's ``logprobs`` is a list (one entry per generated token); each
        entry has ``token``, ``logprob`` and an optional ``top_logprobs``
        list of alternatives.  We expose the first position's distribution —
        that is the next-token prediction the masking engine compares against.
        """
        if isinstance(logprobs, list) and logprobs:
            first = logprobs[0]
            out: list[tuple[str, float]] = []
            alts = first.get("top_logprobs") if isinstance(first, dict) else None
            if isinstance(alts, list) and alts:
                for alt in alts:
                    tok = str(alt.get("token", ""))
                    lp = alt.get("logprob")
                    if lp is not None:
                        out.append((tok, float(math.exp(lp))))
            # Always include the sampled token itself if not already present.
            sampled_tok = str(first.get("token", "")) if isinstance(first, dict) else ""
            sampled_lp = first.get("logprob") if isinstance(first, dict) else None
            if sampled_lp is not None and not any(t == sampled_tok for t, _ in out):
                out.insert(0, (sampled_tok, float(math.exp(sampled_lp))))
            if out:
                out.sort(key=lambda kv: kv[1], reverse=True)
                return out, True

        # Fallback: no logprobs — sampled completion at probability 1.0.
        fallback_tok = whitespace_tokens(text)[0] if text else ""
        return [(fallback_tok, 1.0)], False

    def ping(self) -> bool:
        """Lightweight health check used by the TUI provider-connect screen."""
        import httpx

        try:
            with httpx.Client(timeout=2.0, transport=self._transport) as client:
                resp = client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except Exception:  # noqa: BLE001
            return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _error_detail(resp: httpx.Response) -> str:
    try:
        body = resp.json()
    except Exception:  # noqa: BLE001
        return resp.text[:200]
    if isinstance(body, dict) and body.get("error"):
        return str(body["error"])[:200]
    return str(body)[:200]


def _visible_logprobs(
    logprobs: Any, thinking_present: bool, visible_text: str = ""
) -> tuple[list[Any] | None, str | None]:
    """Logprob entries covering the *visible* answer, or ``(None, reason)``.

    Without thinking the stream is returned unchanged.  With thinking, the
    server's stream starts at ``<think>``: entries up to and including the
    closing tag (and any whitespace-only tokens after it) are dropped.  If the
    stream contains an opening tag but no closing one, no entry belongs to
    the answer and the logprobs are reported unusable.  When thinking was
    reported without any tag in the stream (other reasoning formats), the
    stream is aligned on the visible text instead: the shortest suffix of
    tokens that spells *visible_text* is kept; failing that, the logprobs are
    reported unusable rather than describing a reasoning token.
    """
    if not isinstance(logprobs, list) or not logprobs:
        return None, None
    if not thinking_present:
        return logprobs, None
    toks = [str(e.get("token", "")) if isinstance(e, dict) else "" for e in logprobs]
    close = next((i for i, t in enumerate(toks) if "</think" in t.lower()), None)
    if close is None:
        if any("<think" in t.lower() for t in toks):
            return None, "thinking_unaligned"
        target = visible_text.strip()
        if target:
            for k in range(len(toks) - 1, -1, -1):
                tail = "".join(toks[k:]).strip()
                if tail == target:
                    return logprobs[k:], None
                if len(tail) > len(target):
                    break
        return None, "thinking_unaligned"
    k = close + 1
    while k < len(toks) and not toks[k].strip():
        k += 1
    rest = logprobs[k:]
    return (rest, None) if rest else (None, "no_visible_tokens")


_REASON_TEXT = {
    "disabled": "logprobs were not requested",
    "server_too_old": "the server predates logprobs support (needs Ollama >= 0.12.11)",
    "not_returned": "none were returned; Ollama >= 0.12.11 is needed for graded probabilities",
    "thinking_unaligned": "the completion ended inside a <think> block, so no visible-answer "
    "token has a probability",
    "no_visible_tokens": "no visible-answer token followed the thinking block",
}


def _evidence_note(meta: dict[str, Any]) -> str:
    if meta["has_logprobs"]:
        note = (
            "Ollama exposed real per-token logprobs; top-token "
            "probabilities are genuine model probabilities (black-box: "
            "no activations or attentions are available from this API)."
        )
        if meta.get("thinking"):
            note += " Thinking tokens were skipped: the distribution is for the first answer token."
        return note
    why = _REASON_TEXT.get(str(meta.get("logprobs_unavailable_reason")), "unknown reason")
    return (
        f"This Ollama server did not return logprobs ({why}); the top-token "
        "probability is a 1.0 placeholder for the sampled completion, not a real "
        "probability. No activations or attentions are available from this API."
    )
