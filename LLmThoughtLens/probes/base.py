"""BaseProbe — abstract base class for every LLmThoughtLens probe.

A probe takes a :class:`~LLmThoughtLens.providers.base.BaseProvider`,
runs a small set of prompts against it, and returns a
:class:`ProbeResult` whose ``score`` is in ``[0, 1]`` and whose
``evidence`` dict contains the raw model outputs that motivated the score.

Getting a completion from any provider
--------------------------------------
Behavioural probes score what a model *says*, so they need a completion.
:meth:`ProviderOutput.tokens` cannot serve: for white-box providers it holds
the **prompt** tokens (the forward pass covers the prompt only), and only
black-box providers put their completion there.  :func:`complete` is the one
place that turns ``(provider, prompt)`` into a :class:`Completion`:

* **HuggingFace** (any provider exposing a ``hooked``
  :class:`~LLmThoughtLens.models.hooked.HookedModel` and
  ``supports_gradients``) — real decoding with ``HookedModel.generate``:
  greedy at ``temperature=0`` (deterministic), seeded sampling otherwise.
  The chat template is applied when the tokenizer has one (``framing="chat"``),
  else the prompt is continued as plain text (``framing="completion"``, a
  base LM).  ``first_token_prob`` is the model's real probability of the
  first generated token.
* **Black-box APIs** (OpenAI, Anthropic, Ollama) — ``provider.run`` with
  provider-specific generation kwargs; the completion is
  ``meta["completion"]``.  ``first_token_prob`` is set only when the provider
  reports real logprobs (``meta["has_logprobs"]``); the 1.0 placeholder is
  never used as a probability.
* **Synthetic providers** (the mock) — there is no model, so the "completion"
  is the argmax of the synthetic next-token logits and the result is flagged
  ``synthetic=True``.  Probe numbers from such providers are not model
  findings and are labelled as such.

Any residual ``<think>…</think>`` reasoning is removed from the completion
before scoring (kept in ``Completion.thinking``).

Usage accounting
----------------
Every :func:`complete` call reports its token usage, latency and cost to the
innermost active :func:`record_usage` meter, which is how the benchmark
runner (:mod:`LLmThoughtLens.bench`) measures per-cell usage without changing
the probe API.  Custom probes that call ``provider.run`` directly still work
but are not metered.

Concrete probes live in :mod:`LLmThoughtLens.probes.builtin`.
"""

from __future__ import annotations

import abc
import contextlib
import contextvars
import copy
import math
import re
import time
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal

if TYPE_CHECKING:
    from LLmThoughtLens.providers.base import BaseProvider

__all__ = [
    "BaseProbe",
    "Completion",
    "GenerationConfig",
    "ProbeResult",
    "UsageMeter",
    "answer_token_prob",
    "complete",
    "is_synthetic_provider",
    "record_usage",
]

CompletionSource = Literal["generate", "api", "next_token"]


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class ProbeResult:
    """Structured result returned by every probe.

    ``evidence["synthetic"]`` is ``True`` when the result came from a
    synthetic provider (the mock); :attr:`synthetic` exposes it.
    """

    probe_name: str
    score: float = 0.0
    passed: bool = False
    evidence: dict[str, Any] = field(default_factory=dict)
    summary: str = ""

    @property
    def synthetic(self) -> bool:
        """Whether the result came from a synthetic (non-model) provider."""
        return bool(self.evidence.get("synthetic", False))

    def as_dict(self) -> dict[str, Any]:
        return {
            "probe_name": self.probe_name,
            "score": float(self.score),
            "passed": bool(self.passed),
            "summary": self.summary,
            "synthetic": self.synthetic,
            "evidence": _scrub(self.evidence),
        }

    def __repr__(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        tag = " synthetic" if self.synthetic else ""
        return f"ProbeResult({verdict} {self.probe_name!r} score={self.score:.2f}{tag})"


def _scrub(d: Any) -> Any:
    """Drop non-JSON-safe entries so ``as_dict`` always serialises cleanly."""
    if isinstance(d, dict):
        return {
            k: _scrub(v) for k, v in d.items() if _is_json_safe(v) or isinstance(v, (dict, list))
        }
    if isinstance(d, list):
        return [_scrub(v) for v in d if _is_json_safe(v) or isinstance(v, (dict, list))]
    return d


def _is_json_safe(v: Any) -> bool:
    if isinstance(v, float):
        return math.isfinite(v)
    return isinstance(v, (str, int, bool)) or v is None


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationConfig:
    """Decoding settings shared by every probe call.

    Parameters
    ----------
    max_new_tokens:
        Default completion budget; probes that need more (chain-of-thought)
        or less (refusal openings) pass their own budget to :func:`complete`.
    temperature:
        ``0`` = greedy / deterministic (default).  ``> 0`` samples.
    seed:
        Sampling seed (used only when ``temperature > 0``).
    chat:
        HuggingFace only: ``None`` applies the tokenizer's chat template when it
        has one; ``True`` / ``False`` force it on / off.
    """

    max_new_tokens: int = 48
    temperature: float = 0.0
    seed: int | None = 0
    chat: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "seed": self.seed,
            "chat": self.chat,
        }


@dataclass
class Completion:
    """One model completion as seen by a probe.

    Attributes
    ----------
    text:
        The visible completion that probes score (thinking removed).
    source:
        ``"generate"`` — decoded locally with ``HookedModel.generate``;
        ``"api"`` — returned by a black-box API;
        ``"next_token"`` — a single argmax token (no generation available,
        e.g. the synthetic mock).
    framing:
        ``"chat"`` (chat template / chat API), ``"completion"`` (plain-text
        continuation of a base LM) or ``"synthetic"``.
    first_token_prob:
        The model's real probability of the first completion token, or
        ``None`` when the backend exposes none.  Never a placeholder.  For
        chat models this is often a function word ("The"), not the answer —
        see :func:`answer_token_prob`.
    token_logprobs:
        ``[(token_text, logprob), …]`` for the visible completion when the
        backend reports per-token logprobs (HuggingFace, Ollama >= 0.12.11),
        else ``None``.
    """

    prompt: str
    text: str
    source: CompletionSource
    framing: str
    synthetic: bool
    evidence_kind: str
    first_token_prob: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float = 0.0
    cost_usd: float | None = None
    thinking: str = ""
    stop_reason: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    token_logprobs: list[tuple[str, float]] | None = None

    @property
    def truncated(self) -> bool:
        """``True`` when decoding stopped on the token budget, not on its own."""
        return str(self.stop_reason or "") in ("max_new_tokens", "length", "max_tokens")

    def as_evidence(self) -> dict[str, Any]:
        """Compact, JSON-safe record for ``ProbeResult.evidence``."""
        out: dict[str, Any] = {
            "prompt": self.prompt,
            "response": self.text,
            "source": self.source,
            "framing": self.framing,
            "first_token_prob": self.first_token_prob,
        }
        if self.stop_reason is not None:
            out["stop_reason"] = self.stop_reason
        if self.thinking:
            out["thinking_chars"] = len(self.thinking)
        return out


@dataclass
class UsageMeter:
    """Accumulates usage across :func:`complete` calls (see :func:`record_usage`)."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_tokens_known: bool = True
    completion_tokens_known: bool = True
    latency_ms: float = 0.0
    cost_usd: float | None = None
    completions: list[Completion] = field(default_factory=list)

    def add(self, c: Completion) -> None:
        self.calls += 1
        self.latency_ms += c.latency_ms
        if c.prompt_tokens is None:
            self.prompt_tokens_known = False
        else:
            self.prompt_tokens += int(c.prompt_tokens)
        if c.completion_tokens is None:
            self.completion_tokens_known = False
        else:
            self.completion_tokens += int(c.completion_tokens)
        if c.cost_usd is not None:
            self.cost_usd = (self.cost_usd or 0.0) + float(c.cost_usd)
        self.completions.append(c)

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe totals; token counts are ``None`` when any call lacked them."""
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens if self.prompt_tokens_known else None,
            "completion_tokens": self.completion_tokens if self.completion_tokens_known else None,
            "latency_ms": round(self.latency_ms, 3),
            "cost_usd": self.cost_usd,
        }


_METERS: contextvars.ContextVar[tuple[UsageMeter, ...]] = contextvars.ContextVar(
    "llmthoughtlens_usage_meters", default=()
)


@contextlib.contextmanager
def record_usage() -> Iterator[UsageMeter]:
    """Collect usage of every :func:`complete` call made inside the block.

    Meters nest: an inner block's calls are also counted by outer blocks.
    """
    meter = UsageMeter()
    token = _METERS.set((*_METERS.get(), meter))
    try:
        yield meter
    finally:
        _METERS.reset(token)


def is_synthetic_provider(provider: Any) -> bool:
    """``True`` for providers whose outputs are not produced by a real model.

    A provider may declare itself with a boolean ``is_synthetic`` attribute;
    otherwise the built-in :class:`~LLmThoughtLens.providers.mock_provider.MockProvider`
    (registry name ``"mock"``) is the only synthetic provider.
    """
    flag = getattr(provider, "is_synthetic", None)
    if isinstance(flag, bool):
        return flag
    from LLmThoughtLens.providers.mock_provider import MockProvider

    return isinstance(provider, MockProvider)


def _has_hooked_model(provider: Any) -> bool:
    return bool(getattr(provider, "supports_gradients", False)) and isinstance(
        getattr(type(provider), "hooked", None), property
    )


def _api_generation_kwargs(
    provider: Any, cfg: GenerationConfig, max_new_tokens: int
) -> dict[str, Any]:
    """Map *cfg* onto the built-in black-box providers' ``run`` kwargs."""
    name = getattr(provider, "name", "")
    sampling = cfg.temperature > 0
    if name == "ollama":
        options: dict[str, Any] = {"temperature": cfg.temperature, "num_predict": max_new_tokens}
        if cfg.seed is not None:
            options["seed"] = int(cfg.seed)
        return {"options": options}
    if name == "openai":
        kw: dict[str, Any] = {"temperature": cfg.temperature}
        # Reasoning models share the budget with hidden reasoning: keep the
        # provider's own (larger) default budget for them.
        if not getattr(provider, "is_reasoning_model", False):
            kw["max_tokens"] = max_new_tokens
        if sampling and cfg.seed is not None:
            kw["seed"] = int(cfg.seed)
        return kw
    if name == "anthropic":
        return {"temperature": cfg.temperature, "max_tokens": max_new_tokens}
    return {}


def _usage_from_meta(meta: dict[str, Any]) -> tuple[int | None, int | None]:
    usage = meta.get("usage")
    if not isinstance(usage, dict):
        return None, None
    pt = usage.get("prompt_tokens", usage.get("input_tokens"))
    ct = usage.get("completion_tokens", usage.get("output_tokens"))
    return (int(pt) if isinstance(pt, int) else None, int(ct) if isinstance(ct, int) else None)


def _fold_char(ch: str) -> str:
    out = unicodedata.normalize("NFKD", ch.translate(_QUOTES))
    out = "".join(c for c in out if not unicodedata.combining(c)).lower()
    return out.replace("-", " ")


_QUOTES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"'})


def answer_token_prob(completion: Completion, phrase: str) -> float | None:
    """Probability of the token where *phrase* first appears in the completion.

    The match is case-, accent- and hyphen-insensitive and whole-word (as in
    the probes' scoring).  This is the model's confidence in the *answer*
    token (``"Paris"`` in "The capital of France is Paris"), unlike
    :attr:`Completion.first_token_prob`.  ``None`` when the backend reported
    no per-token logprobs or the phrase does not occur in the tokens.
    """
    lps = completion.token_logprobs
    if not lps or not phrase.strip():
        return None
    folded: list[str] = []
    owner: list[int] = []  # folded char -> token index
    for i, (tok, _) in enumerate(lps):
        for ch in tok:
            f = _fold_char(ch)
            folded.append(f)
            owner.extend([i] * len(f))
    text = "".join(folded)
    target = re.sub(r"\s+", " ", "".join(_fold_char(c) for c in phrase.strip()))
    m = re.search(r"(?<![a-z0-9])" + re.escape(target) + r"(?![a-z0-9])", text)
    if m is None:
        return None
    return float(math.exp(lps[owner[m.start()]][1]))


def _visible_generation_text(tokenizer: Any, gen: Any) -> str:
    """Decode a :class:`GenerateResult` without its stop token or special tokens.

    ``GenerateResult.text`` decodes every generated id, including the EOS /
    end-of-turn token that stopped decoding (``"<|endoftext|>"``); that is not
    part of the answer a probe should score.
    """
    ids = list(gen.token_ids)
    if gen.stop_reason == "eos" and ids:
        ids = ids[:-1]
    if not ids:
        return ""
    try:
        return str(tokenizer.decode(ids, skip_special_tokens=True))
    except TypeError:  # minimal tokenizers without the keyword
        return str(tokenizer.decode(ids))


def complete(
    provider: BaseProvider,
    prompt: str,
    config: GenerationConfig | None = None,
    *,
    max_new_tokens: int | None = None,
) -> Completion:
    """Return *provider*'s completion of *prompt* (see the module docstring).

    Parameters
    ----------
    provider:
        Any :class:`~LLmThoughtLens.providers.base.BaseProvider`.
    prompt:
        User prompt (chat-framed automatically where the backend supports it).
    config:
        Decoding settings; defaults to :class:`GenerationConfig()` (greedy).
    max_new_tokens:
        Overrides ``config.max_new_tokens`` for this call.

    Returns
    -------
    Completion
        Also reported to every active :func:`record_usage` meter.
    """
    from LLmThoughtLens.providers.ollama_provider import strip_think_blocks

    cfg = config or GenerationConfig()
    budget = int(max_new_tokens if max_new_tokens is not None else cfg.max_new_tokens)
    t0 = time.perf_counter()

    if is_synthetic_provider(provider):
        out = provider.run(prompt)
        comp = Completion(
            prompt=prompt,
            text=out.output_token,
            source="next_token",
            framing="synthetic",
            synthetic=True,
            evidence_kind=out.evidence_kind,
            first_token_prob=None,
            meta={"synthetic_top_tokens": [list(t) for t in out.top_tokens[:5]]},
        )
    elif _has_hooked_model(provider):
        hm = provider.hooked  # type: ignore[attr-defined]
        chat = bool(getattr(hm.tokenizer, "chat_template", None)) if cfg.chat is None else cfg.chat
        gen = hm.generate(
            prompt,
            max_new_tokens=budget,
            temperature=float(cfg.temperature),
            seed=cfg.seed,
            chat=chat,
            return_logprobs=True,
        )
        raw_text = _visible_generation_text(hm.tokenizer, gen)
        text, thinking, _ = strip_think_blocks(raw_text)
        # The first generated token's probability describes the answer only
        # when no reasoning block precedes it.
        first_prob = math.exp(gen.logprobs[0]) if gen.logprobs and not thinking else None
        chat_applied = bool(chat and getattr(hm.tokenizer, "chat_template", None))
        token_lps: list[tuple[str, float]] | None = None
        if gen.logprobs and not thinking:
            n_visible = len(gen.token_ids) - (1 if gen.stop_reason == "eos" else 0)
            token_lps = [
                (str(t), float(lp))
                for t, lp in zip(gen.tokens[:n_visible], gen.logprobs[:n_visible], strict=False)
            ]
        comp = Completion(
            prompt=prompt,
            text=text.strip(),
            source="generate",
            framing="chat" if chat_applied else "completion",
            synthetic=False,
            evidence_kind="white_box",
            first_token_prob=first_prob,
            prompt_tokens=len(gen.prompt_token_ids),
            completion_tokens=len(gen.token_ids),
            thinking=thinking,
            stop_reason=gen.stop_reason,
            meta={"family": getattr(hm, "family", None)},
            token_logprobs=token_lps,
        )
    else:
        kwargs = _api_generation_kwargs(provider, cfg, budget)
        dropped: list[str] = []
        try:
            out = provider.run(prompt, **kwargs)
        except Exception:
            if "temperature" not in kwargs:
                raise
            # Some API models reject sampling parameters; retry without them
            # once and record it, rather than failing the probe.
            dropped = ["temperature"]
            kwargs = {k: v for k, v in kwargs.items() if k not in ("temperature", "seed")}
            out = provider.run(prompt, **kwargs)
        meta = out.meta or {}
        if "completion" in meta:
            raw = str(meta.get("completion") or "")
            source: CompletionSource = "api"
        elif out.evidence_kind == "black_box":
            raw = " ".join(out.tokens)
            source = "api"
        else:
            raw = out.output_token
            source = "next_token"
        text, thinking, _ = strip_think_blocks(raw)
        thinking = str(meta.get("thinking") or "") or thinking
        real_probs = meta.get("has_logprobs") is True
        pt, ct = _usage_from_meta(meta)
        cost = meta.get("api_cost_usd")
        framing = "chat" if source == "api" else "completion"
        if meta.get("framing") == "raw":
            framing = "completion"
        extra: dict[str, Any] = {"has_logprobs": real_probs}
        if dropped:
            extra["dropped_generation_params"] = dropped
        stop = meta.get("done_reason") or meta.get("finish_reason") or meta.get("stop_reason")
        api_lps: list[tuple[str, float]] | None = None
        raw_lps = meta.get("token_logprobs") if real_probs else None
        if isinstance(raw_lps, list) and raw_lps:
            try:
                api_lps = [(str(t), float(lp)) for t, lp in raw_lps]
            except (TypeError, ValueError):
                api_lps = None
        comp = Completion(
            prompt=prompt,
            text=text.strip(),
            source=source,
            framing=framing,
            synthetic=False,
            evidence_kind=out.evidence_kind,
            first_token_prob=float(out.output_prob) if real_probs and out.top_tokens else None,
            prompt_tokens=pt,
            completion_tokens=ct,
            cost_usd=float(cost) if isinstance(cost, (int, float)) else None,
            thinking=thinking,
            stop_reason=str(stop) if stop is not None else None,
            meta=extra,
            token_logprobs=api_lps,
        )

    comp.latency_ms = (time.perf_counter() - t0) * 1000.0
    for meter in _METERS.get():
        meter.add(comp)
    return comp


# ---------------------------------------------------------------------------
# BaseProbe
# ---------------------------------------------------------------------------


class BaseProbe(abc.ABC):
    """Every probe must subclass this and implement :meth:`run`.

    Built-in probes obtain completions through :meth:`complete` so they work
    identically on white-box, black-box and synthetic providers, and finish
    with :meth:`annotate` so every result records its provenance
    (``synthetic``, ``framing``, ``completion_source``, ``caveats``).
    """

    #: Short, machine-readable name (snake_case).
    name: str = "base_probe"
    #: One-line human description for the report.
    description: str = ""
    #: Optional citation back to the paper case study.
    citation: str = ""
    #: ``"completion"`` — the prompt is a sentence a base LM can continue;
    #: ``"instruction"`` — the prompt is an instruction that needs an
    #: instruction-tuned (chat) model to be meaningful.
    style: ClassVar[str] = "instruction"
    #: Human-readable pass rule (documented in the scorecard).
    threshold: ClassVar[str] = ""
    #: Decoding settings; override per instance via ``__init__`` or :meth:`with_generation`.
    generation: GenerationConfig = GenerationConfig()

    def __init__(self, generation: GenerationConfig | None = None) -> None:
        if generation is not None:
            self.generation = generation

    def with_generation(self, generation: GenerationConfig) -> BaseProbe:
        """Shallow copy of this probe that decodes with *generation*."""
        clone = copy.copy(self)
        clone.generation = generation
        return clone

    @abc.abstractmethod
    def run(
        self,
        provider: BaseProvider,
        prompt: str | None = None,
    ) -> ProbeResult:
        """Execute the probe and return its :class:`ProbeResult`.

        Parameters
        ----------
        provider:
            Backend that will receive the probe's prompts.
        prompt:
            Optional override of the probe's built-in prompt(s).  When
            ``None`` the probe uses its hard-coded prompt set.
        """

    # ------------------------------------------------------------------
    # Helpers for subclasses
    # ------------------------------------------------------------------

    def complete(
        self, provider: BaseProvider, prompt: str, *, max_new_tokens: int | None = None
    ) -> Completion:
        """:func:`complete` with this probe's :attr:`generation` settings."""
        return complete(provider, prompt, self.generation, max_new_tokens=max_new_tokens)

    def annotate(self, result: ProbeResult, completions: list[Completion]) -> ProbeResult:
        """Stamp provenance onto *result* and label synthetic results.

        Adds ``synthetic``, ``evidence_kind``, ``framing``,
        ``completion_source``, ``generation`` and ``caveats`` to the evidence.
        A synthetic result's summary is prefixed so it can never be read as a
        model finding.
        """
        synthetic = any(c.synthetic for c in completions)
        framings = sorted({c.framing for c in completions})
        sources = sorted({c.source for c in completions})
        caveats: list[str] = list(result.evidence.get("caveats", []))
        if synthetic:
            caveats.append(
                "Synthetic provider: the 'completion' is one argmax token from random "
                "logits; this score says nothing about any real model."
            )
        elif self.style == "instruction" and "completion" in framings:
            caveats.append(
                "Instruction-style probe run as plain-text continuation (base LM, no chat "
                "template): a low score reflects missing instruction tuning, not necessarily "
                "a missing capability."
            )
        result.evidence.update(
            {
                "synthetic": synthetic,
                "evidence_kind": completions[0].evidence_kind if completions else None,
                "framing": framings[0] if len(framings) == 1 else framings,
                "completion_source": sources[0] if len(sources) == 1 else sources,
                "generation": self.generation.as_dict(),
                "caveats": caveats,
            }
        )
        if synthetic and not result.summary.startswith("[synthetic"):
            result.summary = "[synthetic provider - not a model finding] " + result.summary
        return result

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
