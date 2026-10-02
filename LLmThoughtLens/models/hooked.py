"""HookedModel — one residual-stream convention for every HuggingFace causal LM.

:class:`HookedModel` wraps an HF ``*ForCausalLM`` + tokenizer and gives every
white-box feature of LLmThoughtLens (provider capture, logit lens, the live
X-ray, interventions, gradient attribution, SAE hook points, steering) the
same view of the model's internals.  torch / transformers are imported lazily,
so importing this module never requires the ``huggingface`` extra.

Residual-stream convention
--------------------------
For a model with ``L`` transformer blocks (layers ``0 .. L-1``):

``resid_pre[l]``
    The hidden state *entering* block ``l``.  ``resid_pre[0]`` is the
    embedding output exactly as the model feeds it to its first block (token
    + learned position embeddings for GPT-2 / OPT, ``sqrt(d)``-scaled token
    embeddings for Gemma, after OPT's ``project_in``, ...).
``resid_post[l]``
    The hidden state *leaving* block ``l``.  This is the **true residual
    stream**: the final norm is never applied, not even on the last layer.
    (HF's ``output_hidden_states`` replaces its last entry with
    ``final_norm(resid_post[L-1])`` for GPT-2, Llama, Mistral, Qwen, Gemma,
    Phi-3, NeoX, OPT, ... which is why capture here uses forward hooks on the
    blocks instead.)
``resid_post[l] == resid_pre[l + 1]``
    Exactly — they are the same tensor — unless a hook edits one of them.
``logit_lens(resid_post[L - 1]) == model logits``
    :meth:`HookedModel.logit_lens` applies final norm, any post-norm
    projection (OPT ``project_out``), the unembedding and logit soft-capping
    (Gemma-2) exactly once.

Captured tensors record what actually flows downstream, i.e. *after* any
hooks installed at that site.

Hook sites
----------
``"resid_pre"``   forward-pre hook on block ``l`` (edits what enters the block).
``"resid_post"``  forward hook on block ``l`` (edits what leaves the block).
``"mlp_in"``      forward-pre hook on the module whose input is the MLP input
                  (``block.mlp``; OPT: ``fc1``).  For pre-LN models this is the
                  *normalised* residual, not the residual itself.
``"mlp_out"``     forward hook on the MLP output module (``block.mlp``; OPT: ``fc2``).
``"attn_out"``    forward hook on the attention module (its first output).

``mlp_out`` / ``attn_out`` are the sub-module outputs.  For most families
that is exactly what is added to the residual stream; Gemma-2 / Gemma-3 apply
``post_attention_layernorm`` / ``post_feedforward_layernorm`` first.  OPT
flattens tokens before ``fc1``/``fc2``; hook functions still receive
``(1, T, D)`` (single-sequence inputs only).

A hook function is ``fn(hidden) -> Tensor | None`` where ``hidden`` is
``(B, T, D)``.  Return a new tensor to replace the activation, or ``None`` to
observe only.  Do not modify ``hidden`` in place — return a new tensor.
Hooks work under ``torch.no_grad`` and with gradients enabled (edits are
differentiable), and are always removed when the :meth:`HookedModel.hooks`
context exits, including on exceptions.

Token positions and the KV cache
--------------------------------
``ResidHook(positions=...)`` restricts a hook to some token positions; ``fn``
then receives only those rows, ``(B, k, D)``.  Positions are **absolute
indices into the full sequence** (prompt + generated tokens so far).  The
absolute offset of the chunk being processed is read from the
``position_ids`` / ``cache_position`` the model passes to its first block
(falling back to HookedModel's own bookkeeping), so:

* ``positions=None`` — every position of every forward chunk.  With a KV
  cache each position is processed (and edited) exactly once; without a
  cache every step recomputes, and edits, every position.  Both give the
  same result (e.g. steering every token).
* non-negative positions — edit exactly those tokens; identical results with
  or without a KV cache.
* negative positions — resolved against the sequence processed *so far*
  (``-1`` = the newest token).  In cached generation each new token is edited
  when it is fed and the edit persists in the cache; in uncached generation
  only the current last position is edited on each recompute.  Use
  non-negative positions or ``None`` when you need cache-independent
  behaviour.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from LLmThoughtLens.models.families import Architecture, resolve_architecture

if TYPE_CHECKING:
    import torch

__all__ = [
    "SITES",
    "ForwardResult",
    "GenerateResult",
    "HookSite",
    "HookedModel",
    "ResidHook",
    "TokenizedPrompt",
    "infer_device",
    "load_hf_model",
    "register_site_hook",
    "resolve_device",
    "resolve_dtype",
]

HookSite = Literal["resid_pre", "resid_post", "mlp_in", "mlp_out", "attn_out"]
#: Every hook site :class:`HookedModel` understands.
SITES: tuple[str, ...] = ("resid_pre", "resid_post", "mlp_in", "mlp_out", "attn_out")

HookFn = Callable[[Any], Any]

_DTYPE_ALIASES = {
    "float16": "float16",
    "fp16": "float16",
    "half": "float16",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "float32": "float32",
    "fp32": "float32",
    "float": "float32",
}


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def resolve_device(request: Any = "auto") -> Any:
    """Return *request* unchanged unless it is ``"auto"`` / ``None`` (cuda → mps → cpu)."""
    if request not in (None, "auto"):
        return request
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(spec: Any, *, strict: bool = True) -> Any:
    """Map ``"bf16"`` / ``"float16"`` / a ``torch.dtype`` / ``None`` to a torch dtype.

    Unknown strings raise ``ValueError`` when *strict*, else resolve to ``None``
    (the model's default dtype).
    """
    if spec is None:
        return None
    import torch

    if isinstance(spec, torch.dtype):
        return spec
    name = _DTYPE_ALIASES.get(str(spec).lower().removeprefix("torch."))
    if name is None:
        if strict:
            raise ValueError(f"unknown dtype {spec!r}; use one of {sorted(_DTYPE_ALIASES)}")
        return None
    return getattr(torch, name)


def infer_device(model: Any) -> Any:
    """Best-effort device of a loaded model's parameters (cpu for parameterless models)."""
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError, TypeError):
        import torch

        return torch.device("cpu")


def load_hf_model(
    name: str,
    *,
    device: Any = "auto",
    dtype: Any = None,
    attn_implementation: str | None = "eager",
    local_files_only: bool = False,
    **model_kwargs: Any,
) -> tuple[Any, Any, Any]:
    """Load ``(model, tokenizer, device)`` for an HF hub id or a local weights path.

    * Only model-construction kwargs reach ``from_pretrained`` — never the
      ``output_*`` forward flags (transformers 5 stores those in the generation
      config and warns that they "are not valid and may be ignored").
    * ``attn_implementation="eager"`` makes attention weights observable (SDPA
      / flash attention return none).  If a model rejects the requested
      implementation the load is retried with the model's default.
    * A tokenizer without a pad token gets ``pad_token = eos_token``.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = resolve_device(device)
    hub_kw: dict[str, Any] = {"local_files_only": True} if local_files_only else {}
    tokenizer = AutoTokenizer.from_pretrained(name, **hub_kw)
    if getattr(tokenizer, "pad_token_id", None) is None and (
        getattr(tokenizer, "eos_token_id", None) is not None
    ):
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict[str, Any] = dict(model_kwargs)
    kwargs.update(hub_kw)
    torch_dtype = resolve_dtype(dtype, strict=False)
    if torch_dtype is not None:
        kwargs["dtype"] = torch_dtype
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    try:
        model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    except (ValueError, TypeError):
        if "attn_implementation" not in kwargs:
            raise
        kwargs.pop("attn_implementation")
        model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    model.to(dev)
    model.eval()
    return model, tokenizer, dev


# ---------------------------------------------------------------------------
# Low-level hook registration (shared with features.intervention)
# ---------------------------------------------------------------------------


def _is_tensor(x: Any) -> bool:
    try:
        import torch
    except ImportError:  # pragma: no cover — only reachable with torch absent
        return False
    return isinstance(x, torch.Tensor)


def register_site_hook(module: Any, when: Literal["pre", "post"], fn: HookFn) -> Any:
    """Register ``fn(hidden) -> new | None`` on *module*'s input (``"pre"``) or output.

    ``"pre"`` edits the module's first positional argument (or its
    ``hidden_states`` keyword); ``"post"`` edits the module's output, or the
    first element when the output is a tuple (attention modules, older HF
    blocks).  Returns the torch hook handle (call ``.remove()``).
    """
    if when == "pre":

        def _pre(_mod: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
            if args:
                hidden = args[0]
            elif "hidden_states" in kwargs:
                hidden = kwargs["hidden_states"]
            else:
                return None
            if not _is_tensor(hidden):
                return None
            new = fn(hidden)
            if new is None or new is hidden:
                return None
            if args:
                return (new, *args[1:]), kwargs
            return args, {**kwargs, "hidden_states": new}

        return module.register_forward_pre_hook(_pre, with_kwargs=True)

    if when == "post":

        def _post(_mod: Any, _args: Any, output: Any) -> Any:
            if isinstance(output, tuple):
                if not output or not _is_tensor(output[0]):
                    return None
                new = fn(output[0])
                if new is None or new is output[0]:
                    return None
                return (new, *output[1:])
            if _is_tensor(output):
                new = fn(output)
                return None if (new is None or new is output) else new
            raise TypeError(
                f"cannot hook output of type {type(output).__name__} on {type(_mod).__name__}"
            )

        return module.register_forward_hook(_post)

    raise ValueError(f"when must be 'pre' or 'post', got {when!r}")


# ---------------------------------------------------------------------------
# Public value objects
# ---------------------------------------------------------------------------


@dataclass
class ResidHook:
    """A hook on one site of one layer.

    Attributes
    ----------
    layer:
        Block index; negative values count from the end (``-1`` = last block).
    fn:
        ``fn(hidden (B, T, D)) -> new hidden | None`` (``None`` = observe only).
    site:
        One of :data:`SITES`; default ``"resid_post"``.
    positions:
        ``None`` (every position), an int, or a sequence of ints.  See the
        module docstring for absolute / negative position semantics under a
        KV cache.  When set, ``fn`` receives only the selected rows.
    """

    layer: int
    fn: HookFn
    site: HookSite = "resid_post"
    positions: int | Sequence[int] | None = None

    def __post_init__(self) -> None:
        if self.site not in SITES:
            raise ValueError(f"unknown hook site {self.site!r}; expected one of {SITES}")
        if not callable(self.fn):
            raise TypeError("ResidHook.fn must be callable")
        if isinstance(self.positions, (int, np.integer)):
            self.positions = (int(self.positions),)
        elif self.positions is not None:
            self.positions = tuple(int(p) for p in self.positions)


@dataclass
class TokenizedPrompt:
    """Result of :meth:`HookedModel.tokenize`.

    ``input_ids`` is a ``(1, T)`` long tensor on the model's device;
    ``tokens[i] == tokenizer.decode([token_ids[i]])``.
    """

    input_ids: Any
    token_ids: list[int]
    tokens: list[str]
    text: str = ""
    chat_applied: bool = False


@dataclass
class ForwardResult:
    """Everything captured by one :meth:`HookedModel.forward` call (batch size 1).

    Attributes
    ----------
    logits:
        ``(1, T, V)`` model logits (after any hooks).
    resid_pre, resid_post:
        ``(L, T, D)`` stacked residual stream (see module docstring), or
        ``None`` when ``capture=False``.  Detached unless ``grad=True``, in
        which case they are differentiable functions of the model (you can
        backprop *through* them to embeddings / parameters).
    resid_pre_live, resid_post_live:
        The per-layer ``(1, T, D)`` tensors exactly as they flowed through the
        model.  With ``grad=True`` they are on the autograd graph *upstream*
        of the logits and have ``retain_grad()`` set, so after
        ``loss.backward()`` their ``.grad`` holds d loss / d residual (see
        :meth:`grad`).  ``resid_pre_live[0]`` is made a leaf requiring grad
        when the embeddings would not otherwise require it (frozen params).
    attentions:
        ``(L, H, T, T)`` attention weights, or ``None`` (``capture_attentions``
        off, or a non-eager attention backend that returns no weights).
    """

    logits: Any
    tokens: list[str]
    token_ids: list[int]
    input_ids: Any
    resid_pre: Any | None = None
    resid_post: Any | None = None
    attentions: Any | None = None
    resid_pre_live: list[Any] = field(default_factory=list, repr=False)
    resid_post_live: list[Any] = field(default_factory=list, repr=False)
    grad_enabled: bool = False

    @property
    def n_layers(self) -> int:
        return len(self.resid_post_live)

    @property
    def embeddings(self) -> Any | None:
        """``resid_pre[0]`` — ``(T, D)`` embedding output entering block 0."""
        return None if self.resid_pre is None else self.resid_pre[0]

    def resid(self, site: Literal["resid_pre", "resid_post"] = "resid_post") -> Any:
        """Stacked ``(L, T, D)`` residual for *site*."""
        if site not in ("resid_pre", "resid_post"):
            raise ValueError(f"site must be 'resid_pre' or 'resid_post', got {site!r}")
        out = self.resid_pre if site == "resid_pre" else self.resid_post
        if out is None:
            raise RuntimeError("residuals were not captured (forward(..., capture=False))")
        return out

    def grad(self, site: Literal["resid_pre", "resid_post"] = "resid_post") -> Any:
        """Stacked ``(L, T, D)`` gradients of the residual after ``backward()``.

        Layers that received no gradient are zero-filled.  Requires
        ``forward(..., grad=True)`` and a prior ``loss.backward()``.  Like any
        ``retain_grad`` tensor, ``.grad`` accumulates across ``backward()`` /
        ``torch.autograd.grad`` calls — use a fresh forward per objective.
        """
        import torch

        if site not in ("resid_pre", "resid_post"):
            raise ValueError(f"site must be 'resid_pre' or 'resid_post', got {site!r}")
        if not self.grad_enabled:
            raise RuntimeError("forward(..., grad=True) is required for gradients")
        live = self.resid_pre_live if site == "resid_pre" else self.resid_post_live
        rows = [
            (t.grad[0] if t.grad is not None else torch.zeros_like(t[0])).detach() for t in live
        ]
        return torch.stack(rows, dim=0)


@dataclass
class GenerateResult:
    """Result of :meth:`HookedModel.generate`.

    ``token_ids`` / ``tokens`` / ``logprobs`` cover the *generated* tokens
    only; ``logprobs[i]`` is ``log p(token_ids[i])`` under the model's own
    (temperature-1, post-hook) next-token distribution at that step.
    ``stop_reason`` is ``"eos"`` or ``"max_new_tokens"``.
    """

    text: str
    token_ids: list[int]
    tokens: list[str]
    logprobs: list[float] | None
    prompt_token_ids: list[int]
    prompt_tokens: list[str]
    stop_reason: str = "max_new_tokens"

    @property
    def full_token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.token_ids


# ---------------------------------------------------------------------------
# HookedModel
# ---------------------------------------------------------------------------


class HookedModel:
    """An HF causal LM + tokenizer with a single residual-stream / hook convention.

    Parameters
    ----------
    model:
        A loaded HF ``*ForCausalLM`` (any supported family, or a generic model
        whose blocks can be located).  It is used as-is: not moved, not
        re-configured.
    tokenizer:
        The matching tokenizer (anything callable as
        ``tokenizer(text, return_tensors="pt")`` with ``decode``).
    device:
        Where inputs are placed; defaults to the device of the model's
        parameters.

    Raises
    ------
    UnsupportedArchitectureError
        If the model's transformer blocks cannot be located.
    """

    def __init__(self, model: Any, tokenizer: Any, device: Any = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device if device is not None else infer_device(model)
        self.arch: Architecture = resolve_architecture(model)
        self._offset = 0  # absolute position of hidden[:, 0] in the running forward
        self._offset_hint = 0  # HookedModel's own bookkeeping (fallback for _offset)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        name: str,
        device: Any = "auto",
        dtype: Any = None,
        attn_implementation: str | None = "eager",
        local_files_only: bool = False,
        **kw: Any,
    ) -> HookedModel:
        """Load *name* (hub id or local path) via :func:`load_hf_model` and wrap it."""
        model, tokenizer, dev = load_hf_model(
            name,
            device=device,
            dtype=dtype,
            attn_implementation=attn_implementation,
            local_files_only=local_files_only,
            **kw,
        )
        return cls(model, tokenizer, device=dev)

    # ------------------------------------------------------------------
    # Architecture info
    # ------------------------------------------------------------------

    @property
    def family(self) -> str:
        """Canonical family name (``"gpt2"``, ``"llama"``, ..., or ``"generic"``)."""
        return self.arch.family

    @property
    def blocks(self) -> list[Any]:
        return self.arch.blocks

    @property
    def final_norm(self) -> Any | None:
        return self.arch.final_norm

    @property
    def unembed(self) -> Any | None:
        return self.arch.unembed

    @property
    def final_logit_softcap(self) -> float | None:
        """Gemma-2 style ``final_logit_softcapping`` (``None`` when the family has none)."""
        return self.arch.final_logit_softcap

    @property
    def n_layers(self) -> int:
        return self.arch.n_layers

    @property
    def d_model(self) -> int:
        return self.arch.d_model

    @property
    def n_heads(self) -> int:
        return self.arch.n_heads

    @property
    def vocab_size(self) -> int:
        return self.arch.vocab_size

    @property
    def config(self) -> Any:
        return getattr(self.model, "config", None)

    @property
    def attn_implementation(self) -> str | None:
        """The attention backend the model was built with (``"eager"``, ``"sdpa"``, ...)."""
        impl = getattr(self.config, "_attn_implementation", None)
        return None if impl is None else str(impl)

    @property
    def returns_attention_weights(self) -> bool:
        """Whether a forward can return real attention weights (eager attention)."""
        impl = self.attn_implementation
        return impl is None or "eager" in impl

    @property
    def eos_token_ids(self) -> set[int]:
        """EOS ids from the tokenizer and the model's generation config."""
        ids: set[int] = set()
        sources = (
            getattr(self.tokenizer, "eos_token_id", None),
            getattr(getattr(self.model, "generation_config", None), "eos_token_id", None),
        )
        for src in sources:
            if isinstance(src, int):
                ids.add(src)
            elif isinstance(src, (list, tuple, set)):
                ids.update(int(i) for i in src if isinstance(i, int))
        return ids

    @property
    def current_offset(self) -> int:
        """Absolute position of ``hidden[:, 0]`` in the forward currently running.

        Only meaningful inside a hook function while position-targeted hooks
        are installed (it is what ``ResidHook.positions`` is resolved against).
        """
        return self._offset

    def __repr__(self) -> str:
        return (
            f"HookedModel(family={self.family!r}, n_layers={self.n_layers}, "
            f"d_model={self.d_model}, n_heads={self.n_heads}, vocab_size={self.vocab_size}, "
            f"device={str(self.device)!r})"
        )

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    def token_str(self, token_id: int) -> str:
        """Surface form of one token id (``tokenizer.decode([id])``)."""
        return str(self.tokenizer.decode([int(token_id)]))

    def tokenize(self, prompt: str | list[dict[str, str]], chat: bool = False) -> TokenizedPrompt:
        """Tokenise *prompt* into a ``(1, T)`` tensor on :attr:`device`.

        With ``chat=True`` and a tokenizer that has a ``chat_template``, the
        prompt (a user string, or a list of ``{"role", "content"}`` messages)
        is rendered with ``apply_chat_template(..., add_generation_prompt=True)``
        and tokenised without adding special tokens again.  Without a template
        a plain string is tokenised as-is (``chat_applied=False``).
        """
        import torch

        template = getattr(self.tokenizer, "chat_template", None) if chat else None
        if template:
            messages = (
                [{"role": "user", "content": prompt}] if isinstance(prompt, str) else list(prompt)
            )
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
            chat_applied = True
        else:
            if not isinstance(prompt, str):
                raise ValueError("a list of chat messages needs a tokenizer with a chat_template")
            text = prompt
            enc = self.tokenizer(prompt, return_tensors="pt")
            chat_applied = False
        ids = enc["input_ids"]
        if not isinstance(ids, torch.Tensor):
            ids = torch.as_tensor(ids, dtype=torch.long)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        ids = ids.to(self.device)
        token_ids = [int(t) for t in ids[0].tolist()]
        return TokenizedPrompt(
            input_ids=ids,
            token_ids=token_ids,
            tokens=[self.token_str(t) for t in token_ids],
            text=str(text),
            chat_applied=chat_applied,
        )

    def _as_prompt(self, prompt_or_ids: Any, chat: bool = False) -> TokenizedPrompt:
        """Normalise a string / messages / ids / TokenizedPrompt to a single-row prompt."""
        import torch

        if isinstance(prompt_or_ids, TokenizedPrompt):
            return prompt_or_ids
        if isinstance(prompt_or_ids, str) or (
            isinstance(prompt_or_ids, list) and prompt_or_ids and isinstance(prompt_or_ids[0], dict)
        ):
            return self.tokenize(prompt_or_ids, chat=chat)
        ids = (
            prompt_or_ids
            if isinstance(prompt_or_ids, torch.Tensor)
            else torch.as_tensor(list(prompt_or_ids), dtype=torch.long)
        )
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        if ids.dim() != 2 or ids.shape[0] != 1:
            raise ValueError(f"expected a single sequence of ids (1, T), got {tuple(ids.shape)}")
        if ids.shape[1] == 0:
            raise ValueError("cannot run the model on an empty sequence")
        ids = ids.to(self.device)
        token_ids = [int(t) for t in ids[0].tolist()]
        return TokenizedPrompt(
            input_ids=ids, token_ids=token_ids, tokens=[self.token_str(t) for t in token_ids]
        )

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _layer_index(self, layer: int) -> int:
        n = self.n_layers
        idx = int(layer) + n if int(layer) < 0 else int(layer)
        if not 0 <= idx < n:
            raise IndexError(f"layer {layer} out of range for a {n}-layer model")
        return idx

    def site_module(self, layer: int, site: str) -> tuple[Any, Literal["pre", "post"]]:
        """``(module, "pre" | "post")`` that implements *site* at *layer*."""
        li = self._layer_index(layer)
        if site == "resid_pre":
            return self.blocks[li], "pre"
        if site == "resid_post":
            return self.blocks[li], "post"
        if site == "mlp_in":
            mod = self.arch.mlp_in_module(li)
        elif site == "mlp_out":
            mod = self.arch.mlp_out_module(li)
        elif site == "attn_out":
            mod = self.arch.attn_module(li)
        else:
            raise ValueError(f"unknown hook site {site!r}; expected one of {SITES}")
        if mod is None:
            raise ValueError(f"site {site!r} is not available for family {self.family!r}")
        return mod, ("pre" if site == "mlp_in" else "post")

    def _wrap(self, spec: ResidHook) -> HookFn:
        """Adapt a user hook to (B, T, D) tensors and its position restriction."""
        import torch

        fn = spec.fn
        pos = spec.positions
        positions: tuple[int, ...] | None = (
            None if pos is None else (int(pos),) if isinstance(pos, int) else tuple(pos)
        )

        def run(hidden: Any) -> Any:
            h = hidden.unsqueeze(0) if hidden.dim() == 2 else hidden
            if positions is None:
                new = fn(h)
            else:
                off, t_len = self._offset, h.shape[1]
                absolute = (p if p >= 0 else off + t_len + p for p in positions)
                local = sorted({q - off for q in absolute if off <= q < off + t_len})
                if not local:
                    return None
                idx = torch.tensor(local, dtype=torch.long, device=h.device)
                new_rows = fn(h.index_select(1, idx))
                if new_rows is None:
                    return None
                new = h.index_copy(1, idx, new_rows.to(h.dtype))
            if new is None:
                return None
            return new.reshape(hidden.shape) if hidden.dim() == 2 else new

        return run

    def _position_tracker(self) -> Any:
        """Pre-hook on block 0 that records the chunk's absolute offset."""
        import torch

        def _track(_mod: Any, _args: Any, kwargs: dict[str, Any]) -> None:
            pos = kwargs.get("position_ids")
            if not isinstance(pos, torch.Tensor):
                pos = kwargs.get("cache_position")
            if isinstance(pos, torch.Tensor) and pos.numel() > 0:
                self._offset = int(pos.reshape(-1)[0])
            else:
                self._offset = self._offset_hint
            return None

        return self.blocks[0].register_forward_pre_hook(_track, with_kwargs=True, prepend=True)

    @contextlib.contextmanager
    def hooks(self, specs: Iterable[ResidHook] = ()) -> Iterator[list[Any]]:
        """Install *specs* for the duration of the ``with`` block; always removed on exit.

        Yields the list of torch hook handles.  Usable around direct
        ``hm.model(...)`` / ``hm.model.generate(...)`` calls as well as around
        :meth:`forward` / :meth:`generate`.
        """
        specs = list(specs)
        handles: list[Any] = []
        try:
            if any(s.positions is not None for s in specs):
                handles.append(self._position_tracker())
            for spec in specs:
                module, when = self.site_module(spec.layer, spec.site)
                handles.append(register_site_hook(module, when, self._wrap(spec)))
            yield handles
        finally:
            for h in reversed(handles):
                h.remove()
            handles.clear()

    @contextlib.contextmanager
    def _capture(self, pre: list[Any], post: list[Any], grad: bool) -> Iterator[None]:
        handles: list[Any] = []

        def keep(store: list[Any], layer: int) -> HookFn:
            def _fn(hidden: Any) -> Any:
                if grad and layer == 0 and store is pre and not hidden.requires_grad:
                    hidden = hidden.detach().requires_grad_(True)
                    store[layer] = hidden
                    return hidden
                if grad and hidden.requires_grad:
                    hidden.retain_grad()
                store[layer] = hidden
                return None

            return _fn

        try:
            for li, block in enumerate(self.blocks):
                handles.append(register_site_hook(block, "pre", keep(pre, li)))
                handles.append(register_site_hook(block, "post", keep(post, li)))
            yield
        finally:
            for h in reversed(handles):
                h.remove()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        prompt_or_ids: Any,
        *,
        hooks: Iterable[ResidHook] = (),
        capture: bool = True,
        grad: bool = False,
        capture_attentions: bool = True,
        chat: bool = False,
        **model_kwargs: Any,
    ) -> ForwardResult:
        """Run one full forward pass (no KV cache) and capture the residual stream.

        Parameters
        ----------
        prompt_or_ids:
            A prompt string (or chat messages with ``chat=True``), a list of
            token ids, a ``(T,)`` / ``(1, T)`` id tensor, or a
            :class:`TokenizedPrompt`.
        hooks:
            :class:`ResidHook` specs active during this pass only.
        capture:
            Capture ``resid_pre`` / ``resid_post`` for every layer.
        grad:
            Run with autograd enabled; captured tensors stay on the graph
            (see :class:`ForwardResult`).  Otherwise runs under ``no_grad``.
        capture_attentions:
            Request attention weights (only honoured by eager attention).
        model_kwargs:
            Forwarded verbatim to the model call.
        """
        import torch

        tp = self._as_prompt(prompt_or_ids, chat=chat)
        n = self.n_layers
        pre: list[Any] = [None] * n
        post: list[Any] = [None] * n
        want_attn = bool(capture_attentions) and self.returns_attention_weights
        call_kwargs: dict[str, Any] = {"use_cache": False, "output_attentions": want_attn}
        call_kwargs.update(model_kwargs)

        with contextlib.ExitStack() as stack:
            stack.enter_context(torch.enable_grad() if grad else torch.no_grad())
            stack.enter_context(self.hooks(hooks))
            if capture:
                stack.enter_context(self._capture(pre, post, grad))
            self._offset = self._offset_hint = 0
            out = self.model(input_ids=tp.input_ids, **call_kwargs)

        resid_pre = resid_post = None
        if capture:
            if any(t is None for t in pre) or any(t is None for t in post):
                raise RuntimeError("some transformer blocks did not run; cannot capture residuals")
            resid_pre = torch.stack([t[0] for t in pre], dim=0)
            resid_post = torch.stack([t[0] for t in post], dim=0)
            if not grad:
                resid_pre, resid_post = resid_pre.detach(), resid_post.detach()

        attentions = None
        raw_attn = getattr(out, "attentions", None) if want_attn else None
        if raw_attn and all(a is not None for a in raw_attn):
            attentions = torch.stack([a[0] for a in raw_attn], dim=0)
            if not grad:
                attentions = attentions.detach()

        return ForwardResult(
            logits=out.logits,
            tokens=list(tp.tokens),
            token_ids=list(tp.token_ids),
            input_ids=tp.input_ids,
            resid_pre=resid_pre,
            resid_post=resid_post,
            attentions=attentions,
            resid_pre_live=pre if capture else [],
            resid_post_live=post if capture else [],
            grad_enabled=bool(grad),
        )

    __call__ = forward

    # ------------------------------------------------------------------
    # Logit lens
    # ------------------------------------------------------------------

    def logit_lens(self, resid: Any) -> Any:
        """Project residual vector(s) ``(..., D)`` to logits ``(..., V)``.

        Applies final norm → post-norm projection (OPT ``project_out``) →
        unembedding → logit soft-capping (Gemma-2), each exactly once, under
        the caller's grad mode.  ``logit_lens(resid_post[-1])`` reproduces the
        model's own logits.  NumPy inputs are converted to the unembedding's
        device / dtype.
        """
        import torch

        if self.unembed is None:
            raise RuntimeError(f"{type(self.model).__name__} exposes no output embedding")
        h = resid if isinstance(resid, torch.Tensor) else torch.as_tensor(np.asarray(resid))
        ref = next(iter(self.unembed.parameters()), None)
        if ref is not None and (h.dtype != ref.dtype or h.device != ref.device):
            h = h.to(device=ref.device, dtype=ref.dtype)
        if self.final_norm is not None:
            h = self.final_norm(h)
        for mod in self.arch.post_norm:
            h = mod(h)
        logits = self.unembed(h)
        cap = self.final_logit_softcap
        if cap:
            logits = torch.tanh(logits / cap) * cap
        return logits

    def logit_lens_error(self, prompt_or_ids: Any) -> float:
        """Max |logit_lens(resid_post[-1]) - model logits| on *prompt_or_ids*.

        ~0 (up to dtype precision) for every supported family; a large value
        means the generic fallback missed a logit quirk and the lens is not
        faithful for this model.
        """
        import torch

        res = self.forward(prompt_or_ids, capture_attentions=False)
        assert res.resid_post is not None
        with torch.no_grad():
            lens = self.logit_lens(res.resid_post[-1]).to(torch.float32)
        return float((lens - res.logits[0].to(torch.float32)).abs().max())

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: Any,
        *,
        max_new_tokens: int = 20,
        hooks: Iterable[ResidHook] = (),
        temperature: float = 0.0,
        stop_at_eos: bool = True,
        chat: bool = False,
        return_logprobs: bool = True,
        use_cache: bool = True,
        top_k: int | None = None,
        seed: int | None = None,
        stop_token_ids: Iterable[int] | None = None,
    ) -> GenerateResult:
        """Token-by-token generation with hooks active at every step.

        Greedy (deterministic, ties → lowest id) when ``temperature <= 0``;
        otherwise samples from ``softmax(logits / temperature)`` (optionally
        top-k filtered) with a CPU ``torch.Generator`` seeded by *seed*.
        Stops after *max_new_tokens* or when a stop token is produced (the
        stop token is included in the output).  Stop tokens are
        *stop_token_ids* if given, else :attr:`eos_token_ids` when
        *stop_at_eos*.  ``use_cache=False`` recomputes the full sequence each
        step (see the module docstring for hook-position semantics).
        """
        import torch

        tp = self._as_prompt(prompt, chat=chat)
        if stop_token_ids is not None:
            stops = {int(i) for i in stop_token_ids}
        else:
            stops = self.eos_token_ids if stop_at_eos else set()
        gen = None
        if temperature > 0 and seed is not None:
            gen = torch.Generator(device="cpu").manual_seed(int(seed))

        ids = tp.input_ids
        new_ids: list[int] = []
        logprobs: list[float] = []
        stop_reason = "max_new_tokens"
        past: Any = None
        try:
            with torch.no_grad(), self.hooks(hooks):
                for _ in range(int(max_new_tokens)):
                    if use_cache:
                        feed = ids if past is None else ids[:, -1:]
                        self._offset = self._offset_hint = int(ids.shape[1] - feed.shape[1])
                        out = self.model(input_ids=feed, past_key_values=past, use_cache=True)
                        past = out.past_key_values
                    else:
                        self._offset = self._offset_hint = 0
                        out = self.model(input_ids=ids, use_cache=False)
                    logits = out.logits[0, -1].to(torch.float32)
                    next_id = self._choose(logits, temperature, top_k, gen)
                    if return_logprobs:
                        logprobs.append(float(torch.log_softmax(logits, dim=-1)[next_id]))
                    new_ids.append(next_id)
                    nxt = torch.tensor([[next_id]], dtype=ids.dtype, device=ids.device)
                    ids = torch.cat([ids, nxt], dim=1)
                    if next_id in stops:
                        stop_reason = "eos"
                        break
        finally:
            self._offset = self._offset_hint = 0

        return GenerateResult(
            text=str(self.tokenizer.decode(new_ids)) if new_ids else "",
            token_ids=new_ids,
            tokens=[self.token_str(t) for t in new_ids],
            logprobs=logprobs if return_logprobs else None,
            prompt_token_ids=list(tp.token_ids),
            prompt_tokens=list(tp.tokens),
            stop_reason=stop_reason,
        )

    @staticmethod
    def _choose(logits: torch.Tensor, temperature: float, top_k: int | None, gen: Any) -> int:
        import torch

        if temperature <= 0:
            return int(torch.argmax(logits))
        scaled = logits / float(temperature)
        if top_k is not None and 0 < top_k < scaled.shape[-1]:
            kth = torch.topk(scaled, int(top_k)).values[-1]
            scaled = scaled.masked_fill(scaled < kth, float("-inf"))
        probs = torch.softmax(scaled, dim=-1).cpu()
        return int(torch.multinomial(probs, 1, generator=gen)[0])
