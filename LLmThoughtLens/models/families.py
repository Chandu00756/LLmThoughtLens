"""Architecture families — where each HF causal LM keeps its blocks, final norm and unembedding.

Every white-box feature in LLmThoughtLens (residual capture, logit lens, hooks,
interventions, gradient attribution) needs the same four facts about a model:

* the ordered list of transformer **blocks** (the residual stream flows
  through them one after another),
* the **final norm** applied to the residual stream before unembedding,
* any **post-norm projection** between that norm and the unembedding
  (OPT's ``project_out`` when ``word_embed_proj_dim != hidden_size``),
* the **unembedding** (``model.get_output_embeddings()``) and any logit
  post-processing (Gemma-2's ``final_logit_softcapping``).

This module is the single place that knows those facts.  Resolution is
*family-first*: the HF ``config.model_type`` selects an explicit
:class:`FamilySpec` (GPT-2, GPT-NeoX, Llama, Mistral, Qwen2, Qwen3, Gemma,
Gemma-2, Gemma-3 text, Phi-3, OPT).  Unknown models fall back to the
historical attribute-path probes and finally to scanning ``named_modules`` for
a ``ModuleList`` whose children look like transformer blocks (family
``"generic"``).  The generic fallback cannot know a model's logit quirks, so
callers that need a *faithful* logit lens should check
:meth:`HookedModel.logit_lens_error` on a real prompt.

Nothing here imports torch at module import time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "FAMILIES",
    "Architecture",
    "FamilySpec",
    "UnsupportedArchitectureError",
    "family_for_model_type",
    "resolve_architecture",
    "resolve_family",
    "resolve_final_norm",
    "resolve_transformer_blocks",
    "supported_families",
]

Path = tuple[str, ...]


class UnsupportedArchitectureError(ValueError):
    """Raised when no transformer-block list can be located in a model."""


@dataclass(frozen=True)
class FamilySpec:
    """Static description of where a model family keeps its components.

    Attributes
    ----------
    name:
        Canonical family name reported by :attr:`HookedModel.family`.
    model_types:
        HF ``config.model_type`` values that map to this family.
    block_paths:
        Candidate attribute paths (from the ``*ForCausalLM`` object) to the
        ``ModuleList`` of blocks; the first that exists wins.
    final_norm_paths:
        Candidate attribute paths to the final norm module.  A path that
        exists but holds ``None`` (e.g. OPT-350m, ``do_layer_norm_before=False``)
        resolves to "no final norm".
    post_norm_paths:
        Modules applied, in order, between the final norm and the
        unembedding (OPT ``project_out``).  Missing / ``None`` entries are
        skipped.
    attn_attrs, mlp_in_attrs, mlp_out_attrs:
        Attribute names on a block for the attention module, the module whose
        *input* is the MLP input, and the module whose *output* is the MLP
        output.  For OPT the MLP is not a submodule, so ``fc1`` / ``fc2`` are
        used instead.
    """

    name: str
    model_types: tuple[str, ...]
    block_paths: tuple[Path, ...]
    final_norm_paths: tuple[Path, ...]
    post_norm_paths: tuple[Path, ...] = ()
    attn_attrs: tuple[str, ...] = ("self_attn",)
    mlp_in_attrs: tuple[str, ...] = ("mlp",)
    mlp_out_attrs: tuple[str, ...] = ("mlp",)


def _decoder_only(name: str) -> FamilySpec:
    """Llama-style layout: ``model.layers`` blocks, ``model.norm`` final norm."""
    return FamilySpec(
        name=name,
        model_types=(name,),
        block_paths=(("model", "layers"),),
        final_norm_paths=(("model", "norm"),),
    )


#: Every explicitly supported family.  Order is irrelevant (lookup is by model_type).
FAMILIES: tuple[FamilySpec, ...] = (
    FamilySpec(
        name="gpt2",
        model_types=("gpt2",),
        block_paths=(("transformer", "h"),),
        final_norm_paths=(("transformer", "ln_f"),),
        attn_attrs=("attn",),
    ),
    FamilySpec(
        name="gpt_neox",
        model_types=("gpt_neox",),
        block_paths=(("gpt_neox", "layers"),),
        final_norm_paths=(("gpt_neox", "final_layer_norm"),),
        attn_attrs=("attention",),
    ),
    _decoder_only("llama"),
    _decoder_only("mistral"),
    _decoder_only("qwen2"),
    _decoder_only("qwen3"),
    _decoder_only("gemma"),
    _decoder_only("gemma2"),
    FamilySpec(
        name="gemma3",
        model_types=("gemma3_text", "gemma3"),
        # Gemma3ForCausalLM keeps the text stack at ``model``; the multimodal
        # wrapper nests it under ``language_model``.
        block_paths=(
            ("model", "layers"),
            ("model", "language_model", "layers"),
            ("language_model", "model", "layers"),
        ),
        final_norm_paths=(
            ("model", "norm"),
            ("model", "language_model", "norm"),
            ("language_model", "model", "norm"),
        ),
    ),
    _decoder_only("phi3"),
    FamilySpec(
        name="opt",
        model_types=("opt",),
        block_paths=(("model", "decoder", "layers"),),
        final_norm_paths=(("model", "decoder", "final_layer_norm"),),
        post_norm_paths=(("model", "decoder", "project_out"),),
        mlp_in_attrs=("fc1",),
        mlp_out_attrs=("fc2",),
    ),
)

_BY_MODEL_TYPE: dict[str, FamilySpec] = {mt: f for f in FAMILIES for mt in f.model_types}

# Historical probe order — kept verbatim so models without a recognised
# ``config.model_type`` resolve exactly as before the consolidation.
_GENERIC_BLOCK_PATHS: tuple[Path, ...] = (
    ("transformer", "h"),
    ("model", "layers"),
    ("gpt_neox", "layers"),
    ("transformer", "blocks"),
    ("model", "decoder", "layers"),
)
_GENERIC_FINAL_NORM_PATHS: tuple[Path, ...] = (
    ("transformer", "ln_f"),  # GPT-2
    ("model", "norm"),  # Llama / Mistral / Qwen / Gemma / Phi-3
    ("gpt_neox", "final_layer_norm"),  # GPT-NeoX
    ("model", "decoder", "final_layer_norm"),  # OPT
    ("transformer", "norm_f"),  # some others
    ("model", "final_layernorm"),  # Phi-1/2
)
_GENERIC_ATTN_ATTRS = ("self_attn", "attn", "attention")
_GENERIC_MLP_ATTRS = ("mlp", "feed_forward")

_MISSING = object()


def supported_families() -> list[str]:
    """Names of the explicitly supported families (excludes ``"generic"``)."""
    return [f.name for f in FAMILIES]


def family_for_model_type(model_type: str | None) -> FamilySpec | None:
    """Return the :class:`FamilySpec` for an HF ``config.model_type`` (or ``None``)."""
    if not model_type:
        return None
    return _BY_MODEL_TYPE.get(str(model_type))


def resolve_family(model: Any) -> FamilySpec | None:
    """Family of *model* from its ``config.model_type``; ``None`` when unknown."""
    cfg = getattr(model, "config", None)
    return family_for_model_type(getattr(cfg, "model_type", None))


def _walk(obj: Any, path: Path) -> Any:
    """Follow *path* attribute by attribute; return ``_MISSING`` if any hop is absent."""
    for attr in path:
        try:
            obj = getattr(obj, attr)
        except AttributeError:
            return _MISSING
    return obj


def _first_existing(model: Any, paths: tuple[Path, ...]) -> Any:
    for path in paths:
        obj = _walk(model, path)
        if obj is not _MISSING:
            return obj
    return _MISSING


def _scan_for_blocks(model: Any) -> list[Any]:
    """Last resort: first ``ModuleList`` whose head has attention + MLP children."""
    named_modules = getattr(model, "named_modules", None)
    if named_modules is None:
        return []
    try:
        import torch.nn as nn
    except ImportError:  # pragma: no cover — torch-less callers never reach a real module
        return []
    for _name, mod in named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > 0:
            attrs = dir(mod[0])
            if any(a in attrs for a in ("self_attn", "attention", "attn")) and any(
                a in attrs for a in ("mlp", "feed_forward")
            ):
                return list(mod)
    return []


def resolve_transformer_blocks(model: Any) -> list[Any]:
    """Return the ordered list of transformer-block modules (``[]`` when none found).

    Family-first (``config.model_type``), then the historical attribute paths
    (GPT-2, Llama-style, GPT-NeoX, ``transformer.blocks``, OPT), then a scan of
    ``named_modules`` for a block-like ``ModuleList``.
    """
    fam = resolve_family(model)
    if fam is not None:
        obj = _first_existing(model, fam.block_paths)
        if obj is not _MISSING and obj is not None:
            return list(obj)
    obj = _first_existing(model, _GENERIC_BLOCK_PATHS)
    if obj is not _MISSING:
        return list(obj)
    return _scan_for_blocks(model)


def resolve_final_norm(model: Any) -> Any:
    """Return the model's final pre-unembedding norm module, or ``None``.

    Family-first, then the historical probe order (GPT-2 ``ln_f``, Llama-style
    ``model.norm``, GPT-NeoX, OPT, ``transformer.norm_f``, Phi ``final_layernorm``).
    """
    fam = resolve_family(model)
    if fam is not None:
        obj = _first_existing(model, fam.final_norm_paths)
        if obj is not _MISSING:
            return obj
    obj = _first_existing(model, _GENERIC_FINAL_NORM_PATHS)
    return None if obj is _MISSING else obj


def _config_int(cfg: Any, names: tuple[str, ...]) -> int:
    for n in names:
        v = getattr(cfg, n, None)
        if isinstance(v, int) and not isinstance(v, bool) and v > 0:
            return v
    return 0


@dataclass
class Architecture:
    """Resolved, instance-level view of a model's components.

    Built by :func:`resolve_architecture`; consumed by :class:`HookedModel`.
    """

    family: str
    blocks: list[Any]
    final_norm: Any | None
    unembed: Any | None
    post_norm: tuple[Any, ...] = ()
    final_logit_softcap: float | None = None
    n_layers: int = 0
    d_model: int = 0
    n_heads: int = 0
    vocab_size: int = 0
    attn_attrs: tuple[str, ...] = _GENERIC_ATTN_ATTRS
    mlp_in_attrs: tuple[str, ...] = _GENERIC_MLP_ATTRS
    mlp_out_attrs: tuple[str, ...] = _GENERIC_MLP_ATTRS
    spec: FamilySpec | None = field(default=None, repr=False)

    @staticmethod
    def _child(block: Any, attrs: tuple[str, ...]) -> Any | None:
        for a in attrs:
            mod = getattr(block, a, None)
            if mod is not None:
                return mod
        return None

    def attn_module(self, layer: int) -> Any | None:
        """The attention submodule of block *layer* (``None`` if not found)."""
        return self._child(self.blocks[layer], self.attn_attrs)

    def mlp_in_module(self, layer: int) -> Any | None:
        """Module whose forward *input* is block *layer*'s MLP input."""
        return self._child(self.blocks[layer], self.mlp_in_attrs)

    def mlp_out_module(self, layer: int) -> Any | None:
        """Module whose forward *output* is block *layer*'s MLP output."""
        return self._child(self.blocks[layer], self.mlp_out_attrs)


def resolve_architecture(model: Any) -> Architecture:
    """Resolve blocks, final norm, unembedding and logit quirks for *model*.

    Raises
    ------
    UnsupportedArchitectureError
        If no list of transformer blocks can be located.
    """
    blocks = resolve_transformer_blocks(model)
    if not blocks:
        raise UnsupportedArchitectureError(
            f"could not locate transformer blocks in {type(model).__name__}; "
            "supported families: " + ", ".join(supported_families())
        )
    fam = resolve_family(model)
    cfg = getattr(model, "config", None)

    post_norm: list[Any] = []
    if fam is not None:
        for path in fam.post_norm_paths:
            mod = _walk(model, path)
            if mod is not _MISSING and mod is not None:
                post_norm.append(mod)

    unembed = None
    get_out = getattr(model, "get_output_embeddings", None)
    if callable(get_out):
        unembed = get_out()

    softcap = getattr(cfg, "final_logit_softcapping", None)
    softcap_f = float(softcap) if isinstance(softcap, (int, float)) and softcap else None

    d_model = _config_int(cfg, ("hidden_size", "n_embd", "d_model"))
    vocab = int(getattr(unembed, "out_features", 0) or 0) or _config_int(cfg, ("vocab_size",))
    return Architecture(
        family=fam.name if fam is not None else "generic",
        blocks=blocks,
        final_norm=resolve_final_norm(model),
        unembed=unembed,
        post_norm=tuple(post_norm),
        final_logit_softcap=softcap_f,
        n_layers=len(blocks),
        d_model=d_model,
        n_heads=_config_int(cfg, ("num_attention_heads", "n_head", "num_heads")),
        vocab_size=vocab,
        attn_attrs=fam.attn_attrs if fam is not None else _GENERIC_ATTN_ATTRS,
        mlp_in_attrs=fam.mlp_in_attrs if fam is not None else _GENERIC_MLP_ATTRS,
        mlp_out_attrs=fam.mlp_out_attrs if fam is not None else _GENERIC_MLP_ATTRS,
        spec=fam,
    )
