"""Steering vectors — add a direction to the residual stream and measure what changes.

Activation steering edits a white-box model's residual stream during the
forward pass: at one layer / site, ``coeff * direction`` is added to the hidden
state of selected token positions.  The effect of that edit on the model's
next-token distribution and on its completion is *interventional* evidence for
this model, prompt and coefficient — unlike the direction itself, which is
usually derived from correlational statistics (a contrast of activations or an
SAE decoder column).

Everything here runs on :class:`~LLmThoughtLens.models.HookedModel` residual
hooks (``site="resid_pre"`` edits what enters block ``layer``,
``site="resid_post"`` what leaves it; ``resid_post[l]`` is the same tensor as
``resid_pre[l + 1]``).  torch is imported lazily: building, validating, saving
and loading a :class:`SteeringVector` needs only NumPy.

Building a vector
-----------------
* :meth:`SteeringVector.from_contrast` — mean difference of activations between
  positive and negative prompts at one layer (the technique known as
  Contrastive Activation Addition; a single positive/negative pair is the
  ActAdd setting).
* :meth:`SteeringVector.from_sae_feature` — an SAE feature's decoder direction.
* :meth:`SteeringVector.from_vector` — any explicit ``(d_model,)`` direction.

Which positions are steered
---------------------------
``SteeringVector.positions`` is resolved against the *prompt* length ``T`` into
**absolute** token positions (prompt + generated tokens), so steered
generation gives the same tokens with or without a KV cache:

``"all"``        every position: the prompt and every generated token as it is fed.
``"prompt"``     positions ``0 .. T-1`` only.  Under a KV cache these are edited
                 once, during prefill; generated tokens are not edited but
                 attend to the edited prompt.
``"last"``       the final prompt token ``T-1`` only (the position whose output
                 is the first next-token distribution).
``"generated"``  every position that *predicts* a completion token: ``T-1`` and
                 every generated token (``T, T+1, ...``) as it is fed.  This is
                 what "steer while the model writes" means; it equals
                 HookedModel's newest-token (``positions=-1``) hook under a KV
                 cache, but is cache-independent.  ``"prompt"`` and
                 ``"generated"`` share the hinge position ``T-1``.
explicit ints    absolute positions; negative values count back from the end of
                 the *prompt* (``-1`` = ``T-1``), never from the growing
                 sequence, so they are cache-independent too.

Measuring the effect
--------------------
:func:`steer_generate` returns a :class:`SteeringResult` with the baseline and
steered completions (text, tokens, per-step chosen-token log-probs) plus a
**teacher-forced** comparison: both distributions are evaluated along the
*baseline* prefix, so ``kl_per_step[i] = KL(p_steered || p_baseline)`` compares
like with like even after the two completions diverge.  The first-step
promoted / suppressed tokens are ranked by the change in probability.
:func:`coefficient_sweep` tabulates target-token probabilities, first-step KL
and the greedy completion over a list of coefficients.

Steering needs a local model's residual stream; :func:`as_hooked` raises
:class:`SteeringUnavailableError` for the mock provider (synthetic internals)
and for black-box API providers.
"""

from __future__ import annotations

import json
import math
import re
import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Union

import numpy as np

from LLmThoughtLens.models.hooked import HookedModel, ResidHook, TokenizedPrompt

if TYPE_CHECKING:
    import torch

__all__ = [
    "POSITION_MODES",
    "STEERING_SITES",
    "GenerationTrace",
    "PositionSpec",
    "SteeringMismatchError",
    "SteeringResult",
    "SteeringSite",
    "SteeringUnavailableError",
    "SteeringVector",
    "SweepResult",
    "SweepRow",
    "TokenShift",
    "as_hooked",
    "coefficient_sweep",
    "steer_generate",
    "steering_hooks",
]

SteeringSite = Literal["resid_pre", "resid_post"]
#: Residual-stream sites a steering vector can be added at.
STEERING_SITES: tuple[str, ...] = ("resid_pre", "resid_post")
#: Named position modes (see the module docstring).
POSITION_MODES: tuple[str, ...] = ("all", "last", "prompt", "generated")
#: A named mode or explicit absolute token positions.
PositionSpec = Union[str, int, Sequence[int]]

_FORMAT = "LLmThoughtLens.steering_vector"
_FORMAT_VERSION = 1
_NORM_EPS = 1e-12


class SteeringUnavailableError(RuntimeError):
    """The target has no white-box model whose residual stream can be edited."""


class SteeringMismatchError(ValueError):
    """A steering vector does not fit the model (``d_model`` / layer count / layer index)."""


# ---------------------------------------------------------------------------
# Capability check
# ---------------------------------------------------------------------------


def as_hooked(target: Any) -> HookedModel:
    """Return the :class:`HookedModel` behind *target* or raise :class:`SteeringUnavailableError`.

    *target* may be a :class:`HookedModel`, a provider exposing one (the
    HuggingFace provider's ``.hooked``, which loads the model on first use) or
    an object wrapping such a provider in a ``.provider`` attribute (e.g.
    :class:`~LLmThoughtLens.scope.Scope`).  The mock provider is refused even
    though it reports ``white_box``: its activations are synthetic NumPy
    arrays with no forward pass behind them, so there is nothing to steer.
    Black-box API providers expose no residual stream at all.
    """
    if isinstance(target, HookedModel):
        return target
    if not hasattr(target, "evidence_kind"):
        inner = getattr(target, "provider", None)
        if inner is not None and inner is not target:
            return as_hooked(inner)
    name = str(getattr(target, "name", type(target).__name__))
    if getattr(target, "supports_gradients", False) and hasattr(type(target), "hooked"):
        hm = target.hooked
        if isinstance(hm, HookedModel):
            return hm
    evidence = getattr(target, "evidence_kind", None)
    if name == "mock" or type(target).__name__ == "MockProvider":
        why = (
            "the mock provider's activations are synthetic (deterministic NumPy), so there "
            "is no model forward pass to edit"
        )
    elif evidence == "black_box":
        why = (
            f"the {name!r} provider is a black-box API: it exposes no residual stream, so "
            "activations cannot be edited (only the prompt can be changed)"
        )
    else:
        why = f"{type(target).__name__} does not expose a HookedModel"
    raise SteeringUnavailableError(
        "Activation steering needs white-box access to a local model's residual stream "
        f"(a HookedModel, or the 'huggingface' provider); {why}."
    )


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _to_numpy(vec: Any) -> np.ndarray:
    if hasattr(vec, "detach"):  # torch tensor
        vec = vec.detach().to("cpu").float().numpy()
    return np.asarray(vec, dtype=np.float64)


def _norm_positions(positions: Any) -> str | tuple[int, ...]:
    if isinstance(positions, str):
        if positions not in POSITION_MODES:
            raise ValueError(
                f"unknown positions mode {positions!r}; expected one of {POSITION_MODES} "
                "or explicit token indices"
            )
        return positions
    if isinstance(positions, (bool, np.bool_)):
        raise TypeError("positions must be a mode string or integer token indices, not a bool")
    if isinstance(positions, (int, np.integer)):
        return (int(positions),)
    try:
        out = tuple(int(p) for p in positions)
    except TypeError as exc:
        raise TypeError(
            f"positions must be one of {POSITION_MODES} or a sequence of ints, "
            f"got {type(positions).__name__}"
        ) from exc
    if not out:
        raise ValueError("explicit positions must not be empty")
    return out


def _model_name(hm: HookedModel) -> str | None:
    cfg = hm.config
    name = getattr(cfg, "_name_or_path", None) or getattr(cfg, "name_or_path", None)
    return str(name) if name else None


def _same_model(a: str, b: str) -> bool:
    return a == b or a.rstrip("/").rsplit("/", 1)[-1] == b.rstrip("/").rsplit("/", 1)[-1]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    raise TypeError(f"{type(obj).__name__} is not JSON-serialisable")


def _vector_label(v: SteeringVector) -> str:
    return v.name or str(v.source.get("method", "vector"))


# ---------------------------------------------------------------------------
# SteeringVector
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class SteeringVector:
    """A residual-stream direction plus where, where-in-the-sequence and how hard to add it.

    The edit is ``hidden[positions] += coeff * d`` with ``d = direction`` or,
    when ``normalize`` is set, ``d = direction / ||direction||`` (then
    ``coeff`` is in units of residual-stream norm).

    Attributes
    ----------
    direction:
        ``(d_model,)`` float32 direction (stored as given, not normalised).
    layer:
        Block index; negative values count from the end.
    site:
        ``"resid_post"`` (default; output of block ``layer``) or
        ``"resid_pre"`` (input of block ``layer``).
    coeff:
        Default multiplier (callers may override per call).
    normalize:
        Use the unit direction instead of the raw one.
    positions:
        ``"all"`` | ``"last"`` | ``"prompt"`` | ``"generated"`` or explicit
        absolute indices — see the module docstring.
    name:
        Free-form label.
    source:
        How the vector was built (method, prompts, SAE feature, norms, ...).
        Must be JSON-serialisable for :meth:`save`.
    model_name, d_model, n_layers:
        The model the vector was built for; used by :meth:`validate`.
        ``d_model`` defaults to ``len(direction)``.
    """

    direction: np.ndarray
    layer: int
    site: SteeringSite = "resid_post"
    coeff: float = 1.0
    normalize: bool = False
    positions: Any = "all"
    name: str = ""
    source: dict[str, Any] = field(default_factory=dict)
    model_name: str | None = None
    d_model: int | None = None
    n_layers: int | None = None

    def __post_init__(self) -> None:
        vec = _to_numpy(self.direction)
        if vec.ndim != 1 or vec.size == 0:
            raise ValueError(f"direction must be a non-empty 1-D vector, got shape {vec.shape}")
        if not np.all(np.isfinite(vec)):
            raise ValueError("direction contains NaN or infinite values")
        self.direction = vec.astype(np.float32)
        if self.d_model is None:
            self.d_model = int(vec.size)
        elif int(self.d_model) != vec.size:
            raise SteeringMismatchError(
                f"direction has {vec.size} dims but d_model={self.d_model} was given"
            )
        self.d_model = int(self.d_model)
        if self.site not in STEERING_SITES:
            raise ValueError(
                f"unknown steering site {self.site!r}; expected one of {STEERING_SITES}"
            )
        self.layer = int(self.layer)
        if self.n_layers is not None:
            self.n_layers = int(self.n_layers)
            if not -self.n_layers <= self.layer < self.n_layers:
                raise SteeringMismatchError(
                    f"layer {self.layer} out of range for a {self.n_layers}-layer model"
                )
        self.coeff = float(self.coeff)
        if not math.isfinite(self.coeff):
            raise ValueError("coeff must be finite")
        self.normalize = bool(self.normalize)
        if self.normalize and self.norm < _NORM_EPS:
            raise ValueError("cannot normalize a zero direction")
        self.positions = _norm_positions(self.positions)
        self.name = str(self.name)
        self.source = dict(self.source)

    # ------------------------------------------------------------------
    # Vector arithmetic
    # ------------------------------------------------------------------

    @property
    def norm(self) -> float:
        """L2 norm of the raw :attr:`direction`."""
        return float(np.linalg.norm(self.direction.astype(np.float64)))

    @property
    def unit(self) -> np.ndarray:
        """``direction / ||direction||`` (float32; zeros for a zero direction)."""
        n = self.norm
        return self.direction if n < _NORM_EPS else (self.direction / n).astype(np.float32)

    def vector(self, coeff: float | None = None) -> np.ndarray:
        """The ``(d_model,)`` float32 vector actually added: ``coeff * d``."""
        c = self.coeff if coeff is None else float(coeff)
        base = self.unit if self.normalize else self.direction
        return (base.astype(np.float64) * c).astype(np.float32)

    def with_coeff(self, coeff: float) -> SteeringVector:
        """A copy with a different default coefficient."""
        return replace(self, coeff=float(coeff), source=dict(self.source))

    # ------------------------------------------------------------------
    # Validation / positions / hooks
    # ------------------------------------------------------------------

    def validate(self, hooked: Any) -> int:
        """Check the vector fits *hooked*'s model; return the resolved non-negative layer.

        Raises
        ------
        SteeringMismatchError
            ``d_model`` differs, the recorded layer count differs, or the layer
            is out of range.
        SteeringUnavailableError
            *hooked* has no white-box model.

        A different recorded ``model_name`` only warns (the same weights can
        live under a hub id and a local path).
        """
        hm = as_hooked(hooked)
        if hm.d_model and self.direction.shape[0] != hm.d_model:
            raise SteeringMismatchError(
                f"steering vector has d_model={self.direction.shape[0]} but the model's "
                f"residual stream has d_model={hm.d_model}"
            )
        n = hm.n_layers
        if self.n_layers is not None and self.n_layers != n:
            raise SteeringMismatchError(
                f"steering vector was built for a {self.n_layers}-layer model; this model has {n}"
            )
        idx = self.layer + n if self.layer < 0 else self.layer
        if not 0 <= idx < n:
            raise SteeringMismatchError(f"layer {self.layer} out of range for a {n}-layer model")
        current = _model_name(hm)
        if self.model_name and current and not _same_model(self.model_name, current):
            warnings.warn(
                f"steering vector was built on {self.model_name!r} but is applied to {current!r}",
                UserWarning,
                stacklevel=2,
            )
        return idx

    def resolve_positions(
        self, prompt_len: int, total_len: int | None = None
    ) -> tuple[int, ...] | None:
        """Absolute token positions this vector edits (``None`` = every position).

        Parameters
        ----------
        prompt_len:
            Number of prompt tokens ``T`` (named modes and negative explicit
            indices are resolved against it).
        total_len:
            Number of tokens that will be fed through the model (the prompt
            plus every generated token except the last, which is never fed
            back: ``T + max_new_tokens - 1``); bounds the ``"generated"``
            range.  Defaults to ``prompt_len``.
        """
        t = int(prompt_len)
        if t < 1:
            raise ValueError("prompt_len must be >= 1")
        total = max(t, int(total_len) if total_len is not None else t)
        mode = self.positions
        if mode == "all":
            return None
        if mode == "last":
            return (t - 1,)
        if mode == "prompt":
            return tuple(range(t))
        if mode == "generated":
            return tuple(range(t - 1, total))
        out: set[int] = set()
        for p in mode:
            q = p if p >= 0 else t + p
            if q < 0:
                raise IndexError(f"position {p} is before the start of a {t}-token prompt")
            out.add(q)
        return tuple(sorted(out))

    def to_hook(
        self,
        hooked: Any,
        prompt_len: int,
        *,
        total_len: int | None = None,
        coeff: float | None = None,
    ) -> ResidHook:
        """A :class:`ResidHook` that adds :meth:`vector` at :attr:`site` of :attr:`layer`.

        *prompt_len* / *total_len* are as in :meth:`resolve_positions`.  When
        *total_len* is given and none of the explicit positions is below it,
        the hook could never fire and a :class:`UserWarning` is issued.
        """
        import torch

        hm = as_hooked(hooked)
        layer = self.validate(hm)
        delta = torch.as_tensor(self.vector(coeff), dtype=torch.float32, device=hm.device)

        def add(hidden: torch.Tensor) -> torch.Tensor:
            return hidden + delta.to(dtype=hidden.dtype, device=hidden.device)

        positions = self.resolve_positions(prompt_len, total_len)
        if positions is not None and total_len is not None:
            limit = max(int(prompt_len), int(total_len))
            if positions and min(positions) >= limit:
                warnings.warn(
                    f"steering positions {list(positions)} are all beyond the {limit} tokens "
                    "that will be processed; this vector has no effect",
                    UserWarning,
                    stacklevel=2,
                )
        return ResidHook(layer, add, site=self.site, positions=positions)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def metadata(self) -> dict[str, Any]:
        """JSON-safe description of everything except the direction values."""
        positions = self.positions if isinstance(self.positions, str) else list(self.positions)
        return {
            "format": _FORMAT,
            "version": _FORMAT_VERSION,
            "layer": self.layer,
            "site": self.site,
            "coeff": self.coeff,
            "normalize": self.normalize,
            "positions": positions,
            "name": self.name,
            "source": self.source,
            "model_name": self.model_name,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "norm": self.norm,
        }

    def to_dict(self, include_direction: bool = False) -> dict[str, Any]:
        """:meth:`metadata`, optionally with the direction as a list."""
        out = self.metadata()
        if include_direction:
            out["direction"] = self.direction.astype(float).tolist()
        return out

    def save(self, path: str | Path) -> Path:
        """Write ``direction`` + JSON metadata to an ``.npz`` file at exactly *path*."""
        path = Path(path)
        meta = json.dumps(self.metadata(), default=_json_default)
        with path.open("wb") as fh:
            np.savez(fh, direction=self.direction, meta=np.array(meta))
        return path

    @classmethod
    def load(cls, path: str | Path, hooked: Any = None) -> SteeringVector:
        """Read a vector written by :meth:`save`; validate it against *hooked* if given."""
        with np.load(Path(path), allow_pickle=False) as data:
            if "direction" not in data or "meta" not in data:
                raise ValueError(f"{path} is not a saved SteeringVector (missing arrays)")
            direction = np.array(data["direction"], dtype=np.float32)
            meta = json.loads(str(data["meta"]))
        if meta.get("format") != _FORMAT:
            raise ValueError(
                f"{path} is not a saved SteeringVector (format={meta.get('format')!r})"
            )
        if int(meta.get("version", 0)) > _FORMAT_VERSION:
            raise ValueError(
                f"{path} uses steering-vector format v{meta['version']}; "
                f"this version reads up to v{_FORMAT_VERSION}"
            )
        positions = meta.get("positions", "all")
        vec = cls(
            direction=direction,
            layer=int(meta["layer"]),
            site=meta.get("site", "resid_post"),
            coeff=float(meta.get("coeff", 1.0)),
            normalize=bool(meta.get("normalize", False)),
            positions=positions if isinstance(positions, str) else tuple(positions),
            name=str(meta.get("name", "")),
            source=dict(meta.get("source") or {}),
            model_name=meta.get("model_name"),
            d_model=meta.get("d_model"),
            n_layers=meta.get("n_layers"),
        )
        if hooked is not None:
            vec.validate(hooked)
        return vec

    def __repr__(self) -> str:
        pos = self.positions if isinstance(self.positions, str) else list(self.positions)
        label = f"{self.name!r}, " if self.name else ""
        return (
            f"SteeringVector({label}layer={self.layer}, site={self.site!r}, coeff={self.coeff}, "
            f"normalize={self.normalize}, positions={pos!r}, d_model={self.d_model}, "
            f"norm={self.norm:.4g})"
        )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_vector(
        cls,
        vec: Any,
        layer: int,
        *,
        site: SteeringSite = "resid_post",
        coeff: float = 1.0,
        normalize: bool = False,
        positions: PositionSpec = "all",
        name: str = "",
        model_name: str | None = None,
        n_layers: int | None = None,
        source: dict[str, Any] | None = None,
    ) -> SteeringVector:
        """Wrap an explicit ``(d_model,)`` direction (NumPy array, list or torch tensor)."""
        return cls(
            direction=_to_numpy(vec),
            layer=layer,
            site=site,
            coeff=coeff,
            normalize=normalize,
            positions=positions,
            name=name,
            source={"method": "explicit", **(source or {})},
            model_name=model_name,
            n_layers=n_layers,
        )

    @classmethod
    def from_contrast(
        cls,
        hooked: Any,
        positive_prompts: str | Sequence[str],
        negative_prompts: str | Sequence[str],
        layer: int,
        site: SteeringSite = "resid_post",
        position: Literal["last", "mean"] = "last",
        *,
        chat: bool = False,
        coeff: float = 1.0,
        normalize: bool = False,
        positions: PositionSpec = "all",
        name: str = "",
    ) -> SteeringVector:
        """Mean activation difference between positive and negative prompts.

        ``direction = mean_i a(pos_i) - mean_j a(neg_j)`` where ``a(p)`` is the
        residual at *site* of *layer* for prompt ``p``, read at its last token
        (``position="last"``) or averaged over all its tokens
        (``position="mean"``).  This is the mean-difference construction of
        Contrastive Activation Addition; one positive and one negative prompt
        is the ActAdd setting.  With ``normalize=False`` (default) ``coeff=1``
        adds the raw difference.

        Notes
        -----
        * ``position="last"`` on a one-token prompt reads position 0, which
          many models (GPT-2 among them) use as an attention sink with an
          outsized norm — a warning is issued.
        * ``position="mean"`` includes position 0; prompts that share their
          first token cancel it exactly (causal attention).
        * ``source`` records the prompts, counts, the mean activation norm
          (``resid_norm_mean``, a scale reference for ``coeff``) and the
          direction norm.
        """
        hm = as_hooked(hooked)
        pos_list = (
            [positive_prompts] if isinstance(positive_prompts, str) else list(positive_prompts)
        )
        neg_list = (
            [negative_prompts] if isinstance(negative_prompts, str) else list(negative_prompts)
        )
        if not pos_list or not neg_list:
            raise ValueError("from_contrast needs at least one positive and one negative prompt")
        if site not in STEERING_SITES:
            raise ValueError(f"unknown steering site {site!r}; expected one of {STEERING_SITES}")
        if position not in ("last", "mean"):
            raise ValueError(f"position must be 'last' or 'mean', got {position!r}")
        n = hm.n_layers
        li = int(layer) + n if int(layer) < 0 else int(layer)
        if not 0 <= li < n:
            raise SteeringMismatchError(f"layer {layer} out of range for a {n}-layer model")

        def collect(prompts: list[str]) -> np.ndarray:
            rows = []
            for p in prompts:
                res = hm.forward(p, capture_attentions=False, chat=chat)
                acts = res.resid(site)[li].detach().to("cpu", dtype=_torch_float64())
                if position == "last" and acts.shape[0] == 1:
                    warnings.warn(
                        f"prompt {p!r} is a single token: its 'last' activation is position 0, "
                        "often an attention sink with an outsized norm",
                        UserWarning,
                        stacklevel=3,
                    )
                row = acts[-1] if position == "last" else acts.mean(dim=0)
                rows.append(row.numpy())
            return np.stack(rows, axis=0)

        pos_acts, neg_acts = collect(pos_list), collect(neg_list)
        direction = pos_acts.mean(axis=0) - neg_acts.mean(axis=0)
        all_norms = np.linalg.norm(np.concatenate([pos_acts, neg_acts], axis=0), axis=1)
        source = {
            "method": "contrast_mean_difference",
            "technique": "Contrastive Activation Addition (mean difference; ActAdd for one pair)",
            "layer": li,
            "site": site,
            "position": position,
            "chat": bool(chat),
            "n_positive": len(pos_list),
            "n_negative": len(neg_list),
            "positive_prompts": [str(p) for p in pos_list],
            "negative_prompts": [str(p) for p in neg_list],
            "resid_norm_mean": float(all_norms.mean()),
            "direction_norm": float(np.linalg.norm(direction)),
            "family": hm.family,
        }
        return cls(
            direction=direction,
            layer=li,
            site=site,
            coeff=coeff,
            normalize=normalize,
            positions=positions,
            name=name,
            source=source,
            model_name=_model_name(hm),
            d_model=hm.d_model or None,
            n_layers=n,
        )

    @classmethod
    def from_sae_feature(
        cls,
        sae: Any,
        feature_id: int,
        *,
        layer: int | None = None,
        site: SteeringSite | None = None,
        coeff: float = 1.0,
        normalize: bool = True,
        positions: PositionSpec = "all",
        name: str = "",
        model_name: str | None = None,
        n_layers: int | None = None,
    ) -> SteeringVector:
        """An SAE feature's (unit) decoder direction, ``sae.feature_direction(feature_id)``.

        *layer* / *site* default to hook metadata carried by the SAE when
        present (``hook_layer`` / ``layer``, ``hook_site`` / ``site``, or an
        SAELens-style ``hook_name`` such as ``"blocks.6.hook_resid_post"``,
        looked up on the SAE, its ``config`` and a ``metadata`` / ``meta``
        dict).  Without any metadata *layer* is required and *site* defaults
        to ``"resid_post"`` (the historical convention for SAEs saved without
        hook metadata).  An SAE trained on a non-residual site, or whose
        ``hook_name`` names a hook point that is not ``resid_pre`` /
        ``resid_post``, needs an explicit residual *site*: the decoder
        direction is never silently re-interpreted as a residual direction.

        ``source`` records the feature id / label and the SAE's provenance
        (hook point, ``model_name``, ``release`` / ``sae_id``,
        ``normalize_activations``) when the SAE carries it.
        """
        cfg = getattr(sae, "config", None)
        dict_size = getattr(cfg, "dict_size", None)
        fid = int(feature_id)
        if dict_size is not None and not 0 <= fid < int(dict_size):
            raise IndexError(f"feature {fid} out of range for an SAE with {dict_size} features")
        info = _sae_hook_info(sae)
        meta_layer, meta_site = info.get("layer"), info.get("site")
        hook_name = info.get("hook_name")
        use_layer = layer if layer is not None else meta_layer
        if use_layer is None:
            where = f" (hook_name={hook_name!r} is not a residual hook)" if hook_name else ""
            raise ValueError(
                f"this SAE carries no hook-layer metadata{where}; pass layer= explicitly "
                "(the layer whose residual stream the SAE was trained on)"
            )
        if site is None and meta_site is None and hook_name:
            raise ValueError(
                f"the SAE was trained on hook point {hook_name!r}, which is not a residual-"
                f"stream site; steering adds to the residual stream, so pass site= one of "
                f"{STEERING_SITES} explicitly"
            )
        use_site = site if site is not None else (meta_site or "resid_post")
        if use_site not in STEERING_SITES:
            raise ValueError(
                f"the SAE was trained on site {use_site!r}; steering adds to the residual "
                f"stream, so pass site= one of {STEERING_SITES} explicitly"
            )
        direction = _to_numpy(sae.feature_direction(fid))
        if float(np.linalg.norm(direction)) < _NORM_EPS:
            raise ValueError(f"SAE feature {fid} has a zero decoder direction (dead feature)")
        input_dim = getattr(cfg, "input_dim", None)
        label = ""
        labels = getattr(sae, "labels", None)
        if isinstance(labels, dict):
            label = str(labels.get(fid, ""))
        source: dict[str, Any] = {
            "method": "sae_feature",
            "feature_id": fid,
            "feature_label": label,
            "sae_hook_layer": meta_layer,
            "sae_hook_site": meta_site,
            "sae_dict_size": None if dict_size is None else int(dict_size),
        }
        if hook_name:
            source["sae_hook_name"] = hook_name
        for key in ("release", "sae_id", "normalize_activations", "architecture"):
            val = getattr(cfg, key, None)
            if isinstance(val, (str, int, float, bool)) and not callable(val):
                source[f"sae_{key}"] = val
        return cls(
            direction=direction,
            layer=int(use_layer),
            site=use_site,  # type: ignore[arg-type]
            coeff=coeff,
            normalize=normalize,
            positions=positions,
            name=name or (label or f"sae_feature_{fid}"),
            source=source,
            model_name=model_name or info.get("model_name"),
            d_model=None if input_dim is None else int(input_dim),
            n_layers=n_layers,
        )


def _torch_float64() -> Any:
    import torch

    return torch.float64


_HOOK_NAME_RE = re.compile(r"(?:^|\.)(\d+)\.hook_(resid_pre|resid_post|resid_mid|mlp_out|attn_out)")


def _sae_hook_info(sae: Any) -> dict[str, Any]:
    """Best-effort ``{"layer", "site", "hook_name", "model_name"}`` from an SAE's metadata."""
    holders: list[Any] = [sae, getattr(sae, "config", None)]
    for attr in ("metadata", "meta"):
        extra = getattr(sae, attr, None)
        if isinstance(extra, dict):
            holders.append(extra)

    def lookup(keys: tuple[str, ...], kind: type | tuple[type, ...] = object) -> Any:
        for holder in holders:
            if holder is None:
                continue
            for key in keys:
                val = holder.get(key) if isinstance(holder, dict) else getattr(holder, key, None)
                if val is not None and not callable(val) and isinstance(val, kind):
                    return val
        return None

    info: dict[str, Any] = {}
    layer = lookup(("hook_layer", "layer"))
    site = lookup(("hook_site", "site"))
    point = getattr(sae, "hook_point", None)  # SparseAutoencoder: (layer, site) or None
    if isinstance(point, tuple) and len(point) == 2:
        layer = point[0] if layer is None else layer
        site = point[1] if site is None else site
    hook_name = lookup(("hook_name", "hook_point"), str)
    if isinstance(hook_name, str) and hook_name.strip():
        info["hook_name"] = hook_name.strip()
        m = _HOOK_NAME_RE.search(hook_name)
        if m:
            layer = int(m.group(1)) if layer is None else layer
            site = m.group(2) if site is None else site
    if isinstance(layer, (int, np.integer)) and not isinstance(layer, bool):
        info["layer"] = int(layer)
    if isinstance(site, str):
        info["site"] = site
    model = lookup(("model_name",))
    if isinstance(model, str) and model:
        info["model_name"] = model
    return info


# ---------------------------------------------------------------------------
# Hooks for several vectors
# ---------------------------------------------------------------------------


def _coerce_vectors(vectors: Any) -> list[SteeringVector]:
    vecs = [vectors] if isinstance(vectors, SteeringVector) else list(vectors)
    if not vecs:
        raise ValueError("at least one SteeringVector is required")
    for v in vecs:
        if not isinstance(v, SteeringVector):
            raise TypeError(f"expected SteeringVector, got {type(v).__name__}")
    return vecs


def _coerce_coeffs(coeffs: Any, vecs: list[SteeringVector]) -> list[float]:
    if coeffs is None:
        out = [v.coeff for v in vecs]
    elif isinstance(coeffs, (int, float, np.integer, np.floating)):
        out = [float(coeffs)] * len(vecs)
    else:
        out = [float(c) for c in coeffs]
        if len(out) != len(vecs):
            raise ValueError(f"got {len(out)} coeffs for {len(vecs)} steering vectors")
    if not all(math.isfinite(c) for c in out):
        raise ValueError("coefficients must be finite")
    return out


def steering_hooks(
    hooked: Any,
    vectors: SteeringVector | Sequence[SteeringVector],
    prompt_len: int,
    *,
    total_len: int | None = None,
    coeffs: float | Sequence[float] | None = None,
) -> list[ResidHook]:
    """One :class:`ResidHook` per vector (validated against the model).

    Pass the result to ``HookedModel.forward(hooks=...)`` /
    ``generate(hooks=...)``.  Vectors on the same site add up.
    """
    hm = as_hooked(hooked)
    vecs = _coerce_vectors(vectors)
    cs = _coerce_coeffs(coeffs, vecs)
    return [
        v.to_hook(hm, prompt_len, total_len=total_len, coeff=c)
        for v, c in zip(vecs, cs, strict=True)
    ]


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------


@dataclass
class TokenShift:
    """How one token's next-token probability moved under steering."""

    token: str
    token_id: int
    p_baseline: float
    p_steered: float
    delta: float
    logprob_delta: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationTrace:
    """One completion: ``logprobs[i] = log p(token_ids[i])`` under the model that wrote it."""

    text: str
    tokens: list[str]
    token_ids: list[int]
    logprobs: list[float]
    stop_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SteeringResult:
    """Baseline vs steered completion of one prompt, plus a teacher-forced comparison.

    Attributes
    ----------
    baseline, steered:
        The two completions (same decoding settings).
    kl_per_step:
        ``KL(p_steered || p_baseline)`` (nats) on the next-token distribution
        at each step of the *baseline* completion, both models conditioned on
        the same baseline prefix (teacher forcing).
    steered_logprobs_on_baseline:
        ``log p_steered(baseline token i | baseline prefix)`` — compare with
        ``baseline.logprobs``.
    promoted, suppressed:
        First-step tokens with the largest probability increase / decrease.
    vectors:
        Summary of each applied vector (``coeff`` is the one used).
    evidence_kind, effect_semantics, method, note:
        Evidence labels: the numbers are measured on a white-box model under
        an intervention (``effect_semantics="causal_intervention"``); ``note``
        says where the direction itself came from.
    """

    prompt: str
    prompt_tokens: list[str]
    prompt_token_ids: list[int]
    baseline: GenerationTrace
    steered: GenerationTrace
    kl_per_step: list[float]
    steered_logprobs_on_baseline: list[float]
    promoted: list[TokenShift]
    suppressed: list[TokenShift]
    vectors: list[dict[str, Any]]
    temperature: float = 0.0
    seed: int | None = None
    evidence_kind: str = "white_box"
    method: str = "activation_steering"
    note: str = ""
    effect_semantics: str = "causal_intervention"

    @property
    def first_step_kl(self) -> float:
        return self.kl_per_step[0] if self.kl_per_step else 0.0

    @property
    def mean_kl(self) -> float:
        return float(np.mean(self.kl_per_step)) if self.kl_per_step else 0.0

    @property
    def diverged_at(self) -> int | None:
        """Index of the first generated token that differs (``None`` if identical)."""
        a, b = self.baseline.token_ids, self.steered.token_ids
        for i, (x, y) in enumerate(zip(a, b, strict=False)):
            if x != y:
                return i
        return None if len(a) == len(b) else min(len(a), len(b))

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out.update(
            first_step_kl=self.first_step_kl, mean_kl=self.mean_kl, diverged_at=self.diverged_at
        )
        return out


@dataclass
class SweepRow:
    """One coefficient of :func:`coefficient_sweep`."""

    coeff: float
    kl: float
    target_probs: dict[str, float]
    top_tokens: list[tuple[str, float]]
    text: str | None

    @property
    def target_prob_total(self) -> float:
        return float(sum(self.target_probs.values()))

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["top_tokens"] = [list(t) for t in self.top_tokens]
        out["target_prob_total"] = self.target_prob_total
        return out


@dataclass
class SweepResult:
    """Next-token effect of one vector over several coefficients.

    ``kl`` is ``KL(p_coeff || p_baseline)`` on the next-token distribution
    after the prompt; ``target_probs`` are probabilities of the resolved
    target tokens (``targets`` maps label -> token id) at that position;
    ``text`` is the greedy steered completion.
    """

    prompt: str
    vector: dict[str, Any]
    targets: dict[str, int]
    baseline_target_probs: dict[str, float]
    baseline_top_tokens: list[tuple[str, float]]
    baseline_text: str | None
    rows: list[SweepRow]
    evidence_kind: str = "white_box"
    method: str = "activation_steering"
    note: str = ""
    effect_semantics: str = "causal_intervention"

    def table(self) -> list[dict[str, Any]]:
        """Flat rows: ``coeff``, ``kl``, ``p[<target>]`` per target, ``top``, ``text``."""
        out = []
        for r in self.rows:
            row: dict[str, Any] = {"coeff": r.coeff, "kl": r.kl}
            for label, p in r.target_probs.items():
                row[f"p[{label}]"] = p
            row["top"] = r.top_tokens[0][0] if r.top_tokens else ""
            row["text"] = r.text
            out.append(row)
        return out

    def format_table(self, text_width: int = 40) -> str:
        """Plain-text table for terminals / logs."""
        labels = list(self.targets)
        head = ["coeff", "KL"] + [f"p({lab!r})" for lab in labels] + ["top", "text"]
        lines = [" | ".join(head)]
        for r in self.rows:
            text = (r.text or "").replace("\n", "\\n")
            cells = [f"{r.coeff:g}", f"{r.kl:.4f}"]
            cells += [f"{r.target_probs[lab]:.4f}" for lab in labels]
            cells += [repr(r.top_tokens[0][0]) if r.top_tokens else "", repr(text[:text_width])]
            lines.append(" | ".join(cells))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "vector": self.vector,
            "targets": dict(self.targets),
            "baseline_target_probs": dict(self.baseline_target_probs),
            "baseline_top_tokens": [list(t) for t in self.baseline_top_tokens],
            "baseline_text": self.baseline_text,
            "rows": [r.to_dict() for r in self.rows],
            "evidence_kind": self.evidence_kind,
            "method": self.method,
            "note": self.note,
            "effect_semantics": self.effect_semantics,
        }


# ---------------------------------------------------------------------------
# Distribution helpers (torch, CPU float64 for stable KL)
# ---------------------------------------------------------------------------


def _log_probs(logits: Any) -> torch.Tensor:
    import torch

    return torch.log_softmax(logits.detach().to("cpu", dtype=torch.float64), dim=-1)


def _kl(logp: torch.Tensor, logq: torch.Tensor) -> torch.Tensor:
    """Row-wise ``KL(p || q)`` from log-probabilities (zero-probability terms contribute 0)."""
    import torch

    p = logp.exp()
    terms = torch.where(p > 0, p * (logp - logq), torch.zeros_like(p))
    return terms.sum(dim=-1).clamp_min(0.0)


def _top_tokens(hm: HookedModel, logp: torch.Tensor, k: int) -> list[tuple[str, float]]:
    import torch

    k = max(0, min(int(k), logp.shape[-1]))
    if k == 0:
        return []
    vals, idx = torch.topk(logp, k)
    return [(hm.token_str(int(i)), float(v.exp())) for v, i in zip(vals, idx, strict=True)]


def _token_shifts(
    hm: HookedModel, logp_base: torch.Tensor, logp_steer: torch.Tensor, k: int
) -> tuple[list[TokenShift], list[TokenShift]]:
    import torch

    k = max(0, min(int(k), logp_base.shape[-1]))
    if k == 0:
        return [], []
    pb, ps = logp_base.exp(), logp_steer.exp()
    delta = ps - pb

    def shift(i: int) -> TokenShift:
        return TokenShift(
            token=hm.token_str(i),
            token_id=i,
            p_baseline=float(pb[i]),
            p_steered=float(ps[i]),
            delta=float(delta[i]),
            logprob_delta=float(logp_steer[i] - logp_base[i]),
        )

    up = [shift(int(i)) for i in torch.topk(delta, k).indices if float(delta[int(i)]) > 0]
    down = [shift(int(i)) for i in torch.topk(-delta, k).indices if float(delta[int(i)]) < 0]
    return up, down


def _encode_no_special(tokenizer: Any, text: str) -> list[int]:
    try:
        enc = tokenizer(text, add_special_tokens=False)
    except TypeError:
        enc = tokenizer(text)
    ids = enc["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    flat = np.asarray(ids, dtype=np.int64).reshape(-1)
    return [int(i) for i in flat]


def _resolve_targets(hm: HookedModel, targets: Any) -> dict[str, int]:
    if targets is None:
        return {}
    if isinstance(targets, (str, int, np.integer)):
        targets = [targets]
    out: dict[str, int] = {}
    for t in targets:
        if isinstance(t, (int, np.integer)) and not isinstance(t, bool):
            tid = int(t)
            if hm.vocab_size and not 0 <= tid < hm.vocab_size:
                raise IndexError(f"token id {tid} out of range for vocab size {hm.vocab_size}")
            label = hm.token_str(tid)
        elif isinstance(t, str):
            ids = _encode_no_special(hm.tokenizer, t)
            if len(ids) != 1:
                pieces = [hm.token_str(i) for i in ids]
                raise ValueError(
                    f"target {t!r} is {len(ids)} tokens {pieces}; pass a single-token string "
                    "(BPE vocabularies usually need the leading space) or a token id"
                )
            tid, label = ids[0], t
        else:
            raise TypeError(f"target tokens must be str or int, got {type(t).__name__}")
        out.setdefault(label, tid)
    return out


def _prompt_of(hm: HookedModel, prompt: Any, chat: bool) -> TokenizedPrompt:
    if isinstance(prompt, TokenizedPrompt):
        return prompt
    return hm.tokenize(prompt, chat=chat)


def _summaries(vecs: list[SteeringVector], cs: list[float]) -> list[dict[str, Any]]:
    out = []
    for v, c in zip(vecs, cs, strict=True):
        d = v.metadata()
        d["coeff"] = c
        d["added_norm"] = float(np.linalg.norm(v.vector(c).astype(np.float64)))
        out.append(d)
    return out


def _evidence_note(vecs: list[SteeringVector]) -> str:
    origins = sorted({str(v.source.get("method", "explicit")) for v in vecs})
    return (
        "Interventional evidence for this model, prompt and coefficient: the residual stream "
        "was edited at the listed layer/site and the completion re-run. KL and probability "
        "shifts are teacher-forced along the baseline completion. The direction itself comes "
        f"from {', '.join(origins)}; a steering effect shows the direction is causally usable, "
        "not that the model represents the concept only there."
    )


# ---------------------------------------------------------------------------
# Steered generation
# ---------------------------------------------------------------------------


def steer_generate(
    hooked: Any,
    prompt: str | list[dict[str, str]] | TokenizedPrompt,
    vectors: SteeringVector | Sequence[SteeringVector],
    *,
    coeffs: float | Sequence[float] | None = None,
    max_new_tokens: int = 30,
    temperature: float = 0.0,
    chat: bool = False,
    seed: int | None = None,
    stop_at_eos: bool = True,
    top_k_tokens: int = 10,
) -> SteeringResult:
    """Generate with and without steering and compare the two.

    Parameters
    ----------
    hooked:
        A :class:`HookedModel` or a provider exposing one.
    prompt:
        A prompt string, chat messages (``chat=True``) or a
        :class:`TokenizedPrompt`.
    vectors:
        One or more :class:`SteeringVector` (applied together).
    coeffs:
        Override coefficients: one float for all vectors or one per vector
        (default: each vector's own ``coeff``).
    max_new_tokens, temperature, seed, stop_at_eos:
        Decoding settings shared by both runs.  ``temperature <= 0`` is greedy
        and deterministic; otherwise both runs sample with the same *seed*.
    top_k_tokens:
        How many promoted / suppressed first-step tokens to report.

    Raises
    ------
    SteeringUnavailableError
        *hooked* has no white-box model (mock / black-box providers).
    SteeringMismatchError
        A vector does not fit the model.
    """
    hm = as_hooked(hooked)
    vecs = _coerce_vectors(vectors)
    cs = _coerce_coeffs(coeffs, vecs)
    if int(max_new_tokens) < 1:
        raise ValueError("max_new_tokens must be >= 1")
    tp = _prompt_of(hm, prompt, chat)
    t_len = len(tp.token_ids)
    # Generation feeds the prompt plus every generated token but the last.
    processed = t_len + int(max_new_tokens) - 1
    hooks = steering_hooks(hm, vecs, t_len, total_len=processed, coeffs=cs)

    gen_kw: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "stop_at_eos": stop_at_eos,
        "seed": seed,
    }
    base = hm.generate(tp, **gen_kw)
    steer = hm.generate(tp, hooks=hooks, **gen_kw)

    # Teacher-forced comparison along the baseline prefix.  Steering positions
    # are absolute, so a full forward reproduces exactly what cached steered
    # generation would compute had it emitted the baseline tokens.
    n = len(base.token_ids)
    seq = list(tp.token_ids) + list(base.token_ids[:-1])
    window = slice(t_len - 1, t_len - 1 + n)
    lb = hm.forward(seq, capture=False, capture_attentions=False).logits[0, window]
    ls = hm.forward(seq, hooks=hooks, capture=False, capture_attentions=False).logits[0, window]
    logp_b, logp_s = _log_probs(lb), _log_probs(ls)
    kl = _kl(logp_s, logp_b)
    rows = list(range(n))
    on_base = logp_s[rows, list(base.token_ids)]
    promoted, suppressed = _token_shifts(hm, logp_b[0], logp_s[0], top_k_tokens)

    def trace(g: Any) -> GenerationTrace:
        return GenerationTrace(
            text=g.text,
            tokens=list(g.tokens),
            token_ids=list(g.token_ids),
            logprobs=list(g.logprobs or []),
            stop_reason=g.stop_reason,
        )

    return SteeringResult(
        prompt=tp.text or "".join(tp.tokens),
        prompt_tokens=list(tp.tokens),
        prompt_token_ids=list(tp.token_ids),
        baseline=trace(base),
        steered=trace(steer),
        kl_per_step=[float(x) for x in kl],
        steered_logprobs_on_baseline=[float(x) for x in on_base],
        promoted=promoted,
        suppressed=suppressed,
        vectors=_summaries(vecs, cs),
        temperature=float(temperature),
        seed=seed,
        note=_evidence_note(vecs),
    )


def coefficient_sweep(
    hooked: Any,
    prompt: str | list[dict[str, str]] | TokenizedPrompt,
    vector: SteeringVector,
    coeffs: Sequence[float],
    target_tokens: str | int | Sequence[str | int] | None = None,
    *,
    max_new_tokens: int = 20,
    chat: bool = False,
    top_k_tokens: int = 5,
) -> SweepResult:
    """Tabulate the next-token effect of *vector* at each coefficient.

    For every ``c`` in *coeffs*: ``KL(p_c || p_baseline)`` on the next-token
    distribution after the prompt, the probability of each target token
    (single-token strings or ids) at that position, the top tokens, and the
    greedy steered completion (``max_new_tokens=0`` skips generation).  The
    baseline is the unhooked model, so ``c = 0`` gives ``kl == 0`` exactly.
    """
    hm = as_hooked(hooked)
    vector.validate(hm)
    cs = [float(c) for c in coeffs]
    if not cs:
        raise ValueError("coeffs must not be empty")
    if not all(math.isfinite(c) for c in cs):
        raise ValueError("coefficients must be finite")
    tp = _prompt_of(hm, prompt, chat)
    t_len = len(tp.token_ids)
    total = t_len + max(0, int(max_new_tokens) - 1)  # tokens fed while generating
    targets = _resolve_targets(hm, target_tokens)

    def next_token(hooks: list[ResidHook]) -> torch.Tensor:
        out = hm.forward(tp, hooks=hooks, capture=False, capture_attentions=False)
        return _log_probs(out.logits[0, -1])

    def complete(hooks: list[ResidHook]) -> str | None:
        if int(max_new_tokens) <= 0:
            return None
        return hm.generate(tp, hooks=hooks, max_new_tokens=int(max_new_tokens)).text

    def probs(logp: torch.Tensor) -> dict[str, float]:
        return {label: float(logp[tid].exp()) for label, tid in targets.items()}

    logp_b = next_token([])
    rows = []
    for c in cs:
        hook = vector.to_hook(hm, t_len, total_len=total, coeff=c)
        logp_c = next_token([hook])
        rows.append(
            SweepRow(
                coeff=c,
                kl=float(_kl(logp_c, logp_b)),
                target_probs=probs(logp_c),
                top_tokens=_top_tokens(hm, logp_c, top_k_tokens),
                text=complete([hook]),
            )
        )
    summary = vector.metadata()
    summary.pop("coeff", None)
    return SweepResult(
        prompt=tp.text or "".join(tp.tokens),
        vector=summary,
        targets=targets,
        baseline_target_probs=probs(logp_b),
        baseline_top_tokens=_top_tokens(hm, logp_b, top_k_tokens),
        baseline_text=complete([]),
        rows=rows,
        note=_evidence_note([vector]),
    )
