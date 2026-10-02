"""FeatureIntervention — amplify, inhibit, or clamp features mid-inference.

Three execution surfaces:

- :meth:`apply_numpy` — operate on already-collected ``(L, T, D)`` arrays.
  Used by offline analysis (e.g. patched-graph comparison from cached
  activations) and by ``MockProvider.run_with_intervention``.
- :meth:`apply_torch` — operate on a live torch hidden-state tensor
  ``(B, T, D)`` inside a hook installed on the target transformer block.
  This is the path that actually changes the model's forward computation.
- :func:`intervention_context` — a context manager that installs hooks
  on a list of transformer blocks, runs the user's forward pass, and
  guarantees hook removal on exit (even on exception).

Hook site (``FeatureIntervention.site``) for the torch path:

- ``"mlp_in"`` (default, the historical behaviour) — forward-pre hook on the
  block's ``.mlp`` submodule (falling back to ``.feed_forward``, then the
  block itself).  For pre-LN models (GPT-2, Llama, ...) the MLP input is the
  *normalised* residual (after ``ln_2`` / ``post_attention_layernorm``).
- ``"resid_pre"`` — forward-pre hook on the block: edits the residual stream
  entering block ``layer``.
- ``"resid_post"`` — forward hook on the block: edits the residual stream
  leaving block ``layer`` (what :attr:`ProviderOutput.activations` records).

Residual sites follow the convention of :mod:`LLmThoughtLens.models.hooked`.
The NumPy path ignores ``site``: it edits whatever ``(L, T, D)`` array it is
given (the provider's ``resid_post`` activations for cached analysis).

When an SAE is attached, the intervention modifies activations *along the
SAE decoder column for the chosen feature* rather than along a single
coordinate.  Without an SAE attached, the feature_id is interpreted as a
raw hidden-dimension index (wrapped to ``d_model``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from LLmThoughtLens.features.sae import SparseAutoencoder

InterventionMode = Literal["amplify", "inhibit", "clamp"]
InterventionSite = Literal["mlp_in", "resid_pre", "resid_post"]
INTERVENTION_SITES: tuple[str, ...] = ("mlp_in", "resid_pre", "resid_post")


@dataclass
class FeatureIntervention:
    """Specification for a feature-level intervention.

    Attributes
    ----------
    feature_id:
        SAE feature id, or raw hidden-dim index if no SAE is attached.
    mode:
        ``"amplify"`` multiplies the feature projection by ``scale``.
        ``"inhibit"`` multiplies it by ``max(0, 1 - |scale|)`` (scale=1 fully kills it).
        ``"clamp"`` replaces the projection with the constant ``scale``.
    scale:
        Scaling factor (amplify/inhibit) or clamped value (clamp mode).
    layer:
        Transformer layer to target.  ``-1`` means "every layer" (NumPy
        path) or "the SAE layer if one is attached, else layer 0" (torch path).
    token_idx:
        Token position to target.  ``-1`` means "every position".
    sae:
        Optional attached :class:`SparseAutoencoder`.  When supplied, the
        intervention acts along the SAE decoder direction for *feature_id*.
    direction:
        Optional explicit direction vector (overrides *sae* / *feature_id*).
    site:
        Where the torch path hooks: ``"mlp_in"`` (default, historical
        behaviour), ``"resid_pre"`` or ``"resid_post"`` — see the module
        docstring.
    """

    feature_id: int
    mode: InterventionMode = "inhibit"
    scale: float = 1.0
    layer: int = -1
    token_idx: int = -1
    sae: SparseAutoencoder | None = field(default=None, repr=False)
    direction: np.ndarray | None = field(default=None, repr=False)
    site: InterventionSite = "mlp_in"

    def __post_init__(self) -> None:
        if self.site not in INTERVENTION_SITES:
            raise ValueError(
                f"unknown intervention site {self.site!r}; expected one of {INTERVENTION_SITES}"
            )

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def amplify(cls, feature_id: int, scale: float = 2.0, **kwargs: Any) -> FeatureIntervention:
        return cls(feature_id=feature_id, mode="amplify", scale=scale, **kwargs)

    @classmethod
    def inhibit(cls, feature_id: int, scale: float = 1.0, **kwargs: Any) -> FeatureIntervention:
        return cls(feature_id=feature_id, mode="inhibit", scale=scale, **kwargs)

    @classmethod
    def clamp(cls, feature_id: int, value: float = 0.0, **kwargs: Any) -> FeatureIntervention:
        return cls(feature_id=feature_id, mode="clamp", scale=value, **kwargs)

    # ------------------------------------------------------------------
    # Direction resolution
    # ------------------------------------------------------------------

    def _resolve_direction_numpy(self, d_model: int) -> np.ndarray:
        """Return the (unit) direction vector this intervention acts along."""
        if self.direction is not None:
            v = np.asarray(self.direction, dtype=np.float32)
        elif self.sae is not None:
            v = self.sae.feature_direction(int(self.feature_id))
            if v.shape[0] != d_model:
                raise ValueError(
                    f"SAE direction dim ({v.shape[0]}) does not match d_model ({d_model})"
                )
        else:
            v = np.zeros(d_model, dtype=np.float32)
            v[int(self.feature_id) % d_model] = 1.0
        n = float(np.linalg.norm(v))
        return v if n < 1e-9 else (v / n)

    # ------------------------------------------------------------------
    # NumPy path — operate on (L, T, D) cached arrays
    # ------------------------------------------------------------------

    def apply_numpy(self, activations: np.ndarray) -> np.ndarray:
        """Apply the intervention to a cached ``(L, T, D)`` array.

        Parameters
        ----------
        activations:
            Shape ``(n_layers, n_tokens, d_model)``.

        Returns
        -------
        np.ndarray
            A new array with the intervention applied.
        """
        if activations.ndim != 3:
            raise ValueError(f"expected (L, T, D) activations, got shape {activations.shape}")
        result = activations.astype(np.float32, copy=True)
        n_layers, n_tokens, d_model = result.shape
        direction = self._resolve_direction_numpy(d_model)

        layer_indices = list(range(n_layers)) if self.layer == -1 else [int(self.layer) % n_layers]
        token_indices = (
            list(range(n_tokens)) if self.token_idx == -1 else [int(self.token_idx) % n_tokens]
        )

        for lyr in layer_indices:
            for tok in token_indices:
                vec = result[lyr, tok]
                proj = float(np.dot(vec, direction))
                if self.mode == "amplify":
                    delta = (self.scale - 1.0) * proj
                elif self.mode == "inhibit":
                    factor = max(0.0, 1.0 - abs(self.scale))
                    delta = (factor - 1.0) * proj
                else:  # clamp
                    delta = self.scale - proj
                result[lyr, tok] = vec + delta * direction
        return result

    # Backwards-compatible alias.
    def apply(self, activations: np.ndarray) -> np.ndarray:
        return self.apply_numpy(activations)

    # ------------------------------------------------------------------
    # Torch path — operate inside a forward hook
    # ------------------------------------------------------------------

    def apply_torch(self, hidden: Any) -> Any:
        """Apply the intervention to a live torch tensor ``(B, T, D)``."""
        import torch

        if not isinstance(hidden, torch.Tensor):
            return hidden
        if hidden.dim() != 3:
            return hidden

        d_model = hidden.shape[-1]
        direction_np = self._resolve_direction_numpy(d_model)
        direction = torch.as_tensor(direction_np, dtype=hidden.dtype, device=hidden.device)

        token_idx = self.token_idx
        if token_idx == -1:
            target = hidden
        else:
            target = hidden[:, token_idx % hidden.shape[1] : token_idx % hidden.shape[1] + 1, :]

        # Project current activation onto the direction.
        proj = (target * direction).sum(dim=-1, keepdim=True)
        if self.mode == "amplify":
            delta = (self.scale - 1.0) * proj
        elif self.mode == "inhibit":
            factor = max(0.0, 1.0 - abs(self.scale))
            delta = (factor - 1.0) * proj
        else:  # clamp
            delta = self.scale - proj
        update = delta * direction

        if self.token_idx == -1:
            return hidden + update
        new_hidden = hidden.clone()
        idx = self.token_idx % hidden.shape[1]
        new_hidden[:, idx : idx + 1, :] = target + update
        return new_hidden

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        site = "" if self.site == "mlp_in" else f", site={self.site!r}"
        return (
            f"FeatureIntervention(id={self.feature_id}, mode={self.mode!r}, "
            f"scale={self.scale}, layer={self.layer}, token={self.token_idx}, "
            f"sae={'yes' if self.sae is not None else 'no'}{site})"
        )


# ---------------------------------------------------------------------------
# Context manager for hook lifecycle
# ---------------------------------------------------------------------------


class intervention_context:  # noqa: N801 — lowercase to read like a contextmanager call site
    """Install hooks for *interventions* on *blocks*, then guarantee removal on exit.

    Used internally by :class:`HuggingFaceProvider.run_with_intervention` and
    available to advanced users running their own forward passes::

        with intervention_context(blocks, [intervention]):
            outputs = model(**inputs)

    The block list must be the same ``ModuleList`` that holds the transformer
    blocks (e.g. ``model.transformer.h`` for GPT-2, ``model.model.layers``
    for Llama, or ``HookedModel.blocks``).  Each intervention's ``.layer``
    field selects the block and ``.site`` the hook point: ``"mlp_in"`` hooks
    the block's ``.mlp`` submodule input (falling back to ``.feed_forward``
    or the block itself if neither exists); ``"resid_pre"`` / ``"resid_post"``
    hook the block's input / output residual stream.
    """

    def __init__(self, blocks: list[Any], interventions: list[FeatureIntervention]):
        self.blocks = blocks
        self.interventions = interventions
        self._handles: list[Any] = []

    def __enter__(self) -> intervention_context:
        try:
            for spec in self.interventions:
                if getattr(spec, "site", "mlp_in") == "mlp_in":
                    h = _install_mlp_hook(self.blocks, spec)
                else:
                    h = _install_resid_hook(self.blocks, spec)
                if h is not None:
                    self._handles.append(h)
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *_exc: Any) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:  # noqa: BLE001 — best-effort cleanup
                continue
        self._handles.clear()

    @property
    def n_installed(self) -> int:
        return len(self._handles)


def _target_layer(blocks: list[Any], spec: FeatureIntervention) -> int:
    return int(spec.layer) % len(blocks) if spec.layer != -1 else 0


def _install_resid_hook(blocks: list[Any], spec: FeatureIntervention) -> Any | None:
    """Hook the residual stream entering (``resid_pre``) or leaving (``resid_post``)
    ``blocks[spec.layer]``.  Returns the handle, or ``None`` if *blocks* is empty."""
    if not blocks:
        return None
    from LLmThoughtLens.models.hooked import register_site_hook

    block = blocks[_target_layer(blocks, spec)]
    if spec.site == "resid_pre":
        return register_site_hook(block, "pre", spec.apply_torch)
    return register_site_hook(block, "post", spec.apply_torch)


def _install_mlp_hook(blocks: list[Any], spec: FeatureIntervention) -> Any | None:
    """Register a forward-pre hook on the MLP submodule of ``blocks[spec.layer]``.

    Returns the handle (so the caller can ``.remove()`` it).  ``None`` if
    *blocks* is empty.
    """
    if not blocks:
        return None
    block = blocks[_target_layer(blocks, spec)]
    # The tensor that feeds the MLP's forward call: for pre-LN blocks
    # (GPT-2, Llama, ...) this is the *normalised* residual stream.  Use
    # site="resid_pre" / "resid_post" to edit the residual stream itself.
    target_module = getattr(block, "mlp", None)
    if target_module is None:
        target_module = getattr(block, "feed_forward", None)
    if target_module is None:
        # No MLP submodule (unusual): fall back to the block itself.
        target_module = block

    def _pre_hook(_module: Any, inputs: Any) -> Any:
        if not inputs:
            return inputs
        first = inputs[0]
        if first is None:
            return inputs
        modified = spec.apply_torch(first)
        return (modified,) + tuple(inputs[1:])

    return target_module.register_forward_pre_hook(_pre_hook)
