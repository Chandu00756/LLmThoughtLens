"""FeatureIntervention — amplify, inhibit, or clamp features mid-inference.

Interventions let you test causal hypotheses: suppress a feature and see
whether the model's reasoning changes, confirming the feature's causal role.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

InterventionMode = Literal["amplify", "inhibit", "clamp"]


@dataclass
class FeatureIntervention:
    """Specification for a feature-level intervention on activation tensors.

    Attributes
    ----------
    feature_id:
        ID of the feature to intervene on.
    mode:
        ``"amplify"`` scales the feature activation up by ``scale``.
        ``"inhibit"`` suppresses it toward zero (multiplied by ``1 - |scale|``).
        ``"clamp"`` sets it to a fixed value equal to ``scale``.
    scale:
        Scaling factor (amplify/inhibit) or clamped value (clamp mode).
    layer:
        Transformer layer to target. ``-1`` applies to all layers.
    token_idx:
        Token position to target. ``-1`` applies to all positions.
    """

    feature_id: int
    mode: InterventionMode = "inhibit"
    scale: float = 0.0
    layer: int = -1
    token_idx: int = -1

    def apply(self, activations: np.ndarray) -> np.ndarray:
        """Apply this intervention to an activation tensor.

        Parameters
        ----------
        activations:
            Shape ``(n_layers, n_tokens, d_model)``.

        Returns
        -------
        np.ndarray
            Modified copy of the activations array.
        """
        result = activations.copy()
        n_layers, n_tokens, d_model = result.shape

        layer_range: range | list[int] = (
            range(n_layers) if self.layer == -1 else [self.layer % n_layers]
        )
        token_range: range | list[int] = (
            range(n_tokens) if self.token_idx == -1 else [self.token_idx % n_tokens]
        )

        # Map feature_id to a dimension index (wraps around d_model)
        feat_dim = self.feature_id % d_model

        for lyr in layer_range:
            for tok in token_range:
                if self.mode == "amplify":
                    result[lyr, tok, feat_dim] *= self.scale
                elif self.mode == "inhibit":
                    result[lyr, tok, feat_dim] *= max(0.0, 1.0 - abs(self.scale))
                elif self.mode == "clamp":
                    result[lyr, tok, feat_dim] = self.scale

        return result

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def amplify(cls, feature_id: int, scale: float = 2.0, **kwargs) -> "FeatureIntervention":
        """Create an amplification intervention (scale > 1 strengthens the feature)."""
        return cls(feature_id=feature_id, mode="amplify", scale=scale, **kwargs)

    @classmethod
    def inhibit(cls, feature_id: int, scale: float = 1.0, **kwargs) -> "FeatureIntervention":
        """Create an inhibition intervention (scale=1.0 fully suppresses the feature)."""
        return cls(feature_id=feature_id, mode="inhibit", scale=scale, **kwargs)

    @classmethod
    def clamp(cls, feature_id: int, value: float = 0.0, **kwargs) -> "FeatureIntervention":
        """Create a clamping intervention (sets the feature to a fixed value)."""
        return cls(feature_id=feature_id, mode="clamp", scale=value, **kwargs)

    def __repr__(self) -> str:
        return (
            f"FeatureIntervention(id={self.feature_id}, mode={self.mode!r}, "
            f"scale={self.scale}, layer={self.layer}, token={self.token_idx})"
        )
