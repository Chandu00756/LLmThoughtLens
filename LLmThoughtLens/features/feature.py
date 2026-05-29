"""Feature and FeatureSet — interpretable units that flow through the package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class Feature:
    """One interpretable feature.

    Attributes
    ----------
    id:
        Globally unique identifier (SAE dictionary index in white-box mode,
        token position id in black-box mode).
    label:
        Human-readable name, e.g. ``"US capital retrieval"``.
    layer:
        Transformer layer where the feature fired (0-indexed).
    score:
        Activation magnitude (white-box) or causal importance score
        (black-box) at the matched token.
    token_idx:
        Index of the token position where the feature fired.
    node_type:
        One of ``"feature" | "input_token" | "output_token" | "supernode" |
        "safety" | "suppressor" | "error"``.  Used by the graph layout and
        the report's colour coding.
    evidence_kind:
        ``"white_box"`` if score comes from real activations,
        ``"black_box"`` if it comes from token-masking perturbation.
    meta:
        Method-specific metadata (e.g. ``contexts`` from the labeler).
    """

    id: int
    label: str = ""
    layer: int = 0
    score: float = 0.0
    token_idx: int = 0
    node_type: str = "feature"
    evidence_kind: str = "white_box"
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "layer": self.layer,
            "score": float(self.score),
            "token_idx": self.token_idx,
            "node_type": self.node_type,
            "evidence_kind": self.evidence_kind,
        }

    def __repr__(self) -> str:
        return (
            f"Feature(id={self.id}, label={self.label!r}, "
            f"layer={self.layer}, score={self.score:.4f}, "
            f"node={self.node_type}, evidence={self.evidence_kind})"
        )


@dataclass
class FeatureSet:
    """Ordered cluster of related :class:`Feature` objects — a "supernode"."""

    name: str
    features: list[Feature] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def add(self, feature: Feature) -> None:
        self.features.append(feature)

    def top(self, k: int = 5) -> list[Feature]:
        return sorted(self.features, key=lambda f: f.score, reverse=True)[:k]

    def total_score(self) -> float:
        return float(sum(f.score for f in self.features))

    def representative_layer(self) -> int:
        if not self.features:
            return 0
        return max(
            {f.layer for f in self.features},
            key=lambda lyr: sum(f.score for f in self.features if f.layer == lyr),
        )

    def representative_token(self) -> int:
        if not self.features:
            return 0
        return max(
            {f.token_idx for f in self.features},
            key=lambda tok: sum(f.score for f in self.features if f.token_idx == tok),
        )

    def __len__(self) -> int:
        return len(self.features)

    def __iter__(self) -> Iterator[Feature]:
        return iter(self.features)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_features": len(self),
            "total_score": self.total_score(),
            "features": [f.as_dict() for f in self.features],
        }

    def __repr__(self) -> str:
        return f"FeatureSet(name={self.name!r}, n={len(self)}, total={self.total_score():.2f})"
