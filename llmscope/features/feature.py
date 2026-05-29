"""Feature and FeatureSet — core interpretable unit in llmscope."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Feature:
    """A single interpretable feature identified by a sparse autoencoder.

    Attributes
    ----------
    id:
        Globally unique feature identifier (usually an integer index into the
        SAE's feature dictionary).
    label:
        Human-readable description of what this feature represents.
    layer:
        Transformer layer where the feature was detected (0-indexed).
    score:
        Activation magnitude or cosine similarity score at the matched token.
    token_idx:
        Index of the token position where the feature fired.
    meta:
        Provider- or method-specific metadata.
    """

    id: int
    label: str = ""
    layer: int = 0
    score: float = 0.0
    token_idx: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"Feature(id={self.id}, label={self.label!r}, "
            f"layer={self.layer}, score={self.score:.4f})"
        )


@dataclass
class FeatureSet:
    """A named, ordered collection of :class:`Feature` objects.

    Analogous to a "supernode" in a circuit graph — a group of features that
    collectively represent a higher-level concept.

    Attributes
    ----------
    name:
        Human-readable label for this set.
    features:
        Ordered list of :class:`Feature` members.
    """

    name: str
    features: list[Feature] = field(default_factory=list)

    def add(self, feature: Feature) -> None:
        """Append *feature* to this set."""
        self.features.append(feature)

    def top(self, k: int = 5) -> list[Feature]:
        """Return the top-*k* features by score, descending."""
        return sorted(self.features, key=lambda f: f.score, reverse=True)[:k]

    def __len__(self) -> int:
        return len(self.features)

    def __iter__(self):
        return iter(self.features)

    def __repr__(self) -> str:
        return f"FeatureSet(name={self.name!r}, n={len(self)})"
