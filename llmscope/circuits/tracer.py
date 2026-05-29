"""CircuitTracer — builds attribution graphs from feature importance scores.

For white-box models: edges are weighted by cosine similarity between
``(layer, token)`` activation vectors at consecutive layers.

For black-box models: edges are weighted by the product of feature importance
scores (causal importance proxy).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from llmscope.circuits.graph import AttributionGraph

if TYPE_CHECKING:
    from llmscope.features.feature import Feature
    from llmscope.providers.base import ProviderOutput


class CircuitTracer:
    """Build an :class:`~llmscope.circuits.graph.AttributionGraph` from extracted features.

    Parameters
    ----------
    min_weight:
        Edges with absolute weight below this threshold are omitted.
    """

    def __init__(self, min_weight: float = 0.05) -> None:
        self.min_weight = min_weight

    def trace(
        self,
        output: "ProviderOutput",
        features: list["Feature"],
    ) -> AttributionGraph:
        """Build an attribution graph from a :class:`~llmscope.providers.base.ProviderOutput`
        and a list of extracted features.

        Parameters
        ----------
        output:
            Provider output (activations used when available).
        features:
            List of :class:`~llmscope.features.feature.Feature` objects to use
            as graph nodes.

        Returns
        -------
        AttributionGraph
        """
        graph = AttributionGraph(name=f"trace:{output.prompt[:50]}")

        for feat in features:
            graph.add_node(
                feat.id,
                label=feat.label,
                layer=feat.layer,
                score=float(feat.score),
                token_idx=feat.token_idx,
            )

        if output.activations is not None:
            return self._whitebox_trace(graph, features, output.activations)
        return self._blackbox_trace(graph, features)

    # ------------------------------------------------------------------
    # White-box tracing (cosine similarity between successive layer bands)
    # ------------------------------------------------------------------

    def _whitebox_trace(
        self,
        graph: AttributionGraph,
        features: list["Feature"],
        activations: np.ndarray,
    ) -> AttributionGraph:
        """Build edges using cosine similarity between successive-layer features."""
        by_layer: dict[int, list[Feature]] = {}
        for feat in features:
            by_layer.setdefault(feat.layer, []).append(feat)

        layers = sorted(by_layer)
        for i in range(len(layers) - 1):
            src_layer = layers[i]
            dst_layer = layers[i + 1]
            for src_feat in by_layer[src_layer]:
                src_vec = activations[src_feat.layer, src_feat.token_idx]
                for dst_feat in by_layer[dst_layer]:
                    dst_vec = activations[dst_feat.layer, dst_feat.token_idx]
                    weight = float(_cosine_sim(src_vec, dst_vec))
                    if abs(weight) >= self.min_weight:
                        graph.add_edge(src_feat.id, dst_feat.id, weight=weight)

        return graph

    # ------------------------------------------------------------------
    # Black-box tracing (importance product proxy)
    # ------------------------------------------------------------------

    def _blackbox_trace(
        self,
        graph: AttributionGraph,
        features: list["Feature"],
    ) -> AttributionGraph:
        """Build edges based on importance-score product (black-box approximation)."""
        for i, src in enumerate(features[:-1]):
            for dst in features[i + 1 :]:
                if dst.layer == src.layer:
                    continue
                weight = float(src.score * dst.score)
                if weight >= self.min_weight:
                    graph.add_edge(src.id, dst.id, weight=weight)
        return graph


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
