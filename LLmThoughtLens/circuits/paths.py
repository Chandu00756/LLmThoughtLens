"""Top-k attribution paths through an :class:`AttributionGraph`.

Wraps :meth:`AttributionGraph.top_paths` with a richer return type that
includes the per-edge weights and the cumulative log-weight (so callers
can rank the paths or display the underlying numbers in the report).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from LLmThoughtLens.circuits.graph import AttributionGraph, CircuitEdge


@dataclass
class CausalPath:
    """A single causal path through the graph."""

    nodes: list[int]
    edges: list[CircuitEdge]
    total_weight: float
    log_score: float

    @property
    def length(self) -> int:
        return len(self.edges)

    def as_dict(self) -> dict:
        return {
            "nodes": list(self.nodes),
            "edges": [e.as_dict() for e in self.edges],
            "total_weight": float(self.total_weight),
            "log_score": float(self.log_score),
            "length": self.length,
        }


def top_causal_paths(
    graph: AttributionGraph,
    n: int = 5,
    source: int | None = None,
    sink: int | None = None,
) -> list[CausalPath]:
    """Return up to *n* highest-product-of-|weight| causal paths.

    Paths are ordered by ``log_score = sum(log|w|)`` descending, i.e. the
    product of edge magnitudes.  Sign of weights is preserved per edge.
    """
    paths_nodes = graph.top_paths(n=n, source=source, sink=sink)
    out: list[CausalPath] = []
    edge_index: dict[tuple[int, int], list] = {}
    for edge in graph.edges():
        edge_index.setdefault((edge.src, edge.dst), []).append(edge)

    import math

    for path in paths_nodes:
        edges_on_path = []
        log_score = 0.0
        product = 1.0
        for a, b in zip(path, path[1:], strict=False):
            cands = edge_index.get((a, b), [])
            if not cands:
                continue
            best = max(cands, key=lambda e: abs(e.weight))
            edges_on_path.append(best)
            product *= best.weight
            log_score += math.log(abs(best.weight) + 1e-12)
        out.append(
            CausalPath(
                nodes=list(path),
                edges=edges_on_path,
                total_weight=float(product),
                log_score=float(log_score),
            )
        )
    out.sort(key=lambda p: p.log_score, reverse=True)
    return out


def label_path(graph: AttributionGraph, path: CausalPath) -> str:
    """Return a human-readable ``"a → b → c"`` rendering of a path."""
    parts: list[str] = []
    for nid in path.nodes:
        n = graph.node(nid)
        if n is None:
            parts.append(f"?{nid}")
        else:
            parts.append(n.label or str(n.id))
    return " → ".join(parts)
