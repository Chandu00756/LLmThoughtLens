"""GraphDiff — compare two attribution graphs (baseline vs intervention)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from LLmThoughtLens.circuits.graph import AttributionGraph


@dataclass
class GraphDiff:
    """Structured diff between *a* (baseline) and *b* (intervention).

    Attributes
    ----------
    added_nodes:
        Nodes present in *b* but not in *a*.
    removed_nodes:
        Nodes present in *a* but not in *b*.
    added_edges:
        ``(src, dst)`` pairs present in *b* but not in *a*.
    removed_edges:
        ``(src, dst)`` pairs present in *a* but not in *b*.
    changed_edges:
        ``[(src, dst, weight_a, weight_b)]`` for edges whose weight moved
        more than ``threshold``.
    """

    added_nodes: list[int] = field(default_factory=list)
    removed_nodes: list[int] = field(default_factory=list)
    added_edges: list[tuple[int, int]] = field(default_factory=list)
    removed_edges: list[tuple[int, int]] = field(default_factory=list)
    changed_edges: list[tuple[int, int, float, float]] = field(default_factory=list)
    threshold: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def compute(
        cls,
        a: AttributionGraph,
        b: AttributionGraph,
        threshold: float = 0.0,
    ) -> GraphDiff:
        a_nodes = {n.id for n in a.nodes()}
        b_nodes = {n.id for n in b.nodes()}
        added_nodes = sorted(b_nodes - a_nodes)
        removed_nodes = sorted(a_nodes - b_nodes)

        a_edges = {(e.src, e.dst): e.weight for e in a.edges()}
        b_edges = {(e.src, e.dst): e.weight for e in b.edges()}
        added_edges = sorted(set(b_edges) - set(a_edges))
        removed_edges = sorted(set(a_edges) - set(b_edges))
        changed: list[tuple[int, int, float, float]] = []
        for key in set(a_edges) & set(b_edges):
            wa, wb = a_edges[key], b_edges[key]
            if abs(wa - wb) > threshold:
                changed.append((key[0], key[1], float(wa), float(wb)))
        changed.sort(key=lambda t: abs(t[3] - t[2]), reverse=True)

        return cls(
            added_nodes=added_nodes,
            removed_nodes=removed_nodes,
            added_edges=added_edges,
            removed_edges=removed_edges,
            changed_edges=changed,
            threshold=float(threshold),
            meta={"a": a.name, "b": b.name},
        )

    def summary(self) -> str:
        return (
            f"GraphDiff(+{len(self.added_nodes)}/-{len(self.removed_nodes)} nodes, "
            f"+{len(self.added_edges)}/-{len(self.removed_edges)} edges, "
            f"{len(self.changed_edges)} changed)"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "added_nodes": list(self.added_nodes),
            "removed_nodes": list(self.removed_nodes),
            "added_edges": [list(t) for t in self.added_edges],
            "removed_edges": [list(t) for t in self.removed_edges],
            "changed_edges": [list(t) for t in self.changed_edges],
            "threshold": float(self.threshold),
            "meta": dict(self.meta),
        }

    def __repr__(self) -> str:
        return self.summary()
