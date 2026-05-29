"""AttributionGraph — directed weighted graph of feature-level causal edges."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class Edge:
    """A directed causal edge between two features in the attribution graph.

    Attributes
    ----------
    src:
        Source node id (upstream feature).
    dst:
        Destination node id (downstream feature).
    weight:
        Causal attribution weight (e.g. integrated gradient value).
    meta:
        Optional edge metadata.
    """

    src: int
    dst: int
    weight: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


class AttributionGraph:
    """Directed weighted graph connecting :class:`~llmscope.features.feature.Feature` nodes.

    Nodes are represented by integer feature IDs; edges are
    :class:`Edge` objects.  The graph supports forward and backward
    neighbour lookup in O(1) via adjacency dicts.

    Parameters
    ----------
    name:
        Optional human-readable name for this graph.
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self._nodes: dict[int, dict[str, Any]] = {}
        self._edges: list[Edge] = []
        # adjacency: src_id → list[Edge]
        self._out: dict[int, list[Edge]] = {}
        # reverse adjacency: dst_id → list[Edge]
        self._in: dict[int, list[Edge]] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_node(self, node_id: int, **attrs: Any) -> None:
        """Add or update a node with optional attributes."""
        self._nodes.setdefault(node_id, {}).update(attrs)
        self._out.setdefault(node_id, [])
        self._in.setdefault(node_id, [])

    def add_edge(self, src: int, dst: int, weight: float = 0.0, **meta: Any) -> Edge:
        """Add a directed edge from *src* to *dst*.

        Both nodes are auto-created if they do not already exist.
        """
        self.add_node(src)
        self.add_node(dst)
        edge = Edge(src=src, dst=dst, weight=weight, meta=meta)
        self._edges.append(edge)
        self._out[src].append(edge)
        self._in[dst].append(edge)
        return edge

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def successors(self, node_id: int) -> list[int]:
        """Return list of node IDs reachable in one hop from *node_id*."""
        return [e.dst for e in self._out.get(node_id, [])]

    def predecessors(self, node_id: int) -> list[int]:
        """Return list of node IDs that point to *node_id*."""
        return [e.src for e in self._in.get(node_id, [])]

    def out_edges(self, node_id: int) -> list[Edge]:
        """Return all outgoing edges from *node_id*."""
        return list(self._out.get(node_id, []))

    def in_edges(self, node_id: int) -> list[Edge]:
        """Return all incoming edges to *node_id*."""
        return list(self._in.get(node_id, []))

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def nodes(self) -> Iterator[int]:
        """Iterate over node IDs."""
        return iter(self._nodes)

    def edges(self) -> Iterator[Edge]:
        """Iterate over all :class:`Edge` objects."""
        return iter(self._edges)

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    @property
    def num_edges(self) -> int:
        return len(self._edges)

    def __repr__(self) -> str:
        return (
            f"AttributionGraph(name={self.name!r}, "
            f"nodes={self.num_nodes}, edges={self.num_edges})"
        )
