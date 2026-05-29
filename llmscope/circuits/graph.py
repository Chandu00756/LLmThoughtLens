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

    # ------------------------------------------------------------------
    # Pruning and path utilities
    # ------------------------------------------------------------------

    def prune(self, min_weight: float) -> "AttributionGraph":
        """Return a new graph with all edges whose ``|weight| < min_weight`` removed.

        Parameters
        ----------
        min_weight:
            Absolute weight threshold.

        Returns
        -------
        AttributionGraph
            A fresh graph containing only the high-weight edges.
        """
        pruned = AttributionGraph(name=self.name)
        for node_id, attrs in self._nodes.items():
            pruned.add_node(node_id, **attrs)
        for edge in self._edges:
            if abs(edge.weight) >= min_weight:
                pruned.add_edge(edge.src, edge.dst, weight=edge.weight, **edge.meta)
        return pruned

    def top_paths(self, n: int = 5, source: int | None = None) -> list[list[int]]:
        """Return the *n* highest total-weight paths through the graph.

        Uses a best-first DFS that sums edge weights along each path.
        Cycles are avoided by tracking visited nodes per path.

        Parameters
        ----------
        n:
            Number of paths to return.
        source:
            Starting node.  When ``None``, the node with the highest in-degree == 0
            (or first node) is used.

        Returns
        -------
        list of paths, each path being a list of node IDs from source to sink.
        """
        import heapq

        nodes = list(self._nodes)
        if not nodes:
            return []

        if source is None:
            # Pick a root: prefer nodes with no predecessors
            roots = [nid for nid in nodes if not self._in.get(nid)]
            source = roots[0] if roots else nodes[0]

        # Priority queue: (-cumulative_weight, path)
        heap: list[tuple[float, list[int]]] = [(-0.0, [source])]
        results: list[list[int]] = []

        while heap and len(results) < n:
            neg_w, path = heapq.heappop(heap)
            current = path[-1]
            out_edges = self._out.get(current, [])

            if not out_edges:
                # Sink node — record this path
                results.append(path)
                continue

            for edge in sorted(out_edges, key=lambda e: e.weight, reverse=True):
                if edge.dst in path:  # skip cycles
                    continue
                heapq.heappush(heap, (neg_w - edge.weight, path + [edge.dst]))

        return results

    def to_dict(self) -> dict:
        """Serialise the graph to a JSON-compatible dict.

        Returns
        -------
        dict with keys ``name``, ``nodes``, ``edges``.
        """
        return {
            "name": self.name,
            "nodes": [
                {"id": nid, **attrs} for nid, attrs in self._nodes.items()
            ],
            "edges": [
                {"src": e.src, "dst": e.dst, "weight": e.weight, **e.meta}
                for e in self._edges
            ],
        }

    def __repr__(self) -> str:
        return (
            f"AttributionGraph(name={self.name!r}, "
            f"nodes={self.num_nodes}, edges={self.num_edges})"
        )
