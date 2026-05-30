"""AttributionGraph — typed directed weighted graph of circuit nodes/edges.

Node types follow the design-doc taxonomy:
    "input_token", "feature", "supernode", "output_token",
    "error" (unexplained residual), "safety", "suppressor".

Edges carry signed weights — positive weights *promote* the output, negative
weights *suppress* it.  The report's colour scheme is driven directly off
``CircuitEdge.polarity``.
"""

from __future__ import annotations

import csv
import heapq
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal

NodeType = Literal[
    "input_token",
    "feature",
    "supernode",
    "output_token",
    "error",
    "safety",
    "suppressor",
]
EdgePolarity = Literal["promote", "suppress"]


@dataclass
class CircuitNode:
    """One node in an :class:`AttributionGraph`."""

    id: int
    label: str = ""
    node_type: NodeType = "feature"
    layer: int = 0
    token_idx: int = 0
    score: float = 0.0
    evidence_kind: str = "white_box"
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "node_type": self.node_type,
            "layer": self.layer,
            "token_idx": self.token_idx,
            "score": float(self.score),
            "evidence_kind": self.evidence_kind,
            **self.meta,
        }


@dataclass
class CircuitEdge:
    """One directed weighted edge."""

    src: int
    dst: int
    weight: float = 0.0
    polarity: EdgePolarity = "promote"
    method: str = "unknown"
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "src": self.src,
            "dst": self.dst,
            "weight": float(self.weight),
            "polarity": self.polarity,
            "method": self.method,
            **self.meta,
        }


# Backwards-compatibility shim — the previous skeleton used `Edge`.
Edge = CircuitEdge


class AttributionGraph:
    """Directed graph connecting input tokens, features, supernodes, and the output."""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self._nodes: dict[int, CircuitNode] = {}
        self._edges: list[CircuitEdge] = []
        self._out: dict[int, list[CircuitEdge]] = {}
        self._in: dict[int, list[CircuitEdge]] = {}
        self.meta: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_node(
        self,
        node_id: int,
        *,
        label: str = "",
        node_type: NodeType = "feature",
        layer: int = 0,
        token_idx: int = 0,
        score: float = 0.0,
        evidence_kind: str = "white_box",
        **meta: Any,
    ) -> CircuitNode:
        node = self._nodes.get(node_id)
        if node is None:
            node = CircuitNode(
                id=node_id,
                label=label,
                node_type=node_type,
                layer=layer,
                token_idx=token_idx,
                score=score,
                evidence_kind=evidence_kind,
                meta=dict(meta),
            )
            self._nodes[node_id] = node
            self._out.setdefault(node_id, [])
            self._in.setdefault(node_id, [])
        else:
            # Update in place — used when CircuitTracer revisits the output node.
            if label:
                node.label = label
            if node_type:
                node.node_type = node_type
            node.layer = layer or node.layer
            node.token_idx = token_idx if token_idx else node.token_idx
            node.score = score if score else node.score
            node.evidence_kind = evidence_kind or node.evidence_kind
            node.meta.update(meta)
        return node

    def add_edge(
        self,
        src: int,
        dst: int,
        weight: float = 0.0,
        *,
        method: str = "unknown",
        **meta: Any,
    ) -> CircuitEdge:
        if src not in self._nodes:
            self.add_node(src)
        if dst not in self._nodes:
            self.add_node(dst)
        polarity: EdgePolarity = "promote" if weight >= 0 else "suppress"
        edge = CircuitEdge(
            src=src,
            dst=dst,
            weight=float(weight),
            polarity=polarity,
            method=method,
            meta=dict(meta),
        )
        self._edges.append(edge)
        self._out[src].append(edge)
        self._in[dst].append(edge)
        return edge

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def node(self, node_id: int) -> CircuitNode | None:
        return self._nodes.get(node_id)

    def successors(self, node_id: int) -> list[int]:
        return [e.dst for e in self._out.get(node_id, [])]

    def predecessors(self, node_id: int) -> list[int]:
        return [e.src for e in self._in.get(node_id, [])]

    def out_edges(self, node_id: int) -> list[CircuitEdge]:
        return list(self._out.get(node_id, []))

    def in_edges(self, node_id: int) -> list[CircuitEdge]:
        return list(self._in.get(node_id, []))

    def nodes(self) -> Iterator[CircuitNode]:
        return iter(self._nodes.values())

    def edges(self) -> Iterator[CircuitEdge]:
        return iter(self._edges)

    def nodes_by_type(self, node_type: NodeType) -> list[CircuitNode]:
        return [n for n in self._nodes.values() if n.node_type == node_type]

    def output_nodes(self) -> list[CircuitNode]:
        return self.nodes_by_type("output_token")

    def input_nodes(self) -> list[CircuitNode]:
        return self.nodes_by_type("input_token")

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    @property
    def num_edges(self) -> int:
        return len(self._edges)

    @property
    def max_layer(self) -> int:
        return max((n.layer for n in self._nodes.values()), default=0)

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def prune(self, min_weight: float, keep_isolated: bool = True) -> AttributionGraph:
        """Remove edges with |weight| < *min_weight*, optionally dropping isolated nodes."""
        pruned = AttributionGraph(name=self.name)
        pruned.meta = dict(self.meta)
        for n in self._nodes.values():
            pruned.add_node(
                n.id,
                label=n.label,
                node_type=n.node_type,
                layer=n.layer,
                token_idx=n.token_idx,
                score=n.score,
                evidence_kind=n.evidence_kind,
                **n.meta,
            )
        for e in self._edges:
            if abs(e.weight) >= min_weight:
                pruned.add_edge(e.src, e.dst, weight=e.weight, method=e.method, **e.meta)
        if not keep_isolated:
            isolated = {
                nid for nid in pruned._nodes if not pruned._out[nid] and not pruned._in[nid]
            }
            for nid in isolated:
                del pruned._nodes[nid]
                del pruned._out[nid]
                del pruned._in[nid]
        return pruned

    # ------------------------------------------------------------------
    # Indirect / transitive influence
    # ------------------------------------------------------------------

    def indirect_influence(self, src: int, dst: int) -> float:
        """Signed strength of the strongest *multi-hop* path from src→dst.

        Direct (one-hop) edges are excluded — this is the indirect effect that
        flows through intermediate nodes, matching the paper's notion of an
        indirect contribution.  The strength of a path is the product of its
        edge weights; we return the product of the path with the largest
        absolute value.  Returns 0.0 if there is no path of length ≥ 2.
        """
        best = 0.0
        # DFS over acyclic paths (graphs here are layered DAGs), excluding the
        # direct edge so only transitive influence is counted.
        stack: list[tuple[int, float, frozenset[int]]] = [(src, 1.0, frozenset({src}))]
        while stack:
            node, prod, seen = stack.pop()
            for e in self._out.get(node, []):
                if e.dst in seen:
                    continue
                new_prod = prod * e.weight
                # Edges traversed to reach e.dst = len(seen) (seen has src + every
                # intermediate node; the current edge is the len(seen)-th). >= 2
                # means this is at least a two-hop path → genuinely indirect.
                edges_to_dst = len(seen)
                if e.dst == dst:
                    if edges_to_dst >= 2 and abs(new_prod) > abs(best):
                        best = new_prod
                    # do not traverse through the sink
                else:
                    stack.append((e.dst, new_prod, seen | {e.dst}))
        return float(best)

    def indirect_edges(self, min_weight: float = 0.0) -> list[tuple[int, int, float]]:
        """Return ``(src, dst, influence)`` for node pairs connected only
        *transitively* (no direct edge) whose indirect influence ≥ ``min_weight``."""
        direct = {(e.src, e.dst) for e in self._edges}
        out: list[tuple[int, int, float]] = []
        node_ids = list(self._nodes)
        for s in node_ids:
            for d in node_ids:
                if s == d or (s, d) in direct:
                    continue
                infl = self.indirect_influence(s, d)
                if abs(infl) >= min_weight and infl != 0.0:
                    out.append((s, d, infl))
        out.sort(key=lambda t: abs(t[2]), reverse=True)
        return out

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    def top_paths(
        self,
        n: int = 5,
        source: int | None = None,
        sink: int | None = None,
    ) -> list[list[int]]:
        """Return the top-*n* highest-weight paths from a source to a sink.

        Uses a best-first search over ``-log(|weight| + ε)`` so the
        cumulative cost on each path is the negative log of the product
        of |weights|.  Cycles are forbidden by tracking visited nodes
        per-path.

        When *source* is None we pick any input-token node (or, failing that,
        the node with no predecessors).  When *sink* is None we accept any
        ``output_token`` node, falling back to any sink (no outgoing edges).
        """
        if not self._nodes:
            return []

        sources = self._pick_sources(source)
        sinks = self._pick_sinks(sink)
        if not sources or not sinks:
            return []

        results: list[list[int]] = []
        heap: list[tuple[float, list[int]]] = []
        for s in sources:
            heapq.heappush(heap, (0.0, [s]))

        sink_set = set(sinks)
        while heap and len(results) < n:
            cost, path = heapq.heappop(heap)
            head = path[-1]
            if head in sink_set:
                results.append(path)
                continue
            for edge in self._out.get(head, []):
                if edge.dst in path:
                    continue
                w = abs(edge.weight)
                step_cost = -math.log(w + 1e-12)
                heapq.heappush(heap, (cost + step_cost, path + [edge.dst]))
        return results

    def _pick_sources(self, source: int | None) -> list[int]:
        if source is not None:
            return [source] if source in self._nodes else []
        inputs = [n.id for n in self._nodes.values() if n.node_type == "input_token"]
        if inputs:
            return inputs
        return [nid for nid in self._nodes if not self._in.get(nid)]

    def _pick_sinks(self, sink: int | None) -> list[int]:
        if sink is not None:
            return [sink] if sink in self._nodes else []
        outputs = [n.id for n in self._nodes.values() if n.node_type == "output_token"]
        if outputs:
            return outputs
        return [nid for nid in self._nodes if not self._out.get(nid)]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "meta": self.meta,
            "nodes": [n.as_dict() for n in self._nodes.values()],
            "edges": [e.as_dict() for e in self._edges],
        }

    def to_json(self, path: str | Path | None = None, indent: int = 2) -> str | None:
        text = json.dumps(self.to_dict(), indent=indent)
        if path is None:
            return text
        Path(path).write_text(text)
        return None

    def to_csv(self, path: str | Path) -> None:
        """Write edges to a CSV (``src,dst,weight,polarity,method``)."""
        with open(Path(path), "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["src", "dst", "weight", "polarity", "method"])
            writer.writeheader()
            for e in self._edges:
                writer.writerow(
                    {
                        "src": e.src,
                        "dst": e.dst,
                        "weight": e.weight,
                        "polarity": e.polarity,
                        "method": e.method,
                    }
                )

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"AttributionGraph(name={self.name!r}, nodes={self.num_nodes}, edges={self.num_edges})"
        )
