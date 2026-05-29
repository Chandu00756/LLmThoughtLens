"""Circuits layer — AttributionGraph, CircuitTracer, supernodes, paths, diff."""

from LLmThoughtLens.circuits.diff import GraphDiff
from LLmThoughtLens.circuits.graph import (
    AttributionGraph,
    CircuitEdge,
    CircuitNode,
    Edge,  # backwards-compatibility alias
    EdgePolarity,
    NodeType,
)
from LLmThoughtLens.circuits.paths import CausalPath, label_path, top_causal_paths
from LLmThoughtLens.circuits.supernodes import SupernodeGrouper
from LLmThoughtLens.circuits.tracer import CircuitTracer

__all__ = [
    "AttributionGraph",
    "CircuitNode",
    "CircuitEdge",
    "Edge",
    "EdgePolarity",
    "NodeType",
    "CircuitTracer",
    "SupernodeGrouper",
    "CausalPath",
    "top_causal_paths",
    "label_path",
    "GraphDiff",
]
