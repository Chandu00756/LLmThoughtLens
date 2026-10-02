"""Circuits layer — AttributionGraph, CircuitTracer, gradient attribution, patching, diff.

Importing this package never imports torch: :mod:`.attribution` and
:mod:`.patching` load it lazily when a white-box model is actually used.
"""

from LLmThoughtLens.circuits.attribution import (
    AttributionNode,
    AttributionResult,
    GradientAttributor,
    MetricSpec,
)
from LLmThoughtLens.circuits.diff import GraphDiff
from LLmThoughtLens.circuits.graph import (
    AttributionGraph,
    CircuitEdge,
    CircuitNode,
    Edge,  # backwards-compatibility alias
    EdgePolarity,
    NodeType,
)
from LLmThoughtLens.circuits.patching import (
    ActivationPatcher,
    FaithfulnessReport,
    attribution_faithfulness,
)
from LLmThoughtLens.circuits.paths import CausalPath, label_path, top_causal_paths
from LLmThoughtLens.circuits.supernodes import SupernodeGrouper
from LLmThoughtLens.circuits.tracer import TRACE_METHODS, CircuitTracer

__all__ = [
    "AttributionGraph",
    "CircuitNode",
    "CircuitEdge",
    "Edge",
    "EdgePolarity",
    "NodeType",
    "CircuitTracer",
    "TRACE_METHODS",
    "GradientAttributor",
    "AttributionNode",
    "AttributionResult",
    "MetricSpec",
    "ActivationPatcher",
    "FaithfulnessReport",
    "attribution_faithfulness",
    "SupernodeGrouper",
    "CausalPath",
    "top_causal_paths",
    "label_path",
    "GraphDiff",
]
