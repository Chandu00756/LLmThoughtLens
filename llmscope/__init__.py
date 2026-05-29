"""
llmscope — Platform-agnostic LLM interpretability toolkit.

Public API surface:
    Scope           — main entry point for tracing and probing
    Feature         — an interpretable sparse feature
    FeatureSet      — a named collection of features (supernode)
    AttributionGraph — directed causal graph of features
    BaseProbe       — base class for custom probes
    ProbeResult     — structured result returned by a probe
"""

from llmscope.scope import Scope
from llmscope.features.feature import Feature, FeatureSet
from llmscope.circuits.graph import AttributionGraph
from llmscope.probes.base import BaseProbe
from llmscope.probes.runner import ProbeResult

__all__ = [
    "Scope",
    "Feature",
    "FeatureSet",
    "AttributionGraph",
    "BaseProbe",
    "ProbeResult",
]

__version__ = "0.1.0"
