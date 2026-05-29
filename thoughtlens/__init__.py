"""ThoughtLens — platform-agnostic LLM interpretability toolkit.

ThoughtLens operationalises Anthropic's March 2025 Circuit Tracing /
On the Biology of an LLM research as an installable Python package that
works on closed-source API models (GPT-4o, Claude) and locally-hosted
weights (Llama, Mistral, Phi) alike.

Public API
----------
    Scope             — top-level entry point: from_openai / from_anthropic /
                        from_huggingface / from_ollama / from_mock / from_provider.
    TraceResult       — rich envelope returned by Scope.trace_full.
    Feature           — one interpretable feature in the SAE dictionary.
    FeatureSet        — supernode: a cluster of related features.
    AttributionGraph  — directed causal graph of features.
    BaseProbe         — interface for custom probes.
    ProbeResult       — structured probe output.
    FeatureIntervention — clamp / amplify / inhibit a feature mid-inference.
"""

from __future__ import annotations

from thoughtlens.circuits.graph import AttributionGraph, CircuitEdge, CircuitNode
from thoughtlens.features.feature import Feature, FeatureSet
from thoughtlens.features.intervention import FeatureIntervention
from thoughtlens.probes.base import BaseProbe, ProbeResult
from thoughtlens.scope import Scope, TraceResult

__all__ = [
    "Scope",
    "TraceResult",
    "Feature",
    "FeatureSet",
    "AttributionGraph",
    "CircuitNode",
    "CircuitEdge",
    "BaseProbe",
    "ProbeResult",
    "FeatureIntervention",
]

__version__ = "0.1.0"
