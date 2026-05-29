"""LLmThoughtLens — platform-agnostic LLM interpretability toolkit.

ThoughtLens operationalises Anthropic's March 2025 Circuit Tracing /
Biology of an LLM research as an installable Python package that works on
closed-source API models (GPT-4o, Claude) and locally-hosted weights
(Llama, Mistral, Phi).

Public API
----------
    Scope             — entry point: from_openai / from_anthropic /
                        from_huggingface / from_ollama / from_mock / from_provider.
    TraceResult       — rich envelope returned by Scope.trace_full.
    Feature           — one interpretable feature in the SAE dictionary.
    FeatureSet        — supernode: a cluster of related features.
    SparseAutoencoder — TopK SAE with real PyTorch training.
    SAEConfig         — SAE hyperparameters.
    ActivationCache   — collect per-token residual-stream activations.
    FeatureLabeler    — LLM-driven feature auto-labelling.
    FeatureExtractor  — real white-box (SAE/L2) + black-box (masking).
    FeatureIntervention — clamp / amplify / inhibit a feature mid-inference.
    AttributionGraph  — directed causal graph of features.
    CircuitNode/CircuitEdge — typed graph primitives.
    CircuitTracer     — build attribution graphs from real activations / perturbations.
    SupernodeGrouper  — cluster features by SAE-direction or activation similarity.
    GraphDiff         — compare two attribution graphs.
    BaseProbe         — interface for custom probes.
    ProbeResult       — structured probe output.
    ProbeRunner       — execute a battery of probes.
    ReportBuilder     — five-tab self-contained HTML report.
"""

from __future__ import annotations

from LLmThoughtLens.circuits.diff import GraphDiff
from LLmThoughtLens.circuits.graph import (
    AttributionGraph,
    CircuitEdge,
    CircuitNode,
)
from LLmThoughtLens.circuits.supernodes import SupernodeGrouper
from LLmThoughtLens.circuits.tracer import CircuitTracer
from LLmThoughtLens.features.cache import ActivationCache
from LLmThoughtLens.features.extractor import FeatureExtractor
from LLmThoughtLens.features.feature import Feature, FeatureSet
from LLmThoughtLens.features.intervention import FeatureIntervention
from LLmThoughtLens.features.labeler import FeatureLabeler
from LLmThoughtLens.features.sae import SAEConfig, SparseAutoencoder
from LLmThoughtLens.probes.base import BaseProbe, ProbeResult
from LLmThoughtLens.probes.runner import ProbeRunner
from LLmThoughtLens.scope import Scope, TraceResult
from LLmThoughtLens.visualization.report import ReportBuilder

__all__ = [
    "Scope",
    "TraceResult",
    "Feature",
    "FeatureSet",
    "FeatureExtractor",
    "FeatureIntervention",
    "FeatureLabeler",
    "SparseAutoencoder",
    "SAEConfig",
    "ActivationCache",
    "AttributionGraph",
    "CircuitNode",
    "CircuitEdge",
    "CircuitTracer",
    "SupernodeGrouper",
    "GraphDiff",
    "BaseProbe",
    "ProbeResult",
    "ProbeRunner",
    "ReportBuilder",
]

__version__ = "0.1.0"
