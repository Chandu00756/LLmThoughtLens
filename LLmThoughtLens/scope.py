"""Scope — top-level entry point for every LLmThoughtLens workflow.

The :class:`Scope` glues a provider together with the feature extractor,
circuit tracer, supernode grouper, and probe runner.  Users typically reach
for ``Scope.from_openai(...)`` / ``Scope.from_huggingface(...)`` etc., then
call :meth:`trace_full` to receive a :class:`TraceResult` that knows how to
render itself as Plotly figures or a self-contained HTML report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from LLmThoughtLens.providers.base import BaseProvider, ProviderOutput

if TYPE_CHECKING:
    from LLmThoughtLens.circuits.graph import AttributionGraph
    from LLmThoughtLens.features.feature import Feature, FeatureSet
    from LLmThoughtLens.features.intervention import FeatureIntervention
    from LLmThoughtLens.features.sae import SparseAutoencoder
    from LLmThoughtLens.probes.base import BaseProbe, ProbeResult


# ---------------------------------------------------------------------------
# TraceResult
# ---------------------------------------------------------------------------


@dataclass
class TraceResult:
    """Everything the trace pipeline produced for one prompt."""

    prompt: str
    output: ProviderOutput
    features: list[Feature] = field(default_factory=list)
    supernodes: list[FeatureSet] = field(default_factory=list)
    graph: AttributionGraph = field(default_factory=lambda: _empty_graph())
    probe_results: list[ProbeResult] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience getters
    # ------------------------------------------------------------------

    @property
    def output_token(self) -> str:
        return self.output.output_token

    @property
    def top_tokens(self) -> list[tuple[str, float]]:
        return self.output.top_tokens

    @property
    def evidence_kind(self) -> str:
        return self.output.evidence_kind

    def top_features(self, n: int = 5) -> list[Feature]:
        return sorted(self.features, key=lambda f: f.score, reverse=True)[:n]

    def top_paths(self, n: int = 3) -> list[list[int]]:
        return self.graph.top_paths(n=n)

    # ------------------------------------------------------------------
    # Visualisation entry points
    # ------------------------------------------------------------------

    def show(self) -> None:
        from LLmThoughtLens.visualization.graph_viz import GraphVisualizer

        GraphVisualizer(self.graph).to_figure().show()

    def show_heatmap(self) -> None:
        from LLmThoughtLens.visualization.token_heatmap import TokenHeatmap

        TokenHeatmap(self.output, self.features).to_figure().show()

    def show_residual_stream(self) -> None:
        from LLmThoughtLens.visualization.layer_stream import ResidualStreamView

        ResidualStreamView(self.output).to_figure().show()

    def browse_features(self) -> str:
        """Return the searchable-table HTML (useful in notebooks)."""
        from LLmThoughtLens.visualization.feature_browser import FeatureBrowser

        return FeatureBrowser(self.features).to_html()

    # ------------------------------------------------------------------
    # Exports
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Write the full tabbed HTML report to *path*."""
        from LLmThoughtLens.visualization.report import ReportBuilder

        ReportBuilder.from_trace_result(self).save(path)

    def save_graph_json(self, path: str | Path) -> None:
        self.graph.to_json(path)

    def save_graph_csv(self, path: str | Path) -> None:
        self.graph.to_csv(path)

    def save_features_csv(self, path: str | Path) -> None:
        import csv as _csv

        with open(Path(path), "w", newline="") as fh:
            writer = _csv.DictWriter(
                fh,
                fieldnames=[
                    "id",
                    "label",
                    "layer",
                    "token_idx",
                    "score",
                    "node_type",
                    "evidence_kind",
                ],
            )
            writer.writeheader()
            for f in self.features:
                writer.writerow(f.as_dict())

    def __repr__(self) -> str:
        return (
            f"TraceResult(prompt={self.prompt[:40]!r}, "
            f"output_token={self.output_token!r}, "
            f"features={len(self.features)}, "
            f"probes={len(self.probe_results)}, "
            f"evidence={self.evidence_kind})"
        )


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------


class Scope:
    """High-level façade for running a single prompt through every layer.

    Create one via the ``from_*`` factories or by passing a provider
    directly::

        scope = Scope.from_mock()
        result = scope.trace_full("The capital of France is")
        result.save("report.html")
    """

    def __init__(
        self,
        provider: BaseProvider,
        *,
        top_k_features: int = 20,
        attribution_threshold: float = 0.05,
        use_supernodes: bool = True,
        blackbox_budget: int | None = 16,
    ) -> None:
        self._provider = provider
        self._top_k = int(top_k_features)
        self._threshold = float(attribution_threshold)
        self._use_supernodes = bool(use_supernodes)
        self._blackbox_budget = blackbox_budget
        self._extractor: Any = None
        self._tracer: Any = None
        self._grouper: Any = None
        self._sae: SparseAutoencoder | None = None
        self._sae_layer: int = -1

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_mock(cls, **kwargs: Any) -> Scope:
        from LLmThoughtLens.providers.mock_provider import MockProvider

        return cls(MockProvider(**kwargs))

    @classmethod
    def from_openai(
        cls,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        **kwargs: Any,
    ) -> Scope:
        from LLmThoughtLens.providers.openai_provider import OpenAIProvider

        return cls(OpenAIProvider(model=model, api_key=api_key, **kwargs))

    @classmethod
    def from_anthropic(
        cls,
        model: str = "claude-3-5-haiku-20241022",
        api_key: str | None = None,
        **kwargs: Any,
    ) -> Scope:
        from LLmThoughtLens.providers.anthropic_provider import AnthropicProvider

        return cls(AnthropicProvider(model=model, api_key=api_key, **kwargs))

    @classmethod
    def from_huggingface(
        cls,
        model_name: str = "gpt2",
        device: str = "auto",
        **kwargs: Any,
    ) -> Scope:
        from LLmThoughtLens.providers.huggingface_provider import HuggingFaceProvider

        return cls(HuggingFaceProvider(model_name=model_name, device=device, **kwargs))

    @classmethod
    def from_ollama(
        cls,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        **kwargs: Any,
    ) -> Scope:
        from LLmThoughtLens.providers.ollama_provider import OllamaProvider

        return cls(OllamaProvider(model=model, base_url=base_url, **kwargs))

    @classmethod
    def from_provider(cls, provider: BaseProvider, **kwargs: Any) -> Scope:
        return cls(provider, **kwargs)

    # ------------------------------------------------------------------
    # Core: raw provider call
    # ------------------------------------------------------------------

    def trace(self, prompt: str, **kwargs: Any) -> ProviderOutput:
        """Forward *prompt* through the provider and return the raw envelope."""
        return self._provider.run(prompt, **kwargs)

    # ------------------------------------------------------------------
    # Full interpretability pipeline
    # ------------------------------------------------------------------

    def trace_full(
        self,
        prompt: str,
        *,
        top_k_features: int | None = None,
        attribution_threshold: float | None = None,
        use_supernodes: bool | None = None,
        run_probes: bool = False,
        interventions: list[FeatureIntervention] | None = None,
        **kwargs: Any,
    ) -> TraceResult:
        """Run the full interpretability pipeline on *prompt*."""
        from LLmThoughtLens.circuits.supernodes import SupernodeGrouper
        from LLmThoughtLens.circuits.tracer import CircuitTracer
        from LLmThoughtLens.features.extractor import FeatureExtractor

        k = top_k_features if top_k_features is not None else self._top_k
        threshold = attribution_threshold if attribution_threshold is not None else self._threshold
        use_super = self._use_supernodes if use_supernodes is None else bool(use_supernodes)

        # 1. Provider forward (with optional interventions)
        if interventions:
            output = self._provider.run_with_intervention(
                prompt, interventions=interventions, **kwargs
            )
        else:
            output = self._provider.run(prompt, **kwargs)

        # 2. Feature extraction
        extractor = self._extractor
        if extractor is None:
            extractor = FeatureExtractor(top_k=k, blackbox_budget=self._blackbox_budget)
            if self._sae is not None and self._sae_layer >= 0:
                extractor.attach_sae(self._sae, self._sae_layer)
        features = extractor.extract(output, provider=self._provider)

        # 3. Attribution graph
        tracer = self._tracer or CircuitTracer(min_weight=threshold)
        graph = tracer.trace(output, features, provider=self._provider)
        if threshold > 0:
            graph = graph.prune(threshold, keep_isolated=True)

        # 4. Supernodes
        supernodes: list = []
        if use_super:
            grouper = self._grouper or SupernodeGrouper(sae=self._sae)
            supernodes = grouper.group(features, output)

        # 5. Probes
        probe_results: list[ProbeResult] = []
        if run_probes:
            from LLmThoughtLens.probes.builtin import all_probes
            from LLmThoughtLens.probes.runner import ProbeRunner

            probe_results = ProbeRunner(all_probes()).run_all(self._provider).results

        return TraceResult(
            prompt=prompt,
            output=output,
            features=features,
            supernodes=supernodes,
            graph=graph,
            probe_results=probe_results,
            meta={
                "provider": self._provider.name,
                "model": self._provider.model_id,
                "evidence_kind": output.evidence_kind,
            },
        )

    # ------------------------------------------------------------------
    # Convenience: one-shot HTML report
    # ------------------------------------------------------------------

    def report(
        self,
        prompt: str,
        output: str | Path = "report.html",
        run_probes: bool = True,
        **kwargs: Any,
    ) -> TraceResult:
        """Trace *prompt* and write a self-contained HTML report to *output*."""
        result = self.trace_full(prompt, run_probes=run_probes, **kwargs)
        result.save(output)
        return result

    # ------------------------------------------------------------------
    # Run a single probe
    # ------------------------------------------------------------------

    def run_probe(self, probe: BaseProbe, prompt: str | None = None) -> ProbeResult:
        return probe.run(self._provider, prompt=prompt)

    # ------------------------------------------------------------------
    # Component attachment
    # ------------------------------------------------------------------

    def attach_sae(self, sae: SparseAutoencoder, layer: int) -> None:
        """Use a trained SAE for white-box feature extraction at *layer*."""
        from LLmThoughtLens.features.extractor import FeatureExtractor

        self._sae = sae
        self._sae_layer = int(layer)
        if self._extractor is None:
            self._extractor = FeatureExtractor(
                top_k=self._top_k, blackbox_budget=self._blackbox_budget
            )
        self._extractor.attach_sae(sae, layer)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def provider(self) -> BaseProvider:
        return self._provider

    @property
    def sae(self) -> SparseAutoencoder | None:
        return self._sae

    def __repr__(self) -> str:
        return (
            f"Scope(provider={self._provider!r}, top_k={self._top_k}, "
            f"threshold={self._threshold}, sae={'yes' if self._sae else 'no'})"
        )


# ---------------------------------------------------------------------------
# Deferred default for TraceResult.graph
# ---------------------------------------------------------------------------


def _empty_graph() -> AttributionGraph:
    from LLmThoughtLens.circuits.graph import AttributionGraph

    return AttributionGraph(name="(empty)")
