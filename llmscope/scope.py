"""Scope — main entry point for LLM interpretability workflows in llmscope."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from llmscope.providers.base import BaseProvider, ProviderOutput

if TYPE_CHECKING:
    from llmscope.circuits.graph import AttributionGraph
    from llmscope.features.feature import Feature, FeatureSet
    from llmscope.features.intervention import FeatureIntervention
    from llmscope.probes.builtin import ProviderProbe
    from llmscope.probes.runner import ProbeResult


# ---------------------------------------------------------------------------
# TraceResult — rich return value from Scope.trace_full()
# ---------------------------------------------------------------------------


@dataclass
class TraceResult:
    """Rich result returned by :meth:`Scope.trace_full`.

    Attributes
    ----------
    prompt:
        Original input prompt.
    output:
        Raw :class:`~llmscope.providers.base.ProviderOutput` from the provider.
    features:
        Top-k interpretable features extracted from the output.
    supernodes:
        Features grouped into higher-level supernode clusters.
    graph:
        :class:`~llmscope.circuits.graph.AttributionGraph` built from features.
    probe_results:
        Results from any probes that were run.
    meta:
        Freeform metadata dict.
    """

    prompt: str
    output: ProviderOutput
    features: list[Feature] = field(default_factory=list)
    supernodes: list[FeatureSet] = field(default_factory=list)
    graph: AttributionGraph = field(default_factory=lambda: _empty_graph())
    probe_results: list[ProbeResult] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def output_token(self) -> str:
        """Top-predicted next token."""
        return self.output.top_tokens[0][0] if self.output.top_tokens else ""

    @property
    def top_tokens(self) -> list[tuple[str, float]]:
        """Top predicted tokens and their probabilities."""
        return self.output.top_tokens

    def top_features(self, n: int = 5) -> list[Feature]:
        """Return the top-*n* features by activation score."""
        return sorted(self.features, key=lambda f: f.score, reverse=True)[:n]

    def top_paths(self, n: int = 3) -> list[list[int]]:
        """Return the top-*n* causal paths through the attribution graph."""
        return self.graph.top_paths(n=n)

    def show(self) -> None:
        """Open the attribution graph in the default browser."""
        from llmscope.visualization.graph_viz import GraphVisualizer

        fig = GraphVisualizer(self.graph).to_figure()
        fig.show()

    def show_heatmap(self) -> None:
        """Open the token feature heatmap in the default browser."""
        from llmscope.visualization.token_heatmap import TokenHeatmap

        fig = TokenHeatmap(self.output, self.features).to_figure()
        fig.show()

    def save(self, path: str) -> None:
        """Generate and save the full tabbed HTML report.

        Parameters
        ----------
        path:
            Output file path (e.g. ``report.html``).
        """
        from llmscope.visualization.report import ReportBuilder

        builder = ReportBuilder.from_trace_result(self)
        builder.save(path)

    def __repr__(self) -> str:
        return (
            f"TraceResult(prompt={self.prompt[:40]!r}, "
            f"output_token={self.output_token!r}, "
            f"features={len(self.features)}, "
            f"probes={len(self.probe_results)})"
        )


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------


class Scope:
    """High-level façade that wires together a provider, extractor, tracer, and probes.

    Create a :class:`Scope` using one of the class-method factories::

        scope = Scope.from_mock()
        result = scope.trace_full("The capital of France is")
        result.show()
        result.save("report.html")

    Parameters
    ----------
    provider:
        The backend that executes prompts and returns activations.
    top_k_features:
        Default number of features to extract per trace.
    attribution_threshold:
        Default minimum edge weight for the attribution graph.
    """

    def __init__(
        self,
        provider: BaseProvider,
        top_k_features: int = 20,
        attribution_threshold: float = 0.1,
    ) -> None:
        self._provider = provider
        self._top_k = top_k_features
        self._threshold = attribution_threshold
        self._extractor: FeatureExtractor | None = None
        self._tracer: CircuitTracer | None = None
        self._grouper: SupernodeGrouper | None = None

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_mock(cls, **kwargs: Any) -> "Scope":
        """Create a Scope backed by the deterministic :class:`~llmscope.providers.mock_provider.MockProvider`.

        Parameters
        ----------
        **kwargs:
            Forwarded to ``MockProvider.__init__`` (e.g. ``seed=0``, ``n_layers=4``).
        """
        from llmscope.providers.mock_provider import MockProvider

        return cls(MockProvider(**kwargs))

    @classmethod
    def from_openai(cls, api_key: str | None = None, model: str = "gpt-4o-mini") -> "Scope":
        """Create a Scope backed by the OpenAI Chat Completions API."""
        from llmscope.providers.openai_provider import OpenAIProvider

        return cls(OpenAIProvider(api_key=api_key, model=model))

    @classmethod
    def from_anthropic(
        cls,
        api_key: str | None = None,
        model: str = "claude-3-5-haiku-20241022",
    ) -> "Scope":
        """Create a Scope backed by the Anthropic Messages API."""
        from llmscope.providers.anthropic_provider import AnthropicProvider

        return cls(AnthropicProvider(api_key=api_key, model=model))

    @classmethod
    def from_huggingface(cls, model_name: str = "gpt2", device: str = "cpu") -> "Scope":
        """Create a Scope backed by a local HuggingFace model."""
        from llmscope.providers.huggingface_provider import HuggingFaceProvider

        return cls(HuggingFaceProvider(model_name=model_name, device=device))

    @classmethod
    def from_ollama(cls, model: str = "llama3.2", base_url: str = "http://localhost:11434") -> "Scope":
        """Create a Scope backed by a locally-running Ollama instance."""
        from llmscope.providers.ollama_provider import OllamaProvider

        return cls(OllamaProvider(model=model, base_url=base_url))

    @classmethod
    def from_provider(cls, provider: BaseProvider, **kwargs: Any) -> "Scope":
        """Create a Scope from any custom :class:`~llmscope.providers.base.BaseProvider`."""
        return cls(provider, **kwargs)

    # ------------------------------------------------------------------
    # Core: simple trace (raw ProviderOutput)
    # ------------------------------------------------------------------

    def trace(self, prompt: str, **kwargs: Any) -> ProviderOutput:
        """Run the prompt through the provider and return the raw output.

        Parameters
        ----------
        prompt:
            The input text to forward to the model.
        **kwargs:
            Provider-specific options forwarded to :meth:`~BaseProvider.run`.

        Returns
        -------
        ProviderOutput
            Structured envelope containing tokens, activations, attentions,
            logits, and metadata.
        """
        return self._provider.run(prompt, **kwargs)

    # ------------------------------------------------------------------
    # Full interpretability trace
    # ------------------------------------------------------------------

    def trace_full(
        self,
        prompt: str,
        top_k_features: int | None = None,
        run_probes: bool = False,
        attribution_threshold: float | None = None,
        use_supernodes: bool = True,
        **kwargs: Any,
    ) -> TraceResult:
        """Run the full interpretability pipeline on *prompt*.

        Steps:
        1. Forward the prompt through the provider.
        2. Extract top-k features (white-box or black-box).
        3. Build an attribution graph (with optional pruning).
        4. Group features into supernodes.
        5. Optionally run all 5 built-in probes.

        Parameters
        ----------
        prompt:
            Input text.
        top_k_features:
            Number of features to extract (overrides instance default).
        run_probes:
            Whether to run all 5 built-in probes.
        attribution_threshold:
            Minimum edge weight (overrides instance default).
        use_supernodes:
            Whether to group features into supernode clusters.
        **kwargs:
            Forwarded to the provider's ``run()`` call.

        Returns
        -------
        TraceResult
        """
        from llmscope.circuits.graph import AttributionGraph
        from llmscope.circuits.tracer import CircuitTracer
        from llmscope.circuits.supernodes import SupernodeGrouper
        from llmscope.features.extractor import FeatureExtractor

        k = top_k_features or self._top_k
        threshold = attribution_threshold if attribution_threshold is not None else self._threshold

        # Step 1: forward pass
        output = self._provider.run(prompt, **kwargs)

        # Step 2: feature extraction
        extractor = FeatureExtractor(top_k=k)
        if self._extractor is not None:
            extractor = self._extractor
        features = extractor.extract(output)

        # Step 3: attribution graph
        tracer = self._tracer or CircuitTracer(min_weight=threshold)
        graph = tracer.trace(output, features)

        # Prune weak edges
        if threshold > 0:
            graph = graph.prune(threshold)

        # Step 4: supernodes
        supernodes = []
        if use_supernodes:
            grouper = self._grouper or SupernodeGrouper()
            supernodes = grouper.group(features, output)

        # Step 5: probes
        probe_results: list[ProbeResult] = []
        if run_probes:
            from llmscope.probes.builtin import all_probes

            for probe in all_probes():
                try:
                    pr = probe.run(self._provider)
                    probe_results.append(pr)
                except Exception as exc:
                    from llmscope.probes.runner import ProbeResult

                    probe_results.append(
                        ProbeResult(
                            probe_name=probe.name,
                            score=0.0,
                            meta={"error": str(exc)},
                        )
                    )

        return TraceResult(
            prompt=prompt,
            output=output,
            features=features,
            supernodes=supernodes,
            graph=graph,
            probe_results=probe_results,
            meta={"provider": self._provider.name},
        )

    # ------------------------------------------------------------------
    # Convenience: one-shot HTML report
    # ------------------------------------------------------------------

    def report(
        self,
        prompt: str,
        output: str = "report.html",
        run_probes: bool = True,
        **kwargs: Any,
    ) -> "TraceResult":
        """Trace *prompt* and save a self-contained HTML report to *output*.

        Parameters
        ----------
        prompt:
            Input text.
        output:
            Path to write the HTML report.
        run_probes:
            Whether to include probe results in the report.
        **kwargs:
            Forwarded to :meth:`trace_full`.

        Returns
        -------
        TraceResult
        """
        result = self.trace_full(prompt, run_probes=run_probes, **kwargs)
        result.save(output)
        return result

    # ------------------------------------------------------------------
    # Run a single probe
    # ------------------------------------------------------------------

    def run_probe(self, probe: "ProviderProbe", prompt: str | None = None) -> "ProbeResult":
        """Run a single probe against this scope's provider.

        Parameters
        ----------
        probe:
            Any :class:`~llmscope.probes.builtin.ProviderProbe` instance.
        prompt:
            Optional prompt override.

        Returns
        -------
        ProbeResult
        """
        return probe.run(self._provider, prompt=prompt)

    # ------------------------------------------------------------------
    # SAE attachment
    # ------------------------------------------------------------------

    def attach_sae(self, sae: Any, layer: int) -> None:
        """Attach a trained SAE for white-box feature extraction.

        Parameters
        ----------
        sae:
            Trained :class:`~llmscope.features.sae.SparseAutoencoder`.
        layer:
            The transformer layer the SAE was trained on.
        """
        from llmscope.features.extractor import FeatureExtractor

        if self._extractor is None:
            self._extractor = FeatureExtractor(top_k=self._top_k)
        self._extractor.attach_sae(sae, layer)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def provider(self) -> BaseProvider:
        """The underlying provider instance."""
        return self._provider

    def __repr__(self) -> str:
        return f"Scope(provider={self._provider!r})"


# ---------------------------------------------------------------------------
# Deferred imports for type annotations inside TraceResult
# ---------------------------------------------------------------------------


def _empty_graph() -> "AttributionGraph":
    from llmscope.circuits.graph import AttributionGraph

    return AttributionGraph()


# ---------------------------------------------------------------------------
# Type aliases (resolved at runtime to avoid circular imports at load time)
# ---------------------------------------------------------------------------

FeatureExtractor = None  # type: ignore[assignment,misc]
CircuitTracer = None  # type: ignore[assignment,misc]
SupernodeGrouper = None  # type: ignore[assignment,misc]
