"""Scope — top-level entry point for every LLmThoughtLens workflow.

The :class:`Scope` glues a provider together with the feature extractor,
circuit tracer, supernode grouper, and probe runner.  Users typically reach
for ``Scope.from_openai(...)`` / ``Scope.from_huggingface(...)`` etc., then
call :meth:`trace_full` to receive a :class:`TraceResult` that knows how to
render itself as Plotly figures or a self-contained HTML report.

Attribution defaults (user-facing layer)
----------------------------------------
:class:`~LLmThoughtLens.circuits.tracer.CircuitTracer` keeps its own class
defaults; the user-facing surfaces (``Scope``, the CLI, the dashboard, the
SDK) use :data:`ATTRIBUTION_DEFAULTS` instead, every one overridable per Scope
and per :meth:`Scope.trace_full` call:

* ``attribution="auto"`` — gradient attribution (``causal_linearised`` edges)
  when the provider exposes a differentiable model (HuggingFace), activation
  flow (``correlational``) for the mock, input masking for API models.
* ``metric="logprob"`` — the target metric is the log-probability of the
  predicted token, which removes the large common-mode logit offset (about
  -100 on GPT-2) that inflates attributions under ``"logit"``.
* ``attribution_nodes=10`` — the ten residual nodes with the largest
  attribution join the graph, so paths can run through the prediction
  position (on GPT-2 "...Dallas is" this cut uncovered attribution mass from
  0.90 to 0.19).
* ``validate=False`` — real-ablation faithfulness is opt-in (``validate=k``
  ablates the top-``k`` nodes); when it runs, Spearman, Pearson, ``n`` and
  sign agreement are reported together (:attr:`TraceResult.faithfulness`).

Interventions and attached SAEs are passed to the tracer, so intervened HF
traces and SAE-feature nodes get gradient edges too.

Steering and SAEs
-----------------
:meth:`Scope.load_sae` / :meth:`Scope.attach_saes` attach pretrained (SAELens
/ Gemma Scope) or trained SAEs; :meth:`Scope.steer`,
:meth:`Scope.steering_vector_from_contrast` and
:meth:`Scope.coefficient_sweep` run activation steering.  Both need a local
white-box model: black-box providers raise a clear ``ValueError`` /
:class:`~LLmThoughtLens.features.steering.SteeringUnavailableError`, and the
mock provider cannot be steered (its activations are synthetic).
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from LLmThoughtLens.providers.base import BaseProvider, ProviderOutput
from LLmThoughtLens.providers.defaults import default_ollama_url, resolve_model

if TYPE_CHECKING:
    from LLmThoughtLens.circuits.graph import AttributionGraph
    from LLmThoughtLens.features.extractor import FeatureExtractor, WhiteboxScoring
    from LLmThoughtLens.features.feature import Feature, FeatureSet
    from LLmThoughtLens.features.intervention import FeatureIntervention
    from LLmThoughtLens.features.sae import SparseAutoencoder
    from LLmThoughtLens.features.steering import SteeringResult, SteeringVector, SweepResult
    from LLmThoughtLens.models.hooked import HookedModel
    from LLmThoughtLens.probes.base import BaseProbe, ProbeResult


#: User-facing attribution defaults (CircuitTracer's own class defaults differ).
ATTRIBUTION_DEFAULTS: dict[str, Any] = {
    "attribution": "auto",
    "metric": "logprob",
    "attribution_nodes": 10,
    "validate": False,
    "node_kind": "delta",
    "baseline": "zero",
    "max_edge_targets": None,
}

#: Keyword arguments of :class:`Scope` (``from_*`` factories route these to the
#: Scope instead of the provider constructor).
SCOPE_OPTIONS: frozenset[str] = frozenset(
    {
        "top_k_features",
        "attribution_threshold",
        "use_supernodes",
        "blackbox_budget",
        "scoring",
        "exclude_outlier_positions",
        "exclude_positions",
        *ATTRIBUTION_DEFAULTS,
    }
)


def _split_scope_options(kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(scope options, provider kwargs)`` from one ``**kwargs`` dict."""
    scope_kw = {k: v for k, v in kwargs.items() if k in SCOPE_OPTIONS}
    provider_kw = {k: v for k, v in kwargs.items() if k not in SCOPE_OPTIONS}
    return scope_kw, provider_kw


def _validate_attribution_options(
    attribution: str | None,
    metric: str | None,
    node_kind: str | None,
    baseline: str | None,
    attribution_nodes: int | None,
    validate: bool | int | None,
) -> None:
    from LLmThoughtLens.circuits.attribution import BASELINES, METRICS, NODE_KINDS
    from LLmThoughtLens.circuits.tracer import TRACE_METHODS

    for name, value, allowed in (
        ("attribution", attribution, TRACE_METHODS),
        ("metric", metric, METRICS),
        ("node_kind", node_kind, NODE_KINDS),
        ("baseline", baseline, BASELINES),
    ):
        if value is not None and value not in allowed:
            raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
    if attribution_nodes is not None and int(attribution_nodes) < 0:
        raise ValueError(f"attribution_nodes must be >= 0, got {attribution_nodes}")
    if validate is not None and not isinstance(validate, bool) and int(validate) < 0:
        raise ValueError(f"validate must be False, True or an int >= 0, got {validate}")


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

    @property
    def input_tokens(self) -> list[str]:
        """The prompt tokens ``feature.token_idx`` indexes into.

        White-box: the provider's own tokens.  Black-box: the whitespace-split
        prompt (``output.tokens`` is the completion there).
        """
        from LLmThoughtLens.utils.tokenizer_utils import whitespace_tokens

        if self.output.has_internals:
            return list(self.output.tokens)
        return whitespace_tokens(self.prompt) or list(self.output.tokens)

    def top_features(self, n: int = 5) -> list[Feature]:
        return sorted(self.features, key=lambda f: f.score, reverse=True)[:n]

    def top_paths(self, n: int = 3) -> list[list[int]]:
        return self.graph.top_paths(n=n)

    # ------------------------------------------------------------------
    # Attribution method / faithfulness (read from graph.meta)
    # ------------------------------------------------------------------

    @property
    def attribution_method(self) -> str | None:
        """``"gradient"`` / ``"activation_flow"`` / ``"mask_perturbation"`` (``None`` if untraced)."""
        value = self.graph.meta.get("attribution_method")
        return str(value) if value is not None else None

    @property
    def edge_semantics(self) -> str | None:
        """``"causal_linearised"`` / ``"correlational"`` / ``"causal_input_masking"``."""
        value = self.graph.meta.get("edge_semantics")
        return str(value) if value is not None else None

    @property
    def method_fallback(self) -> str | None:
        """Why gradient attribution was not used although the provider supports it."""
        value = self.graph.meta.get("method_fallback")
        return str(value) if value is not None else None

    @property
    def faithfulness(self) -> dict[str, Any] | None:
        """Real-ablation check of the gradient attributions (``validate=k``), else ``None``.

        Spearman, Pearson, ``n`` and sign agreement always come together, with
        a plain-language caveat (see
        :data:`~LLmThoughtLens.visualization.attribution_view.FAITHFULNESS_CAVEAT`).
        """
        return self.attribution_summary()["faithfulness"]

    def attribution_summary(self) -> dict[str, Any]:
        """JSON-safe description of the attribution method, edge semantics and faithfulness."""
        from LLmThoughtLens.visualization.attribution_view import attribution_summary

        return attribution_summary(self.graph)

    # ------------------------------------------------------------------
    # JSON payload for the live dashboard / API
    # ------------------------------------------------------------------

    def to_payload(self, max_features: int = 200) -> dict[str, Any]:
        """Compose a JSON-safe dict from the existing per-object serializers.

        Reuses :meth:`ProviderOutput.to_summary`, :meth:`Feature.as_dict`,
        :meth:`AttributionGraph.to_dict`, and :meth:`ProbeResult.as_dict` so
        there is a single source of truth for each object's serialization.

        Besides those, the payload carries what a renderer needs to stay
        honest about the score scale: ``input_tokens`` (the prompt tokens
        ``feature.token_idx`` indexes into — the completion tokens in
        ``summary.tokens`` for black-box traces), ``score_method`` (the
        extractor's ``meta["method"]``, e.g. ``"centered_norm"``),
        ``excluded_positions`` and ``exclusion_reasons`` (positions left out
        of the feature ranking, e.g. attention-sink outliers), ``attribution``
        (:meth:`attribution_summary`: method, edge semantics, faithfulness),
        ``saes`` (attached SAEs) and ``sae_warnings`` (input caveats).
        """
        from LLmThoughtLens.circuits.paths import label_path, top_causal_paths
        from LLmThoughtLens.features.extractor import exclusion_reasons

        paths = top_causal_paths(self.graph, n=5)
        reasons = exclusion_reasons(self.features, self.graph.meta.get("excluded_positions"))
        methods = {f.meta.get("method") for f in self.features}
        method = next(iter(methods)) if len(methods) == 1 else None
        input_tokens = self.input_tokens
        return {
            "prompt": self.prompt,
            "output_token": self.output_token,
            "top_tokens": [[t, float(p)] for t, p in self.top_tokens],
            "evidence_kind": self.evidence_kind,
            "provider": self.meta.get("provider", ""),
            "model": self.meta.get("model", ""),
            "summary": self.output.to_summary(),
            "input_tokens": input_tokens,
            "score_method": method if isinstance(method, str) else "",
            "excluded_positions": list(reasons),
            "exclusion_reasons": {str(p): r for p, r in reasons.items()},
            "features": [f.as_dict() for f in self.features[:max_features]],
            "supernodes": [s.as_dict() for s in self.supernodes],
            "graph": self.graph.to_dict(),
            "paths": [
                {"label": label_path(self.graph, p), "log_score": p.log_score} for p in paths
            ],
            "probes": [r.as_dict() for r in self.probe_results],
            "attribution": self.attribution_summary(),
            "saes": list(self.meta.get("saes", [])),
            "sae_warnings": list(self.meta.get("sae_warnings", [])),
        }

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

    def save(self, path: str | Path, steering: SteeringResult | None = None) -> None:
        """Write the full tabbed HTML report to *path* (plus a Steering tab when given)."""
        from LLmThoughtLens.visualization.report import ReportBuilder

        ReportBuilder.from_trace_result(self, steering=steering).save(path)

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
            f"evidence={self.evidence_kind}, "
            f"edges={self.edge_semantics})"
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

    White-box feature scoring is configurable without touching the
    extractor: ``scoring`` (``"centered"`` — the default unitless score — or
    the legacy raw-norm ``"l2"``), ``exclude_outlier_positions`` (``None`` =
    on for ``"centered"``, off for ``"l2"``) and ``exclude_positions``
    (always-excluded token positions; negative indices count from the end)
    are passed straight to :class:`~LLmThoughtLens.features.extractor.FeatureExtractor`.
    :meth:`trace_full` accepts the same three keywords as per-call overrides.

    Attribution is configured the same way (defaults in
    :data:`ATTRIBUTION_DEFAULTS`): ``attribution`` (``"auto"`` /
    ``"gradient"`` / ``"activation_flow"``), ``metric`` (``"logprob"`` /
    ``"logit"`` / ``"logit_diff"``), ``attribution_nodes``, ``validate``
    (``False`` / ``True`` / ``k``), ``node_kind``, ``baseline`` and
    ``max_edge_targets`` are passed to
    :class:`~LLmThoughtLens.circuits.tracer.CircuitTracer`; :meth:`trace_full`
    takes the same keywords (plus ``target`` / ``runner_up``) per call.
    """

    def __init__(
        self,
        provider: BaseProvider,
        *,
        top_k_features: int = 20,
        attribution_threshold: float = 0.05,
        use_supernodes: bool = True,
        blackbox_budget: int | None = 16,
        scoring: WhiteboxScoring = "centered",
        exclude_outlier_positions: bool | None = None,
        exclude_positions: Iterable[int] | None = None,
        attribution: str = ATTRIBUTION_DEFAULTS["attribution"],
        metric: str = ATTRIBUTION_DEFAULTS["metric"],
        attribution_nodes: int = ATTRIBUTION_DEFAULTS["attribution_nodes"],
        validate: bool | int = ATTRIBUTION_DEFAULTS["validate"],
        node_kind: str = ATTRIBUTION_DEFAULTS["node_kind"],
        baseline: str = ATTRIBUTION_DEFAULTS["baseline"],
        max_edge_targets: int | None = ATTRIBUTION_DEFAULTS["max_edge_targets"],
    ) -> None:
        from LLmThoughtLens.features.extractor import SCORING_METHODS

        if scoring not in SCORING_METHODS:
            raise ValueError(f"scoring must be one of {SCORING_METHODS}, got {scoring!r}")
        _validate_attribution_options(
            attribution, metric, node_kind, baseline, attribution_nodes, validate
        )
        self._provider = provider
        self._top_k = int(top_k_features)
        self._threshold = float(attribution_threshold)
        self._use_supernodes = bool(use_supernodes)
        self._blackbox_budget = blackbox_budget
        self._scoring: WhiteboxScoring = scoring
        self._exclude_outlier_positions = exclude_outlier_positions
        self._exclude_positions: tuple[int, ...] = tuple(int(p) for p in exclude_positions or ())
        self._attribution: dict[str, Any] = {
            "attribution": attribution,
            "metric": metric,
            "attribution_nodes": int(attribution_nodes),
            "validate": validate,
            "node_kind": node_kind,
            "baseline": baseline,
            "max_edge_targets": max_edge_targets,
        }
        self._extractor: Any = None
        self._tracer: Any = None
        self._grouper: Any = None
        self._sae: SparseAutoencoder | None = None
        self._sae_layer: int = -1
        # Attachments (SAEAttachment) re-applied to every freshly built extractor.
        self._sae_attachments: list[Any] = []

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------
    #
    # Every factory routes Scope options (top_k_features, scoring,
    # exclude_*_positions, attribution, metric, validate, ...; see
    # SCOPE_OPTIONS) to the Scope and everything else to the provider.

    @classmethod
    def from_mock(cls, **kwargs: Any) -> Scope:
        from LLmThoughtLens.providers.mock_provider import MockProvider

        scope_kw, provider_kw = _split_scope_options(kwargs)
        return cls(MockProvider(**provider_kw), **scope_kw)

    @classmethod
    def from_openai(
        cls,
        model: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> Scope:
        """OpenAI Chat Completions; *model* defaults to ``default_model("openai")``."""
        from LLmThoughtLens.providers.openai_provider import OpenAIProvider

        model = resolve_model("openai", model)
        scope_kw, provider_kw = _split_scope_options(kwargs)
        return cls(OpenAIProvider(model=model, api_key=api_key, **provider_kw), **scope_kw)

    @classmethod
    def from_anthropic(
        cls,
        model: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> Scope:
        """Anthropic Messages API; *model* defaults to ``default_model("anthropic")``."""
        from LLmThoughtLens.providers.anthropic_provider import AnthropicProvider

        model = resolve_model("anthropic", model)
        scope_kw, provider_kw = _split_scope_options(kwargs)
        return cls(AnthropicProvider(model=model, api_key=api_key, **provider_kw), **scope_kw)

    @classmethod
    def from_huggingface(
        cls,
        model_name: str | None = None,
        device: str = "auto",
        **kwargs: Any,
    ) -> Scope:
        """Local white-box HF model; defaults to ``default_model("huggingface")``."""
        from LLmThoughtLens.providers.huggingface_provider import HuggingFaceProvider

        model_name = resolve_model("huggingface", model_name)
        scope_kw, provider_kw = _split_scope_options(kwargs)
        provider = HuggingFaceProvider(model_name=model_name, device=device, **provider_kw)
        return cls(provider, **scope_kw)

    @classmethod
    def from_ollama(
        cls,
        model: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> Scope:
        """Local Ollama server; defaults come from :mod:`LLmThoughtLens.providers.defaults`."""
        from LLmThoughtLens.providers.ollama_provider import OllamaProvider

        model = resolve_model("ollama", model)
        base_url = base_url or default_ollama_url()
        scope_kw, provider_kw = _split_scope_options(kwargs)
        return cls(OllamaProvider(model=model, base_url=base_url, **provider_kw), **scope_kw)

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
        scoring: WhiteboxScoring | None = None,
        exclude_outlier_positions: bool | None = None,
        exclude_positions: Iterable[int] | None = None,
        attribution: str | None = None,
        validate: bool | int | None = None,
        target: int | str | None = None,
        runner_up: int | str | None = None,
        metric: str | None = None,
        attribution_nodes: int | None = None,
        node_kind: str | None = None,
        baseline: str | None = None,
        max_edge_targets: int | None = None,
        **kwargs: Any,
    ) -> TraceResult:
        """Run the full interpretability pipeline on *prompt*.

        ``scoring`` / ``exclude_outlier_positions`` / ``exclude_positions``
        override the Scope's extractor options for this call only (``None``
        inherits the Scope setting); any override builds a fresh extractor
        (with the attached SAEs, if any).

        ``attribution`` (``"auto"`` / ``"gradient"`` / ``"activation_flow"``),
        ``validate`` (``False`` / ``True`` / ``k`` real ablations), ``metric``,
        ``target`` / ``runner_up`` (token id or single-token string; default
        the model's top-1 / top-2), ``attribution_nodes``, ``node_kind``,
        ``baseline`` and ``max_edge_targets`` configure the
        :class:`~LLmThoughtLens.circuits.tracer.CircuitTracer` for this call
        (``None`` inherits the Scope setting).  ``attribution="gradient"`` on
        a provider without a differentiable model raises ``ValueError``;
        ``"auto"`` falls back to activation flow and records why in
        :attr:`TraceResult.method_fallback`.

        Remaining ``**kwargs`` go to the provider's ``run``.
        """
        from LLmThoughtLens.circuits.supernodes import SupernodeGrouper
        from LLmThoughtLens.circuits.tracer import CircuitTracer

        k = top_k_features if top_k_features is not None else self._top_k
        threshold = attribution_threshold if attribution_threshold is not None else self._threshold
        use_super = self._use_supernodes if use_supernodes is None else bool(use_supernodes)
        _validate_attribution_options(
            attribution, metric, node_kind, baseline, attribution_nodes, validate
        )
        settings = dict(self._attribution)
        for key, value in (
            ("attribution", attribution),
            ("validate", validate),
            ("metric", metric),
            ("attribution_nodes", attribution_nodes),
            ("node_kind", node_kind),
            ("baseline", baseline),
            ("max_edge_targets", max_edge_targets),
        ):
            if value is not None:
                settings[key] = value

        # 1. Provider forward (with optional interventions)
        if interventions:
            output = self._provider.run_with_intervention(
                prompt, interventions=interventions, **kwargs
            )
        else:
            output = self._provider.run(prompt, **kwargs)

        # 2. Feature extraction
        overrides: dict[str, Any] = {}
        if scoring is not None:
            overrides["scoring"] = scoring
        if exclude_outlier_positions is not None:
            overrides["exclude_outlier_positions"] = exclude_outlier_positions
        if exclude_positions is not None:
            overrides["exclude_positions"] = tuple(int(p) for p in exclude_positions)
        extractor = self._extractor
        if extractor is None or overrides:
            extractor = self._build_extractor(k, **overrides)
        features = extractor.extract(output, provider=self._provider)
        attachments = tuple(getattr(extractor, "saes", ()) or ())
        sae_map = getattr(extractor, "sae_map", None) if attachments else None

        # 3. Attribution graph
        tracer = self._tracer
        if tracer is None:
            tracer = CircuitTracer(
                min_weight=threshold,
                method=settings["attribution"],
                validate=settings["validate"],
                metric=settings["metric"],
                target=target,
                runner_up=runner_up,
                node_kind=settings["node_kind"],
                baseline=settings["baseline"],
                sae=sae_map or None,
                max_edge_targets=settings["max_edge_targets"],
                attribution_nodes=settings["attribution_nodes"],
            )
        graph = tracer.trace(
            output, features, provider=self._provider, interventions=interventions or None
        )
        if threshold > 0:
            graph = graph.prune(threshold, keep_isolated=True)

        # 4. Supernodes
        supernodes: list = []
        if use_super:
            grouper = self._grouper
            if grouper is None:
                # Several SAEs: match features to their own SAE by meta["sae_name"].
                sae_source: Any = sae_map if len(attachments) > 1 else self._first_sae(extractor)
                grouper = SupernodeGrouper(sae=sae_source)
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
                "scoring": getattr(extractor, "scoring", None),
                "excluded_positions": list(getattr(extractor, "last_excluded_positions", [])),
                "outlier_positions": list(getattr(extractor, "last_outlier_positions", [])),
                "outlier_stats": dict(getattr(extractor, "last_outlier_stats", {})),
                "attribution_settings": {
                    **{k_: v for k_, v in settings.items()},
                    "target": target,
                    "runner_up": runner_up,
                    "interventions": len(interventions or ()),
                },
                "saes": [_sae_description(a) for a in attachments],
                "sae_warnings": list(getattr(extractor, "last_sae_warnings", []) or []),
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
    # SAE attachment
    # ------------------------------------------------------------------

    def _require_internals(self, what: str) -> None:
        if getattr(self._provider, "evidence_kind", "black_box") != "white_box":
            raise ValueError(
                f"{what} read a model's residual stream, but the {self._provider.name!r} "
                "provider is black-box (an API returns text and token probabilities, never "
                "activations). Use a local model, e.g. Scope.from_huggingface('gpt2')."
            )

    def attach_sae(
        self,
        sae: SparseAutoencoder,
        layer: int | None = None,
        *,
        site: str | None = None,
        name: str | None = None,
    ) -> Any:
        """Use *sae* (and only it) for white-box feature extraction.

        *layer* / *site* default from the SAE's hook metadata
        (``blocks.{layer}.hook_{site}``); SAEs without metadata need *layer*
        (it then indexes ``ProviderOutput.activations``, the historical
        meaning).  Returns the :class:`~LLmThoughtLens.features.extractor.SAEAttachment`.
        Raises ``ValueError`` for black-box providers.
        """
        self._require_internals("SAEs")
        if self._extractor is None:
            self._extractor = self._build_extractor(self._top_k)
        attachment = self._extractor.attach_sae(sae, layer, site=site, name=name)
        self._sync_saes()
        return attachment

    def add_sae(
        self,
        sae: SparseAutoencoder,
        layer: int | None = None,
        *,
        site: str | None = None,
        name: str | None = None,
    ) -> Any:
        """Attach *sae* in addition to the SAEs already attached (same arguments as :meth:`attach_sae`)."""
        self._require_internals("SAEs")
        if self._extractor is None:
            self._extractor = self._build_extractor(self._top_k)
        attachment = self._extractor.add_sae(sae, layer, site=site, name=name)
        self._sync_saes()
        return attachment

    def attach_saes(self, saes: Any) -> list[Any]:
        """Replace the attached SAEs with several at once.

        *saes* is anything :meth:`FeatureExtractor.attach_saes
        <LLmThoughtLens.features.extractor.FeatureExtractor.attach_saes>` takes:
        ``{layer: sae}`` / ``{name: sae}``, or an iterable of SAEs, ``(sae, layer)``
        / ``(sae, layer, site)`` tuples or attachments.
        """
        self._require_internals("SAEs")
        if self._extractor is None:
            self._extractor = self._build_extractor(self._top_k)
        attachments = self._extractor.attach_saes(saes)
        self._sync_saes()
        return list(attachments)

    def detach_saes(self) -> None:
        """Remove every attached SAE (back to residual-site scoring)."""
        if self._extractor is not None and hasattr(self._extractor, "detach_saes"):
            self._extractor.detach_saes()
        self._sae_attachments = []
        self._sae = None
        self._sae_layer = -1

    def load_sae(
        self,
        release: str,
        sae_id: str | None = None,
        *,
        layer: int | None = None,
        site: str | None = None,
        name: str | None = None,
        add: bool = False,
        local_files_only: bool = False,
        revision: str | None = None,
        device: str = "cpu",
        **loader_kwargs: Any,
    ) -> SparseAutoencoder:
        """Load a pretrained / saved SAE and attach it (replacing others unless ``add=True``).

        Parameters
        ----------
        release:
            A known release (``"gpt2-small-res-jb"``, ``"gemma-scope-2b-pt-res"``;
            see :func:`~LLmThoughtLens.features.sae_loaders.list_pretrained`), a
            Hub repo id, or — with ``sae_id=None`` — a local path: a SAELens
            folder (``cfg.json`` + ``sae_weights.safetensors``) or a file written
            by :meth:`SparseAutoencoder.save`.
        sae_id:
            E.g. ``"blocks.6.hook_resid_pre"`` or ``"layer_20/width_16k/average_l0_71"``.
        local_files_only:
            Read only the local Hugging Face cache (no download).
            ``HF_HUB_OFFLINE=1`` has the same effect.

        A pretrained SAE records the model it was trained on; attaching it to a
        different HuggingFace model warns.  Raises ``ValueError`` for
        black-box providers.
        """
        self._require_internals("SAEs")
        sae = _load_sae_spec(
            release,
            sae_id,
            local_files_only=local_files_only,
            revision=revision,
            device=device,
            **loader_kwargs,
        )
        _warn_on_model_mismatch(sae, self._provider)
        if add:
            self.add_sae(sae, layer, site=site, name=name)
        else:
            self.attach_sae(sae, layer, site=site, name=name)
        return sae

    def _sync_saes(self) -> None:
        attachments = list(getattr(self._extractor, "saes", ()) or ())
        self._sae_attachments = attachments
        self._sae = attachments[0].sae if attachments else None
        self._sae_layer = int(attachments[0].layer) if attachments else -1

    @staticmethod
    def _first_sae(extractor: Any) -> Any:
        return getattr(extractor, "sae", None)

    def _build_extractor(self, top_k: int, **overrides: Any) -> FeatureExtractor:
        """A :class:`FeatureExtractor` with this Scope's options (+ *overrides*) and SAEs."""
        from LLmThoughtLens.features.extractor import FeatureExtractor

        options: dict[str, Any] = {
            "scoring": self._scoring,
            "exclude_outlier_positions": self._exclude_outlier_positions,
            "exclude_positions": self._exclude_positions,
        }
        options.update(overrides)
        extractor = FeatureExtractor(top_k=top_k, blackbox_budget=self._blackbox_budget, **options)
        if self._sae_attachments:
            extractor.attach_saes(self._sae_attachments)
        return extractor

    # ------------------------------------------------------------------
    # Activation steering (white-box local models only)
    # ------------------------------------------------------------------

    @property
    def hooked(self) -> HookedModel:
        """The provider's :class:`~LLmThoughtLens.models.hooked.HookedModel`.

        Raises :class:`~LLmThoughtLens.features.steering.SteeringUnavailableError`
        for the mock provider (synthetic activations) and black-box providers.
        """
        from LLmThoughtLens.features.steering import as_hooked

        return as_hooked(self._provider)

    def steer(
        self,
        prompt: str | list[dict[str, str]],
        vectors: SteeringVector | Sequence[SteeringVector],
        *,
        coeffs: float | Sequence[float] | None = None,
        max_new_tokens: int = 30,
        temperature: float = 0.0,
        chat: bool = False,
        seed: int | None = None,
        stop_at_eos: bool = True,
        top_k_tokens: int = 10,
    ) -> SteeringResult:
        """Generate with and without the steering *vectors* and compare the two.

        Returns a :class:`~LLmThoughtLens.features.steering.SteeringResult`
        (baseline vs steered completion, teacher-forced per-step KL, promoted /
        suppressed tokens, evidence labels).  See
        :func:`~LLmThoughtLens.features.steering.steer_generate`.
        """
        from LLmThoughtLens.features.steering import steer_generate

        return steer_generate(
            self.hooked,
            prompt,
            vectors,
            coeffs=coeffs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            chat=chat,
            seed=seed,
            stop_at_eos=stop_at_eos,
            top_k_tokens=top_k_tokens,
        )

    def steering_vector_from_contrast(
        self,
        positive_prompts: str | Sequence[str],
        negative_prompts: str | Sequence[str],
        layer: int,
        site: str = "resid_post",
        position: str = "last",
        **kwargs: Any,
    ) -> SteeringVector:
        """Mean-difference (contrastive activation addition) vector on this Scope's model.

        Keyword arguments (``coeff``, ``normalize``, ``positions``, ``chat``,
        ``name``) go to :meth:`SteeringVector.from_contrast
        <LLmThoughtLens.features.steering.SteeringVector.from_contrast>`.
        """
        from LLmThoughtLens.features.steering import SteeringVector

        return SteeringVector.from_contrast(
            self.hooked,
            positive_prompts,
            negative_prompts,
            layer,
            site,  # type: ignore[arg-type]
            position,  # type: ignore[arg-type]
            **kwargs,
        )

    def steering_vector_from_sae_feature(
        self, feature_id: int, sae: SparseAutoencoder | None = None, **kwargs: Any
    ) -> SteeringVector:
        """An SAE feature's decoder direction (default: the first attached SAE)."""
        from LLmThoughtLens.features.steering import SteeringVector

        sae = sae if sae is not None else self._sae
        if sae is None:
            raise ValueError("no SAE attached: pass sae=... or call load_sae / attach_sae first")
        vec = SteeringVector.from_sae_feature(sae, int(feature_id), **kwargs)
        vec.validate(self.hooked)
        return vec

    def load_steering_vector(self, path: str | Path) -> SteeringVector:
        """Load a saved :class:`SteeringVector` and validate it against this Scope's model."""
        from LLmThoughtLens.features.steering import SteeringVector

        return SteeringVector.load(path, hooked=self.hooked)

    def coefficient_sweep(
        self,
        prompt: str | list[dict[str, str]],
        vector: SteeringVector,
        coeffs: Sequence[float],
        target_tokens: str | int | Sequence[str | int] | None = None,
        **kwargs: Any,
    ) -> SweepResult:
        """Next-token effect of *vector* at each coefficient (see :func:`coefficient_sweep`)."""
        from LLmThoughtLens.features.steering import coefficient_sweep

        return coefficient_sweep(self.hooked, prompt, vector, coeffs, target_tokens, **kwargs)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def provider(self) -> BaseProvider:
        return self._provider

    @property
    def sae(self) -> SparseAutoencoder | None:
        return self._sae

    @property
    def saes(self) -> tuple[Any, ...]:
        """The attached SAEs (:class:`~LLmThoughtLens.features.extractor.SAEAttachment`)."""
        return tuple(self._sae_attachments)

    @property
    def attribution_settings(self) -> dict[str, Any]:
        """This Scope's attribution defaults (a copy)."""
        return dict(self._attribution)

    def __repr__(self) -> str:
        return (
            f"Scope(provider={self._provider!r}, top_k={self._top_k}, "
            f"threshold={self._threshold}, scoring={self._scoring!r}, "
            f"attribution={self._attribution['attribution']!r}, "
            f"metric={self._attribution['metric']!r}, "
            f"sae={'yes' if self._sae else 'no'})"
        )


# ---------------------------------------------------------------------------
# SAE helpers
# ---------------------------------------------------------------------------


def _sae_description(attachment: Any) -> dict[str, Any]:
    """JSON-safe description of one attached SAE."""
    cfg = getattr(getattr(attachment, "sae", None), "config", None)
    return {
        "name": getattr(attachment, "name", ""),
        "hook_name": getattr(attachment, "hook_name", ""),
        "layer": int(getattr(attachment, "layer", -1)),
        "site": getattr(attachment, "site", ""),
        "release": getattr(cfg, "release", None),
        "sae_id": getattr(cfg, "sae_id", None),
        "architecture": getattr(cfg, "architecture", None),
        "d_sae": getattr(cfg, "dict_size", None),
        "d_in": getattr(cfg, "input_dim", None),
    }


def _load_sae_spec(
    release: str,
    sae_id: str | None,
    *,
    local_files_only: bool = False,
    revision: str | None = None,
    device: str = "cpu",
    **loader_kwargs: Any,
) -> SparseAutoencoder:
    """Load an SAE from a release + id, a Hub repo id + id, or a local path."""
    if sae_id is None:
        path = Path(release).expanduser()
        if path.is_dir():
            from LLmThoughtLens.features.sae_loaders import load_saelens

            return load_saelens(path, device=device, **loader_kwargs)
        if path.is_file():
            from LLmThoughtLens.features.sae import SparseAutoencoder

            return SparseAutoencoder.load(path)
        raise FileNotFoundError(
            f"{release!r} is not a local SAE path; pass a release and an sae_id "
            "(e.g. load_sae('gpt2-small-res-jb', 'blocks.6.hook_resid_pre'))"
        )
    from LLmThoughtLens.features.sae_loaders import from_pretrained

    return from_pretrained(
        release,
        sae_id,
        revision=revision,
        local_files_only=local_files_only,
        device=device,
        **loader_kwargs,
    )


def _warn_on_model_mismatch(sae: Any, provider: Any) -> None:
    """Warn when a pretrained SAE's recorded HF model differs from the provider's."""
    cfg = getattr(sae, "config", None)
    trained_on = (getattr(cfg, "extra", None) or {}).get("hf_model")
    model_name = getattr(provider, "model_name", None)
    if not trained_on or not isinstance(model_name, str):
        return
    if str(trained_on).lower() != model_name.lower():
        warnings.warn(
            f"this SAE was trained on {trained_on!r} but the provider runs {model_name!r}; "
            "its features are only meaningful on the model it was trained on",
            UserWarning,
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Deferred default for TraceResult.graph
# ---------------------------------------------------------------------------


def _empty_graph() -> AttributionGraph:
    from LLmThoughtLens.circuits.graph import AttributionGraph

    return AttributionGraph(name="(empty)")


# ---------------------------------------------------------------------------
# Before / after intervention comparison
# ---------------------------------------------------------------------------


def compare_traces(
    baseline: TraceResult,
    intervention: TraceResult,
    threshold: float = 0.01,
) -> Any:
    """Diff two traces (baseline vs intervention) → a :class:`GraphDiff`.

    The diff reports nodes added/removed and edges whose weight moved by more
    than *threshold* — the concrete causal effect of the intervention.
    """
    from LLmThoughtLens.circuits.diff import GraphDiff

    return GraphDiff.compute(baseline.graph, intervention.graph, threshold=threshold)


def save_intervention_report(
    baseline: TraceResult,
    intervention: TraceResult,
    path: str | Path,
    threshold: float = 0.01,
) -> Any:
    """Write a before/after HTML intervention report and return the diff.

    Three tabs: the baseline attribution graph, the intervention graph, and a
    side-by-side diff (added / removed / changed edges).
    """
    from LLmThoughtLens.visualization.graph_viz import GraphVisualizer
    from LLmThoughtLens.visualization.report import ReportBuilder

    diff = compare_traces(baseline, intervention, threshold=threshold)
    builder = ReportBuilder(
        title="LLmThoughtLens — Intervention Report (before / after)",
        prompt=baseline.prompt,
        evidence_kind=baseline.evidence_kind,
    )
    builder.add_tab("baseline", "Baseline Graph", GraphVisualizer(baseline.graph).to_html())
    builder.add_tab(
        "intervention", "Intervention Graph", GraphVisualizer(intervention.graph).to_html()
    )
    builder.add_tab("diff", "Before / After Diff", diff.to_html())
    builder.save(path)
    return diff
