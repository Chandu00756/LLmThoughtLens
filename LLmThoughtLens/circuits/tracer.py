"""CircuitTracer — builds attribution graphs from real activations / perturbations.

Three edge semantics, chosen per trace and recorded truthfully in
``graph.meta["attribution_method"]`` / ``graph.meta["edge_semantics"]``:

* **Gradient** (``method="gradient"``; what ``"auto"`` picks for a provider
  that exposes a differentiable :class:`~LLmThoughtLens.models.hooked.HookedModel`,
  i.e. :class:`~LLmThoughtLens.providers.huggingface_provider.HuggingFaceProvider`)
  — ``edge_semantics="causal_linearised"``, every edge ``method="grad_x_act"``.
  :class:`~LLmThoughtLens.circuits.attribution.GradientAttributor` replays the
  prompt with the residual stream on the autograd graph and computes, for a
  target metric at the last position (``metric="logit"`` of the predicted
  token by default, or ``"logprob"`` / ``"logit_diff"``):

  - feature -> output: the node attribution ``A = (v - b) . d metric / d v``
    where ``v`` is the node's value — by default the **block write**
    ``resid_post[l][t] - resid_pre[l][t]`` (``node_kind="delta"``) so the
    skip connection cannot make "one token walking down the layers" the
    dominant path; ``node_kind="resid"`` uses the cumulative residual — and
    ``b`` the zero or per-layer mean baseline.  ``A`` is the first-order
    estimate of ``metric(clean) - metric(node ablated)``, in metric units.
  - feature -> feature (``layer_src < layer_dst``): the linearised effect of
    the source's value on the destination's own output-relevant scalar
    ``s_dst = stopgrad(d metric / d v_dst) . v_dst``.  Edges are total
    (direct + mediated) effects.
  - input token -> feature: the same, with the token's embedding output as
    the source.  Each input node carries its own ``meta["attribution"]``.
  - SAE-feature nodes (``Feature.meta["method"] == "sae"``) use the SAE code
    ``z_f`` computed differentiably from the activation at the SAE's hook site
    (``resid_post`` / ``resid_pre`` / ``mlp_out`` / ``attn_out``); ablating one
    removes its decoder write ``z_f * scale * W_dec[:, f]``.  Pass the SAEs as
    ``CircuitTracer(sae=extractor)`` (or ``extractor.sae_map``); without them
    SAE features fall back to activation flow (``meta["method_fallback"]``).

  ``validate=k`` then runs *real* ablations of the top-``k`` nodes with
  :class:`~LLmThoughtLens.circuits.patching.ActivationPatcher` and stores
  ``graph.meta["faithfulness"]`` (Spearman / Pearson between predicted and
  measured effect) plus per-node ``meta["patched_effect"]``.

* **Activation flow** (``method="activation_flow"``, and ``"auto"`` for
  white-box outputs without a differentiable model, e.g. the mock provider)
  — ``edge_semantics="correlational"``.  For each ordered pair
  ``(src_feat, dst_feat)`` in consecutive feature layers

      w = |a_src| * |a_dst| * cos(a_src, a_dst) * attention_share

  where ``a`` are the residual vectors and ``attention_share`` averages the
  attention mass ``dst.token_idx -> src.token_idx`` over the heads of the
  destination layer (1.0 when attentions are unavailable).  This is a
  co-activation heuristic computed from real activations, **not** a causal
  measurement: adjacent-layer residuals at one position are nearly
  identical, so it favours the identity path.  Magnitudes use the raw
  activation norm (``feature.meta["raw_norm"]``, falling back to
  ``feature.score``).  Edge methods: ``"input_activation"``,
  ``"activation_flow"``, ``"last_layer_to_output"``.

* **Black-box** (no internals) — ``edge_semantics="causal_input_masking"``:
  for each input token we mask it and measure
  ``P_baseline(target) - P_masked(target)``; that delta IS the edge weight
  (``method="mask_perturbation"``).

The graph always gets one ``input_token`` node per input position, one
``output_token`` node for the predicted next token, and (when large enough)
one ``error`` node:

* gradient — ``meta["error_kind"]="attribution_mass"``: the share of total
  attribution mass ``sum |A|`` over *every* residual node of the same kind
  (every layer x every non-excluded position, plus SAE reconstruction-error
  terms for SAE features) that the selected features do not cover.  Because
  attributions are total effects they overlap, so this is a coverage
  measure, not an additive share of the metric.  ``score`` is the uncovered
  mass; its edge to the output carries the signed uncovered attribution.
* activation flow — ``meta["error_kind"]="activation_energy"``: residual
  energy ``sum ||a||^2`` not covered by the selected features' raw norms.

Positions the extractor excluded from ranking (attention sinks) are listed
in ``graph.meta["excluded_positions"]`` and left out of both totals.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from LLmThoughtLens.circuits.graph import AttributionGraph
from LLmThoughtLens.utils.math_utils import cosine_sim
from LLmThoughtLens.utils.tokenizer_utils import mask_positions, token_join, whitespace_tokens

if TYPE_CHECKING:
    from LLmThoughtLens.circuits.attribution import AttributionResult, GradientAttributor
    from LLmThoughtLens.circuits.patching import FaithfulnessReport
    from LLmThoughtLens.features.feature import Feature
    from LLmThoughtLens.models.hooked import HookedModel
    from LLmThoughtLens.providers.base import BaseProvider, ProviderOutput


# Node id offsets so input / output / error nodes never collide with feature ids.
_INPUT_OFFSET = 1_000_000_000
_OUTPUT_OFFSET = 2_000_000_000
_ERROR_OFFSET = 3_000_000_000
_ATTRIBUTION_OFFSET = 4_000_000_000  # added attribution nodes whose natural id is taken

TraceMethod = Literal["auto", "gradient", "activation_flow"]
#: Valid ``CircuitTracer(method=...)`` values.
TRACE_METHODS: tuple[str, ...] = ("auto", "gradient", "activation_flow")
#: ``CircuitTracer(validate=True)`` ablates this many nodes.
DEFAULT_VALIDATE_K = 10
#: Replayed residuals must match ``ProviderOutput.activations`` this closely
#: (max abs error relative to the max activation) before gradients are trusted.
REPLAY_TOLERANCE = 1e-2

_GRADIENT_EDGE_DEFINITION = (
    "feature->feature: (d s_dst / d v_src) . (v_src - b_src) with "
    "s_dst = stopgrad(d metric / d v_dst) . v_dst; input->feature: same with the token "
    "embedding as source; feature->output: (v - b) . d metric / d v (attribution patching)"
)
_FLOW_EDGE_DEFINITION = (
    "|a_src| * |a_dst| * cos(a_src, a_dst) * mean attention(dst -> src) between "
    "consecutive feature layers (correlational co-activation heuristic)"
)
_MASK_EDGE_DEFINITION = "P_baseline(target) - P_masked(target) per masked input token"


class _NotApplicable(Exception):
    """Gradient attribution cannot describe this trace (reason in ``args[0]``)."""


class CircuitTracer:
    """Build an attribution graph from features + (optional) provider.

    Parameters
    ----------
    min_weight:
        Edges with ``|weight| < min_weight`` are dropped (gradient weights are
        in metric units, e.g. logits).
    max_blackbox_calls:
        Cap on masking calls for black-box traces without precomputed features.
    method:
        ``"auto"`` (default): gradient attribution when the provider exposes a
        differentiable :class:`HookedModel` (``provider.supports_gradients``),
        else activation flow.  ``"gradient"`` requires it (``ValueError``
        otherwise); ``"activation_flow"`` always uses the correlational
        heuristic.  Ignored for black-box outputs.
    validate:
        ``False`` (default), ``True`` (= ``10``) or ``k``: after gradient
        attribution, ablate the top-``k`` nodes and record faithfulness.
    metric, target, runner_up:
        Target metric of gradient attribution (see
        :mod:`LLmThoughtLens.circuits.attribution`): ``"logit"`` (default),
        ``"logprob"`` or ``"logit_diff"``; the target / runner-up token
        (id or single-token string) default to the model's top-1 / top-2.
    node_kind:
        ``"delta"`` (block writes, default) or ``"resid"`` (cumulative residual).
    baseline:
        ``"zero"`` (default) or ``"mean"`` (per-layer mean over non-excluded positions).
    sae:
        SAEs for SAE-feature nodes (``Feature.meta["method"] == "sae"``): the
        ``FeatureExtractor`` that produced the features, its ``sae_map``
        (``{attachment name: sae}``, matched through ``meta["sae_name"]``), a
        single :class:`~LLmThoughtLens.features.sae.SparseAutoencoder`, or a
        mapping keyed by ``(layer, site)`` / ``"blocks.L.hook_site"`` / ``layer``
        (see :func:`~LLmThoughtLens.circuits.attribution.lookup_sae`).  Without
        it, SAE features are traced with activation flow.
    max_edge_targets:
        Compute incoming edges only for the *n* nodes with the largest ``|A|``
        (one backward pass each); ``None`` = every feature.
    attribution_nodes:
        Gradient path only: also add the *n* residual nodes with the largest
        ``|A|`` that the extracted features do not cover (``0`` = graph nodes
        are exactly the features).  Extractor rankings favour distinctive
        residuals, not causal relevance, and typically miss the last position
        where the prediction is computed; added nodes are ``feature`` nodes
        with ``meta["selected_by"] = "attribution"``, score ``0.0`` and id
        ``layer * n_tokens + token_idx`` (the extractor's residual-site id).
        Ignored for SAE features.
    """

    def __init__(
        self,
        min_weight: float = 0.05,
        max_blackbox_calls: int = 16,
        *,
        method: str = "auto",
        validate: bool | int = False,
        metric: str = "logit",
        target: int | str | None = None,
        runner_up: int | str | None = None,
        node_kind: str = "delta",
        baseline: str = "zero",
        sae: Any = None,
        max_edge_targets: int | None = None,
        attribution_nodes: int = 0,
    ) -> None:
        from LLmThoughtLens.circuits.attribution import (
            BASELINES,
            METRICS,
            NODE_KINDS,
            normalise_sae_source,
        )

        if method not in TRACE_METHODS:
            raise ValueError(f"method must be one of {TRACE_METHODS}, got {method!r}")
        for name, value, allowed in (
            ("metric", metric, METRICS),
            ("node_kind", node_kind, NODE_KINDS),
            ("baseline", baseline, BASELINES),
        ):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
        if isinstance(validate, bool):
            k = DEFAULT_VALIDATE_K if validate else 0
        else:
            k = int(validate)
            if k < 0:
                raise ValueError(f"validate must be False, True or a positive int, got {validate}")
        if int(attribution_nodes) < 0:
            raise ValueError(f"attribution_nodes must be >= 0, got {attribution_nodes}")
        self.min_weight = float(min_weight)
        self.max_blackbox_calls = int(max_blackbox_calls)
        self.method = method
        self.validate_k = k
        self.metric = metric
        self.target = target
        self.runner_up = runner_up
        self.node_kind = node_kind
        self.baseline = baseline
        self.sae = normalise_sae_source(sae)
        self.max_edge_targets = max_edge_targets
        self.attribution_nodes = int(attribution_nodes)
        #: The :class:`AttributionResult` of the last gradient trace (else ``None``).
        self.last_attribution: AttributionResult | None = None
        #: The :class:`FaithfulnessReport` of the last validated trace (else ``None``).
        self.last_faithfulness: FaithfulnessReport | None = None

    # ------------------------------------------------------------------
    # Top-level dispatch
    # ------------------------------------------------------------------

    def trace(
        self,
        output: ProviderOutput,
        features: Iterable[Feature],
        provider: BaseProvider | HookedModel | None = None,
        *,
        interventions: Sequence[Any] | None = None,
    ) -> AttributionGraph:
        """Build the graph for *output* / *features*.

        Parameters
        ----------
        output:
            The provider output the features were extracted from.
        features:
            Extracted features (graph nodes).
        provider:
            The provider that produced *output* (black-box masking; gradient
            attribution through ``provider.hooked``), or a :class:`HookedModel`.
        interventions:
            The :class:`FeatureIntervention` specs *output* was produced with
            (``provider.run_with_intervention``).  Gradient attribution replays
            them; an intervened output without them falls back to activation
            flow under ``"auto"`` (``ValueError`` under ``"gradient"``).
        """
        feats = list(features)
        self.last_attribution = None
        self.last_faithfulness = None
        graph = AttributionGraph(name=f"trace:{output.prompt[:60]}")
        graph.meta["evidence_kind"] = output.evidence_kind
        graph.meta["prompt"] = output.prompt
        graph.meta["model"] = output.meta.get("model", "")
        graph.meta["method_requested"] = self.method

        self._add_feature_nodes(graph, feats, output.evidence_kind)
        self._add_input_nodes(graph, output)
        output_node_id = self._add_output_node(graph, output)

        if not output.has_internals:
            graph.meta["attribution_method"] = "mask_perturbation"
            graph.meta["edge_semantics"] = "causal_input_masking"
            graph.meta["edge_definition"] = _MASK_EDGE_DEFINITION
            self._blackbox_edges(graph, feats, output, output_node_id, provider)
            return graph

        excluded = _excluded_positions(feats)
        if excluded:
            graph.meta["excluded_positions"] = excluded

        if self.method != "activation_flow":
            try:
                attributor, result = self._gradient_attribution(
                    output, feats, provider, interventions, excluded
                )
            except _NotApplicable as exc:
                if self.method == "gradient":
                    raise ValueError(f"gradient attribution unavailable: {exc}") from None
                if _supports_gradients(provider):
                    graph.meta["method_fallback"] = str(exc)
            else:
                self._gradient_graph(graph, feats, output, output_node_id, attributor, result)
                return graph

        graph.meta["attribution_method"] = "activation_flow"
        graph.meta["edge_semantics"] = "correlational"
        graph.meta["edge_definition"] = _FLOW_EDGE_DEFINITION
        if self.validate_k:
            graph.meta["faithfulness_skipped"] = (
                "activation-flow edges carry no per-node attribution to validate"
            )
        self._whitebox_edges(graph, feats, output, output_node_id)
        self._add_error_residual(graph, feats, output, output_node_id)
        return graph

    # ------------------------------------------------------------------
    # Node helpers
    # ------------------------------------------------------------------

    def _add_feature_nodes(
        self, graph: AttributionGraph, feats: list[Feature], evidence: str
    ) -> None:
        for f in feats:
            node_type = f.node_type if f.node_type else "feature"
            graph.add_node(
                f.id,
                label=f.label,
                node_type=node_type,  # type: ignore[arg-type]
                layer=f.layer,
                token_idx=f.token_idx,
                score=f.score,
                evidence_kind=f.evidence_kind or evidence,
            )

    def _add_input_nodes(self, graph: AttributionGraph, output: ProviderOutput) -> None:
        # White-box: ``output.tokens`` IS the tokenised prompt, so feature
        # token_idx values index straight into it.  Black-box: ``output.tokens``
        # is the *completion*, but the masking engine scores the *prompt*
        # tokens (whitespace split), so input nodes must reflect the prompt.
        input_tokens = output.tokens if output.has_internals else whitespace_tokens(output.prompt)
        for i, tok in enumerate(input_tokens):
            graph.add_node(
                _INPUT_OFFSET + i,
                label=tok,
                node_type="input_token",
                layer=-1,
                token_idx=i,
                score=1.0,
                evidence_kind=output.evidence_kind,
            )

    def _add_output_node(self, graph: AttributionGraph, output: ProviderOutput) -> int:
        nid = _OUTPUT_OFFSET
        graph.add_node(
            nid,
            label=output.output_token or "<unk>",
            node_type="output_token",
            layer=max(0, output.n_layers) + 1,
            token_idx=max(0, output.n_tokens - 1),
            score=float(output.output_prob),
            evidence_kind=output.evidence_kind,
        )
        return nid

    # ------------------------------------------------------------------
    # Gradient attribution (causal, linearised)
    # ------------------------------------------------------------------

    def _gradient_attribution(
        self,
        output: ProviderOutput,
        feats: list[Feature],
        provider: Any,
        interventions: Sequence[Any] | None,
        excluded: list[int],
    ) -> tuple[GradientAttributor, AttributionResult]:
        """Run :class:`GradientAttributor`; raise :class:`_NotApplicable` with a reason."""
        hooked = _hooked_from(provider)
        n_hooks = int(output.meta.get("n_intervention_hooks", 0) or 0)
        if n_hooks and not interventions:
            raise _NotApplicable(
                f"the output was produced with {n_hooks} intervention hook(s); pass "
                "trace(..., interventions=[...]) so attribution replays the same computation"
            )

        from LLmThoughtLens.circuits.attribution import AttributionNode, GradientAttributor

        nodes = [AttributionNode.from_feature(f, self.node_kind) for f in feats]
        has_sae = any(nd.kind == "sae" for nd in nodes)
        if has_sae and self.sae is None:
            raise _NotApplicable("SAE-feature nodes need the SAE: CircuitTracer(sae=...)")
        attributor = GradientAttributor(
            hooked,
            metric=self.metric,
            target=self.target,
            runner_up=self.runner_up,
            node_kind=self.node_kind,
            baseline=self.baseline,
            exclude_positions=excluded,
            sae=self.sae,
            interventions=interventions,
        )
        try:
            # Layer range, SAE resolution and hook-site availability, before any
            # model work, so "auto" can fall back with the reason instead of failing.
            attributor.check_nodes(nodes)
        except (ValueError, IndexError) as exc:
            raise _NotApplicable(str(exc)) from None
        if output.activations is not None and nodes:
            n_tokens = int(output.activations.shape[1])
            bad = [nd.token_idx for nd in nodes if not 0 <= nd.token_idx < n_tokens]
            if bad:
                raise _NotApplicable(
                    f"feature token positions {sorted(set(bad))} are outside the "
                    f"{n_tokens}-token prompt"
                )
        ids: Any = list(output.token_ids) or output.prompt
        result = attributor.attribute(
            ids,
            nodes,
            edges=True,
            max_edge_targets=self.max_edge_targets,
            reference_activations=output.activations,
            add_top_nodes=0 if has_sae else self.attribution_nodes,
        )
        err = float(result.meta.get("replay_max_rel_error", 0.0))
        if not err <= REPLAY_TOLERANCE:
            raise _NotApplicable(
                f"replaying the prompt does not reproduce output.activations (max relative "
                f"error {err:.3g} > {REPLAY_TOLERANCE}); the output came from a different "
                "computation"
            )
        return attributor, result

    def _gradient_graph(
        self,
        graph: AttributionGraph,
        feats: list[Feature],
        output: ProviderOutput,
        output_node_id: int,
        attributor: GradientAttributor,
        result: AttributionResult,
    ) -> None:
        self.last_attribution = result
        spec = result.metric
        excluded = list(graph.meta.get("excluded_positions", []))
        feats = feats + self._attribution_selected(graph, result, output, len(feats))
        graph.meta.update(
            {
                "attribution_method": "gradient",
                "edge_semantics": "causal_linearised",
                "edge_method": "grad_x_act",
                "edge_definition": _GRADIENT_EDGE_DEFINITION,
                **spec.as_dict(),
                "node_kind": result.node_kind,
                "baseline": result.baseline,
                "baseline_positions": list(result.baseline_positions),
                "token_ids": [int(t) for t in result.token_ids],
                "family": str(result.meta.get("family", "")),
                "replay_max_rel_error": float(result.meta.get("replay_max_rel_error", 0.0)),
                "attribution_runtime_s": float(result.runtime_s),
                "top_attribution_nodes": [
                    {
                        "layer": lyr,
                        "token_idx": tok,
                        "token": output.tokens[tok] if 0 <= tok < output.n_tokens else "",
                        "attribution": a,
                    }
                    for lyr, tok, a in result.top_residual_nodes(10, exclude_positions=excluded)
                ],
            }
        )

        # Node annotations.
        for i, (f, nd) in enumerate(zip(feats, result.nodes, strict=True)):
            node = graph.node(f.id)
            if node is None:  # pragma: no cover — every feature was added above
                continue
            node.meta.update(
                {
                    "attribution": float(result.node_attr[i]),
                    "node_kind": nd.kind,
                    "value_norm": float(result.node_value_norm[i]),
                }
            )
            if nd.kind == "sae":
                node.meta.update(
                    {
                        "sae_feature_id": int(nd.sae_feature_id or 0),
                        "sae_layer": int(nd.layer),
                        "sae_site": nd.sae_site,
                        "sae_name": nd.sae_name,
                    }
                )
        for u in range(len(result.input_attr)):
            inp = graph.node(_INPUT_OFFSET + u)
            if inp is not None:
                inp.meta["attribution"] = float(result.input_attr[u])

        # Edges: input -> feature, feature -> feature, feature -> output.
        for j, f_dst in enumerate(feats):
            for u in range(result.input_edges.shape[0]):
                w = float(result.input_edges[u, j])
                if w != 0.0 and abs(w) >= self.min_weight:
                    graph.add_edge(_INPUT_OFFSET + u, f_dst.id, weight=w, method="grad_x_act")
            for i, f_src in enumerate(feats):
                w = float(result.edge_matrix[i, j])
                if w != 0.0 and abs(w) >= self.min_weight:
                    graph.add_edge(f_src.id, f_dst.id, weight=w, method="grad_x_act")
        for i, f in enumerate(feats):
            w = float(result.node_attr[i])
            if abs(w) >= self.min_weight:
                graph.add_edge(f.id, output_node_id, weight=w, method="grad_x_act")

        self._add_attribution_error(graph, feats, result, output_node_id, excluded)

        if self.validate_k and feats:
            from LLmThoughtLens.circuits.patching import ActivationPatcher

            report = ActivationPatcher.from_attributor(attributor).faithfulness(
                result, k=self.validate_k
            )
            self.last_faithfulness = report
            graph.meta["faithfulness"] = report.as_dict()
            for row in report.nodes:
                f_node = graph.node(row["node_id"]) if row.get("node_id") is not None else None
                if f_node is not None:
                    f_node.meta["patched_effect"] = float(row["measured"])

    def _attribution_selected(
        self,
        graph: AttributionGraph,
        result: AttributionResult,
        output: ProviderOutput,
        n_features: int,
    ) -> list[Feature]:
        """Graph nodes for the residual nodes ``add_top_nodes`` appended (if any)."""
        from LLmThoughtLens.features.feature import Feature

        added: list[Feature] = []
        n_tokens = int(result.all_attr.shape[1])
        for rank, nd in enumerate(result.nodes[n_features:]):
            nid = nd.layer * n_tokens + nd.token_idx
            if graph.node(nid) is not None:  # id taken by an SAE / custom feature
                nid = _ATTRIBUTION_OFFSET + nid
            tok = output.tokens[nd.token_idx] if 0 <= nd.token_idx < output.n_tokens else ""
            feat = Feature(
                id=nid,
                label=f"L{nd.layer} {tok!r} (attribution)",
                layer=nd.layer,
                score=0.0,
                token_idx=nd.token_idx,
                node_type="feature",
                evidence_kind="white_box",
                meta={"selected_by": "attribution", "attribution_rank": rank},
            )
            graph.add_node(
                nid,
                label=feat.label,
                node_type="feature",
                layer=feat.layer,
                token_idx=feat.token_idx,
                score=0.0,
                evidence_kind="white_box",
                selected_by="attribution",
            )
            added.append(feat)
        if added:
            graph.meta["n_attribution_nodes"] = len(added)
        return added

    def _add_attribution_error(
        self,
        graph: AttributionGraph,
        feats: list[Feature],
        result: AttributionResult,
        output_node_id: int,
        excluded: list[int],
    ) -> None:
        """Error node = attribution mass of every candidate node the features do not cover."""
        if not feats:
            return
        n_tokens = int(result.all_attr.shape[1])
        skip = {p for p in excluded if 0 <= p < n_tokens}
        cand = [t for t in range(n_tokens) if t not in skip]
        total = 0.0
        signed_total = 0.0
        excluded_mass = 0.0
        explained = 0.0
        signed_explained = 0.0

        resid_sites = {(nd.layer, nd.token_idx) for nd in result.nodes if nd.kind != "sae"}
        if resid_sites:
            all_attr = result.all_attr
            total += float(np.abs(all_attr[:, cand]).sum())
            signed_total += float(all_attr[:, cand].sum())
            excluded_mass += float(np.abs(all_attr[:, sorted(skip)]).sum()) if skip else 0.0
            for lyr, tok in resid_sites:
                if tok in skip:
                    continue
                explained += abs(float(all_attr[lyr, tok]))
                signed_explained += float(all_attr[lyr, tok])
        for key, abs_row in result.sae_all_abs.items():
            err_row = result.sae_error_attr[key]
            total += float(abs_row[cand].sum() + np.abs(err_row[cand]).sum())
            signed_total += float(result.sae_all_sum[key][cand].sum() + err_row[cand].sum())
            if skip:
                idx = sorted(skip)
                excluded_mass += float(abs_row[idx].sum() + np.abs(err_row[idx]).sum())
        seen_sae: set[tuple[Any, ...]] = set()
        for i, nd in enumerate(result.nodes):
            if nd.kind == "sae" and nd.token_idx not in skip and nd.key not in seen_sae:
                seen_sae.add(nd.key)
                explained += abs(float(result.node_attr[i]))
                signed_explained += float(result.node_attr[i])

        if total <= 0.0:
            return
        unexplained = max(0.0, total - explained)
        fraction = unexplained / total
        if fraction < self.min_weight:
            return
        extra: dict[str, Any] = {}
        if skip:
            extra["excluded_positions"] = sorted(skip)
            extra["excluded_attribution_fraction"] = excluded_mass / (total + excluded_mass)
        graph.add_node(
            _ERROR_OFFSET,
            label="error residual",
            node_type="error",
            layer=max(f.layer for f in feats) + 1,
            token_idx=0,
            score=unexplained,
            evidence_kind="white_box",
            unexplained_fraction=fraction,
            error_kind="attribution_mass",
            unexplained_attribution=signed_total - signed_explained,
            **extra,
        )
        graph.add_edge(
            _ERROR_OFFSET,
            output_node_id,
            weight=float(signed_total - signed_explained),
            method="residual",
        )

    # ------------------------------------------------------------------
    # White-box edges (activation flow — correlational)
    # ------------------------------------------------------------------

    def _whitebox_edges(
        self,
        graph: AttributionGraph,
        feats: list[Feature],
        output: ProviderOutput,
        output_node_id: int,
    ) -> None:
        activations = output.activations
        attentions = output.attentions  # (L, H, T, T) or None
        assert activations is not None
        n_layers, n_tokens, _ = activations.shape

        by_layer: dict[int, list[Feature]] = {}
        for f in feats:
            by_layer.setdefault(f.layer, []).append(f)

        # 1. Input-token → first-layer features
        first_layer_feats = by_layer.get(min(by_layer), []) if by_layer else []
        for f in first_layer_feats:
            input_nid = _INPUT_OFFSET + f.token_idx
            w = float(np.linalg.norm(activations[f.layer, f.token_idx]))
            if abs(w) >= self.min_weight:
                graph.add_edge(input_nid, f.id, weight=w, method="input_activation")

        # 2. Feature → feature across layers
        sorted_layers = sorted(by_layer)
        for li, lj in zip(sorted_layers, sorted_layers[1:], strict=False):
            for src in by_layer[li]:
                src_vec = activations[src.layer, src.token_idx]
                src_norm = float(np.linalg.norm(src_vec))
                for dst in by_layer[lj]:
                    dst_vec = activations[dst.layer, dst.token_idx]
                    dst_norm = float(np.linalg.norm(dst_vec))
                    align = cosine_sim(src_vec, dst_vec)
                    if abs(align) < 1e-6 or src_norm < 1e-9 or dst_norm < 1e-9:
                        continue
                    attn_share = self._attention_share(
                        attentions, dst.layer, dst.token_idx, src.token_idx, n_tokens
                    )
                    weight = align * src_norm * dst_norm * attn_share
                    if abs(weight) >= self.min_weight:
                        graph.add_edge(
                            src.id,
                            dst.id,
                            weight=weight,
                            method="activation_flow",
                            attn_share=float(attn_share),
                        )

        # 3. Last-layer features → output_token
        if sorted_layers:
            last_layer = sorted_layers[-1]
            for f in by_layer[last_layer]:
                w = _raw_norm(f)
                if abs(w) >= self.min_weight:
                    graph.add_edge(f.id, output_node_id, weight=w, method="last_layer_to_output")

    def _attention_share(
        self,
        attentions: np.ndarray | None,
        dst_layer: int,
        dst_token: int,
        src_token: int,
        n_tokens: int,
    ) -> float:
        if attentions is None:
            return 1.0
        if not (0 <= dst_layer < attentions.shape[0]):
            return 1.0
        if not (0 <= dst_token < n_tokens) or not (0 <= src_token < n_tokens):
            return 1.0
        head_block = attentions[dst_layer, :, dst_token, src_token]
        return float(head_block.mean())

    # ------------------------------------------------------------------
    # Error residual node (activation flow)
    # ------------------------------------------------------------------

    def _add_error_residual(
        self,
        graph: AttributionGraph,
        feats: list[Feature],
        output: ProviderOutput,
        output_node_id: int,
    ) -> None:
        activations = output.activations
        assert activations is not None
        if not feats:
            return
        # Energy unaccounted for by top features = total energy − Σ raw_norm².
        # Positions the extractor deliberately excluded (attention sinks) are
        # not "unexplained" mass, so they are left out of the total.
        site_energy = np.sum(np.square(activations, dtype=np.float64), axis=-1)  # (L, T)
        full_total = float(site_energy.sum())
        n_tokens = site_energy.shape[1]
        excluded = [p for p in _excluded_positions(feats) if 0 <= p < n_tokens]
        excluded_energy = float(site_energy[:, excluded].sum()) if excluded else 0.0
        total = full_total - excluded_energy
        explained = float(sum(_raw_norm(f) ** 2 for f in feats))
        residual = max(0.0, total - explained)
        if residual < self.min_weight * total:
            return
        extra: dict[str, Any] = {}
        if excluded:
            extra["excluded_positions"] = excluded
            extra["excluded_energy_fraction"] = excluded_energy / (full_total + 1e-9)
        nid = _ERROR_OFFSET
        graph.add_node(
            nid,
            label="error residual",
            node_type="error",
            layer=max(f.layer for f in feats) + 1,
            token_idx=0,
            score=residual,
            evidence_kind="white_box",
            unexplained_fraction=residual / (total + 1e-9),
            error_kind="activation_energy",
            **extra,
        )
        graph.add_edge(
            nid,
            output_node_id,
            weight=float(np.sqrt(residual)),
            method="residual",
        )

    # ------------------------------------------------------------------
    # Black-box edges (real prob deltas via masking)
    # ------------------------------------------------------------------

    def _blackbox_edges(
        self,
        graph: AttributionGraph,
        feats: list[Feature],
        output: ProviderOutput,
        output_node_id: int,
        provider: Any,
    ) -> None:
        target = output.output_token
        baseline_prob = output.output_prob

        # Use feature scores when they were produced by token-masking; they
        # already encode prob deltas, so we re-use them rather than burning
        # more API calls.
        masked_features = [f for f in feats if f.meta.get("method") == "token_masking"]

        # If we don't have prob-delta features (e.g. caller didn't pass a
        # provider), and we DO have a provider here, generate them now.
        if not masked_features and provider is not None and hasattr(provider, "run"):
            tokens = whitespace_tokens(output.prompt)
            limit = min(self.max_blackbox_calls, len(tokens))
            for i in range(limit):
                masked = mask_positions(tokens, [i])
                out_m = provider.run(token_join(masked))
                top = dict(out_m.top_tokens)
                p_m = float(top.get(target, 0.0))
                weight = baseline_prob - p_m
                if abs(weight) >= self.min_weight:
                    graph.add_edge(
                        _INPUT_OFFSET + i,
                        output_node_id,
                        weight=weight,
                        method="mask_perturbation",
                    )
            return

        for f in masked_features:
            weight = float(f.score)
            if abs(weight) >= self.min_weight:
                graph.add_edge(
                    _INPUT_OFFSET + f.token_idx,
                    output_node_id,
                    weight=weight,
                    method="mask_perturbation",
                )

    # ------------------------------------------------------------------
    # Constants (used by tests / external code)
    # ------------------------------------------------------------------

    @staticmethod
    def input_node_id(token_idx: int) -> int:
        return _INPUT_OFFSET + token_idx

    @staticmethod
    def output_node_id() -> int:
        return _OUTPUT_OFFSET

    @staticmethod
    def error_node_id() -> int:
        return _ERROR_OFFSET


def _supports_gradients(provider: Any) -> bool:
    """Whether *provider* is (or exposes) a differentiable :class:`HookedModel`."""
    from LLmThoughtLens.models.hooked import HookedModel

    return isinstance(provider, HookedModel) or bool(getattr(provider, "supports_gradients", False))


def _hooked_from(provider: Any) -> Any:
    """The differentiable :class:`HookedModel` behind *provider* (``_NotApplicable`` if none)."""
    if provider is None:
        raise _NotApplicable("no provider was passed, so there is no model to differentiate")
    from LLmThoughtLens.models.hooked import HookedModel

    if isinstance(provider, HookedModel):
        return provider
    if not getattr(provider, "supports_gradients", False):
        raise _NotApplicable(
            f"{type(provider).__name__} exposes no differentiable model "
            "(gradient attribution needs a HookedModel, e.g. HuggingFaceProvider)"
        )
    from LLmThoughtLens.models.families import UnsupportedArchitectureError

    try:
        return provider.hooked
    except UnsupportedArchitectureError as exc:
        raise _NotApplicable(str(exc)) from None


def _raw_norm(feature: Feature) -> float:
    """Activation-norm magnitude of *feature* (``meta["raw_norm"]``, else its score)."""
    return float(feature.meta.get("raw_norm", feature.score))


def _excluded_positions(feats: list[Feature]) -> list[int]:
    """Token positions the extractor left out of ranking, as recorded in feature meta."""
    for f in feats:
        excluded = f.meta.get("excluded_positions")
        if excluded is not None:
            return sorted(int(p) for p in excluded)
    return []
