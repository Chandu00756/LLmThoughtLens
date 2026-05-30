"""CircuitTracer — builds attribution graphs from real activations / perturbations.

Edge weight semantics:

* **White-box** mode — for each ordered pair ``(src_feat, dst_feat)`` where
  ``dst.layer > src.layer`` we form

      w = sign * |a_src| * |a_dst| * align * attention_share

  where ``a_src`` and ``a_dst`` are the source/destination activation
  vectors, ``align`` is the cosine similarity (sign included), and
  ``attention_share`` averages the attention mass that flows from
  ``dst.token_idx`` to ``src.token_idx`` across the heads of the
  intervening transformer block (1.0 if attentions are unavailable).
  This is a real activation-flow attribution (NOT random-embedding cosine).
* **Black-box** mode — for each input-token / output-token pair we mask the
  source token and measure ``P_baseline(target) − P_masked(target)``.  That
  delta IS the attribution weight (the same prob-delta that drives the
  black-box importance engine).

After building edges, the tracer adds:
* one ``input_token`` node per input position (left column of the layered view);
* one ``output_token`` node for the top predicted next token (right column);
* one ``error`` residual node summarising activation mass not explained by
  any captured feature (when computable).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np

from LLmThoughtLens.circuits.graph import AttributionGraph
from LLmThoughtLens.utils.math_utils import cosine_sim
from LLmThoughtLens.utils.tokenizer_utils import mask_positions, token_join, whitespace_tokens

if TYPE_CHECKING:
    from LLmThoughtLens.features.feature import Feature
    from LLmThoughtLens.providers.base import BaseProvider, ProviderOutput


# Node id offsets so input / output / error nodes never collide with feature ids.
_INPUT_OFFSET = 1_000_000_000
_OUTPUT_OFFSET = 2_000_000_000
_ERROR_OFFSET = 3_000_000_000


class CircuitTracer:
    """Build an attribution graph from features + (optional) provider for masking."""

    def __init__(
        self,
        min_weight: float = 0.05,
        max_blackbox_calls: int = 16,
    ) -> None:
        self.min_weight = float(min_weight)
        self.max_blackbox_calls = int(max_blackbox_calls)

    # ------------------------------------------------------------------
    # Top-level dispatch
    # ------------------------------------------------------------------

    def trace(
        self,
        output: ProviderOutput,
        features: Iterable[Feature],
        provider: BaseProvider | None = None,
    ) -> AttributionGraph:
        feats = list(features)
        graph = AttributionGraph(name=f"trace:{output.prompt[:60]}")
        graph.meta["evidence_kind"] = output.evidence_kind
        graph.meta["prompt"] = output.prompt
        graph.meta["model"] = output.meta.get("model", "")

        self._add_feature_nodes(graph, feats, output.evidence_kind)
        self._add_input_nodes(graph, output)
        output_node_id = self._add_output_node(graph, output)

        if output.has_internals:
            self._whitebox_edges(graph, feats, output, output_node_id)
            self._add_error_residual(graph, feats, output, output_node_id)
        else:
            self._blackbox_edges(graph, feats, output, output_node_id, provider)

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
    # White-box edges (real activation flow)
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
                w = float(f.score)
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
    # Error residual node (white-box only)
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
        # Variance unaccounted for by top features = total var − sum of feature scores².
        total = float(np.linalg.norm(activations) ** 2)
        explained = float(sum(f.score**2 for f in feats))
        residual = max(0.0, total - explained)
        if residual < self.min_weight * total:
            return
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
        provider: BaseProvider | None,
    ) -> None:
        target = output.output_token
        baseline_prob = output.output_prob

        # Use feature scores when they were produced by token-masking; they
        # already encode prob deltas, so we re-use them rather than burning
        # more API calls.
        masked_features = [f for f in feats if f.meta.get("method") == "token_masking"]

        # If we don't have prob-delta features (e.g. caller didn't pass a
        # provider), and we DO have a provider here, generate them now.
        if not masked_features and provider is not None:
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
