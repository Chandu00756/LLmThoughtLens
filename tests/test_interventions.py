"""Causal validation tests for :class:`FeatureIntervention`.

These tests confirm three causal claims by routing a real intervention
through the trace pipeline against :class:`MockProvider` (which honours
interventions by re-running its activation generator with each
intervention's :meth:`apply_numpy` applied before the downstream
feature / graph computation sees it):

(a) inhibiting a feature reduces its contribution in the output graph,
(b) amplifying a feature increases the edge weight to the output node,
(c) clamping to zero removes the feature node from the pruned graph entirely.

The tests do NOT rely on probabilistic mock behaviour — each assertion
compares concrete numerical quantities (sum of edge weights, presence of
a specific node id) so the failure mode is unambiguous.
"""

from __future__ import annotations

import pytest

from LLmThoughtLens.features.intervention import FeatureIntervention, intervention_context
from LLmThoughtLens.providers.mock_provider import MockProvider
from LLmThoughtLens.scope import Scope


_PROMPT = "the capital of France is Paris and the population is large"


def _build_scope(seed: int = 11, n_layers: int = 4) -> Scope:
    # A small mock so the test stays under 50 ms but still exercises real
    # multi-layer / multi-token activation paths.  Scope.from_mock forwards
    # every kwarg to MockProvider, so configure Scope explicitly here.
    provider = MockProvider(n_layers=n_layers, n_heads=2, d_model=16, seed=seed)
    return Scope(provider, top_k_features=20, attribution_threshold=0.05)


def _feature_score_at_token(result, token_idx: int) -> float:
    """Sum of feature scores at *token_idx* — what 'contribution' means for the test."""
    return float(sum(f.score for f in result.features if f.token_idx == token_idx))


def _edges_into_output_weight(result) -> float:
    """Sum |w| of every edge ending at the unique output_token node."""
    output_nodes = result.graph.output_nodes()
    assert output_nodes, "trace should always have one output_token node"
    out_id = output_nodes[0].id
    return float(sum(abs(e.weight) for e in result.graph.in_edges(out_id)))


# ---------------------------------------------------------------------------
# (a) Inhibit reduces feature contribution
# ---------------------------------------------------------------------------


def test_inhibit_reduces_feature_contribution_in_output_graph() -> None:
    """Inhibiting an entire token's contribution must lower its aggregated
    feature score in the output trace."""

    scope = _build_scope()
    baseline = scope.trace_full(_PROMPT, run_probes=False)
    target_token = 0  # "the"
    baseline_contribution = _feature_score_at_token(baseline, target_token)
    assert baseline_contribution > 0.0, "fixture must have nonzero baseline at token 0"

    # Inhibit every hidden dimension at that token across every layer.
    # `feature_id=0` with no SAE attached selects coord 0; targeting all
    # layers (layer=-1) and the chosen token isolates the effect.
    intervention = FeatureIntervention.inhibit(
        feature_id=0, scale=1.0, layer=-1, token_idx=target_token
    )
    modified = scope.trace_full(_PROMPT, interventions=[intervention], run_probes=False)
    modified_contribution = _feature_score_at_token(modified, target_token)

    # Inhibition zeroes the projection along the chosen direction, so the
    # L2 norm of every (layer, token=0) vector can only stay the same or
    # decrease.  A strict inequality is the load-bearing assertion.
    assert modified_contribution < baseline_contribution, (
        f"inhibit failed to reduce contribution: "
        f"baseline={baseline_contribution:.4f}, modified={modified_contribution:.4f}"
    )


# ---------------------------------------------------------------------------
# (b) Amplify increases edge weight to the output node
# ---------------------------------------------------------------------------


def test_amplify_increases_edge_weight_to_output_node() -> None:
    """Amplifying activations must increase the sum of weights of edges
    flowing INTO the output_token node — that is the causal definition of
    "contribution to the prediction"."""

    scope = _build_scope(seed=13)
    baseline = scope.trace_full(_PROMPT, run_probes=False)
    baseline_w = _edges_into_output_weight(baseline)
    assert baseline_w > 0.0

    # Amplify with a large scale so the effect on |a_dst| dominates noise.
    intervention = FeatureIntervention.amplify(
        feature_id=0, scale=8.0, layer=-1, token_idx=-1
    )
    modified = scope.trace_full(_PROMPT, interventions=[intervention], run_probes=False)
    modified_w = _edges_into_output_weight(modified)

    assert modified_w > baseline_w, (
        f"amplify failed to raise total inbound weight to output_token: "
        f"baseline={baseline_w:.4f}, modified={modified_w:.4f}"
    )


# ---------------------------------------------------------------------------
# (c) Clamp-to-zero removes the feature node from the pruned graph
# ---------------------------------------------------------------------------


def test_clamp_to_zero_removes_feature_node_from_pruned_graph() -> None:
    """Clamping every layer × dim of the target token to zero must remove
    that token's features from a pruned graph: either the L2 norm collapses
    so the extractor never selects them, or every incident edge falls below
    the prune threshold and the node is dropped as isolated."""

    provider = MockProvider(n_layers=3, n_heads=2, d_model=16, seed=21)
    scope = Scope(provider, top_k_features=20, attribution_threshold=0.05)

    baseline = scope.trace_full(_PROMPT, attribution_threshold=0.0, run_probes=False)

    # Pick the token whose baseline features are most prominent — that is
    # the one whose disappearance gives the strongest causal signal.
    contribution_per_token: dict[int, float] = {}
    for f in baseline.features:
        contribution_per_token[f.token_idx] = (
            contribution_per_token.get(f.token_idx, 0.0) + f.score
        )
    target_token, baseline_contribution = max(
        contribution_per_token.items(), key=lambda kv: kv[1]
    )
    assert baseline_contribution > 0.0

    baseline_ids_for_target = {
        f.id for f in baseline.features if f.token_idx == target_token
    }
    assert baseline_ids_for_target, "baseline must have at least one feature at target"

    # Strong threshold so weak surviving edges don't keep the node alive.
    threshold = 0.5
    pruned_baseline = baseline.graph.prune(threshold, keep_isolated=False)
    surviving_baseline = {
        nid for nid in baseline_ids_for_target if pruned_baseline.node(nid) is not None
    }
    # The strongest-contribution token must have at least one feature node
    # surviving the prune in the baseline (otherwise the test premise fails
    # before the intervention even runs).
    assert surviving_baseline, (
        f"premise: baseline pruned graph must keep at least one feature at "
        f"token {target_token}; got 0 of {len(baseline_ids_for_target)}"
    )

    # Without an SAE, a single intervention clamps one coordinate of the
    # hidden state.  To fully zero out the activation at *target_token* we
    # apply one clamp per dimension of the model — this is the d-dimensional
    # equivalent of "kill every feature at that position".  In SAE mode a
    # single intervention along a learned direction would suffice; here we
    # iterate to make the causal effect unambiguous on the raw hidden axes.
    interventions = [
        FeatureIntervention.clamp(
            feature_id=d, value=0.0, layer=-1, token_idx=target_token
        )
        for d in range(provider.d_model)
    ]
    modified = scope.trace_full(
        _PROMPT,
        interventions=interventions,
        attribution_threshold=0.0,
        run_probes=False,
    )
    modified_ids_for_target = {
        f.id for f in modified.features if f.token_idx == target_token
    }
    pruned_modified = modified.graph.prune(threshold, keep_isolated=False)

    # Strongest possible causal signal: extraction itself drops the feature.
    if not modified_ids_for_target:
        return

    # Otherwise: every baseline-surviving id must be gone from the pruned
    # modified graph (its edges have all fallen below threshold).
    for nid in surviving_baseline:
        assert pruned_modified.node(nid) is None, (
            f"clamp(value=0) failed to remove feature node {nid} at "
            f"token {target_token} from the pruned graph"
        )


# ---------------------------------------------------------------------------
# Bonus: the context manager really removes hooks even on exception
# ---------------------------------------------------------------------------


def test_intervention_context_removes_hooks_on_exception() -> None:
    """Hook lifecycle must be exception-safe (the design doc's Phase-9
    invariant: 'release the hook after the forward pass completes')."""

    torch = pytest.importorskip("torch")
    import torch.nn as nn

    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = nn.Linear(4, 4)
            self.attn = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.mlp(self.attn(x))

    blocks = [_Block(), _Block()]
    spec = FeatureIntervention.clamp(feature_id=0, value=0.0, layer=1)

    with pytest.raises(RuntimeError):
        with intervention_context(blocks, [spec]):
            assert len(blocks[1].mlp._forward_pre_hooks) == 1
            raise RuntimeError("simulated forward-pass failure")

    # After context exit (even via exception), no hooks remain.
    assert not blocks[0].mlp._forward_pre_hooks
    assert not blocks[1].mlp._forward_pre_hooks
