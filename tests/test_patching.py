"""Activation patching (``circuits.patching``) — real ablations vs gradient attribution.

* Correlation helpers are checked against hand-computed values (and SciPy
  when it is installed).
* Every ablation is checked against the residual stream it should produce on
  a tiny random GPT-2 (float64): a zero-ablated block write leaves
  ``resid_post == resid_pre`` at that site, joint ablations compose, an SAE
  ablation removes exactly ``z * scale * W_dec[:, f]`` at its hook site.
* ``faithfulness`` is exactly 1 on the linear toy transformer of
  ``test_attribution.py`` (where first-order attribution is exact), and is
  recomputed identically from a traced graph.

Nothing here downloads weights.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pytest
from LLmThoughtLens.circuits.patching import FaithfulnessReport, _rankdata, pearson, spearman

PROMPT = "a b c d e f"
N_TOK = 6


def _torch() -> Any:
    return pytest.importorskip("torch")


def gpt2_hooked() -> Any:
    """Tiny random GPT-2 in float64 (perturbed norms) wrapped in a HookedModel."""
    _torch()
    pytest.importorskip("transformers")
    from LLmThoughtLens.models import HookedModel

    from _tiny_hf import WordTokenizer, make_tiny_gpt2

    return HookedModel(make_tiny_gpt2(perturb_norms=True).double(), WordTokenizer())


def toy() -> Any:
    _torch()
    from test_attribution import toy_hooked

    return toy_hooked()


def tiny_provider() -> Any:
    _torch()
    pytest.importorskip("transformers")
    from _tiny_hf import make_provider

    return make_provider(perturb_norms=True)


def captured(hm: Any, hooks: list[Any]) -> Any:
    return hm.forward(PROMPT, hooks=hooks, capture_attentions=False)


# ---------------------------------------------------------------------------
# Correlation helpers and the report (NumPy only)
# ---------------------------------------------------------------------------


class TestCorrelations:
    def test_rankdata_averages_ties(self):
        np.testing.assert_array_equal(_rankdata(np.array([3.0, 1.0, 3.0, 2.0])), [3.5, 1, 3.5, 2])

    def test_known_values(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert pearson(x, [2 * v for v in x]) == pytest.approx(1.0)
        assert pearson(x, [-v for v in x]) == pytest.approx(-1.0)
        cubes = [v**3 for v in x]  # monotone, not linear
        assert spearman(x, cubes) == pytest.approx(1.0)
        assert pearson(x, cubes) < 0.99
        # Hand-computed: one adjacent swap out of four -> r = 0.8 for both.
        assert pearson([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(0.8)
        assert spearman([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(0.8)

    def test_undefined_cases_are_nan(self):
        assert math.isnan(pearson([1.0], [2.0]))
        assert math.isnan(pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))
        assert math.isnan(spearman([1.0, 2.0], [1.0, 2.0, 3.0]))

    def test_matches_scipy_when_installed(self):
        stats = pytest.importorskip("scipy.stats")
        rng = np.random.default_rng(0)
        x = rng.normal(size=40)
        y = x + rng.normal(size=40)
        y[::5] = y[0]  # ties
        assert spearman(x, y) == pytest.approx(float(stats.spearmanr(x, y)[0]), abs=1e-12)
        assert pearson(x, y) == pytest.approx(float(stats.pearsonr(x, y)[0]), abs=1e-12)


class TestReport:
    def test_as_dict_is_strict_json(self):
        rep = FaithfulnessReport(
            spearman=float("nan"),
            pearson=0.5,
            n=1,
            k=3,
            method="zero_ablation",
            metric="logit",
            target_token="x",
            clean_metric=1.0,
            sign_agreement=float("nan"),
            nodes=[{"layer": 0, "token_idx": 1, "predicted": 1.0, "measured": 0.5}],
        )
        d = rep.as_dict()
        assert d["spearman"] is None and d["sign_agreement"] is None and d["pearson"] == 0.5
        assert (d["n"], d["k"], d["node_kind"]) == (1, 3, "delta")
        json.dumps(d, allow_nan=False)
        np.testing.assert_array_equal(rep.predicted, [1.0])
        np.testing.assert_array_equal(rep.measured, [0.5])


# ---------------------------------------------------------------------------
# Ablations do exactly what they claim
# ---------------------------------------------------------------------------


class TestAblations:
    def test_zero_delta_ablation_removes_the_block_write(self):
        from LLmThoughtLens.circuits.attribution import AttributionNode
        from LLmThoughtLens.circuits.patching import ActivationPatcher

        hm = gpt2_hooked()
        patcher = ActivationPatcher(hm)
        clean = patcher.clean_run(PROMPT)
        node = AttributionNode(1, 3)
        out = captured(hm, patcher.ablation_hooks(node, clean))
        post, pre = out.resid_post.double(), out.resid_pre.double()
        np.testing.assert_allclose(post[1, 3], pre[1, 3], atol=1e-12)
        # Only (layer 1, position 3) changes at layer 1; earlier layers are untouched.
        others = [t for t in range(N_TOK) if t != 3]
        np.testing.assert_allclose(post[1, others], clean.resid_post[1, others], atol=1e-12)
        np.testing.assert_allclose(post[0], clean.resid_post[0], atol=1e-12)
        manual = float(out.logits[0, -1, clean.spec.target_id])
        assert patcher.ablate(PROMPT, node, clean=clean) == pytest.approx(clean.metric - manual)

    def test_mean_ablation_uses_the_mean_write_over_baseline_positions(self):
        from LLmThoughtLens.circuits.attribution import AttributionNode
        from LLmThoughtLens.circuits.patching import ActivationPatcher

        hm = gpt2_hooked()
        patcher = ActivationPatcher(hm, baseline="mean", exclude_positions=[0])
        clean = patcher.clean_run(PROMPT)
        assert clean.positions == [1, 2, 3, 4, 5] and patcher.method == "mean_ablation"
        writes = clean.resid_post - clean.resid_pre
        out = captured(hm, patcher.ablation_hooks(AttributionNode(0, 2), clean))
        expected = out.resid_pre[0, 2].double() + writes[0, 1:].mean(dim=0)
        np.testing.assert_allclose(out.resid_post[0, 2].double(), expected, atol=1e-12)

    @pytest.mark.parametrize("baseline", ["zero", "mean"])
    def test_resid_ablation_sets_the_residual(self, baseline):
        from LLmThoughtLens.circuits.attribution import AttributionNode
        from LLmThoughtLens.circuits.patching import ActivationPatcher

        hm = gpt2_hooked()
        patcher = ActivationPatcher(hm, node_kind="resid", baseline=baseline)
        clean = patcher.clean_run(PROMPT)
        out = captured(hm, patcher.ablation_hooks(AttributionNode(1, 4, kind="resid"), clean))
        np.testing.assert_allclose(out.resid_post[1, 4].double(), clean.base[1, 0], atol=1e-12)
        # The per-kind baseline does not depend on the patcher's own node_kind.
        delta = ActivationPatcher(hm, baseline=baseline).clean_run(PROMPT)
        np.testing.assert_allclose(delta.base_for("resid", 1), clean.base[1, 0], atol=1e-12)

    def test_joint_delta_ablations_compose(self):
        """Ablating writes of blocks 0 and 1 at t leaves resid_post[1][t] == embedding[t]."""
        from LLmThoughtLens.circuits.attribution import AttributionNode
        from LLmThoughtLens.circuits.patching import ActivationPatcher

        hm = gpt2_hooked()
        patcher = ActivationPatcher(hm)
        clean = patcher.clean_run(PROMPT)
        nodes = [AttributionNode(0, 2), AttributionNode(1, 2)]
        hooks = [h for nd in nodes for h in patcher.ablation_hooks(nd, clean)]
        out = captured(hm, hooks)
        np.testing.assert_allclose(out.resid_post[1, 2].double(), clean.resid_pre[0, 2], atol=1e-12)
        joint = patcher.ablate(PROMPT, nodes, clean=clean)
        manual = clean.metric - float(out.logits[0, -1, clean.spec.target_id])
        assert joint == pytest.approx(manual)

    @pytest.mark.parametrize("site", ["mlp_out", "attn_out", "resid_pre"])
    def test_sae_ablation_removes_exactly_the_feature_write(self, site):
        from LLmThoughtLens.circuits.patching import ActivationPatcher
        from LLmThoughtLens.models import ResidHook

        from test_attribution import _active_sae_nodes, _feature_write, _site_acts, _toy_sae

        hm = gpt2_hooked()
        sae = _toy_sae(d=hm.d_model, normalize_activations="constant_norm_rescale")
        node = _active_sae_nodes(hm, sae, 1, site, 1)[0]
        patcher = ActivationPatcher(hm, sae=sae)
        clean = patcher.clean_run(PROMPT, nodes=[node])
        x = _site_acts(hm, 1, site)
        seen: dict[str, Any] = {}

        def after(h: Any) -> None:
            seen["x"] = h[0].detach().double()
            return None

        hooks = [*patcher.ablation_hooks(node, clean), ResidHook(1, after, site=site)]
        captured(hm, hooks)
        t, fid = node.token_idx, int(node.sae_feature_id or 0)
        expected = x[t] - _feature_write(sae, x[t], fid).numpy()
        np.testing.assert_allclose(seen["x"][t].numpy(), expected, rtol=1e-5, atol=1e-6)
        others = [p for p in range(N_TOK) if p != t]
        np.testing.assert_allclose(seen["x"][others].numpy(), x[others], atol=1e-12)


# ---------------------------------------------------------------------------
# Faithfulness
# ---------------------------------------------------------------------------


class TestFaithfulness:
    @pytest.mark.parametrize("baseline", ["zero", "mean"])
    def test_exact_on_a_linear_model(self, baseline):
        from LLmThoughtLens.circuits.attribution import AttributionNode, GradientAttributor
        from LLmThoughtLens.circuits.patching import ActivationPatcher

        hm = toy()
        ga = GradientAttributor(hm, baseline=baseline)
        nodes = [AttributionNode(lyr, t) for lyr in range(hm.n_layers) for t in range(N_TOK)]
        res = ga.attribute(PROMPT, nodes, edges=False)
        rep = ActivationPatcher.from_attributor(ga).faithfulness(res, k=8)
        assert rep.n == rep.k == 8 and rep.method == f"{baseline}_ablation"
        assert rep.spearman == pytest.approx(1.0) and rep.pearson == pytest.approx(1.0)
        assert rep.sign_agreement == 1.0
        np.testing.assert_allclose(rep.measured, rep.predicted, rtol=1e-9)
        # Rows are the top-k nodes by |predicted|.
        top = np.sort(np.abs(res.node_attr))[::-1][:8]
        np.testing.assert_allclose(np.abs(rep.predicted), top)
        assert rep.metric == "logit" and rep.clean_metric == pytest.approx(res.metric.value)
        json.dumps(rep.as_dict(), allow_nan=False)

    def test_pairs_source_needs_the_prompt(self):
        from LLmThoughtLens.circuits.attribution import AttributionNode, GradientAttributor
        from LLmThoughtLens.circuits.patching import ActivationPatcher

        hm = toy()
        ga = GradientAttributor(hm)
        res = ga.attribute(PROMPT, [AttributionNode(1, t) for t in range(N_TOK)], edges=False)
        patcher = ActivationPatcher.from_attributor(ga)
        pairs = list(zip(res.nodes, res.node_attr.tolist(), strict=True))
        with pytest.raises(ValueError, match="needs the prompt"):
            patcher.faithfulness(pairs, k=3)
        rep = patcher.faithfulness(pairs, k=3, prompt=PROMPT)
        ref = patcher.faithfulness(res, k=3)
        np.testing.assert_allclose(rep.measured, ref.measured)
        assert patcher.faithfulness(pairs, k=50, prompt=PROMPT).n == N_TOK

    def test_near_exact_on_tiny_gpt2(self):
        """Random-init GPT-2 writes are small, so the linearisation holds closely."""
        from LLmThoughtLens.circuits.attribution import GradientAttributor
        from LLmThoughtLens.circuits.patching import ActivationPatcher

        hm = gpt2_hooked()
        ga = GradientAttributor(hm, exclude_positions=[0])
        res = ga.attribute(PROMPT, None, edges=False, add_top_nodes=8)
        rep = ActivationPatcher.from_attributor(ga).faithfulness(res, k=8)
        assert rep.n == 8
        assert rep.spearman > 0.9 and rep.pearson > 0.9 and rep.sign_agreement >= 0.875

    def test_from_a_traced_graph(self):
        from LLmThoughtLens.circuits.patching import ActivationPatcher
        from LLmThoughtLens.circuits.tracer import CircuitTracer
        from LLmThoughtLens.features.extractor import FeatureExtractor

        provider = tiny_provider()
        out = provider.run(PROMPT)
        feats = FeatureExtractor(top_k=6).extract(out, provider=provider)
        tracer = CircuitTracer(min_weight=0.0, validate=4)
        graph = tracer.trace(out, feats, provider=provider)
        rep = ActivationPatcher(provider.hooked).faithfulness(graph, k=4)
        stored = graph.meta["faithfulness"]
        assert rep.n == stored["n"] == 4
        np.testing.assert_allclose(rep.measured, [r["measured"] for r in stored["nodes"]])
        assert rep.target_token == graph.meta["target_token"] == out.output_token
        graph.meta["metric"] = "bogus"
        with pytest.raises(ValueError, match="unknown metric"):
            ActivationPatcher(provider.hooked).faithfulness(graph, k=2)

    def test_edge_check_needs_edges(self):
        from LLmThoughtLens.circuits.attribution import GradientAttributor
        from LLmThoughtLens.circuits.patching import ActivationPatcher

        hm = toy()
        ga = GradientAttributor(hm)
        res = ga.node_attributions(PROMPT)
        with pytest.raises(ValueError, match="edges=True"):
            ActivationPatcher.from_attributor(ga).edge_check(res)

    def test_edge_check_with_explicit_pairs(self):
        from LLmThoughtLens.circuits.attribution import AttributionNode, GradientAttributor
        from LLmThoughtLens.circuits.patching import ActivationPatcher

        hm = toy()
        ga = GradientAttributor(hm, node_kind="resid")
        nodes = [AttributionNode(lyr, t, kind="resid") for lyr in range(2) for t in range(N_TOK)]
        res = ga.attribute(PROMPT, nodes)
        checks = ActivationPatcher.from_attributor(ga).edge_check(res, pairs=[(0, 6), (2, 9)])
        assert [(c["src"], c["dst"]) for c in checks] == [(0, 6), (2, 9)]
        for c in checks:
            assert c["measured"] == pytest.approx(c["predicted"], rel=1e-9, abs=1e-12)


# ---------------------------------------------------------------------------
# One-call faithfulness (benchmark hook)
# ---------------------------------------------------------------------------


class TestAttributionFaithfulnessFn:
    def test_on_a_white_box_provider(self):
        from LLmThoughtLens.circuits.patching import attribution_faithfulness

        provider = tiny_provider()
        out = attribution_faithfulness(provider, PROMPT, k=5)
        assert set(out) == {
            "spearman",
            "pearson",
            "sign_agreement",
            "n",
            "clean_metric",
            "runtime_s",
        }
        assert out["n"] == 5.0
        assert out["spearman"] is not None and -1.0 <= out["spearman"] <= 1.0
        json.dumps(out, allow_nan=False)
        # A HookedModel works directly and gives the same numbers.
        again = attribution_faithfulness(provider.hooked, PROMPT, k=5)
        assert again["spearman"] == pytest.approx(out["spearman"])

    def test_benchmark_contract(self):
        """bench.metrics.AttributionFaithfulness calls fn(provider, prompt) -> Mapping."""
        import importlib

        fn = importlib.import_module("LLmThoughtLens.circuits.patching").attribution_faithfulness
        out = fn(tiny_provider(), PROMPT)
        assert all(v is None or isinstance(v, float) for v in out.values())

    def test_rejects_providers_without_gradients(self):
        from LLmThoughtLens.circuits.patching import attribution_faithfulness
        from LLmThoughtLens.providers.mock_provider import MockProvider

        with pytest.raises(ValueError, match="no differentiable model"):
            attribution_faithfulness(MockProvider(seed=0), PROMPT)
