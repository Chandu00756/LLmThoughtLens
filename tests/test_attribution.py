"""Gradient attribution (``circuits.attribution``) — correctness against finite differences.

Every attribution and edge is checked numerically on tiny random models:

* GPT-2 in float64 — central differences agree to ~1e-7 relative.
* Llama / Gemma-2 / OPT in float64 — HF computes RMSNorm / softmax in float32
  internally even for float64 weights, so differences use a larger step and
  a float32-level tolerance.
* A purely *linear* toy transformer (fixed causal mixing + linear MLPs, no
  norm) — the metric is linear in every node, so first-order attribution is
  **exact**: it equals the real ablation effect to ~1e-12.

Nothing here downloads weights.  The real-GPT-2 checks at the bottom load
``gpt2`` with ``local_files_only=True`` and skip when it is not cached.
"""

from __future__ import annotations

import time
import types
from typing import Any

import numpy as np
import pytest
from LLmThoughtLens.circuits.attribution import (
    BASELINES,
    METRICS,
    NODE_KINDS,
    SAE_SITES,
    AttributionNode,
    GradientAttributor,
    MetricSpec,
    lookup_sae,
    normalise_sae_source,
)
from LLmThoughtLens.features.feature import Feature

PROMPT = "a b c d e f"
N_TOK = 6


# ---------------------------------------------------------------------------
# Shared builders (torch imported lazily so the value-object tests run core-only)
# ---------------------------------------------------------------------------


def _torch() -> Any:
    return pytest.importorskip("torch")


def make_linear_toy(n_layers: int = 3, d: int = 16, vocab: int = 64, seed: int = 0) -> Any:
    """A transformer-shaped model that is *linear* in its residual stream.

    Each block adds ``mix @ attn(h) + mlp(h)`` (fixed lower-triangular mixing
    across positions, linear maps, no norm), the unembedding is linear and
    there is no final norm, so every metric of type ``"logit"`` is an affine
    function of every node: gradient attribution must equal the measured
    ablation effect exactly.  Blocks live at ``transformer.h`` so
    :class:`HookedModel` resolves them through the generic fallback.
    """
    torch = _torch()
    from torch import nn

    gen = torch.Generator().manual_seed(seed)

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = nn.Linear(d, d, bias=False)
            self.mlp = nn.Linear(d, d, bias=True)
            with torch.no_grad():
                self.attn.weight.copy_(torch.randn(d, d, generator=gen) * 0.3)
                self.mlp.weight.copy_(torch.randn(d, d, generator=gen) * 0.3)
                self.mlp.bias.copy_(torch.randn(d, generator=gen) * 0.1)
            self.register_buffer("mix", torch.tril(torch.rand(32, 32, generator=gen)))

        def forward(self, hidden_states: Any, **_kw: Any) -> Any:
            t = hidden_states.shape[1]
            mixed = torch.einsum("ts,bsd->btd", self.mix[:t, :t], self.attn(hidden_states))
            return hidden_states + mixed + self.mlp(hidden_states)

    class Toy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = types.SimpleNamespace(
                model_type="linear_toy", hidden_size=d, vocab_size=vocab, num_attention_heads=1
            )
            self.transformer = nn.Module()
            self.transformer.wte = nn.Embedding(vocab, d)
            self.transformer.h = nn.ModuleList([Block() for _ in range(n_layers)])
            self.lm_head = nn.Linear(d, vocab, bias=False)
            with torch.no_grad():
                self.transformer.wte.weight.copy_(torch.randn(vocab, d, generator=gen))
                self.lm_head.weight.copy_(torch.randn(vocab, d, generator=gen))

        def get_output_embeddings(self) -> Any:
            return self.lm_head

        def forward(self, input_ids: Any, **_kw: Any) -> Any:
            h = self.transformer.wte(input_ids)
            for block in self.transformer.h:
                h = block(h)
            return types.SimpleNamespace(logits=self.lm_head(h), attentions=None)

    return Toy().double().eval()


def toy_hooked(**kw: Any) -> Any:
    model = make_linear_toy(**kw)  # importorskip("torch") before _tiny_hf imports torch
    from LLmThoughtLens.models import HookedModel

    from _tiny_hf import WordTokenizer

    return HookedModel(model, WordTokenizer())


def family_hooked(family: str) -> Any:
    _torch()
    pytest.importorskip("transformers")
    from LLmThoughtLens.models import HookedModel

    from _tiny_hf import WordTokenizer, make_tiny_family_model

    model = make_tiny_family_model(family, seed=0)
    if model is None:
        pytest.skip(f"installed transformers lacks the {family} config")
    return HookedModel(model.double(), WordTokenizer())


def all_nodes(hm: Any, kind: str = "delta") -> list[AttributionNode]:
    return [AttributionNode(lyr, t, kind=kind) for lyr in range(hm.n_layers) for t in range(N_TOK)]


def metric_under(hm: Any, spec: MetricSpec, hooks: list[Any]) -> float:
    from LLmThoughtLens.circuits.attribution import metric_from_logits

    res = hm.forward(PROMPT, hooks=hooks, capture_attentions=False)
    return float(metric_from_logits(res.logits[0, -1], spec))


def fd_along(hm: Any, spec: MetricSpec, layer: int, t: int, direction: Any, eps: float, site: str):
    """Central difference of the metric when *direction* is added at (layer, site, t)."""
    from LLmThoughtLens.models import ResidHook

    def at(e: float) -> float:
        hook = ResidHook(layer, lambda h: h + (e * direction).to(h.dtype), site=site, positions=[t])
        return metric_under(hm, spec, [hook])

    return (at(eps) - at(-eps)) / (2.0 * eps)


#: family -> (finite-difference step, rtol, atol as a fraction of max |reference|)
FD_SETTINGS = {
    "gpt2": (1e-6, 1e-5, 1e-7),
    "llama": (1e-2, 1e-2, 5e-3),
    "gemma2": (1e-2, 1e-2, 5e-3),
    "opt": (1e-2, 1e-2, 5e-3),
}


def assert_close(actual: Any, expected: Any, rtol: float, atol_frac: float) -> None:
    exp = np.asarray(expected, dtype=np.float64)
    atol = atol_frac * max(1e-12, float(np.max(np.abs(exp))))
    np.testing.assert_allclose(np.asarray(actual, dtype=np.float64), exp, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# Value objects (no torch needed)
# ---------------------------------------------------------------------------


class TestValueObjects:
    def test_choices_are_exposed(self):
        assert METRICS == ("logit", "logprob", "logit_diff")
        assert NODE_KINDS == ("delta", "resid")
        assert BASELINES == ("zero", "mean")

    def test_node_validation(self):
        with pytest.raises(ValueError, match="kind"):
            AttributionNode(0, 0, kind="bogus")
        with pytest.raises(ValueError, match="sae_feature_id"):
            AttributionNode(0, 0, kind="sae")
        with pytest.raises(ValueError, match="sae_site"):
            AttributionNode(0, 0, kind="sae", sae_feature_id=3, sae_site="mlp_in")
        for site in SAE_SITES:  # every extractor SAE site is accepted
            AttributionNode(0, 0, kind="sae", sae_feature_id=3, sae_site=site)

    def test_node_key_ignores_id_and_label(self):
        a = AttributionNode(1, 2, node_id=7, label="x")
        b = AttributionNode(1, 2, node_id=9, label="y")
        assert a.key == b.key == ("delta", 1, 2)
        s = AttributionNode(1, 2, kind="sae", sae_feature_id=5)
        assert s.key == ("sae", 1, 2, 5, "resid_post", None)
        assert s.as_dict()["sae_feature_id"] == 5 and "sae_site" in s.as_dict()
        assert "sae_feature_id" not in a.as_dict()
        named = AttributionNode(1, 2, kind="sae", sae_feature_id=5, sae_name="b")
        assert named.key != s.key and named.sae_key == (1, "resid_post", "b")

    def test_order_places_resid_pre_before_its_block(self):
        sae = {"kind": "sae", "sae_feature_id": 0}
        assert AttributionNode(2, 0).order == 2.0
        assert AttributionNode(2, 0, sae_site="mlp_out", **sae).order == 2.0
        assert AttributionNode(2, 0, sae_site="attn_out", **sae).order == 2.0
        # resid_pre[2] is resid_post[1]; resid_pre[0] is the embeddings.
        assert AttributionNode(2, 0, sae_site="resid_pre", **sae).order == 1.0
        assert AttributionNode(0, 0, sae_site="resid_pre", **sae).order == -0.5
        assert AttributionNode(2, 0, sae_site="mlp_out", **sae).site == (2, "mlp_out")
        assert AttributionNode(2, 0).site == (2, "resid_post")

    def test_from_feature_residual_and_sae(self):
        f = Feature(id=13, label="L2 t", layer=2, token_idx=1, meta={"method": "centered_norm"})
        nd = AttributionNode.from_feature(f, node_kind="resid")
        assert (nd.layer, nd.token_idx, nd.kind, nd.node_id) == (2, 1, "resid", 13)

        legacy = Feature(id=40, layer=3, token_idx=2, meta={"method": "sae"})
        nd = AttributionNode.from_feature(legacy)
        assert (nd.kind, nd.layer, nd.sae_feature_id, nd.sae_site) == ("sae", 3, 40, "resid_post")

        rich = Feature(
            id=99,
            layer=4,
            token_idx=4,
            meta={
                "method": "sae",
                "sae_feature_id": 7,
                "sae_layer": 5,
                "sae_site": "resid_pre",
                "sae_name": "blocks.5.hook_resid_pre",
            },
        )
        nd = AttributionNode.from_feature(rich)
        assert (nd.layer, nd.sae_feature_id, nd.sae_site, nd.node_id) == (5, 7, "resid_pre", 99)
        assert nd.sae_name == "blocks.5.hook_resid_pre"

    @pytest.mark.parametrize(
        ("kw", "match"),
        [
            ({"metric": "prob"}, "metric"),
            ({"node_kind": "sae"}, "node_kind"),
            ({"baseline": "median"}, "baseline"),
        ],
    )
    def test_attributor_rejects_bad_settings(self, kw, match):
        with pytest.raises(ValueError, match=match):
            GradientAttributor(None, **kw)  # type: ignore[arg-type]

    def test_missing_sae_is_a_clear_error(self):
        ga = GradientAttributor(None, sae={3: object()})  # type: ignore[arg-type]
        assert ga.sae_for(3) is not None
        with pytest.raises(ValueError, match="needs an SAE"):
            ga.sae_for(1)

    def test_sae_lookup_order(self):
        by_name, by_site, by_hook, by_layer = object(), object(), object(), object()
        saes = {
            "mine": by_name,
            (2, "mlp_out"): by_site,
            "blocks.3.hook_attn_out": by_hook,
            4: by_layer,
        }
        assert lookup_sae(saes, 9, "resid_post", "mine") is by_name
        assert lookup_sae(saes, 2, "mlp_out") is by_site
        assert lookup_sae(saes, 3, "attn_out") is by_hook
        assert lookup_sae(saes, 4, "resid_pre") is by_layer
        assert lookup_sae(saes, 5, "resid_post") is None
        single = object()
        assert lookup_sae(single, 7, "mlp_out", "anything") is single

    def test_extractor_is_accepted_as_sae_source(self):
        sae = object()
        extractor = types.SimpleNamespace(sae_map={"blocks.1.hook_resid_post": sae})
        assert normalise_sae_source(extractor) == {"blocks.1.hook_resid_post": sae}
        assert normalise_sae_source(types.SimpleNamespace(sae_map={})) is None
        assert normalise_sae_source({}) is None and normalise_sae_source(None) is None
        ga = GradientAttributor(None, sae=extractor)  # type: ignore[arg-type]
        assert ga.sae_for(1, "resid_post", "blocks.1.hook_resid_post") is sae


# ---------------------------------------------------------------------------
# Finite-difference correctness on real (tiny) architectures
# ---------------------------------------------------------------------------


class TestFiniteDifferences:
    @pytest.mark.parametrize("family", ["gpt2", "llama", "gemma2", "opt"])
    @pytest.mark.parametrize("node_kind", ["delta", "resid"])
    @pytest.mark.parametrize("baseline", ["zero", "mean"])
    def test_node_attributions(self, family, node_kind, baseline):
        from LLmThoughtLens.circuits.patching import ActivationPatcher

        eps, rtol, atol = FD_SETTINGS[family]
        hm = family_hooked(family)
        ga = GradientAttributor(hm, node_kind=node_kind, baseline=baseline)
        res = ga.attribute(PROMPT, all_nodes(hm, node_kind), edges=False)
        clean = ActivationPatcher.from_attributor(ga).clean_run(PROMPT, spec=res.metric)
        fd = [
            fd_along(
                hm,
                res.metric,
                nd.layer,
                nd.token_idx,
                clean.values[nd.layer, nd.token_idx] - clean.base[nd.layer, 0],
                eps,
                "resid_post",
            )
            for nd in res.nodes
        ]
        assert_close(res.node_attr, fd, rtol, atol)
        # all_attr is the same quantity for every (layer, position).
        assert_close(res.all_attr.reshape(-1), res.node_attr, 1e-12, 0.0)

    @pytest.mark.parametrize("family", ["gpt2", "llama"])
    @pytest.mark.parametrize("metric", ["logprob", "logit_diff"])
    def test_other_metrics(self, family, metric):
        eps, rtol, atol = FD_SETTINGS[family]
        hm = family_hooked(family)
        ga = GradientAttributor(hm, metric=metric)
        res = ga.attribute(PROMPT, all_nodes(hm), edges=False)
        torch = _torch()
        pre = hm.forward(PROMPT).resid_pre
        post = hm.forward(PROMPT).resid_post
        fd = [
            fd_along(
                hm,
                res.metric,
                nd.layer,
                nd.token_idx,
                (post - pre)[nd.layer, nd.token_idx],
                eps,
                "resid_post",
            )
            for nd in res.nodes
        ]
        assert_close(res.node_attr, fd, rtol, atol)
        assert isinstance(torch, types.ModuleType)

    @pytest.mark.parametrize("family", ["gpt2", "llama"])
    def test_input_attributions(self, family):
        eps, rtol, atol = FD_SETTINGS[family]
        hm = family_hooked(family)
        res = GradientAttributor(hm, baseline="mean").node_attributions(PROMPT)
        emb = hm.forward(PROMPT).resid_pre[0].double()
        base = emb.mean(dim=0)
        fd = [fd_along(hm, res.metric, 0, t, emb[t] - base, eps, "resid_pre") for t in range(N_TOK)]
        assert_close(res.input_attr, fd, rtol, atol)

    @pytest.mark.parametrize("family", ["gpt2", "llama"])
    @pytest.mark.parametrize("node_kind", ["delta", "resid"])
    def test_edges_match_finite_differences_of_dst_scalar(self, family, node_kind):
        """edge(i -> j) == d/de s_j(v_i + e (v_i - b_i)) with s_j = g_j . v_j (g_j fixed)."""
        from LLmThoughtLens.circuits.patching import ActivationPatcher
        from LLmThoughtLens.models import ResidHook

        torch = _torch()
        eps, rtol, atol = FD_SETTINGS[family]
        hm = family_hooked(family)
        ga = GradientAttributor(hm, node_kind=node_kind)
        res = ga.attribute(PROMPT, all_nodes(hm, node_kind))
        edges = res.edges()
        assert edges, "a 2-layer model must have layer-0 -> layer-1 edges"
        assert all(res.nodes[i].layer < res.nodes[j].layer for i, j, _ in edges)
        clean = ActivationPatcher.from_attributor(ga).clean_run(PROMPT, spec=res.metric)

        def s_dst(j: int, hooks: list[Any]) -> float:
            out = hm.forward(PROMPT, hooks=hooks, capture_attentions=False)
            post, pre = out.resid_post.double(), out.resid_pre.double()
            vals = post - pre if node_kind == "delta" else post
            nd = res.nodes[j]
            g = torch.as_tensor(res.node_grads[j], dtype=torch.float64)
            return float((g * vals[nd.layer, nd.token_idx]).sum())

        fd, pred = [], []
        for i, j, w in edges:
            src = res.nodes[i]
            dv = clean.values[src.layer, src.token_idx] - clean.base[src.layer, 0]

            def hook(e: float, src: AttributionNode = src, dv: Any = dv) -> list[Any]:
                fn = lambda h: h + (e * dv).to(h.dtype)  # noqa: E731
                return [ResidHook(src.layer, fn, site="resid_post", positions=[src.token_idx])]

            fd.append((s_dst(j, hook(eps)) - s_dst(j, hook(-eps))) / (2 * eps))
            pred.append(w)
        assert_close(pred, fd, rtol, atol)

    def test_input_edges_match_finite_differences(self):
        from LLmThoughtLens.models import ResidHook

        torch = _torch()
        hm = family_hooked("gpt2")
        res = GradientAttributor(hm).attribute(PROMPT, all_nodes(hm))
        emb = hm.forward(PROMPT).resid_pre[0].double()
        j = 7  # layer 1, token 1
        nd = res.nodes[j]
        g = torch.as_tensor(res.node_grads[j], dtype=torch.float64)

        def s_j(u: int, e: float) -> float:
            hook = ResidHook(
                0, lambda h: h + (e * emb[u]).to(h.dtype), site="resid_pre", positions=[u]
            )
            out = hm.forward(PROMPT, hooks=[hook])
            vals = out.resid_post.double() - out.resid_pre.double()
            return float((g * vals[nd.layer, nd.token_idx]).sum())

        fd = [(s_j(u, 1e-6) - s_j(u, -1e-6)) / 2e-6 for u in range(N_TOK)]
        assert_close(res.input_edges[:, j], fd, 1e-5, 1e-7)
        # Causal masking: later positions cannot influence an earlier destination.
        assert np.all(res.input_edges[nd.token_idx + 1 :, j] == 0.0)


# ---------------------------------------------------------------------------
# Exactness on a linear model
# ---------------------------------------------------------------------------


class TestLinearToyIsExact:
    @pytest.mark.parametrize("node_kind", ["delta", "resid"])
    @pytest.mark.parametrize("baseline", ["zero", "mean"])
    def test_attribution_equals_real_ablation(self, node_kind, baseline):
        from LLmThoughtLens.circuits.patching import ActivationPatcher

        hm = toy_hooked()
        assert hm.family == "generic" and hm.final_norm is None
        ga = GradientAttributor(hm, node_kind=node_kind, baseline=baseline)
        res = ga.attribute(PROMPT, all_nodes(hm, node_kind))
        patcher = ActivationPatcher.from_attributor(ga)
        measured = patcher.node_effects(PROMPT, res.nodes)
        assert_close(res.node_attr, measured, 1e-9, 1e-12)
        # Edges: the measured change in s_dst equals the linearised edge exactly.
        checks = patcher.edge_check(res, k=12)
        assert len(checks) == 12
        assert_close([c["predicted"] for c in checks], [c["measured"] for c in checks], 1e-9, 1e-12)

    def test_identity_path_is_not_an_edge_for_block_writes(self):
        """A residual edge (l, t) -> (l+1, t) contains the skip connection; a delta edge does not."""
        hm = toy_hooked()
        d = GradientAttributor(hm, node_kind="delta").attribute(PROMPT, all_nodes(hm, "delta"))
        r = GradientAttributor(hm, node_kind="resid").attribute(PROMPT, all_nodes(hm, "resid"))
        # For the cumulative residual, s_dst = g . resid_post[l+1][t] contains g . resid_post[l][t]
        # verbatim (skip path), so the same-position edge is at least that large in magnitude.
        n = N_TOK
        same_pos_resid = [abs(r.edge_matrix[t, n + t]) for t in range(n)]
        same_pos_delta = [abs(d.edge_matrix[t, n + t]) for t in range(n)]
        assert sum(same_pos_resid) > sum(same_pos_delta)


# ---------------------------------------------------------------------------
# SAE-feature nodes
# ---------------------------------------------------------------------------

#: SAE pre-processing variants; a feature always writes ``z * scale_t * W_dec[:, f]``.
SAE_VARIANTS: dict[str, dict[str, Any]] = {
    "topk": {},
    "relu": {"architecture": "relu"},
    "scaled_centered": {
        "normalize_activations": "expected_average_only_in",
        "norm_scaling_factor": 0.37,
        "center_input": True,
    },
    "norm_rescale": {"normalize_activations": "constant_norm_rescale"},
    "layer_norm": {"normalize_activations": "layer_norm"},
}


def _toy_sae(d: int = 16, seed: int = 0, **cfg: Any) -> Any:
    _torch()
    from LLmThoughtLens.features.sae import SAEConfig, SparseAutoencoder

    return SparseAutoencoder(
        SAEConfig(input_dim=d, dict_size=32, k=6, seed=seed, device="cpu", **cfg)
    )


def _site_acts(hm: Any, layer: int, site: str, hooks: tuple[Any, ...] = ()) -> np.ndarray:
    """``(T, D)`` activation at ``(layer, site)`` (after *hooks*) as float64 NumPy."""
    from LLmThoughtLens.models import ResidHook

    seen: dict[str, Any] = {}

    def keep(h: Any) -> None:
        seen["x"] = h[0].detach()
        return None

    hm.forward(PROMPT, hooks=[*hooks, ResidHook(layer, keep, site=site)])
    return seen["x"].double().numpy()


def _active_sae_nodes(
    hm: Any, sae: Any, layer: int, site: str, n: int, name: str | None = None
) -> list[AttributionNode]:
    z = sae.encode(_site_acts(hm, layer, site).astype(np.float32))
    nodes = []
    for t in range(1, N_TOK):
        for fid in np.argsort(-z[t], kind="stable")[:n]:
            if z[t, fid] > 0:
                nodes.append(
                    AttributionNode(
                        layer, t, kind="sae", sae_feature_id=int(fid), sae_site=site, sae_name=name
                    )
                )
    return nodes


def _feature_write(sae: Any, x_row: np.ndarray, fid: int) -> Any:
    """``z_f * scale * W_dec[:, f]`` for one activation row, as a float64 tensor."""
    torch = _torch()
    z, scale = sae.encode_with_output_scale(x_row[None].astype(np.float32))
    col = sae.W_dec[:, fid].detach().double()
    return float(z[0, fid]) * float(scale[0]) * col.to(torch.float64)


class TestSAENodes:
    @pytest.mark.parametrize("site", SAE_SITES)
    @pytest.mark.parametrize("baseline", ["zero", "mean"])
    def test_sae_attribution_equals_decoder_ablation_on_linear_model(self, site, baseline):
        from LLmThoughtLens.circuits.patching import ActivationPatcher

        hm = toy_hooked()
        sae = _toy_sae()
        nodes = _active_sae_nodes(hm, sae, 1, site, 2)
        assert nodes
        ga = GradientAttributor(hm, sae=sae, baseline=baseline)
        res = ga.attribute(PROMPT, nodes, edges=False)
        measured = ActivationPatcher.from_attributor(ga).node_effects(PROMPT, nodes)
        # SAE weights are float32, so agreement is at float32 precision.
        assert_close(res.node_attr, measured, 1e-4, 1e-5)
        key = (1, site, None)
        assert res.sae_error_attr[key].shape == (N_TOK,)
        assert res.sae_all_abs[key].shape == (N_TOK,)

    @pytest.mark.parametrize("variant", sorted(SAE_VARIANTS))
    def test_normalised_saes_write_in_model_units(self, variant):
        """Input normalisation rescales the decoder write; attribution must follow it."""
        from LLmThoughtLens.circuits.patching import ActivationPatcher

        hm = toy_hooked()
        sae = _toy_sae(**SAE_VARIANTS[variant])
        nodes = _active_sae_nodes(hm, sae, 1, "resid_post", 2)
        ga = GradientAttributor(hm, sae=sae)
        res = ga.attribute(PROMPT, nodes, edges=False)
        measured = ActivationPatcher.from_attributor(ga).node_effects(PROMPT, nodes)
        assert_close(res.node_attr, measured, 1e-4, 1e-5)

    @pytest.mark.parametrize("variant", ["topk", "scaled_centered", "layer_norm"])
    def test_feature_and_error_terms_decompose_the_site(self, variant):
        """g . x == sum_f A_f + A_error + g . (reconstruction constant) at every position."""
        torch = _torch()
        hm = toy_hooked()
        sae = _toy_sae(**SAE_VARIANTS[variant])
        nodes = _active_sae_nodes(hm, sae, 1, "resid_post", 1)
        res = GradientAttributor(hm, sae=sae).attribute(PROMPT, nodes, edges=False)
        x = _site_acts(hm, 1, "resid_post").astype(np.float32)
        z, scale = sae.encode_with_output_scale(x)
        w_dec = sae.W_dec.detach().double().numpy()
        recon_const = sae.reconstruct(x).astype(np.float64) - scale[:, None] * (z @ w_dec.T)
        key = (1, "resid_post", None)
        for i, nd in enumerate(res.nodes):
            t, g = nd.token_idx, res.node_grads[i]
            lhs = float(g @ x[t].astype(np.float64))
            rhs = res.sae_all_sum[key][t] + res.sae_error_attr[key][t] + float(g @ recon_const[t])
            assert lhs == pytest.approx(rhs, rel=1e-4, abs=1e-4 * abs(lhs) + 1e-5)
        assert isinstance(torch.zeros(1), torch.Tensor)

    def test_edges_into_and_out_of_sae_nodes(self):
        from LLmThoughtLens.circuits.patching import ActivationPatcher

        hm = toy_hooked()
        sae = {1: _toy_sae(), 2: _toy_sae(seed=1)}
        resid = [AttributionNode(0, t) for t in range(N_TOK)]
        sae_nodes = _active_sae_nodes(hm, sae[1], 1, "resid_post", 1)
        sae_nodes += _active_sae_nodes(hm, sae[2], 2, "resid_post", 1)
        ga = GradientAttributor(hm, sae=sae)
        res = ga.attribute(PROMPT, resid + sae_nodes)
        edges = res.edges()
        kinds = {(res.nodes[i].kind, res.nodes[j].kind) for i, j, _ in edges}
        assert ("delta", "sae") in kinds and ("sae", "sae") in kinds
        assert all(res.nodes[i].order < res.nodes[j].order for i, j, _ in edges)
        checks = ActivationPatcher.from_attributor(ga).edge_check(res, k=len(edges))
        # The model is linear and TopK is piecewise linear: when ablating the source
        # leaves the destination feature active (z > 0 after ablation) its code is
        # linear along the ablation, so the measured change equals the edge.  When
        # the ablation switches the feature off the linearisation does not apply.
        z_clean = {
            lyr: sae[lyr].encode(_site_acts(hm, lyr, "resid_post").astype(np.float32))
            for lyr in (1, 2)
        }
        exact, switched_off = [], 0
        for c in checks:
            dst = res.nodes[c["dst"]]
            z0 = float(z_clean[dst.layer][dst.token_idx, dst.sae_feature_id])
            z_after = z0 - c["measured"] / float(res.node_coef[c["dst"]])
            if z_after > 1e-4:
                exact.append(c)
            else:
                switched_off += 1
        assert len(exact) >= 10
        pred = np.array([c["predicted"] for c in exact])
        meas = np.array([c["measured"] for c in exact])
        assert_close(pred, meas, 1e-4, 1e-5)
        assert switched_off >= 1  # this prompt does exercise the non-linear case

    @pytest.mark.parametrize("variant", ["relu", "norm_rescale"])
    def test_sae_edges_match_finite_differences(self, variant):
        """edge(i -> j) == d/de [coef_j * z_j(x_j)] when e * (feature i's write) is added."""
        from LLmThoughtLens.models import ResidHook

        hm = toy_hooked()
        cfg = SAE_VARIANTS[variant]
        sae = {"src": _toy_sae(**cfg), "dst": _toy_sae(seed=1, **cfg)}
        nodes = _active_sae_nodes(hm, sae["src"], 1, "mlp_out", 1, name="src")
        nodes += _active_sae_nodes(hm, sae["dst"], 2, "resid_post", 1, name="dst")
        res = GradientAttributor(hm, sae=sae).attribute(PROMPT, nodes)
        edges = res.edges()
        assert edges and all(res.nodes[i].sae_name == "src" for i, _, _ in edges)
        x_src = _site_acts(hm, 1, "mlp_out")
        eps = 1e-3
        fd, pred = [], []
        for i, j, w in edges:
            src, dst = res.nodes[i], res.nodes[j]
            write = _feature_write(sae["src"], x_src[src.token_idx], int(src.sae_feature_id))

            def s_dst(
                e: float, src: Any = src, dst: Any = dst, write: Any = write, j: int = j
            ) -> float:
                fn = lambda h: h + (e * write).to(h.dtype)  # noqa: E731
                hook = ResidHook(1, fn, site="mlp_out", positions=[src.token_idx])
                x = _site_acts(hm, 2, "resid_post", (hook,))[dst.token_idx]
                z = sae["dst"].encode(x[None].astype(np.float32))[0, dst.sae_feature_id]
                return float(res.node_coef[j]) * float(z)

            fd.append((s_dst(eps) - s_dst(-eps)) / (2 * eps))
            pred.append(w)
        assert_close(pred, fd, 2e-3, 2e-3)

    @pytest.mark.parametrize("site", ["mlp_out", "attn_out", "resid_pre"])
    def test_sae_sites_on_gpt2_match_finite_differences(self, site):
        """A_f = d metric / d e of adding e * (feature write) at the SAE's real hook site."""
        hm = family_hooked("gpt2")
        sae = _toy_sae(d=hm.d_model, architecture="relu")
        nodes = _active_sae_nodes(hm, sae, 1, site, 2)
        assert nodes
        res = GradientAttributor(hm, sae=sae).attribute(PROMPT, nodes, edges=False)
        x = _site_acts(hm, 1, site)
        fd = [
            fd_along(
                hm,
                res.metric,
                1,
                nd.token_idx,
                _feature_write(sae, x[nd.token_idx], int(nd.sae_feature_id)),
                1e-5,
                site,
            )
            for nd in nodes
        ]
        assert_close(res.node_attr, fd, 1e-4, 1e-5)

    def test_unresolvable_sae_is_rejected_before_any_forward(self):
        hm = toy_hooked()
        node = AttributionNode(1, 2, kind="sae", sae_feature_id=0, sae_name="missing")
        with pytest.raises(ValueError, match="needs an SAE"):
            GradientAttributor(hm, sae={"other": _toy_sae()}).attribute(PROMPT, [node])


# ---------------------------------------------------------------------------
# Targets, options and guards
# ---------------------------------------------------------------------------


class TestTargetsAndOptions:
    def test_default_target_is_top1_and_metric_values(self):
        torch = _torch()
        hm = toy_hooked()
        logits = hm.forward(PROMPT).logits[0, -1]
        top = torch.topk(logits, 2).indices.tolist()
        for metric, expected in (
            ("logit", logits[top[0]]),
            ("logprob", torch.log_softmax(logits, -1)[top[0]]),
            ("logit_diff", logits[top[0]] - logits[top[1]]),
        ):
            res = GradientAttributor(hm, metric=metric).node_attributions(PROMPT)
            assert res.metric.target_id == top[0]
            assert res.metric.value == pytest.approx(float(expected), rel=1e-12)
            assert (res.metric.runner_up_id == top[1]) == (metric == "logit_diff")
        spec = res.metric.as_dict()
        assert spec["metric"] == "logit_diff" and spec["runner_up_token"] == f"<{top[1]}>"

    def test_explicit_targets(self):
        hm = toy_hooked()
        from _tiny_hf import WordTokenizer

        word_id = WordTokenizer.word_id("zebra")
        res = GradientAttributor(hm, target="zebra").node_attributions(PROMPT)
        assert res.metric.target_id == word_id
        res = GradientAttributor(hm, metric="logit_diff", target=5, runner_up=6).node_attributions(
            PROMPT
        )
        assert (res.metric.target_id, res.metric.runner_up_id) == (5, 6)
        with pytest.raises(ValueError, match="tokenises to 2 tokens"):
            GradientAttributor(hm, target="two words").node_attributions(PROMPT)
        with pytest.raises(ValueError, match="outside the vocabulary"):
            GradientAttributor(hm, target=10_000).node_attributions(PROMPT)
        with pytest.raises(ValueError, match="different from the target"):
            GradientAttributor(hm, metric="logit_diff", target=5, runner_up=5).node_attributions(
                PROMPT
            )

    def test_node_guards(self):
        hm = toy_hooked()
        ga = GradientAttributor(hm)
        with pytest.raises(ValueError, match="outside"):
            ga.attribute(PROMPT, [AttributionNode(9, 0)])
        with pytest.raises(ValueError, match="does not match"):
            ga.attribute(PROMPT, [AttributionNode(0, 0, kind="resid")])

    def test_max_edge_targets_limits_backward_passes(self):
        hm = toy_hooked()
        nodes = all_nodes(hm)
        full = GradientAttributor(hm).attribute(PROMPT, nodes)
        part = GradientAttributor(hm).attribute(PROMPT, nodes, max_edge_targets=2)
        top2 = np.argsort(-np.abs(full.node_attr), kind="stable")[:2]
        computed = {j for j in range(len(nodes)) if np.any(part.edge_matrix[:, j] != 0)}
        assert computed <= set(top2.tolist())
        np.testing.assert_allclose(part.edge_matrix[:, top2], full.edge_matrix[:, top2])
        none = GradientAttributor(hm).attribute(PROMPT, nodes, edges=False)
        assert not none.edges_computed and not np.any(none.edge_matrix)

    def test_add_top_nodes(self):
        hm = toy_hooked()
        ga = GradientAttributor(hm, exclude_positions=[0])
        given = [AttributionNode(0, 3)]
        res = ga.attribute(PROMPT, given, edges=False, add_top_nodes=4)
        assert res.meta["n_added_nodes"] == 4 and len(res.nodes) == 5
        added = res.nodes[1:]
        assert all(nd.token_idx != 0 and nd.key != given[0].key for nd in added)
        expected = [
            (lyr, tok)
            for lyr, tok, _ in res.top_residual_nodes(10, exclude_positions=[0])
            if (lyr, tok) != (0, 3)
        ][:4]
        assert [(nd.layer, nd.token_idx) for nd in added] == expected
        assert res.top_residual_nodes(3)[0][2] == pytest.approx(
            float(np.max(np.abs(res.all_attr))) * np.sign(res.top_residual_nodes(3)[0][2])
        )

    def test_replay_error(self):
        hm = toy_hooked()
        ref = hm.forward(PROMPT).resid_post.float().numpy()
        ga = GradientAttributor(hm)
        assert ga.attribute(PROMPT, reference_activations=ref).meta["replay_max_rel_error"] == 0.0
        bad = ref.copy()
        bad[1, 2] += 1.0
        err = ga.attribute(PROMPT, reference_activations=bad).meta["replay_max_rel_error"]
        assert err == pytest.approx(1.0 / np.max(np.abs(ref)), rel=1e-4)
        assert ga.attribute(PROMPT, reference_activations=ref[:1]).meta[
            "replay_max_rel_error"
        ] == float("inf")

    def test_interventions_are_replayed(self):
        """Attributing an intervened run describes the intervened computation."""
        pytest.importorskip("transformers")
        from LLmThoughtLens.features.intervention import FeatureIntervention

        from _tiny_hf import make_provider

        provider = make_provider(perturb_norms=True)
        iv = FeatureIntervention.clamp(3, value=5.0, layer=0, site="resid_post")
        out = provider.run_with_intervention(PROMPT, [iv])
        plain = GradientAttributor(provider.hooked).attribute(
            out.token_ids, reference_activations=out.activations
        )
        replayed = GradientAttributor(provider.hooked, interventions=[iv]).attribute(
            out.token_ids, reference_activations=out.activations
        )
        assert plain.meta["replay_max_rel_error"] > 1e-3
        assert replayed.meta["replay_max_rel_error"] < 1e-6
        target = int(np.argmax(out.logits))
        assert replayed.metric.target_id == target
        assert replayed.metric.value == pytest.approx(float(out.logits[target]), rel=1e-5)


# ---------------------------------------------------------------------------
# Real GPT-2 (skipped unless cached locally)
# ---------------------------------------------------------------------------

DALLAS = "The capital of the state containing Dallas is"


@pytest.fixture(scope="module")
def gpt2_provider() -> Any:
    _torch()
    pytest.importorskip("transformers")
    from LLmThoughtLens.models import HookedModel
    from LLmThoughtLens.providers.huggingface_provider import HuggingFaceProvider

    try:
        hm = HookedModel.from_pretrained("gpt2", device="cpu", local_files_only=True)
    except Exception as exc:  # OSError when not cached
        pytest.skip(f"gpt2 weights not available offline: {exc}")
    provider = HuggingFaceProvider("gpt2", device="cpu")
    provider._model, provider._tokenizer, provider._device = hm.model, hm.tokenizer, hm.device
    return provider


class TestRealGPT2:
    def test_dallas_gradient_trace_is_faithful(self, gpt2_provider):
        from LLmThoughtLens.circuits.tracer import CircuitTracer
        from LLmThoughtLens.features.extractor import FeatureExtractor

        out = gpt2_provider.run(DALLAS)
        feats = FeatureExtractor(top_k=20).extract(out, provider=gpt2_provider)
        t0 = time.perf_counter()
        graph = CircuitTracer(validate=20).trace(out, feats, provider=gpt2_provider)
        elapsed = time.perf_counter() - t0
        meta = graph.meta
        assert meta["attribution_method"] == "gradient"
        assert meta["edge_semantics"] == "causal_linearised"
        assert meta["replay_max_rel_error"] < 1e-4
        assert meta["target_token"] == out.output_token
        faith = meta["faithfulness"]
        assert faith["n"] == 20
        # Measured on CPU: Spearman 0.72, sign agreement 0.85.
        assert faith["spearman"] > 0.5
        assert faith["sign_agreement"] >= 0.7
        ff = [
            e
            for e in graph.edges()
            if graph.node(e.src).node_type == "feature" and graph.node(e.dst).node_type == "feature"
        ]
        assert ff and all(e.method == "grad_x_act" for e in ff)
        # Block-write edges are not confined to one position.
        assert any(graph.node(e.src).token_idx != graph.node(e.dst).token_idx for e in ff)
        assert elapsed < 30.0

    def test_edge_spot_check_on_real_model(self, gpt2_provider):
        from LLmThoughtLens.circuits.patching import ActivationPatcher, pearson
        from LLmThoughtLens.features.extractor import FeatureExtractor

        out = gpt2_provider.run(DALLAS)
        feats = FeatureExtractor(top_k=20).extract(out, provider=gpt2_provider)
        ga = GradientAttributor(gpt2_provider.hooked, exclude_positions=[0])
        res = ga.attribute(out.token_ids, [AttributionNode.from_feature(f) for f in feats])
        checks = ActivationPatcher.from_attributor(ga).edge_check(res, k=10)
        # Measured on CPU: Pearson 0.98 between predicted and measured edge effects.
        assert pearson([c["predicted"] for c in checks], [c["measured"] for c in checks]) > 0.8

    def test_attribution_nodes_route_paths_through_the_prediction_position(self, gpt2_provider):
        from LLmThoughtLens.circuits.tracer import CircuitTracer
        from LLmThoughtLens.features.extractor import FeatureExtractor

        out = gpt2_provider.run(DALLAS)
        feats = FeatureExtractor(top_k=20).extract(out, provider=gpt2_provider)

        def positions(graph: Any, path: list[int]) -> set[int]:
            return {graph.node(n).token_idx for n in path if graph.node(n).node_type == "feature"}

        flow = CircuitTracer(method="activation_flow").trace(out, feats, provider=gpt2_provider)
        # The correlational heuristic's best path walks one token down the layers.
        assert len(positions(flow, flow.top_paths(n=1)[0])) == 1

        graph = CircuitTracer(attribution_nodes=10).trace(out, feats, provider=gpt2_provider)
        assert graph.meta["n_attribution_nodes"] == 10
        last = out.n_tokens - 1
        paths = graph.top_paths(n=6)
        assert any(len(positions(graph, p)) >= 2 and last in positions(graph, p) for p in paths)
        err = graph.node(CircuitTracer.error_node_id())
        default = CircuitTracer().trace(out, feats, provider=gpt2_provider)
        err_default = default.node(CircuitTracer.error_node_id())
        assert err is not None and err_default is not None
        assert err.meta["unexplained_fraction"] < err_default.meta["unexplained_fraction"]
