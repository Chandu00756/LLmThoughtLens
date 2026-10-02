"""Tests for :class:`LLmThoughtLens.models.HookedModel` across every supported family.

Every family is exercised on a tiny random model built from its transformers
config (no downloads, see ``_tiny_hf.TINY_FAMILIES``) with weights perturbed
so norms, soft-capping and projections are non-trivial.  Assertions compare
against the model's own forward pass or independently captured internals.
A final section checks the real GPT-2 when it is already in the local HF
cache (skipped otherwise).
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from LLmThoughtLens.models import (  # noqa: E402
    SITES,
    HookedModel,
    ResidHook,
    UnsupportedArchitectureError,
    family_for_model_type,
    load_hf_model,
    resolve_architecture,
    resolve_dtype,
    supported_families,
)

from _tiny_hf import (  # noqa: E402
    N_EMBD,
    N_HEAD,
    N_LAYER,
    TINY_FAMILIES,
    VOCAB,
    WordTokenizer,
    make_tiny_family_model,
    make_tiny_gpt2,
)

PROMPT = "the quick brown fox jumps over"


def _tf() -> Any:
    """The live ``transformers`` module.

    transformers 5 can swap ``sys.modules["transformers"]`` for a fresh lazy
    module while building models, so a reference captured at import time may
    be stale — monkeypatches must target the current one.
    """
    return importlib.import_module("transformers")


# family -> (blocks path, final-norm path) on the *ForCausalLM object
_EXPECTED_PATHS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "gpt2": (("transformer", "h"), ("transformer", "ln_f")),
    "gpt_neox": (("gpt_neox", "layers"), ("gpt_neox", "final_layer_norm")),
    "opt": (("model", "decoder", "layers"), ("model", "decoder", "final_layer_norm")),
}
_LLAMA_STYLE = (("model", "layers"), ("model", "norm"))


def _walk(obj: Any, path: tuple[str, ...]) -> Any:
    for attr in path:
        obj = getattr(obj, attr)
    return obj


def _family_model(family: str, seed: int = 0) -> Any:
    model = make_tiny_family_model(family, seed=seed)
    if model is None:
        pytest.skip(f"transformers {_tf().__version__} has no {TINY_FAMILIES[family][0]}")
    return model


@pytest.fixture(params=sorted(TINY_FAMILIES))
def family(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture
def hm(family: str) -> HookedModel:
    return HookedModel(_family_model(family), WordTokenizer())


def _steer_vector(d: int, seed: int = 3, scale: float = 25.0) -> Any:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(d, generator=g) * scale


def _hook_ids(model: Any) -> list[tuple[str, frozenset[int], frozenset[int]]]:
    """Snapshot every module's hook ids (transformers keeps some of its own)."""
    return [
        (name, frozenset(m._forward_hooks), frozenset(m._forward_pre_hooks))
        for name, m in model.named_modules()
    ]


# ===========================================================================
# Family resolution
# ===========================================================================


class TestResolution:
    def test_every_tiny_family_is_supported(self):
        assert set(TINY_FAMILIES) == set(supported_families())
        assert family_for_model_type("gemma3_text").name == "gemma3"
        assert family_for_model_type("nope") is None
        assert family_for_model_type(None) is None

    def test_blocks_final_norm_unembed_and_dims(self, family: str, hm: HookedModel):
        model = hm.model
        blocks_path, norm_path = _EXPECTED_PATHS.get(family, _LLAMA_STYLE)
        assert hm.family == family
        assert hm.blocks == list(_walk(model, blocks_path))
        assert hm.final_norm is _walk(model, norm_path)
        assert hm.unembed is model.get_output_embeddings()
        assert (hm.n_layers, hm.d_model, hm.n_heads, hm.vocab_size) == (
            N_LAYER,
            N_EMBD,
            N_HEAD,
            VOCAB,
        )
        assert hm.final_logit_softcap == (3.0 if family == "gemma2" else None)
        if family == "opt":
            assert hm.arch.post_norm == (model.model.decoder.project_out,)
        else:
            assert hm.arch.post_norm == ()
        assert family in repr(hm)

    def test_every_site_resolves_to_a_module(self, hm: HookedModel):
        for site in SITES:
            module, when = hm.site_module(-1, site)
            assert module is not None
            assert when == ("pre" if site in ("resid_pre", "mlp_in") else "post")

    def test_generic_fallback_for_unknown_model_type(self):
        model = make_tiny_gpt2()
        model.config.model_type = "my_custom_arch"
        hm = HookedModel(model, WordTokenizer())
        assert hm.family == "generic"
        assert hm.blocks == list(model.transformer.h)
        assert hm.final_norm is model.transformer.ln_f
        assert hm.logit_lens_error(PROMPT) < 1e-4

    def test_unresolvable_model_raises(self):
        with pytest.raises(UnsupportedArchitectureError, match="could not locate"):
            HookedModel(torch.nn.Sequential(torch.nn.Linear(2, 2)), WordTokenizer())

    def test_architecture_of_namespace_without_config(self):
        blocks = [torch.nn.Linear(2, 2)]
        arch = resolve_architecture(SimpleNamespace(model=SimpleNamespace(layers=blocks)))
        assert arch.family == "generic" and arch.blocks == blocks
        assert arch.unembed is None and arch.d_model == 0


# ===========================================================================
# Residual-stream convention
# ===========================================================================


class TestResidualConvention:
    def test_shapes_and_tokens(self, hm: HookedModel):
        res = hm.forward(PROMPT)
        t = len(PROMPT.split())
        assert res.logits.shape == (1, t, VOCAB)
        assert res.resid_pre.shape == res.resid_post.shape == (N_LAYER, t, N_EMBD)
        assert res.attentions.shape == (N_LAYER, N_HEAD, t, t)
        assert res.token_ids == WordTokenizer().encode_ids(PROMPT)
        assert res.tokens == [f"<{i}>" for i in res.token_ids]
        assert res.embeddings.shape == (t, N_EMBD)
        assert res.n_layers == N_LAYER
        assert not res.resid_post.requires_grad

    def test_resid_post_chains_into_resid_pre(self, hm: HookedModel):
        res = hm.forward(PROMPT)
        for layer in range(hm.n_layers - 1):
            torch.testing.assert_close(res.resid_post[layer], res.resid_pre[layer + 1])

    def test_resid_post_matches_independent_block_hooks(self, hm: HookedModel):
        outs: list[Any] = []
        handles = [
            b.register_forward_hook(
                lambda _m, _a, o: outs.append(o[0] if isinstance(o, tuple) else o)
            )
            for b in hm.blocks
        ]
        ids = torch.tensor([WordTokenizer().encode_ids(PROMPT)])
        try:
            with torch.no_grad():
                hm.model(input_ids=ids, use_cache=False)
        finally:
            for h in handles:
                h.remove()
        torch.testing.assert_close(hm.forward(PROMPT).resid_post, torch.cat(outs, dim=0))

    def test_last_layer_is_pre_final_norm_and_lens_reproduces_logits(self, hm: HookedModel):
        res = hm.forward(PROMPT)
        last = res.resid_post[-1]
        with torch.no_grad():
            normed = hm.final_norm(last)
            lens_all = hm.logit_lens(last)
            lens_last = hm.logit_lens(last[-1])
        # The norm is non-trivial, so "pre-final-norm" is a real distinction.
        assert not torch.allclose(normed, last, atol=1e-3)
        torch.testing.assert_close(lens_last, res.logits[0, -1], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(lens_all, res.logits[0], atol=1e-4, rtol=1e-4)
        assert hm.logit_lens_error(PROMPT) < 1e-4
        # HF's output_hidden_states disagrees exactly on the last layer: it is
        # already final-normed (and for OPT even projected), never the residual.
        with torch.no_grad():
            hf = hm.model(input_ids=res.input_ids, output_hidden_states=True, use_cache=False)
        hs = [h[0] for h in hf.hidden_states]
        torch.testing.assert_close(hs[0], res.resid_pre[0])
        for layer in range(hm.n_layers - 1):
            torch.testing.assert_close(hs[layer + 1], res.resid_post[layer])
        if hs[-1].shape == last.shape:
            assert not torch.allclose(hs[-1], last, atol=1e-3)
            torch.testing.assert_close(hs[-1], normed, atol=1e-5, rtol=1e-5)

    def test_softcap_is_applied_exactly_once(self):
        hm = HookedModel(_family_model("gemma2"), WordTokenizer())
        res = hm.forward(PROMPT)
        assert float(res.logits.abs().max()) <= 3.0 + 1e-5  # soft-capped by the model
        with torch.no_grad():
            raw = hm.unembed(hm.final_norm(res.resid_post[-1]))
        assert float(raw.abs().max()) > 3.0  # the cap really bites on this model
        torch.testing.assert_close(
            hm.logit_lens(res.resid_post[-1]), res.logits[0], atol=1e-5, rtol=1e-5
        )

    def test_logit_lens_accepts_numpy(self):
        hm = HookedModel(make_tiny_gpt2(), WordTokenizer())
        res = hm.forward(PROMPT)
        lens = hm.logit_lens(res.resid_post[-1].numpy())
        torch.testing.assert_close(lens, res.logits[0], atol=1e-5, rtol=1e-5)

    def test_capture_off_and_attention_off(self):
        hm = HookedModel(make_tiny_gpt2(), WordTokenizer())
        res = hm.forward(PROMPT, capture=False, capture_attentions=False)
        assert res.resid_pre is None and res.resid_post is None and res.attentions is None
        assert res.embeddings is None and res.n_layers == 0
        with pytest.raises(RuntimeError, match="not captured"):
            res.resid()
        with pytest.raises(ValueError, match="site must be"):
            res.resid("mlp_in")  # type: ignore[arg-type]

    def test_sdpa_model_reports_no_attention_weights(self):
        tf = _tf()
        cfg = tf.GPT2Config(n_layer=1, n_head=2, n_embd=8, vocab_size=VOCAB)
        model = tf.GPT2LMHeadModel(cfg).eval()
        model.config._attn_implementation = "sdpa"
        hm = HookedModel(model, WordTokenizer())
        assert hm.attn_implementation == "sdpa"
        assert hm.returns_attention_weights is False
        res = hm.forward(PROMPT)
        assert res.attentions is None and res.resid_post is not None

    def test_model_kwargs_are_forwarded(self):
        hm = HookedModel(make_tiny_gpt2(), WordTokenizer())
        seen: dict[str, Any] = {}
        real = hm.model.forward

        def spy(*args: Any, **kwargs: Any) -> Any:
            seen.update(kwargs)
            return real(*args, **kwargs)

        hm.model.forward = spy  # type: ignore[method-assign]
        hm.forward(PROMPT, output_hidden_states=True)
        assert seen["output_hidden_states"] is True and seen["use_cache"] is False


# ===========================================================================
# Hooks
# ===========================================================================


class TestHooks:
    def test_resid_hook_modifies_downstream_and_is_removed(self, hm: HookedModel):
        base = hm.forward(PROMPT)
        before = _hook_ids(hm.model)
        v = _steer_vector(hm.d_model)
        with hm.hooks([ResidHook(0, lambda h: h + v, site="resid_post")]) as handles:
            assert len(handles) == 1
            res = hm.forward(PROMPT)
        torch.testing.assert_close(res.resid_post[0], base.resid_post[0] + v)
        torch.testing.assert_close(res.resid_pre[1], base.resid_post[0] + v)  # captured post-edit
        assert not torch.allclose(res.logits, base.logits)
        assert _hook_ids(hm.model) == before
        torch.testing.assert_close(hm.forward(PROMPT).logits, base.logits)

    def test_hooks_removed_on_exception(self, hm: HookedModel):
        before = _hook_ids(hm.model)
        specs = [ResidHook(i, lambda h: None, site=s) for i in (0, -1) for s in SITES]
        with pytest.raises(RuntimeError, match="boom"), hm.hooks(specs):
            assert _hook_ids(hm.model) != before
            raise RuntimeError("boom")
        assert _hook_ids(hm.model) == before

    def test_forward_hooks_argument_is_scoped_to_the_call(self, hm: HookedModel):
        base = hm.forward(PROMPT)
        v = _steer_vector(hm.d_model)
        steered = hm.forward(PROMPT, hooks=[ResidHook(1, lambda h: h + v, site="resid_pre")])
        assert not torch.allclose(steered.logits, base.logits)
        torch.testing.assert_close(steered.resid_pre[1], base.resid_pre[1] + v)
        torch.testing.assert_close(hm.forward(PROMPT).logits, base.logits)

    @pytest.mark.parametrize("site", SITES)
    def test_every_site_observes_and_edits(self, hm: HookedModel, site: str):
        base = hm.forward(PROMPT)
        shapes: list[tuple[int, ...]] = []
        obs = hm.forward(
            PROMPT, hooks=[ResidHook(0, lambda h: shapes.append(tuple(h.shape)), site)]
        )
        torch.testing.assert_close(obs.logits, base.logits)  # observe-only changes nothing
        assert len(shapes) == 1 and shapes[0][0] == 1 and shapes[0][1] == len(PROMPT.split())
        edited = hm.forward(PROMPT, hooks=[ResidHook(0, lambda h: h * 0.0 + 3.0, site)])
        assert not torch.allclose(edited.logits, base.logits)

    def test_position_restricted_hook_in_single_forward(self, hm: HookedModel):
        base = hm.forward(PROMPT)
        v = _steer_vector(hm.d_model)
        seen: list[tuple[int, ...]] = []

        def fn(h: Any) -> Any:
            seen.append(tuple(h.shape))
            return h + v

        res = hm.forward(PROMPT, hooks=[ResidHook(0, fn, "resid_post", positions=[1, -1])])
        t = len(PROMPT.split())
        assert seen == [(1, 2, hm.d_model)]
        delta = (res.resid_post[0] - base.resid_post[0]).abs().sum(-1)
        assert [i for i in range(t) if float(delta[i]) > 1e-4] == [1, t - 1]
        # Out-of-range positions never fire.
        res2 = hm.forward(PROMPT, hooks=[ResidHook(0, fn, "resid_post", positions=99)])
        torch.testing.assert_close(res2.logits, base.logits)

    def test_hook_validation(self):
        hm = HookedModel(make_tiny_gpt2(), WordTokenizer())
        with pytest.raises(ValueError, match="unknown hook site"):
            ResidHook(0, lambda h: None, site="nowhere")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="callable"):
            ResidHook(0, "not callable")  # type: ignore[arg-type]
        assert ResidHook(0, lambda h: None, positions=3).positions == (3,)
        with (
            pytest.raises(IndexError, match="out of range"),
            hm.hooks([ResidHook(5, lambda h: None)]),
        ):
            pass
        with pytest.raises(ValueError, match="unknown hook site"):
            hm.site_module(0, "nowhere")


# ===========================================================================
# Generation
# ===========================================================================


def _manual_greedy(model: Any, ids: list[int], n: int) -> tuple[list[int], list[float]]:
    out_ids, lps = [], []
    seq = torch.tensor([ids])
    with torch.no_grad():
        for _ in range(n):
            logits = model(input_ids=seq, use_cache=False).logits[0, -1].float()
            nxt = int(torch.argmax(logits))
            out_ids.append(nxt)
            lps.append(float(torch.log_softmax(logits, -1)[nxt]))
            seq = torch.cat([seq, torch.tensor([[nxt]])], dim=1)
    return out_ids, lps


class TestGenerate:
    N = 6

    def test_greedy_is_deterministic_and_matches_manual_loop(self, hm: HookedModel):
        a = hm.generate(PROMPT, max_new_tokens=self.N, stop_at_eos=False)
        b = hm.generate(PROMPT, max_new_tokens=self.N, stop_at_eos=False)
        c = hm.generate(PROMPT, max_new_tokens=self.N, stop_at_eos=False, use_cache=False)
        ref_ids, ref_lp = _manual_greedy(hm.model, WordTokenizer().encode_ids(PROMPT), self.N)
        assert a.token_ids == b.token_ids == c.token_ids == ref_ids
        assert a.logprobs == pytest.approx(ref_lp, abs=1e-4)
        assert a.tokens == [f"<{i}>" for i in ref_ids]
        assert a.text == "".join(a.tokens)
        assert a.stop_reason == "max_new_tokens"
        assert a.full_token_ids == a.prompt_token_ids + a.token_ids
        assert a.prompt_tokens == [f"<{i}>" for i in a.prompt_token_ids]

    def test_resid_hook_changes_generation_and_cache_is_consistent(self, hm: HookedModel):
        base = hm.generate(PROMPT, max_new_tokens=self.N, stop_at_eos=False)
        v = _steer_vector(hm.d_model, scale=60.0)
        hook = ResidHook(0, lambda h: h + v, site="resid_post")
        cached = hm.generate(PROMPT, max_new_tokens=self.N, hooks=[hook], stop_at_eos=False)
        uncached = hm.generate(
            PROMPT, max_new_tokens=self.N, hooks=[hook], stop_at_eos=False, use_cache=False
        )
        assert cached.token_ids != base.token_ids
        assert cached.token_ids == uncached.token_ids  # positions=None: cache-independent
        # Hooks are gone afterwards.
        assert hm.generate(PROMPT, max_new_tokens=self.N, stop_at_eos=False).token_ids == (
            base.token_ids
        )

    def test_absolute_positions_are_cache_independent(self, hm: HookedModel):
        v = _steer_vector(hm.d_model, seed=5, scale=60.0)
        t = len(PROMPT.split())
        targets = (1, t + 1)  # one prompt token and one generated token
        offsets: list[tuple[int, int]] = []

        def fn(h: Any) -> Any:
            offsets.append((hm.current_offset, int(h.shape[1])))
            return h + v

        hook = ResidHook(0, fn, site="resid_pre", positions=targets)
        cached = hm.generate(PROMPT, max_new_tokens=4, hooks=[hook], stop_at_eos=False)
        # Cached: prefill (offset 0) hits position 1; step feeding position t+1 hits it once.
        assert offsets == [(0, 1), (t + 1, 1)]
        uncached = hm.generate(
            PROMPT, max_new_tokens=4, hooks=[hook], stop_at_eos=False, use_cache=False
        )
        assert cached.token_ids == uncached.token_ids

    def test_negative_positions_follow_the_newest_token_under_cache(self):
        hm = HookedModel(make_tiny_gpt2(), WordTokenizer())
        fed: list[int] = []

        def fn(h: Any) -> None:
            fed.append(hm.current_offset)

        t = len(PROMPT.split())
        hm.generate(
            PROMPT, max_new_tokens=3, stop_at_eos=False, hooks=[ResidHook(0, fn, positions=-1)]
        )
        # Offsets of the chunk containing the newest token: prefill, then each fed token.
        assert fed == [0, t, t + 1]

    def test_stop_at_eos(self):
        model = make_tiny_gpt2()
        first = _manual_greedy(model, WordTokenizer().encode_ids(PROMPT), 1)[0][0]
        model.generation_config.eos_token_id = None
        hm = HookedModel(model, WordTokenizer(eos_token_id=first))
        assert hm.eos_token_ids == {first}
        stopped = hm.generate(PROMPT, max_new_tokens=5)
        assert stopped.token_ids == [first] and stopped.stop_reason == "eos"
        full = hm.generate(PROMPT, max_new_tokens=5, stop_at_eos=False)
        assert len(full.token_ids) == 5 and full.stop_reason == "max_new_tokens"
        stop = full.token_ids[-1]
        custom = hm.generate(PROMPT, max_new_tokens=5, stop_token_ids=[stop])
        assert custom.token_ids == full.token_ids[: full.token_ids.index(stop) + 1]
        assert custom.stop_reason == "eos"

    def test_eos_ids_merge_tokenizer_and_generation_config(self):
        model = make_tiny_gpt2()
        model.generation_config.eos_token_id = [7, 9]
        assert HookedModel(model, WordTokenizer(eos_token_id=0)).eos_token_ids == {0, 7, 9}

    def test_sampling_is_seeded(self):
        hm = HookedModel(make_tiny_gpt2(), WordTokenizer())
        kw = {"max_new_tokens": 8, "temperature": 1.5, "stop_at_eos": False}
        a = hm.generate(PROMPT, seed=11, **kw)
        b = hm.generate(PROMPT, seed=11, **kw)
        assert a.token_ids == b.token_ids
        greedy = hm.generate(PROMPT, max_new_tokens=8, stop_at_eos=False)
        top1 = hm.generate(PROMPT, seed=3, top_k=1, **kw)
        assert top1.token_ids == greedy.token_ids
        assert hm.generate(PROMPT, return_logprobs=False, max_new_tokens=2).logprobs is None
        assert hm.generate(PROMPT, max_new_tokens=0).text == ""


# ===========================================================================
# Gradients
# ===========================================================================


class TestGradients:
    def test_gradients_flow_to_captured_residuals(self, hm: HookedModel):
        # Reference: plain autograd.grad on the live tensors of a separate pass
        # (retain_grad tensors also accumulate .grad under autograd.grad).
        ref = hm.forward(PROMPT, grad=True)
        target = int(ref.logits[0, -1].argmax())
        auto = torch.autograd.grad(ref.logits[0, -1, target], ref.resid_post_live)

        res = hm.forward(PROMPT, grad=True)
        assert res.grad_enabled and res.resid_post.requires_grad
        assert res.resid_post.grad_fn is not None
        loss = res.logits[0, -1, target]
        loss.backward()
        g_post = res.grad("resid_post")
        g_pre = res.grad("resid_pre")
        assert g_post.shape == g_pre.shape == (N_LAYER, len(PROMPT.split()), N_EMBD)
        for layer in range(hm.n_layers):
            assert float(g_post[layer, -1].abs().sum()) > 0
            torch.testing.assert_close(g_post[layer], auto[layer][0])
        # Same tensor feeds the next block -> same gradient.
        torch.testing.assert_close(g_post[0], g_pre[1])

    def test_gradients_with_frozen_parameters(self):
        hm = HookedModel(make_tiny_gpt2(), WordTokenizer())
        for p in hm.model.parameters():
            p.requires_grad_(False)
        res = hm.forward(PROMPT, grad=True)
        assert res.resid_pre_live[0].is_leaf and res.resid_pre_live[0].requires_grad
        hm.logit_lens(res.resid_post_live[-1][0, -1])[5].backward()
        assert float(res.grad("resid_pre")[0].abs().sum()) > 0

    def test_hook_edits_are_differentiable(self):
        hm = HookedModel(make_tiny_gpt2(), WordTokenizer())
        alpha = torch.tensor(2.0, requires_grad=True)
        v = _steer_vector(hm.d_model, scale=1.0)
        res = hm.forward(
            PROMPT, grad=True, hooks=[ResidHook(0, lambda h: h + alpha * v, positions=[-1])]
        )
        res.logits[0, -1, 3].backward()
        assert alpha.grad is not None and float(alpha.grad.abs()) > 0

    def test_grad_requires_grad_mode(self):
        hm = HookedModel(make_tiny_gpt2(), WordTokenizer())
        res = hm.forward(PROMPT)
        with pytest.raises(RuntimeError, match="grad=True"):
            res.grad()
        with pytest.raises(ValueError, match="site must be"):
            res.grad("attn_out")  # type: ignore[arg-type]


# ===========================================================================
# Tokenisation / inputs
# ===========================================================================


class TestTokenize:
    def test_plain_and_chat(self):
        hm = HookedModel(make_tiny_gpt2(), WordTokenizer(chat_template="{{ messages }}"))
        plain = hm.tokenize(PROMPT)
        assert plain.input_ids.shape == (1, len(PROMPT.split())) and not plain.chat_applied
        chat = hm.tokenize("hello there", chat=True)
        assert chat.chat_applied and chat.text == "<user> hello there <assistant>"
        assert chat.token_ids == WordTokenizer().encode_ids(chat.text)
        assert hm.tokenizer.calls[-1].get("add_special_tokens") is False
        msgs = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]
        assert hm.tokenize(msgs, chat=True).text == "<system> be brief <user> hi <assistant>"
        g = hm.generate("hello there", chat=True, max_new_tokens=2, stop_at_eos=False)
        assert g.prompt_token_ids == chat.token_ids

    def test_chat_without_template_falls_back_to_plain(self):
        hm = HookedModel(make_tiny_gpt2(), WordTokenizer())
        tp = hm.tokenize(PROMPT, chat=True)
        assert not tp.chat_applied and tp.token_ids == WordTokenizer().encode_ids(PROMPT)
        with pytest.raises(ValueError, match="chat_template"):
            hm.tokenize([{"role": "user", "content": "hi"}], chat=True)

    def test_id_inputs(self):
        hm = HookedModel(make_tiny_gpt2(), WordTokenizer())
        ref = hm.forward(PROMPT).logits
        ids = WordTokenizer().encode_ids(PROMPT)
        for given in (ids, torch.tensor(ids), torch.tensor([ids]), hm.tokenize(PROMPT)):
            torch.testing.assert_close(hm.forward(given).logits, ref)
        torch.testing.assert_close(hm(PROMPT).logits, ref)
        with pytest.raises(ValueError, match="single sequence"):
            hm.forward(torch.tensor([ids, ids]))
        with pytest.raises(ValueError, match="empty"):
            hm.forward([])


# ===========================================================================
# Loading
# ===========================================================================


class TestLoading:
    def test_resolve_dtype(self):
        assert resolve_dtype(None) is None
        assert resolve_dtype("bf16") is torch.bfloat16
        assert resolve_dtype("torch.float16") is torch.float16
        assert resolve_dtype(torch.float32) is torch.float32
        assert resolve_dtype("int3", strict=False) is None
        with pytest.raises(ValueError, match="unknown dtype"):
            resolve_dtype("int3")

    def test_from_pretrained_wires_kwargs(self, monkeypatch):
        calls: dict[str, Any] = {}

        def tok_fp(name: str, **kw: Any) -> WordTokenizer:
            calls["tok"] = (name, kw)
            return WordTokenizer(pad_token_id=None)

        def model_fp(name: str, **kw: Any) -> Any:
            calls["model"] = (name, kw)
            return make_tiny_gpt2()

        tf = _tf()
        monkeypatch.setattr(tf, "AutoTokenizer", SimpleNamespace(from_pretrained=tok_fp))
        monkeypatch.setattr(tf, "AutoModelForCausalLM", SimpleNamespace(from_pretrained=model_fp))
        hm = HookedModel.from_pretrained("org/m", device="cpu", dtype="fp32", local_files_only=True)
        assert calls["tok"] == ("org/m", {"local_files_only": True})
        assert calls["model"] == (
            "org/m",
            {"local_files_only": True, "dtype": torch.float32, "attn_implementation": "eager"},
        )
        assert hm.family == "gpt2" and hm.device == "cpu"
        assert hm.tokenizer.pad_token == "<eos>"
        assert not hm.model.training

    def test_load_without_attn_implementation(self, monkeypatch):
        seen: list[dict[str, Any]] = []
        tf = _tf()
        monkeypatch.setattr(
            tf, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda name, **kw: WordTokenizer())
        )
        monkeypatch.setattr(
            tf,
            "AutoModelForCausalLM",
            SimpleNamespace(from_pretrained=lambda name, **kw: seen.append(kw) or make_tiny_gpt2()),
        )
        model, tok, dev = load_hf_model("m", device="cpu", attn_implementation=None)
        assert seen == [{}] and dev == "cpu" and isinstance(tok, WordTokenizer)


# ===========================================================================
# Real GPT-2 (only when already cached locally — never downloads)
# ===========================================================================


@pytest.fixture(scope="module")
def real_gpt2() -> HookedModel:
    try:
        return HookedModel.from_pretrained("gpt2", device="cpu", local_files_only=True)
    except Exception as exc:  # OSError when not cached; anything else = unusable here
        pytest.skip(f"gpt2 weights not available offline: {exc}")


class TestRealGPT2:
    PROMPT = "The capital of France is"

    def test_architecture_and_eager_attention(self, real_gpt2: HookedModel):
        assert real_gpt2.family == "gpt2"
        assert (real_gpt2.n_layers, real_gpt2.d_model, real_gpt2.n_heads) == (12, 768, 12)
        assert real_gpt2.attn_implementation == "eager"
        res = real_gpt2.forward(self.PROMPT)
        assert res.attentions is not None and res.attentions.shape[:2] == (12, 12)

    def test_logit_lens_on_last_layer_matches_model_logits(self, real_gpt2: HookedModel):
        res = real_gpt2.forward(self.PROMPT)
        lens = real_gpt2.logit_lens(res.resid_post[-1])
        torch.testing.assert_close(lens, res.logits[0], atol=2e-3, rtol=1e-4)
        # The double-norm reading (ln_f applied to HF's already-normed last
        # hidden state) is measurably different — the bug this convention fixes.
        with torch.no_grad():
            hf = real_gpt2.model(input_ids=res.input_ids, output_hidden_states=True)
            double = real_gpt2.model.lm_head(real_gpt2.final_norm(hf.hidden_states[-1][0]))
        assert float((double - res.logits[0]).abs().max()) > 1.0

    def test_xray_last_layer_top1_is_the_models_next_token(self, real_gpt2: HookedModel):
        from LLmThoughtLens.xray_core import run_xray_loop

        events: list[tuple[str, dict[str, Any]]] = []
        run_xray_loop(
            real_gpt2.model,
            real_gpt2.tokenizer,
            "cpu",
            self.PROMPT,
            3,
            emit=lambda k, d: events.append((k, d)),
        )
        steps = [d for k, d in events if k == "xray_step"]
        assert len(steps) == 3
        ids = real_gpt2.tokenize(self.PROMPT).input_ids
        for step in steps:
            with torch.no_grad():
                probs = torch.softmax(real_gpt2.model(input_ids=ids).logits[0, -1], -1)
            nxt = int(probs.argmax())
            top_tok, top_p = step["logit_lens"][-1]["top"][0]
            assert step["token"] == real_gpt2.token_str(nxt) == top_tok
            assert top_p == pytest.approx(float(probs[nxt]), rel=1e-3)
            assert len(step["grid"]) == 12 and len(step["logit_lens"]) == 12
            ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)

    def test_greedy_generation_matches_manual_loop(self, real_gpt2: HookedModel):
        g = real_gpt2.generate(self.PROMPT, max_new_tokens=5, stop_at_eos=False)
        ref, _ = _manual_greedy(real_gpt2.model, real_gpt2.tokenize(self.PROMPT).token_ids, 5)
        assert g.token_ids == ref

    def test_provider_load_emits_no_generation_flag_warning(self, real_gpt2: HookedModel, capfd):
        from LLmThoughtLens.providers.huggingface_provider import HuggingFaceProvider

        tf = _tf()
        real_fp = tf.AutoModelForCausalLM.from_pretrained
        tok_fp = tf.AutoTokenizer.from_pretrained
        p = HuggingFaceProvider(model_name="gpt2", device="cpu")
        # Offline: force cache-only lookups without touching global HF settings.
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                tf,
                "AutoModelForCausalLM",
                SimpleNamespace(
                    from_pretrained=lambda n, **kw: real_fp(n, local_files_only=True, **kw)
                ),
            )
            mp.setattr(
                tf,
                "AutoTokenizer",
                SimpleNamespace(
                    from_pretrained=lambda n, **kw: tok_fp(n, local_files_only=True, **kw)
                ),
            )
            p._load()
        err = capfd.readouterr().err
        assert "generation flags" not in err
        assert p.hooked.attn_implementation == "eager"

    def test_provider_activations_are_true_residual(self, real_gpt2: HookedModel):
        from LLmThoughtLens.providers.huggingface_provider import HuggingFaceProvider

        p = HuggingFaceProvider(model_name="gpt2", device="cpu")
        p._model, p._tokenizer, p._device = real_gpt2.model, real_gpt2.tokenizer, "cpu"
        out = p.run(self.PROMPT)
        lens = real_gpt2.logit_lens(torch.from_numpy(out.activations[-1][-1])).detach().numpy()
        np.testing.assert_allclose(lens, out.logits, atol=2e-3)
        assert out.embeddings is not None and out.embeddings.shape == (5, 768)


# ===========================================================================
# Generic / low-level behaviour on a hand-written toy LM
# ===========================================================================


class _ToyBlock(torch.nn.Module):
    """Block called with ``hidden_states=`` as a keyword and returning a tuple."""

    def __init__(self, d: int, with_attn: bool = True) -> None:
        super().__init__()
        if with_attn:
            self.self_attn = torch.nn.Linear(d, d)
        self.mlp = torch.nn.Linear(d, d)

    def forward(self, hidden_states: Any) -> Any:  # type: ignore[override]
        mixed = self.self_attn(hidden_states) if hasattr(self, "self_attn") else hidden_states
        return (hidden_states + self.mlp(torch.tanh(mixed)), "aux")


class _ToyLM(torch.nn.Module):
    """No config, no position ids, no KV cache: exercises every generic fallback."""

    def __init__(self, d: int = 8, n: int = 3, skip_last: bool = False) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(VOCAB, d)
        self.model = SimpleNamespace(layers=torch.nn.ModuleList([_ToyBlock(d) for _ in range(n)]))
        self.blocks_ = self.model.layers  # registered so parameters()/named_modules see them
        self.head = torch.nn.Linear(d, VOCAB)
        self.skip_last = skip_last

    def get_output_embeddings(self) -> Any:
        return self.head

    def forward(self, input_ids: Any = None, **_kw: Any) -> Any:  # type: ignore[override]
        h = self.embed(input_ids)
        layers = list(self.model.layers)[: -1 if self.skip_last else None]
        for block in layers:
            h = block(hidden_states=h)[0]
        return SimpleNamespace(logits=self.head(h), attentions=None, past_key_values=None)


class _ListTokenizer(WordTokenizer):
    def __call__(self, prompt: str, return_tensors: str | None = None, **kw: Any) -> Any:
        return {"input_ids": self.encode_ids(prompt)}  # plain 1-D list


class TestGenericToyModel:
    def test_generic_resolution_hooks_and_generation(self):
        model = _ToyLM()
        hm = HookedModel(model, _ListTokenizer())
        assert hm.family == "generic" and hm.final_norm is None and hm.n_layers == 3
        assert hm.device == torch.device("cpu")
        assert hm.attn_implementation is None and hm.returns_attention_weights
        assert hm.tokenize(PROMPT).input_ids.shape == (1, len(PROMPT.split()))
        res = hm.forward(PROMPT)
        assert res.attentions is None
        assert hm.logit_lens_error(PROMPT) < 1e-5
        torch.testing.assert_close(res.resid_post[0], res.resid_pre[1])

        # Keyword-only block input + tuple block output are both hookable.
        v = torch.ones(8)
        edited = hm.forward(PROMPT, hooks=[ResidHook(1, lambda h: h + v, site="resid_pre")])
        torch.testing.assert_close(edited.resid_pre[1], res.resid_pre[1] + v)
        edited = hm.forward(PROMPT, hooks=[ResidHook(1, lambda h: h + v, site="resid_post")])
        torch.testing.assert_close(edited.resid_post[1], res.resid_post[1] + v)

        # No cache returned -> generation silently recomputes the full sequence;
        # no position ids -> hooks fall back to HookedModel's own offsets.
        offsets: list[int] = []

        def spy(h: Any) -> None:
            offsets.append(hm.current_offset)

        g = hm.generate(
            PROMPT, max_new_tokens=3, stop_at_eos=False, hooks=[ResidHook(0, spy, positions=0)]
        )
        ref, _ = _manual_greedy(model, WordTokenizer().encode_ids(PROMPT), 3)
        assert g.token_ids == ref
        assert offsets == [0, 0, 0]

    def test_unavailable_site_and_missing_unembed(self):
        model = _ToyLM()
        for block in model.model.layers:
            del block.self_attn
        hm = HookedModel(model, WordTokenizer())
        with pytest.raises(ValueError, match="not available"):
            hm.site_module(0, "attn_out")
        hm.arch.unembed = None
        with pytest.raises(RuntimeError, match="no output embedding"):
            hm.logit_lens(torch.zeros(8))

    def test_logit_lens_casts_inputs(self):
        hm = HookedModel(_ToyLM(), WordTokenizer())
        res = hm.forward(PROMPT)
        lens = hm.logit_lens(res.resid_post[-1].double().numpy())
        assert lens.dtype == torch.float32
        torch.testing.assert_close(lens, res.logits[0], atol=1e-5, rtol=1e-5)

    def test_skipped_block_cannot_be_captured(self):
        hm = HookedModel(_ToyLM(skip_last=True), WordTokenizer())
        with pytest.raises(RuntimeError, match="did not run"):
            hm.forward(PROMPT)
        assert hm.forward(PROMPT, capture=False).logits.shape[-1] == VOCAB


class TestRegisterSiteHook:
    def test_pre_hook_edge_cases(self):
        from LLmThoughtLens.models import register_site_hook

        calls: list[Any] = []

        class _M(torch.nn.Module):
            def forward(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
                calls.append((args, kwargs))
                return args[0] if args else kwargs.get("x")

        m = _M()
        h = register_site_hook(m, "pre", lambda t: t * 2)
        m("not a tensor")  # non-tensor input: untouched
        m(x=torch.ones(1))  # neither positional nor hidden_states: untouched
        assert m(hidden_states=torch.ones(1)) is None and calls[-1][1]["hidden_states"].item() == 2
        h.remove()
        with pytest.raises(ValueError, match="'pre' or 'post'"):
            register_site_hook(m, "sideways", lambda t: t)  # type: ignore[arg-type]

    def test_post_hook_edge_cases(self):

        class _Out(torch.nn.Module):
            def __init__(self, value: Any) -> None:
                super().__init__()
                self.value = value

            def forward(self, x: Any) -> Any:  # type: ignore[override]
                return self.value

        same = lambda t: t  # noqa: E731
        assert _hook_and_call(_Out(("a", 1)), same) == ("a", 1)  # non-tensor head
        assert _hook_and_call(_Out(()), same) == ()
        t = torch.ones(2)
        assert _hook_and_call(_Out((t, 5)), same)[0] is t  # identity -> unchanged
        doubled = _hook_and_call(_Out((t, 5)), lambda x: x * 2)
        assert doubled[1] == 5 and float(doubled[0].sum()) == 4.0
        with pytest.raises(TypeError, match="cannot hook output"):
            _hook_and_call(_Out({"logits": t}), same)

    def test_infer_device_fallbacks(self):
        from LLmThoughtLens.models import infer_device

        assert infer_device(torch.nn.Identity()) == torch.device("cpu")
        assert infer_device(SimpleNamespace()) == torch.device("cpu")

    def test_load_errors_without_attn_implementation_propagate(self, monkeypatch):
        tf = _tf()
        monkeypatch.setattr(
            tf, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda n, **kw: WordTokenizer())
        )

        def boom(name: str, **kw: Any) -> Any:
            raise ValueError("bad config")

        monkeypatch.setattr(tf, "AutoModelForCausalLM", SimpleNamespace(from_pretrained=boom))
        with pytest.raises(ValueError, match="bad config"):
            load_hf_model("m", device="cpu", attn_implementation=None)


def _hook_and_call(module: Any, fn: Any) -> Any:
    from LLmThoughtLens.models import register_site_hook

    handle = register_site_hook(module, "post", fn)
    try:
        return module(torch.zeros(1))
    finally:
        handle.remove()
