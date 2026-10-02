"""Offline tests for the white-box :class:`HuggingFaceProvider`.

A tiny randomly initialised GPT-2 (built from a config, no downloads) and a
deterministic word tokenizer are injected into the provider's private
``_model`` / ``_tokenizer`` / ``_device`` slots, so ``_load`` is a no-op and
every assertion is checked against the model's own forward pass.

``activations`` must be the *true* residual stream (block outputs captured
independently here with forward hooks), not HF's ``output_hidden_states``,
whose last entry is already passed through ``ln_f``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from LLmThoughtLens.features.intervention import FeatureIntervention  # noqa: E402
from LLmThoughtLens.models import HookedModel  # noqa: E402
from LLmThoughtLens.providers.huggingface_provider import (  # noqa: E402
    HuggingFaceProvider,
    _register_intervention_hook,
    _resolve_transformer_blocks,
)
from LLmThoughtLens.utils.math_utils import softmax  # noqa: E402

from _tiny_hf import N_EMBD, N_HEAD, N_LAYER, VOCAB, WordTokenizer, make_provider  # noqa: E402

PROMPT = "the capital of France is"


def _reference_forward(provider: HuggingFaceProvider, prompt: str) -> Any:
    ids = torch.tensor([provider._tokenizer.encode_ids(prompt)])
    with torch.no_grad():
        return provider._model(
            input_ids=ids, output_hidden_states=True, output_attentions=True, use_cache=False
        )


def _block_outputs(provider: HuggingFaceProvider, prompt: str) -> np.ndarray:
    """Independent capture of every GPT-2 block's output: the true residual stream."""
    ids = torch.tensor([provider._tokenizer.encode_ids(prompt)])
    outs: list[Any] = []
    handles = [
        b.register_forward_hook(lambda _m, _a, o: outs.append(o[0] if isinstance(o, tuple) else o))
        for b in provider._model.transformer.h
    ]
    try:
        with torch.no_grad():
            provider._model(input_ids=ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return torch.stack(outs).squeeze(1).numpy()


# ---------------------------------------------------------------------------
# Construction / identity / device + dtype resolution
# ---------------------------------------------------------------------------


class TestIdentityAndResolution:
    def test_identity(self):
        p = HuggingFaceProvider(model_name="org/some-model", device="cpu")
        assert p.name == "huggingface"
        assert p.model_id == "hf/org/some-model"
        assert p.evidence_kind == "white_box"
        assert p.supports_internals is True
        assert p._model is None and p._tokenizer is None  # lazy

    def test_explicit_device_is_returned_verbatim(self):
        assert HuggingFaceProvider(device="cuda:3")._resolve_device() == "cuda:3"

    @pytest.mark.parametrize(
        ("cuda", "mps", "expected"),
        [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")],
    )
    def test_auto_device_priority(self, monkeypatch, cuda, mps, expected):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
        assert HuggingFaceProvider(device="auto")._resolve_device() == expected

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            (None, None),
            ("float16", "float16"),
            ("fp16", "float16"),
            ("bfloat16", "bfloat16"),
            ("bf16", "bfloat16"),
            ("float32", "float32"),
            ("fp32", "float32"),
            ("int3", None),
        ],
    )
    def test_dtype_resolution(self, name, expected):
        got = HuggingFaceProvider(torch_dtype=name)._resolve_dtype()
        assert got == (None if expected is None else getattr(torch, expected))


class TestLazyLoad:
    def test_load_is_noop_when_model_injected(self):
        p = make_provider()
        model = p._model
        p._load()
        assert p._model is model

    def test_load_wires_tokenizer_model_device_and_dtype(self, monkeypatch):
        import transformers

        calls: dict[str, Any] = {}

        class FakeTok:
            pad_token_id = None
            eos_token_id = 7
            eos_token = "<eos>"
            pad_token = None

        class FakeModel:
            def to(self, device: Any) -> FakeModel:
                calls["to"] = device
                return self

            def eval(self) -> FakeModel:
                calls["eval"] = True
                return self

        def tok_from_pretrained(name: str, **kwargs: Any) -> FakeTok:
            calls["tok_name"] = name
            calls["tok_kwargs"] = kwargs
            return FakeTok()

        def model_from_pretrained(name: str, **kwargs: Any) -> FakeModel:
            calls["model_name"] = name
            calls["model_kwargs"] = kwargs
            return FakeModel()

        monkeypatch.setattr(
            transformers, "AutoTokenizer", SimpleNamespace(from_pretrained=tok_from_pretrained)
        )
        monkeypatch.setattr(
            transformers,
            "AutoModelForCausalLM",
            SimpleNamespace(from_pretrained=model_from_pretrained),
        )

        p = HuggingFaceProvider(
            model_name="/local/weights", device="cpu", torch_dtype="bf16", capture_internals=False
        )
        p._load()

        assert calls["tok_name"] == "/local/weights"
        assert calls["tok_kwargs"] == {}
        assert calls["model_name"] == "/local/weights"
        # Only construction kwargs: never the output_* forward flags (transformers 5
        # warns they are invalid generation flags), and no eager attention when
        # internals are not captured.
        assert calls["model_kwargs"] == {"dtype": torch.bfloat16}
        assert calls["to"] == "cpu"
        assert calls["eval"] is True
        assert p._device == "cpu"
        # Missing pad token falls back to EOS so batching never crashes.
        assert p._tokenizer.pad_token == "<eos>"

    @staticmethod
    def _patch_loaders(monkeypatch, model_from_pretrained: Any) -> None:
        import transformers

        class FakeTok:
            pad_token_id = 0
            eos_token_id = 0
            eos_token = "<eos>"

        monkeypatch.setattr(
            transformers,
            "AutoTokenizer",
            SimpleNamespace(from_pretrained=lambda name, **kw: FakeTok()),
        )
        monkeypatch.setattr(
            transformers,
            "AutoModelForCausalLM",
            SimpleNamespace(from_pretrained=model_from_pretrained),
        )

    def test_capture_internals_requests_eager_attention(self, monkeypatch):
        from _tiny_hf import make_tiny_gpt2

        seen: list[dict[str, Any]] = []

        def model_from_pretrained(name: str, **kwargs: Any) -> Any:
            seen.append(kwargs)
            return make_tiny_gpt2()

        self._patch_loaders(monkeypatch, model_from_pretrained)
        p = HuggingFaceProvider(model_name="m", device="cpu")
        p._load()
        assert seen == [{"attn_implementation": "eager"}]

    def test_model_rejecting_eager_attention_falls_back_to_default(self, monkeypatch):
        from _tiny_hf import make_tiny_gpt2

        seen: list[dict[str, Any]] = []

        def model_from_pretrained(name: str, **kwargs: Any) -> Any:
            seen.append(dict(kwargs))
            if "attn_implementation" in kwargs:
                raise ValueError("this model does not support eager attention")
            return make_tiny_gpt2()

        self._patch_loaders(monkeypatch, model_from_pretrained)
        p = HuggingFaceProvider(model_name="m", device="cpu", torch_dtype="fp32")
        p._load()
        assert seen == [
            {"dtype": torch.float32, "attn_implementation": "eager"},
            {"dtype": torch.float32},
        ]
        assert p._model is not None

    def test_unrelated_load_errors_propagate(self, monkeypatch):
        def model_from_pretrained(name: str, **kwargs: Any) -> Any:
            raise OSError("weights not found")

        self._patch_loaders(monkeypatch, model_from_pretrained)
        with pytest.raises(OSError, match="weights not found"):
            HuggingFaceProvider(model_name="m", device="cpu")._load()


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


class TestForward:
    def test_run_exposes_real_internals_matching_model(self):
        p = make_provider()
        out = p.run(PROMPT)
        ref = _reference_forward(p, PROMPT)
        n_tok = len(PROMPT.split())

        assert out.evidence_kind == "white_box"
        assert out.has_internals
        assert out.token_ids == p._tokenizer.encode_ids(PROMPT)
        assert out.tokens == [f"<{i}>" for i in out.token_ids]

        assert out.activations.shape == (N_LAYER, n_tok, N_EMBD)
        assert out.activations.dtype == np.float32
        # True residual stream = every block's output, the last one included.
        np.testing.assert_allclose(out.activations, _block_outputs(p, PROMPT), atol=1e-5)
        # HF hidden_states agree on every layer except the last, where HF has
        # already applied ln_f — the provider must NOT expose that normed copy.
        hs = torch.stack(ref.hidden_states).squeeze(1).numpy()
        np.testing.assert_allclose(out.activations[:-1], hs[1:-1], atol=1e-5)
        assert not np.allclose(out.activations[-1], hs[-1], atol=1e-3)
        with torch.no_grad():
            normed = p._model.transformer.ln_f(torch.from_numpy(out.activations[-1])).numpy()
        np.testing.assert_allclose(normed, hs[-1], atol=1e-5)
        # Embedding output entering block 0.
        assert out.embeddings is not None and out.embeddings.shape == (n_tok, N_EMBD)
        np.testing.assert_allclose(out.embeddings, hs[0], atol=1e-6)

        assert out.attentions.shape == (N_LAYER, N_HEAD, n_tok, n_tok)
        np.testing.assert_allclose(out.attentions.sum(-1), 1.0, atol=1e-5)
        # Causal mask: no attention to future positions.
        assert np.allclose(np.triu(out.attentions[0, 0], k=1), 0.0, atol=1e-6)

        assert out.logits.shape == (VOCAB,)
        np.testing.assert_allclose(out.logits, ref.logits[0, -1].numpy(), rtol=1e-5, atol=1e-5)

    def test_top_tokens_are_softmax_top5_sorted(self):
        p = make_provider()
        out = p.run(PROMPT)
        probs = softmax(out.logits)
        expected_ids = np.argsort(-probs)[:5]
        assert [t for t, _ in out.top_tokens] == [f"<{i}>" for i in expected_ids]
        np.testing.assert_allclose([pr for _, pr in out.top_tokens], probs[expected_ids], rtol=1e-5)
        assert out.output_token == f"<{expected_ids[0]}>"

    def test_meta(self):
        out = make_provider().run(PROMPT)
        assert out.meta["provider"] == "HuggingFaceProvider"
        assert out.meta["model"] == "tiny-gpt2"
        assert out.meta["device"] == "cpu"
        assert out.meta["family"] == "gpt2"
        assert out.meta["activation_site"] == "resid_post"
        assert out.meta["n_intervention_hooks"] == 0
        assert "forward hooks" in out.meta["evidence_note"]
        assert "no final norm" in out.meta["evidence_note"]

    def test_deterministic(self):
        p = make_provider()
        a, b = p.run(PROMPT), p.run(PROMPT)
        np.testing.assert_array_equal(a.activations, b.activations)
        np.testing.assert_array_equal(a.logits, b.logits)

    def test_capture_internals_false_keeps_logits_only(self):
        p = make_provider(capture_internals=False)
        out = p.run(PROMPT)
        assert out.activations is None
        assert out.attentions is None
        assert out.embeddings is None
        assert out.logits is not None and out.logits.shape == (VOCAB,)
        assert out.top_tokens


class TestHookedAccess:
    def test_hooked_wraps_injected_model_and_is_cached(self):
        p = make_provider()
        hm = p.hooked
        assert isinstance(hm, HookedModel)
        assert hm.model is p._model and hm.tokenizer is p._tokenizer
        assert hm.family == "gpt2" and hm.n_layers == N_LAYER and hm.d_model == N_EMBD
        assert p.hooked is hm  # cached

    def test_hooked_rebuilds_when_model_is_replaced(self):
        from _tiny_hf import make_tiny_gpt2

        p = make_provider()
        first = p.hooked
        p._model = make_tiny_gpt2(seed=1)
        assert p.hooked is not first
        assert p.hooked.model is p._model

    def test_hooked_triggers_lazy_load(self, monkeypatch):
        from _tiny_hf import install_tiny_loader

        loaded = install_tiny_loader(monkeypatch)
        p = HuggingFaceProvider(model_name="tiny-gpt2", device="cpu")
        assert p._model is None
        assert p.hooked.model is p._model is not None
        assert loaded == [p]

    def test_supports_gradients_flag(self):
        assert HuggingFaceProvider.supports_gradients is True
        from LLmThoughtLens.providers.mock_provider import MockProvider

        assert MockProvider().supports_gradients is False

    def test_logit_lens_on_last_activation_reproduces_logits(self):
        p = make_provider(perturb_norms=True)
        out = p.run(PROMPT)
        with torch.no_grad():
            lens = p.hooked.logit_lens(torch.from_numpy(out.activations[-1][-1])).numpy()
        np.testing.assert_allclose(lens, out.logits, atol=1e-4)


class TestInterventions:
    def test_no_interventions_matches_plain_run(self):
        p = make_provider()
        base = p.run(PROMPT)
        for empty in (None, []):
            out = p.run_with_intervention(PROMPT, empty)
            np.testing.assert_array_equal(out.logits, base.logits)
            assert out.meta["n_intervention_hooks"] == 0

    def test_intervention_installs_hook_changes_forward_and_cleans_up(self):
        p = make_provider()
        base = p.run(PROMPT)
        spec = FeatureIntervention.clamp(feature_id=3, value=25.0, layer=0)
        out = p.run_with_intervention(PROMPT, [spec])

        assert out.meta["n_intervention_hooks"] == 1
        # The hook edits block 0's MLP input, so block 0's output residual and the
        # final logits both move — the intervention is a real causal edit.
        assert not np.allclose(out.activations[0], base.activations[0])
        assert not np.allclose(out.logits, base.logits)
        # Hooks are removed afterwards: a plain run reproduces the baseline exactly.
        np.testing.assert_array_equal(p.run(PROMPT).logits, base.logits)
        assert all(len(b.mlp._forward_pre_hooks) == 0 for b in p._model.transformer.h)

    @pytest.mark.parametrize("site", ["resid_pre", "resid_post"])
    def test_residual_site_interventions(self, site):
        p = make_provider()
        base = p.run(PROMPT)
        blocks = p._model.transformer.h

        def hook_ids() -> list[set[int]]:
            # transformers 5 keeps its own persistent output-capturing hooks on
            # blocks; only hooks *we* add must be gone afterwards.
            return [set(b._forward_hooks) | set(b._forward_pre_hooks) for b in blocks]

        before = hook_ids()
        spec = FeatureIntervention.clamp(feature_id=5, value=40.0, layer=1, site=site)
        out = p.run_with_intervention(PROMPT, [spec])
        assert out.meta["n_intervention_hooks"] == 1
        # Layer 0 is upstream of the edit and unchanged.
        np.testing.assert_allclose(out.activations[0], base.activations[0], atol=1e-6)
        assert not np.allclose(out.logits, base.logits)
        if site == "resid_post":
            # The edited block output is exactly what activations record.
            np.testing.assert_allclose(out.activations[1][:, 5], 40.0, atol=1e-4)
        # Hooks are removed afterwards.
        np.testing.assert_array_equal(p.run(PROMPT).logits, base.logits)
        assert hook_ids() == before

    def test_register_intervention_hook_shim(self):
        p = make_provider()
        blocks = _resolve_transformer_blocks(p._model)
        handle = _register_intervention_hook(blocks, FeatureIntervention.inhibit(0, layer=1))
        assert len(blocks[1].mlp._forward_pre_hooks) == 1
        handle.remove()
        assert len(blocks[1].mlp._forward_pre_hooks) == 0
        assert _register_intervention_hook([], FeatureIntervention.inhibit(0)) is None


# ---------------------------------------------------------------------------
# Transformer-block resolution across architectures
# ---------------------------------------------------------------------------


class TestResolveTransformerBlocks:
    def test_gpt2(self):
        p = make_provider()
        blocks = _resolve_transformer_blocks(p._model)
        assert len(blocks) == N_LAYER
        assert blocks[0] is p._model.transformer.h[0]

    @pytest.mark.parametrize(
        "path",
        [
            ("model", "layers"),
            ("gpt_neox", "layers"),
            ("transformer", "blocks"),
            ("model", "decoder", "layers"),
        ],
    )
    def test_known_attribute_paths(self, path):
        leaf: Any = ["b0", "b1", "b2"]
        obj: Any = leaf
        for attr in reversed(path):
            obj = SimpleNamespace(**{attr: obj})
        assert _resolve_transformer_blocks(obj) == leaf

    def test_fallback_scans_named_modules_for_block_like_modulelist(self):
        nn = torch.nn

        class Block(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.self_attn = nn.Identity()
                self.mlp = nn.Identity()

        class Exotic(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.misc = nn.ModuleList([nn.Linear(2, 2)])  # not block-like
                self.stack = nn.ModuleList([Block(), Block(), Block()])

        model = Exotic()
        blocks = _resolve_transformer_blocks(model)
        assert len(blocks) == 3
        assert blocks[2] is model.stack[2]

    def test_returns_empty_when_nothing_matches(self):
        assert _resolve_transformer_blocks(torch.nn.Sequential(torch.nn.Linear(2, 2))) == []


# ---------------------------------------------------------------------------
# End-to-end through Scope
# ---------------------------------------------------------------------------


class TestScopeIntegration:
    def test_trace_full_on_tiny_hf_model(self):
        from LLmThoughtLens.scope import Scope

        p = make_provider()
        result = Scope(p, top_k_features=5).trace_full(PROMPT)
        assert result.evidence_kind == "white_box"
        assert result.output.n_layers == N_LAYER
        assert result.output_token == p.run(PROMPT).output_token
        assert 0 < len(result.features) <= 5 * N_LAYER
        assert all(0 <= f.layer < N_LAYER for f in result.features)
        assert result.graph.num_nodes > 0

    def test_eos_and_pad_ids_are_not_required_for_forward(self):
        p = make_provider(eos_token_id=None, pad_token_id=None)
        assert isinstance(p._tokenizer, WordTokenizer)
        assert p.run("hello").activations.shape == (N_LAYER, 1, N_EMBD)
