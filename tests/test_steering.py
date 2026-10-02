"""Tests for :mod:`LLmThoughtLens.features.steering`.

* The first section needs only NumPy (it also runs in the core-only CI job):
  construction, position resolution, save / load, SAE metadata parsing and the
  capability errors for providers without a HookedModel.
* The tiny-model sections build random 2-layer models from transformers
  configs (several families, no downloads) and compare every steering effect
  against an independent computation: the logit lens of the unsteered
  residual, a raw torch forward hook, ``inputs_embeds``, HF's
  ``output_hidden_states`` or a closed form.
* The last section runs real GPT-2 when it is already in the local HF cache
  (skipped otherwise) and checks a contrast vector measurably moves the
  next-token distribution and the completion.
"""

from __future__ import annotations

import json
import math
import warnings
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from LLmThoughtLens.features.steering import (
    POSITION_MODES,
    STEERING_SITES,
    SteeringMismatchError,
    SteeringUnavailableError,
    SteeringVector,
    as_hooked,
    coefficient_sweep,
    steer_generate,
    steering_hooks,
)
from LLmThoughtLens.providers.base import BaseProvider, ProviderOutput
from LLmThoughtLens.providers.mock_provider import MockProvider

PROMPT = "the quick brown fox jumps over"
T = len(PROMPT.split())


def _torch() -> Any:
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    return torch


def _vec(d: int = 8, seed: int = 0, scale: float = 1.0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(d).astype(np.float32) * scale


class _BlackBox(BaseProvider):
    evidence_kind = "black_box"

    @property
    def name(self) -> str:
        return "fakeapi"

    def run(self, prompt: str, **kwargs: Any) -> ProviderOutput:  # pragma: no cover
        raise AssertionError("steering must not call run()")


# ===========================================================================
# NumPy-only: construction, positions, persistence, SAE metadata, capability
# ===========================================================================


class TestConstruction:
    def test_defaults_and_casting(self):
        v = SteeringVector(direction=[1, 2, 2], layer=3)
        assert v.direction.dtype == np.float32
        assert v.direction.tolist() == [1.0, 2.0, 2.0]
        assert (v.d_model, v.site, v.coeff, v.normalize, v.positions) == (
            3,
            "resid_post",
            1.0,
            False,
            "all",
        )
        assert v.norm == pytest.approx(3.0)
        np.testing.assert_allclose(v.unit, [1 / 3, 2 / 3, 2 / 3], rtol=1e-6)
        assert set(STEERING_SITES) == {"resid_pre", "resid_post"}
        assert POSITION_MODES == ("all", "last", "prompt", "generated")

    def test_vector_scaling_raw_and_normalized(self):
        raw = SteeringVector(direction=[3, 4], layer=0, coeff=2.0)
        np.testing.assert_allclose(raw.vector(), [6, 8])
        np.testing.assert_allclose(raw.vector(-0.5), [-1.5, -2])
        unit = SteeringVector(direction=[3, 4], layer=0, coeff=10.0, normalize=True)
        np.testing.assert_allclose(unit.vector(), [6, 8], rtol=1e-6)
        assert float(np.linalg.norm(unit.vector(5.0))) == pytest.approx(5.0)
        assert raw.vector(0.0).tolist() == [0.0, 0.0]
        assert raw.with_coeff(3.0).coeff == 3.0 and raw.coeff == 2.0

    def test_positions_normalisation(self):
        assert SteeringVector(direction=[1.0], layer=0, positions=4).positions == (4,)
        assert SteeringVector(direction=[1.0], layer=0, positions=[3, -1]).positions == (3, -1)
        assert SteeringVector(direction=[1.0], layer=0, positions=np.int64(2)).positions == (2,)
        with pytest.raises(ValueError, match="unknown positions mode"):
            SteeringVector(direction=[1.0], layer=0, positions="first")
        with pytest.raises(ValueError, match="must not be empty"):
            SteeringVector(direction=[1.0], layer=0, positions=[])
        with pytest.raises(TypeError, match="bool"):
            SteeringVector(direction=[1.0], layer=0, positions=True)
        with pytest.raises(TypeError, match="sequence of ints"):
            SteeringVector(direction=[1.0], layer=0, positions=3.5j)

    @pytest.mark.parametrize(
        ("kwargs", "exc", "match"),
        [
            ({"direction": [[1.0, 2.0]]}, ValueError, "1-D"),
            ({"direction": []}, ValueError, "1-D"),
            ({"direction": [1.0, np.nan]}, ValueError, "NaN"),
            ({"direction": [1.0, 2.0], "d_model": 3}, SteeringMismatchError, "d_model=3"),
            ({"direction": [1.0], "site": "mlp_in"}, ValueError, "unknown steering site"),
            ({"direction": [0.0, 0.0], "normalize": True}, ValueError, "zero direction"),
            ({"direction": [1.0], "coeff": math.inf}, ValueError, "finite"),
            ({"direction": [1.0], "layer": 4, "n_layers": 4}, SteeringMismatchError, "range"),
            ({"direction": [1.0], "layer": -5, "n_layers": 4}, SteeringMismatchError, "range"),
        ],
    )
    def test_invalid(self, kwargs: dict[str, Any], exc: type[Exception], match: str):
        kwargs = {"layer": 0, **kwargs}
        with pytest.raises(exc, match=match):
            SteeringVector(**kwargs)

    def test_from_vector_and_repr(self):
        v = SteeringVector.from_vector([0.0, 1.0], 2, name="up", source={"note": "hand-made"})
        assert v.source == {"method": "explicit", "note": "hand-made"}
        assert "'up'" in repr(v) and "layer=2" in repr(v) and "positions='all'" in repr(v)
        assert "positions=[1, 2]" in repr(SteeringVector.from_vector([1.0], 0, positions=[1, 2]))
        assert v.to_dict(include_direction=True)["direction"] == [0.0, 1.0]


class TestResolvePositions:
    def test_named_modes(self):
        mk = lambda p: SteeringVector(direction=[1.0], layer=0, positions=p)  # noqa: E731
        assert mk("all").resolve_positions(5, 9) is None
        assert mk("last").resolve_positions(5, 9) == (4,)
        assert mk("prompt").resolve_positions(5, 9) == (0, 1, 2, 3, 4)
        assert mk("generated").resolve_positions(5, 9) == (4, 5, 6, 7, 8)
        # Without a total length "generated" is just the hinge position T-1.
        assert mk("generated").resolve_positions(5) == (4,)

    def test_explicit_indices_resolve_against_the_prompt(self):
        v = SteeringVector(direction=[1.0], layer=0, positions=[-1, 0, 7, -1])
        assert v.resolve_positions(5, 9) == (0, 4, 7)
        assert SteeringVector(direction=[1.0], layer=0, positions=-5).resolve_positions(5) == (0,)
        with pytest.raises(IndexError, match="before the start"):
            SteeringVector(direction=[1.0], layer=0, positions=-6).resolve_positions(5)
        with pytest.raises(ValueError, match="prompt_len"):
            SteeringVector(direction=[1.0], layer=0).resolve_positions(0)


class TestPersistence:
    def test_round_trip_is_exact(self, tmp_path):
        v = SteeringVector(
            direction=_vec(16, seed=3),
            layer=-2,
            site="resid_pre",
            coeff=-1.25,
            normalize=True,
            positions=[0, -1],
            name="contrast",
            source={"method": "contrast_mean_difference", "n": np.int64(3), "w": np.ones(2)},
            model_name="gpt2",
            n_layers=12,
        )
        path = v.save(tmp_path / "vec.steer")
        assert path == tmp_path / "vec.steer" and path.exists()  # no ".npz" appended
        w = SteeringVector.load(path)
        np.testing.assert_array_equal(w.direction, v.direction)
        for attr in ("layer", "site", "coeff", "normalize", "positions", "name", "model_name"):
            assert getattr(w, attr) == getattr(v, attr), attr
        assert (w.d_model, w.n_layers) == (16, 12)
        assert w.source == {"method": "contrast_mean_difference", "n": 3, "w": [1.0, 1.0]}
        json.dumps(w.metadata())  # JSON-safe

    def test_load_rejects_foreign_or_future_files(self, tmp_path):
        foreign = tmp_path / "foreign.npz"
        np.savez(foreign, x=np.zeros(3))
        with pytest.raises(ValueError, match="missing arrays"):
            SteeringVector.load(foreign)
        other = tmp_path / "other.npz"
        np.savez(other, direction=np.zeros(3), meta=np.array(json.dumps({"format": "x"})))
        with pytest.raises(ValueError, match="format="):
            SteeringVector.load(other)
        v = SteeringVector(direction=[1.0, 2.0], layer=0)
        meta = {**v.metadata(), "version": 99}
        future = tmp_path / "future.npz"
        np.savez(future, direction=v.direction, meta=np.array(json.dumps(meta)))
        with pytest.raises(ValueError, match="v99"):
            SteeringVector.load(future)

    def test_load_detects_tampered_d_model(self, tmp_path):
        v = SteeringVector(direction=[1.0, 2.0], layer=0)
        bad = tmp_path / "bad.npz"
        meta = {**v.metadata(), "d_model": 5}
        np.savez(bad, direction=v.direction, meta=np.array(json.dumps(meta)))
        with pytest.raises(SteeringMismatchError, match="d_model=5"):
            SteeringVector.load(bad)

    def test_unserialisable_source_fails_loudly(self, tmp_path):
        v = SteeringVector(direction=[1.0], layer=0, source={"obj": object()})
        with pytest.raises(TypeError, match="JSON"):
            v.save(tmp_path / "x.npz")


def _fake_sae(direction: np.ndarray, dict_size: int = 10, **meta: Any) -> Any:
    return SimpleNamespace(
        config=SimpleNamespace(dict_size=dict_size, input_dim=len(direction), **meta),
        feature_direction=lambda fid: direction / np.linalg.norm(direction),
        labels={2: "Golden Gate"},
    )


class TestFromSaeFeature:
    def test_layer_and_site_from_config(self):
        d = _vec(8)
        v = SteeringVector.from_sae_feature(
            _fake_sae(d, hook_layer=5, hook_site="resid_pre"), 2, coeff=4.0
        )
        assert (v.layer, v.site, v.d_model, v.normalize, v.name) == (
            5,
            "resid_pre",
            8,
            True,
            "Golden Gate",
        )
        np.testing.assert_allclose(v.direction, d / np.linalg.norm(d), rtol=1e-6)
        assert v.source["method"] == "sae_feature" and v.source["feature_id"] == 2
        assert v.source["sae_hook_layer"] == 5 and v.source["sae_dict_size"] == 10

    def test_saelens_hook_name_and_metadata_dict(self):
        sae = _fake_sae(_vec(8), hook_name="blocks.7.hook_resid_post")
        v = SteeringVector.from_sae_feature(sae, 0)
        assert (v.layer, v.site, v.name) == (7, "resid_post", "sae_feature_0")
        sae2 = _fake_sae(_vec(8))
        sae2.metadata = {"layer": 3, "model_name": "gpt2"}
        v2 = SteeringVector.from_sae_feature(sae2, 1)
        assert (v2.layer, v2.site, v2.model_name) == (3, "resid_post", "gpt2")

    def test_hook_point_tuple_does_not_shadow_hook_name(self):
        sae = _fake_sae(_vec(8), hook_name="blocks.4.hook_resid_pre")
        sae.hook_point = None  # SparseAutoencoder exposes (layer, site) | None here
        v = SteeringVector.from_sae_feature(sae, 0)
        assert (v.layer, v.site) == (4, "resid_pre")
        sae.hook_point = (6, "resid_post")
        sae.config.hook_name = None
        v = SteeringVector.from_sae_feature(sae, 0)
        assert (v.layer, v.site) == (6, "resid_post")

    def test_non_residual_hook_name_needs_an_explicit_site(self):
        sae = _fake_sae(_vec(8), hook_name="blocks.3.attn.hook_z", hook_layer=3)
        with pytest.raises(ValueError, match="hook_z"):
            SteeringVector.from_sae_feature(sae, 0)
        v = SteeringVector.from_sae_feature(sae, 0, site="resid_post")
        assert (v.layer, v.site) == (3, "resid_post")
        assert v.source["sae_hook_name"] == "blocks.3.attn.hook_z"
        no_layer = _fake_sae(_vec(8), hook_name="blocks.3.attn.hook_z")
        with pytest.raises(ValueError, match="not a residual hook"):
            SteeringVector.from_sae_feature(no_layer, 0)
        mid = _fake_sae(_vec(8), hook_name="blocks.2.hook_resid_mid")
        with pytest.raises(ValueError, match="resid_mid"):
            SteeringVector.from_sae_feature(mid, 0)

    def test_explicit_arguments_override_metadata(self):
        sae = _fake_sae(_vec(8), hook_layer=5, hook_site="mlp_out")
        v = SteeringVector.from_sae_feature(sae, 1, layer=2, site="resid_post")
        assert (v.layer, v.site) == (2, "resid_post")
        assert v.source["sae_hook_layer"] == 5 and v.source["sae_hook_site"] == "mlp_out"

    def test_errors(self):
        with pytest.raises(ValueError, match="pass layer="):
            SteeringVector.from_sae_feature(_fake_sae(_vec(8)), 0)
        with pytest.raises(ValueError, match="mlp_out"):
            SteeringVector.from_sae_feature(
                _fake_sae(_vec(8), hook_layer=1, hook_site="mlp_out"), 0
            )
        with pytest.raises(IndexError, match="out of range"):
            SteeringVector.from_sae_feature(_fake_sae(_vec(8)), 10, layer=0)
        dead = SimpleNamespace(
            config=SimpleNamespace(dict_size=4, input_dim=3),
            feature_direction=lambda fid: np.zeros(3),
        )
        with pytest.raises(ValueError, match="dead feature"):
            SteeringVector.from_sae_feature(dead, 1, layer=0)


class TestCapability:
    def test_mock_provider_is_refused(self):
        with pytest.raises(SteeringUnavailableError, match="synthetic"):
            as_hooked(MockProvider(n_layers=2, n_heads=1, d_model=8))

    def test_black_box_provider_is_refused(self):
        with pytest.raises(SteeringUnavailableError, match="black-box"):
            as_hooked(_BlackBox())

    def test_arbitrary_object_is_refused(self):
        with pytest.raises(SteeringUnavailableError, match="white-box"):
            as_hooked(object())

    def test_provider_wrappers_are_unwrapped(self):
        from LLmThoughtLens.scope import Scope

        with pytest.raises(SteeringUnavailableError, match="synthetic"):
            as_hooked(Scope(MockProvider(n_layers=2, n_heads=1, d_model=8)))
        with pytest.raises(SteeringUnavailableError, match="black-box"):
            as_hooked(SimpleNamespace(provider=_BlackBox()))

    def test_entry_points_check_capability_before_touching_torch(self):
        mock = MockProvider(n_layers=2, n_heads=1, d_model=8)
        v = SteeringVector(direction=_vec(8), layer=0)
        with pytest.raises(SteeringUnavailableError):
            steer_generate(mock, "hi", v)
        with pytest.raises(SteeringUnavailableError):
            coefficient_sweep(_BlackBox(), "hi", v, [0.0, 1.0])
        with pytest.raises(SteeringUnavailableError):
            SteeringVector.from_contrast(mock, ["a b"], ["c d"], layer=0)
        with pytest.raises(SteeringUnavailableError):
            steering_hooks(mock, v, 3)
        with pytest.raises(SteeringUnavailableError):
            v.validate(_BlackBox())


# ===========================================================================
# Tiny random models
# ===========================================================================

EXACT_FAMILIES = ("gpt2", "llama", "gemma2", "opt")
GEN_FAMILIES = ("gpt2", "llama")


def _tiny(family: str, seed: int = 0) -> Any:
    _torch()
    from LLmThoughtLens.models import HookedModel

    from _tiny_hf import TINY_FAMILIES, WordTokenizer, make_tiny_family_model

    model = make_tiny_family_model(family, seed=seed)
    if model is None:
        pytest.skip(f"installed transformers has no {TINY_FAMILIES[family][0]}")
    return HookedModel(model, WordTokenizer())


@pytest.fixture(params=EXACT_FAMILIES)
def exact_hm(request: pytest.FixtureRequest) -> Any:
    return _tiny(str(request.param))


@pytest.fixture(params=GEN_FAMILIES)
def gen_hm(request: pytest.FixtureRequest) -> Any:
    return _tiny(str(request.param))


def _strong(hm: Any, layer: int = 0, seed: int = 3, **kw: Any) -> SteeringVector:
    """A random vector large enough to change greedy tokens on the tiny models."""
    return SteeringVector.from_vector(_vec(hm.d_model, seed=seed, scale=8.0), layer, **kw)


def _hook_ids(model: Any) -> list[tuple[str, frozenset[int], frozenset[int]]]:
    return [
        (n, frozenset(m._forward_hooks), frozenset(m._forward_pre_hooks))
        for n, m in model.named_modules()
    ]


class TestExactEffect:
    """The edit is exactly ``hidden + coeff * d`` at the requested site."""

    def test_last_layer_resid_post_matches_logit_lens_of_shifted_residual(self, exact_hm: Any):
        torch = _torch()
        hm = exact_hm
        v = SteeringVector.from_vector(_vec(hm.d_model, seed=1), -1, coeff=1.5)
        base = hm.forward(PROMPT)
        steered = hm.forward(PROMPT, hooks=steering_hooks(hm, v, T))
        shift = torch.as_tensor(v.vector())
        expected = hm.logit_lens(base.resid_post[-1] + shift)
        torch.testing.assert_close(steered.logits[0], expected, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(steered.resid_post[-1], base.resid_post[-1] + shift)
        torch.testing.assert_close(steered.resid_post[:-1], base.resid_post[:-1])

    def test_mid_layer_resid_post_matches_a_raw_torch_hook(self, exact_hm: Any):
        torch = _torch()
        hm = exact_hm
        v = SteeringVector.from_vector(_vec(hm.d_model, seed=2), 0, coeff=-2.0)
        shift = torch.as_tensor(v.vector())

        def raw(_mod: Any, _args: Any, out: Any) -> Any:
            if isinstance(out, tuple):
                return (out[0] + shift, *out[1:])
            return out + shift

        handle = hm.blocks[0].register_forward_hook(raw)
        try:
            with torch.no_grad():
                manual = hm.model(input_ids=hm.tokenize(PROMPT).input_ids, use_cache=False).logits
        finally:
            handle.remove()
        steered = hm.forward(PROMPT, hooks=steering_hooks(hm, v, T), capture=False)
        torch.testing.assert_close(steered.logits, manual, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("family", ["gpt2", "llama"])
    def test_resid_pre_layer0_equals_shifted_inputs_embeds(self, family: str):
        torch = _torch()
        hm = _tiny(family)
        v = SteeringVector.from_vector(_vec(hm.d_model, seed=4), 0, site="resid_pre", coeff=3.0)
        ids = hm.tokenize(PROMPT).input_ids
        with torch.no_grad():
            emb = hm.model.get_input_embeddings()(ids) + torch.as_tensor(v.vector())
            manual = hm.model(inputs_embeds=emb, use_cache=False).logits
        steered = hm.forward(PROMPT, hooks=steering_hooks(hm, v, T), capture=False)
        torch.testing.assert_close(steered.logits, manual, rtol=1e-5, atol=1e-5)

    def test_resid_post_l_equals_resid_pre_l_plus_1(self, exact_hm: Any):
        torch = _torch()
        hm = exact_hm
        d = _vec(hm.d_model, seed=5, scale=3.0)
        post0 = SteeringVector.from_vector(d, 0, site="resid_post")
        pre1 = SteeringVector.from_vector(d, 1, site="resid_pre")
        a = hm.forward(PROMPT, hooks=steering_hooks(hm, post0, T), capture=False).logits
        b = hm.forward(PROMPT, hooks=steering_hooks(hm, pre1, T), capture=False).logits
        torch.testing.assert_close(a, b)

    def test_positions_restrict_the_edit(self, exact_hm: Any):
        torch = _torch()
        hm = exact_hm
        v = SteeringVector.from_vector(_vec(hm.d_model, seed=6, scale=4.0), 0, positions=[2, -1])
        base = hm.forward(PROMPT)
        steered = hm.forward(PROMPT, hooks=steering_hooks(hm, v, T))
        diff = (steered.resid_post[0] - base.resid_post[0]).abs().sum(-1)
        assert [i for i in range(T) if float(diff[i]) > 0] == [2, T - 1]
        torch.testing.assert_close(
            steered.resid_post[0, 2], base.resid_post[0, 2] + torch.as_tensor(v.vector())
        )
        # Causal: positions before the first edit are untouched everywhere.
        torch.testing.assert_close(steered.logits[0, :2], base.logits[0, :2])
        assert not torch.allclose(steered.logits[0, 2], base.logits[0, 2])

    def test_several_vectors_add_up(self, exact_hm: Any):
        torch = _torch()
        hm = exact_hm
        d1, d2 = _vec(hm.d_model, seed=7), _vec(hm.d_model, seed=8)
        v1 = SteeringVector.from_vector(d1, 0)
        v2 = SteeringVector.from_vector(d2, 0)
        both = steering_hooks(hm, [v1, v2], T, coeffs=[2.0, -3.0])
        combined = steering_hooks(hm, SteeringVector.from_vector(2 * d1 - 3 * d2, 0), T)
        a = hm.forward(PROMPT, hooks=both, capture=False).logits
        b = hm.forward(PROMPT, hooks=combined, capture=False).logits
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)


class TestValidation:
    def test_mismatches_raise(self):
        _torch()
        hm = _tiny("gpt2")
        with pytest.raises(SteeringMismatchError, match="d_model=7"):
            SteeringVector(direction=np.ones(7), layer=0).validate(hm)
        with pytest.raises(SteeringMismatchError, match="12-layer"):
            SteeringVector(direction=np.ones(hm.d_model), layer=0, n_layers=12).validate(hm)
        with pytest.raises(SteeringMismatchError, match="out of range"):
            SteeringVector(direction=np.ones(hm.d_model), layer=2).validate(hm)
        assert SteeringVector(direction=np.ones(hm.d_model), layer=-1).validate(hm) == 1
        bad = SteeringVector(direction=np.ones(7), layer=0)
        with pytest.raises(SteeringMismatchError):
            steer_generate(hm, PROMPT, bad, max_new_tokens=2)
        with pytest.raises(SteeringMismatchError):
            coefficient_sweep(hm, PROMPT, bad, [1.0])

    def test_model_name_mismatch_only_warns(self):
        _torch()
        hm = _tiny("gpt2")
        hm.model.config._name_or_path = "/models/gpt2"
        v = SteeringVector(direction=np.ones(hm.d_model), layer=0, model_name="gpt2")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            v.validate(hm)  # same basename: no warning
        v.model_name = "distilgpt2"
        with pytest.warns(UserWarning, match="distilgpt2"):
            v.validate(hm)

    def test_load_validates_against_a_model(self, tmp_path):
        _torch()
        hm = _tiny("gpt2")
        good = SteeringVector(direction=np.ones(hm.d_model), layer=1, n_layers=2)
        assert SteeringVector.load(good.save(tmp_path / "good.npz"), hooked=hm).layer == 1
        wrong = SteeringVector(direction=np.ones(hm.d_model + 1), layer=1)
        with pytest.raises(SteeringMismatchError):
            SteeringVector.load(wrong.save(tmp_path / "wrong.npz"), hooked=hm)

    def test_vector_and_coeff_arguments(self):
        _torch()
        hm = _tiny("gpt2")
        v = SteeringVector(direction=np.ones(hm.d_model), layer=0)
        with pytest.raises(ValueError, match="at least one"):
            steering_hooks(hm, [], T)
        with pytest.raises(TypeError, match="SteeringVector"):
            steering_hooks(hm, [np.ones(hm.d_model)], T)
        with pytest.raises(ValueError, match="2 coeffs for 1"):
            steering_hooks(hm, v, T, coeffs=[1.0, 2.0])
        with pytest.raises(ValueError, match="finite"):
            steering_hooks(hm, v, T, coeffs=math.nan)
        with pytest.raises(ValueError, match="max_new_tokens"):
            steer_generate(hm, PROMPT, v, max_new_tokens=0)
        with pytest.raises(ValueError, match="empty"):
            coefficient_sweep(hm, PROMPT, v, [])
        with pytest.raises(ValueError, match="finite"):
            coefficient_sweep(hm, PROMPT, v, [math.inf])
        hook = v.to_hook(hm, T, coeff=0.5)
        assert (hook.layer, hook.site, hook.positions) == (0, "resid_post", None)


class TestFromContrast:
    POS = ["the quick brown fox", "a lazy dog sleeps here"]
    NEG = ["one two three"]

    def _hidden(self, hm: Any, prompt: str, index: int) -> np.ndarray:
        torch = _torch()
        with torch.no_grad():
            out = hm.model(
                input_ids=hm.tokenize(prompt).input_ids,
                output_hidden_states=True,
                use_cache=False,
            )
        return out.hidden_states[index][0].double().numpy()

    @pytest.mark.parametrize("family", ["gpt2", "llama"])
    def test_last_position_mean_difference_by_hand(self, family: str):
        hm = _tiny(family)
        v = SteeringVector.from_contrast(hm, self.POS, self.NEG, layer=0)
        # HF hidden_states[1] is the output of block 0 (not final-normed for l < L-1).
        pos = np.mean([self._hidden(hm, p, 1)[-1] for p in self.POS], axis=0)
        neg = np.mean([self._hidden(hm, p, 1)[-1] for p in self.NEG], axis=0)
        np.testing.assert_allclose(v.direction, pos - neg, rtol=1e-5, atol=1e-5)
        assert (v.layer, v.site, v.n_layers, v.d_model) == (0, "resid_post", 2, hm.d_model)
        src = v.source
        assert src["method"] == "contrast_mean_difference" and src["position"] == "last"
        assert (src["n_positive"], src["n_negative"]) == (2, 1)
        assert src["positive_prompts"] == self.POS and src["negative_prompts"] == self.NEG
        assert src["direction_norm"] == pytest.approx(float(np.linalg.norm(pos - neg)), rel=1e-5)
        norms = [np.linalg.norm(self._hidden(hm, p, 1)[-1]) for p in self.POS + self.NEG]
        assert src["resid_norm_mean"] == pytest.approx(float(np.mean(norms)), rel=1e-5)
        assert src["family"] == family

    def test_mean_position_and_resid_pre(self):
        hm = _tiny("gpt2")
        v = SteeringVector.from_contrast(
            hm, self.POS, self.NEG, layer=-1, site="resid_pre", position="mean"
        )
        # resid_pre[1] == output of block 0 == hidden_states[1]
        pos = np.mean([self._hidden(hm, p, 1).mean(axis=0) for p in self.POS], axis=0)
        neg = np.mean([self._hidden(hm, p, 1).mean(axis=0) for p in self.NEG], axis=0)
        np.testing.assert_allclose(v.direction, pos - neg, rtol=1e-5, atol=1e-5)
        assert (v.layer, v.site) == (1, "resid_pre")

    def test_single_pair_strings_and_identical_sets(self):
        hm = _tiny("gpt2")
        same = SteeringVector.from_contrast(hm, "a b c", "a b c", layer=0)
        assert same.norm == 0.0 and same.source["n_positive"] == 1
        with pytest.warns(UserWarning, match="single token"):
            SteeringVector.from_contrast(hm, ["fox"], ["dog"], layer=0)

    def test_errors(self):
        hm = _tiny("gpt2")
        with pytest.raises(ValueError, match="at least one"):
            SteeringVector.from_contrast(hm, [], ["a"], layer=0)
        with pytest.raises(SteeringMismatchError, match="out of range"):
            SteeringVector.from_contrast(hm, ["a b"], ["c d"], layer=5)
        with pytest.raises(ValueError, match="site"):
            SteeringVector.from_contrast(hm, ["a b"], ["c d"], layer=0, site="mlp_in")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="position"):
            SteeringVector.from_contrast(hm, ["a b"], ["c d"], layer=0, position="first")  # type: ignore[arg-type]

    def test_provider_and_chat_template(self):
        _torch()
        from _tiny_hf import make_provider

        provider = make_provider(chat_template="tmpl")
        v = SteeringVector.from_contrast(provider, ["a b c"], ["d e f"], layer=0, chat=True)
        hm = provider.hooked
        tp = hm.tokenize("a b c", chat=True)
        assert tp.chat_applied and len(tp.token_ids) > 3
        ref_pos = hm.forward(tp).resid_post[0, -1]
        ref_neg = hm.forward(hm.tokenize("d e f", chat=True)).resid_post[0, -1]
        np.testing.assert_allclose(v.direction, (ref_pos - ref_neg).numpy(), rtol=1e-5, atol=1e-5)
        assert v.source["chat"] is True


class TestCachedGenerationPositions:
    """Every mode gives cache-independent tokens; the edited positions are exactly as documented."""

    N = 4

    def _spy(self, hm: Any, v: SteeringVector) -> tuple[Any, list[tuple[int, int]]]:
        hook = v.to_hook(hm, T, total_len=T + self.N)
        calls: list[tuple[int, int]] = []
        inner = hook.fn

        def fn(h: Any) -> Any:
            calls.append((hm.current_offset, int(h.shape[1])))
            return inner(h)

        hook.fn = fn
        return hook, calls

    @pytest.mark.parametrize(
        ("positions", "expected"),
        [
            ("all", [(0, T), (T, 1), (T + 1, 1), (T + 2, 1)]),
            ("prompt", [(0, T)]),
            ("last", [(0, 1)]),
            ("generated", [(0, 1), (T, 1), (T + 1, 1), (T + 2, 1)]),
            ((1, T + 1), [(0, 1), (T + 1, 1)]),
            (-1, [(0, 1)]),
        ],
    )
    def test_edited_positions_and_cache_independence(
        self, gen_hm: Any, positions: Any, expected: list[tuple[int, int]]
    ):
        hm = gen_hm
        v = _strong(hm, positions=positions)
        hook, calls = self._spy(hm, v)
        cached = hm.generate(PROMPT, max_new_tokens=self.N, hooks=[hook], stop_at_eos=False)
        assert calls == expected
        uncached = hm.generate(
            PROMPT, max_new_tokens=self.N, hooks=[hook], stop_at_eos=False, use_cache=False
        )
        assert cached.token_ids == uncached.token_ids
        assert cached.logprobs == pytest.approx(uncached.logprobs, abs=1e-4)

    def test_generated_equals_newest_token_hook_under_cache(self, gen_hm: Any):
        torch = _torch()
        from LLmThoughtLens.models import ResidHook

        hm = gen_hm
        v = _strong(hm, positions="generated")
        shift = torch.as_tensor(v.vector())
        newest = ResidHook(0, lambda h: h + shift, positions=-1)
        a = steer_generate(hm, PROMPT, v, max_new_tokens=6, stop_at_eos=False).steered
        b = hm.generate(PROMPT, max_new_tokens=6, hooks=[newest], stop_at_eos=False)
        assert a.token_ids == b.token_ids
        assert a.logprobs == pytest.approx(b.logprobs, abs=1e-6)

    def test_modes_share_the_first_step_where_documented(self, gen_hm: Any):
        hm = gen_hm
        runs = {
            mode: steer_generate(
                hm, PROMPT, _strong(hm, positions=mode), max_new_tokens=6, stop_at_eos=False
            )
            for mode in ("all", "prompt", "last", "generated")
        }
        first = {m: r.steered_logprobs_on_baseline[0] for m, r in runs.items()}
        # "all"/"prompt" edit the same positions at prefill; so do "last"/"generated".
        assert first["all"] == pytest.approx(first["prompt"], abs=1e-6)
        assert first["last"] == pytest.approx(first["generated"], abs=1e-6)
        assert runs["all"].kl_per_step[0] == pytest.approx(runs["prompt"].kl_per_step[0], abs=1e-9)
        assert runs["last"].kl_per_step[0] == pytest.approx(
            runs["generated"].kl_per_step[0], abs=1e-9
        )
        # ... but later steps differ: generated tokens are edited only by "all"/"generated".
        assert runs["all"].kl_per_step[1:] != pytest.approx(runs["prompt"].kl_per_step[1:])
        assert runs["generated"].kl_per_step[1:] != pytest.approx(runs["last"].kl_per_step[1:])


class TestSteerGenerate:
    def test_coeff_zero_is_exactly_the_baseline(self, gen_hm: Any):
        hm = gen_hm
        r = steer_generate(hm, PROMPT, _strong(hm), coeffs=0.0, max_new_tokens=6)
        assert r.steered.token_ids == r.baseline.token_ids
        assert r.steered.logprobs == r.baseline.logprobs  # bit-identical
        assert r.steered.text == r.baseline.text
        assert r.kl_per_step == [0.0] * len(r.baseline.token_ids)
        assert r.promoted == [] and r.suppressed == []
        assert r.diverged_at is None and r.first_step_kl == 0.0 and r.mean_kl == 0.0
        assert r.vectors[0]["coeff"] == 0.0 and r.vectors[0]["added_norm"] == 0.0

    def test_teacher_forced_numbers_by_hand(self, gen_hm: Any):
        torch = _torch()
        hm = gen_hm
        v = _strong(hm, coeff=0.5)
        r = steer_generate(hm, PROMPT, v, max_new_tokens=6, stop_at_eos=False, top_k_tokens=3)
        n = len(r.baseline.token_ids)
        assert n == 6 and len(r.kl_per_step) == n == len(r.steered_logprobs_on_baseline)
        seq = r.prompt_token_ids + r.baseline.token_ids[:-1]
        shift = torch.as_tensor(v.vector())

        def raw(_m: Any, _a: Any, out: Any) -> Any:
            return (out[0] + shift, *out[1:]) if isinstance(out, tuple) else out + shift

        ids = torch.tensor([seq])
        with torch.no_grad():
            lb = hm.model(input_ids=ids, use_cache=False).logits[0, T - 1 :].double()
            handle = hm.blocks[0].register_forward_hook(raw)
            try:
                ls = hm.model(input_ids=ids, use_cache=False).logits[0, T - 1 :].double()
            finally:
                handle.remove()
        pb = np.exp(torch.log_softmax(lb, -1).numpy())
        ps = np.exp(torch.log_softmax(ls, -1).numpy())
        kl = (ps * (np.log(ps) - np.log(pb))).sum(-1)
        assert r.kl_per_step == pytest.approx(kl.tolist(), abs=1e-6)
        assert all(k >= 0 for k in r.kl_per_step) and r.first_step_kl > 0
        on_base = np.log(ps[np.arange(n), r.baseline.token_ids])
        assert r.steered_logprobs_on_baseline == pytest.approx(on_base.tolist(), abs=1e-5)
        assert r.baseline.logprobs == pytest.approx(
            np.log(pb[np.arange(n), r.baseline.token_ids]).tolist(), abs=1e-4
        )
        # Up to the divergence point the steered run saw the baseline prefix.
        d = r.diverged_at if r.diverged_at is not None else n
        for i in range(min(d, n)):
            assert r.steered.logprobs[i] == pytest.approx(
                r.steered_logprobs_on_baseline[i], abs=1e-4
            )
        # First-step shifts are the top probability movers, sorted.
        delta = ps[0] - pb[0]
        assert [s.token_id for s in r.promoted] == [
            int(i) for i in np.argsort(-delta)[:3] if delta[i] > 0
        ]
        assert [s.token_id for s in r.suppressed] == [
            int(i) for i in np.argsort(delta)[:3] if delta[i] < 0
        ]
        top = r.promoted[0]
        assert top.delta == pytest.approx(top.p_steered - top.p_baseline)
        assert top.logprob_delta == pytest.approx(
            math.log(top.p_steered / top.p_baseline), abs=1e-6
        )
        assert top.token == hm.token_str(top.token_id)

    def test_strong_vector_changes_the_completion(self, gen_hm: Any):
        hm = gen_hm
        r = steer_generate(hm, PROMPT, _strong(hm, coeff=3.0), max_new_tokens=6, stop_at_eos=False)
        assert r.steered.token_ids != r.baseline.token_ids
        assert r.diverged_at is not None and r.mean_kl > 0

    def test_deterministic_and_seeded_sampling(self, gen_hm: Any):
        hm = gen_hm
        v = _strong(hm)
        a = steer_generate(hm, PROMPT, v, max_new_tokens=5)
        b = steer_generate(hm, PROMPT, v, max_new_tokens=5)
        assert a.to_dict() == b.to_dict()
        s1 = steer_generate(hm, PROMPT, v, max_new_tokens=5, temperature=1.0, seed=11)
        s2 = steer_generate(hm, PROMPT, v, max_new_tokens=5, temperature=1.0, seed=11)
        assert s1.steered.token_ids == s2.steered.token_ids
        assert s1.baseline.token_ids == s2.baseline.token_ids
        assert (s1.temperature, s1.seed) == (1.0, 11)

    def test_hooks_are_removed_and_result_is_json_safe(self):
        hm = _tiny("gpt2")
        before = _hook_ids(hm.model)
        r = steer_generate(hm, PROMPT, [_strong(hm), _strong(hm, layer=1, seed=9)], coeffs=[1, -1])
        assert _hook_ids(hm.model) == before
        d = json.loads(json.dumps(r.to_dict()))
        assert d["evidence_kind"] == "white_box" and d["method"] == "activation_steering"
        assert "Interventional" in d["note"] and "explicit" in d["note"]
        assert d["effect_semantics"] == "causal_intervention"
        assert [v["coeff"] for v in d["vectors"]] == [1.0, -1.0]
        assert d["prompt"] == PROMPT and d["prompt_tokens"] == r.prompt_tokens

    def test_through_the_huggingface_provider_with_chat(self):
        _torch()
        from LLmThoughtLens.scope import Scope

        from _tiny_hf import make_provider

        provider = make_provider(chat_template="tmpl")
        assert as_hooked(provider) is provider.hooked
        assert as_hooked(Scope(provider)) is provider.hooked
        v = _strong(provider.hooked)
        r = steer_generate(provider, PROMPT, v, max_new_tokens=3, chat=True, stop_at_eos=False)
        tp = provider.hooked.tokenize(PROMPT, chat=True)
        assert r.prompt_token_ids == tp.token_ids and r.prompt.startswith("<user>")
        tp_r = steer_generate(provider, tp, v, max_new_tokens=3, stop_at_eos=False)
        assert tp_r.steered.token_ids == r.steered.token_ids


class TestProcessedRange:
    """Generation feeds ``T + n - 1`` tokens; later positions can never be steered."""

    def test_last_fed_position_steers_only_the_last_step(self, gen_hm: Any):
        hm, n = gen_hm, 3
        v = _strong(hm, positions=T + n - 2)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            r = steer_generate(hm, PROMPT, v, max_new_tokens=n, stop_at_eos=False)
        assert r.kl_per_step[: n - 1] == pytest.approx([0.0] * (n - 1), abs=1e-9)
        assert r.kl_per_step[n - 1] > 0
        assert r.steered.token_ids[: n - 1] == r.baseline.token_ids[: n - 1]

    def test_unreachable_positions_warn_and_do_nothing(self, gen_hm: Any):
        hm, n = gen_hm, 3
        v = _strong(hm, positions=[T + n - 1, T + 10])
        with pytest.warns(UserWarning, match="no effect"):
            r = steer_generate(hm, PROMPT, v, max_new_tokens=n, stop_at_eos=False)
        assert r.steered.token_ids == r.baseline.token_ids
        assert r.kl_per_step == [0.0] * n
        with pytest.warns(UserWarning, match="no effect"):
            coefficient_sweep(hm, PROMPT, _strong(hm, positions=T), [1.0], max_new_tokens=1)


class TestCoefficientSweep:
    def test_constructed_case_is_monotone(self):
        """Identity final norm + ``d = W_U[t]`` with ``t`` the largest-norm unembedding row.

        Then the steered last-position logits are ``z + c * W_U @ W_U[t]`` and,
        by Cauchy-Schwarz, ``W_U[t] . W_U[t] >= W_U[j] . W_U[t]`` for every j, so
        ``p(t)`` strictly increases with ``c`` and ``KL(p_c || p_0)`` (an
        exponential family, derivative ``c * Var_c``) grows with ``|c|``.
        """
        torch = _torch()
        from LLmThoughtLens.models import HookedModel

        from _tiny_hf import WordTokenizer, make_tiny_gpt2

        model = make_tiny_gpt2(seed=0)
        model.transformer.ln_f = torch.nn.Identity()
        hm = HookedModel(model, WordTokenizer())
        w = model.lm_head.weight.detach()
        t = int(w.norm(dim=1).argmax())
        v = SteeringVector.from_vector(w[t], -1, positions="last")
        coeffs = [-4.0, -2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 4.0]
        sw = coefficient_sweep(hm, PROMPT, v, coeffs, target_tokens=[t], max_new_tokens=0)
        label = hm.token_str(t)
        assert sw.targets == {label: t}
        p = [r.target_probs[label] for r in sw.rows]
        kl = [r.kl for r in sw.rows]
        assert all(b > a for a, b in zip(p, p[1:], strict=False))
        assert kl[3] == 0.0
        assert all(a > b for a, b in zip(kl[:4], kl[1:4], strict=False))  # shrinking to c=0
        assert all(b > a for a, b in zip(kl[3:], kl[4:], strict=False))  # growing after
        # Closed form for the target probability.
        z = hm.forward(PROMPT, capture=False).logits[0, -1].double()
        a = w.double() @ w[t].double()
        for c, row in zip(coeffs, sw.rows, strict=True):
            ref = float(torch.softmax(z + c * a, -1)[t])
            assert row.target_probs[label] == pytest.approx(ref, rel=1e-4, abs=1e-7)
        assert sw.baseline_target_probs[label] == pytest.approx(sw.rows[3].target_probs[label])
        assert sw.rows[0].text is None and sw.baseline_text is None

    def test_table_text_and_serialisation(self):
        hm = _tiny("gpt2")  # importorskip torch before touching _tiny_hf
        from _tiny_hf import WordTokenizer

        words = PROMPT.split()
        v = _strong(hm, positions="generated")
        sw = coefficient_sweep(
            hm, PROMPT, v, [0.0, 2.0], target_tokens=[words[0], 5], max_new_tokens=3
        )
        label_int = hm.token_str(5)
        assert sw.targets == {words[0]: WordTokenizer.word_id(words[0]), label_int: 5}
        assert sw.rows[0].kl == 0.0
        assert sw.rows[0].target_probs == sw.baseline_target_probs
        assert sw.rows[0].text == sw.baseline_text
        assert sw.rows[0].top_tokens == sw.baseline_top_tokens and len(sw.baseline_top_tokens) == 5
        assert sw.rows[1].target_prob_total == pytest.approx(sum(sw.rows[1].target_probs.values()))
        table = sw.table()
        assert [r["coeff"] for r in table] == [0.0, 2.0]
        assert set(table[0]) == {"coeff", "kl", f"p[{words[0]}]", f"p[{label_int}]", "top", "text"}
        text = sw.format_table()
        assert text.splitlines()[0].startswith("coeff | KL") and len(text.splitlines()) == 3
        d = json.loads(json.dumps(sw.to_dict()))
        assert "coeff" not in d["vector"] and d["vector"]["positions"] == "generated"
        assert len(d["rows"]) == 2 and d["rows"][1]["coeff"] == 2.0
        assert (d["evidence_kind"], d["method"], d["effect_semantics"]) == (
            "white_box",
            "activation_steering",
            "causal_intervention",
        )
        assert "Interventional" in d["note"]

    def test_target_resolution_errors(self):
        hm = _tiny("gpt2")
        v = _strong(hm)
        with pytest.raises(ValueError, match="2 tokens"):
            coefficient_sweep(hm, PROMPT, v, [1.0], target_tokens="two words")
        with pytest.raises(IndexError, match="vocab"):
            coefficient_sweep(hm, PROMPT, v, [1.0], target_tokens=10_000)
        with pytest.raises(TypeError, match="str or int"):
            coefficient_sweep(hm, PROMPT, v, [1.0], target_tokens=[1.5])  # type: ignore[list-item]
        one = coefficient_sweep(hm, PROMPT, v, [1.0], target_tokens="fox", max_new_tokens=0)
        assert list(one.targets) == ["fox"]


class TestRealSaeFeature:
    def test_trained_free_sae_direction_steers(self):
        torch = _torch()
        from LLmThoughtLens.features.sae import SAEConfig, SparseAutoencoder

        hm = _tiny("gpt2")
        sae = SparseAutoencoder(SAEConfig(input_dim=hm.d_model, dict_size=16, k=4, device="cpu"))
        v = SteeringVector.from_sae_feature(sae, 3, layer=0, coeff=5.0)
        np.testing.assert_allclose(v.direction, sae.feature_direction(3), rtol=1e-6)
        assert v.normalize and float(np.linalg.norm(v.vector())) == pytest.approx(5.0, rel=1e-5)
        base = hm.forward(PROMPT)
        steered = hm.forward(PROMPT, hooks=steering_hooks(hm, v, T))
        torch.testing.assert_close(
            steered.resid_post[0], base.resid_post[0] + torch.as_tensor(v.vector())
        )

    def test_hook_metadata_on_the_sae_config(self):
        torch = _torch()
        from LLmThoughtLens.features.sae import SAEConfig, SparseAutoencoder

        hm = _tiny("gpt2")
        try:
            cfg = SAEConfig(
                input_dim=hm.d_model,
                dict_size=16,
                k=4,
                device="cpu",
                hook_name="blocks.1.hook_resid_pre",
                model_name="tiny",
            )
        except TypeError:
            pytest.skip("this SAEConfig carries no hook metadata")
        sae = SparseAutoencoder(cfg)
        v = SteeringVector.from_sae_feature(sae, 0, coeff=2.0)
        assert (v.layer, v.site, v.model_name, v.d_model) == (1, "resid_pre", "tiny", hm.d_model)
        assert v.source["sae_hook_name"] == "blocks.1.hook_resid_pre"
        assert v.source.get("sae_architecture", "topk") == "topk"
        assert "sae_release" not in v.source  # None values are not recorded
        base = hm.forward(PROMPT)
        steered = hm.forward(PROMPT, hooks=steering_hooks(hm, v, T))
        torch.testing.assert_close(
            steered.resid_pre[1], base.resid_pre[1] + torch.as_tensor(v.vector())
        )
        torch.testing.assert_close(steered.resid_post[0], base.resid_post[0])


# ===========================================================================
# Real GPT-2 (only when already cached locally)
# ===========================================================================

PARIS = [
    "I traveled to Paris, France",
    "The Eiffel Tower in Paris",
    "We ate croissants in Paris",
    "Paris is the capital of France",
]
LONDON = [
    "I traveled to London, England",
    "Big Ben in London",
    "We ate fish and chips in London",
    "London is the capital of England",
]
NEUTRAL = "My favourite city in the world is"


@pytest.fixture(scope="module")
def real_gpt2() -> Any:
    _torch()
    from LLmThoughtLens.models import HookedModel

    try:
        return HookedModel.from_pretrained("gpt2", device="cpu", local_files_only=True)
    except Exception as exc:  # OSError when not cached
        pytest.skip(f"gpt2 weights not available offline: {exc}")


@pytest.fixture(scope="module")
def paris_vector(real_gpt2: Any) -> SteeringVector:
    return SteeringVector.from_contrast(real_gpt2, PARIS, LONDON, layer=6, name="paris-london")


class TestRealGPT2:
    def test_contrast_vector_metadata(self, real_gpt2: Any, paris_vector: SteeringVector):
        v = paris_vector
        assert (v.layer, v.d_model, v.n_layers, v.model_name) == (6, 768, 12, "gpt2")
        assert 10 < v.norm < v.source["resid_norm_mean"]

    def test_steering_shifts_next_token_and_completion(
        self, real_gpt2: Any, paris_vector: SteeringVector
    ):
        r = steer_generate(real_gpt2, NEUTRAL, paris_vector, coeffs=1.0, max_new_tokens=8)
        assert r.baseline.text.startswith(" London")
        assert r.steered.text.startswith(" Paris")
        assert r.diverged_at == 0 and r.first_step_kl > 0.3
        promoted = {s.token: s for s in r.promoted[:5]}
        assert " Paris" in promoted
        assert promoted[" Paris"].p_steered > 3 * promoted[" Paris"].p_baseline
        assert " London" in {s.token for s in r.suppressed}

    def test_coefficient_sweep(self, real_gpt2: Any, paris_vector: SteeringVector):
        coeffs = [-1.0, -0.5, 0.0, 0.25, 0.5, 1.0]
        sw = coefficient_sweep(
            real_gpt2, NEUTRAL, paris_vector, coeffs, [" Paris", " London"], max_new_tokens=4
        )
        by_c = {r.coeff: r for r in sw.rows}
        paris = [by_c[c].target_probs[" Paris"] for c in coeffs[:5]]
        london = [by_c[c].target_probs[" London"] for c in coeffs[1:]]
        assert all(b > a for a, b in zip(paris, paris[1:], strict=False))
        assert all(b < a for a, b in zip(london, london[1:], strict=False))
        assert by_c[0.0].kl == 0.0
        assert by_c[0.25].kl < by_c[0.5].kl < by_c[1.0].kl
        assert by_c[-0.5].kl < by_c[-1.0].kl
        assert by_c[-1.0].target_probs[" London"] > 2 * sw.baseline_target_probs[" London"]
        assert by_c[1.0].text.startswith(" Paris") and by_c[-1.0].text.startswith(" London")
