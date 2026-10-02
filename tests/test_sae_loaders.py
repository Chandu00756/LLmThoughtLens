"""Tests for pretrained-SAE loading: SAELens folders, Gemma Scope ``params.npz``, the registry.

Everything here is offline and deterministic: SAE files are synthesised in
``tmp_path`` (random weights, the real formats' names / shapes / orientation)
and every Hugging Face Hub call is monkeypatched.  The one real-weights test
(:class:`TestRealGPT2SmallSAE`) runs only when both gpt2 and the
``gpt2-small-res-jb`` SAE are already in the local Hugging Face cache.
"""

from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
safetensors_numpy = pytest.importorskip("safetensors.numpy")

from LLmThoughtLens.features.sae import SparseAutoencoder  # noqa: E402
from LLmThoughtLens.features.sae_loaders import (  # noqa: E402
    PRETRAINED_RELEASES,
    describe_pretrained,
    from_pretrained,
    gemma_scope_hook,
    infer_center_input,
    list_pretrained,
    load_gemma_scope,
    load_saelens,
    parse_safetensors_header,
    read_safetensors_header,
    saelens_config_kwargs,
    tensor_shapes,
    validate_saelens_shapes,
)

D, F = 8, 24


# ---------------------------------------------------------------------------
# Synthetic files in the real formats
# ---------------------------------------------------------------------------


def _weights(seed: int = 0, d: int = D, f: int = F) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "W_enc": (rng.standard_normal((d, f)) / np.sqrt(d)).astype(np.float32),  # (d_in, d_sae)
        "W_dec": (rng.standard_normal((f, d)) / np.sqrt(f)).astype(np.float32),  # (d_sae, d_in)
        "b_enc": (0.1 * rng.standard_normal(f)).astype(np.float32),
        "b_dec": (0.2 * rng.standard_normal(d)).astype(np.float32),
        "threshold": rng.uniform(0.05, 0.4, f).astype(np.float32),
    }


def _write_saelens(folder: Path, cfg: dict[str, Any], tensors: dict[str, np.ndarray]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "cfg.json").write_text(json.dumps(cfg))
    safetensors_numpy.save_file(dict(tensors), str(folder / "sae_weights.safetensors"))
    return folder


def _legacy_cfg(**over: Any) -> dict[str, Any]:
    """jbloom/GPT2-Small-SAEs-Reformatted style (the real file's keys, tiny dims)."""
    cfg = {
        "model_name": "gpt2-small",
        "hook_point": "blocks.1.hook_resid_pre",
        "hook_point_layer": 1,
        "hook_point_head_index": None,
        "dataset_path": "Skylion007/openwebtext",
        "context_size": 128,
        "d_in": D,
        "d_sae": F,
        "dtype": "torch.float32",
        "expansion_factor": F // D,
        "l1_coefficient": 8e-5,
        "run_name": "tiny",
    }
    cfg.update(over)
    return cfg


def _v5_jumprelu_cfg(**over: Any) -> dict[str, Any]:
    cfg = {
        "architecture": "jumprelu",
        "d_in": D,
        "d_sae": F,
        "activation_fn_str": "relu",
        "apply_b_dec_to_input": False,
        "finetuning_scaling_factor": False,
        "context_size": 1024,
        "model_name": "gemma-2-2b",
        "hook_name": "blocks.0.hook_resid_post",
        "hook_layer": 0,
        "hook_head_index": None,
        "prepend_bos": True,
        "dataset_path": "monology/pile-uncopyrighted",
        "normalize_activations": "none",
        "model_from_pretrained_kwargs": {},
        "neuronpedia_id": "gemma-2-2b/0-gemmascope-res-16k",
    }
    cfg.update(over)
    return cfg


def _v6_topk_cfg(**over: Any) -> dict[str, Any]:
    cfg = {
        "architecture": "topk",
        "d_in": D,
        "d_sae": F,
        "k": 4,
        "apply_b_dec_to_input": True,
        "normalize_activations": "none",
        "metadata": {
            "model_name": "pythia-70m-deduped",
            "hook_name": "blocks.1.hook_mlp_out",
            "hook_head_index": None,
            "prepend_bos": True,
            "model_from_pretrained_kwargs": {"center_writing_weights": False},
            "sae_lens_version": "6.0.0",
        },
    }
    cfg.update(over)
    return cfg


def _ref(x: np.ndarray, w: dict[str, np.ndarray], arch: str, *, b_dec_in: bool, k: int = 4):
    xin = x.astype(np.float64) - (w["b_dec"] if b_dec_in else 0.0)
    pre = xin @ w["W_enc"] + w["b_enc"]
    if arch == "relu":
        return np.maximum(pre, 0)
    if arch == "jumprelu":
        return (pre > w["threshold"]) * np.maximum(pre, 0)
    out = np.zeros_like(pre)
    idx = np.argsort(-pre, axis=-1)[:, :k]
    np.put_along_axis(out, idx, np.maximum(np.take_along_axis(pre, idx, -1), 0), -1)
    return out


def _x(n: int = 7, seed: int = 3, d: int = D) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)


def _core(w: dict[str, np.ndarray], *extra: str) -> dict[str, np.ndarray]:
    return {k: w[k] for k in ("W_enc", "W_dec", "b_enc", "b_dec", *extra)}


# ---------------------------------------------------------------------------
# SAELens
# ---------------------------------------------------------------------------


class TestSAELensFormats:
    def test_legacy_jbloom_config(self, tmp_path):
        w = _weights()
        root = tmp_path / "repo"
        _write_saelens(root / "blocks.1.hook_resid_pre", _legacy_cfg(), _core(w))
        sae = load_saelens(root, "blocks.1.hook_resid_pre", release="rel")
        cfg = sae.config
        assert (cfg.architecture, cfg.d_in, cfg.d_sae) == ("relu", D, F)
        assert cfg.apply_b_dec_to_input is True and cfg.normalize_activations == "none"
        assert (cfg.hook_name, cfg.hook_layer, cfg.hook_site) == (
            "blocks.1.hook_resid_pre",
            1,
            "resid_pre",
        )
        assert cfg.center_input is True  # TransformerLens gpt2 => center_writing_weights
        assert (cfg.model_name, cfg.release, cfg.sae_id) == (
            "gpt2-small",
            "rel",
            "blocks.1.hook_resid_pre",
        )
        assert cfg.source_format == "saelens" and cfg.extra["context_size"] == 128
        x = _x()
        xc = x - x.mean(-1, keepdims=True)
        np.testing.assert_allclose(
            sae.encode(x), _ref(xc, w, "relu", b_dec_in=True), rtol=1e-5, atol=1e-5
        )
        # orientation: in-memory W_dec column i == published W_dec row i
        np.testing.assert_array_equal(sae.W_dec[:, 5].numpy(), w["W_dec"][5])

    def test_v5_jumprelu_config(self, tmp_path):
        w = _weights(1)
        folder = _write_saelens(tmp_path / "sae", _v5_jumprelu_cfg(), _core(w, "threshold"))
        sae = load_saelens(folder)
        cfg = sae.config
        assert cfg.architecture == "jumprelu" and cfg.apply_b_dec_to_input is False
        assert sae.hook_point == (0, "resid_post")
        assert cfg.center_input is False  # gemma: RMSNorm, never centred
        assert cfg.extra["prepend_bos"] is True and cfg.sae_id == "sae"
        x = _x()
        np.testing.assert_allclose(
            sae.encode(x), _ref(x, w, "jumprelu", b_dec_in=False), rtol=1e-5, atol=1e-5
        )

    def test_v6_topk_config_with_metadata_block(self, tmp_path):
        w = _weights(2)
        folder = _write_saelens(tmp_path / "sae", _v6_topk_cfg(), _core(w))
        sae = load_saelens(folder / "cfg.json")  # a file inside the folder works too
        cfg = sae.config
        assert (cfg.architecture, cfg.k) == ("topk", 4)
        assert sae.hook_point == (1, "mlp_out")
        assert cfg.model_name == "pythia-70m-deduped"
        assert cfg.center_input is False  # recorded center_writing_weights=False
        x = _x()
        np.testing.assert_allclose(
            sae.encode(x), _ref(x, w, "topk", b_dec_in=True), rtol=1e-5, atol=1e-5
        )

    def test_standard_topk_activation_fn(self, tmp_path):
        w = _weights(3)
        cfg = _legacy_cfg(
            architecture="standard",
            activation_fn_str="topk",
            activation_fn_kwargs={"k": 3},
            model_from_pretrained_kwargs={"center_writing_weights": False},
        )
        sae = load_saelens(_write_saelens(tmp_path / "s", cfg, _core(w)))
        assert (sae.config.architecture, sae.config.k) == ("topk", 3)
        np.testing.assert_allclose(
            sae.encode(_x()), _ref(_x(), w, "topk", b_dec_in=True, k=3), rtol=1e-5, atol=1e-5
        )

    def test_log_threshold_and_low_precision_weights(self, tmp_path):
        w = _weights(4)
        tensors = {k: w[k].astype(np.float16) for k in ("W_enc", "W_dec", "b_enc", "b_dec")}
        tensors["log_threshold"] = np.log(w["threshold"]).astype(np.float32)
        sae = load_saelens(_write_saelens(tmp_path / "s", _v5_jumprelu_cfg(), tensors))
        assert sae.W_enc.dtype == torch.float32
        np.testing.assert_allclose(sae.threshold.numpy(), w["threshold"], rtol=1e-5)

    def test_finetuning_scaling_factor_is_folded_into_decoder(self, tmp_path):
        w = _weights(5)
        scale = np.linspace(0.5, 2.0, F).astype(np.float32)
        cfg = _v5_jumprelu_cfg(architecture="standard", finetuning_scaling_factor=True)
        tensors = {**_core(w), "finetuning_scaling_factor": scale}
        sae = load_saelens(_write_saelens(tmp_path / "s", cfg, tensors))
        assert sae.config.extra["folded_finetuning_scaling_factor"] is True
        x = _x()
        z = _ref(x, w, "relu", b_dec_in=False)
        expected = (z * scale) @ w["W_dec"] + w["b_dec"]  # SAELens decode
        np.testing.assert_allclose(sae.reconstruct(x), expected, rtol=1e-5, atol=1e-5)

    def test_normalize_activations_modes(self, tmp_path):
        w = _weights(6)
        cfg = _v5_jumprelu_cfg(architecture="standard", normalize_activations=True)
        folder = _write_saelens(tmp_path / "s", cfg, _core(w))
        with pytest.raises(ValueError, match="norm_scaling_factor"):
            load_saelens(folder)  # bool True == expected_average_only_in, factor unknown
        sae = load_saelens(folder, norm_scaling_factor=0.5)
        assert sae.config.normalize_activations == "expected_average_only_in"
        np.testing.assert_allclose(
            sae.encode(_x()), _ref(_x() * 0.5, w, "relu", b_dec_in=False), rtol=1e-5, atol=1e-5
        )
        cfg2 = _v5_jumprelu_cfg(architecture="standard", normalize_activations="layer_norm")
        sae2 = load_saelens(_write_saelens(tmp_path / "s2", cfg2, _core(w)))
        assert sae2.config.normalize_activations == "layer_norm"
        cfg3 = _v5_jumprelu_cfg(
            architecture="standard",
            normalize_activations="expected_average_only_in",
            norm_scaling_factor=2.0,
        )
        sae3 = load_saelens(_write_saelens(tmp_path / "s3", cfg3, _core(w)))
        assert sae3.config.norm_scaling_factor == 2.0

    def test_center_input_override(self, tmp_path):
        w = _weights()
        folder = _write_saelens(tmp_path / "s", _legacy_cfg(), _core(w))
        assert load_saelens(folder, center_input=False).config.center_input is False


class TestSAELensValidation:
    def test_transposed_weights_are_rejected_with_hint(self, tmp_path):
        w = _weights()
        bad = {**_core(w), "W_enc": np.ascontiguousarray(w["W_enc"].T)}
        folder = _write_saelens(tmp_path / "s", _legacy_cfg(), bad)
        with pytest.raises(ValueError, match="transposed"):
            load_saelens(folder)

    def test_missing_and_gated_tensors(self, tmp_path):
        w = _weights()
        folder = _write_saelens(tmp_path / "a", _v5_jumprelu_cfg(), _core(w))
        with pytest.raises(ValueError, match="threshold"):
            load_saelens(folder)
        gated = {**_core(w), "b_gate": w["b_enc"], "r_mag": w["b_enc"]}
        folder = _write_saelens(tmp_path / "b", _legacy_cfg(), gated)
        with pytest.raises(ValueError, match="gated"):
            load_saelens(folder)

    @pytest.mark.parametrize(
        ("over", "match"),
        [
            ({"architecture": "gated"}, "architecture"),
            ({"architecture": "standard", "activation_fn_str": "tanh-relu"}, "activation"),
            ({"architecture": "topk"}, "'k'"),
            ({"rescale_acts_by_decoder_norm": True}, "rescale_acts_by_decoder_norm"),
            ({"normalize_activations": "mystery"}, "normalize_activations"),
        ],
    )
    def test_unsupported_options_error_clearly(self, over, match):
        with pytest.raises(ValueError, match=match):
            kwargs = saelens_config_kwargs(_legacy_cfg(**over))
            SparseAutoencoder.from_weights(**_core(_weights()), **kwargs)

    def test_missing_dims_and_paths(self, tmp_path):
        cfg = _legacy_cfg()
        del cfg["d_sae"]
        with pytest.raises(ValueError, match="d_sae"):
            saelens_config_kwargs(cfg)
        folder = tmp_path / "only_cfg"
        folder.mkdir()
        (folder / "cfg.json").write_text(json.dumps(_legacy_cfg()))
        with pytest.raises(FileNotFoundError, match="sae_weights"):
            load_saelens(folder)
        with pytest.raises(FileNotFoundError, match="neither"):
            load_saelens(tmp_path / "does-not-exist")

    def test_hook_layer_must_agree_with_hook_name(self):
        with pytest.raises(ValueError, match="disagrees"):
            saelens_config_kwargs(_legacy_cfg(hook_point_layer=3))

    def test_unsupported_hook_point_is_recorded_not_guessed(self):
        kw = saelens_config_kwargs(
            _legacy_cfg(hook_point="blocks.2.attn.hook_z", hook_point_layer=2)
        )
        assert kw["hook_site"] is None and kw["hook_layer"] == 2
        assert kw["hook_name"] == "blocks.2.attn.hook_z"


class TestCenterInputInference:
    @pytest.mark.parametrize(
        ("over", "expected"),
        [
            ({}, True),  # TransformerLens default center_writing_weights=True, LayerNorm
            ({"model_from_pretrained_kwargs": {"center_writing_weights": False}}, False),
            ({"model_from_pretrained_kwargs": {"center_writing_weights": True}}, True),
            ({"model_name": "pythia-70m-deduped"}, True),
            ({"model_name": "gemma-2-2b"}, False),
            ({"model_name": "meta-llama/Llama-3.1-8B"}, False),
            ({"model_class_name": "AutoModelForCausalLM"}, False),
        ],
    )
    def test_inference(self, over, expected):
        assert infer_center_input(_legacy_cfg(**over)) is expected

    def test_unknown_model_warns_and_assumes_uncentred(self):
        with pytest.warns(UserWarning, match="LayerNorm"):
            assert infer_center_input(_legacy_cfg(model_name="my-custom-net")) is False

    def test_center_input_reproduces_transformerlens_centred_stream(self):
        """``center_writing_weights`` makes every residual write mean-free, so the
        TransformerLens residual equals the HF residual minus its per-token mean.
        Verified on a tiny GPT-2 by centring its writing weights by hand."""
        pytest.importorskip("transformers")
        import copy

        from LLmThoughtLens.models import HookedModel

        from _tiny_hf import WordTokenizer, make_tiny_gpt2

        model = make_tiny_gpt2(seed=4, perturb_norms=True)
        with torch.no_grad():  # give every write a large mean so centring is not a no-op
            gen = torch.Generator().manual_seed(0)
            model.transformer.wpe.weight.add_(1.0 + torch.rand(1, generator=gen))
            for blk in model.transformer.h:
                blk.mlp.c_proj.bias.add_(
                    0.5 * torch.randn(blk.mlp.c_proj.bias.shape, generator=gen)
                )
        centred = copy.deepcopy(model)
        with torch.no_grad():
            tr = centred.transformer
            for p in (tr.wte.weight, tr.wpe.weight):
                p.sub_(p.mean(-1, keepdim=True))
            for blk in tr.h:
                for proj in (blk.attn.c_proj, blk.mlp.c_proj):  # Conv1D: output dim last
                    proj.weight.sub_(proj.weight.mean(-1, keepdim=True))
                    proj.bias.sub_(proj.bias.mean())
        tok = WordTokenizer()
        prompt = "the capital of texas is"
        a = HookedModel(model, tok).forward(prompt, capture_attentions=False)
        b = HookedModel(centred, tok).forward(prompt, capture_attentions=False)
        for site in ("resid_pre", "resid_post"):
            ra, rb = a.resid(site).numpy(), b.resid(site).numpy()
            np.testing.assert_allclose(rb, ra - ra.mean(-1, keepdims=True), atol=2e-5)
            assert np.abs(ra.mean(-1)).min() > 0.5  # the HF stream is far from centred
        w = _weights(7, d=ra.shape[-1], f=40)
        with_centre = SparseAutoencoder.from_weights(**_core(w), center_input=True)
        on_tl = SparseAutoencoder.from_weights(**_core(w), center_input=False)
        np.testing.assert_allclose(
            with_centre.encode(a.resid("resid_pre").numpy()[1]),
            on_tl.encode(b.resid("resid_pre").numpy()[1]),
            atol=1e-4,
        )


# ---------------------------------------------------------------------------
# safetensors headers (validate a remote file without its data)
# ---------------------------------------------------------------------------

#: The real header of jbloom/GPT2-Small-SAEs-Reformatted/blocks.6.hook_resid_pre/
#: sae_weights.safetensors (151,096,640 bytes), fetched with an HTTP range request.
_REAL_JB_HEADER = {
    "W_dec": {"dtype": "F32", "shape": [24576, 768], "data_offsets": [0, 75497472]},
    "W_enc": {"dtype": "F32", "shape": [768, 24576], "data_offsets": [75497472, 150994944]},
    "b_dec": {"dtype": "F32", "shape": [768], "data_offsets": [150994944, 150998016]},
    "b_enc": {"dtype": "F32", "shape": [24576], "data_offsets": [150998016, 151096320]},
}


def _encode_header(header: dict[str, Any]) -> bytes:
    raw = json.dumps(header).encode()
    return struct.pack("<Q", len(raw)) + raw


class TestSafetensorsHeader:
    def test_local_header_matches_tensors(self, tmp_path):
        w = _weights()
        folder = _write_saelens(tmp_path / "s", _legacy_cfg(), _core(w))
        header = read_safetensors_header(folder / "sae_weights.safetensors")
        assert tensor_shapes(header) == {k: v.shape for k, v in _core(w).items()}

    def test_real_gpt2_small_res_jb_header_validates(self):
        buf = _encode_header(_REAL_JB_HEADER) + b"\x00" * 64  # range-request prefix
        shapes = tensor_shapes(parse_safetensors_header(buf))
        cfg = _legacy_cfg(
            hook_point="blocks.6.hook_resid_pre", hook_point_layer=6, d_in=768, d_sae=24576
        )
        kwargs = saelens_config_kwargs(cfg)
        validate_saelens_shapes(kwargs, shapes)
        assert (kwargs["hook_layer"], kwargs["hook_site"]) == (6, "resid_pre")
        transposed = {**shapes, "W_enc": (24576, 768)}
        with pytest.raises(ValueError, match="transposed"):
            validate_saelens_shapes(kwargs, transposed)

    def test_bad_buffers(self, tmp_path):
        with pytest.raises(ValueError, match="shorter"):
            parse_safetensors_header(b"\x01\x02")
        with pytest.raises(ValueError, match="need"):
            parse_safetensors_header(struct.pack("<Q", 100) + b"{}")
        with pytest.raises(ValueError, match="implausible"):
            parse_safetensors_header(struct.pack("<Q", 2**40))
        with pytest.raises(ValueError, match="JSON object"):
            parse_safetensors_header(_encode_header([1, 2]))  # type: ignore[arg-type]
        short = tmp_path / "short.safetensors"
        short.write_bytes(b"abc")
        with pytest.raises(ValueError, match="too short"):
            read_safetensors_header(short)


# ---------------------------------------------------------------------------
# Gemma Scope
# ---------------------------------------------------------------------------


def _write_gemma(path: Path, w: dict[str, np.ndarray], drop: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in w.items() if k != drop}
    np.savez(path, **arrays)
    return path


class TestGemmaScope:
    def test_load_infers_hook_and_matches_reference(self, tmp_path):
        w = _weights(8)
        npz = _write_gemma(
            tmp_path / "gemma-scope-2b-pt-res/layer_3/width_16k/average_l0_10/params.npz", w
        )
        sae = load_gemma_scope(npz)
        cfg = sae.config
        assert cfg.architecture == "jumprelu" and cfg.apply_b_dec_to_input is False
        assert cfg.normalize_activations == "none" and cfg.center_input is False
        assert sae.hook_point == (3, "resid_post") and cfg.hook_name == "blocks.3.hook_resid_post"
        assert cfg.model_name == "google/gemma-2-2b" and cfg.source_format == "gemma_scope"
        assert cfg.sae_id == "layer_3/width_16k/average_l0_10"
        x = _x()
        np.testing.assert_allclose(
            sae.encode(x), _ref(x, w, "jumprelu", b_dec_in=False), rtol=1e-5, atol=1e-5
        )
        # Gemma Scope reconstruction: decode adds b_dec back, no input shift was applied
        z = _ref(x, w, "jumprelu", b_dec_in=False)
        np.testing.assert_allclose(
            sae.reconstruct(x), z @ w["W_dec"] + w["b_dec"], rtol=1e-5, atol=1e-5
        )

    def test_overrides_and_validation(self, tmp_path):
        w = _weights(9)
        npz = _write_gemma(tmp_path / "plain/params.npz", w)
        sae = load_gemma_scope(npz)
        assert sae.hook_point is None and sae.config.sae_id == "plain"
        sae = load_gemma_scope(npz, hook_layer=5, hook_site="resid_post", model_name="m")
        assert sae.hook_point == (5, "resid_post") and sae.config.model_name == "m"
        with pytest.raises(ValueError, match="hook_site"):
            load_gemma_scope(npz, hook_layer=5, hook_site="resid_mid")
        bad = _write_gemma(tmp_path / "bad/params.npz", w, drop="threshold")
        with pytest.raises(ValueError, match="threshold"):
            load_gemma_scope(bad)
        with pytest.raises(FileNotFoundError):
            load_gemma_scope(tmp_path / "missing.npz")

    @pytest.mark.parametrize(
        ("identifier", "expected"),
        [
            (
                "google/gemma-scope-2b-pt-res/layer_20/width_16k/average_l0_71/params.npz",
                (20, "resid_post", "blocks.20.hook_resid_post", "google/gemma-2-2b"),
            ),
            (
                "google/gemma-scope-9b-it-res/layer_9/width_16k/average_l0_88/params.npz",
                (9, "resid_post", "blocks.9.hook_resid_post", "google/gemma-2-9b-it"),
            ),
            (
                "google/gemma-scope-2b-pt-mlp/layer_4/width_16k/average_l0_50/params.npz",
                (4, "mlp_out", "blocks.4.hook_mlp_out", "google/gemma-2-2b"),
            ),
            (
                "google/gemma-scope-2b-pt-att/layer_4/width_16k/average_l0_50/params.npz",
                (4, None, "blocks.4.attn.hook_z", "google/gemma-2-2b"),
            ),
            (
                "google/gemma-scope-2b-pt-res/embedding/width_4k/average_l0_6/params.npz",
                (None, None, "hook_embed", "google/gemma-2-2b"),
            ),
        ],
    )
    def test_hook_inference(self, identifier, expected):
        assert gemma_scope_hook(identifier) == expected


# ---------------------------------------------------------------------------
# Registry + Hub access (all monkeypatched; nothing touches the network)
# ---------------------------------------------------------------------------


class _FakeHub:
    """Serves files from a local mirror and records every hf_hub_download call."""

    def __init__(self, mirror: Path) -> None:
        self.mirror = mirror
        self.calls: list[dict[str, Any]] = []

    def __call__(self, repo_id: str, filename: str, revision: str | None = None, **kw: Any) -> str:
        self.calls.append({"repo_id": repo_id, "filename": filename, "revision": revision, **kw})
        path = self.mirror / repo_id / filename
        if not path.is_file():
            raise FileNotFoundError(f"not in mirror: {repo_id}/{filename}")
        return str(path)


@pytest.fixture
def fake_hub(tmp_path, monkeypatch) -> _FakeHub:
    hub = pytest.importorskip("huggingface_hub")
    fake = _FakeHub(tmp_path / "mirror")
    monkeypatch.setattr(hub, "hf_hub_download", fake)
    return fake


class TestRegistry:
    def test_known_releases(self):
        listing = list_pretrained()
        assert {"gpt2-small-res-jb", "gemma-scope-2b-pt-res"} <= set(listing)
        jb = PRETRAINED_RELEASES["gpt2-small-res-jb"]
        assert jb.repo_id == "jbloom/GPT2-Small-SAEs-Reformatted" and jb.model == "gpt2"
        assert jb.path_for("blocks.6.hook_resid_pre") == "blocks.6.hook_resid_pre"
        assert jb.path_for("blocks.11.hook_resid_post") == "blocks.11.hook_resid_post"
        gs = PRETRAINED_RELEASES["gemma-scope-2b-pt-res"]
        assert gs.path_for("layer_20/width_16k/average_l0_71") == (
            "layer_20/width_16k/average_l0_71/params.npz"
        )

    @pytest.mark.parametrize(
        ("release", "sae_id"),
        [
            ("gpt2-small-res-jb", "blocks.12.hook_resid_pre"),
            ("gpt2-small-res-jb", "blocks.6.hook_mlp_out"),
            ("gemma-scope-2b-pt-res", "layer_26/width_16k/average_l0_71"),
        ],
    )
    def test_invalid_ids_raise(self, release, sae_id):
        with pytest.raises(ValueError, match="not a valid sae_id"):
            PRETRAINED_RELEASES[release].path_for(sae_id)

    def test_unknown_release(self):
        with pytest.raises(KeyError, match="unknown SAE release"):
            from_pretrained("no-such-release", "x")

    def test_from_pretrained_saelens_downloads_lazily(self, fake_hub):
        w = _weights(10)
        repo = fake_hub.mirror / "jbloom/GPT2-Small-SAEs-Reformatted"
        _write_saelens(
            repo / "blocks.1.hook_resid_pre",
            _legacy_cfg(d_in=D, d_sae=F),
            _core(w),
        )
        sae = from_pretrained("gpt2-small-res-jb", "blocks.1.hook_resid_pre", revision="abc")
        assert [c["filename"] for c in fake_hub.calls] == [
            "blocks.1.hook_resid_pre/cfg.json",
            "blocks.1.hook_resid_pre/sae_weights.safetensors",
        ]
        assert all(
            c["repo_id"] == "jbloom/GPT2-Small-SAEs-Reformatted" and c["revision"] == "abc"
            for c in fake_hub.calls
        )
        assert all(c["local_files_only"] is False for c in fake_hub.calls)
        cfg = sae.config
        assert (cfg.release, cfg.sae_id) == ("gpt2-small-res-jb", "blocks.1.hook_resid_pre")
        assert cfg.extra["hf_model"] == "gpt2" and sae.hook_point == (1, "resid_pre")

    def test_from_pretrained_gemma_scope(self, fake_hub):
        w = _weights(11)
        sae_id = "layer_2/width_16k/average_l0_9"
        _write_gemma(fake_hub.mirror / "google/gemma-scope-2b-pt-res" / sae_id / "params.npz", w)
        sae = from_pretrained("gemma-scope-2b-pt-res", sae_id, local_files_only=True)
        assert fake_hub.calls[0]["filename"] == f"{sae_id}/params.npz"
        assert fake_hub.calls[0]["local_files_only"] is True
        assert sae.hook_point == (2, "resid_post") and sae.config.sae_id == sae_id
        assert sae.config.extra["hf_model"] == "google/gemma-2-2b"

    def test_repo_id_release_and_repo_file_spec(self, fake_hub):
        w = _weights(12)
        _write_saelens(fake_hub.mirror / "org/sae-repo/layer1", _legacy_cfg(), _core(w))
        sae = from_pretrained("org/sae-repo", "layer1")
        assert sae.config.release == "org/sae-repo" and "hf_model" not in sae.config.extra
        _write_gemma(fake_hub.mirror / "org/gs/layer_1/x/params.npz", w)
        assert from_pretrained("org/gs", "layer_1/x/params.npz").config.source_format == (
            "gemma_scope"
        )
        sae = load_gemma_scope("org/gs/layer_1/x/params.npz")
        assert sae.config.sae_id == "layer_1/x"
        with pytest.raises(FileNotFoundError, match="neither"):
            load_gemma_scope("params.npz")

    def test_offline_failure_says_so(self, fake_hub, monkeypatch):
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        with pytest.raises(FileNotFoundError, match="HF_HUB_OFFLINE"):
            from_pretrained("gpt2-small-res-jb", "blocks.3.hook_resid_pre")
        monkeypatch.setenv("HF_HUB_OFFLINE", "0")
        with pytest.raises(FileNotFoundError) as info:
            from_pretrained("gpt2-small-res-jb", "blocks.3.hook_resid_pre")
        assert "HF_HUB_OFFLINE" not in str(info.value)

    def test_describe_pretrained_reads_metadata_only(self, fake_hub, monkeypatch):
        hub = pytest.importorskip("huggingface_hub")
        cfg_path = fake_hub.mirror / "jbloom/GPT2-Small-SAEs-Reformatted/blocks.6.hook_resid_pre"
        cfg_path.mkdir(parents=True)
        (cfg_path / "cfg.json").write_text(
            json.dumps(
                _legacy_cfg(
                    hook_point="blocks.6.hook_resid_pre",
                    hook_point_layer=6,
                    d_in=768,
                    d_sae=24576,
                )
            )
        )
        sizes = {
            "blocks.6.hook_resid_pre/cfg.json": 1274,
            "blocks.6.hook_resid_pre/sae_weights.safetensors": 151096640,
            "layer_20/width_16k/average_l0_71/params.npz": 302131416,
        }

        class _Api:
            def get_paths_info(self, repo_id, files, revision=None):
                return [SimpleNamespace(path=f, size=sizes[f]) for f in files if f in sizes]

            def parse_safetensors_file_metadata(self, repo_id, filename, revision=None):
                return SimpleNamespace(
                    tensors={
                        k: SimpleNamespace(dtype=v["dtype"], shape=v["shape"])
                        for k, v in _REAL_JB_HEADER.items()
                    }
                )

        monkeypatch.setattr(hub, "HfApi", _Api)
        info = describe_pretrained("gpt2-small-res-jb", "blocks.6.hook_resid_pre")
        assert info["files"] == {k: v for k, v in sizes.items() if k.startswith("blocks.6")}
        assert info["tensors"]["W_enc"] == ("F32", (768, 24576))
        assert info["config"]["hook_site"] == "resid_pre" and info["config"]["center_input"]
        assert [c["filename"] for c in fake_hub.calls] == ["blocks.6.hook_resid_pre/cfg.json"]
        with pytest.raises(FileNotFoundError, match="has no"):
            describe_pretrained("gpt2-small-res-jb", "blocks.7.hook_resid_pre")
        gemma = describe_pretrained("gemma-scope-2b-pt-res", "layer_20/width_16k/average_l0_71")
        assert gemma["format"] == "gemma_scope" and "cfg" not in gemma
        assert gemma["files"] == {"layer_20/width_16k/average_l0_71/params.npz": 302131416}
        assert len(fake_hub.calls) == 1  # sizes only: the 302 MB file was not fetched


def test_importing_loaders_needs_no_hub_or_torch():
    """Downloads are lazy: importing the package must not import huggingface_hub / torch."""
    code = (
        "import sys, LLmThoughtLens.features, LLmThoughtLens.features.sae_loaders;"
        "print(sorted(m for m in ('torch', 'huggingface_hub', 'safetensors') if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=60
    )
    assert out.stdout.strip() == "[]"


# ---------------------------------------------------------------------------
# Real weights (only when already cached — never downloads)
# ---------------------------------------------------------------------------


def _reconstruction_report(sae: Any, acts: np.ndarray) -> dict[str, float]:
    """MSE, explained variance (per-dimension centred) and L0 of *sae* on ``(N, D)`` acts."""
    codes = sae.encode(acts)
    recon = sae.reconstruct(acts)
    resid = ((acts - recon) ** 2).sum(-1)
    total = ((acts - acts.mean(0)) ** 2).sum(-1)
    return {
        "mse": float(((acts - recon) ** 2).mean()),
        "explained_variance": float(1.0 - resid.sum() / total.sum()),
        "l0": float((codes > 0).sum(-1).mean()),
    }


def _gpt2_provider() -> Any:
    pytest.importorskip("transformers")
    from LLmThoughtLens.models.hooked import load_hf_model
    from LLmThoughtLens.providers.huggingface_provider import HuggingFaceProvider

    try:
        model, tok, device = load_hf_model("gpt2", device="cpu", local_files_only=True)
    except Exception as exc:  # not cached
        pytest.skip(f"gpt2 not available offline: {exc}")
    provider = HuggingFaceProvider(model_name="gpt2", device="cpu")
    provider._model, provider._tokenizer, provider._device = model, tok, device
    return provider


_PROMPTS = (
    "The capital of the state containing Dallas is",
    "In 1969 Neil Armstrong became the first person to walk on the Moon.",
    "def fibonacci(n): return n if n < 2 else fibonacci(n - 1) + fibonacci(n - 2)",
)


@pytest.mark.slow
class TestRealGPT2SmallSAE:
    def test_report_helper_on_real_gpt2_with_random_sae(self):
        """Exercises the real-weights code path with a random SAE (no SAE download)."""
        provider = _gpt2_provider()
        out = provider.run(_PROMPTS[0])
        w = _weights(13, d=768, f=1536)
        sae = SparseAutoencoder.from_weights(
            **_core(w), hook_name="blocks.6.hook_resid_pre", center_input=True
        )
        from LLmThoughtLens.features.extractor import sae_input_activations

        acts, prov = sae_input_activations(out, 6, "resid_pre")
        assert prov == {"activation_source": "activations", "activation_layer": 5}
        report = _reconstruction_report(sae, acts[1:])
        assert np.isfinite(list(report.values())).all()

    def test_pretrained_sae_if_cached(self):
        try:
            sae = from_pretrained(
                "gpt2-small-res-jb", "blocks.6.hook_resid_pre", local_files_only=True
            )
        except FileNotFoundError as exc:
            pytest.skip(f"gpt2-small-res-jb weights not cached (151 MB, not downloaded): {exc}")
        assert sae.config.center_input and sae.hook_point == (6, "resid_pre")
        provider = _gpt2_provider()
        from LLmThoughtLens.features.extractor import FeatureExtractor, sae_input_activations

        rows = []
        for prompt in _PROMPTS:
            acts, _ = sae_input_activations(provider.run(prompt), 6, "resid_pre")
            rows.append(acts[1:])  # position 0 is the attention sink (no BOS here)
        report = _reconstruction_report(sae, np.concatenate(rows))
        assert report["explained_variance"] > 0.5, report
        assert 1.0 < report["l0"] < 1000.0, report
        ex = FeatureExtractor(top_k=10, exclude_positions=[0])
        ex.attach_sae(sae)
        feats = ex.extract(provider.run(_PROMPTS[0]), provider=provider)
        assert feats and all(f.meta["sae_layer"] == 6 and f.layer == 5 for f in feats)
