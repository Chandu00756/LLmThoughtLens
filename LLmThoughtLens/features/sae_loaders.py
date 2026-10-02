"""Load pretrained sparse autoencoders: SAELens folders, Gemma Scope ``params.npz``, a registry.

Formats
-------
**SAELens** — a folder with ``cfg.json`` and ``sae_weights.safetensors``.  The
weights use the published orientation ``W_enc (d_in, d_sae)``,
``W_dec (d_sae, d_in)``, ``b_enc (d_sae,)``, ``b_dec (d_in,)`` plus
``threshold (d_sae,)`` for JumpReLU SAEs.  Three generations of ``cfg.json``
are understood:

* legacy (e.g. ``jbloom/GPT2-Small-SAEs-Reformatted``): ``hook_point``,
  ``hook_point_layer``, no ``architecture`` (a ReLU SAE that subtracts ``b_dec``);
* SAELens 3-5: ``hook_name``, ``hook_layer``, ``architecture``
  (``"standard"`` / ``"jumprelu"`` / ``"topk"``), ``activation_fn_str`` /
  ``activation_fn_kwargs``, ``apply_b_dec_to_input``, ``normalize_activations``;
* SAELens 6: the same with the provenance under ``metadata``.

**Gemma Scope** — one ``params.npz`` with ``W_enc (d_model, d_sae)``,
``W_dec (d_sae, d_model)``, ``b_enc``, ``b_dec``, ``threshold``: a JumpReLU SAE
that does *not* subtract ``b_dec`` from its input.  Residual SAEs read
``blocks.L.hook_resid_post``; the layer and site are inferred from the path
(``…/gemma-scope-2b-pt-res/layer_20/…``) unless given explicitly.

Hook-point convention
---------------------
SAELens hook names are TransformerLens names; they map 1:1 onto
:class:`~LLmThoughtLens.models.HookedModel` sites (``blocks.L.hook_resid_pre``
= input to block ``L``).  TransformerLens loads LayerNorm models such as GPT-2
with ``center_writing_weights=True``, so their residual stream is the
HuggingFace residual minus its per-token mean; the loader records that as
``SAEConfig.center_input`` (inferred from ``model_from_pretrained_kwargs`` and
the model family, overridable with ``center_input=``).

Downloads
---------
Nothing here touches the network at import time.  Remote loading
(:func:`from_pretrained`, or a repo id passed to :func:`load_saelens` /
:func:`load_gemma_scope`) lazily imports ``huggingface_hub`` and calls
``hf_hub_download``, which honours ``HF_HUB_OFFLINE`` (cached files only).
:func:`describe_pretrained` reports file sizes and tensor shapes from the Hub
*without* downloading the weights.
"""

from __future__ import annotations

import json
import os
import re
import struct
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from LLmThoughtLens.features.sae import SAE_SITES, parse_hook_name

if TYPE_CHECKING:
    from LLmThoughtLens.features.sae import SparseAutoencoder

SAELENS_CFG = "cfg.json"
SAELENS_WEIGHTS = "sae_weights.safetensors"
GEMMA_SCOPE_FILE = "params.npz"

#: Provenance keys copied from a SAELens cfg into ``SAEConfig.extra``.
_EXTRA_KEYS: tuple[str, ...] = (
    "context_size",
    "prepend_bos",
    "dataset_path",
    "neuronpedia_id",
    "model_class_name",
    "model_from_pretrained_kwargs",
    "hook_head_index",
    "run_name",
    "sae_lens_training_version",
    "sae_lens_version",
    "dtype",
    "l1_coefficient",
    "expansion_factor",
)

# TransformerLens model families with LayerNorm (writing weights centred by
# default) vs RMSNorm (centring is skipped).  Matched as substrings of the
# lower-cased model name.
_LAYERNORM_MODELS: tuple[str, ...] = (
    "gpt2",
    "pythia",
    "gpt-neo",
    "gpt-j",
    "opt-",
    "solu-",
    "gelu-",
    "attn-only",
    "bloom",
    "santacoder",
    "codegen",
    "phi-1",
    "phi-2",
    "tiny-stories",
)
_RMSNORM_MODELS: tuple[str, ...] = (
    "llama",
    "mistral",
    "mixtral",
    "gemma",
    "qwen",
    "phi-3",
)

_MAX_HEADER_BYTES = 100 * 1024 * 1024


# ---------------------------------------------------------------------------
# safetensors header (pure Python — no weights read)
# ---------------------------------------------------------------------------


def parse_safetensors_header(buf: bytes) -> dict[str, Any]:
    """Parse a safetensors header from the first bytes of a file.

    *buf* must hold at least the 8-byte little-endian header length and the
    JSON header that follows (e.g. the result of an HTTP range request).
    Returns ``{tensor_name: {"dtype", "shape", "data_offsets"}}`` plus the
    optional ``"__metadata__"`` entry.
    """
    if len(buf) < 8:
        raise ValueError("safetensors buffer is shorter than its 8-byte length prefix")
    (n,) = struct.unpack("<Q", buf[:8])
    if n > _MAX_HEADER_BYTES:
        raise ValueError(f"implausible safetensors header length {n}")
    if len(buf) < 8 + n:
        raise ValueError(f"need {8 + n} bytes for the safetensors header, got {len(buf)}")
    header = json.loads(buf[8 : 8 + n].decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("safetensors header is not a JSON object")
    return header


def read_safetensors_header(path: str | Path) -> dict[str, Any]:
    """Read only the header of a local ``.safetensors`` file (no tensor data)."""
    with open(path, "rb") as fh:
        prefix = fh.read(8)
        if len(prefix) < 8:
            raise ValueError(f"{path}: not a safetensors file (too short)")
        (n,) = struct.unpack("<Q", prefix)
        if n > _MAX_HEADER_BYTES:
            raise ValueError(f"{path}: implausible safetensors header length {n}")
        return parse_safetensors_header(prefix + fh.read(n))


def tensor_shapes(header: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    """``{name: shape}`` for every tensor in a parsed safetensors header."""
    return {
        name: tuple(int(d) for d in info["shape"])
        for name, info in header.items()
        if name != "__metadata__"
    }


# ---------------------------------------------------------------------------
# SAELens cfg.json -> SAEConfig fields
# ---------------------------------------------------------------------------


def _flatten_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Merge SAELens 6 ``metadata`` into the top level (top-level keys win)."""
    meta = cfg.get("metadata")
    flat = dict(meta) if isinstance(meta, dict) else {}
    flat.update({k: v for k, v in cfg.items() if k != "metadata"})
    return flat


def _architecture(flat: dict[str, Any]) -> tuple[str, int | None]:
    arch = str(flat.get("architecture") or "standard").lower()
    fn_kwargs = flat.get("activation_fn_kwargs") or {}
    if not isinstance(fn_kwargs, dict):
        fn_kwargs = {}
    k = flat.get("k", fn_kwargs.get("k"))
    if flat.get("rescale_acts_by_decoder_norm"):
        raise ValueError("SAELens option rescale_acts_by_decoder_norm=True is not supported")
    if arch == "standard":
        fn = str(flat.get("activation_fn_str") or flat.get("activation_fn") or "relu").lower()
        if fn == "relu":
            return "relu", None
        if fn == "topk":
            arch = "topk"
        else:
            raise ValueError(
                f"unsupported SAELens activation function {fn!r} (supported: 'relu', 'topk')"
            )
    if arch == "topk":
        if k is None:
            raise ValueError("SAELens TopK config has no 'k' (activation_fn_kwargs.k)")
        return "topk", int(k)
    if arch == "jumprelu":
        return "jumprelu", None
    raise ValueError(
        f"unsupported SAELens architecture {arch!r} (supported: 'standard', 'topk', 'jumprelu')"
    )


def _normalisation(flat: dict[str, Any]) -> str:
    mode = flat.get("normalize_activations", "none")
    if isinstance(mode, bool):  # SAELens backward compatibility
        return "expected_average_only_in" if mode else "none"
    return "none" if mode is None else str(mode)


def _uses_layernorm(model_name: str | None) -> bool | None:
    name = (model_name or "").lower()
    if not name:
        return None
    if any(p in name for p in _RMSNORM_MODELS):
        return False
    if any(p in name for p in _LAYERNORM_MODELS):
        return True
    return None


def infer_center_input(cfg: dict[str, Any]) -> bool:
    """Whether a SAELens SAE expects HF activations with their per-token mean removed.

    True when the SAE was trained on TransformerLens activations of a
    LayerNorm model loaded with ``center_writing_weights=True`` (the
    TransformerLens default; newer SAELens runs record
    ``model_from_pretrained_kwargs={"center_writing_weights": False}``).
    """
    flat = _flatten_cfg(cfg)
    model_class = str(flat.get("model_class_name") or "HookedTransformer")
    if model_class not in ("HookedTransformer", "HookedSAETransformer"):
        return False
    kwargs = flat.get("model_from_pretrained_kwargs") or {}
    if not isinstance(kwargs, dict) or not kwargs.get("center_writing_weights", True):
        return False
    uses_ln = _uses_layernorm(flat.get("model_name"))
    if uses_ln is None:
        warnings.warn(
            f"cannot tell whether {flat.get('model_name')!r} uses LayerNorm; assuming the SAE "
            "reads uncentred activations. Pass center_input=True if it was trained on a "
            "TransformerLens LayerNorm model with center_writing_weights=True.",
            UserWarning,
            stacklevel=3,
        )
        return False
    return uses_ln


def saelens_config_kwargs(
    cfg: dict[str, Any],
    *,
    norm_scaling_factor: float | None = None,
    center_input: bool | None = None,
) -> dict[str, Any]:
    """Translate a SAELens ``cfg.json`` dict into :class:`SAEConfig` keyword arguments."""
    flat = _flatten_cfg(cfg)
    try:
        d_in, d_sae = int(flat["d_in"]), int(flat["d_sae"])
    except KeyError as exc:
        raise ValueError(f"SAELens cfg.json is missing {exc.args[0]!r}") from exc
    arch, k = _architecture(flat)

    hook_name = flat.get("hook_name") or flat.get("hook_point")
    cfg_layer = flat.get("hook_layer", flat.get("hook_point_layer"))
    head = flat.get("hook_head_index", flat.get("hook_point_head_index"))
    hook_layer: int | None = None if cfg_layer is None else int(cfg_layer)
    hook_site: str | None = None
    parsed = parse_hook_name(hook_name) if hook_name else None
    if parsed is not None and head is None:
        if hook_layer is not None and hook_layer != parsed[0]:
            raise ValueError(
                f"cfg.json hook_name {hook_name!r} disagrees with hook_layer={hook_layer}"
            )
        hook_layer, hook_site = parsed

    factor = norm_scaling_factor
    if factor is None:
        for key in ("norm_scaling_factor", "activation_norm_scaling_factor"):
            if flat.get(key) is not None:
                factor = float(flat[key])
                break

    extra = {key: flat[key] for key in _EXTRA_KEYS if key in flat}
    kwargs: dict[str, Any] = {
        "input_dim": d_in,
        "dict_size": d_sae,
        "architecture": arch,
        "apply_b_dec_to_input": bool(flat.get("apply_b_dec_to_input", True)),
        "normalize_activations": _normalisation(flat),
        "norm_scaling_factor": factor,
        "center_input": infer_center_input(cfg) if center_input is None else bool(center_input),
        "hook_name": hook_name,
        "hook_layer": hook_layer,
        "hook_site": hook_site,
        "model_name": flat.get("model_name"),
        "source_format": "saelens",
        "extra": extra,
        "device": "cpu",
    }
    if k is not None:
        kwargs["k"] = k
    return kwargs


def validate_saelens_shapes(
    config_kwargs: dict[str, Any], shapes: dict[str, tuple[int, ...]]
) -> None:
    """Check tensor names / shapes / orientation against a translated SAELens config.

    Works from a safetensors *header* (see :func:`tensor_shapes`), so it can
    validate a remote file without downloading its data.
    """
    d_in, d_sae = int(config_kwargs["input_dim"]), int(config_kwargs["dict_size"])
    expect: dict[str, tuple[int, ...]] = {
        "W_enc": (d_in, d_sae),
        "W_dec": (d_sae, d_in),
        "b_enc": (d_sae,),
        "b_dec": (d_in,),
    }
    if config_kwargs["architecture"] == "jumprelu":
        name = "threshold" if "threshold" in shapes else "log_threshold"
        expect[name] = (d_sae,)
    for gated in ("b_gate", "r_mag", "b_mag"):
        if gated in shapes:
            raise ValueError(f"tensor {gated!r} found: gated SAEs are not supported")
    for name, shape in expect.items():
        if name not in shapes:
            raise ValueError(f"SAELens weights are missing {name!r} (have {sorted(shapes)})")
        if shapes[name] != shape:
            hint = ""
            if len(shape) == 2 and shapes[name] == shape[::-1]:
                hint = (
                    " (transposed: SAELens stores W_enc as (d_in, d_sae), W_dec as (d_sae, d_in))"
                )
            raise ValueError(
                f"SAELens tensor {name!r} has shape {shapes[name]}, expected {shape} "
                f"for d_in={d_in}, d_sae={d_sae}{hint}"
            )


# ---------------------------------------------------------------------------
# Hub access (lazy; honours HF_HUB_OFFLINE)
# ---------------------------------------------------------------------------


def _offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").strip().lower() not in ("", "0", "false", "no")


def _hub_download(
    repo_id: str, filename: str, revision: str | None, local_files_only: bool
) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover — gated by extras
        raise ImportError(
            "downloading SAEs needs huggingface_hub: pip install 'LLmThoughtLens[huggingface]'"
        ) from exc
    try:
        return Path(
            hf_hub_download(repo_id, filename, revision=revision, local_files_only=local_files_only)
        )
    except Exception as exc:
        hint = (
            " (HF_HUB_OFFLINE is set: only files already in the local Hugging Face cache "
            "can be used)"
            if _offline()
            else ""
        )
        raise FileNotFoundError(f"could not fetch {repo_id}/{filename}{hint}: {exc}") from exc


def _looks_like_repo_id(spec: str) -> bool:
    return (
        "/" in spec
        and not spec.startswith(("/", ".", "~"))
        and not re.match(r"^[A-Za-z]:[\\/]", spec)
    )


# ---------------------------------------------------------------------------
# SAELens
# ---------------------------------------------------------------------------


def _load_safetensors(path: Path) -> dict[str, Any]:
    from safetensors.torch import load_file

    return load_file(str(path), device="cpu")


def load_saelens(
    path_or_repo: str | Path,
    subfolder: str | None = None,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
    device: str = "cpu",
    norm_scaling_factor: float | None = None,
    center_input: bool | None = None,
    release: str | None = None,
    sae_id: str | None = None,
) -> SparseAutoencoder:
    """Load a SAELens SAE from a local folder or a Hugging Face repo.

    Parameters
    ----------
    path_or_repo:
        A local folder (or its ``cfg.json`` / weights file), or a Hub repo id
        such as ``"jbloom/GPT2-Small-SAEs-Reformatted"`` (downloaded with
        ``hf_hub_download``; honours ``HF_HUB_OFFLINE``).
    subfolder:
        Folder inside *path_or_repo*, e.g. ``"blocks.8.hook_resid_pre"``.
    norm_scaling_factor:
        Dataset scaling factor for ``normalize_activations="expected_average_only_in"``
        (overrides any value in ``cfg.json``).
    center_input:
        Override the inferred TransformerLens ``center_writing_weights`` handling.
    release, sae_id:
        Provenance recorded in the config (``sae_id`` defaults to *subfolder*).
    """
    import torch

    from LLmThoughtLens.features.sae import SparseAutoencoder

    spec = str(path_or_repo)
    local = Path(spec).expanduser()
    if local.exists():
        folder = local.parent if local.is_file() else local
        if subfolder:
            folder = folder / subfolder
        cfg_path, weights_path = folder / SAELENS_CFG, folder / SAELENS_WEIGHTS
        for p in (cfg_path, weights_path):
            if not p.is_file():
                raise FileNotFoundError(f"{p} not found (expected a SAELens folder)")
        default_id = subfolder or folder.name
    elif _looks_like_repo_id(spec):
        prefix = f"{subfolder.strip('/')}/" if subfolder else ""
        cfg_path = _hub_download(spec, prefix + SAELENS_CFG, revision, local_files_only)
        weights_path = _hub_download(spec, prefix + SAELENS_WEIGHTS, revision, local_files_only)
        default_id = subfolder or spec
    else:
        raise FileNotFoundError(f"{spec!r} is neither an existing path nor a Hub repo id")

    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    kwargs = saelens_config_kwargs(
        cfg, norm_scaling_factor=norm_scaling_factor, center_input=center_input
    )
    shapes = tensor_shapes(read_safetensors_header(weights_path))
    validate_saelens_shapes(kwargs, shapes)

    tensors = _load_safetensors(Path(weights_path))
    threshold = tensors.get("threshold")
    if kwargs["architecture"] == "jumprelu" and threshold is None:
        threshold = torch.exp(tensors["log_threshold"].to(torch.float32))
    W_dec = tensors["W_dec"].to(torch.float32)
    scale = tensors.get("finetuning_scaling_factor", tensors.get("scaling_factor"))
    if scale is not None:
        scale = scale.to(torch.float32).reshape(-1)
        if scale.shape[0] != W_dec.shape[0]:
            raise ValueError(f"finetuning scaling factor has shape {tuple(scale.shape)}")
        if not torch.allclose(scale, torch.ones_like(scale)):
            # SAELens multiplies feature activations by it before decoding;
            # folding it into the decoder rows is exactly equivalent.
            W_dec = W_dec * scale[:, None]
            kwargs["extra"]["folded_finetuning_scaling_factor"] = True

    kwargs["release"] = release
    kwargs["sae_id"] = sae_id or default_id
    sae = SparseAutoencoder.from_weights(
        tensors["W_enc"],
        W_dec,
        tensors["b_enc"],
        tensors["b_dec"],
        threshold if kwargs["architecture"] == "jumprelu" else None,
        **kwargs,
    )
    return sae.to(device) if device != "cpu" else sae


# ---------------------------------------------------------------------------
# Gemma Scope
# ---------------------------------------------------------------------------

_GEMMA_REPO_RE = re.compile(r"gemma-scope-(\d+b)-(pt|it)-(res|mlp|att|transcoders)")


def gemma_scope_hook(identifier: str) -> tuple[int | None, str | None, str | None, str | None]:
    """Infer ``(hook_layer, hook_site, hook_name, model_name)`` from a Gemma Scope path / id.

    ``-res`` releases read ``blocks.L.hook_resid_post`` and ``-mlp`` releases
    ``blocks.L.hook_mlp_out``.  Attention SAEs read the per-head ``hook_z``
    and the ``embedding`` SAEs the token embeddings — neither is a supported
    site, so ``hook_site`` is ``None`` for them.
    """
    layer_m = re.search(r"layer_(\d+)", identifier)
    layer = int(layer_m.group(1)) if layer_m else None
    repo_m = _GEMMA_REPO_RE.search(identifier)
    model = None
    site: str | None = None
    hook_name: str | None = None
    if repo_m:
        size, variant, kind = repo_m.groups()
        model = f"google/gemma-2-{size}" + ("-it" if variant == "it" else "")
        if kind == "res":
            site = "resid_post"
        elif kind == "mlp":
            site = "mlp_out"
        elif kind == "att" and layer is not None:
            hook_name = f"blocks.{layer}.attn.hook_z"
    if re.search(r"(^|/)embedding(/|$)", identifier):
        layer, site, hook_name = None, None, "hook_embed"
    if site is not None and layer is not None:
        hook_name = f"blocks.{layer}.hook_{site}"
    elif site is not None:
        site = None
    return layer, site, hook_name, model


def load_gemma_scope(
    npz_path_or_repo_file: str | Path,
    *,
    repo_id: str | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
    hook_layer: int | None = None,
    hook_site: str | None = None,
    model_name: str | None = None,
    device: str = "cpu",
    release: str | None = None,
    sae_id: str | None = None,
) -> SparseAutoencoder:
    """Load a Gemma Scope JumpReLU SAE from ``params.npz``.

    Parameters
    ----------
    npz_path_or_repo_file:
        A local ``params.npz``; or a file inside *repo_id*; or
        ``"google/gemma-scope-2b-pt-res/layer_20/width_16k/average_l0_71/params.npz"``
        (repo id followed by the file path).
    hook_layer, hook_site, model_name:
        Override what is inferred from the path (see :func:`gemma_scope_hook`).
    """
    from LLmThoughtLens.features.sae import SparseAutoencoder

    spec = str(npz_path_or_repo_file)
    local = Path(spec).expanduser()
    if repo_id is None and local.is_file():
        path, identifier = local, str(local.resolve())
    else:
        if repo_id is None:
            parts = spec.strip("/").split("/")
            if len(parts) < 3 or not _looks_like_repo_id(spec):
                raise FileNotFoundError(
                    f"{spec!r} is neither an existing file nor '<org>/<repo>/<path>/params.npz'"
                )
            repo_id, filename = "/".join(parts[:2]), "/".join(parts[2:])
        else:
            filename = spec.strip("/")
        path = _hub_download(repo_id, filename, revision, local_files_only)
        identifier = f"{repo_id}/{filename}"

    inf_layer, inf_site, inf_hook, inf_model = gemma_scope_hook(identifier)
    layer = inf_layer if hook_layer is None else int(hook_layer)
    site = inf_site if hook_site is None else hook_site
    if site is not None and site not in SAE_SITES:
        raise ValueError(f"unknown hook_site {site!r}; expected one of {SAE_SITES}")
    hook_name = inf_hook
    if layer is not None and site is not None:
        hook_name = f"blocks.{layer}.hook_{site}"

    with np.load(path) as params:
        missing = {"W_enc", "W_dec", "b_enc", "b_dec", "threshold"} - set(params.files)
        if missing:
            raise ValueError(f"{path}: Gemma Scope params are missing {sorted(missing)}")
        arrays = {k: np.asarray(params[k], dtype=np.float32) for k in params.files}

    if sae_id is None:
        m = re.search(r"(layer_\d+/.*?)/params\.npz$", identifier) or re.search(
            r"(embedding/.*?)/params\.npz$", identifier
        )
        sae_id = m.group(1) if m else Path(identifier).parent.name
    sae = SparseAutoencoder.from_weights(
        arrays["W_enc"],
        arrays["W_dec"],
        arrays["b_enc"],
        arrays["b_dec"],
        arrays["threshold"],
        architecture="jumprelu",
        apply_b_dec_to_input=False,
        normalize_activations="none",
        center_input=False,
        hook_name=hook_name,
        hook_layer=layer,
        hook_site=site,
        model_name=model_name or inf_model,
        release=release,
        sae_id=sae_id,
        source_format="gemma_scope",
        device="cpu",
    )
    return sae.to(device) if device != "cpu" else sae


# ---------------------------------------------------------------------------
# Registry of well-known releases
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PretrainedRelease:
    """A known family of pretrained SAEs on the Hugging Face Hub.

    ``sae_id_pattern`` is a full-match regex over valid ids; the file(s) of an
    id live at ``path_template.format(sae_id=...)`` inside ``repo_id``.
    """

    name: str
    repo_id: str
    format: str  # "saelens" | "gemma_scope"
    model: str  # Hugging Face id of the model the SAEs read
    sae_id_pattern: str
    path_template: str
    example_sae_id: str
    description: str = ""

    def path_for(self, sae_id: str) -> str:
        if not re.fullmatch(self.sae_id_pattern, sae_id):
            raise ValueError(
                f"{sae_id!r} is not a valid sae_id for release {self.name!r} "
                f"(e.g. {self.example_sae_id!r})"
            )
        return self.path_template.format(sae_id=sae_id)


PRETRAINED_RELEASES: dict[str, PretrainedRelease] = {
    r.name: r
    for r in (
        PretrainedRelease(
            name="gpt2-small-res-jb",
            repo_id="jbloom/GPT2-Small-SAEs-Reformatted",
            format="saelens",
            model="gpt2",
            sae_id_pattern=r"blocks\.([0-9]|1[01])\.hook_resid_pre|blocks\.11\.hook_resid_post",
            path_template="{sae_id}",
            example_sae_id="blocks.8.hook_resid_pre",
            description=(
                "Joseph Bloom's GPT-2 small residual-stream ReLU SAEs, 24576 features "
                "per layer (about 151 MB of float32 weights each)."
            ),
        ),
        PretrainedRelease(
            name="gemma-scope-2b-pt-res",
            repo_id="google/gemma-scope-2b-pt-res",
            format="gemma_scope",
            model="google/gemma-2-2b",
            sae_id_pattern=r"layer_([0-9]|1[0-9]|2[0-5])/width_\d+[km]/average_l0_\d+",
            path_template="{sae_id}/params.npz",
            example_sae_id="layer_20/width_16k/average_l0_71",
            description=(
                "Gemma Scope JumpReLU SAEs on the Gemma-2-2B residual stream "
                "(blocks.L.hook_resid_post); width_16k files are about 302 MB."
            ),
        ),
    )
}


def list_pretrained() -> dict[str, dict[str, str]]:
    """Summary of :data:`PRETRAINED_RELEASES` (no network access)."""
    return {
        name: {
            "repo_id": r.repo_id,
            "format": r.format,
            "model": r.model,
            "example_sae_id": r.example_sae_id,
            "description": r.description,
        }
        for name, r in PRETRAINED_RELEASES.items()
    }


def _resolve(release: str, sae_id: str) -> tuple[str, str, str, PretrainedRelease | None]:
    """``(repo_id, path_in_repo, format, registry entry)`` for a release + id."""
    entry = PRETRAINED_RELEASES.get(release)
    if entry is not None:
        return entry.repo_id, entry.path_for(sae_id), entry.format, entry
    if _looks_like_repo_id(release):
        fmt = "gemma_scope" if sae_id.endswith(".npz") else "saelens"
        return release, sae_id.strip("/"), fmt, None
    raise KeyError(
        f"unknown SAE release {release!r}; known releases: {sorted(PRETRAINED_RELEASES)} "
        "(or pass a Hub repo id such as 'org/repo')"
    )


def from_pretrained(
    release: str,
    sae_id: str,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
    device: str = "cpu",
    **loader_kwargs: Any,
) -> SparseAutoencoder:
    """Download (or read from the local cache) and load a pretrained SAE.

    Parameters
    ----------
    release:
        A key of :data:`PRETRAINED_RELEASES` (e.g. ``"gpt2-small-res-jb"``), or a
        Hub repo id, in which case *sae_id* is the SAELens folder or the path of
        a Gemma Scope ``params.npz`` inside it.
    sae_id:
        E.g. ``"blocks.8.hook_resid_pre"`` or ``"layer_20/width_16k/average_l0_71"``.
    loader_kwargs:
        Passed to :func:`load_saelens` / :func:`load_gemma_scope`.

    Downloads happen only here (``hf_hub_download``, cached under the
    Hugging Face cache, ``HF_HUB_OFFLINE`` honoured).  Check sizes first with
    :func:`describe_pretrained`.
    """
    repo_id, path, fmt, entry = _resolve(release, sae_id)
    common: dict[str, Any] = {
        "revision": revision,
        "local_files_only": local_files_only,
        "device": device,
    }
    if fmt == "gemma_scope":
        sae = load_gemma_scope(
            path, repo_id=repo_id, release=release, sae_id=sae_id, **common, **loader_kwargs
        )
    else:
        sae = load_saelens(repo_id, path, release=release, sae_id=sae_id, **common, **loader_kwargs)
    if entry is not None:
        sae.config.extra.setdefault("hf_model", entry.model)
    return sae


def describe_pretrained(
    release: str, sae_id: str, *, revision: str | None = None
) -> dict[str, Any]:
    """File sizes (and, for SAELens, ``cfg.json`` + tensor shapes) without downloading weights.

    Uses ``HfApi.get_paths_info`` for sizes and, for SAELens releases, fetches
    the small ``cfg.json`` and parses the weights' safetensors *header* through
    HTTP range requests (``HfApi.parse_safetensors_file_metadata``), then
    validates names / shapes / orientation with :func:`validate_saelens_shapes`.
    """
    from huggingface_hub import HfApi

    repo_id, path, fmt, _ = _resolve(release, sae_id)
    api = HfApi()
    if fmt == "gemma_scope":
        files = [path]
    else:
        files = [f"{path}/{SAELENS_CFG}", f"{path}/{SAELENS_WEIGHTS}"]
    infos = api.get_paths_info(repo_id, files, revision=revision)
    sizes = {i.path: getattr(i, "size", None) for i in infos}
    missing = [f for f in files if f not in sizes]
    if missing:
        raise FileNotFoundError(f"{repo_id} has no {missing}")
    out: dict[str, Any] = {"repo_id": repo_id, "format": fmt, "files": sizes}
    if fmt == "saelens":
        cfg_path = _hub_download(repo_id, files[0], revision, False)
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        meta = api.parse_safetensors_file_metadata(repo_id, files[1], revision=revision)
        shapes = {name: tuple(int(d) for d in t.shape) for name, t in meta.tensors.items()}
        kwargs = saelens_config_kwargs(cfg)
        validate_saelens_shapes(kwargs, shapes)
        out["cfg"] = cfg
        out["tensors"] = {name: (t.dtype, shapes[name]) for name, t in meta.tensors.items()}
        out["config"] = {
            k: kwargs[k]
            for k in (
                "input_dim",
                "dict_size",
                "architecture",
                "apply_b_dec_to_input",
                "normalize_activations",
                "center_input",
                "hook_name",
                "hook_layer",
                "hook_site",
                "model_name",
            )
        }
    return out
