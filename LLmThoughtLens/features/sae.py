"""Sparse autoencoders — the trainable TopK SAE plus pretrained ReLU / JumpReLU / TopK SAEs.

Architectures (``SAEConfig.architecture``)
-----------------------------------------
Every architecture shares one affine encoder / decoder; only the activation
function differs.  With ``x_in`` the pre-processed input (see below)::

    pre  = x_in · W_encᵀ + b_enc                     # (…, d_sae)
    "topk"     z = TopK(ReLU(pre), k)                 # trainable here (Anthropic CLT-style)
    "relu"     z = ReLU(pre)                          # SAELens "standard"
    "jumprelu" z = ReLU(pre) · 1[pre > threshold]     # Gemma Scope / SAELens "jumprelu"
    x̂_sae = z · W_decᵀ + b_dec

Weights are stored in the *legacy* orientation of this module: ``W_enc`` is
``(d_sae, d_in)`` and ``W_dec`` is ``(d_in, d_sae)`` (decoder *columns* are
feature directions).  SAELens / Gemma Scope publish the transposes
(``W_enc (d_in, d_sae)``, ``W_dec (d_sae, d_in)``); :meth:`SparseAutoencoder.from_weights`
takes that published orientation and asserts every shape.

Input pre-processing (applied by :meth:`encode`, in this order)
--------------------------------------------------------------
1. ``center_input`` — subtract the per-token mean over ``d_in``.  TransformerLens
   loads LayerNorm models (GPT-2, Pythia, …) with ``center_writing_weights=True``,
   which makes its residual stream equal to the HuggingFace residual minus its
   per-token mean; SAEs trained on such activations need this to read HF
   activations.  The mean is added back by :meth:`reconstruct`.
2. ``normalize_activations`` (SAELens semantics):

   * ``"none"`` — identity.
   * ``"expected_average_only_in"`` — multiply by the dataset constant
     ``norm_scaling_factor`` (``sqrt(d_in) / E‖x‖`` over the training data);
     the decoder output is divided by it again.  The factor is required
     (``1.0`` when it is already folded into the weights).
   * ``"constant_norm_rescale"`` — per token, ``x · sqrt(d_in) / ‖x‖``; undone
     per token by :meth:`reconstruct`.
   * ``"layer_norm"`` — per token, ``(x − μ) / (σ + 1e-5)`` (σ unbiased, as
     ``torch.std``); undone as ``x̂ · σ + μ``.

3. ``apply_b_dec_to_input`` — subtract ``b_dec`` (SAELens default ``True``;
   Gemma Scope ``False``).

Hook-point metadata
-------------------
``hook_layer`` / ``hook_site`` name the activation the SAE reads, in the
TransformerLens / SAELens convention, which is also the convention of
:class:`~LLmThoughtLens.models.HookedModel` sites: ``blocks.L.hook_resid_pre``
is the input to block ``L`` and ``blocks.L.hook_resid_post`` its output (the
true residual, no final norm).  :func:`parse_hook_name` /
:func:`hook_name_for` convert between the two spellings.  Old SAEs saved by
the trainer carry no hook metadata; :class:`FeatureExtractor` then treats the
layer the caller passes as a ``resid_post`` index (the historical behaviour).

The class takes NumPy inputs externally and converts to torch internally, so
callers never need torch at use sites outside training.  For gradient-based
attribution the torch API is differentiable end to end:
:meth:`encode_torch` (codes), :meth:`reconstruct_torch` (reconstruction in
model units, every pre-processing step undone) and :meth:`output_scale_torch`
(``∂ reconstruct / ∂ z_i = output_scale · W_dec[:, i]``).  ``W_dec`` is
``(d_in, d_sae)``, so ``W_dec[:, i]`` is feature ``i``'s decoder direction.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np

#: Encoder activation functions this module implements.
SAE_ARCHITECTURES: tuple[str, ...] = ("topk", "relu", "jumprelu")
#: Hook sites an SAE can be attached to (TransformerLens ``hook_*`` names).
SAE_SITES: tuple[str, ...] = ("resid_pre", "resid_post", "mlp_out", "attn_out")
#: SAELens ``normalize_activations`` modes this module implements.
SAE_NORMALIZATIONS: tuple[str, ...] = (
    "none",
    "expected_average_only_in",
    "constant_norm_rescale",
    "layer_norm",
)
#: Architectures :meth:`SparseAutoencoder.fit` can train.
TRAINABLE_ARCHITECTURES: tuple[str, ...] = ("topk", "relu")

_HOOK_RE = re.compile(r"^blocks\.(\d+)\.hook_(resid_pre|resid_post|mlp_out|attn_out)$")
_LAYER_NORM_EPS = 1e-5


def parse_hook_name(hook_name: str) -> tuple[int, str] | None:
    """``"blocks.8.hook_resid_pre"`` -> ``(8, "resid_pre")``; ``None`` for other hook points.

    Only the sites in :data:`SAE_SITES` are recognised; e.g. ``blocks.L.attn.hook_z``
    or ``blocks.L.hook_resid_mid`` return ``None``.
    """
    m = _HOOK_RE.match(str(hook_name).strip())
    if m is None:
        return None
    return int(m.group(1)), m.group(2)


def hook_name_for(layer: int, site: str) -> str:
    """``(8, "resid_pre")`` -> ``"blocks.8.hook_resid_pre"``."""
    if site not in SAE_SITES:
        raise ValueError(f"unknown SAE site {site!r}; expected one of {SAE_SITES}")
    if int(layer) < 0:
        raise ValueError(f"hook layer must be >= 0, got {layer!r}")
    return f"blocks.{int(layer)}.hook_{site}"


def estimate_norm_scaling_factor(activations: np.ndarray) -> float:
    """SAELens' ``expected_average_only_in`` factor: ``sqrt(d_in) / mean ‖x‖``.

    Parameters
    ----------
    activations:
        ``(N, d_in)`` sample of the activations the SAE reads (after any
        ``center_input`` centring).
    """
    x = np.asarray(activations, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError(f"expected a non-empty (N, d_in) array, got shape {x.shape}")
    mean_norm = float(np.linalg.norm(x, axis=-1).mean())
    if mean_norm <= 0.0:
        raise ValueError("activations have zero mean norm; cannot estimate a scaling factor")
    return math.sqrt(x.shape[1]) / mean_norm


@dataclass
class SAEConfig:
    """Hyperparameters + provenance of a sparse autoencoder.

    The first block of fields are the TopK trainer's hyperparameters (kept
    verbatim so configs saved by older versions still load).  The remaining
    fields describe the encoder and where the SAE reads its input:

    Attributes
    ----------
    architecture:
        One of :data:`SAE_ARCHITECTURES`.
    apply_b_dec_to_input:
        Subtract ``b_dec`` from the input before encoding.
    normalize_activations, norm_scaling_factor:
        See the module docstring.  ``norm_scaling_factor`` is required for
        ``"expected_average_only_in"`` and ignored otherwise.
    center_input:
        Subtract the per-token mean before encoding (TransformerLens
        ``center_writing_weights`` activations; see the module docstring).
    hook_name, hook_layer, hook_site:
        The activation the SAE was trained on (e.g. ``"blocks.8.hook_resid_pre"``,
        ``8``, ``"resid_pre"``).  ``hook_name`` may name an unsupported hook
        point, in which case ``hook_site`` stays ``None``.
    model_name, release, sae_id, source_format:
        Provenance: the model it was trained on, the pretrained release /
        id it was loaded from, and the file format (``"native"``,
        ``"saelens"``, ``"gemma_scope"``).
    extra:
        Format-specific metadata (e.g. SAELens ``context_size``,
        ``prepend_bos``).  JSON-friendly values only.
    """

    input_dim: int = 768
    dict_size: int = 3072
    k: int = 64
    lr: float = 2e-4
    batch_size: int = 2048
    n_steps: int = 5_000
    l1_coeff: float = 8e-4
    seed: int = 0
    dead_window: int = 1_000
    log_every: int = 200
    device: str = "auto"
    # --- encoder semantics -------------------------------------------------
    architecture: str = "topk"
    apply_b_dec_to_input: bool = True
    normalize_activations: str = "none"
    norm_scaling_factor: float | None = None
    center_input: bool = False
    # --- hook point + provenance -----------------------------------------
    hook_name: str | None = None
    hook_layer: int | None = None
    hook_site: str | None = None
    model_name: str | None = None
    release: str | None = None
    sae_id: str | None = None
    source_format: str = "native"
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.architecture not in SAE_ARCHITECTURES:
            raise ValueError(
                f"unsupported SAE architecture {self.architecture!r}; "
                f"supported: {SAE_ARCHITECTURES}"
            )
        if self.normalize_activations not in SAE_NORMALIZATIONS:
            raise ValueError(
                f"unsupported normalize_activations {self.normalize_activations!r}; "
                f"supported: {SAE_NORMALIZATIONS}"
            )
        if self.normalize_activations == "expected_average_only_in":
            if self.norm_scaling_factor is None:
                raise ValueError(
                    "normalize_activations='expected_average_only_in' needs "
                    "norm_scaling_factor (sqrt(d_in) / mean ||x|| over the training data; "
                    "1.0 if it is already folded into the weights). Estimate it with "
                    "estimate_norm_scaling_factor(activations)."
                )
            if not float(self.norm_scaling_factor) > 0.0:
                raise ValueError(
                    f"norm_scaling_factor must be > 0, got {self.norm_scaling_factor!r}"
                )
        if self.norm_scaling_factor is not None:
            self.norm_scaling_factor = float(self.norm_scaling_factor)
        if self.hook_site is not None and self.hook_site not in SAE_SITES:
            raise ValueError(f"unknown hook_site {self.hook_site!r}; expected one of {SAE_SITES}")
        if self.hook_layer is not None:
            self.hook_layer = int(self.hook_layer)
            if self.hook_layer < 0:
                raise ValueError(f"hook_layer must be >= 0, got {self.hook_layer}")
        if self.hook_name is not None:
            parsed = parse_hook_name(self.hook_name)
            if parsed is not None:
                layer, site = parsed
                if self.hook_layer is None:
                    self.hook_layer = layer
                if self.hook_site is None:
                    self.hook_site = site
                if (self.hook_layer, self.hook_site) != (layer, site):
                    raise ValueError(
                        f"hook_name {self.hook_name!r} disagrees with hook_layer="
                        f"{self.hook_layer!r} / hook_site={self.hook_site!r}"
                    )
        elif self.hook_layer is not None and self.hook_site is not None:
            self.hook_name = hook_name_for(self.hook_layer, self.hook_site)
        if self.architecture == "topk" and int(self.k) < 1:
            raise ValueError(f"k must be >= 1 for a TopK SAE, got {self.k!r}")

    # Aliases in the SAELens vocabulary.
    @property
    def d_in(self) -> int:
        return int(self.input_dim)

    @property
    def d_sae(self) -> int:
        return int(self.dict_size)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SAEConfig:
        """Build from a saved dict, ignoring keys this version does not know."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


class SparseAutoencoder:
    """Sparse autoencoder: trainable TopK, or a loaded ReLU / JumpReLU / TopK SAE.

    Parameters
    ----------
    config:
        :class:`SAEConfig` instance.  Use the keyword form to override
        only the hyperparameters you care about::

            sae = SparseAutoencoder(SAEConfig(input_dim=64, dict_size=256, k=16))

        Pretrained SAEs come from :meth:`from_weights` or the loaders in
        :mod:`LLmThoughtLens.features.sae_loaders`.
    """

    def __init__(self, config: SAEConfig | None = None) -> None:
        _require_torch()
        self._setup(config or SAEConfig())
        self._init_weights()

    def _setup(self, config: SAEConfig) -> None:
        self.config = config
        self._labels: dict[int, str] = {}
        self._trained: bool = False
        self._steps_run: int = 0
        self.threshold: Any = None

    # ------------------------------------------------------------------
    # Construction from published weights
    # ------------------------------------------------------------------

    @classmethod
    def from_weights(
        cls,
        W_enc: Any,
        W_dec: Any,
        b_enc: Any = None,
        b_dec: Any = None,
        threshold: Any = None,
        *,
        config: SAEConfig | None = None,
        **config_kwargs: Any,
    ) -> SparseAutoencoder:
        """Build an SAE from weights in the published (SAELens / Gemma Scope) orientation.

        Parameters
        ----------
        W_enc:
            ``(d_in, d_sae)`` encoder (NumPy array or torch tensor).
        W_dec:
            ``(d_sae, d_in)`` decoder; row ``i`` is feature ``i``'s direction.
        b_enc, b_dec:
            ``(d_sae,)`` / ``(d_in,)`` biases (zeros when ``None``).
        threshold:
            ``(d_sae,)`` JumpReLU thresholds; required iff the architecture is
            ``"jumprelu"``.
        config:
            Base config; ``input_dim`` / ``dict_size`` are taken from the
            weight shapes (a disagreeing config raises).
        config_kwargs:
            Field overrides applied to *config* (e.g. ``architecture="relu"``).
        """
        _require_torch()
        import torch

        def _t(name: str, x: Any) -> Any:
            t = torch.as_tensor(x.detach() if hasattr(x, "detach") else np.asarray(x))
            if not torch.is_floating_point(t):
                raise TypeError(f"{name} must be a floating-point array, got {t.dtype}")
            return t.to(torch.float32).cpu()

        enc = _t("W_enc", W_enc)
        dec = _t("W_dec", W_dec)
        if enc.ndim != 2:
            raise ValueError(f"W_enc must be 2-D (d_in, d_sae), got shape {tuple(enc.shape)}")
        d_in, d_sae = int(enc.shape[0]), int(enc.shape[1])
        if tuple(dec.shape) != (d_sae, d_in):
            raise ValueError(
                f"W_dec must be (d_sae, d_in) = {(d_sae, d_in)} to match W_enc (d_in, d_sae) = "
                f"{(d_in, d_sae)}, got {tuple(dec.shape)}"
            )
        benc = torch.zeros(d_sae) if b_enc is None else _t("b_enc", b_enc)
        bdec = torch.zeros(d_in) if b_dec is None else _t("b_dec", b_dec)
        if tuple(benc.shape) != (d_sae,):
            raise ValueError(f"b_enc must be ({d_sae},), got {tuple(benc.shape)}")
        if tuple(bdec.shape) != (d_in,):
            raise ValueError(f"b_dec must be ({d_in},), got {tuple(bdec.shape)}")

        unknown = sorted(set(config_kwargs) - {f.name for f in fields(SAEConfig)})
        if unknown:
            raise TypeError(f"unknown SAEConfig field(s) {unknown}")
        base = asdict(config) if config is not None else {}
        for key, want in (("input_dim", d_in), ("dict_size", d_sae)):
            if int(base.get(key, want)) != want:
                raise ValueError(f"config.{key}={base[key]} but the weights give {want}")
            if key in config_kwargs and int(config_kwargs[key]) != want:
                raise ValueError(f"{key}={config_kwargs[key]} but the weights give {want}")
            base[key] = want
        base.update({k: v for k, v in config_kwargs.items() if k not in ("input_dim", "dict_size")})
        cfg = SAEConfig.from_dict(base)

        thr = None
        if cfg.architecture == "jumprelu":
            if threshold is None:
                raise ValueError("a JumpReLU SAE needs a (d_sae,) threshold")
            thr = _t("threshold", threshold)
            if tuple(thr.shape) != (d_sae,):
                raise ValueError(f"threshold must be ({d_sae},), got {tuple(thr.shape)}")
        elif threshold is not None:
            raise ValueError(f"threshold given for a {cfg.architecture!r} SAE (JumpReLU only)")

        sae = cls.__new__(cls)
        sae._setup(cfg)
        sae.W_enc = enc.T.contiguous()  # (d_sae, d_in)
        sae.W_dec = dec.T.contiguous()  # (d_in, d_sae)
        sae.b_enc = benc.contiguous()
        sae.b_dec = bdec.contiguous()
        sae.threshold = thr
        sae._steps_since_active = torch.zeros(d_sae, dtype=torch.int64)
        sae._trained = True
        return sae

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        import torch

        cfg = self.config
        gen = torch.Generator().manual_seed(cfg.seed)
        scale = 1.0 / math.sqrt(cfg.input_dim)

        self.W_enc = (
            torch.randn((cfg.dict_size, cfg.input_dim), generator=gen, dtype=torch.float32) * scale
        )
        self.b_enc = torch.zeros(cfg.dict_size, dtype=torch.float32)
        # Initialise decoder as transpose of encoder; renormalise columns.
        self.W_dec = self.W_enc.T.clone().contiguous()
        self._renormalise_decoder()
        self.b_dec = torch.zeros(cfg.input_dim, dtype=torch.float32)
        if cfg.architecture == "jumprelu":
            self.threshold = torch.zeros(cfg.dict_size, dtype=torch.float32)

        # Track how many steps since each feature last fired (for resampling).
        self._steps_since_active = torch.zeros(cfg.dict_size, dtype=torch.int64)

    def _renormalise_decoder(self) -> None:
        norms = self.W_dec.norm(dim=0, keepdim=True).clamp_min(1e-9)
        self.W_dec.div_(norms)

    # ------------------------------------------------------------------
    # Device helpers
    # ------------------------------------------------------------------

    def _resolve_device(self) -> Any:
        import torch

        req = self.config.device
        if req != "auto":
            return torch.device(req)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _to(self, device: Any) -> None:
        self.W_enc = self.W_enc.to(device)
        self.b_enc = self.b_enc.to(device)
        self.W_dec = self.W_dec.to(device)
        self.b_dec = self.b_dec.to(device)
        if self.threshold is not None:
            self.threshold = self.threshold.to(device)

    def to(self, device: Any) -> SparseAutoencoder:
        """Move every weight to *device* (e.g. the model's, for use inside hooks)."""
        import torch

        self._to(torch.device(device) if isinstance(device, str) else device)
        return self

    @property
    def device(self) -> Any:
        return self.W_enc.device

    # ------------------------------------------------------------------
    # Encode / decode (numpy in/out for downstream extractor)
    # ------------------------------------------------------------------

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode model activations into sparse codes.

        Parameters
        ----------
        x:
            ``(d_in,)`` or ``(..., d_in)`` activations in model units (the
            pre-processing in the module docstring is applied here).

        Returns
        -------
        np.ndarray
            ``(d_sae,)`` or ``(..., d_sae)`` float32 codes.
        """
        return self.encode_with_output_scale(x)[0]

    def encode_with_output_scale(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Codes plus the per-row factor that maps SAE-space vectors back to model units.

        ``‖z_i · W_dec[:, i]‖ · scale`` is the norm of feature ``i``'s
        contribution to the reconstruction in model units.  ``scale`` is 1
        unless ``normalize_activations`` rescales the input.

        Returns
        -------
        tuple
            ``(codes (..., d_sae), scale (...,))``.
        """
        import torch

        arr = np.asarray(x)
        lead = arr.shape[:-1]
        xt = self._as_rows(arr)
        with torch.no_grad():
            x_in, ctx = self._preprocess(xt)
            z = self._activate(x_in @ self.W_enc.T + self.b_enc)
            scale = ctx["out_scale"]
        codes = z.cpu().numpy().reshape(*lead, self.config.dict_size)
        scale_np = scale.reshape(-1).cpu().numpy() if scale is not None else None
        if scale_np is None or scale_np.size == 1:
            value = 1.0 if scale_np is None else float(scale_np[0])
            scale_np = np.full(int(np.prod(lead, dtype=np.int64)), value, dtype=np.float32)
        return codes, scale_np.reshape(lead).astype(np.float32)

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode codes: ``z · W_decᵀ + b_dec``, with any constant input scaling undone.

        Per-token pre-processing (``center_input``, ``"constant_norm_rescale"``,
        ``"layer_norm"``) cannot be undone from the codes alone; use
        :meth:`reconstruct` for a reconstruction in model units.
        """
        import torch

        zarr = np.asarray(z)
        lead = zarr.shape[:-1]
        if zarr.shape[-1] != self.config.dict_size:
            raise ValueError(
                f"codes have last dim {zarr.shape[-1]}, expected d_sae={self.config.dict_size}"
            )
        zt = torch.as_tensor(
            zarr.reshape(-1, zarr.shape[-1]), dtype=torch.float32, device=self.W_enc.device
        )
        with torch.no_grad():
            xh = self.decode_torch(zt)
        return xh.cpu().numpy().reshape(*lead, self.config.input_dim)

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        """Full round trip in model units: pre-process, encode, decode, undo pre-processing."""
        import torch

        arr = np.asarray(x)
        xt = self._as_rows(arr)
        with torch.no_grad():
            xh = self.reconstruct_torch(xt)
        return xh.cpu().numpy().reshape(arr.shape)

    def encode_torch(self, x: Any) -> Any:
        """Differentiable torch encode of model activations ``(..., d_in)`` -> ``(..., d_sae)``.

        *x* is cast to the SAE's device / dtype (see :meth:`to`); gradients
        flow back to *x* through the active features.
        """
        x_in, _ = self._preprocess(self._cast(x))
        return self._activate(x_in @ self.W_enc.T + self.b_enc)

    def output_scale_torch(self, x: Any) -> Any:
        """Per-token factor ``(..., 1)`` mapping SAE-space decoder outputs to model units.

        With ``s = output_scale_torch(x)``, feature ``i``'s contribution to
        :meth:`reconstruct_torch` ``(x)`` is ``z_i · s · W_dec[:, i]``, so
        ``∂ reconstruct / ∂ z_i = s · W_dec[:, i]`` (the per-token normalisation
        statistics depend on *x*, not on the codes).  ``s`` is ``1`` without
        input normalisation, ``1 / norm_scaling_factor`` for
        ``"expected_average_only_in"``, ``‖x‖ / sqrt(d_in)`` for
        ``"constant_norm_rescale"`` and the token's standard deviation for
        ``"layer_norm"``.  ``center_input`` does not scale.
        """
        xc = self._cast(x)
        _, ctx = self._preprocess(xc)
        scale = ctx["out_scale"]
        ones = xc.new_ones((*xc.shape[:-1], 1))
        return ones if scale is None else ones * scale

    def decode_torch(self, z: Any) -> Any:
        """Torch-native decode (same semantics as :meth:`decode`)."""
        out = z @ self.W_dec.T + self.b_dec
        if self.config.normalize_activations == "expected_average_only_in":
            out = out / float(self.config.norm_scaling_factor or 1.0)
        return out

    def reconstruct_torch(self, x: Any) -> Any:
        """Differentiable torch version of :meth:`reconstruct`."""
        x = self._cast(x)
        x_in, ctx = self._preprocess(x)
        z = self._activate(x_in @ self.W_enc.T + self.b_enc)
        out = z @ self.W_dec.T + self.b_dec
        mode = self.config.normalize_activations
        if mode == "layer_norm":
            out = out * ctx["ln_std"] + ctx["ln_mu"]
        elif ctx["out_scale"] is not None:
            out = out * ctx["out_scale"]
        if ctx["mean"] is not None:
            out = out + ctx["mean"]
        return out

    # ------------------------------------------------------------------
    # Internal forward (torch tensors)
    # ------------------------------------------------------------------

    def _as_rows(self, arr: np.ndarray) -> Any:
        import torch

        if arr.ndim == 0 or arr.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"activations have shape {arr.shape}; expected (..., {self.config.input_dim}) "
                f"for an SAE with d_in={self.config.input_dim}"
            )
        rows = arr.reshape(-1, arr.shape[-1])
        return torch.as_tensor(rows, dtype=torch.float32, device=self.W_enc.device)

    def _cast(self, x: Any) -> Any:
        if x.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"activations have last dim {x.shape[-1]}, expected d_in={self.config.input_dim}"
            )
        return x.to(device=self.W_enc.device, dtype=self.W_enc.dtype)

    def _preprocess(self, x: Any) -> tuple[Any, dict[str, Any]]:
        """Apply centring, normalisation and the b_dec shift; return what undoes them."""
        cfg = self.config
        ctx: dict[str, Any] = {"mean": None, "out_scale": None}
        if cfg.center_input:
            mean = x.mean(dim=-1, keepdim=True)
            x = x - mean
            ctx["mean"] = mean
        mode = cfg.normalize_activations
        if mode == "expected_average_only_in":
            factor = float(cfg.norm_scaling_factor or 1.0)
            x = x * factor
            ctx["out_scale"] = x.new_full((1,), 1.0 / factor)
        elif mode == "constant_norm_rescale":
            coeff = math.sqrt(cfg.input_dim) / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            x = x * coeff
            ctx["out_scale"] = 1.0 / coeff
        elif mode == "layer_norm":
            mu = x.mean(dim=-1, keepdim=True)
            xc = x - mu
            std = xc.std(dim=-1, keepdim=True)
            x = xc / (std + _LAYER_NORM_EPS)
            ctx["ln_mu"], ctx["ln_std"] = mu, std
            ctx["out_scale"] = std
        if cfg.apply_b_dec_to_input:
            x = x - self.b_dec
        return x, ctx

    def _activate(self, pre: Any) -> Any:
        import torch

        arch = self.config.architecture
        if arch == "topk":
            return _topk_tensor(torch.relu(pre), self.config.k)
        if arch == "relu":
            return torch.relu(pre)
        # jumprelu
        return torch.relu(pre) * (pre > self.threshold).to(pre.dtype)

    def _encode_torch(self, x: Any) -> Any:
        """Encode already-cast tensors (training loop / resampling)."""
        x_in, _ = self._preprocess(x)
        return self._activate(x_in @ self.W_enc.T + self.b_enc)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def fit(
        self,
        activations: np.ndarray,
        verbose: bool = False,
    ) -> SparseAutoencoder:
        """Train the SAE on cached activations using real PyTorch autograd.

        Only :data:`TRAINABLE_ARCHITECTURES` without input normalisation /
        centring can be trained here.

        Parameters
        ----------
        activations:
            ``(N, input_dim)`` — token activations collected from a corpus.
        verbose:
            Print loss + sparsity stats every ``config.log_every`` steps.
        """
        import torch

        cfg = self.config
        if cfg.architecture not in TRAINABLE_ARCHITECTURES:
            raise ValueError(
                f"training a {cfg.architecture!r} SAE is not supported "
                f"(trainable: {TRAINABLE_ARCHITECTURES})"
            )
        if cfg.normalize_activations != "none" or cfg.center_input:
            raise ValueError("fit() does not support input normalisation or centring")
        device = self._resolve_device()
        self._to(device)

        x_all = torch.as_tensor(activations, dtype=torch.float32, device=device)
        n = x_all.shape[0]
        if n == 0:
            raise ValueError("activations array is empty")
        if x_all.shape[1] != cfg.input_dim:
            raise ValueError(
                f"activations have d={x_all.shape[1]}, expected input_dim={cfg.input_dim}"
            )

        # b_dec is initialised to the mean of the data (CLT recommendation).
        with torch.no_grad():
            self.b_dec.copy_(x_all.mean(dim=0))

        # Optimiser holds W_enc, b_enc, W_dec, b_dec as leaf tensors with grad.
        for t in (self.W_enc, self.b_enc, self.W_dec, self.b_dec):
            t.requires_grad_(True)
        opt = torch.optim.Adam(
            [self.W_enc, self.b_enc, self.W_dec, self.b_dec],
            lr=cfg.lr,
            betas=(0.9, 0.999),
        )

        steps_since_active = self._steps_since_active.to(device)

        gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed + 1)

        for step in range(1, cfg.n_steps + 1):
            idx = torch.randint(0, n, (cfg.batch_size,), generator=gen)
            x = x_all[idx]

            z = self._encode_torch(x)
            x_hat = z @ self.W_dec.T + self.b_dec
            mse = (x - x_hat).pow(2).mean()
            l1 = z.abs().mean() * cfg.l1_coeff
            loss = mse + l1

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            with torch.no_grad():
                self._renormalise_decoder()

                active = (z != 0).any(dim=0)
                steps_since_active = torch.where(
                    active, torch.zeros_like(steps_since_active), steps_since_active + 1
                )

                if cfg.dead_window > 0 and step % cfg.dead_window == 0:
                    self._resample_dead_features(steps_since_active, x_all, rng)
                    steps_since_active.zero_()

            if verbose and step % max(1, cfg.log_every) == 0:
                with torch.no_grad():
                    l0 = (z != 0).float().sum(dim=-1).mean().item()
                    print(
                        f"  step {step:>6}/{cfg.n_steps} "
                        f"mse={mse.item():.4f} l1={l1.item():.4f} "
                        f"l0={l0:.1f}"
                    )

            self._steps_run += 1

        # Detach parameters once training completes.
        for t in (self.W_enc, self.b_enc, self.W_dec, self.b_dec):
            t.requires_grad_(False)
        self._steps_since_active = steps_since_active.detach().cpu()
        self._trained = True
        return self

    def _resample_dead_features(
        self,
        steps_since_active: Any,
        x_all: Any,
        rng: np.random.Generator,
    ) -> None:
        import torch

        dead = (steps_since_active >= self.config.dead_window).nonzero(as_tuple=True)[0]
        if dead.numel() == 0:
            return
        # Sample residuals — examples where current reconstruction is worst.
        n_sample = min(2048, x_all.shape[0])
        idx = torch.randint(0, x_all.shape[0], (n_sample,), device=x_all.device)
        sample = x_all[idx]
        x_hat = self._encode_torch(sample) @ self.W_dec.T + self.b_dec
        residual = (sample - x_hat).detach()
        norms = residual.norm(dim=-1)
        order = torch.argsort(-norms)
        chosen = residual[order[: dead.numel()]]
        # Encoder rows pointing toward worst residuals (with unit norm).
        chosen_unit = chosen / chosen.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        self.W_enc.data[dead] = chosen_unit
        self.W_dec.data[:, dead] = chosen_unit.T
        # Reset their bias so they re-enter via ReLU again.
        self.b_enc.data[dead] = 0.0

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def reconstruction_loss(self, x: np.ndarray) -> float:
        """Return ``MSE + λ·||z||₁`` for the given batch."""
        z = self.encode(x)
        x_hat = self.reconstruct(x)
        mse = float(np.mean((x - x_hat) ** 2))
        l1 = float(self.config.l1_coeff * np.mean(np.abs(z)))
        return mse + l1

    def sparsity_stats(self, activations: np.ndarray) -> dict[str, float]:
        """Return L0 sparsity stats, dead-feature fraction and reconstruction quality.

        ``mse`` / ``explained_variance`` compare :meth:`reconstruct` with the
        input in model units; ``explained_variance`` is ``1 − MSE / Var(x)``
        clipped to ``[0, 1]``.
        """
        z = self.encode(activations)
        l0 = (z != 0).sum(axis=-1).astype(float)
        feature_active = (z != 0).any(axis=0)
        dead_fraction = float(1.0 - feature_active.mean())
        mse = float(np.mean((activations - self.reconstruct(activations)) ** 2))
        explained = 1.0 - mse / float(np.var(activations) + 1e-12)
        return {
            "l0_mean": float(l0.mean()),
            "l0_std": float(l0.std()),
            "dead_fraction": dead_fraction,
            "mse": mse,
            "explained_variance": float(max(0.0, min(1.0, explained))),
        }

    def feature_density(self, activations: np.ndarray, bins: int = 20) -> dict[str, list[float]]:
        """Return density histogram of how often each feature fires across a batch."""
        z = self.encode(activations)
        firing_rate = (z != 0).mean(axis=0)
        hist, edges = np.histogram(firing_rate, bins=bins, range=(0.0, 1.0))
        return {
            "edges": edges.astype(float).tolist(),
            "counts": hist.astype(int).tolist(),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save weights + config (incl. hook metadata) + labels to one torch file.

        The format is the trainer's historical one (``torch.save`` of a dict),
        whatever the file extension; ``threshold`` is added for JumpReLU SAEs.
        """
        import torch

        path = Path(path)
        payload = {
            "config": asdict(self.config),
            "W_enc": self.W_enc.detach().cpu(),
            "b_enc": self.b_enc.detach().cpu(),
            "W_dec": self.W_dec.detach().cpu(),
            "b_dec": self.b_dec.detach().cpu(),
            "threshold": None if self.threshold is None else self.threshold.detach().cpu(),
            "labels": self._labels,
            "trained": self._trained,
            "steps_run": self._steps_run,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> SparseAutoencoder:
        """Load a file written by :meth:`save` (including files from older versions)."""
        _require_torch()
        import torch

        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        cfg = SAEConfig.from_dict(dict(payload["config"]))
        sae = cls.__new__(cls)
        sae._setup(cfg)
        sae.W_enc = payload["W_enc"].to(torch.float32)
        sae.b_enc = payload["b_enc"].to(torch.float32)
        sae.W_dec = payload["W_dec"].to(torch.float32)
        sae.b_dec = payload["b_dec"].to(torch.float32)
        thr = payload.get("threshold")
        sae.threshold = None if thr is None else thr.to(torch.float32)
        if cfg.architecture == "jumprelu" and sae.threshold is None:
            raise ValueError(f"{path}: JumpReLU SAE file has no threshold")
        expect = {
            "W_enc": (cfg.dict_size, cfg.input_dim),
            "W_dec": (cfg.input_dim, cfg.dict_size),
            "b_enc": (cfg.dict_size,),
            "b_dec": (cfg.input_dim,),
        }
        for name, shape in expect.items():
            got = tuple(getattr(sae, name).shape)
            if got != shape:
                raise ValueError(f"{path}: {name} has shape {got}, config implies {shape}")
        sae._steps_since_active = torch.zeros(cfg.dict_size, dtype=torch.int64)
        sae._labels = {int(k): str(v) for k, v in dict(payload.get("labels", {})).items()}
        sae._trained = bool(payload.get("trained", True))
        sae._steps_run = int(payload.get("steps_run", 0))
        return sae

    def save_with_labels(self, path: str | Path, labels: dict[int, str]) -> None:
        """Persist labels alongside weights (merges with existing)."""
        self._labels.update(labels)
        self.save(path)

    def export_labels(self, path: str | Path) -> None:
        """Dump just the labels dict as JSON (handy for sharing)."""
        Path(path).write_text(json.dumps(self._labels, indent=2))

    # ------------------------------------------------------------------
    # Label management
    # ------------------------------------------------------------------

    @property
    def labels(self) -> dict[int, str]:
        return dict(self._labels)

    def set_label(self, feature_id: int, label: str) -> None:
        self._labels[int(feature_id)] = str(label)

    # ------------------------------------------------------------------
    # Feature direction (used by FeatureIntervention for white-box hooks)
    # ------------------------------------------------------------------

    def feature_direction(self, feature_id: int) -> np.ndarray:
        """Return the (unit) decoder column for *feature_id* (taken modulo ``d_sae``).

        Interventions clamp/amplify the activation along this direction in
        the residual stream, rather than mutating a single coordinate.
        """
        fid = int(feature_id) % self.config.dict_size
        col = self.W_dec[:, fid].detach().cpu().numpy()
        norm = float(np.linalg.norm(col))
        if norm < 1e-9:
            return col
        return col / norm

    def decoder_norms(self) -> np.ndarray:
        """``(d_sae,)`` L2 norm of every decoder direction."""
        return self.W_dec.detach().norm(dim=0).cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    @property
    def hook_point(self) -> tuple[int, str] | None:
        """``(hook_layer, hook_site)`` when the SAE's hook point is known and supported."""
        cfg = self.config
        if cfg.hook_layer is None or cfg.hook_site is None:
            return None
        return int(cfg.hook_layer), str(cfg.hook_site)

    def __repr__(self) -> str:
        cfg = self.config
        if cfg.source_format == "native" and cfg.architecture == "topk":
            status = "trained" if self._trained else "untrained"
            return (
                f"SparseAutoencoder({status}, "
                f"input_dim={cfg.input_dim}, "
                f"dict_size={cfg.dict_size}, "
                f"k={cfg.k}, steps={self._steps_run})"
            )
        parts = [cfg.source_format, f"arch={cfg.architecture}"]
        parts += [f"input_dim={cfg.input_dim}", f"dict_size={cfg.dict_size}"]
        if cfg.architecture == "topk":
            parts.append(f"k={cfg.k}")
        if cfg.hook_name:
            parts.append(f"hook={cfg.hook_name}")
        if cfg.sae_id:
            parts.append(f"sae_id={cfg.sae_id}")
        return f"SparseAutoencoder({', '.join(parts)})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_torch() -> None:
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover — gated via extras
        raise ImportError(
            "SparseAutoencoder needs torch. Install with: pip install 'LLmThoughtLens[huggingface]'"
        ) from exc


def _topk_tensor(h: Any, k: int) -> Any:
    """Exact TopK along the last dim — keeps top *k* magnitudes, zeros the rest.

    Operates on a torch tensor without breaking autograd.
    """
    import torch

    if k >= h.shape[-1]:
        return h
    topk_vals, topk_idx = torch.topk(h, k, dim=-1)
    mask = torch.zeros_like(h)
    mask.scatter_(-1, topk_idx, 1.0)
    return h * mask
