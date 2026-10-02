"""FeatureExtractor — real white-box (SAE/L2) and black-box (token-masking) features.

White-box mode (``output.activations is not None``):
- With one or more SAEs attached → SAE-encoded sparse codes as per-token
  features (see *SAE features* below).
- Without SAE → score every ``(layer, token)`` site of the residual stream.
  The default ``scoring="centered"`` measures how far ``a[l, t]`` sits from
  the layer's robust centre (coordinate-wise median over the rankable
  tokens), divided by the layer's median token norm so layers are
  comparable.  ``scoring="l2"`` keeps the legacy raw-L2-norm score.

  Real decoder LMs (GPT-2, Llama, Qwen, Mistral, ...) park a *massive
  activation* ("attention sink") on the first position whose norm is 10-50x
  every other token at nearly every layer.  Under raw L2 scoring that one
  position wins every layer and the trace degenerates into "token 0 at
  layers 1..L".  Such positions are detected automatically (norm greater
  than ``outlier_ratio`` x the median norm of the *other* tokens, in at
  least ``outlier_min_layer_frac`` of the layers) and, by default, left out
  of the ranking and of the per-layer centre / scale statistics.  Excluded
  positions are recorded on every feature's ``meta["excluded_positions"]``
  and on the extractor (``last_excluded_positions`` / ``last_outlier_stats``)
  so reports can say so; ``meta["outlier_positions"]`` lists the detected
  massive-activation positions, so a report can tell "attention-sink
  outlier" apart from a position the caller excluded explicitly.  The
  ``last_*`` record is reset at the start of every :meth:`extract` call.  The raw norm is always kept in
  ``meta["raw_norm"]`` for consumers that need activation-energy units.

Black-box mode (``output.activations is None``):
- Run real token-masking forward passes (``prob_baseline − prob_masked``).
- Optionally compute pairwise interactions for multi-hop detection.

SAE features
------------
Each attached SAE reads the activation at its hook point
``blocks.{layer}.hook_{site}`` (TransformerLens / SAELens convention, identical
to :class:`~LLmThoughtLens.models.HookedModel` sites).  With
``ProviderOutput.activations[l] = resid_post[l]`` and
``ProviderOutput.embeddings = resid_pre[0]`` (see :func:`sae_input_activations`):

* ``resid_post`` at ``L``     -> ``activations[L]``
* ``resid_pre`` at ``L > 0``  -> ``activations[L - 1]`` (the same tensor)
* ``resid_pre`` at ``0``      -> ``embeddings`` (never ``activations[0]``)
* ``mlp_out`` / ``attn_out``  -> not in a ``ProviderOutput``; captured by
  re-running the prompt through ``provider.hooked`` and checked against the
  recorded residual stream, else :class:`SAEHookUnavailableError`.

Every SAE feature is a ``(sae, token position, dictionary index)`` triple with
``score`` = the SAE activation and ``Feature.layer`` = the residual-stream
layer it lives in (the ``activations`` index it was read from, or the block
whose output it is written into; ``0`` for an SAE on the embeddings).  Its
``meta`` carries (stable contract, consumed by attribution code):

``method`` (``"sae"``), ``sae_name`` (unique attachment name), ``sae_slot``
(index into :attr:`FeatureExtractor.saes`), ``sae_feature_id`` (dictionary
index), ``sae_layer`` / ``sae_site`` / ``sae_hook_name`` (hook point, usable
directly as ``ResidHook(sae_layer, fn, site=sae_site)``), ``sae_id``,
``sae_release``, ``sae_architecture``, ``activation`` (= score), ``position``
(= ``token_idx``), ``activation_source`` (``"activations"`` /
``"embeddings"`` / ``"recomputed"``), ``activation_layer`` (``activations``
index read, else ``None``), ``raw_norm`` (norm of the feature's decoded
contribution ``a·W_dec[:, i]`` in model units, when the SAE exposes its
decoder), ``excluded_positions`` (positions left out of the ranking),
``outlier_positions`` (detected massive-activation positions, as on the
residual-site path), and, only when the SAE was trained with a BOS token at
position 0 (SAELens ``prepend_bos``, default ``True``) but the prompt does not
start with the tokenizer's BOS id, ``sae_input_warning`` (also collected in
:attr:`FeatureExtractor.last_sae_warnings`) and ``bos_mismatch_positions``
(``[0]`` when that caveat excluded position 0).

SAE ranking uses the same exclusion semantics as residual-site scoring:
explicit ``exclude_positions`` always, and, by default
(``exclude_outlier_positions=None`` with ``scoring="centered"``), detected
attention-sink positions plus — when an attached SAE's BOS caveat fires —
position 0, whose codes are out of the SAE's training distribution.  On real
GPT-2 the sink at position 0 otherwise takes every top SAE feature (codes in
the hundreds vs. single digits elsewhere).  ``exclude_outlier_positions=False``
(or ``scoring="l2"``) opts out of both automatic exclusions; the attribute
:attr:`FeatureExtractor.sae_exclude_outlier_positions` can also be set directly.

``Feature.id`` is unique across SAEs, positions and dictionary indices
(``base(sae) + position * d_sae + index``, so ``id % d_sae`` is still the
dictionary index); if that would exceed ``1e9`` the ids are the features' rank
instead.  Always read ``meta["sae_feature_id"]`` for the dictionary index.

Returns :class:`~LLmThoughtLens.features.feature.Feature` objects tagged
with the correct ``evidence_kind`` so the report can colour and caveat
them appropriately.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from LLmThoughtLens.features.feature import Feature
from LLmThoughtLens.features.sae import SAE_SITES
from LLmThoughtLens.utils.tokenizer_utils import mask_positions, token_join, whitespace_tokens

if TYPE_CHECKING:
    from LLmThoughtLens.features.sae import SparseAutoencoder
    from LLmThoughtLens.providers.base import BaseProvider, ProviderOutput


WhiteboxScoring = Literal["centered", "l2"]

#: Valid values of ``FeatureExtractor(scoring=...)`` (also accepted by ``Scope``).
SCORING_METHODS: tuple[str, ...] = ("centered", "l2")
_SCORING_METHODS = SCORING_METHODS
_EPS = 1e-12
#: SAE feature ids stay below the tracer's first synthetic node id (input tokens).
_SAE_ID_LIMIT = 1_000_000_000
#: Hook-model families whose ``mlp_out`` / ``attn_out`` hooks sit *before* a
#: post-sublayer norm, so they differ from SAELens' ``hook_mlp_out`` / ``hook_attn_out``.
_POST_NORM_FAMILIES = ("gemma2", "gemma3")


class SAEHookUnavailableError(NotImplementedError):
    """The provider output does not contain the activation an SAE's hook point reads."""


@dataclass(frozen=True)
class SAEAttachment:
    """One SAE attached to a :class:`FeatureExtractor`.

    ``layer`` / ``site`` name the hook point ``blocks.{layer}.hook_{site}``
    the SAE reads; ``name`` is unique within the extractor.
    """

    sae: Any
    layer: int
    site: str
    name: str

    @property
    def hook_name(self) -> str:
        return f"blocks.{self.layer}.hook_{self.site}"

    def residual_layer(self) -> int:
        """Residual-stream layer (``activations`` index) a feature of this SAE lives in."""
        return max(self.layer - 1, 0) if self.site == "resid_pre" else self.layer


class FeatureExtractor:
    """Extract :class:`Feature` objects from a :class:`ProviderOutput`.

    Parameters
    ----------
    top_k:
        Maximum number of features to return per trace.
    blackbox_budget:
        Cap on number of token-masking forward passes per trace
        (each mask is one API call).  ``None`` means "all tokens".
    blackbox_cache:
        When ``True``, memoise masking results inside a single extractor
        instance so the tracer and probe runner share work.
    scoring:
        No-SAE white-box score.  ``"centered"`` (default) is the distance of
        ``a[l, t]`` from the layer's median activation over the rankable
        tokens, divided by the layer's median token norm.  ``"l2"`` is the
        legacy raw L2 norm of ``a[l, t]``.
    exclude_outlier_positions:
        Drop automatically detected massive-activation positions (attention
        sinks) from white-box ranking.  ``None`` (default) means "on for
        ``scoring="centered"``, off for ``scoring="l2"``" so that ``"l2"``
        reproduces the legacy behaviour exactly.
    exclude_positions:
        Token positions that are always excluded from white-box ranking, in
        addition to any detected outliers.  Negative indices count from the
        end of the prompt; out-of-range positions are ignored.
    outlier_ratio:
        A ``(layer, token)`` site is flagged when its norm exceeds
        ``outlier_ratio`` x the median norm of the other tokens in that layer.
    outlier_min_layer_frac:
        A position is treated as a massive-activation outlier when it is
        flagged in at least this fraction of the layers.
    top_k_per_sae:
        With SAEs attached: ``None`` (default) returns the global ``top_k``
        SAE features; an int returns up to that many features *per attached
        SAE* (all of them, sorted by score), so every layer is represented.

    SAE features use the same exclusions as residual-site scoring:
    ``exclude_positions`` always; detected attention-sink outliers — and
    position 0 when an attached SAE's BOS caveat fires (the SAE was trained
    with a BOS token the prompt lacks) — whenever outlier exclusion is on
    (the default for ``scoring="centered"``).  ``exclude_outlier_positions=False``
    opts out (attribute :attr:`sae_exclude_outlier_positions`).
    """

    def __init__(
        self,
        top_k: int = 20,
        blackbox_budget: int | None = 16,
        blackbox_cache: bool = True,
        *,
        scoring: WhiteboxScoring = "centered",
        exclude_outlier_positions: bool | None = None,
        exclude_positions: Iterable[int] | None = None,
        outlier_ratio: float = 6.0,
        outlier_min_layer_frac: float = 0.3,
        top_k_per_sae: int | None = None,
    ) -> None:
        if scoring not in _SCORING_METHODS:
            raise ValueError(f"scoring must be one of {_SCORING_METHODS}, got {scoring!r}")
        if not outlier_ratio > 1.0:
            raise ValueError(f"outlier_ratio must be > 1, got {outlier_ratio!r}")
        if not 0.0 < outlier_min_layer_frac <= 1.0:
            raise ValueError(
                f"outlier_min_layer_frac must be in (0, 1], got {outlier_min_layer_frac!r}"
            )
        if top_k_per_sae is not None and int(top_k_per_sae) < 1:
            raise ValueError(f"top_k_per_sae must be >= 1, got {top_k_per_sae!r}")
        self.top_k = int(top_k)
        self.top_k_per_sae: int | None = None if top_k_per_sae is None else int(top_k_per_sae)
        self.blackbox_budget = blackbox_budget
        self._cache: dict[str, float] = {} if blackbox_cache else {}
        self._cache_enabled = blackbox_cache
        self._saes: list[SAEAttachment] = []
        # Provider used during the last extract() call — needed for black-box masking.
        self._last_provider: BaseProvider | None = None
        #: ``(attachment name, reason)`` for SAEs skipped by the last extract() call.
        self.last_skipped_saes: list[tuple[str, str]] = []
        #: Input caveats for the last extract() call's SAEs (also in ``meta["sae_input_warning"]``).
        self.last_sae_warnings: list[str] = []

        self.scoring: WhiteboxScoring = scoring
        self.exclude_outlier_positions: bool = (
            scoring == "centered"
            if exclude_outlier_positions is None
            else exclude_outlier_positions
        )
        self.exclude_positions: tuple[int, ...] = tuple(int(p) for p in exclude_positions or ())
        #: Exclude detected attention-sink outliers (and position 0 under an SAE BOS
        #: mismatch) from SAE ranking; follows ``exclude_outlier_positions``.
        self.sae_exclude_outlier_positions: bool = self.exclude_outlier_positions
        self.outlier_ratio = float(outlier_ratio)
        self.outlier_min_layer_frac = float(outlier_min_layer_frac)

        # Inspectable record of the last white-box (no-SAE) extraction.
        #: Positions actually left out of the ranking (explicit + detected outliers).
        self.last_excluded_positions: list[int] = []
        #: Positions detected as massive-activation outliers (excluded or not).
        self.last_outlier_positions: list[int] = []
        #: Per detected outlier position: ``max_ratio`` and ``layer_frac``.
        self.last_outlier_stats: dict[int, dict[str, float]] = {}

    # ------------------------------------------------------------------
    # SAE attachment
    # ------------------------------------------------------------------

    def attach_sae(
        self,
        sae: SparseAutoencoder,
        layer: int | None = None,
        *,
        site: str | None = None,
        name: str | None = None,
    ) -> SAEAttachment:
        """Use *sae* (and only it) for white-box feature extraction.

        Parameters
        ----------
        sae:
            A :class:`~LLmThoughtLens.features.sae.SparseAutoencoder` (or any
            object with ``encode(x (T, D)) -> (T, d_sae)``).
        layer:
            Hook layer, in the SAE's own site convention.  Defaults to the
            SAE's ``config.hook_layer``; required for SAEs without hook
            metadata (e.g. trained with ``train-sae``).
        site:
            One of :data:`~LLmThoughtLens.features.sae.SAE_SITES`.  Defaults to
            the SAE's ``config.hook_site``, else ``"resid_post"`` — i.e. *layer*
            indexes ``ProviderOutput.activations``, the historical meaning.
        name:
            Unique attachment name (default ``blocks.{layer}.hook_{site}``).

        Replaces any previously attached SAEs; see :meth:`add_sae` /
        :meth:`attach_saes` for several.
        """
        attachment = self._make_attachment(sae, layer, site, name, taken=())
        self._saes = [attachment]
        return attachment

    def add_sae(
        self,
        sae: SparseAutoencoder,
        layer: int | None = None,
        *,
        site: str | None = None,
        name: str | None = None,
    ) -> SAEAttachment:
        """Attach *sae* in addition to the SAEs already attached (same arguments as :meth:`attach_sae`)."""
        attachment = self._make_attachment(
            sae, layer, site, name, taken=tuple(a.name for a in self._saes)
        )
        self._saes.append(attachment)
        return attachment

    def attach_saes(self, saes: Any) -> list[SAEAttachment]:
        """Replace the attached SAEs with several at once.

        *saes* may be a mapping ``{layer: sae}`` or ``{name: sae}``, or an
        iterable of SAEs (hook point from their metadata), ``(sae, layer)`` /
        ``(sae, layer, site)`` tuples, or :class:`SAEAttachment` objects.
        Returns the new attachments in order.
        """
        specs: list[tuple[Any, int | None, str | None, str | None]] = []
        if isinstance(saes, Mapping):
            for key, sae in saes.items():
                if isinstance(key, (int, np.integer)) and not isinstance(key, bool):
                    specs.append((sae, int(key), None, None))
                elif isinstance(key, str):
                    specs.append((sae, None, None, key))
                else:
                    raise TypeError(
                        f"attach_saes mapping keys must be int layers or str names: {key!r}"
                    )
        else:
            for item in saes:
                if isinstance(item, SAEAttachment):
                    specs.append((item.sae, item.layer, item.site, item.name))
                elif isinstance(item, tuple):
                    if not 2 <= len(item) <= 3:
                        raise TypeError(
                            f"expected (sae, layer) or (sae, layer, site), got {item!r}"
                        )
                    specs.append((item[0], item[1], item[2] if len(item) == 3 else None, None))
                else:
                    specs.append((item, None, None, None))
        new: list[SAEAttachment] = []
        for sae, layer, site, name in specs:
            new.append(
                self._make_attachment(sae, layer, site, name, taken=tuple(a.name for a in new))
            )
        self._saes = new
        return list(new)

    def detach_saes(self) -> None:
        """Remove every attached SAE (back to residual-site scoring)."""
        self._saes = []

    @staticmethod
    def _make_attachment(
        sae: Any,
        layer: int | None,
        site: str | None,
        name: str | None,
        *,
        taken: tuple[str, ...],
    ) -> SAEAttachment:
        if not callable(getattr(sae, "encode", None)):
            raise TypeError(f"{type(sae).__name__} has no encode(); not an SAE")
        cfg = getattr(sae, "config", None)
        hook_layer = getattr(cfg, "hook_layer", None)
        hook_site = getattr(cfg, "hook_site", None)
        hook_name = getattr(cfg, "hook_name", None)
        if site is None:
            if hook_site is not None:
                site = str(hook_site)
            elif hook_name:
                raise ValueError(
                    f"the SAE reads hook point {hook_name!r}, which is not one of {SAE_SITES}; "
                    "it cannot be read from a ProviderOutput"
                )
            else:
                site = "resid_post"
        if site not in SAE_SITES:
            raise ValueError(f"unknown SAE site {site!r}; expected one of {SAE_SITES}")
        if layer is None:
            if hook_layer is None:
                raise ValueError(
                    "layer is required: this SAE carries no hook_layer metadata "
                    "(pass the layer it was trained on)"
                )
            layer = int(hook_layer)
        layer = int(layer)
        if (
            hook_layer is not None
            and hook_site is not None
            and (layer, site)
            != (
                int(hook_layer),
                hook_site,
            )
        ):
            warnings.warn(
                f"SAE trained on blocks.{hook_layer}.hook_{hook_site} attached at "
                f"blocks.{layer}.hook_{site}",
                UserWarning,
                stacklevel=3,
            )
        base = name or f"blocks.{layer}.hook_{site}"
        unique, n = base, 2
        while unique in taken:
            if name is not None:
                raise ValueError(f"an SAE named {name!r} is already attached")
            unique, n = f"{base}#{n}", n + 1
        return SAEAttachment(sae=sae, layer=layer, site=site, name=unique)

    @property
    def saes(self) -> tuple[SAEAttachment, ...]:
        """The attached SAEs, in attachment order (``sae_slot`` indexes this)."""
        return tuple(self._saes)

    @property
    def sae_map(self) -> dict[str, Any]:
        """``{attachment name: sae}`` — e.g. for ``SupernodeGrouper(sae=extractor.sae_map)``."""
        return {a.name: a.sae for a in self._saes}

    @property
    def sae(self) -> SparseAutoencoder | None:
        """The first attached SAE (``None`` when none is attached)."""
        return self._saes[0].sae if self._saes else None

    @property
    def sae_layer(self) -> int:
        """Hook layer of the first attached SAE (``-1`` when none is attached)."""
        return self._saes[0].layer if self._saes else -1

    # ------------------------------------------------------------------
    # Main dispatch
    # ------------------------------------------------------------------

    def extract(
        self,
        output: ProviderOutput,
        provider: BaseProvider | None = None,
    ) -> list[Feature]:
        """Top-level extraction dispatch.

        Parameters
        ----------
        output:
            The provider's :class:`ProviderOutput` for the prompt.
        provider:
            The provider instance — required for black-box masking (so we
            can issue masked forward passes).  When ``None`` and the
            output is black-box, we fall back to a single-pass importance
            heuristic that is clearly labelled as such.
        """
        self._last_provider = provider
        # Reset the exclusion record on *every* call so the SAE and black-box
        # paths never leave a previous white-box trace's values behind.
        self._reset_exclusion_record()
        self.last_skipped_saes = []
        self.last_sae_warnings = []
        if output.has_internals:
            return self._whitebox_features(output)
        return self._blackbox_features(output, provider)

    def _reset_exclusion_record(self) -> None:
        self.last_excluded_positions = []
        self.last_outlier_positions = []
        self.last_outlier_stats = {}

    # ------------------------------------------------------------------
    # White-box: SAE-based features when available, else residual-site scoring.
    # ------------------------------------------------------------------

    def _whitebox_features(self, output: ProviderOutput) -> list[Feature]:
        activations = output.activations
        assert activations is not None
        n_layers, n_tokens, _ = activations.shape

        if self._saes:
            sae_features = self._sae_features(output)
            if sae_features is not None:
                return sae_features

        self._reset_exclusion_record()
        if n_layers == 0 or n_tokens == 0:
            return []

        norms = _site_norms(activations)

        outliers, stats = detect_outlier_positions(
            norms, ratio=self.outlier_ratio, min_layer_frac=self.outlier_min_layer_frac
        )
        self.last_outlier_positions = outliers
        self.last_outlier_stats = stats

        excluded = {p % n_tokens for p in self.exclude_positions if -n_tokens <= p < n_tokens}
        if self.exclude_outlier_positions:
            auto = set(outliers) - excluded
            # Never let auto-detection remove every remaining position.
            if len(excluded | auto) < n_tokens:
                excluded |= auto
        keep = [t for t in range(n_tokens) if t not in excluded]
        excluded_list = sorted(excluded)
        self.last_excluded_positions = excluded_list
        if not keep:
            return []

        method = "l2_norm"
        fallback: str | None = None
        scores: np.ndarray = norms
        scales: np.ndarray | None = None
        if self.scoring == "centered":
            if len(keep) >= 2:
                scores, scales = _centered_scores(activations, norms, keep)
                method = "centered_norm"
            else:
                # A single rankable position has no layer centre to compare
                # against — rank its layers by raw norm and say so.
                fallback = "centered_needs_2_positions"

        features: list[Feature] = []
        for layer in range(n_layers):
            for tok_idx in keep:
                meta: dict[str, Any] = {
                    "method": method,
                    "raw_norm": float(norms[layer, tok_idx]),
                    "excluded_positions": list(excluded_list),
                    "outlier_positions": list(outliers),
                }
                if scales is not None:
                    meta["layer_scale"] = float(scales[layer])
                if fallback is not None:
                    meta["fallback"] = fallback
                features.append(
                    Feature(
                        # Stable id: identical to the legacy sequential numbering.
                        id=layer * n_tokens + tok_idx,
                        label=_layer_band_label(output.tokens, tok_idx, layer, n_layers),
                        layer=layer,
                        score=float(scores[layer, tok_idx]),
                        token_idx=tok_idx,
                        node_type="feature",
                        evidence_kind="white_box",
                        meta=meta,
                    )
                )
        features.sort(key=lambda f: f.score, reverse=True)
        return features[: self.top_k]

    def _sae_features(self, output: ProviderOutput) -> list[Feature] | None:
        """SAE features from every usable attached SAE; ``None`` when none applies.

        SAEs whose hook layer is outside this output's layer range are skipped
        with a warning (recorded in :attr:`last_skipped_saes`); when every SAE
        is skipped the caller falls back to residual-site scoring, as before
        multi-SAE support.
        """
        activations = output.activations
        assert activations is not None
        n_layers, n_tokens, d_model = activations.shape

        usable: list[tuple[int, SAEAttachment]] = []
        for slot, att in enumerate(self._saes):
            if not 0 <= att.layer < n_layers:
                reason = f"hook layer {att.layer} is outside this {n_layers}-layer model"
                self.last_skipped_saes.append((att.name, reason))
                warnings.warn(f"SAE {att.name!r} skipped: {reason}", RuntimeWarning, stacklevel=4)
                continue
            d_in = getattr(getattr(att.sae, "config", None), "input_dim", None)
            if d_in is not None and int(d_in) != d_model:
                raise ValueError(
                    f"SAE {att.name!r} expects d_in={d_in} but this model's residual stream "
                    f"has d_model={d_model} (an SAE for a different model?)"
                )
            usable.append((slot, att))
        if not usable:
            return None
        if n_tokens == 0:
            return []

        bos_first = _first_token_is_bos(output, self._last_provider)
        bos_warning: dict[int, str] = {}
        for slot, att in usable:
            if bos_first is False and _sae_expects_bos(att.sae):
                bos_warning[slot] = (
                    f"SAE {att.name!r} was trained with a BOS token at position 0 but this "
                    "prompt does not start with one; position-0 codes are out of distribution "
                    + (
                        "(position 0 is excluded from the ranking; prepend the BOS token to "
                        "rank it)"
                        if self.sae_exclude_outlier_positions
                        else "(prepend the BOS token, or exclude position 0)"
                    )
                )
                self.last_sae_warnings.append(bos_warning[slot])

        # Same exclusion semantics as residual-site scoring: explicit positions
        # always; detected attention sinks (+ position 0 under a BOS mismatch)
        # unless outlier exclusion is off.  Outliers are always detected and
        # recorded so renderers can say why a position is greyed.
        excluded = {p % n_tokens for p in self.exclude_positions if -n_tokens <= p < n_tokens}
        outliers, stats = detect_outlier_positions(
            _site_norms(activations),
            ratio=self.outlier_ratio,
            min_layer_frac=self.outlier_min_layer_frac,
        )
        self.last_outlier_positions = outliers
        self.last_outlier_stats = stats
        bos_excluded: list[int] = []
        if self.sae_exclude_outlier_positions:
            auto = set(outliers) - excluded
            # Never let automatic exclusion remove every remaining position.
            if len(excluded | auto) < n_tokens:
                excluded |= auto
            if bos_warning and 0 not in excluded and len(excluded) + 1 < n_tokens:
                excluded.add(0)
                bos_excluded = [0]
        excluded_list = sorted(excluded)
        self.last_excluded_positions = excluded_list
        keep = [t for t in range(n_tokens) if t not in excluded]

        inputs = _gather_sae_inputs(
            output, [(att.layer, att.site) for _, att in usable], self._last_provider
        )

        # (score, slot, token, dictionary index) candidates, best first per SAE.
        per_sae: dict[int, list[tuple[float, int, int, int]]] = {}
        details: dict[int, dict[str, Any]] = {}
        bases: dict[int, int] = {}
        next_base = 0
        for slot, att in usable:
            x, provenance = inputs[(att.layer, att.site)]
            codes, out_scale = _encode_with_scale(att.sae, x)
            if codes.ndim != 2 or codes.shape[0] != n_tokens:
                raise ValueError(
                    f"SAE {att.name!r} returned codes of shape {codes.shape}; "
                    f"expected ({n_tokens}, d_sae)"
                )
            d_sae = int(codes.shape[1])
            next_base = -(-next_base // d_sae) * d_sae  # base(slot) is a multiple of d_sae
            bases[slot] = next_base
            next_base += n_tokens * d_sae
            dec_norms = _decoder_norms(att.sae)
            details[slot] = {
                "codes": codes,
                "out_scale": out_scale,
                "dec_norms": dec_norms,
                "d_sae": d_sae,
                "provenance": provenance,
                "labels": dict(getattr(att.sae, "labels", None) or {}),
            }
            limit = self.top_k if self.top_k_per_sae is None else self.top_k_per_sae
            local_top = min(max(limit, 1), d_sae)
            candidates: list[tuple[float, int, int, int]] = []
            for tok in keep:
                row = codes[tok]
                if not np.any(row > 0):
                    continue
                top_ids = np.argpartition(-row, local_top - 1)[:local_top]
                top_ids = top_ids[np.argsort(-row[top_ids], kind="stable")]
                candidates.extend(
                    (float(row[fid]), slot, tok, int(fid)) for fid in top_ids if row[fid] > 0
                )
            candidates.sort(key=lambda c: c[0], reverse=True)
            per_sae[slot] = candidates

        if self.top_k_per_sae is None:
            chosen = sorted(
                (c for cands in per_sae.values() for c in cands), key=lambda c: c[0], reverse=True
            )[: self.top_k]
        else:
            chosen = sorted(
                (c for cands in per_sae.values() for c in cands[: self.top_k_per_sae]),
                key=lambda c: c[0],
                reverse=True,
            )
        sequential_ids = next_base > _SAE_ID_LIMIT

        single = len(usable) == 1
        by_slot = dict(usable)
        features: list[Feature] = []
        for rank, (score, slot, tok, fid) in enumerate(chosen):
            att, info = by_slot[slot], details[slot]
            cfg = getattr(att.sae, "config", None)
            default_label = f"feature_{fid}" if single else f"{att.name}/feature_{fid}"
            meta: dict[str, Any] = {
                "method": "sae",
                "sae_name": att.name,
                "sae_slot": slot,
                "sae_feature_id": fid,
                "sae_layer": att.layer,
                "sae_site": att.site,
                "sae_hook_name": att.hook_name,
                "sae_id": getattr(cfg, "sae_id", None),
                "sae_release": getattr(cfg, "release", None),
                "sae_architecture": getattr(cfg, "architecture", None),
                "activation": score,
                "position": tok,
                **info["provenance"],
                "excluded_positions": list(excluded_list),
                "outlier_positions": list(outliers),
            }
            if bos_excluded:
                meta["bos_mismatch_positions"] = list(bos_excluded)
            if slot in bos_warning:
                meta["sae_input_warning"] = bos_warning[slot]
            if info["dec_norms"] is not None:
                meta["raw_norm"] = float(score * info["dec_norms"][fid] * info["out_scale"][tok])
            features.append(
                Feature(
                    id=rank if sequential_ids else bases[slot] + tok * info["d_sae"] + fid,
                    label=str(info["labels"].get(fid, default_label)),
                    layer=att.residual_layer(),
                    score=score,
                    token_idx=tok,
                    node_type="feature",
                    evidence_kind="white_box",
                    meta=meta,
                )
            )
        return features

    # ------------------------------------------------------------------
    # Black-box: real token-masking importance.
    # ------------------------------------------------------------------

    def _blackbox_features(
        self, output: ProviderOutput, provider: BaseProvider | None
    ) -> list[Feature]:
        tokens = output.tokens
        if not tokens:
            return []

        # If we don't have a provider handle, we cannot actually mask.  Return
        # one feature per token with a *clearly approximated* heuristic score
        # taken from the surface position so downstream code degrades gracefully.
        if provider is None:
            heuristic_score = output.output_prob
            features = [
                Feature(
                    id=i,
                    label=f"token:{t}",
                    layer=0,
                    score=float(heuristic_score / (1.0 + i)),
                    token_idx=i,
                    node_type="input_token",
                    evidence_kind="black_box",
                    meta={"method": "position_heuristic", "approximation": True},
                )
                for i, t in enumerate(tokens)
            ]
            features.sort(key=lambda f: f.score, reverse=True)
            return features[: self.top_k]

        importance = self.compute_token_importance(
            provider=provider, prompt=output.prompt, baseline=output
        )
        # importance is [(token, score)] — convert to Feature list.
        features = [
            Feature(
                id=i,
                label=f"token:{tok}",
                layer=0,
                score=float(score),
                token_idx=i,
                node_type="input_token",
                evidence_kind="black_box",
                meta={"method": "token_masking"},
            )
            for i, (tok, score) in enumerate(importance)
        ]
        features.sort(key=lambda f: f.score, reverse=True)
        return features[: self.top_k]

    # ------------------------------------------------------------------
    # Black-box: real masking + pairwise interactions.
    # ------------------------------------------------------------------

    def compute_token_importance(
        self,
        provider: BaseProvider,
        prompt: str,
        baseline: ProviderOutput | None = None,
    ) -> list[tuple[str, float]]:
        """Compute per-token causal importance via masking.

        ``score_i = prob_baseline(top) − prob_masked_i(top)`` with
        ``top`` fixed to the baseline's argmax.  Positive ⇒ token was
        load-bearing; negative ⇒ token was actively suppressing that prediction.
        """
        tokens = whitespace_tokens(prompt)
        if not tokens:
            return []

        if baseline is None:
            baseline = provider.run(prompt)
        target_token = baseline.output_token
        baseline_prob = baseline.output_prob

        limit = min(self.blackbox_budget or len(tokens), len(tokens))
        scores: list[tuple[str, float]] = []
        for i in range(limit):
            masked = mask_positions(tokens, [i])
            masked_prompt = token_join(masked)
            masked_prob = self._masked_prob(provider, masked_prompt, target_token)
            scores.append((tokens[i], float(baseline_prob - masked_prob)))
        for i in range(limit, len(tokens)):
            scores.append((tokens[i], 0.0))
        return scores

    def compute_pairwise_interactions(
        self,
        provider: BaseProvider,
        prompt: str,
        budget: int = 8,
    ) -> dict[tuple[int, int], float]:
        """Pairwise interaction scores for multi-hop circuit detection.

        ``interaction(i, j) = P_full(top) − P_mask_i(top) − P_mask_j(top) + P_mask_{i,j}(top)``

        Positive ⇒ tokens i and j are *jointly* required.
        """
        tokens = whitespace_tokens(prompt)
        n = min(len(tokens), int(budget))
        baseline = provider.run(prompt)
        target = baseline.output_token

        p_full = baseline.output_prob
        p_single = {
            i: self._masked_prob(provider, token_join(mask_positions(tokens, [i])), target)
            for i in range(n)
        }
        interactions: dict[tuple[int, int], float] = {}
        for i in range(n):
            for j in range(i + 1, n):
                p_both = self._masked_prob(
                    provider, token_join(mask_positions(tokens, [i, j])), target
                )
                interactions[(i, j)] = float(p_full - p_single[i] - p_single[j] + p_both)
        return interactions

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _masked_prob(self, provider: BaseProvider, prompt: str, target_token: str) -> float:
        """Return P(target_token | masked_prompt) using *provider*.

        Looks up *target_token* in the masked run's ``top_tokens`` list; if
        the target isn't in the top-k we approximate its probability as 0.0
        (the model has effectively given up on that prediction at that mask).
        """
        if self._cache_enabled and prompt in self._cache:
            base_prob = self._cache[prompt]
            return base_prob if not target_token else base_prob

        out = provider.run(prompt)
        top = dict(out.top_tokens)
        prob = float(top.get(target_token, 0.0)) if target_token else out.output_prob
        if self._cache_enabled:
            self._cache[prompt] = prob
        return prob


# ---------------------------------------------------------------------------
# SAE hook-point mapping
# ---------------------------------------------------------------------------


def sae_input_activations(
    output: ProviderOutput,
    layer: int,
    site: str,
    provider: BaseProvider | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """The ``(T, D)`` activation an SAE hooked at ``blocks.{layer}.hook_{site}`` reads.

    Mapping (HookedModel / TransformerLens convention; ``activations[l]`` is
    ``resid_post[l]``, ``embeddings`` is ``resid_pre[0]``):

    * ``resid_post``, ``layer`` ``L`` -> ``output.activations[L]``
    * ``resid_pre``, ``L > 0``        -> ``output.activations[L - 1]``
    * ``resid_pre``, ``L == 0``       -> ``output.embeddings``
    * ``mlp_out`` / ``attn_out``      -> re-run ``provider.hooked`` on
      ``output.token_ids`` with an observe-only hook at that site; the re-run's
      residual stream must reproduce ``output.activations``.

    Returns ``(activations, provenance)`` where provenance holds
    ``activation_source`` (``"activations"`` / ``"embeddings"`` /
    ``"recomputed"``) and ``activation_layer`` (the ``activations`` index read,
    or ``None``).  Raises :class:`SAEHookUnavailableError` when the tensor is
    not available — never substitutes a different one.
    """
    return _gather_sae_inputs(output, [(int(layer), str(site))], provider)[(int(layer), str(site))]


def _gather_sae_inputs(
    output: ProviderOutput,
    keys: list[tuple[int, str]],
    provider: BaseProvider | None,
) -> dict[tuple[int, str], tuple[np.ndarray, dict[str, Any]]]:
    activations = output.activations
    if activations is None:
        raise SAEHookUnavailableError("black-box output: no activations to feed an SAE")
    n_layers, n_tokens, d_model = activations.shape
    out: dict[tuple[int, str], tuple[np.ndarray, dict[str, Any]]] = {}
    to_recompute: list[tuple[int, str]] = []
    for layer, site in dict.fromkeys(keys):
        if site not in SAE_SITES:
            raise ValueError(f"unknown SAE site {site!r}; expected one of {SAE_SITES}")
        if not 0 <= layer < n_layers:
            raise IndexError(f"hook layer {layer} out of range for a {n_layers}-layer output")
        if site == "resid_post":
            out[(layer, site)] = (
                activations[layer],
                {"activation_source": "activations", "activation_layer": layer},
            )
        elif site == "resid_pre" and layer > 0:
            out[(layer, site)] = (
                activations[layer - 1],
                {"activation_source": "activations", "activation_layer": layer - 1},
            )
        elif site == "resid_pre":
            emb = output.embeddings
            if emb is None:
                raise SAEHookUnavailableError(
                    "blocks.0.hook_resid_pre is the embedding output, but this provider's "
                    "ProviderOutput.embeddings is None (only the HuggingFace provider records it)"
                )
            if tuple(emb.shape) != (n_tokens, d_model):
                raise ValueError(
                    f"embeddings have shape {tuple(emb.shape)}, expected {(n_tokens, d_model)}"
                )
            out[(layer, site)] = (
                emb,
                {"activation_source": "embeddings", "activation_layer": None},
            )
        else:
            to_recompute.append((layer, site))
    if to_recompute:
        for key, arr in _recompute_sites(output, to_recompute, provider).items():
            out[key] = (arr, {"activation_source": "recomputed", "activation_layer": None})
    return out


def _recompute_sites(
    output: ProviderOutput,
    keys: list[tuple[int, str]],
    provider: BaseProvider | None,
) -> dict[tuple[int, str], np.ndarray]:
    """Capture ``mlp_out`` / ``attn_out`` activations by re-running the prompt."""
    names = ", ".join(f"blocks.{layer}.hook_{site}" for layer, site in keys)
    hooked = getattr(provider, "hooked", None) if provider is not None else None
    if hooked is None:
        raise SAEHookUnavailableError(
            f"{names}: a ProviderOutput carries only the residual stream (resid_pre / "
            "resid_post); reading mlp_out / attn_out needs a provider with a HookedModel "
            "(HuggingFaceProvider), passed to FeatureExtractor.extract(output, provider=...)"
        )
    family = str(getattr(hooked, "family", ""))
    if family in _POST_NORM_FAMILIES:
        raise SAEHookUnavailableError(
            f"{names}: for {family} the HookedModel mlp_out / attn_out hooks sit before the "
            "post-sublayer norm, while SAELens / Gemma Scope hook_mlp_out / hook_attn_out "
            "read after it"
        )
    if output.meta.get("n_intervention_hooks"):
        raise SAEHookUnavailableError(
            f"{names}: this output was produced with interventions, which a re-run "
            "would not reproduce"
        )
    if not output.token_ids:
        raise SAEHookUnavailableError(f"{names}: the output has no token ids to re-run")

    from LLmThoughtLens.models.hooked import ResidHook

    store: dict[tuple[int, str], np.ndarray] = {}

    def _keep(key: tuple[int, str]) -> Any:
        def _fn(hidden: Any) -> None:
            store[key] = hidden[0].detach().float().cpu().numpy()
            return None

        return _fn

    hooks = [ResidHook(layer, _keep((layer, site)), site=site) for layer, site in keys]  # type: ignore[arg-type]
    res = hooked.forward(
        list(output.token_ids), hooks=hooks, capture=True, capture_attentions=False
    )
    got = res.resid_post.detach().float().cpu().numpy()
    ref = output.activations
    assert ref is not None
    scale = max(1.0, float(np.max(np.abs(ref)))) if ref.size else 1.0
    if got.shape != ref.shape or float(np.max(np.abs(got - ref))) > 1e-4 * scale:
        raise SAEHookUnavailableError(
            f"{names}: re-running the prompt did not reproduce this output's residual stream "
            "(different tokens, model state or interventions)"
        )
    missing = [k for k in keys if k not in store]
    if missing:
        raise SAEHookUnavailableError(f"hooks at {missing} did not fire during the re-run")
    return store


def _sae_expects_bos(sae: Any) -> bool:
    """True when the SAE's provenance says it was trained with a BOS token prepended.

    SAELens records ``prepend_bos`` (default ``True`` when absent); other
    sources are trusted only when they record it explicitly.
    """
    cfg = getattr(sae, "config", None)
    extra = getattr(cfg, "extra", None) or {}
    if "prepend_bos" in extra:
        return bool(extra["prepend_bos"])
    return getattr(cfg, "source_format", None) == "saelens"


def _first_token_is_bos(output: ProviderOutput, provider: BaseProvider | None) -> bool | None:
    """Whether the prompt starts with the tokenizer's BOS id; ``None`` when unknown.

    The tokenizer is read only from an already-loaded provider (never triggers a load).
    """
    if not output.token_ids:
        return None
    bos = output.meta.get("bos_token_id")
    if bos is None and provider is not None and getattr(provider, "_model", None) is not None:
        tokenizer = getattr(getattr(provider, "hooked", None), "tokenizer", None)
        bos = getattr(tokenizer, "bos_token_id", None)
    if bos is None:
        return None
    return int(output.token_ids[0]) == int(bos)


def _encode_with_scale(sae: Any, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(codes, per-token output scale)``; duck-typed SAEs get a scale of 1."""
    fn = getattr(sae, "encode_with_output_scale", None)
    if callable(fn):
        codes, scale = fn(x)
        return np.asarray(codes), np.asarray(scale, dtype=np.float64)
    codes = np.asarray(sae.encode(x))
    return codes, np.ones(codes.shape[0], dtype=np.float64)


def _decoder_norms(sae: Any) -> np.ndarray | None:
    fn = getattr(sae, "decoder_norms", None)
    return np.asarray(fn(), dtype=np.float64) if callable(fn) else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _site_norms(activations: np.ndarray) -> np.ndarray:
    """``(L, T)`` residual L2 norms, in float64 one layer at a time (no fp16 overflow)."""
    return np.stack(
        [
            np.linalg.norm(activations[layer].astype(np.float64), axis=-1)
            for layer in range(activations.shape[0])
        ]
    )


#: Why a position was left out of white-box ranking (used by every renderer).
EXCLUSION_REASON_OUTLIER = "attention-sink outlier"
EXCLUSION_REASON_REQUESTED = "excluded by request"
EXCLUSION_REASON_UNKNOWN = "excluded from feature ranking"
EXCLUSION_REASON_SAE_BOS = "SAE input out of distribution (prompt lacks the SAE's BOS token)"


def exclusion_reasons(
    features: Iterable[Feature],
    excluded_positions: Iterable[int] | None = None,
) -> dict[int, str]:
    """Map each position left out of white-box ranking to a human-readable reason.

    The excluded / detected-outlier positions are read from the first
    feature whose ``meta`` records them (every white-box feature of one trace
    carries the same lists).  A recorded position that was detected as a
    massive-activation outlier is :data:`EXCLUSION_REASON_OUTLIER`; position 0
    excluded because an SAE's BOS caveat fired (``meta["bos_mismatch_positions"]``)
    is :data:`EXCLUSION_REASON_SAE_BOS`; any other recorded position was
    requested by the caller (:data:`EXCLUSION_REASON_REQUESTED`).
    *excluded_positions* (e.g. ``graph.meta["excluded_positions"]``) is merged
    in; positions known only from it get :data:`EXCLUSION_REASON_UNKNOWN`.
    Black-box traces exclude nothing and yield ``{}``; SAE traces use the same
    exclusions as residual-site traces.
    """
    recorded: set[int] = set()
    outliers: set[int] = set()
    bos: set[int] = set()
    for f in features:
        positions = f.meta.get("excluded_positions")
        if positions is None:
            continue
        recorded = {int(p) for p in positions}
        outliers = {int(p) for p in f.meta.get("outlier_positions") or ()}
        bos = {int(p) for p in f.meta.get("bos_mismatch_positions") or ()}
        break
    reasons = {int(p): EXCLUSION_REASON_UNKNOWN for p in excluded_positions or ()}
    for p in recorded:
        if p in outliers:
            reasons[p] = EXCLUSION_REASON_OUTLIER
        elif p in bos:
            reasons[p] = EXCLUSION_REASON_SAE_BOS
        else:
            reasons[p] = EXCLUSION_REASON_REQUESTED
    return dict(sorted(reasons.items()))


def detect_outlier_positions(
    norms: np.ndarray,
    ratio: float = 6.0,
    min_layer_frac: float = 0.3,
) -> tuple[list[int], dict[int, dict[str, float]]]:
    """Find massive-activation ("attention sink") token positions.

    Parameters
    ----------
    norms:
        ``(n_layers, n_tokens)`` array of per-site residual-stream L2 norms.
    ratio:
        A site ``(l, t)`` is flagged when ``norms[l, t]`` exceeds ``ratio`` x
        the median norm of the *other* tokens in layer ``l``.
    min_layer_frac:
        A position is an outlier when it is flagged in at least this
        fraction of the layers (massive activations persist across depth;
        ordinary content tokens do not).

    Returns
    -------
    tuple
        ``(positions, stats)`` — the sorted outlier positions and, for each,
        ``{"max_ratio": ..., "layer_frac": ...}``.
    """
    if norms.ndim != 2:
        raise ValueError(f"expected (L, T) norms, got shape {norms.shape}")
    n_layers, n_tokens = norms.shape
    if n_layers == 0 or n_tokens < 2:
        return [], {}
    ref = _median_excluding_each(norms)
    valid = ref > _EPS
    ratios = np.where(valid, norms / np.where(valid, ref, 1.0), 0.0)
    flagged = valid & (norms > ratio * ref)
    layer_frac = flagged.mean(axis=0)
    positions = [int(t) for t in np.flatnonzero(layer_frac >= min_layer_frac)]
    stats = {
        t: {"max_ratio": float(ratios[:, t].max()), "layer_frac": float(layer_frac[t])}
        for t in positions
    }
    return positions, stats


def _median_excluding_each(x: np.ndarray) -> np.ndarray:
    """Return ``out[l, t] = median(x[l, j] for j != t)`` for a ``(L, T)`` array, ``T >= 2``."""
    n_rows, n_cols = x.shape
    order = np.argsort(x, axis=1, kind="stable")
    sorted_x = np.take_along_axis(x, order, axis=1)
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.broadcast_to(np.arange(n_cols), (n_rows, n_cols)), axis=1)
    remaining = n_cols - 1
    lo, hi = (remaining - 1) // 2, remaining // 2

    def _at(i: int) -> np.ndarray:
        # Index i of the sorted row with element ``t`` removed.
        idx = np.where(i < ranks, i, i + 1)
        return np.take_along_axis(sorted_x, idx, axis=1)

    return 0.5 * (_at(lo) + _at(hi))


def _centered_scores(
    activations: np.ndarray, norms: np.ndarray, keep: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Distance of each site from its layer's robust centre, in layer-scale units.

    The centre is the coordinate-wise median of ``activations[l, keep]`` and
    the scale is the median of ``norms[l, keep]``; excluded positions affect
    neither.  Returns ``(scores (L, T), scales (L,))``; columns outside
    *keep* are scored too but callers only read the kept ones.
    """
    n_layers = activations.shape[0]
    keep_idx = np.asarray(keep, dtype=np.intp)
    scores = np.zeros_like(norms)
    scales = np.ones(n_layers, dtype=np.float64)
    for layer in range(n_layers):
        layer_acts = activations[layer].astype(np.float64)
        centre = np.median(layer_acts[keep_idx], axis=0)
        scale = float(np.median(norms[layer, keep_idx]))
        if scale <= _EPS:
            scale = 1.0
        scales[layer] = scale
        scores[layer] = np.linalg.norm(layer_acts - centre, axis=-1) / scale
    return scores, scales


def _layer_band_label(tokens: list[str], tok_idx: int, layer: int, n_layers: int) -> str:
    tok = tokens[tok_idx] if 0 <= tok_idx < len(tokens) else "?"
    if n_layers <= 1:
        band = "only"
    elif layer < n_layers // 3:
        band = "early"
    elif layer < 2 * n_layers // 3:
        band = "mid"
    else:
        band = "late"
    return f"{tok}@{band}"
