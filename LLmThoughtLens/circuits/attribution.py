"""Gradient attribution over the residual stream (attribution patching).

:class:`GradientAttributor` turns one prompt into *linearised causal*
attributions on a :class:`~LLmThoughtLens.models.hooked.HookedModel`: how much
each residual-stream node contributes to a scalar **target metric** at the
last position, and how much each node drives every later node.  torch is
imported lazily; importing this module needs only NumPy.

Target metric (``metric=``), always read at the last prompt position
---------------------------------------------------------------------
``"logit"`` (default)
    The logit of the target token (default target: the model's own top-1
    prediction).
``"logprob"``
    ``log_softmax(logits)[target]``.
``"logit_diff"``
    ``logit[target] - logit[runner_up]``; the runner-up defaults to the
    highest-scoring token other than the target.

Nodes
-----
A residual node is a ``(layer l, position t)`` site.  Its **value** ``v``
depends on ``node_kind``:

``"delta"`` (default)
    The block's *write*: ``delta[l, t] = resid_post[l][t] - resid_pre[l][t]``,
    i.e. what block ``l`` added at position ``t``.  This is the default
    because adjacent-layer residuals at one position are nearly identical
    (the skip connection carries everything forward), so attributing the
    *cumulative* residual lets the trivial identity path dominate: a trace
    collapses into "one token walking down the layers".  A block's write
    is new information, so edges between writes reflect computation, not
    copying.
``"resid"``
    The cumulative residual ``resid_post[l][t]`` itself (kept for
    comparison; it reproduces the identity-path artefact).
``"sae"`` (per node)
    An SAE feature at ``(sae_layer, t, feature f)``: its value is the code
    ``z_f`` computed differentiably (``sae.encode_torch``) from the live
    activation at the SAE's hook site ``sae_site`` (``resid_post``,
    ``resid_pre``, ``mlp_out`` or ``attn_out`` of ``sae_layer``); it writes
    ``z_f * scale_t * W_dec[:, f]`` into that activation, where ``scale_t`` is
    the SAE's output scale at position ``t`` (1 unless the SAE normalises its
    input; see :meth:`SparseAutoencoder.encode_with_output_scale`).  The SAE
    reconstruction error ``x - reconstruct(x)`` and the normalisation
    statistics are treated as constants; the error's own attribution is
    reported per position (``AttributionResult.sae_error_attr``).

Node -> metric attribution (input x gradient against a baseline)
    ``A = (v - b) . d metric / d v`` with ``b = 0`` (``baseline="zero"``) or
    the mean of ``v`` over the baseline positions (``baseline="mean"``).
    ``d metric / d v`` is the *total* derivative (every downstream path), so
    ``A`` is the first-order estimate of ``metric(clean) - metric(v -> b)``,
    which :class:`~LLmThoughtLens.circuits.patching.ActivationPatcher`
    measures exactly.  ``A`` is in metric units (logits / nats).

Node -> node edges (linearised effect on the destination's own contribution)
    For a destination node ``j`` let ``s_j = stopgrad(d metric / d v_j) . v_j``
    (the part of ``j``'s value that matters for the metric).  For a source
    ``i`` in an earlier layer, ``edge(i -> j) = (d s_j / d v_i) . (v_i - b_i)``:
    the first-order change in ``s_j`` if ``i`` were ablated to its baseline.
    One backward pass per destination gives the edges from every earlier
    node.  Edges are *total* effects (direct + mediated through
    intermediate layers), so path products are influence indicators, not an
    additive decomposition of the metric.

Input-token edges
    The same formula with the embedding output ``resid_pre[0][u]`` as the
    source value (``input -> node``), and ``A_input[u] = (e_u - b) . d metric
    / d e_u`` for the token's total attribution.

Edge direction
    Edges only run from an earlier to a strictly later **residual position**
    (:attr:`AttributionNode.order`): a block write / residual / ``resid_post``,
    ``mlp_out`` or ``attn_out`` SAE node at layer ``l`` sits at ``l``; a
    ``resid_pre`` SAE node at ``l`` reads ``resid_post[l - 1]`` and sits at
    ``l - 1`` (``-0.5`` for the embeddings at ``l = 0``).  Nodes at the same
    position share or contain each other's tensors, so they get no edge.

Every number here is exact calculus on the real model (checked against
finite differences in ``tests/test_attribution.py``); what is approximate is
the *linearisation*: for a large ablation the true effect can differ from
``A`` (e.g. when ablating a node switches an SAE feature or an MLP neuron
off).  ``ActivationPatcher.faithfulness`` measures how well it holds.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from LLmThoughtLens.features.feature import Feature
    from LLmThoughtLens.models.hooked import ForwardResult, HookedModel

__all__ = [
    "BASELINES",
    "METRICS",
    "NODE_KINDS",
    "SAE_SITES",
    "AttributionNode",
    "AttributionResult",
    "GradientAttributor",
    "MetricSpec",
    "SAEKey",
    "TargetMetric",
    "metric_from_logits",
    "resolve_target",
]

TargetMetric = Literal["logit", "logprob", "logit_diff"]
NodeKind = Literal["delta", "resid"]
BaselineKind = Literal["zero", "mean"]
#: ``(sae_layer, sae_site, sae_name)`` — identifies one attached SAE's hook point.
SAEKey = tuple[int, str, "str | None"]

#: Valid ``metric=`` values.
METRICS: tuple[str, ...] = ("logit", "logprob", "logit_diff")
#: Valid residual ``node_kind=`` values (``"sae"`` is per-node, from SAE features).
NODE_KINDS: tuple[str, ...] = ("delta", "resid")
#: Valid ``baseline=`` values.
BASELINES: tuple[str, ...] = ("zero", "mean")
#: Hook sites an SAE feature may be read from (the extractor's ``SAE_SITES``).
SAE_SITES: tuple[str, ...] = ("resid_post", "resid_pre", "mlp_out", "attn_out")
#: SAE sites that are not part of the captured residual stream (need their own hook).
_SUBLAYER_SITES: tuple[str, ...] = ("mlp_out", "attn_out")
#: Families whose HookedModel ``mlp_out`` / ``attn_out`` hooks sit *before* a
#: post-sublayer norm, so they are not the tensor SAELens / Gemma Scope
#: ``hook_mlp_out`` / ``hook_attn_out`` SAEs read (mirrors ``FeatureExtractor``).
_POST_NORM_FAMILIES: tuple[str, ...] = ("gemma2", "gemma3")


def _check_choice(name: str, value: str, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}")


def _work(t: Any) -> Any:
    """*t* in the working precision: float64 stays float64, everything else float32."""
    import torch

    return t if t.dtype == torch.float64 else t.to(torch.float32)


def normalise_sae_source(sae: Any) -> Any:
    """Accept an SAE, a mapping of SAEs, or a ``FeatureExtractor`` (its ``sae_map``).

    Returns ``None`` for ``None`` / an empty mapping / an extractor without SAEs.
    """
    if sae is None:
        return None
    if (
        not isinstance(sae, Mapping)
        and not hasattr(sae, "encode_torch")
        and hasattr(sae, "sae_map")
    ):
        sae = sae.sae_map
    if isinstance(sae, Mapping) and not sae:
        return None
    return sae


def lookup_sae(sae: Any, layer: int, site: str = "resid_post", name: str | None = None) -> Any:
    """Resolve the SAE for one hook point from *sae* (see :func:`normalise_sae_source`).

    A single SAE is returned as-is.  A mapping is searched by, in order: the
    extractor attachment name *name* (``Feature.meta["sae_name"]``, i.e. keys of
    ``FeatureExtractor.sae_map``), ``(layer, site)``, the hook name
    ``"blocks.{layer}.hook_{site}"`` and the bare ``layer``.  Returns ``None``
    when nothing matches.
    """
    sae = normalise_sae_source(sae)
    if not isinstance(sae, Mapping):
        return sae
    for key in (name, (int(layer), site), f"blocks.{int(layer)}.hook_{site}", int(layer)):
        if key is not None and key in sae:
            return sae[key]
    return None


def sae_output_scale(sae: Any, x: Any) -> Any:
    """``(N,)`` float32 factor mapping SAE-space decoder vectors to model units.

    Uses :meth:`SparseAutoencoder.encode_with_output_scale` when the SAE has
    it (input normalisation); 1 otherwise.  *x* is ``(N, D)`` (no grad needed).
    """
    import torch

    rows = x.detach().reshape(-1, x.shape[-1])
    dev = getattr(getattr(sae, "W_dec", None), "device", rows.device)
    fn = getattr(sae, "encode_with_output_scale", None)
    if fn is None:
        return torch.ones(rows.shape[0], dtype=torch.float32, device=dev)
    _, scale = fn(rows.to(torch.float32).cpu().numpy())
    flat = np.asarray(scale, dtype=np.float32).reshape(-1)
    if flat.size == 1 and rows.shape[0] != 1:
        flat = np.full(rows.shape[0], float(flat[0]), dtype=np.float32)
    return torch.as_tensor(flat, dtype=torch.float32, device=dev)


def site_capture_hooks(
    sites: Iterable[tuple[int, str]], store: dict[tuple[int, str], Any], *, differentiable: bool
) -> list[Any]:
    """:class:`ResidHook` observers for ``mlp_out`` / ``attn_out`` sites.

    Each hook records the ``(1, T, D)`` activation in ``store[(layer, site)]``.
    With *differentiable* it passes on a clone, so the stored tensor is the one
    that flows downstream (upstream of the logits) and can be differentiated
    against even when HookedModel presents a reshaped view.  Residual sites
    are skipped (they come from ``ForwardResult.resid_*_live``).
    """
    from LLmThoughtLens.models.hooked import ResidHook

    hooks = []
    for layer, site in sorted({(int(lyr), str(st)) for lyr, st in sites}):
        if site not in _SUBLAYER_SITES:
            continue

        def keep(h: Any, key: tuple[int, str] = (layer, site)) -> Any:
            if differentiable:
                out = h.clone()
                store[key] = out
                return out
            store[key] = h.detach()
            return None

        hooks.append(ResidHook(layer, keep, site=site))  # type: ignore[arg-type]
    return hooks


def site_tensor(res: Any, store: Mapping[tuple[int, str], Any], layer: int, site: str) -> Any:
    """The ``(1, T, D)`` activation at ``(layer, site)`` of one forward pass."""
    if site == "resid_post":
        return res.resid_post_live[layer]
    if site == "resid_pre":
        return res.resid_pre_live[layer]
    try:
        return store[(int(layer), site)]
    except KeyError:
        raise RuntimeError(
            f"site {site!r} of layer {layer} was not reached during the forward pass"
        ) from None


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttributionNode:
    """One node to attribute: a residual site ``(layer, token_idx)`` or an SAE feature.

    Attributes
    ----------
    layer:
        Block index (``resid_post`` / block write of that block, or the SAE layer).
    token_idx:
        Absolute prompt position.
    kind:
        ``"delta"`` (block write), ``"resid"`` (cumulative residual) or ``"sae"``.
    sae_feature_id:
        Dictionary index of the SAE feature (``kind="sae"`` only).
    sae_site:
        Hook site of ``layer`` the SAE reads (one of :data:`SAE_SITES`;
        default ``"resid_post"``), SAE nodes only.
    node_id:
        Optional id of the matching :class:`~LLmThoughtLens.circuits.graph.CircuitNode`.
    label:
        Optional human-readable label.
    sae_name:
        Extractor attachment name (``Feature.meta["sae_name"]``) used to pick
        the SAE from a mapping; ``None`` = resolve by hook point.
    """

    layer: int
    token_idx: int
    kind: str = "delta"
    sae_feature_id: int | None = None
    sae_site: str = "resid_post"
    node_id: int | None = None
    label: str = ""
    sae_name: str | None = None

    def __post_init__(self) -> None:
        _check_choice("kind", self.kind, (*NODE_KINDS, "sae"))
        if self.kind == "sae":
            if self.sae_feature_id is None:
                raise ValueError("an SAE node needs sae_feature_id")
            _check_choice("sae_site", self.sae_site, SAE_SITES)

    @property
    def key(self) -> tuple[Any, ...]:
        """Identity of the underlying model quantity (ignores id / label)."""
        if self.kind == "sae":
            return (
                "sae",
                self.layer,
                self.token_idx,
                self.sae_feature_id,
                self.sae_site,
                self.sae_name,
            )
        return (self.kind, self.layer, self.token_idx)

    @property
    def sae_key(self) -> SAEKey:
        """``(layer, sae_site, sae_name)`` — the SAE hook point of an SAE node."""
        return (int(self.layer), self.sae_site, self.sae_name)

    @property
    def site(self) -> tuple[int, str]:
        """``(layer, site)`` of the tensor this node lives in (its edge source)."""
        if self.kind == "sae":
            return (int(self.layer), self.sae_site)
        return (int(self.layer), "resid_post")

    @property
    def order(self) -> float:
        """Residual-stream position used to direct edges (see the module docstring)."""
        if self.kind == "sae" and self.sae_site == "resid_pre":
            return float(self.layer) - 1.0 if self.layer > 0 else -0.5
        return float(self.layer)

    @classmethod
    def from_feature(cls, feature: Feature, node_kind: str = "delta") -> AttributionNode:
        """Map an extractor :class:`Feature` to a node.

        Features whose ``meta["method"] == "sae"`` become SAE nodes
        (``meta["sae_feature_id"]`` / ``"sae_layer"`` / ``"sae_site"`` /
        ``"sae_name"`` when present, else ``feature.id`` / ``feature.layer`` /
        ``"resid_post"`` / ``None``).  Every other white-box feature is a
        residual node of *node_kind*.
        """
        meta = feature.meta or {}
        if meta.get("method") == "sae":
            name = meta.get("sae_name")
            return cls(
                layer=int(meta.get("sae_layer", feature.layer)),
                token_idx=int(feature.token_idx),
                kind="sae",
                sae_feature_id=int(meta.get("sae_feature_id", feature.id)),
                sae_site=str(meta.get("sae_site", "resid_post")),
                node_id=int(feature.id),
                label=feature.label,
                sae_name=None if name is None else str(name),
            )
        return cls(
            layer=int(feature.layer),
            token_idx=int(feature.token_idx),
            kind=node_kind,
            node_id=int(feature.id),
            label=feature.label,
        )

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "node_id": self.node_id,
            "layer": self.layer,
            "token_idx": self.token_idx,
            "kind": self.kind,
        }
        if self.kind == "sae":
            out["sae_feature_id"] = self.sae_feature_id
            out["sae_site"] = self.sae_site
            out["sae_name"] = self.sae_name
        return out


@dataclass
class MetricSpec:
    """The resolved scalar being attributed (fixed token ids, clean value)."""

    metric: str
    target_id: int
    target_token: str
    runner_up_id: int | None = None
    runner_up_token: str | None = None
    value: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "target_token_id": self.target_id,
            "target_token": self.target_token,
            "runner_up_token_id": self.runner_up_id,
            "runner_up_token": self.runner_up_token,
            "metric_value": float(self.value),
        }


@dataclass
class AttributionResult:
    """Everything one :meth:`GradientAttributor.attribute` call computed.

    Attributes
    ----------
    metric:
        The resolved :class:`MetricSpec` (target ids and the clean value).
    node_kind, baseline:
        Settings used for residual nodes.
    nodes:
        The requested nodes, in order.
    node_attr:
        ``(n,)`` attribution ``A`` of each requested node.
    node_value_norm:
        ``(n,)`` L2 norm of ``v - b`` (residual nodes) or ``|z - b|`` (SAE nodes).
    all_attr:
        ``(L, T)`` attribution of every residual node of ``node_kind``.
    input_attr:
        ``(T,)`` attribution of every token embedding (``resid_pre[0]``).
    edge_matrix:
        ``(n, n)`` node -> node edge weights, ``edge_matrix[i, j]`` = edge
        ``i -> j``; zero where ``layer_i >= layer_j`` or edges were not computed.
    input_edges:
        ``(T, n)`` input-token -> node edge weights.
    edges_computed:
        Whether edges were computed (``attribute(..., edges=True)``).
    sae_error_attr:
        ``{(layer, site, sae_name): (T,)}`` attribution of the SAE
        reconstruction error at every position, for each SAE hook point used.
    sae_all_abs, sae_all_sum:
        ``{(layer, site, sae_name): (T,)}`` sum of ``|A|`` / of ``A`` over every
        SAE feature per position (used for the SAE error-node accounting).
    node_grads:
        ``(n, D)`` d metric / d (the node's site tensor) at each node.
    node_coef:
        ``(n,)`` d metric / d z for SAE nodes (``scale_t * g . W_dec[:, f]``);
        ``0`` for residual nodes.  ``s_dst = node_coef * z`` for an SAE destination.
    """

    metric: MetricSpec
    node_kind: str
    baseline: str
    tokens: list[str]
    token_ids: list[int]
    nodes: list[AttributionNode]
    node_attr: np.ndarray
    node_value_norm: np.ndarray
    all_attr: np.ndarray
    input_attr: np.ndarray
    edge_matrix: np.ndarray
    input_edges: np.ndarray
    edges_computed: bool = False
    baseline_positions: list[int] = field(default_factory=list)
    sae_error_attr: dict[SAEKey, np.ndarray] = field(default_factory=dict)
    sae_all_abs: dict[SAEKey, np.ndarray] = field(default_factory=dict)
    sae_all_sum: dict[SAEKey, np.ndarray] = field(default_factory=dict)
    runtime_s: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)
    node_grads: np.ndarray | None = field(default=None, repr=False)
    node_coef: np.ndarray | None = field(default=None, repr=False)

    def top_residual_nodes(
        self, n: int = 10, exclude_positions: Iterable[int] = ()
    ) -> list[tuple[int, int, float]]:
        """``[(layer, token_idx, A), ...]`` — the *n* residual nodes with largest ``|A|``."""
        skip = {int(p) for p in exclude_positions}
        rows = [
            (int(lyr), int(tok), float(self.all_attr[lyr, tok]))
            for lyr in range(self.all_attr.shape[0])
            for tok in range(self.all_attr.shape[1])
            if tok not in skip
        ]
        rows.sort(key=lambda r: abs(r[2]), reverse=True)
        return rows[: max(0, int(n))]

    def edges(self, min_abs: float = 0.0) -> list[tuple[int, int, float]]:
        """Non-zero node -> node edges as ``(src_index, dst_index, weight)``."""
        out = []
        n = len(self.nodes)
        for i in range(n):
            for j in range(n):
                w = float(self.edge_matrix[i, j])
                if w != 0.0 and abs(w) >= min_abs:
                    out.append((i, j, w))
        return out


# ---------------------------------------------------------------------------
# Target resolution (shared with ActivationPatcher)
# ---------------------------------------------------------------------------


def _token_id_for(hooked: HookedModel, token: int | str) -> int:
    """Resolve an int id, or a string that tokenises to exactly one token."""
    if isinstance(token, (int, np.integer)):
        tid = int(token)
    else:
        enc = hooked.tokenizer(str(token), add_special_tokens=False)
        raw = enc["input_ids"] if isinstance(enc, Mapping) or hasattr(enc, "keys") else enc
        if hasattr(raw, "tolist"):
            raw = raw.tolist()
        ids = [int(i) for i in np.asarray(raw, dtype=np.int64).reshape(-1)]
        if len(ids) != 1:
            raise ValueError(
                f"target {token!r} tokenises to {len(ids)} tokens; pass a single-token string "
                "or a token id"
            )
        tid = ids[0]
    vocab = hooked.vocab_size
    if tid < 0 or (vocab and tid >= vocab):
        raise ValueError(f"token id {tid} is outside the vocabulary (size {vocab})")
    return tid


def resolve_target(
    hooked: HookedModel,
    last_logits: Any,
    metric: str = "logit",
    target: int | str | None = None,
    runner_up: int | str | None = None,
) -> MetricSpec:
    """Fix the token ids the metric reads (default: the model's own top-1 / top-2)."""
    import torch

    _check_choice("metric", metric, METRICS)
    logits = last_logits.detach().to(torch.float32).reshape(-1)
    tid = int(torch.argmax(logits)) if target is None else _token_id_for(hooked, target)
    rid: int | None = None
    if metric == "logit_diff":
        if runner_up is None:
            masked = logits.clone()
            masked[tid] = float("-inf")
            rid = int(torch.argmax(masked))
        else:
            rid = _token_id_for(hooked, runner_up)
        if rid == tid:
            raise ValueError("logit_diff needs a runner-up token different from the target")
    return MetricSpec(
        metric=metric,
        target_id=tid,
        target_token=hooked.token_str(tid),
        runner_up_id=rid,
        runner_up_token=None if rid is None else hooked.token_str(rid),
    )


def _replay_error(replayed: Any, reference: np.ndarray) -> float:
    """``max |replayed - reference| / max |reference|`` (``inf`` if shapes differ)."""
    import torch

    got = replayed.detach().to(torch.float32).cpu().numpy()
    if got.shape != reference.shape:
        return float("inf")
    ref = reference.astype(np.float32, copy=False)
    scale = float(np.max(np.abs(ref))) if ref.size else 0.0
    return float(np.max(np.abs(got - ref)) / (scale + 1e-12)) if ref.size else 0.0


def metric_from_logits(last_logits: Any, spec: MetricSpec) -> Any:
    """The metric as a differentiable torch scalar from ``(V,)`` last-position logits."""
    import torch

    logits = _work(last_logits).reshape(-1)
    if spec.metric == "logit":
        return logits[spec.target_id]
    if spec.metric == "logprob":
        return torch.log_softmax(logits, dim=-1)[spec.target_id]
    assert spec.runner_up_id is not None
    return logits[spec.target_id] - logits[spec.runner_up_id]


# ---------------------------------------------------------------------------
# GradientAttributor
# ---------------------------------------------------------------------------


@dataclass
class _Clean:
    """Clean forward pass with the residual stream on the autograd graph."""

    res: ForwardResult
    spec: MetricSpec
    metric: Any  # torch scalar (on the graph)
    values: Any  # (L, T, D) node values (detached, working precision)
    base: Any  # (L, 1, D) baseline
    grads: Any  # (L, T, D) d metric / d resid_post
    emb: Any  # (T, D) embeddings (detached)
    emb_base: Any  # (1, D)
    emb_grad: Any  # (T, D)
    positions: list[int]
    sites: dict[tuple[int, str], Any] = field(default_factory=dict)  # live (1, T, D)


class GradientAttributor:
    """Gradient (attribution-patching) attributions on a :class:`HookedModel`.

    Parameters
    ----------
    hooked:
        The model, e.g. ``HuggingFaceProvider(...).hooked``.
    metric:
        ``"logit"`` (default), ``"logprob"`` or ``"logit_diff"`` — see the module
        docstring.  Always read at the last prompt position.
    target:
        Target token (id, or a string that is exactly one token).  ``None``
        uses the model's own top-1 prediction.
    runner_up:
        Contrast token for ``"logit_diff"``; ``None`` uses the top token
        other than the target.
    node_kind:
        ``"delta"`` (block writes, default) or ``"resid"`` (cumulative residual).
    baseline:
        ``"zero"`` (default) or ``"mean"`` (per layer, over *baseline positions*).
    baseline_positions:
        Positions averaged for ``baseline="mean"``; ``None`` = every position
        not in *exclude_positions*.
    exclude_positions:
        Positions left out of the mean baseline (e.g. the attention sink the
        extractor excluded).
    sae:
        SAEs for SAE-feature nodes: one
        :class:`~LLmThoughtLens.features.sae.SparseAutoencoder` (used for every
        SAE node), a mapping (keys: extractor attachment name — e.g.
        ``FeatureExtractor.sae_map`` —, ``(layer, site)``, hook name
        ``"blocks.L.hook_site"`` or bare ``layer``; see :func:`lookup_sae`), or
        a ``FeatureExtractor`` (its ``sae_map`` is used).
    interventions:
        :class:`~LLmThoughtLens.features.intervention.FeatureIntervention` specs
        installed during every forward, so attributions describe the *same*
        (intervened) computation the provider ran.
    """

    def __init__(
        self,
        hooked: HookedModel,
        *,
        metric: str = "logit",
        target: int | str | None = None,
        runner_up: int | str | None = None,
        node_kind: str = "delta",
        baseline: str = "zero",
        baseline_positions: Sequence[int] | None = None,
        exclude_positions: Iterable[int] = (),
        sae: Any = None,
        interventions: Sequence[Any] | None = None,
    ) -> None:
        _check_choice("metric", metric, METRICS)
        _check_choice("node_kind", node_kind, NODE_KINDS)
        _check_choice("baseline", baseline, BASELINES)
        self.hooked = hooked
        self.metric = metric
        self.target = target
        self.runner_up = runner_up
        self.node_kind = node_kind
        self.baseline = baseline
        self.baseline_positions = (
            None if baseline_positions is None else [int(p) for p in baseline_positions]
        )
        self.exclude_positions = tuple(int(p) for p in exclude_positions)
        self.sae = normalise_sae_source(sae)
        self.interventions = list(interventions or [])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def sae_for(self, layer: int, site: str = "resid_post", name: str | None = None) -> Any:
        """The SAE for one hook point (see :func:`lookup_sae`); ``ValueError`` if none."""
        sae = lookup_sae(self.sae, layer, site, name)
        if sae is None:
            where = f"blocks.{layer}.hook_{site}" + (f" ({name!r})" if name else "")
            raise ValueError(
                f"SAE-feature node at {where} needs an SAE: pass "
                "GradientAttributor(sae=...) / CircuitTracer(sae=...), e.g. "
                "FeatureExtractor.sae_map"
            )
        return sae

    def check_nodes(self, nodes: Sequence[AttributionNode], n_layers: int | None = None) -> None:
        """Raise ``ValueError`` if a node cannot be attributed on this model.

        Checks the layer range, the node kind against ``node_kind``, that every
        SAE node resolves to an SAE and that its hook site exists for the
        model's family (``mlp_out`` / ``attn_out`` SAE nodes are refused on
        Gemma-2 / Gemma-3, whose post-sublayer norms put those hooks before the
        tensor the SAE was trained on).  (Token positions are checked by
        :meth:`attribute`.)
        """
        n_layers = self.hooked.n_layers if n_layers is None else int(n_layers)
        for nd in nodes:
            if not 0 <= nd.layer < n_layers:
                raise ValueError(f"node layer {nd.layer} is outside the {n_layers}-layer model")
            if nd.kind not in ("sae", self.node_kind):
                raise ValueError(
                    f"node kind {nd.kind!r} does not match the attributor's node_kind "
                    f"{self.node_kind!r}"
                )
            if nd.kind == "sae":
                self.sae_for(*nd.sae_key)
                if nd.sae_site in _SUBLAYER_SITES:
                    family = str(getattr(self.hooked, "family", ""))
                    if family in _POST_NORM_FAMILIES:
                        raise ValueError(
                            f"SAE-feature node at blocks.{nd.layer}.hook_{nd.sae_site}: for "
                            f"{family} the HookedModel {nd.sae_site} hook sits before the "
                            "post-sublayer norm, while SAELens / Gemma Scope "
                            f"hook_{nd.sae_site} reads after it, so the SAE's input cannot be "
                            "reproduced"
                        )
                    self.hooked.site_module(nd.layer, nd.sae_site)

    @contextlib.contextmanager
    def intervened(self) -> Any:
        """Install this attributor's interventions (if any) for the ``with`` block."""
        if not self.interventions:
            yield
            return
        from LLmThoughtLens.features.intervention import intervention_context

        with intervention_context(self.hooked.blocks, self.interventions):
            yield

    def _positions(self, n_tokens: int) -> list[int]:
        if self.baseline_positions is not None:
            pos = [p % n_tokens for p in self.baseline_positions if -n_tokens <= p < n_tokens]
        else:
            skip = {p % n_tokens for p in self.exclude_positions if -n_tokens <= p < n_tokens}
            pos = [t for t in range(n_tokens) if t not in skip]
        return sorted(set(pos)) or list(range(n_tokens))

    def _forward(
        self, prompt_or_ids: Any, sites: Iterable[tuple[int, str]] = ()
    ) -> tuple[ForwardResult, dict[tuple[int, str], Any]]:
        store: dict[tuple[int, str], Any] = {}
        hooks = site_capture_hooks(sites, store, differentiable=True)
        with self.intervened():
            res = self.hooked.forward(
                prompt_or_ids, hooks=hooks, grad=True, capture_attentions=False
            )
        return res, store

    def _clean(self, prompt_or_ids: Any, sites: Iterable[tuple[int, str]] = ()) -> _Clean:
        import torch

        sites = sorted(set(sites))
        res, store = self._forward(prompt_or_ids, sites)
        spec = resolve_target(
            self.hooked, res.logits[0, -1], self.metric, self.target, self.runner_up
        )
        metric = metric_from_logits(res.logits[0, -1], spec)
        spec.value = float(metric.detach())

        post = res.resid_post_live
        pre0 = res.resid_pre_live[0]
        if not metric.requires_grad:
            raise RuntimeError("the metric does not depend on the residual stream (no grad)")
        grads = torch.autograd.grad(metric, [*post, pre0], retain_graph=True, allow_unused=True)
        g_post = torch.stack(
            [
                _work(g[0] if g is not None else torch.zeros_like(p[0]))
                for g, p in zip(grads[:-1], post, strict=True)
            ]
        )
        g_emb = grads[-1]
        emb_grad = _work(g_emb[0] if g_emb is not None else torch.zeros_like(pre0[0]))

        resid_post = _work(res.resid("resid_post").detach())
        resid_pre = _work(res.resid("resid_pre").detach())
        values = resid_post - resid_pre if self.node_kind == "delta" else resid_post
        positions = self._positions(values.shape[1])
        emb = resid_pre[0]
        if self.baseline == "mean":
            base = values[:, positions].mean(dim=1, keepdim=True)
            emb_base = emb[positions].mean(dim=0, keepdim=True)
        else:
            base = torch.zeros_like(values[:, :1])
            emb_base = torch.zeros_like(emb[:1])
        return _Clean(
            res=res,
            spec=spec,
            metric=metric,
            values=values,
            base=base,
            grads=g_post,
            emb=emb,
            emb_base=emb_base,
            emb_grad=emb_grad,
            positions=positions,
            sites={key: site_tensor(res, store, *key) for key in sites},
        )

    # -- SAE helpers ---------------------------------------------------

    @staticmethod
    def _encode(sae: Any, x: Any) -> Any:
        """Differentiable SAE codes ``(N, F)`` for ``x (N, D)`` on the SAE's device."""
        import torch

        dev = getattr(getattr(sae, "W_enc", None), "device", None)
        xs = x.to(torch.float32)
        if dev is not None:
            xs = xs.to(dev)
        return sae.encode_torch(xs)

    @staticmethod
    def _decoder_col(sae: Any, fid: int) -> Any:
        import torch

        return sae.W_dec[:, int(fid)].detach().to(torch.float32)

    def _sae_site_stats(self, clean: _Clean, key: SAEKey) -> dict[str, Any]:
        """Codes, output scale, per-feature metric gradients and baselines at one SAE."""
        import torch

        layer, site, _name = key
        sae = self.sae_for(*key)
        live = clean.sites[(layer, site)]
        x = live[0].detach().to(torch.float32)
        (g_site,) = torch.autograd.grad(clean.metric, [live], retain_graph=True, allow_unused=True)
        g = (g_site[0] if g_site is not None else torch.zeros_like(live[0])).to(torch.float32)
        with torch.no_grad():
            z = self._encode(sae, x)  # (T, F)
            w_dec = sae.W_dec.detach().to(torch.float32)  # (D, F)
            scale = sae_output_scale(sae, x).to(w_dec.device)  # (T,)
            g_dev = g.to(w_dec.device)
            x_dev = x.to(w_dec.device)
            coef = (g_dev @ w_dec) * scale[:, None]  # (T, F) d metric / d z
            z_base = (
                z[clean.positions].mean(dim=0, keepdim=True)
                if self.baseline == "mean"
                else torch.zeros_like(z[:1])
            )
            attr = (z - z_base) * coef
            if hasattr(sae, "reconstruct_torch"):
                recon = sae.reconstruct_torch(x_dev)
            elif hasattr(sae, "decode_torch"):
                recon = sae.decode_torch(z)
            else:
                recon = z @ w_dec.T
            err_attr = ((x_dev - recon.to(torch.float32)) * g_dev).sum(-1)
        return {
            "z": z,
            "scale": scale,
            "coef": coef,
            "z_base": z_base,
            "attr": attr,
            "grad": g,
            "err_attr": err_attr,
            "all_abs": attr.abs().sum(-1),
            "all_sum": attr.sum(-1),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def attribute(
        self,
        prompt_or_ids: Any,
        nodes: Sequence[AttributionNode] | None = None,
        *,
        edges: bool = True,
        max_edge_targets: int | None = None,
        reference_activations: np.ndarray | None = None,
        add_top_nodes: int = 0,
    ) -> AttributionResult:
        """Attribute the metric to *nodes* (and, with ``edges=True``, node -> node).

        Parameters
        ----------
        prompt_or_ids:
            Anything :meth:`HookedModel.forward` accepts (prefer the exact
            token ids the provider ran).
        nodes:
            Nodes to attribute; ``None`` = no individual nodes (``all_attr`` /
            ``input_attr`` are always filled).
        edges:
            Compute node -> node and input -> node edges (one backward pass
            per destination node).
        max_edge_targets:
            Only compute incoming edges for the *n* destination nodes with the
            largest ``|A|`` (``None`` = all).
        reference_activations:
            Optional ``(L, T, D)`` residual stream the caller already has (e.g.
            ``ProviderOutput.activations``).  The replayed forward is compared
            against it and ``meta["replay_max_rel_error"]`` records
            ``max |replay - reference| / max |reference|`` (``inf`` on a shape
            mismatch), so callers can refuse to attribute a different computation.
        add_top_nodes:
            Append the *n* residual nodes (of this attributor's ``node_kind``)
            with the largest ``|A|`` that are not already in *nodes*, skipping
            ``exclude_positions``; they get ``node_id=None`` and
            ``meta["n_added_nodes"]`` records how many were added.
        """
        import torch

        t0 = time.perf_counter()
        node_list = list(nodes or [])
        self.check_nodes(node_list)
        sae_keys = sorted(
            {nd.sae_key for nd in node_list if nd.kind == "sae"},
            key=lambda k: (k[0], k[1], k[2] or ""),
        )
        clean = self._clean(prompt_or_ids, {(k[0], k[1]) for k in sae_keys})
        n_layers, n_tokens = int(clean.values.shape[0]), int(clean.values.shape[1])
        meta: dict[str, Any] = {
            "family": self.hooked.family,
            "n_layers": n_layers,
            "n_tokens": n_tokens,
        }
        if reference_activations is not None:
            meta["replay_max_rel_error"] = _replay_error(
                clean.res.resid("resid_post"), np.asarray(reference_activations)
            )
        for nd in node_list:
            if not (0 <= nd.layer < n_layers and 0 <= nd.token_idx < n_tokens):
                raise ValueError(
                    f"node (layer={nd.layer}, token={nd.token_idx}) is outside the "
                    f"{n_layers}-layer x {n_tokens}-token prompt"
                )

        with torch.no_grad():
            all_attr = ((clean.values - clean.base) * clean.grads).sum(-1)
            input_attr = ((clean.emb - clean.emb_base) * clean.emb_grad).sum(-1)

        if add_top_nodes > 0:
            present = {nd.key for nd in node_list}
            skip = {p % n_tokens for p in self.exclude_positions if -n_tokens <= p < n_tokens}
            flat = all_attr.abs().reshape(-1).cpu().numpy()
            added = 0
            for idx in np.argsort(-flat, kind="stable"):
                lyr, tok = divmod(int(idx), n_tokens)
                cand = AttributionNode(lyr, tok, kind=self.node_kind)
                if tok in skip or cand.key in present:
                    continue
                node_list.append(cand)
                present.add(cand.key)
                added += 1
                if added >= int(add_top_nodes):
                    break
            meta["n_added_nodes"] = added

        sae_stats = {key: self._sae_site_stats(clean, key) for key in sae_keys}

        n = len(node_list)
        node_attr = np.zeros(n, dtype=np.float64)
        node_norm = np.zeros(n, dtype=np.float64)
        node_coef = np.zeros(n, dtype=np.float64)
        d_model = int(clean.values.shape[-1])
        node_grads = np.zeros((n, d_model), dtype=np.float64)
        # Per-node source direction: what ablating the node removes from its site tensor.
        src_dirs: list[Any] = []
        for i, nd in enumerate(node_list):
            if nd.kind == "sae":
                st = sae_stats[nd.sae_key]
                fid = int(nd.sae_feature_id or 0)
                t = nd.token_idx
                dz = st["z"][t, fid] - st["z_base"][0, fid]
                node_attr[i] = float(st["attr"][t, fid])
                node_norm[i] = float(dz.abs())
                node_coef[i] = float(st["coef"][t, fid])
                node_grads[i] = st["grad"][t].cpu().numpy()
                col = self._decoder_col(self.sae_for(*nd.sae_key), fid)
                src_dirs.append(col * (dz * st["scale"][t]))
            else:
                dv = clean.values[nd.layer, nd.token_idx] - clean.base[nd.layer, 0]
                node_attr[i] = float(all_attr[nd.layer, nd.token_idx])
                node_norm[i] = float(dv.norm())
                node_grads[i] = clean.grads[nd.layer, nd.token_idx].cpu().numpy()
                src_dirs.append(dv)

        edge_matrix = np.zeros((n, n), dtype=np.float64)
        input_edges = np.zeros((n_tokens, n), dtype=np.float64)
        if edges and n:
            order = sorted(range(n), key=lambda j: abs(node_attr[j]), reverse=True)
            targets = order if max_edge_targets is None else order[: max(0, int(max_edge_targets))]
            for j in targets:
                self._edges_into(clean, node_list, j, src_dirs, sae_stats, edge_matrix, input_edges)

        return AttributionResult(
            metric=clean.spec,
            node_kind=self.node_kind,
            baseline=self.baseline,
            tokens=list(clean.res.tokens),
            token_ids=list(clean.res.token_ids),
            nodes=node_list,
            node_attr=node_attr,
            node_value_norm=node_norm,
            all_attr=all_attr.cpu().numpy().astype(np.float64),
            input_attr=input_attr.cpu().numpy().astype(np.float64),
            edge_matrix=edge_matrix,
            input_edges=input_edges,
            edges_computed=bool(edges),
            baseline_positions=list(clean.positions),
            sae_error_attr={
                k: v["err_attr"].cpu().numpy().astype(np.float64) for k, v in sae_stats.items()
            },
            sae_all_abs={
                k: v["all_abs"].cpu().numpy().astype(np.float64) for k, v in sae_stats.items()
            },
            sae_all_sum={
                k: v["all_sum"].cpu().numpy().astype(np.float64) for k, v in sae_stats.items()
            },
            runtime_s=time.perf_counter() - t0,
            meta=meta,
            node_grads=node_grads,
            node_coef=node_coef,
        )

    def _edges_into(
        self,
        clean: _Clean,
        nodes: list[AttributionNode],
        j: int,
        src_dirs: list[Any],
        sae_stats: dict[SAEKey, dict[str, Any]],
        edge_matrix: np.ndarray,
        input_edges: np.ndarray,
    ) -> None:
        """One backward pass: every edge into destination node *j*."""
        import torch

        res = clean.res
        dst = nodes[j]
        t = dst.token_idx
        # s_j = stopgrad(d metric / d v_j) . v_j  (v_j live, on the graph)
        if dst.kind == "sae":
            st = sae_stats[dst.sae_key]
            fid = int(dst.sae_feature_id or 0)
            live = clean.sites[(dst.layer, dst.sae_site)][0, t : t + 1]
            z = self._encode(self.sae_for(*dst.sae_key), live)[0, fid]
            s_j = st["coef"][t, fid].detach() * z
        else:
            post = res.resid_post_live[dst.layer][0, t]
            v = post - res.resid_pre_live[dst.layer][0, t] if dst.kind == "delta" else post
            s_j = (clean.grads[dst.layer, t].to(v.dtype).detach() * v).sum()
        if not getattr(s_j, "requires_grad", False):
            return

        sources = [i for i, nd in enumerate(nodes) if i != j and nd.order < dst.order]
        tensors: list[Any] = []
        index: dict[int, int] = {}

        def slot(tensor: Any) -> int:
            key = id(tensor)
            if key not in index:
                index[key] = len(tensors)
                tensors.append(tensor)
            return index[key]

        src_slot: dict[int, int] = {}
        for i in sources:
            nd = nodes[i]
            if nd.kind == "sae":
                src_slot[i] = slot(clean.sites[nd.site])
            else:
                src_slot[i] = slot(res.resid_post_live[nd.layer])
        emb_slot = slot(res.resid_pre_live[0])

        grads = torch.autograd.grad(s_j, tensors, retain_graph=True, allow_unused=True)
        with torch.no_grad():
            for i in sources:
                g = grads[src_slot[i]]
                if g is None:
                    continue
                d = src_dirs[i]
                gi = g[0, nodes[i].token_idx].to(device=d.device, dtype=d.dtype)
                edge_matrix[i, j] = float((gi * d).sum())
            g_emb = grads[emb_slot]
            if g_emb is not None:
                ge = g_emb[0].to(clean.emb.dtype)
                input_edges[:, j] = ((clean.emb - clean.emb_base) * ge).sum(-1).cpu().numpy()

    def node_attributions(self, prompt_or_ids: Any) -> AttributionResult:
        """Attribution of every residual node and token embedding (no edges, one backward)."""
        return self.attribute(prompt_or_ids, None, edges=False)
