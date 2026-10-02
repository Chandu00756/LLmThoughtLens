"""Activation patching — measure what gradient attributions only predict.

:class:`ActivationPatcher` runs *real* ablations through
:class:`~LLmThoughtLens.models.hooked.HookedModel` hooks and measures the
change in the same target metric :class:`~LLmThoughtLens.circuits.attribution.GradientAttributor`
attributes (see that module for the metric / node / baseline definitions):

``measured effect = metric(clean) - metric(node ablated)``

so a positive effect means the node *supported* the metric, exactly the sign
convention of the attribution ``A``.  Ablations:

* ``"delta"`` node ``(l, t)`` — block ``l``'s write at position ``t`` is replaced
  by the baseline: ``resid_post[l][t] := resid_pre[l][t] + b`` (``b = 0`` or
  the clean per-layer mean write).  ``resid_pre[l][t]`` is read live, so
  joint ablations of several nodes compose correctly.
* ``"resid"`` node — ``resid_post[l][t] := b``.
* SAE feature — its decoder contribution is removed at its hook site
  (``resid_post`` / ``resid_pre`` / ``mlp_out`` / ``attn_out``):
  ``x := x - (z_f(x) - b_f) * scale(x) * W_dec[:, f]`` with ``z_f`` and the
  SAE's output scale read live.

:meth:`ActivationPatcher.faithfulness` ablates the top-``k`` nodes one at a
time and reports Spearman / Pearson correlation between predicted
attribution and measured effect: this is how much the linearised graph can
be trusted for *this* prompt.  :meth:`ActivationPatcher.edge_check` does the
same for edges: ablate the source, measure the change in the destination's
own output-relevant scalar ``s_j``.  :func:`attribution_faithfulness` is the
one-call version used by the benchmark harness.

torch is imported lazily.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from LLmThoughtLens.circuits.attribution import (
    METRICS,
    NODE_KINDS,
    AttributionNode,
    AttributionResult,
    GradientAttributor,
    MetricSpec,
    _work,
    metric_from_logits,
    resolve_target,
    sae_output_scale,
    site_capture_hooks,
    site_tensor,
)

if TYPE_CHECKING:
    from LLmThoughtLens.circuits.graph import AttributionGraph
    from LLmThoughtLens.models.hooked import HookedModel, ResidHook

__all__ = [
    "ActivationPatcher",
    "FaithfulnessReport",
    "attribution_faithfulness",
    "pearson",
    "spearman",
]


# ---------------------------------------------------------------------------
# Correlation helpers (NumPy only — no scipy dependency)
# ---------------------------------------------------------------------------


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank), 1-based like ``scipy.stats.rankdata``."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def pearson(x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray) -> float:
    """Pearson correlation; ``nan`` when n < 2 or either side is constant."""
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.shape != b.shape or a.size < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    if denom == 0.0:
        return float("nan")
    return float(np.clip((a * b).sum() / denom, -1.0, 1.0))


def spearman(x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray) -> float:
    """Spearman rank correlation (Pearson of average ranks); ``nan`` when undefined."""
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.shape != b.shape or a.size < 2:
        return float("nan")
    return pearson(_rankdata(a), _rankdata(b))


def _json_float(v: float | None) -> float | None:
    return None if v is None or not math.isfinite(float(v)) else float(v)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class FaithfulnessReport:
    """Predicted attribution vs measured ablation effect for the top-``k`` nodes.

    Attributes
    ----------
    spearman, pearson:
        Correlation between ``predicted`` and ``measured`` (``nan`` if undefined).
    n:
        Number of nodes actually ablated.
    k:
        Requested number of nodes.
    method:
        ``"zero_ablation"`` or ``"mean_ablation"``.
    metric, target_token, clean_metric:
        The metric that was measured (identical to the attributed one).
    sign_agreement:
        Fraction of nodes whose measured effect has the predicted sign.
    node_kind:
        Residual node kind of the attribution (``"delta"`` / ``"resid"``).
    nodes:
        Per node: ``node_id``, ``layer``, ``token_idx``, ``kind`` (plus the
        SAE fields for SAE nodes), ``predicted``, ``measured``.
    runtime_s:
        Wall time of the ablations.
    """

    spearman: float
    pearson: float
    n: int
    k: int
    method: str
    metric: str
    target_token: str
    clean_metric: float
    sign_agreement: float
    node_kind: str = "delta"
    nodes: list[dict[str, Any]] = field(default_factory=list)
    runtime_s: float = 0.0

    @property
    def predicted(self) -> np.ndarray:
        return np.array([r["predicted"] for r in self.nodes], dtype=np.float64)

    @property
    def measured(self) -> np.ndarray:
        return np.array([r["measured"] for r in self.nodes], dtype=np.float64)

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe dict (undefined correlations become ``None``)."""
        return {
            "spearman": _json_float(self.spearman),
            "pearson": _json_float(self.pearson),
            "n": int(self.n),
            "k": int(self.k),
            "method": self.method,
            "metric": self.metric,
            "target_token": self.target_token,
            "clean_metric": float(self.clean_metric),
            "sign_agreement": _json_float(self.sign_agreement),
            "node_kind": self.node_kind,
            "runtime_s": float(self.runtime_s),
            "nodes": [dict(r) for r in self.nodes],
        }


# ---------------------------------------------------------------------------
# Patcher
# ---------------------------------------------------------------------------


@dataclass
class _CleanRun:
    """Clean (unablated) pass: metric value, residual stream, SAE sites, baselines."""

    spec: MetricSpec
    metric: float
    values: Any  # (L, T, D) node values of the patcher's node_kind, working precision
    base: Any  # (L, 1, D) baseline of the patcher's node_kind
    resid_post: Any  # (L, T, D)
    resid_pre: Any  # (L, T, D)
    positions: list[int]
    sites: dict[tuple[int, str], Any] = field(default_factory=dict)  # (T, D) per SAE site
    baseline: str = "zero"

    def base_for(self, kind: str, layer: int) -> Any:
        """``(D,)`` ablation baseline of a ``"delta"`` / ``"resid"`` node at *layer*."""
        import torch

        if self.baseline != "mean":
            return torch.zeros_like(self.resid_post[layer, 0])
        if kind == "delta":
            vals = self.resid_post[layer] - self.resid_pre[layer]
        else:
            vals = self.resid_post[layer]
        return vals[self.positions].mean(dim=0)

    def site(self, layer: int, site: str) -> Any:
        """``(T, D)`` clean activation at ``(layer, site)``."""
        if site == "resid_post":
            return self.resid_post[layer]
        if site == "resid_pre":
            return self.resid_pre[layer]
        return self.sites[(int(layer), site)]


def _sae_sites(nodes: Iterable[AttributionNode]) -> set[tuple[int, str]]:
    return {nd.site for nd in nodes if nd.kind == "sae"}


class ActivationPatcher:
    """Real ablations on a :class:`HookedModel`, scored with the attribution metric.

    Parameters mirror :class:`~LLmThoughtLens.circuits.attribution.GradientAttributor`
    (use :meth:`from_attributor` to copy them, so predicted and measured
    effects describe the same metric, nodes and baseline).
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
        exclude_positions: Sequence[int] = (),
        sae: Any = None,
        interventions: Sequence[Any] | None = None,
    ) -> None:
        # Validation + SAE / intervention plumbing are shared with the attributor.
        self._attr = GradientAttributor(
            hooked,
            metric=metric,
            target=target,
            runner_up=runner_up,
            node_kind=node_kind,
            baseline=baseline,
            baseline_positions=baseline_positions,
            exclude_positions=exclude_positions,
            sae=sae,
            interventions=interventions,
        )
        self.hooked = hooked

    @classmethod
    def from_attributor(cls, attributor: GradientAttributor) -> ActivationPatcher:
        """A patcher with exactly the attributor's metric / node / baseline settings."""
        return cls(
            attributor.hooked,
            metric=attributor.metric,
            target=attributor.target,
            runner_up=attributor.runner_up,
            node_kind=attributor.node_kind,
            baseline=attributor.baseline,
            baseline_positions=attributor.baseline_positions,
            exclude_positions=attributor.exclude_positions,
            sae=attributor.sae,
            interventions=attributor.interventions,
        )

    @property
    def metric(self) -> str:
        return self._attr.metric

    @property
    def node_kind(self) -> str:
        return self._attr.node_kind

    @property
    def baseline(self) -> str:
        return self._attr.baseline

    @property
    def method(self) -> str:
        return f"{self.baseline}_ablation"

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _run(
        self,
        prompt_or_ids: Any,
        hooks: Sequence[ResidHook] = (),
        capture: bool = False,
        sites: Iterable[tuple[int, str]] = (),
    ) -> tuple[Any, dict[tuple[int, str], Any]]:
        store: dict[tuple[int, str], Any] = {}
        # Observers go last so they see the activation after any ablation at that site.
        all_hooks = [*hooks, *site_capture_hooks(sites, store, differentiable=False)]
        with self._attr.intervened():
            res = self.hooked.forward(
                prompt_or_ids, hooks=all_hooks, capture=capture, capture_attentions=False
            )
        return res, store

    def _site_values(
        self, res: Any, store: dict[tuple[int, str], Any], sites: Iterable[tuple[int, str]]
    ) -> dict[tuple[int, str], Any]:
        return {
            key: _work(site_tensor(res, store, *key)[0].detach())
            for key in sites
            if key[1] not in ("resid_pre", "resid_post")
        }

    def clean_run(
        self,
        prompt_or_ids: Any,
        spec: MetricSpec | None = None,
        nodes: Iterable[AttributionNode] = (),
    ) -> _CleanRun:
        """Clean forward: resolved metric spec, its value and per-layer baselines.

        *nodes* names the SAE nodes that will be ablated, so their
        ``mlp_out`` / ``attn_out`` sites are captured too.
        """
        import torch

        sites = _sae_sites(nodes)
        res, store = self._run(prompt_or_ids, capture=True, sites=sites)
        if spec is None:
            a = self._attr
            spec = resolve_target(self.hooked, res.logits[0, -1], a.metric, a.target, a.runner_up)
        value = float(metric_from_logits(res.logits[0, -1], spec))
        post = _work(res.resid("resid_post"))
        pre = _work(res.resid("resid_pre"))
        values = post - pre if self.node_kind == "delta" else post
        positions = self._attr._positions(int(values.shape[1]))
        if self.baseline == "mean":
            base = values[:, positions].mean(dim=1, keepdim=True)
        else:
            base = torch.zeros_like(values[:, :1])
        spec = MetricSpec(**{**spec.__dict__, "value": value})
        return _CleanRun(
            spec=spec,
            metric=value,
            values=values,
            base=base,
            resid_post=post,
            resid_pre=pre,
            positions=positions,
            sites=self._site_values(res, store, sites),
            baseline=self.baseline,
        )

    def metric_value(
        self, prompt_or_ids: Any, spec: MetricSpec, hooks: Sequence[ResidHook] = ()
    ) -> float:
        """The metric under *hooks* (token ids fixed by *spec*)."""
        res, _ = self._run(prompt_or_ids, hooks=hooks)
        return float(metric_from_logits(res.logits[0, -1], spec))

    # ------------------------------------------------------------------
    # Ablation hooks
    # ------------------------------------------------------------------

    def ablation_hooks(self, node: AttributionNode, clean: _CleanRun) -> list[ResidHook]:
        """:class:`ResidHook` specs that ablate *node* to its baseline."""
        import torch

        from LLmThoughtLens.models.hooked import ResidHook

        layer, t = int(node.layer), int(node.token_idx)
        if node.kind == "sae":
            sae = self._attr.sae_for(*node.sae_key)
            fid = int(node.sae_feature_id or 0)
            col = GradientAttributor._decoder_col(sae, fid)
            z_base = 0.0
            if self.baseline == "mean":
                with torch.no_grad():
                    z_all = GradientAttributor._encode(sae, clean.site(layer, node.sae_site))
                z_base = float(z_all[clean.positions, fid].mean())

            def remove_feature(h: Any) -> Any:
                rows = h[0]  # (k, D): the selected position(s), live
                z = GradientAttributor._encode(sae, rows)[:, fid]  # (k,)
                scale = sae_output_scale(sae, rows).to(z.device)
                delta = ((z - z_base) * scale).unsqueeze(-1) * col.to(z.device)
                return h - delta.to(device=h.device, dtype=h.dtype).unsqueeze(0)

            return [ResidHook(layer, remove_feature, site=node.sae_site, positions=[t])]  # type: ignore[arg-type]

        if node.kind not in NODE_KINDS:
            raise ValueError(f"cannot ablate node kind {node.kind!r}")
        base = clean.base_for(node.kind, layer)
        if node.kind == "resid":

            def set_resid(h: Any) -> Any:
                return base.to(device=h.device, dtype=h.dtype).expand_as(h)

            return [ResidHook(layer, set_resid, site="resid_post", positions=[t])]

        seen: dict[str, Any] = {}

        def observe_pre(h: Any) -> None:
            seen["pre"] = h
            return None

        def replace_write(h: Any) -> Any:
            return seen["pre"].to(h.dtype) + base.to(device=h.device, dtype=h.dtype)

        return [
            ResidHook(layer, observe_pre, site="resid_pre", positions=[t]),
            ResidHook(layer, replace_write, site="resid_post", positions=[t]),
        ]

    def ablate(
        self,
        prompt_or_ids: Any,
        nodes: AttributionNode | Sequence[AttributionNode],
        *,
        clean: _CleanRun | None = None,
    ) -> float:
        """Measured effect ``metric(clean) - metric(all *nodes* ablated jointly)``."""
        node_list = [nodes] if isinstance(nodes, AttributionNode) else list(nodes)
        clean = clean or self.clean_run(prompt_or_ids, nodes=node_list)
        hooks = [h for nd in node_list for h in self.ablation_hooks(nd, clean)]
        return clean.metric - self.metric_value(prompt_or_ids, clean.spec, hooks)

    def node_effects(
        self,
        prompt_or_ids: Any,
        nodes: Sequence[AttributionNode],
        *,
        clean: _CleanRun | None = None,
    ) -> np.ndarray:
        """Measured effect of ablating each node on its own."""
        clean = clean or self.clean_run(prompt_or_ids, nodes=nodes)
        return np.array(
            [self.ablate(prompt_or_ids, nd, clean=clean) for nd in nodes], dtype=np.float64
        )

    # ------------------------------------------------------------------
    # Faithfulness
    # ------------------------------------------------------------------

    def _from_graph(self, graph: AttributionGraph) -> tuple[list[AttributionNode], list[float]]:
        nodes: list[AttributionNode] = []
        predicted: list[float] = []
        for n in graph.nodes():
            meta = n.meta
            if n.node_type in ("input_token", "output_token", "error"):
                continue
            if "attribution" not in meta or "node_kind" not in meta:
                continue
            kind = str(meta["node_kind"])
            name = meta.get("sae_name")
            nodes.append(
                AttributionNode(
                    layer=int(meta.get("sae_layer", n.layer)) if kind == "sae" else int(n.layer),
                    token_idx=int(n.token_idx),
                    kind=kind,
                    sae_feature_id=meta.get("sae_feature_id"),
                    sae_site=str(meta.get("sae_site", "resid_post")),
                    node_id=int(n.id),
                    label=n.label,
                    sae_name=None if name is None else str(name),
                )
            )
            predicted.append(float(meta["attribution"]))
        return nodes, predicted

    def faithfulness(
        self,
        source: AttributionResult | AttributionGraph | Sequence[tuple[AttributionNode, float]],
        k: int = 10,
        prompt: Any = None,
    ) -> FaithfulnessReport:
        """Ablate the top-*k* nodes (by ``|predicted|``) and correlate with prediction.

        Parameters
        ----------
        source:
            An :class:`AttributionResult`, a gradient :class:`AttributionGraph`
            (feature nodes carrying ``meta["attribution"]``; prompt, token ids
            and target come from ``graph.meta``), or ``(node, predicted)`` pairs.
        k:
            Number of nodes to ablate (each in its own forward pass).
        prompt:
            Prompt / token ids; required for ``(node, predicted)`` pairs,
            otherwise taken from *source*.
        """
        from LLmThoughtLens.circuits.graph import AttributionGraph

        t0 = time.perf_counter()
        spec: MetricSpec | None = None
        if isinstance(source, AttributionResult):
            nodes, predicted = list(source.nodes), [float(a) for a in source.node_attr]
            prompt = prompt if prompt is not None else source.token_ids
            spec = source.metric
        elif isinstance(source, AttributionGraph):
            nodes, predicted = self._from_graph(source)
            meta = source.meta
            if prompt is None:
                prompt = meta.get("token_ids") or meta.get("prompt")
            if "target_token_id" in meta:
                spec = MetricSpec(
                    metric=str(meta.get("metric", self.metric)),
                    target_id=int(meta["target_token_id"]),
                    target_token=str(meta.get("target_token", "")),
                    runner_up_id=meta.get("runner_up_token_id"),
                    runner_up_token=meta.get("runner_up_token"),
                )
        else:
            pairs = list(source)
            nodes = [p[0] for p in pairs]
            predicted = [float(p[1]) for p in pairs]
        if prompt is None or (isinstance(prompt, (list, tuple)) and not prompt):
            raise ValueError("faithfulness needs the prompt (or token ids) that was attributed")
        if spec is not None and spec.metric not in METRICS:
            raise ValueError(f"unknown metric {spec.metric!r} in attribution source")

        order = sorted(range(len(nodes)), key=lambda i: abs(predicted[i]), reverse=True)
        chosen = order[: max(0, int(k))]
        clean = self.clean_run(prompt, spec=spec, nodes=[nodes[i] for i in chosen])
        rows: list[dict[str, Any]] = []
        for i in chosen:
            nd = nodes[i]
            measured = self.ablate(prompt, nd, clean=clean)
            rows.append({**nd.as_dict(), "predicted": predicted[i], "measured": float(measured)})

        pred = np.array([r["predicted"] for r in rows], dtype=np.float64)
        meas = np.array([r["measured"] for r in rows], dtype=np.float64)
        signs = float(np.mean(np.sign(pred) == np.sign(meas))) if rows else float("nan")
        return FaithfulnessReport(
            spearman=spearman(pred, meas),
            pearson=pearson(pred, meas),
            n=len(rows),
            k=int(k),
            method=self.method,
            metric=clean.spec.metric,
            target_token=clean.spec.target_token,
            clean_metric=clean.metric,
            sign_agreement=signs,
            node_kind=self.node_kind,
            nodes=rows,
            runtime_s=time.perf_counter() - t0,
        )

    # ------------------------------------------------------------------
    # Edge spot-check
    # ------------------------------------------------------------------

    def _dst_scalar(self, result: AttributionResult, j: int, run: _CleanRun) -> float:
        """``s_j = stopgrad(d metric / d v_j) . v_j`` evaluated on one (ablated) run."""
        import torch

        nd = result.nodes[j]
        if nd.kind == "sae":
            assert result.node_coef is not None
            sae = self._attr.sae_for(*nd.sae_key)
            x = run.site(nd.layer, nd.sae_site)[nd.token_idx : nd.token_idx + 1]
            with torch.no_grad():
                z = GradientAttributor._encode(sae, x)[0, int(nd.sae_feature_id or 0)]
            return float(result.node_coef[j]) * float(z)
        assert result.node_grads is not None
        g = torch.as_tensor(result.node_grads[j], dtype=torch.float64)
        post = run.resid_post[nd.layer, nd.token_idx]
        v = post - run.resid_pre[nd.layer, nd.token_idx] if nd.kind == "delta" else post
        return float((g * v.to(torch.float64).cpu()).sum())

    def edge_check(
        self,
        result: AttributionResult,
        pairs: Sequence[tuple[int, int]] | None = None,
        k: int = 5,
        prompt: Any = None,
    ) -> list[dict[str, Any]]:
        """Ablate edge sources and measure the change in each destination's ``s_j``.

        Parameters
        ----------
        result:
            An :class:`AttributionResult` computed with ``edges=True``.
        pairs:
            ``(src_index, dst_index)`` pairs into ``result.nodes``; default: the
            *k* node -> node edges with the largest ``|weight|``.

        Returns
        -------
        list of dict
            ``src``, ``dst`` (indices), ``predicted`` (edge weight) and
            ``measured`` (``s_dst(clean) - s_dst(src ablated)``).
        """
        if not result.edges_computed:
            raise ValueError("edge_check needs an AttributionResult computed with edges=True")
        if pairs is None:
            ranked = sorted(result.edges(), key=lambda e: abs(e[2]), reverse=True)
            pairs = [(i, j) for i, j, _ in ranked[: max(0, int(k))]]
        prompt = prompt if prompt is not None else result.token_ids
        involved = [result.nodes[i] for pair in pairs for i in pair]
        clean = self.clean_run(prompt, spec=result.metric, nodes=involved)
        sites = _sae_sites(involved)
        out: list[dict[str, Any]] = []
        for i, j in pairs:
            hooks = self.ablation_hooks(result.nodes[i], clean)
            res, store = self._run(prompt, hooks=hooks, capture=True, sites=sites)
            ablated = _CleanRun(
                spec=clean.spec,
                metric=float("nan"),
                values=clean.values,
                base=clean.base,
                resid_post=_work(res.resid("resid_post")),
                resid_pre=_work(res.resid("resid_pre")),
                positions=clean.positions,
                sites=self._site_values(res, store, sites),
                baseline=clean.baseline,
            )
            s_clean = self._dst_scalar(result, j, clean)
            s_abl = self._dst_scalar(result, j, ablated)
            out.append(
                {
                    "src": int(i),
                    "dst": int(j),
                    "predicted": float(result.edge_matrix[i, j]),
                    "measured": float(s_clean - s_abl),
                }
            )
        return out


# ---------------------------------------------------------------------------
# One-call faithfulness (benchmark hook)
# ---------------------------------------------------------------------------


def attribution_faithfulness(
    provider: Any,
    prompt: str,
    *,
    k: int = 10,
    metric: str = "logit",
    node_kind: str = "delta",
    baseline: str = "zero",
    exclude_positions: Sequence[int] = (0,),
) -> dict[str, float | None]:
    """Faithfulness of gradient attribution on *prompt*: predicted vs patched effects.

    Attributes the metric to every residual node of *node_kind*, takes the
    *k* nodes with the largest ``|A|`` (skipping *exclude_positions*), ablates
    each one for real and correlates.  This is the implementation the
    benchmark's ``attribution_faithfulness`` metric looks up
    (``fn(provider, prompt) -> Mapping[str, float]``).

    Parameters
    ----------
    provider:
        A provider with ``supports_gradients`` (e.g. ``HuggingFaceProvider``)
        or a :class:`HookedModel`.
    prompt:
        The prompt to attribute.
    k:
        Number of nodes ablated.
    metric, node_kind, baseline:
        As for :class:`GradientAttributor`.
    exclude_positions:
        Positions whose nodes are not ranked.  Default ``(0,)``, matching the
        extractor's usual attention-sink exclusion (position 0 of GPT-2-style
        and BOS-prefixed models); pass ``()`` to rank every position.
        Ablations far outside the linear regime (e.g. zero-ablating GPT-2's
        very large layer-0 writes) lower the scores; they are measured, not
        hidden.

    Returns
    -------
    dict
        ``spearman``, ``pearson``, ``sign_agreement`` (``None`` when
        undefined), ``n`` (nodes ablated), ``clean_metric`` and ``runtime_s``.

    Raises
    ------
    ValueError
        If *provider* exposes no differentiable model.
    """
    from LLmThoughtLens.models.hooked import HookedModel

    if isinstance(provider, HookedModel):
        hooked = provider
    elif getattr(provider, "supports_gradients", False):
        hooked = provider.hooked
    else:
        raise ValueError(
            f"{type(provider).__name__} exposes no differentiable model; attribution "
            "faithfulness needs a white-box provider with gradients (HuggingFaceProvider)"
        )
    t0 = time.perf_counter()
    ga = GradientAttributor(
        hooked,
        metric=metric,
        node_kind=node_kind,
        baseline=baseline,
        exclude_positions=exclude_positions,
    )
    result = ga.attribute(prompt, None, edges=False, add_top_nodes=int(k))
    report = ActivationPatcher.from_attributor(ga).faithfulness(result, k=int(k))
    return {
        "spearman": _json_float(report.spearman),
        "pearson": _json_float(report.pearson),
        "sign_agreement": _json_float(report.sign_agreement),
        "n": float(report.n),
        "clean_metric": float(report.clean_metric),
        "runtime_s": float(time.perf_counter() - t0),
    }
