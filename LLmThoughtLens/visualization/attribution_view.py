"""Attribution method, edge semantics and faithfulness — one honest summary for every surface.

Every renderer (HTML report, graph figure, dashboard payload, CLI, TUI) reads
the same JSON-safe summary built by :func:`attribution_summary` from an
:class:`~LLmThoughtLens.circuits.graph.AttributionGraph`'s ``meta``, so the
wording about *what the edges mean* is defined in exactly one place:

* ``causal_linearised`` — gradient attribution (``grad_x_act``) on a local
  model: each number is a first-order (linearised) estimate of what ablating
  the node would do to the target metric.  It is causal in kind but an
  *estimate*, not a measured ablation.
* ``correlational`` — activation flow (the mock provider, or a white-box
  trace without gradients): a co-activation heuristic between consecutive
  layers.  Not a causal measurement.
* ``causal_input_masking`` — black-box API models: each edge is the measured
  change in the output probability when one input token is masked.  Causal
  for the *input*, but no internals are observed.

Faithfulness (``CircuitTracer(validate=k)``) compares the linearised
predictions with *real* ablations of the top-``k`` nodes.  It is always
reported as Spearman **and** Pearson **and** ``n`` **and** sign agreement,
with :data:`FAITHFULNESS_CAVEAT` — never as a single cherry-picked number.
"""

from __future__ import annotations

import html
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from LLmThoughtLens.circuits.graph import AttributionGraph

__all__ = [
    "EDGE_SEMANTICS",
    "FAITHFULNESS_CAVEAT",
    "attribution_summary",
    "attribution_html",
    "edge_semantics_label",
    "format_faithfulness",
    "graph_title",
]

#: ``edge_semantics`` -> (short badge label, one-sentence plain-language meaning).
EDGE_SEMANTICS: dict[str, tuple[str, str]] = {
    "causal_linearised": (
        "causal (linearised estimate)",
        "Gradient x activation: each weight is a first-order estimate of how much the "
        "target metric would change if that node were ablated. Causal in kind, but an "
        "estimate - not a measured ablation.",
    ),
    "correlational": (
        "correlational",
        "Activation flow: a co-activation heuristic between consecutive layers computed "
        "from real activations. It is not a causal measurement.",
    ),
    "causal_input_masking": (
        "causal (input masking)",
        "Each edge is the measured change in the output probability when one input token "
        "is masked. Causal for the input tokens; the model's internals are not observed.",
    ),
}

#: One-line plain-language caveat shown next to every faithfulness number.
FAITHFULNESS_CAVEAT = (
    "Predicted = linearised (gradient x activation) estimate; measured = real ablation of "
    "the same node. Nonlinear components (e.g. GPT-2's large layer-0 writes) can make them "
    "disagree, so read Spearman, Pearson, n and sign agreement together."
)

_FAITH_KEYS = (
    "spearman",
    "pearson",
    "n",
    "k",
    "sign_agreement",
    "method",
    "metric",
    "target_token",
    "clean_metric",
    "node_kind",
    "runtime_s",
)


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    f = float(value)
    return f if math.isfinite(f) else None


def edge_semantics_label(semantics: str | None) -> str:
    """Short badge text for *semantics* (``"unknown"`` when not recorded)."""
    if not semantics:
        return "unknown"
    return EDGE_SEMANTICS.get(semantics, (semantics, ""))[0]


def graph_title(meta: dict[str, Any], evidence_kind: str = "") -> str:
    """Figure title that states the edge semantics truthfully."""
    semantics = meta.get("edge_semantics")
    if semantics == "causal_linearised":
        metric = meta.get("metric")
        suffix = f", metric {metric}" if metric else ""
        return f"Attribution graph - causal, linearised estimate (gradient x activation{suffix})"
    if semantics == "correlational":
        return "Attribution graph - correlational (activation flow, not causal)"
    if semantics == "causal_input_masking" or evidence_kind == "black_box":
        return "Input -> output attribution - causal input masking (internals not observed)"
    return "Attribution graph"


def _faithfulness(meta: dict[str, Any]) -> dict[str, Any] | None:
    raw = meta.get("faithfulness")
    if not isinstance(raw, dict):
        return None
    out: dict[str, Any] = {k: raw.get(k) for k in _FAITH_KEYS if k in raw}
    for key in ("spearman", "pearson", "sign_agreement", "clean_metric", "runtime_s"):
        if key in out:
            out[key] = _finite(out[key])
    nodes = raw.get("nodes")
    if isinstance(nodes, list):
        out["nodes"] = [
            {
                k: n.get(k)
                for k in ("node_id", "layer", "token_idx", "kind", "predicted", "measured")
                if k in n
            }
            for n in nodes
            if isinstance(n, dict)
        ]
    out["caveat"] = FAITHFULNESS_CAVEAT
    return out


def format_faithfulness(faith: dict[str, Any] | None) -> str:
    """``"Spearman 0.53 | Pearson -0.21 | n=10 | sign agreement 70%"`` (all four, always)."""
    if not faith:
        return "not validated"

    def corr(v: Any) -> str:
        f = _finite(v)
        return "undefined" if f is None else f"{f:+.2f}"

    sign = _finite(faith.get("sign_agreement"))
    sign_text = "undefined" if sign is None else f"{100.0 * sign:.0f}%"
    n = faith.get("n")
    parts = [
        f"Spearman {corr(faith.get('spearman'))}",
        f"Pearson {corr(faith.get('pearson'))}",
        f"n={int(n) if isinstance(n, (int, float)) else '?'}",
        f"sign agreement {sign_text}",
    ]
    method = faith.get("method")
    metric = faith.get("metric")
    detail = ", ".join(str(x) for x in (method, f"metric {metric}" if metric else None) if x)
    return " | ".join(parts) + (f" ({detail})" if detail else "")


def attribution_summary(graph: AttributionGraph) -> dict[str, Any]:
    """JSON-safe summary of how *graph*'s edges were computed and how faithful they are.

    Keys: ``method`` (``"gradient"`` / ``"activation_flow"`` / ``"mask_perturbation"``),
    ``method_requested``, ``edge_semantics``, ``semantics_label``,
    ``semantics_explanation``, ``edge_definition``, ``method_fallback`` (why
    gradients were not used, or ``None``), ``metric`` / ``target_token`` /
    ``runner_up_token`` / ``metric_value`` / ``node_kind`` / ``baseline`` (gradient
    traces), ``n_attribution_nodes``, ``error_kind`` / ``unexplained_fraction``
    (the error node, if any), ``faithfulness`` (``None`` unless validated; else
    Spearman, Pearson, ``n``, sign agreement, per-node predicted vs measured and
    :data:`FAITHFULNESS_CAVEAT`) and ``faithfulness_skipped``.
    """
    meta = dict(getattr(graph, "meta", {}) or {})
    semantics = meta.get("edge_semantics")
    label, explanation = EDGE_SEMANTICS.get(str(semantics), (str(semantics or "unknown"), ""))
    err = None
    for node in graph.nodes():
        if node.node_type == "error":
            err = node
            break
    out: dict[str, Any] = {
        "method": meta.get("attribution_method"),
        "method_requested": meta.get("method_requested"),
        "edge_semantics": semantics,
        "semantics_label": label,
        "semantics_explanation": explanation,
        "edge_definition": meta.get("edge_definition"),
        "method_fallback": meta.get("method_fallback"),
        "metric": meta.get("metric"),
        "target_token": meta.get("target_token"),
        "runner_up_token": meta.get("runner_up_token"),
        "metric_value": _finite(meta.get("metric_value")),
        "node_kind": meta.get("node_kind"),
        "baseline": meta.get("baseline"),
        "n_attribution_nodes": meta.get("n_attribution_nodes", 0),
        "error_kind": err.meta.get("error_kind") if err is not None else None,
        "unexplained_fraction": (
            _finite(err.meta.get("unexplained_fraction")) if err is not None else None
        ),
        "faithfulness": _faithfulness(meta),
        "faithfulness_skipped": meta.get("faithfulness_skipped"),
    }
    return out


def attribution_html(summary: dict[str, Any]) -> str:
    """A small HTML panel (report / notebook) describing *summary* honestly."""
    esc = html.escape
    rows: list[str] = []
    method = summary.get("method") or "unknown"
    rows.append(
        f"<b>Edges:</b> {esc(str(summary.get('semantics_label') or 'unknown'))} "
        f"&middot; method <code>{esc(str(method))}</code>"
        + (
            f" (requested <code>{esc(str(summary['method_requested']))}</code>)"
            if summary.get("method_requested") and summary.get("method_requested") != method
            else ""
        )
    )
    if summary.get("semantics_explanation"):
        rows.append(esc(str(summary["semantics_explanation"])))
    if summary.get("metric"):
        target = summary.get("target_token")
        rows.append(
            f"<b>Target metric:</b> {esc(str(summary['metric']))}"
            + (f" of {esc(repr(target))}" if target is not None else "")
            + (
                f" vs {esc(repr(summary['runner_up_token']))}"
                if summary.get("metric") == "logit_diff" and summary.get("runner_up_token")
                else ""
            )
            + (
                f" &middot; node kind {esc(str(summary['node_kind']))}"
                if summary.get("node_kind")
                else ""
            )
            + (
                f" &middot; baseline {esc(str(summary['baseline']))}"
                if summary.get("baseline")
                else ""
            )
        )
    if summary.get("method_fallback"):
        rows.append(
            "<b>Fallback:</b> gradient attribution was not used - "
            f"{esc(str(summary['method_fallback']))}"
        )
    frac = summary.get("unexplained_fraction")
    if isinstance(frac, (int, float)):
        what = (
            "attribution mass not covered by the graph's nodes"
            if summary.get("error_kind") == "attribution_mass"
            else "residual activation energy not covered by the graph's nodes"
        )
        rows.append(f"<b>Error node:</b> {100.0 * float(frac):.1f}% {what}.")
    faith = summary.get("faithfulness")
    if faith:
        rows.append(f"<b>Faithfulness (real ablations):</b> {esc(format_faithfulness(faith))}")
        rows.append(f"<em>{esc(FAITHFULNESS_CAVEAT)}</em>")
        nodes = faith.get("nodes") or []
        if nodes:
            body = "".join(
                "<tr>"
                f"<td>L{esc(str(n.get('layer')))}</td><td>{esc(str(n.get('token_idx')))}</td>"
                f"<td>{esc(str(n.get('kind', '')))}</td>"
                f"<td style='text-align:right'>{_fmt(n.get('predicted'))}</td>"
                f"<td style='text-align:right'>{_fmt(n.get('measured'))}</td>"
                "</tr>"
                for n in nodes
            )
            rows.append(
                "<table class='tl-faith'><thead><tr><th>layer</th><th>token</th><th>kind</th>"
                "<th>predicted</th><th>measured</th></tr></thead>"
                f"<tbody>{body}</tbody></table>"
            )
    elif summary.get("faithfulness_skipped"):
        rows.append(
            f"<b>Faithfulness:</b> not computed - {esc(str(summary['faithfulness_skipped']))}"
        )
    elif summary.get("edge_semantics") == "causal_linearised":
        rows.append(
            "<b>Faithfulness:</b> not validated (run with <code>validate=k</code> to compare "
            "the estimates with real ablations)."
        )
    return (
        "<div class='tl-attr'>"
        + "".join(f"<div class='tl-attr-row'>{r}</div>" for r in rows)
        + "</div>"
    )


def _fmt(value: Any) -> str:
    f = _finite(value)
    return "-" if f is None else f"{f:+.4g}"
