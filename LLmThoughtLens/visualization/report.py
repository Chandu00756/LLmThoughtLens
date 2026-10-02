# ruff: noqa: UP031  (the CSS template uses %-formatting because { } collide with format-string braces)
"""ReportBuilder — self-contained HTML report with five real tabs.

The five tabs (per the design document):

1. **Token heatmap** — Plotly heatmap of per-token feature evidence.
2. **Attribution graph** — layered Plotly directed graph (real nodes + edges).
3. **Residual stream** — PCA trajectory of selected tokens across layers.
4. **Feature browser** — searchable / filterable HTML table.
5. **Probe dashboard** — scorecard + radar chart.

The whole document is a single ``.html`` file with zero external assets
except a CDN-loaded ``plotly.min.js``.  Open it in any browser.

The header carries honest caveats about how the numbers were produced: the
white-box score scale (unitless centred score vs legacy raw L2 norm), any
token positions the extractor left out of the ranking (attention-sink
outliers), read from ``graph.meta["excluded_positions"]`` and feature meta,
what the attribution edges mean (``graph.meta["edge_semantics"]``: causal
linearised estimate / correlational / causal input masking), the
real-ablation faithfulness check when it ran (Spearman, Pearson, ``n`` and
sign agreement together, with a caveat), and SAE input caveats
(``meta["sae_input_warning"]``).  The Attribution Graph tab opens with the
same attribution summary.  Passing a
:class:`~LLmThoughtLens.features.steering.SteeringResult` adds a Steering tab
(baseline vs steered completion, per-step KL, token shifts).
"""

from __future__ import annotations

import datetime as _dt
import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from LLmThoughtLens.features.extractor import EXCLUSION_REASON_OUTLIER, exclusion_reasons
from LLmThoughtLens.utils.colors import THOUGHTLENS_COLORS
from LLmThoughtLens.visualization.attribution_view import (
    attribution_html,
    attribution_summary,
    format_faithfulness,
)
from LLmThoughtLens.visualization.feature_browser import FeatureBrowser
from LLmThoughtLens.visualization.graph_viz import GraphVisualizer
from LLmThoughtLens.visualization.layer_stream import ResidualStreamView
from LLmThoughtLens.visualization.probe_dashboard import ProbeDashboard
from LLmThoughtLens.visualization.steering_view import SteeringView, steering_dict
from LLmThoughtLens.visualization.token_heatmap import TokenHeatmap

if TYPE_CHECKING:
    from LLmThoughtLens.scope import TraceResult

_PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"

_CSS = (
    """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
  background: %(bg)s;
  color: %(text)s;
  max-width: 1180px;
  margin: 0 auto;
  padding: 18px 24px 64px;
}
header.tl-header {
  background: %(accent)s;
  color: #fff;
  padding: 18px 22px;
  border-radius: 10px;
  margin-bottom: 18px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.06);
}
header.tl-header h1 { font-size: 1.35rem; font-weight: 700; letter-spacing: 0.2px; }
header.tl-header .tl-sub { font-size: 0.85rem; opacity: 0.85; margin-top: 4px; }
header.tl-header .tl-evidence { display: inline-block; padding: 2px 8px; border-radius: 12px;
  background: rgba(255,255,255,0.18); font-size: 0.72rem; margin-left: 6px; }
.tl-tabbar { display: flex; gap: 4px; }
.tl-tab-btn {
  padding: 8px 16px;
  border: none;
  border-radius: 8px 8px 0 0;
  background: #e6e2d6;
  color: %(text)s;
  cursor: pointer;
  font-size: 0.88rem;
  font-weight: 500;
  transition: background 0.15s;
}
.tl-tab-btn.active { background: %(accent)s; color: #fff; }
.tl-tab-panel {
  background: #fff;
  border: 1px solid #d8d4c8;
  border-radius: 0 10px 10px 10px;
  padding: 18px;
  min-height: 260px;
  display: none;
}
.tl-tab-panel.active { display: block; }
.probe-row { display: flex; align-items: center; gap: 12px; padding: 8px 0; border-bottom: 1px solid #eee; font-size: 0.9rem; }
.probe-badge { padding: 3px 10px; border-radius: 12px; font-weight: 700; font-size: 0.72rem; min-width: 52px; text-align: center; }
.tl-pass { background: #d1f3d1; color: %(pass_)s; }
.tl-fail { background: #f8d7da; color: %(fail_)s; }
.probe-name { font-weight: 600; min-width: 180px; }
.probe-summary { flex: 1; color: %(muted)s; font-size: 0.85rem; }
.probe-score { font-weight: 600; min-width: 36px; text-align: right; }
.probe-bar { width: 180px; background: #e6e2d6; border-radius: 4px; height: 10px; display: inline-block; }
.probe-fill { background: %(accent)s; border-radius: 4px; height: 10px; display: block; }
.probe-detail { font-size: 0.78rem; color: %(muted)s; margin-bottom: 12px; }
.probe-detail pre { background: #f4f3ef; padding: 8px 12px; border-radius: 6px; overflow-x: auto; white-space: pre-wrap; }
.probe-overall { font-size: 1.05rem; padding: 6px 0 12px; }
.tl-footer { text-align: center; color: %(muted)s; font-size: 0.78rem; margin-top: 28px; }
.tl-theme-btn { float: right; cursor: pointer; border: 1px solid rgba(255,255,255,0.4);
  background: rgba(255,255,255,0.12); color: #fff; border-radius: 8px; padding: 4px 10px;
  font-size: 0.75rem; }
.tl-legend { font-size: 0.74rem; margin-top: 8px; opacity: 0.9; }
.tl-legend b { font-weight: 700; }
.tl-caveat { font-size: 0.74rem; margin-top: 6px; padding: 5px 9px; border-radius: 6px;
  background: rgba(255,255,255,0.14); }
.tl-caveat b { font-weight: 700; }
.tl-method { font-size: 0.74rem; margin-top: 6px; padding: 5px 9px; border-radius: 6px;
  background: rgba(255,255,255,0.10); border-left: 3px solid rgba(255,255,255,0.45); }
.tl-method b { font-weight: 700; }
.tl-attr { font-size: 0.85rem; margin-bottom: 12px; padding: 10px 12px; border-radius: 8px;
  background: rgba(1,105,111,0.07); border: 1px solid rgba(1,105,111,0.18); }
.tl-attr-row { margin: 3px 0; }
.tl-faith { border-collapse: collapse; margin-top: 6px; font-size: 0.8rem; }
.tl-faith th, .tl-faith td { padding: 2px 8px; border-bottom: 1px solid rgba(0,0,0,0.08); }
.tl-steer-cols { display: flex; gap: 14px; flex-wrap: wrap; margin: 10px 0; }
.tl-steer-col { flex: 1 1 280px; min-width: 0; }
.tl-steer-text { font-family: ui-monospace, Menlo, monospace; font-size: 0.85rem;
  white-space: pre-wrap; padding: 8px 10px; border-radius: 6px; background: rgba(0,0,0,0.04); }
.tl-diverge { background: rgba(209,153,0,0.35); border-radius: 3px; }
.tl-steer-labels { font-size: 0.78rem; opacity: 0.85; }
/* Dark mode: respects the OS setting, and a manual toggle via [data-theme]. */
@media (prefers-color-scheme: dark) {
  html:not([data-theme="light"]) body { background: #15140f; color: #ece7da; }
  html:not([data-theme="light"]) .tl-tab-panel { background: #1f1c16; border-color: #322d22; }
  html:not([data-theme="light"]) .tl-tab-btn { background: #2a261d; color: #cfc8b6; }
  html:not([data-theme="light"]) .probe-detail pre { background: #15140f; }
}
html[data-theme="dark"] body { background: #15140f; color: #ece7da; }
html[data-theme="dark"] .tl-tab-panel { background: #1f1c16; border-color: #322d22; }
html[data-theme="dark"] .tl-tab-btn { background: #2a261d; color: #cfc8b6; }
html[data-theme="dark"] .probe-detail pre { background: #15140f; }
"""
    % {  # noqa: UP031  (CSS contains `{}` so %-formatting is the cleanest interpolation)
        "bg": THOUGHTLENS_COLORS["bg"],
        "text": THOUGHTLENS_COLORS["text"],
        "accent": THOUGHTLENS_COLORS["accent"],
        "muted": THOUGHTLENS_COLORS["muted"],
        "pass_": THOUGHTLENS_COLORS["pass"],
        "fail_": THOUGHTLENS_COLORS["fail"],
    }
)

_JS_TABS = """
function tlsShowTab(id) {
  document.querySelectorAll('.tl-tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tl-tab-panel').forEach(p => p.classList.remove('active'));
  document.getElementById('tlb-' + id).classList.add('active');
  document.getElementById('tlp-' + id).classList.add('active');
  if (window.Plotly) {
    document.querySelectorAll('#tlp-' + id + ' .js-plotly-plot').forEach(el => {
      window.Plotly.Plots.resize(el);
    });
  }
}
function tlsToggleTheme() {
  var h = document.documentElement;
  var cur = h.getAttribute('data-theme');
  h.setAttribute('data-theme', cur === 'dark' ? 'light' : 'dark');
}
"""


@dataclass
class ReportTab:
    tab_id: str
    title: str
    content: str
    meta: dict[str, Any] = field(default_factory=dict)


class ReportBuilder:
    """Assemble the five tabs into a self-contained HTML report."""

    def __init__(
        self,
        title: str = "LLmThoughtLens — Interpretability Report",
        model: str = "",
        prompt: str = "",
        evidence_kind: str = "",
    ) -> None:
        self.title = title
        self.model = model
        self.prompt = prompt
        self.evidence_kind = evidence_kind
        self._tabs: list[ReportTab] = []
        self._extra_js: list[str] = []
        self._caveats: list[str] = []
        self._method_notes: list[str] = []

    def add_tab(self, tab_id: str, title: str, content: str, **meta: Any) -> ReportBuilder:
        self._tabs.append(ReportTab(tab_id=tab_id, title=title, content=content, meta=meta))
        return self

    def add_js(self, snippet: str) -> ReportBuilder:
        self._extra_js.append(snippet)
        return self

    def add_caveat(self, snippet: str) -> ReportBuilder:
        """Add a header caveat line.  *snippet* is trusted HTML — escape user text first."""
        self._caveats.append(snippet)
        return self

    def add_method_note(self, snippet: str) -> ReportBuilder:
        """Add a header line about *how* the evidence was computed (attribution method,
        edge semantics, faithfulness).  *snippet* is trusted HTML — escape user text first."""
        self._method_notes.append(snippet)
        return self

    def add_section(self, title: str, content: str, **meta: Any) -> ReportBuilder:
        return self.add_tab(title.lower().replace(" ", "_"), title, content, **meta)

    def add_graph_diff(self, diff: Any, title: str = "Graph Diff") -> ReportBuilder:
        """Embed a :class:`GraphDiff` rendering as a new tab in the report."""
        return self.add_tab("diff", title, diff.to_html(), diff_summary=diff.summary())

    def add_steering(self, steering: Any, title: str = "Steering") -> ReportBuilder:
        """Add a Steering tab for a :class:`SteeringResult` (or its ``to_dict()``)."""
        try:
            content = SteeringView(steering).to_html()
        except Exception as exc:  # noqa: BLE001
            content = f"<p>Steering view unavailable: {html.escape(repr(exc))}</p>"
        return self.add_tab("steering", title, content)

    @classmethod
    def from_steering_result(cls, steering: Any) -> ReportBuilder:
        """A one-tab report for a steering comparison (``LLmThoughtLens steer --report``)."""
        data = steering_dict(steering)
        vectors = data.get("vectors") or []
        model = next((str(v["model_name"]) for v in vectors if v.get("model_name")), "")
        builder = cls(
            title="LLmThoughtLens — Steering Report",
            model=model,
            prompt=str(data.get("prompt", "")),
            evidence_kind=str(data.get("evidence_kind", "white_box")),
        )
        if data.get("note"):
            builder.add_caveat(f"<b>Steering evidence:</b> {html.escape(str(data['note']))}")
        builder.add_steering(data)
        return builder

    def render(self) -> str:
        if not self._tabs:
            self.add_tab("empty", "Report", "<p>No content.</p>")

        tab_buttons = " ".join(
            f'<button class="tl-tab-btn{" active" if i == 0 else ""}" '
            f'id="tlb-{t.tab_id}" onclick="tlsShowTab(\'{t.tab_id}\')">{html.escape(t.title)}</button>'
            for i, t in enumerate(self._tabs)
        )
        panels = "\n".join(
            f'<div class="tl-tab-panel{" active" if i == 0 else ""}" id="tlp-{t.tab_id}">{t.content}</div>'
            for i, t in enumerate(self._tabs)
        )

        sub_parts: list[str] = []
        if self.model:
            sub_parts.append(f"Model: <b>{html.escape(self.model)}</b>")
        if self.prompt:
            sub_parts.append(f'Prompt: <em>"{html.escape(self.prompt)}"</em>')
        if self.evidence_kind:
            sub_parts.append(
                f'<span class="tl-evidence">evidence: {html.escape(self.evidence_kind)}</span>'
            )
        sub = " &middot; ".join(sub_parts)
        generated = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
        extra_js = "\n".join(self._extra_js)
        legend = _evidence_legend(self.evidence_kind)
        caveats = "".join(f'<div class="tl-caveat">{c}</div>' for c in self._caveats)
        method_notes = "".join(f'<div class="tl-method">{m}</div>' for m in self._method_notes)

        return (
            "<!DOCTYPE html>\n<html lang='en'>\n<head>\n"
            f"<meta charset='utf-8'><title>{html.escape(self.title)}</title>\n"
            f'<script src="{_PLOTLY_CDN}"></script>\n'
            f"<style>{_CSS}</style>\n"
            "</head>\n<body>\n"
            f'<header class="tl-header">'
            '<button class="tl-theme-btn" onclick="tlsToggleTheme()">◐ theme</button>'
            f"<h1>{html.escape(self.title)}</h1>"
            f'<div class="tl-sub">{sub} &middot; generated {generated}</div>'
            f'<div class="tl-legend">{legend}</div>'
            f"{method_notes}"
            f"{caveats}"
            "</header>\n"
            f'<div class="tl-tabbar">{tab_buttons}</div>\n'
            f"{panels}\n"
            '<footer class="tl-footer">LLmThoughtLens &middot; '
            '<a href="https://github.com/Chandu00756/LLmThoughtLens">github.com/Chandu00756/LLmThoughtLens</a>'
            "</footer>\n"
            f"<script>{_JS_TABS}{extra_js}</script>\n"
            "<script>document.addEventListener('DOMContentLoaded', function() { "
            "if (typeof tlsFeatureBrowser === 'function') tlsFeatureBrowser(); });</script>\n"
            "</body>\n</html>"
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.render(), encoding="utf-8")

    @classmethod
    def from_trace_result(cls, result: TraceResult, steering: Any = None) -> ReportBuilder:
        """The tabbed report for *result*; *steering* (a SteeringResult) adds a Steering tab."""
        evidence_kind = result.output.evidence_kind
        builder = cls(
            title="LLmThoughtLens — Interpretability Report",
            model=result.output.meta.get("model", result.meta.get("provider", "")),
            prompt=result.prompt,
            evidence_kind=evidence_kind,
        )

        excluded_meta = result.graph.meta.get("excluded_positions")
        exclusions = exclusion_reasons(result.features, excluded_meta)
        for note in _attribution_notes(result):
            builder.add_method_note(note)
        for caveat in _trace_caveats(result, exclusions):
            builder.add_caveat(caveat)

        # 1. Token heatmap
        try:
            heatmap_html = TokenHeatmap(
                result.output, result.features, excluded_positions=excluded_meta
            ).to_html()
        except Exception as exc:  # noqa: BLE001
            heatmap_html = f"<p>Token heatmap unavailable: {html.escape(repr(exc))}</p>"
        builder.add_tab("heatmap", "Token Heatmap", heatmap_html)

        # 2. Attribution graph (preceded by the method / semantics / faithfulness panel)
        summary = attribution_summary(result.graph)
        try:
            graph_html = GraphVisualizer(result.graph, excluded=exclusions).to_html()
        except Exception as exc:  # noqa: BLE001
            graph_html = f"<p>Attribution graph unavailable: {html.escape(repr(exc))}</p>"
        builder.add_tab("graph", "Attribution Graph", attribution_html(summary) + graph_html)

        # 3. Residual stream
        if result.output.has_internals:
            try:
                stream_html = ResidualStreamView(
                    result.output, exclude_positions=list(exclusions)
                ).to_html()
            except Exception as exc:  # noqa: BLE001
                stream_html = f"<p>Residual stream view unavailable: {html.escape(repr(exc))}</p>"
        else:
            stream_html = (
                "<p>Residual stream trajectory requires a white-box provider with "
                "real activations.  This trace was generated in black-box mode.</p>"
            )
        builder.add_tab("stream", "Residual Stream", stream_html)

        # 4. Feature browser
        fb = FeatureBrowser(result.features)
        builder.add_tab("features", "Feature Browser", fb.to_html())
        builder.add_js(fb.js())

        # 5. Probe dashboard
        probe_html = ProbeDashboard(result.probe_results).to_html()
        builder.add_tab("probes", "Probe Dashboard", probe_html)

        # Optional: steering comparison
        if steering is not None:
            builder.add_steering(steering)

        # Optional 6th tab: raw JSON for inspection.
        raw = {
            "prompt": result.prompt,
            "output_token": result.output.output_token,
            "top_tokens": result.output.top_tokens,
            "evidence_kind": evidence_kind,
            "score_method": _score_method(result.features),
            "excluded_positions": {str(p): r for p, r in exclusions.items()},
            "attribution": summary,
            "features": [{**f.as_dict(), "meta": f.meta} for f in result.features[:30]],
            "graph": result.graph.to_dict(),
            "probes": [p.as_dict() for p in result.probe_results],
        }
        builder.add_tab(
            "json",
            "Raw JSON",
            f"<pre style='background:#f4f3ef;padding:12px;border-radius:6px;overflow:auto'>"
            f"{html.escape(json.dumps(raw, indent=2, default=str))}</pre>",
        )

        return builder

    def __repr__(self) -> str:
        return f"ReportBuilder(title={self.title!r}, tabs={len(self._tabs)})"


def _evidence_legend(evidence_kind: str) -> str:
    """Explain the observation taxonomy so a reader never over-trusts a number.

    * **observed** — measured directly from real activations (white-box).
    * **inferred** — derived from real output probabilities / logprobs (black-box).
    * **approximated** — estimated by input perturbation / token masking (black-box).
    """
    this = "observed" if evidence_kind == "white_box" else "inferred / approximated"
    return (
        "<b>Evidence key:</b> "
        "<b>observed</b> = measured directly from real activations (white-box) &middot; "
        "<b>inferred</b> = from real output probabilities / logprobs (black-box) &middot; "
        "<b>approximated</b> = estimated by input perturbation / token masking (black-box). "
        f"This trace is <b>{html.escape(evidence_kind)}</b> ({this})."
    )


def _score_method(features: list[Any]) -> str:
    """The extractor scoring method shared by *features* (``""`` when mixed / unknown)."""
    methods = {f.meta.get("method") for f in features}
    if len(methods) == 1:
        method = next(iter(methods))
        return method if isinstance(method, str) else ""
    return ""


def _attribution_notes(result: TraceResult) -> list[str]:
    """Header method notes (trusted HTML): edge semantics, fallback, faithfulness."""
    caveats: list[str] = []
    summary = attribution_summary(result.graph)
    if summary.get("edge_semantics"):
        caveats.append(
            f"<b>Attribution edges:</b> {html.escape(str(summary['semantics_label']))} — "
            f"{html.escape(str(summary.get('semantics_explanation') or ''))}"
        )
    if summary.get("method_fallback"):
        caveats.append(
            "<b>Attribution fallback:</b> gradient attribution was not used — "
            f"{html.escape(str(summary['method_fallback']))}"
        )
    faith = summary.get("faithfulness")
    if faith:
        caveats.append(
            f"<b>Faithfulness (real ablations):</b> {html.escape(format_faithfulness(faith))}. "
            f"{html.escape(str(faith.get('caveat', '')))}"
        )
    return caveats


def _sae_caveats(result: TraceResult) -> list[str]:
    """One caveat per distinct ``meta["sae_input_warning"]`` (e.g. SAELens BOS mismatch)."""
    seen: list[str] = []
    for f in result.features:
        w = f.meta.get("sae_input_warning")
        if isinstance(w, str) and w not in seen:
            seen.append(w)
    return [f"<b>SAE input:</b> {html.escape(w)}" for w in seen]


def _trace_caveats(result: TraceResult, exclusions: dict[int, str]) -> list[str]:
    """Header caveats (trusted HTML) describing the score scale and excluded positions."""
    caveats: list[str] = _sae_caveats(result)
    method = _score_method(result.features)
    if method == "centered_norm":
        caveats.append(
            "<b>Score scale:</b> white-box feature scores are <b>unitless</b> — each site's "
            "distance from its layer's robust centre divided by the layer's median token norm. "
            "Raw residual norms are in the feature browser's <em>Raw ‖h‖</em> column and the "
            "JSON <code>meta.raw_norm</code>."
        )
    elif method == "l2_norm":
        caveats.append(
            '<b>Score scale:</b> legacy <code>scoring="l2"</code> — feature scores are raw '
            "residual L2 norms, which a massive-activation (attention-sink) position can dominate."
        )
    if not exclusions:
        return caveats

    tokens = result.output.tokens
    stats = result.meta.get("outlier_stats") or {}
    parts: list[str] = []
    for pos, reason in exclusions.items():
        tok = f" {html.escape(repr(tokens[pos]))}" if 0 <= pos < len(tokens) else ""
        detail = ""
        st = (stats.get(pos) or stats.get(str(pos))) if isinstance(stats, dict) else None
        if reason == EXCLUSION_REASON_OUTLIER and isinstance(st, dict):
            detail = (
                f" — up to {float(st.get('max_ratio', 0.0)):.1f}x the other tokens' median norm "
                f"in {100.0 * float(st.get('layer_frac', 0.0)):.0f}% of layers"
            )
        parts.append(f"position {pos}{tok}: {html.escape(reason)}{detail}")
    line = (
        "<b>Excluded from feature ranking:</b> "
        + "; ".join(parts)
        + ". These positions are left out of the per-layer statistics and shown greyed in the "
        "token heatmap."
    )
    err = next((n for n in result.graph.nodes() if n.node_type == "error"), None)
    frac = err.meta.get("excluded_energy_fraction") if err is not None else None
    if isinstance(frac, (int, float)):
        line += (
            f" Their activation energy ({100.0 * float(frac):.1f}% of the total) is not counted "
            "as unexplained in the error residual."
        )
    caveats.append(line)
    return caveats
