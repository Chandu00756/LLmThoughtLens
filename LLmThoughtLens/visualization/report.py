# ruff: noqa: UP031  (the CSS template uses %-formatting because { } collide with format-string braces)
"""ReportBuilder — self-contained HTML report with five real tabs.

The five tabs (per the design document):

1. **Token heatmap** — Plotly heatmap of per-token activation magnitudes.
2. **Attribution graph** — layered Plotly directed graph (real nodes + edges).
3. **Residual stream** — PCA trajectory of selected tokens across layers.
4. **Feature browser** — searchable / filterable HTML table.
5. **Probe dashboard** — scorecard + radar chart.

The whole document is a single ``.html`` file with zero external assets
except a CDN-loaded ``plotly.min.js``.  Open it in any browser.
"""

from __future__ import annotations

import datetime as _dt
import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from LLmThoughtLens.utils.colors import THOUGHTLENS_COLORS
from LLmThoughtLens.visualization.feature_browser import FeatureBrowser
from LLmThoughtLens.visualization.graph_viz import GraphVisualizer
from LLmThoughtLens.visualization.layer_stream import ResidualStreamView
from LLmThoughtLens.visualization.probe_dashboard import ProbeDashboard
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

    def add_tab(self, tab_id: str, title: str, content: str, **meta: Any) -> ReportBuilder:
        self._tabs.append(ReportTab(tab_id=tab_id, title=title, content=content, meta=meta))
        return self

    def add_js(self, snippet: str) -> ReportBuilder:
        self._extra_js.append(snippet)
        return self

    def add_section(self, title: str, content: str, **meta: Any) -> ReportBuilder:
        return self.add_tab(title.lower().replace(" ", "_"), title, content, **meta)

    def add_graph_diff(self, diff: Any, title: str = "Graph Diff") -> ReportBuilder:
        """Embed a :class:`GraphDiff` rendering as a new tab in the report."""
        return self.add_tab("diff", title, diff.to_html(), diff_summary=diff.summary())

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
    def from_trace_result(cls, result: TraceResult) -> ReportBuilder:
        evidence_kind = result.output.evidence_kind
        builder = cls(
            title="LLmThoughtLens — Interpretability Report",
            model=result.output.meta.get("model", result.meta.get("provider", "")),
            prompt=result.prompt,
            evidence_kind=evidence_kind,
        )

        # 1. Token heatmap
        try:
            heatmap_html = TokenHeatmap(result.output, result.features).to_html()
        except Exception as exc:  # noqa: BLE001
            heatmap_html = f"<p>Token heatmap unavailable: {html.escape(repr(exc))}</p>"
        builder.add_tab("heatmap", "Token Heatmap", heatmap_html)

        # 2. Attribution graph
        try:
            graph_html = GraphVisualizer(result.graph).to_html()
        except Exception as exc:  # noqa: BLE001
            graph_html = f"<p>Attribution graph unavailable: {html.escape(repr(exc))}</p>"
        builder.add_tab("graph", "Attribution Graph", graph_html)

        # 3. Residual stream
        if result.output.has_internals:
            try:
                stream_html = ResidualStreamView(result.output).to_html()
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

        # Optional 6th tab: raw JSON for inspection.
        raw = {
            "prompt": result.prompt,
            "output_token": result.output.output_token,
            "top_tokens": result.output.top_tokens,
            "evidence_kind": evidence_kind,
            "features": [f.as_dict() for f in result.features[:30]],
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
