"""ReportBuilder — assembles interpretability results into a self-contained HTML report.

The report uses a tabbed layout (no server required) with one tab per view:
  Attribution Graph | Token Heatmap | Probe Results | Raw JSON
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: #f7f6f2;
  color: #28251d;
  max-width: 1100px;
  margin: 0 auto;
  padding: 1rem 1.5rem 3rem;
}
header {
  background: #01696f;
  color: #fff;
  padding: 1rem 1.5rem;
  border-radius: 8px;
  margin-bottom: 1.5rem;
}
header h1 { font-size: 1.3rem; font-weight: 700; }
header p  { font-size: 0.85rem; opacity: 0.85; margin-top: 0.3rem; }
.tabs { display: flex; gap: 4px; margin-bottom: 0; }
.tab-btn {
  padding: 0.5rem 1.1rem;
  border: none;
  border-radius: 6px 6px 0 0;
  background: #e2e0d9;
  color: #28251d;
  cursor: pointer;
  font-size: 0.88rem;
  font-weight: 500;
  transition: background 0.15s;
}
.tab-btn.active { background: #01696f; color: #fff; }
.tab-content {
  background: #fff;
  border: 1px solid #dddbd3;
  border-radius: 0 6px 6px 6px;
  padding: 1.2rem;
  min-height: 220px;
  display: none;
}
.tab-content.active { display: block; }
.probe-row {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 0.6rem 0;
  border-bottom: 1px solid #eee;
  font-size: 0.9rem;
}
.probe-badge {
  padding: 2px 8px;
  border-radius: 12px;
  font-size: 0.75rem;
  font-weight: 700;
  min-width: 48px;
  text-align: center;
}
.badge-pass { background: #d1f3d1; color: #1a6b1a; }
.badge-fail { background: #f8d7da; color: #842029; }
.bar-bg { background: #e2e0d9; border-radius: 4px; width: 160px; height: 10px; }
.bar-fill { height: 10px; border-radius: 4px; background: #01696f; }
pre { background: #f4f3ef; padding: 1rem; border-radius: 6px; overflow-x: auto; font-size: 0.8rem; }
footer { text-align: center; color: #7a7974; font-size: 0.78rem; margin-top: 2rem; }
"""

_JS = """
function showTab(id) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.getElementById('btn-' + id).classList.add('active');
  document.getElementById('tab-' + id).classList.add('active');
}
"""


@dataclass
class ReportSection:
    tab_id: str
    title: str
    content: str
    meta: dict[str, Any] = field(default_factory=dict)


class ReportBuilder:
    """Incrementally build a self-contained tabbed HTML interpretability report.

    Parameters
    ----------
    title:
        Top-level report title displayed in the header bar.
    model:
        Model identifier shown in the header.
    prompt:
        Prompt shown in the header.
    """

    def __init__(
        self,
        title: str = "llmscope Report",
        model: str = "",
        prompt: str = "",
    ) -> None:
        self.title = title
        self.model = model
        self.prompt = prompt
        self._sections: list[ReportSection] = []

    def add_tab(
        self,
        tab_id: str,
        title: str,
        content: str,
        **meta: Any,
    ) -> "ReportBuilder":
        """Add a tab to the report.

        Parameters
        ----------
        tab_id:
            Short identifier used in JavaScript (no spaces).
        title:
            Tab button label.
        content:
            HTML content for this tab's panel.
        """
        self._sections.append(ReportSection(tab_id=tab_id, title=title, content=content, meta=meta))
        return self

    # Keep legacy add_section as an alias
    def add_section(self, title: str, content: str, **meta: Any) -> "ReportBuilder":
        """Add a section/tab (legacy alias for :meth:`add_tab`)."""
        tab_id = title.lower().replace(" ", "_")
        return self.add_tab(tab_id, title, content, **meta)

    def render(self) -> str:
        """Return the complete self-contained HTML report as a string."""
        if not self._sections:
            self.add_tab("empty", "Report", "<p>No content yet.</p>")

        # Tab buttons
        tab_buttons = " ".join(
            f'<button class="tab-btn{"  active" if i == 0 else ""}" '
            f'id="btn-{s.tab_id}" onclick="showTab(\'{s.tab_id}\')">{s.title}</button>'
            for i, s in enumerate(self._sections)
        )

        # Tab panels
        tab_panels = "\n".join(
            f'<div class="tab-content{"  active" if i == 0 else ""}" id="tab-{s.tab_id}">'
            f"{s.content}</div>"
            for i, s in enumerate(self._sections)
        )

        header_sub = ""
        if self.model:
            header_sub += f"Model: <b>{self.model}</b>  "
        if self.prompt:
            header_sub += f'Prompt: <em>"{self.prompt}"</em>'

        return (
            "<!DOCTYPE html>\n<html lang='en'>\n<head>\n"
            f"<meta charset='utf-8'><title>{self.title}</title>\n"
            f"<style>{_CSS}</style>\n"
            "</head>\n<body>\n"
            f"<header><h1>{self.title}</h1>"
            f"{'<p>' + header_sub + '</p>' if header_sub else ''}"
            "</header>\n"
            f"<div class='tabs'>{tab_buttons}</div>\n"
            f"{tab_panels}\n"
            "<footer>llmscope &nbsp;·&nbsp; "
            "<a href='https://github.com/Chandu00756/LLmThoughtLens'>GitHub</a></footer>\n"
            f"<script>{_JS}</script>\n"
            "</body>\n</html>"
        )

    def save(self, path: str) -> None:
        """Write the self-contained HTML report to *path*.

        Parameters
        ----------
        path:
            Output file path (typically ending in ``.html``).
        """
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.render())

    # ------------------------------------------------------------------
    # Convenience factory
    # ------------------------------------------------------------------

    @classmethod
    def from_trace_result(cls, result: Any) -> "ReportBuilder":
        """Build a report from a :class:`~llmscope.scope.TraceResult`.

        Parameters
        ----------
        result:
            :class:`~llmscope.scope.TraceResult` returned by
            :meth:`~llmscope.scope.Scope.trace_full`.
        """
        builder = cls(
            title="llmscope — Interpretability Report",
            model=result.meta.get("provider", ""),
            prompt=result.prompt,
        )

        # --- Attribution graph tab ---
        try:
            from llmscope.visualization.graph_viz import GraphVisualizer

            gv = GraphVisualizer(result.graph)
            graph_html = gv.to_html()
        except Exception:
            graph_html = "<p>Attribution graph unavailable.</p>"
        builder.add_tab("graph", "Attribution Graph", graph_html)

        # --- Token heatmap tab ---
        try:
            from llmscope.visualization.token_heatmap import TokenHeatmap

            th = TokenHeatmap(result.output, result.features)
            heatmap_html = th.to_html()
        except Exception:
            heatmap_html = "<p>Token heatmap unavailable.</p>"
        builder.add_tab("heatmap", "Token Heatmap", heatmap_html)

        # --- Probe results tab ---
        probe_rows = ""
        for pr in result.probe_results:
            passed = pr.meta.get("passed", pr.score >= 0.5)
            badge_cls = "badge-pass" if passed else "badge-fail"
            badge_txt = "PASS" if passed else "FAIL"
            bar_w = int(pr.score * 160)
            summary = pr.meta.get("summary", "")
            probe_rows += (
                f'<div class="probe-row">'
                f'<span class="probe-badge {badge_cls}">{badge_txt}</span>'
                f'<span style="flex:1"><b>{pr.probe_name}</b> — {summary}</span>'
                f'<span style="font-size:0.82rem;color:#7a7974">{pr.score:.2f}</span>'
                f'<div class="bar-bg"><div class="bar-fill" style="width:{bar_w}px"></div></div>'
                f'</div>'
            )
        if not probe_rows:
            probe_rows = "<p>No probes were run.</p>"
        builder.add_tab("probes", "Probe Results", probe_rows)

        # --- Raw JSON tab ---
        raw = {
            "prompt": result.prompt,
            "output_token": result.output_token,
            "top_tokens": result.top_tokens,
            "features": [
                {"id": f.id, "label": f.label, "layer": f.layer, "score": f.score}
                for f in result.features[:20]
            ],
            "graph": result.graph.to_dict() if hasattr(result.graph, "to_dict") else {},
        }
        builder.add_tab("json", "Raw JSON", f"<pre>{json.dumps(raw, indent=2)}</pre>")

        return builder

    def __repr__(self) -> str:
        return f"ReportBuilder(title={self.title!r}, tabs={len(self._sections)})"
