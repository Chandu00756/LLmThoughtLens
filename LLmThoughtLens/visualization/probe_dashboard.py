"""ProbeDashboard — render probe-runner results as a scorecard + radar chart."""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

from LLmThoughtLens.utils.colors import THOUGHTLENS_COLORS

if TYPE_CHECKING:
    from LLmThoughtLens.probes.base import ProbeResult


class ProbeDashboard:
    """Render a probe report as HTML (scorecard rows + radar Plotly figure)."""

    def __init__(self, results: list[ProbeResult]) -> None:
        self.results = list(results)

    # ------------------------------------------------------------------
    # Scorecard rows
    # ------------------------------------------------------------------

    def _rows_html(self) -> str:
        rows: list[str] = []
        for r in self.results:
            passed = r.passed
            badge_cls = "tl-pass" if passed else "tl-fail"
            badge_txt = "PASS" if passed else "FAIL"
            bar_w = int(180 * max(0.0, min(1.0, r.score)))
            summary = html.escape(r.summary or "")
            evidence = html.escape(_truncate(_format_evidence(r.evidence), 280))
            rows.append(
                f'<div class="probe-row">'
                f'<span class="probe-badge {badge_cls}">{badge_txt}</span>'
                f'<span class="probe-name">{html.escape(r.probe_name)}</span>'
                f'<span class="probe-summary">{summary}</span>'
                f'<span class="probe-score">{r.score:.2f}</span>'
                f'<span class="probe-bar"><span class="probe-fill" style="width:{bar_w}px"></span></span>'
                f"</div>"
                f'<details class="probe-detail"><summary>evidence</summary>'
                f"<pre>{evidence}</pre></details>"
            )
        return "".join(rows)

    # ------------------------------------------------------------------
    # Radar chart
    # ------------------------------------------------------------------

    def radar_html(self) -> str:
        if not self.results:
            return ""
        try:
            import plotly.graph_objects as go
        except ImportError:  # pragma: no cover
            return ""

        names = [r.probe_name for r in self.results]
        scores = [max(0.0, min(1.0, r.score)) for r in self.results]
        # Close the polygon by repeating the first point.
        names_closed = names + [names[0]]
        scores_closed = scores + [scores[0]]
        fig = go.Figure(
            data=go.Scatterpolar(
                r=scores_closed,
                theta=names_closed,
                fill="toself",
                line={"color": THOUGHTLENS_COLORS["accent"]},
                fillcolor="rgba(1, 105, 111, 0.2)",
            )
        )
        fig.update_layout(
            polar={"radialaxis": {"range": [0, 1], "tickfont": {"size": 10}}},
            title="Interpretability radar",
            showlegend=False,
            paper_bgcolor=THOUGHTLENS_COLORS["surface"],
            height=420,
            margin={"t": 60, "b": 50, "l": 50, "r": 30},
        )
        return fig.to_html(full_html=False, include_plotlyjs=False)

    # ------------------------------------------------------------------
    # Top-level HTML
    # ------------------------------------------------------------------

    def to_html(self) -> str:
        if not self.results:
            return "<div class='probe-empty'>No probes were run.</div>"
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        header = (
            f'<div class="probe-overall"><b>{passed}/{total}</b> probes passed · '
            f"mean score {sum(r.score for r in self.results) / total:.2f}</div>"
        )
        return (
            header + self.radar_html() + '<div class="probe-list">' + self._rows_html() + "</div>"
        )


def _format_evidence(d: dict) -> str:
    import json

    try:
        return json.dumps(d, indent=2, default=str)
    except Exception:  # noqa: BLE001
        return repr(d)


def _truncate(text: str, max_len: int) -> str:
    return text if len(text) <= max_len else text[:max_len] + "…"
