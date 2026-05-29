"""GraphVisualizer — render an :class:`AttributionGraph` as a layered Plotly DAG.

Nodes are positioned on an x-axis layer band (input nodes on the left,
output nodes on the right, features in between by their layer attribute)
and a y-axis spread by token position + score.  Edge colour encodes
polarity (teal = promote, magenta = suppress) and edge width encodes
absolute weight magnitude.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from LLmThoughtLens.utils.colors import THOUGHTLENS_COLORS, edge_color, node_color

if TYPE_CHECKING:
    from LLmThoughtLens.circuits.graph import AttributionGraph, CircuitNode


_NODE_SYMBOL = {
    "input_token": "square",
    "feature": "circle",
    "supernode": "hexagon",
    "output_token": "star",
    "error": "diamond",
    "safety": "diamond-wide",
    "suppressor": "x",
}


class GraphVisualizer:
    """Layered DAG renderer."""

    def __init__(
        self,
        graph: AttributionGraph,
        max_nodes: int | None = 60,
        compact: bool = False,
    ) -> None:
        self.graph = graph
        self.max_nodes = max_nodes
        self.compact = compact

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _layout(self) -> dict[int, tuple[float, float]]:
        nodes = list(self.graph.nodes())
        if self.max_nodes is not None and len(nodes) > self.max_nodes:
            nodes = sorted(nodes, key=lambda n: abs(n.score), reverse=True)[: self.max_nodes]

        max_layer = max((n.layer for n in nodes if n.node_type == "feature"), default=0)
        column: dict[int, list[CircuitNode]] = defaultdict(list)
        for n in nodes:
            if n.node_type == "input_token":
                x = -1
            elif n.node_type == "output_token" or n.node_type == "error":
                x = max_layer + 1
            else:
                x = max(0, min(max_layer, n.layer))
            column[x].append(n)

        positions: dict[int, tuple[float, float]] = {}
        for x, members in column.items():
            members.sort(key=lambda n: (n.token_idx, -abs(n.score)))
            n_members = len(members)
            for i, m in enumerate(members):
                y = (i - (n_members - 1) / 2.0) * 1.2 if n_members > 1 else 0.0
                positions[m.id] = (float(x), float(y))
        return positions

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------

    def to_figure(self):
        try:
            import plotly.graph_objects as go
        except ImportError as exc:  # pragma: no cover
            raise ImportError("plotly is required for GraphVisualizer.") from exc

        positions = self._layout()
        if not positions:
            return go.Figure()

        promote_x: list[float | None] = []
        promote_y: list[float | None] = []
        suppress_x: list[float | None] = []
        suppress_y: list[float | None] = []
        for e in self.graph.edges():
            if e.src not in positions or e.dst not in positions:
                continue
            x0, y0 = positions[e.src]
            x1, y1 = positions[e.dst]
            mid_x = (x0 + x1) / 2.0
            mid_y = (y0 + y1) / 2.0 + 0.3 * (1.0 if x1 > x0 else -1.0)
            if e.polarity == "promote":
                promote_x.extend([x0, mid_x, x1, None])
                promote_y.extend([y0, mid_y, y1, None])
            else:
                suppress_x.extend([x0, mid_x, x1, None])
                suppress_y.extend([y0, mid_y, y1, None])

        promote_trace = go.Scatter(
            x=promote_x,
            y=promote_y,
            mode="lines",
            line={"color": edge_color(1.0), "width": 1.4},
            hoverinfo="skip",
            name="Promoting",
        )
        suppress_trace = go.Scatter(
            x=suppress_x,
            y=suppress_y,
            mode="lines",
            line={"color": edge_color(-1.0), "width": 1.4, "dash": "dash"},
            hoverinfo="skip",
            name="Suppressing",
        )

        node_traces = self._node_traces(positions)

        fig = go.Figure(data=[promote_trace, suppress_trace] + node_traces)
        fig.update_layout(
            title="Attribution graph (real causal flow)",
            showlegend=True,
            xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
            yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
            plot_bgcolor=THOUGHTLENS_COLORS["surface"],
            paper_bgcolor=THOUGHTLENS_COLORS["surface"],
            margin={"t": 60, "b": 30, "l": 30, "r": 30},
            height=440 if self.compact else 620,
        )
        return fig

    def _node_traces(self, positions: dict[int, tuple[float, float]]) -> list:
        import plotly.graph_objects as go

        traces: list = []
        by_type: dict[str, list[CircuitNode]] = defaultdict(list)
        for n in self.graph.nodes():
            if n.id in positions:
                by_type[n.node_type].append(n)

        for node_type, members in by_type.items():
            symbol = _NODE_SYMBOL.get(node_type, "circle")
            colour = node_color(node_type)
            x = [positions[n.id][0] for n in members]
            y = [positions[n.id][1] for n in members]
            text = [n.label or f"id={n.id}" for n in members]
            hover = [
                f"<b>{_html_escape(n.label) or n.id}</b><br>"
                f"type: {n.node_type}<br>"
                f"layer: {n.layer}<br>"
                f"token_idx: {n.token_idx}<br>"
                f"score: {n.score:.3f}<br>"
                f"evidence: {n.evidence_kind}"
                for n in members
            ]
            traces.append(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers+text",
                    text=text,
                    textposition="top center",
                    marker={
                        "size": 14,
                        "symbol": symbol,
                        "color": colour,
                        "line": {"width": 1, "color": "#28251d"},
                    },
                    name=node_type,
                    hoverinfo="text",
                    hovertext=hover,
                )
            )
        return traces

    def to_html(self) -> str:
        return self.to_figure().to_html(full_html=False, include_plotlyjs=False)


def _html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
