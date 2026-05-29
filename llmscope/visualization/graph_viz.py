"""GraphVisualizer — renders an AttributionGraph as an interactive Plotly figure."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llmscope.circuits.graph import AttributionGraph


class GraphVisualizer:
    """Render an :class:`~llmscope.circuits.graph.AttributionGraph` as a
    Plotly figure.

    Parameters
    ----------
    graph:
        The attribution graph to visualize.
    layout:
        Networkx-style layout name: ``"spring"``, ``"circular"``,
        ``"kamada_kawai"``, or ``"shell"``.
    """

    def __init__(
        self,
        graph: "AttributionGraph",
        layout: str = "spring",
    ) -> None:
        self.graph = graph
        self.layout = layout

    def to_figure(self):
        """Return a ``plotly.graph_objects.Figure`` representing the graph.

        Requires ``plotly`` (included in base dependencies).
        """
        try:
            import plotly.graph_objects as go
        except ImportError as exc:
            raise ImportError("plotly is required for GraphVisualizer.") from exc

        # Simple spring-like layout using node indices as positions
        nodes = list(self.graph.nodes())
        n = len(nodes)
        if n == 0:
            return go.Figure()

        import math

        positions = {
            nid: (math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n))
            for i, nid in enumerate(nodes)
        }

        edge_x, edge_y = [], []
        for edge in self.graph.edges():
            x0, y0 = positions[edge.src]
            x1, y1 = positions[edge.dst]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line={"width": 1, "color": "#888"},
            hoverinfo="none",
        )

        node_x = [positions[n][0] for n in nodes]
        node_y = [positions[n][1] for n in nodes]
        node_trace = go.Scatter(
            x=node_x, y=node_y, mode="markers+text",
            text=[str(n) for n in nodes],
            textposition="top center",
            marker={"size": 12, "color": "royalblue"},
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=self.graph.name or "Attribution Graph",
                showlegend=False,
                xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
                yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
            ),
        )
        return fig

    def to_html(self) -> str:
        """Render to an HTML string."""
        fig = self.to_figure()
        return fig.to_html(full_html=False)
