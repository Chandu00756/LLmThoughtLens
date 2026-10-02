"""GraphVisualizer — render an :class:`AttributionGraph` as a layered Plotly DAG.

Nodes are positioned on an x-axis layer band (input nodes on the left,
output nodes on the right, features in between by their layer attribute)
and a y-axis spread by token position + score.  Edge colour encodes
polarity (teal = promote, magenta = suppress) and edge width encodes
absolute weight magnitude.

When the graph has more than ``max_nodes`` nodes the renderer keeps a
type-aware subset: node scores are only comparable *within* a node type
(input tokens carry a fixed ``1.0``, white-box features a unitless centred
score or a raw norm, the error node raw residual energy), so one global
``|score|`` sort would let input tokens and the error node crowd out the
features the trace is actually about.  See :meth:`GraphVisualizer.select_nodes`.

The title states the edge semantics recorded by the tracer
(``graph.meta["edge_semantics"]``: causal linearised estimate, correlational
activation flow, or causal input masking) instead of a blanket "causal" claim,
and the error node's hover distinguishes uncovered *attribution mass*
(gradient traces) from uncovered *activation energy* (activation flow).
Gradient nodes show their attribution and, when validated, the measured
ablation effect.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import TYPE_CHECKING

from LLmThoughtLens.features.extractor import exclusion_reasons
from LLmThoughtLens.utils.colors import THOUGHTLENS_COLORS, edge_color, node_color
from LLmThoughtLens.visualization.attribution_view import graph_title

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

#: Node types that are always drawn, whatever ``max_nodes`` says.
_ALWAYS_KEEP = ("output_token", "error")
_EXCLUDED_NODE_COLOUR = "#b9b4a7"


class GraphVisualizer:
    """Layered DAG renderer."""

    def __init__(
        self,
        graph: AttributionGraph,
        max_nodes: int | None = 60,
        compact: bool = False,
        excluded: Mapping[int, str] | None = None,
    ) -> None:
        self.graph = graph
        self.max_nodes = max_nodes
        self.compact = compact
        # Token position -> reason it was left out of feature ranking.
        self.excluded: dict[int, str] = (
            dict(excluded)
            if excluded is not None
            else exclusion_reasons([], graph.meta.get("excluded_positions"))
        )

    # ------------------------------------------------------------------
    # Node selection + layout
    # ------------------------------------------------------------------

    def select_nodes(self) -> list[CircuitNode]:
        """The nodes drawn, honouring ``max_nodes`` with a type-aware priority.

        1. every ``output_token`` and ``error`` node (kept even past the budget);
        2. supernodes, then features (and any other internal node type), each
           ranked by ``|score|`` within its own type;
        3. input tokens fill whatever budget is left: first those with edges
           into already-kept nodes (by total ``|weight|`` into them), then
           those at a kept feature's token position, then in prompt order.
        """
        nodes = list(self.graph.nodes())
        if self.max_nodes is None or len(nodes) <= self.max_nodes:
            return nodes

        keep = [n for n in nodes if n.node_type in _ALWAYS_KEEP]
        remaining = int(self.max_nodes) - len(keep)

        def by_score(members: list[CircuitNode]) -> list[CircuitNode]:
            return sorted(members, key=lambda n: (-abs(n.score), n.id))

        supernodes = by_score([n for n in nodes if n.node_type == "supernode"])
        internal = by_score(
            [n for n in nodes if n.node_type not in (*_ALWAYS_KEEP, "supernode", "input_token")]
        )
        for n in (*supernodes, *internal):
            if remaining <= 0:
                break
            keep.append(n)
            remaining -= 1
        if remaining <= 0:
            return keep

        kept_ids = {n.id for n in keep}
        kept_positions = {n.token_idx for n in keep if n.node_type not in _ALWAYS_KEEP}
        flow: dict[int, float] = defaultdict(float)
        for e in self.graph.edges():
            if e.dst in kept_ids:
                flow[e.src] += abs(e.weight)

        def input_rank(n: CircuitNode) -> tuple[int, float, int]:
            tier = 0 if flow.get(n.id, 0.0) > 0.0 else (1 if n.token_idx in kept_positions else 2)
            return (tier, -flow.get(n.id, 0.0), n.token_idx)

        inputs = sorted((n for n in nodes if n.node_type == "input_token"), key=input_rank)
        keep.extend(inputs[:remaining])
        return keep

    def _layout(self) -> dict[int, tuple[float, float]]:
        nodes = self.select_nodes()

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
            title=graph_title(self.graph.meta, str(self.graph.meta.get("evidence_kind", ""))),
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
            colour: str | list[str] = node_color(node_type)
            if node_type == "input_token" and self.excluded:
                colour = [
                    _EXCLUDED_NODE_COLOUR if n.token_idx in self.excluded else node_color(node_type)
                    for n in members
                ]
            x = [positions[n.id][0] for n in members]
            y = [positions[n.id][1] for n in members]
            text = [n.label or f"id={n.id}" for n in members]
            hover = [self._hover(n) for n in members]
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

    def _hover(self, n: CircuitNode) -> str:
        if n.node_type == "error":
            frac = n.meta.get("unexplained_fraction")
            if n.meta.get("error_kind") == "attribution_mass":
                # Gradient traces: uncovered attribution mass, in metric units.
                score_line = f"uncovered attribution mass: {n.score:.4g} (metric units)"
            else:
                # Activation flow: raw residual energy, not comparable with feature scores.
                score_line = f"residual energy: {n.score:.4g}"
            if isinstance(frac, (int, float)):
                score_line += f"<br>unexplained: {100.0 * float(frac):.1f}%"
        else:
            score_line = f"score: {n.score:.3f}"
            attr = n.meta.get("attribution")
            if isinstance(attr, (int, float)):
                score_line += f"<br>attribution: {float(attr):+.4g}"
            patched = n.meta.get("patched_effect")
            if isinstance(patched, (int, float)):
                score_line += f"<br>measured ablation effect: {float(patched):+.4g}"
            if n.meta.get("selected_by") == "attribution":
                score_line += "<br>added by attribution (not an extracted feature)"
        lines = [
            f"<b>{_html_escape(n.label) or n.id}</b>",
            f"type: {n.node_type}",
            f"layer: {n.layer}",
            f"token_idx: {n.token_idx}",
            score_line,
            f"evidence: {n.evidence_kind}",
        ]
        if n.node_type == "input_token" and n.token_idx in self.excluded:
            lines.append(f"excluded: {_html_escape(self.excluded[n.token_idx])}")
        return "<br>".join(lines)

    def to_html(self) -> str:
        return self.to_figure().to_html(full_html=False, include_plotlyjs=False)


def _html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
