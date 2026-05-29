"""ResidualStreamView — PCA trajectory of a token's residual stream across layers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from LLmThoughtLens.utils.colors import THOUGHTLENS_COLORS
from LLmThoughtLens.utils.math_utils import pca_2d

if TYPE_CHECKING:
    from LLmThoughtLens.providers.base import ProviderOutput


class ResidualStreamView:
    """Render the residual-stream trajectory of one or more tokens."""

    def __init__(
        self,
        output: ProviderOutput,
        focus_tokens: list[int] | None = None,
        compact: bool = False,
    ) -> None:
        if not output.has_internals or output.activations is None:
            raise ValueError(
                "ResidualStreamView requires a white-box ProviderOutput with activations."
            )
        self.output = output
        self.focus_tokens = focus_tokens or list(range(output.n_tokens))[:6]
        self.compact = compact

    def to_figure(self):
        try:
            import plotly.graph_objects as go
        except ImportError as exc:  # pragma: no cover
            raise ImportError("plotly is required for ResidualStreamView.") from exc

        acts = self.output.activations
        assert acts is not None
        n_layers = acts.shape[0]

        # Project the union of all (layer, token) activations into 2D so the
        # trajectories share a common axis.  Use SVD-based PCA.
        flat = acts[:, self.focus_tokens, :].reshape(-1, acts.shape[-1])
        if flat.shape[0] < 2:
            return go.Figure()
        coords = pca_2d(flat).reshape(n_layers, len(self.focus_tokens), 2)

        traces: list = []
        colours = ["#01696f", "#d19900", "#4f98a3", "#a12c7b", "#437a22", "#964219"]
        for i, t in enumerate(self.focus_tokens):
            xs = coords[:, i, 0]
            ys = coords[:, i, 1]
            label = self.output.tokens[t] if 0 <= t < len(self.output.tokens) else f"tok{t}"
            traces.append(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines+markers+text",
                    line={"color": colours[i % len(colours)], "width": 2},
                    marker={"size": 8, "color": colours[i % len(colours)]},
                    text=[f"L{lyr}" for lyr in range(n_layers)],
                    textposition="top right",
                    name=f"token '{label}'",
                )
            )

        fig = go.Figure(data=traces)
        fig.update_layout(
            title="Residual stream trajectory (PCA across layers)",
            xaxis={"title": "PC 1", "showgrid": True, "zeroline": True},
            yaxis={"title": "PC 2", "showgrid": True, "zeroline": True},
            plot_bgcolor=THOUGHTLENS_COLORS["surface"],
            paper_bgcolor=THOUGHTLENS_COLORS["surface"],
            margin={"t": 60, "b": 50, "l": 50, "r": 30},
            height=420 if self.compact else 540,
        )
        return fig

    def to_html(self) -> str:
        return self.to_figure().to_html(full_html=False, include_plotlyjs=False)
