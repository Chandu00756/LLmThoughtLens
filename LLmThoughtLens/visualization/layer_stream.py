"""ResidualStreamView — PCA trajectory of a token's residual stream across layers.

The projection is fitted on raw residual activations, so a massive-activation
"attention sink" position (norm 10-50x every other token in GPT-2 / Llama /
Qwen ...) would own the first principal component and squash every other
trajectory into a point.  When no ``focus_tokens`` are given, such positions
(explicit ``exclude_positions`` plus automatically detected outliers, the
same rule the feature extractor uses) are left out of the default focus set
and named in the figure title.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np

from LLmThoughtLens.features.extractor import detect_outlier_positions
from LLmThoughtLens.utils.colors import THOUGHTLENS_COLORS
from LLmThoughtLens.utils.math_utils import pca_2d

if TYPE_CHECKING:
    from LLmThoughtLens.providers.base import ProviderOutput

_DEFAULT_FOCUS = 6


class ResidualStreamView:
    """Render the residual-stream trajectory of one or more tokens.

    Parameters
    ----------
    output:
        White-box provider output with ``activations``.
    focus_tokens:
        Positions to plot.  Used verbatim when given.
    compact:
        Smaller figure for embedding in dashboards.
    exclude_positions:
        Positions to leave out of the *default* focus set (e.g.
        ``graph.meta["excluded_positions"]``), in addition to detected
        massive-activation outliers.
    """

    def __init__(
        self,
        output: ProviderOutput,
        focus_tokens: list[int] | None = None,
        compact: bool = False,
        exclude_positions: Iterable[int] | None = None,
    ) -> None:
        if not output.has_internals or output.activations is None:
            raise ValueError(
                "ResidualStreamView requires a white-box ProviderOutput with activations."
            )
        self.output = output
        self.compact = compact
        #: Positions left out of the default focus set (empty when focus_tokens is given).
        self.skipped_positions: list[int] = []
        if focus_tokens:
            self.focus_tokens = list(focus_tokens)
        else:
            self.focus_tokens, self.skipped_positions = _default_focus(
                output.activations, exclude_positions
            )

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

        title = "Residual stream trajectory (PCA across layers)"
        if self.skipped_positions:
            names = ", ".join(
                f"{p} {self.output.tokens[p]!r}" if 0 <= p < len(self.output.tokens) else str(p)
                for p in self.skipped_positions
            )
            title += (
                f"<br><sup>left out: position {_html_escape(names)} "
                "(massive-activation outlier / excluded; would dominate the PCA)</sup>"
            )
        fig = go.Figure(data=traces)
        fig.update_layout(
            title=title,
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


def _html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _default_focus(
    activations: np.ndarray, exclude_positions: Iterable[int] | None
) -> tuple[list[int], list[int]]:
    """First positions that are neither excluded nor massive-activation outliers.

    Returns ``(focus, skipped)``; falls back to every position (skipping
    nothing) rather than return an empty focus set.
    """
    n_layers, n_tokens = activations.shape[0], activations.shape[1]
    skipped = {p % n_tokens for p in exclude_positions or () if -n_tokens <= p < n_tokens}
    if n_layers:
        norms = np.stack(
            [np.linalg.norm(activations[li].astype(np.float64), axis=-1) for li in range(n_layers)]
        )
        skipped |= set(detect_outlier_positions(norms)[0])
    focus = [t for t in range(n_tokens) if t not in skipped]
    if not focus:
        return list(range(n_tokens))[:_DEFAULT_FOCUS], []
    return focus[:_DEFAULT_FOCUS], sorted(skipped)
