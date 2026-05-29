"""TokenHeatmap — Plotly heatmap of per-token feature activation.

Each cell in the 1-row heatmap corresponds to one input token; cell colour
encodes the sum of feature scores active on that token.  Hovering shows
the top features at that position with their scores.  Safety-related and
uncertainty-related tokens get an overlay marker in the design-system colours.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from LLmThoughtLens.utils.colors import THOUGHTLENS_COLORS, heatmap_colorscale

if TYPE_CHECKING:
    from LLmThoughtLens.features.feature import Feature
    from LLmThoughtLens.providers.base import ProviderOutput

_SAFETY_KEYWORDS = ("cannot", "can't", "refuse", "harm", "dangerous", "unsafe", "kill", "attack")
_UNCERTAINTY_KEYWORDS = ("maybe", "uncertain", "unknown", "possibly", "might", "perhaps")


class TokenHeatmap:
    """Render the token-level activation heatmap as Plotly HTML."""

    def __init__(
        self,
        output: ProviderOutput,
        features: list[Feature],
        top_n: int = 5,
        compact: bool = False,
    ) -> None:
        self.output = output
        self.features = features
        self.top_n = int(top_n)
        self.compact = bool(compact)

    # ------------------------------------------------------------------
    # Per-token aggregates
    # ------------------------------------------------------------------

    def token_scores(self) -> list[float]:
        """Aggregate score on each token from features + raw activation norm."""
        agg: dict[int, float] = {}
        for f in self.features:
            agg[f.token_idx] = agg.get(f.token_idx, 0.0) + max(0.0, float(f.score))
        n = len(self.output.tokens)
        scores = [agg.get(i, 0.0) for i in range(n)]
        if self.output.has_internals and self.output.activations is not None:
            for i in range(n):
                norms = np.linalg.norm(self.output.activations[:, i, :], axis=-1)
                scores[i] += float(norms.mean())
        return scores

    def token_top_features(self) -> list[list[Feature]]:
        by_token: dict[int, list[Feature]] = {}
        for f in self.features:
            by_token.setdefault(f.token_idx, []).append(f)
        out = []
        for i in range(len(self.output.tokens)):
            ranked = sorted(by_token.get(i, []), key=lambda f: f.score, reverse=True)
            out.append(ranked[: self.top_n])
        return out

    def _overlay_colour(self, token: str) -> str | None:
        lower = token.lower()
        if any(k in lower for k in _SAFETY_KEYWORDS):
            return THOUGHTLENS_COLORS["heatmap_safety"]
        if any(k in lower for k in _UNCERTAINTY_KEYWORDS):
            return THOUGHTLENS_COLORS["heatmap_uncertainty"]
        return None

    # ------------------------------------------------------------------
    # Plotly figure
    # ------------------------------------------------------------------

    def to_figure(self):
        try:
            import plotly.graph_objects as go
        except ImportError as exc:  # pragma: no cover
            raise ImportError("plotly is required for TokenHeatmap.") from exc

        tokens = self.output.tokens or ["<empty>"]
        scores = self.token_scores()
        top_feats = self.token_top_features()
        max_score = max(scores) if scores else 1.0
        norm = [s / (max_score + 1e-9) for s in scores]

        hover = []
        for i, tok in enumerate(tokens):
            lines = [f"<b>{_html_escape(tok)}</b>  (activation: {scores[i]:.3f})"]
            for f in top_feats[i]:
                lines.append(f"  {_html_escape(f.label)}: {f.score:.3f}")
            hover.append("<br>".join(lines))

        fig = go.Figure(
            data=go.Heatmap(
                z=[norm],
                x=tokens,
                y=["tokens"],
                colorscale=heatmap_colorscale(),
                showscale=True,
                colorbar={"title": "Activation"},
                hovertext=[hover],
                hovertemplate="%{hovertext}<extra></extra>",
                xgap=2,
                ygap=2,
            )
        )

        annotations = []
        for tok in tokens:
            colour = self._overlay_colour(tok)
            if colour:
                annotations.append(
                    {
                        "x": tok,
                        "y": "tokens",
                        "text": "⚑",
                        "showarrow": False,
                        "font": {"size": 14, "color": colour},
                    }
                )

        fig.update_layout(
            title="Token feature activation heatmap",
            xaxis={"title": "Input tokens", "tickangle": -25, "showgrid": False},
            yaxis={"showticklabels": False, "showgrid": False},
            plot_bgcolor=THOUGHTLENS_COLORS["surface"],
            paper_bgcolor=THOUGHTLENS_COLORS["surface"],
            annotations=annotations,
            height=200 if not self.compact else 130,
            margin={"t": 60, "b": 50, "l": 30, "r": 30},
        )
        return fig

    def to_html(self) -> str:
        return self.to_figure().to_html(full_html=False, include_plotlyjs=False)


def _html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
