"""TokenHeatmap — renders token-level feature activation as a Plotly heatmap.

Each token cell is shaded by the sum of feature activation magnitudes on it.
Hovering a token shows its top active features with scores.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llmscope.features.feature import Feature
    from llmscope.providers.base import ProviderOutput

# ThoughtLens design-system colours
_HEATMAP_LOW = "#f7f6f2"
_HEATMAP_HIGH = "#01696f"
_SAFETY_COLOR = "#964219"
_UNCERTAINTY_COLOR = "#d19900"

_SAFETY_KEYWORDS = {"cannot", "can't", "refuse", "harm", "dangerous", "unsafe"}
_UNCERTAINTY_KEYWORDS = {"maybe", "uncertain", "unknown", "possibly", "might"}


class TokenHeatmap:
    """Render a token-level feature activation heatmap.

    Parameters
    ----------
    output:
        :class:`~llmscope.providers.base.ProviderOutput` supplying tokens and activations.
    features:
        Extracted feature list (used to compute per-token activation scores).
    top_n:
        Number of top features to list in each token tooltip.
    """

    def __init__(
        self,
        output: "ProviderOutput",
        features: list["Feature"],
        top_n: int = 5,
    ) -> None:
        self.output = output
        self.features = features
        self.top_n = top_n

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _token_scores(self) -> list[float]:
        """Compute per-token activation magnitude (sum of feature scores)."""
        token_agg: dict[int, float] = {}
        for feat in self.features:
            token_agg[feat.token_idx] = token_agg.get(feat.token_idx, 0.0) + feat.score

        n = len(self.output.tokens)
        return [token_agg.get(i, 0.0) for i in range(n)]

    def _token_top_features(self) -> list[list[Feature]]:
        """Return top-n features for each token position."""
        by_token: dict[int, list[Feature]] = {}
        for feat in self.features:
            by_token.setdefault(feat.token_idx, []).append(feat)

        result = []
        for i in range(len(self.output.tokens)):
            feats = sorted(by_token.get(i, []), key=lambda f: f.score, reverse=True)
            result.append(feats[: self.top_n])
        return result

    def _overlay_color(self, token: str) -> str | None:
        """Return overlay colour for special tokens (safety / uncertainty)."""
        lower = token.lower()
        if any(kw in lower for kw in _SAFETY_KEYWORDS):
            return _SAFETY_COLOR
        if any(kw in lower for kw in _UNCERTAINTY_KEYWORDS):
            return _UNCERTAINTY_COLOR
        return None

    # ------------------------------------------------------------------
    # Plotly figure
    # ------------------------------------------------------------------

    def to_figure(self):
        """Return an interactive ``plotly.graph_objects.Figure``.

        Returns a 1-row heatmap where each column is a token.  Hovering shows
        the top active features and their scores.

        Raises
        ------
        ImportError
            If ``plotly`` is not installed.
        """
        try:
            import plotly.graph_objects as go
        except ImportError as exc:
            raise ImportError("plotly is required for TokenHeatmap.") from exc

        tokens = self.output.tokens
        scores = self._token_scores()
        top_feats = self._token_top_features()

        # Normalise scores to [0, 1] for colour mapping
        max_score = max(scores) if scores else 1.0
        norm_scores = [s / (max_score + 1e-9) for s in scores]

        hover_texts = []
        for i, tok in enumerate(tokens):
            lines = [f"<b>{tok}</b>  (score: {scores[i]:.3f})"]
            for f in top_feats[i]:
                lines.append(f"  {f.label}: {f.score:.3f}")
            hover_texts.append("<br>".join(lines))

        fig = go.Figure(
            data=go.Heatmap(
                z=[norm_scores],
                x=tokens,
                y=["tokens"],
                colorscale=[[0, _HEATMAP_LOW], [1, _HEATMAP_HIGH]],
                showscale=True,
                colorbar={"title": "Activation"},
                hovertext=[hover_texts],
                hovertemplate="%{hovertext}<extra></extra>",
                xgap=2,
                ygap=2,
            )
        )

        # Add overlay annotations for safety / uncertainty tokens
        annotations = []
        for i, tok in enumerate(tokens):
            overlay = self._overlay_color(tok)
            if overlay:
                annotations.append(
                    dict(
                        x=tok,
                        y="tokens",
                        text="⚠",
                        showarrow=False,
                        font={"size": 12, "color": overlay},
                    )
                )

        fig.update_layout(
            title="Token Feature Activation Heatmap",
            xaxis={
                "title": "Input Tokens",
                "tickangle": -30,
                "showgrid": False,
            },
            yaxis={"showticklabels": False, "showgrid": False},
            plot_bgcolor="#f9f8f5",
            paper_bgcolor="#f9f8f5",
            annotations=annotations,
            height=200,
        )
        return fig

    def to_html(self) -> str:
        """Return the heatmap as an HTML fragment."""
        return self.to_figure().to_html(full_html=False)
