"""TokenHeatmap — Plotly heatmap of per-token feature evidence.

Each cell in the 1-row heatmap corresponds to one input token; cell colour
encodes the sum of the (positive) scores of the features extracted on that
token, so the heatmap shows exactly the evidence the trace is built on.
Hovering shows the top features at that position with their scores.

The colour is never mixed with raw activation-norm units: white-box feature
scores are unitless by default (``scoring="centered"``) while raw residual
norms run into the hundreds or thousands and are dominated by the
massive-activation "attention sink" position.  Positions the extractor left
out of the ranking (recorded in feature ``meta["excluded_positions"]`` /
``graph.meta["excluded_positions"]``) are drawn as grey cells with a hover
note explaining why, rather than silently hot or silently zero.

Safety-related and uncertainty-related tokens get an overlay marker in the
design-system colours.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np

from LLmThoughtLens.features.extractor import exclusion_reasons
from LLmThoughtLens.utils.colors import THOUGHTLENS_COLORS, heatmap_colorscale
from LLmThoughtLens.utils.tokenizer_utils import whitespace_tokens

if TYPE_CHECKING:
    from LLmThoughtLens.features.feature import Feature
    from LLmThoughtLens.providers.base import ProviderOutput

_SAFETY_KEYWORDS = ("cannot", "can't", "refuse", "harm", "dangerous", "unsafe", "kill", "attack")
_UNCERTAINTY_KEYWORDS = ("maybe", "uncertain", "unknown", "possibly", "might", "perhaps")

_EXCLUDED_CELL = "#b9b4a7"

#: Colour-bar caption per extractor scoring method (``Feature.meta["method"]``).
_SCORE_UNITS = {
    "centered_norm": "Σ centred score",
    "l2_norm": "Σ raw ‖h‖",
    "sae": "Σ SAE code",
    "token_masking": "Σ Δp (masking)",
    "position_heuristic": "Σ heuristic score",
}


class TokenHeatmap:
    """Render the token-level feature-evidence heatmap as Plotly HTML.

    Parameters
    ----------
    output:
        The provider output the features were extracted from.
    features:
        The extracted features (any evidence kind).
    top_n:
        Number of features listed per token in the hover text.
    compact:
        Smaller figure for embedding in dashboards.
    excluded_positions:
        Extra positions to treat as excluded (e.g.
        ``graph.meta["excluded_positions"]``).  Positions recorded in the
        features' meta are always honoured.
    """

    def __init__(
        self,
        output: ProviderOutput,
        features: list[Feature],
        top_n: int = 5,
        compact: bool = False,
        excluded_positions: Iterable[int] | None = None,
    ) -> None:
        self.output = output
        self.features = features
        self.top_n = int(top_n)
        self.compact = bool(compact)
        self._exclusions = exclusion_reasons(features, excluded_positions)

    # ------------------------------------------------------------------
    # Per-token aggregates
    # ------------------------------------------------------------------

    def input_tokens(self) -> list[str]:
        """The prompt tokens that ``Feature.token_idx`` indexes into.

        White-box outputs carry the tokenised prompt in ``output.tokens``;
        black-box outputs carry the *completion* there, while the masking
        engine scores whitespace-split prompt tokens (same rule as the tracer).
        """
        if self.output.has_internals:
            return list(self.output.tokens)
        return whitespace_tokens(self.output.prompt) or list(self.output.tokens)

    def excluded_positions(self) -> dict[int, str]:
        """In-range positions left out of feature ranking → human-readable reason."""
        n = len(self.input_tokens())
        return {p: r for p, r in self._exclusions.items() if 0 <= p < n}

    def token_scores(self) -> list[float]:
        """Per-token sum of positive feature scores (the extracted evidence).

        Excluded positions score ``0.0``; they are reported by
        :meth:`excluded_positions` and rendered grey, never coloured.
        """
        n = len(self.input_tokens())
        excluded = self.excluded_positions()
        scores = [0.0] * n
        for f in self.features:
            if 0 <= f.token_idx < n and f.token_idx not in excluded:
                scores[f.token_idx] += max(0.0, float(f.score))
        return scores

    def token_top_features(self) -> list[list[Feature]]:
        by_token: dict[int, list[Feature]] = {}
        for f in self.features:
            by_token.setdefault(f.token_idx, []).append(f)
        out = []
        for i in range(len(self.input_tokens())):
            ranked = sorted(by_token.get(i, []), key=lambda f: f.score, reverse=True)
            out.append(ranked[: self.top_n])
        return out

    def score_unit(self) -> str:
        """Caption for the colour scale, derived from the features' scoring method."""
        methods = {f.meta.get("method") for f in self.features}
        if len(methods) == 1:
            method = next(iter(methods))
            if isinstance(method, str) and method in _SCORE_UNITS:
                return _SCORE_UNITS[method]
        return "Σ feature score"

    def _mean_raw_norms(self) -> np.ndarray | None:
        """Per-token mean residual L2 norm over layers (hover context only)."""
        acts = self.output.activations
        if not self.output.has_internals or acts is None or acts.ndim != 3:
            return None
        if acts.shape[0] == 0 or acts.shape[1] != len(self.input_tokens()):
            return None
        norms = np.stack(
            [np.linalg.norm(acts[layer].astype(np.float64), axis=-1) for layer in range(len(acts))]
        )
        return norms.mean(axis=0)

    def _overlay_colour(self, token: str) -> str | None:
        lower = token.lower()
        if any(k in lower for k in _SAFETY_KEYWORDS):
            return THOUGHTLENS_COLORS["heatmap_safety"]
        if any(k in lower for k in _UNCERTAINTY_KEYWORDS):
            return THOUGHTLENS_COLORS["heatmap_uncertainty"]
        return None

    def _excluded_hover(self, idx: int, token: str, reason: str, raw: np.ndarray | None) -> str:
        lines = [
            f"<b>{_html_escape(token)}</b>",
            f"excluded: {_html_escape(reason)}",
            "left out of feature ranking and layer statistics",
        ]
        if raw is not None:
            others = [float(raw[j]) for j in range(len(raw)) if j not in self._exclusions]
            if others:
                ref = float(np.median(others))
                ratio = f" ({raw[idx] / ref:.1f}x the other tokens' median)" if ref > 0 else ""
                lines.append(f"mean raw ‖h‖ = {raw[idx]:.1f}{ratio}")
        return "<br>".join(lines)

    # ------------------------------------------------------------------
    # Plotly figure
    # ------------------------------------------------------------------

    def to_figure(self):
        try:
            import plotly.graph_objects as go
        except ImportError as exc:  # pragma: no cover
            raise ImportError("plotly is required for TokenHeatmap.") from exc

        tokens = self.input_tokens() or ["<empty>"]
        scores = self.token_scores() or [0.0]
        top_feats = self.token_top_features() or [[]]
        excluded = self.excluded_positions()
        raw = self._mean_raw_norms() if excluded else None
        unit = self.score_unit()
        max_score = max((s for i, s in enumerate(scores) if i not in excluded), default=0.0)

        # Numeric x positions + tick labels: repeated tokens (" the" twice)
        # would otherwise collapse onto one categorical column.
        xs = list(range(len(tokens)))
        z: list[float | None] = []
        z_excl: list[float | None] = []
        hover: list[str] = []
        for i, tok in enumerate(tokens):
            if i in excluded:
                z.append(None)
                z_excl.append(1.0)
                hover.append(self._excluded_hover(i, tok, excluded[i], raw))
                continue
            z.append(scores[i] / max_score if max_score > 0 else 0.0)
            z_excl.append(None)
            lines = [f"<b>{_html_escape(tok)}</b>  ({unit}: {scores[i]:.3f})"]
            for f in top_feats[i]:
                lines.append(f"  {_html_escape(f.label)}: {f.score:.3f}")
            hover.append("<br>".join(lines))

        data: list[Any] = [
            go.Heatmap(
                z=[z],
                x=xs,
                y=["tokens"],
                zmin=0.0,
                zmax=1.0,
                colorscale=heatmap_colorscale(),
                showscale=True,
                colorbar={"title": f"{unit}<br>(÷ max)"},
                hovertext=[hover],
                hovertemplate="%{hovertext}<extra></extra>",
                hoverongaps=False,
                name="evidence",
                xgap=2,
                ygap=2,
            )
        ]
        if excluded:
            data.append(
                go.Heatmap(
                    z=[z_excl],
                    x=xs,
                    y=["tokens"],
                    zmin=0.0,
                    zmax=1.0,
                    colorscale=[[0.0, _EXCLUDED_CELL], [1.0, _EXCLUDED_CELL]],
                    showscale=False,
                    hovertext=[hover],
                    hovertemplate="%{hovertext}<extra></extra>",
                    hoverongaps=False,
                    name="excluded",
                    xgap=2,
                    ygap=2,
                )
            )

        annotations = []
        for i, tok in enumerate(tokens):
            colour = self._overlay_colour(tok)
            if colour:
                annotations.append(
                    {
                        "x": i,
                        "y": "tokens",
                        "text": "⚑",
                        "showarrow": False,
                        "font": {"size": 14, "color": colour},
                    }
                )
            if i in excluded:
                annotations.append(
                    {
                        "x": i,
                        "y": "tokens",
                        "text": "excl.",
                        "showarrow": False,
                        "font": {"size": 10, "color": THOUGHTLENS_COLORS["text"]},
                    }
                )

        title = "Token feature evidence heatmap"
        if excluded:
            notes = ", ".join(f"pos {p} {tokens[p]!r}: {r}" for p, r in excluded.items())
            title += f"<br><sup>grey = left out of feature ranking ({_html_escape(notes)})</sup>"

        fig = go.Figure(data=data)
        fig.update_layout(
            title=title,
            xaxis={
                "title": "Input tokens",
                "tickangle": -25,
                "showgrid": False,
                "tickmode": "array",
                "tickvals": xs,
                "ticktext": tokens,
            },
            yaxis={"showticklabels": False, "showgrid": False},
            plot_bgcolor=THOUGHTLENS_COLORS["surface"],
            paper_bgcolor=THOUGHTLENS_COLORS["surface"],
            annotations=annotations,
            height=(200 if not self.compact else 130) + (20 if excluded else 0),
            margin={"t": 80 if excluded else 60, "b": 50, "l": 30, "r": 30},
        )
        return fig

    def to_html(self) -> str:
        return self.to_figure().to_html(full_html=False, include_plotlyjs=False)


def _html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
