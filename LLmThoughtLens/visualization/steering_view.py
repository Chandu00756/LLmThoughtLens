"""SteeringView — baseline vs steered completion, per-step KL and token shifts.

Renders a :class:`~LLmThoughtLens.features.steering.SteeringResult` (or its
``to_dict()`` JSON) for the HTML report: the two completions side by side
with the first diverging token marked, a per-step KL chart and the
first-step promoted / suppressed tokens.

Evidence labels are shown verbatim from the result: the *effect* numbers are
measured on a white-box model under an intervention
(``effect_semantics="causal_intervention"``), while ``note`` says where the
steering direction itself came from (usually correlational statistics such as
a contrast of activations or an SAE decoder column).  ``kl_per_step[i]`` is
teacher-forced along the *baseline* prefix, so it compares like with like
even after the two completions diverge.
"""

from __future__ import annotations

import html
from typing import Any

from LLmThoughtLens.utils.colors import THOUGHTLENS_COLORS

__all__ = ["SteeringView", "steering_dict"]


def steering_dict(result: Any) -> dict[str, Any]:
    """``result.to_dict()`` for a SteeringResult, or *result* itself when it is already a dict."""
    if isinstance(result, dict):
        return result
    to_dict = getattr(result, "to_dict", None)
    if callable(to_dict):
        out = to_dict()
        if isinstance(out, dict):
            return out
    raise TypeError(f"expected a SteeringResult or its to_dict(), got {type(result).__name__}")


class SteeringView:
    """HTML / Plotly rendering of one steering comparison."""

    def __init__(self, result: Any, compact: bool = False) -> None:
        self.data = steering_dict(result)
        self.compact = bool(compact)

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------

    def to_figure(self) -> Any:
        """Per-step ``KL(p_steered || p_baseline)`` (nats) along the baseline completion."""
        import plotly.graph_objects as go

        kl = [float(x) for x in self.data.get("kl_per_step") or []]
        tokens = list((self.data.get("baseline") or {}).get("tokens") or [])
        xs = list(range(len(kl)))
        labels = [tokens[i] if i < len(tokens) else f"step {i}" for i in xs]
        fig = go.Figure(
            go.Bar(
                x=xs,
                y=kl,
                marker_color=THOUGHTLENS_COLORS["accent"],
                text=[repr(t) for t in labels],
                hovertemplate="step %{x} (baseline token %{text}): KL=%{y:.4f} nats<extra></extra>",
            )
        )
        fig.update_layout(
            title="KL(steered || baseline) per step, teacher-forced on the baseline completion",
            xaxis={
                "title": "baseline token",
                "tickmode": "array",
                "tickvals": xs,
                "ticktext": labels,
            },
            yaxis={"title": "KL (nats)"},
            plot_bgcolor=THOUGHTLENS_COLORS["surface"],
            paper_bgcolor=THOUGHTLENS_COLORS["surface"],
            margin={"t": 50, "b": 60, "l": 50, "r": 20},
            height=260 if self.compact else 320,
        )
        return fig

    # ------------------------------------------------------------------
    # HTML
    # ------------------------------------------------------------------

    def _completion(self, which: str, diverged: int | None) -> str:
        trace = self.data.get(which) or {}
        tokens = list(trace.get("tokens") or [])
        if not tokens:
            return f"<em>{html.escape(str(trace.get('text', '')))}</em>"
        parts = []
        for i, tok in enumerate(tokens):
            cls = " class='tl-diverge'" if diverged is not None and i == diverged else ""
            parts.append(f"<span{cls}>{html.escape(str(tok))}</span>")
        return "".join(parts)

    @staticmethod
    def _shift_rows(rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "<tr><td colspan='3'>-</td></tr>"
        return "".join(
            "<tr>"
            f"<td><code>{html.escape(repr(r.get('token', '')))}</code></td>"
            f"<td style='text-align:right'>{float(r.get('p_baseline', 0.0)):.4f}</td>"
            f"<td style='text-align:right'>{float(r.get('p_steered', 0.0)):.4f}</td>"
            "</tr>"
            for r in rows
        )

    def to_html(self, include_figure: bool = True) -> str:
        d = self.data
        esc = html.escape
        diverged = d.get("diverged_at")
        diverged = int(diverged) if isinstance(diverged, int) else None
        vectors = d.get("vectors") or []
        vec_lines = "".join(
            "<li>"
            + esc(
                f"{v.get('name') or 'vector'}: layer {v.get('layer')} {v.get('site', '')}, "
                f"coeff {v.get('coeff')}, positions {v.get('positions')}"
                + (
                    f", added norm {float(v['added_norm']):.3g}"
                    if isinstance(v.get("added_norm"), (int, float))
                    else ""
                )
            )
            + "</li>"
            for v in vectors
        )
        kl = [float(x) for x in d.get("kl_per_step") or []]
        first_kl = d.get("first_step_kl", kl[0] if kl else 0.0)
        mean_kl = d.get("mean_kl", sum(kl) / len(kl) if kl else 0.0)
        figure = ""
        if include_figure and kl:
            try:
                figure = self.to_figure().to_html(full_html=False, include_plotlyjs=False)
            except Exception as exc:  # noqa: BLE001 — the text comparison is still useful
                figure = f"<p>KL chart unavailable: {esc(repr(exc))}</p>"
        labels = (
            f"evidence: {d.get('evidence_kind', 'white_box')} &middot; "
            f"method: {esc(str(d.get('method', 'activation_steering')))} &middot; "
            f"effect: {esc(str(d.get('effect_semantics', 'causal_intervention')))}"
        )
        diverge_text = (
            "the completions are identical"
            if diverged is None
            else f"the completions first differ at generated token {diverged}"
        )
        return (
            "<div class='tl-steer'>"
            f"<p class='tl-steer-labels'>{labels}</p>"
            + (f"<p><em>{esc(str(d.get('note', '')))}</em></p>" if d.get("note") else "")
            + f"<p><b>Prompt:</b> <code>{esc(str(d.get('prompt', '')))}</code></p>"
            + (f"<ul class='tl-steer-vectors'>{vec_lines}</ul>" if vec_lines else "")
            + "<div class='tl-steer-cols'>"
            "<div class='tl-steer-col'><h4>Baseline</h4>"
            f"<div class='tl-steer-text'>{self._completion('baseline', diverged)}</div></div>"
            "<div class='tl-steer-col'><h4>Steered</h4>"
            f"<div class='tl-steer-text'>{self._completion('steered', diverged)}</div></div>"
            "</div>"
            f"<p>KL first step {float(first_kl):.4f} nats &middot; mean {float(mean_kl):.4f} "
            f"nats over {len(kl)} steps &middot; {esc(diverge_text)}.</p>"
            f"{figure}"
            "<div class='tl-steer-cols'>"
            "<div class='tl-steer-col'><h4>Promoted (first step)</h4><table class='tl-faith'>"
            "<thead><tr><th>token</th><th>p baseline</th><th>p steered</th></tr></thead>"
            f"<tbody>{self._shift_rows(list(d.get('promoted') or []))}</tbody></table></div>"
            "<div class='tl-steer-col'><h4>Suppressed (first step)</h4><table class='tl-faith'>"
            "<thead><tr><th>token</th><th>p baseline</th><th>p steered</th></tr></thead>"
            f"<tbody>{self._shift_rows(list(d.get('suppressed') or []))}</tbody></table></div>"
            "</div></div>"
        )
