"""LLmThoughtLens design system — single source of truth for every visualisation colour.

Mirrors the palette specified in the design document so the HTML report,
Plotly figures, and the Textual TUI share a coherent look.
"""

from __future__ import annotations

THOUGHTLENS_COLORS: dict[str, str] = {
    "input_token": "#4f98a3",
    "feature": "#01696f",
    "supernode": "#d19900",
    "output_token": "#437a22",
    "safety_feature": "#964219",
    "suppressor": "#a12c7b",
    "error_term": "#a13544",
    "edge_promoting": "rgba(1, 105, 111, 0.7)",
    "edge_suppressing": "rgba(161, 44, 123, 0.7)",
    "edge_neutral": "rgba(122, 121, 116, 0.45)",
    "heatmap_low": "#f7f6f2",
    "heatmap_high": "#01696f",
    "heatmap_safety": "#964219",
    "heatmap_uncertainty": "#d19900",
    "bg": "#f7f6f2",
    "surface": "#f9f8f5",
    "text": "#28251d",
    "muted": "#7a7974",
    "accent": "#01696f",
    "pass": "#1a6b1a",
    "fail": "#842029",
}


def heatmap_colorscale() -> list[list[float | str]]:
    """Return a Plotly colour scale for activation heatmaps (low→high)."""
    return [
        [0.0, THOUGHTLENS_COLORS["heatmap_low"]],
        [0.4, "#bcd3d4"],
        [0.75, "#4f98a3"],
        [1.0, THOUGHTLENS_COLORS["heatmap_high"]],
    ]


def node_color(node_type: str) -> str:
    """Canonical colour for an attribution-graph node type."""
    mapping = {
        "input_token": THOUGHTLENS_COLORS["input_token"],
        "feature": THOUGHTLENS_COLORS["feature"],
        "supernode": THOUGHTLENS_COLORS["supernode"],
        "output_token": THOUGHTLENS_COLORS["output_token"],
        "safety": THOUGHTLENS_COLORS["safety_feature"],
        "suppressor": THOUGHTLENS_COLORS["suppressor"],
        "error": THOUGHTLENS_COLORS["error_term"],
    }
    return mapping.get(node_type, THOUGHTLENS_COLORS["feature"])


def edge_color(weight: float) -> str:
    """Promoting vs suppressing colour for a signed edge weight."""
    return (
        THOUGHTLENS_COLORS["edge_promoting"]
        if weight >= 0
        else THOUGHTLENS_COLORS["edge_suppressing"]
    )
