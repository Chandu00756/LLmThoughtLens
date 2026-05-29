# ruff: noqa: UP031  (CSS templates use %-formatting because { } collide with format-string braces)
"""GraphDiff — compare two attribution graphs (baseline vs intervention).

Produces three concrete result buckets per the design document:

* ``added_nodes`` / ``removed_nodes`` — nodes present in only one graph.
* ``added_edges`` / ``removed_edges`` — ``(src, dst)`` pairs present in only one.
* ``changed_edges`` — edges whose weight moved by more than ``threshold``.

Render as JSON via :meth:`as_dict` / :meth:`to_json`, or as a self-contained
HTML fragment via :meth:`to_html` — the latter can be dropped into a
:class:`ReportBuilder` tab as a side-by-side baseline-vs-intervention view.
"""

from __future__ import annotations

import html as _html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from LLmThoughtLens.utils.colors import THOUGHTLENS_COLORS

if TYPE_CHECKING:
    from LLmThoughtLens.circuits.graph import AttributionGraph


@dataclass
class GraphDiff:
    """Structured diff between *a* (baseline) and *b* (intervention).

    Attributes
    ----------
    added_nodes:
        Nodes present in *b* but not in *a*.
    removed_nodes:
        Nodes present in *a* but not in *b*.
    added_edges:
        ``(src, dst)`` pairs present in *b* but not in *a*.
    removed_edges:
        ``(src, dst)`` pairs present in *a* but not in *b*.
    changed_edges:
        ``[(src, dst, weight_a, weight_b)]`` for edges whose weight moved
        more than ``threshold``.
    """

    added_nodes: list[int] = field(default_factory=list)
    removed_nodes: list[int] = field(default_factory=list)
    added_edges: list[tuple[int, int]] = field(default_factory=list)
    removed_edges: list[tuple[int, int]] = field(default_factory=list)
    changed_edges: list[tuple[int, int, float, float]] = field(default_factory=list)
    threshold: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def compute(
        cls,
        a: AttributionGraph,
        b: AttributionGraph,
        threshold: float = 0.0,
    ) -> GraphDiff:
        a_nodes = {n.id for n in a.nodes()}
        b_nodes = {n.id for n in b.nodes()}
        added_nodes = sorted(b_nodes - a_nodes)
        removed_nodes = sorted(a_nodes - b_nodes)

        a_edges = {(e.src, e.dst): e.weight for e in a.edges()}
        b_edges = {(e.src, e.dst): e.weight for e in b.edges()}
        added_edges = sorted(set(b_edges) - set(a_edges))
        removed_edges = sorted(set(a_edges) - set(b_edges))
        changed: list[tuple[int, int, float, float]] = []
        for key in set(a_edges) & set(b_edges):
            wa, wb = a_edges[key], b_edges[key]
            if abs(wa - wb) > threshold:
                changed.append((key[0], key[1], float(wa), float(wb)))
        changed.sort(key=lambda t: abs(t[3] - t[2]), reverse=True)

        return cls(
            added_nodes=added_nodes,
            removed_nodes=removed_nodes,
            added_edges=added_edges,
            removed_edges=removed_edges,
            changed_edges=changed,
            threshold=float(threshold),
            meta={"a": a.name, "b": b.name},
        )

    def summary(self) -> str:
        return (
            f"GraphDiff(+{len(self.added_nodes)}/-{len(self.removed_nodes)} nodes, "
            f"+{len(self.added_edges)}/-{len(self.removed_edges)} edges, "
            f"{len(self.changed_edges)} changed)"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "added_nodes": list(self.added_nodes),
            "removed_nodes": list(self.removed_nodes),
            "added_edges": [list(t) for t in self.added_edges],
            "removed_edges": [list(t) for t in self.removed_edges],
            "changed_edges": [list(t) for t in self.changed_edges],
            "threshold": float(self.threshold),
            "meta": dict(self.meta),
            "summary": self.summary(),
        }

    def to_json(self, path: str | Path | None = None, indent: int = 2) -> str | None:
        text = json.dumps(self.as_dict(), indent=indent)
        if path is None:
            return text
        Path(path).write_text(text)
        return None

    # ------------------------------------------------------------------
    # HTML side-by-side rendering (drop into a ReportBuilder tab)
    # ------------------------------------------------------------------

    def to_html(self) -> str:
        """Self-contained HTML fragment showing added / removed / changed."""
        css = (
            "<style>"
            ".tl-diff{font:inherit;display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;}"
            ".tl-diff h4{margin-bottom:8px;font-size:0.9rem;}"
            ".tl-diff .col{background:%s;border:1px solid #e1ddd1;border-radius:8px;padding:10px 12px;}"
            ".tl-diff .added{border-left:4px solid %s;}"
            ".tl-diff .removed{border-left:4px solid %s;}"
            ".tl-diff .changed{border-left:4px solid %s;}"
            ".tl-diff table{width:100%%;border-collapse:collapse;font-size:0.82rem;}"
            ".tl-diff td,.tl-diff th{padding:3px 6px;border-bottom:1px solid #efeae0;text-align:left;}"
            ".tl-diff th{font-weight:600;color:%s;}"
            ".tl-diff .none{color:%s;font-style:italic;}"
            "</style>"
        ) % (  # noqa: UP031 — CSS uses %% so f-string interpolation would collide
            THOUGHTLENS_COLORS["surface"],
            THOUGHTLENS_COLORS["pass"],
            THOUGHTLENS_COLORS["fail"],
            THOUGHTLENS_COLORS["supernode"],
            THOUGHTLENS_COLORS["text"],
            THOUGHTLENS_COLORS["muted"],
        )

        added_rows = self._rows_for_ids(self.added_nodes) + self._rows_for_edges(self.added_edges)
        removed_rows = self._rows_for_ids(self.removed_nodes) + self._rows_for_edges(
            self.removed_edges
        )
        changed_rows = self._rows_for_changes(self.changed_edges)

        return (
            f"{css}"
            f'<div class="tl-diff">'
            f'<div class="col added"><h4>Added ({len(self.added_nodes)} nodes, '
            f"{len(self.added_edges)} edges)</h4>{added_rows or self._empty()}</div>"
            f'<div class="col removed"><h4>Removed ({len(self.removed_nodes)} nodes, '
            f"{len(self.removed_edges)} edges)</h4>{removed_rows or self._empty()}</div>"
            f'<div class="col changed"><h4>Changed ({len(self.changed_edges)} edges'
            f", |Δ|&gt;{self.threshold:g})</h4>{changed_rows or self._empty()}</div>"
            f"</div>"
        )

    @staticmethod
    def _empty() -> str:
        return "<div class='none'>none</div>"

    @staticmethod
    def _rows_for_ids(ids: list[int]) -> str:
        if not ids:
            return ""
        rows = "".join(f"<tr><td>node {_html.escape(str(i))}</td></tr>" for i in ids)
        return f"<table><thead><tr><th>node</th></tr></thead><tbody>{rows}</tbody></table>"

    @staticmethod
    def _rows_for_edges(edges: list[tuple[int, int]]) -> str:
        if not edges:
            return ""
        rows = "".join(f"<tr><td>{s}</td><td>{d}</td></tr>" for s, d in edges)
        return (
            f"<table><thead><tr><th>src</th><th>dst</th></tr></thead><tbody>{rows}</tbody></table>"
        )

    @staticmethod
    def _rows_for_changes(changes: list[tuple[int, int, float, float]]) -> str:
        if not changes:
            return ""
        rows = "".join(
            f"<tr><td>{s}</td><td>{d}</td><td>{wa:+.3f}</td><td>{wb:+.3f}</td>"
            f"<td>{wb - wa:+.3f}</td></tr>"
            for s, d, wa, wb in changes
        )
        return (
            "<table><thead><tr><th>src</th><th>dst</th><th>baseline</th>"
            "<th>intervention</th><th>Δw</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )

    def __repr__(self) -> str:
        return self.summary()
