"""FeatureBrowser — a searchable HTML table of extracted features."""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

from LLmThoughtLens.utils.colors import THOUGHTLENS_COLORS

if TYPE_CHECKING:
    from LLmThoughtLens.features.feature import Feature

_TABLE_CSS = (
    """
.fb-wrap { font-family: inherit; }
.fb-controls { display: flex; gap: 8px; margin-bottom: 12px; align-items: center; }
.fb-controls input, .fb-controls select { padding: 6px 10px; border: 1px solid #ddd; border-radius: 6px; font: inherit; }
.fb-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
.fb-table th, .fb-table td { padding: 6px 10px; border-bottom: 1px solid #ececec; text-align: left; vertical-align: middle; }
.fb-table th { background: #eee9df; cursor: pointer; user-select: none; }
.fb-table tr.fb-row-feature td:nth-child(1) { color: """
    + THOUGHTLENS_COLORS["feature"]
    + """; font-weight: 600; }
.fb-table tr.fb-row-input_token td:nth-child(1) { color: """
    + THOUGHTLENS_COLORS["input_token"]
    + """; }
.fb-bar { display: inline-block; height: 8px; background: """
    + THOUGHTLENS_COLORS["accent"]
    + """; border-radius: 3px; }
.fb-empty { color: """
    + THOUGHTLENS_COLORS["muted"]
    + """; font-style: italic; padding: 12px; }
"""
)


class FeatureBrowser:
    """Searchable / filterable HTML table of features."""

    def __init__(self, features: list[Feature]) -> None:
        self.features = sorted(features, key=lambda f: f.score, reverse=True)

    def to_html(self) -> str:
        rows = []
        max_score = max((abs(f.score) for f in self.features), default=1.0)
        for f in self.features:
            bar_w = int(140 * abs(f.score) / (max_score + 1e-9))
            rows.append(
                f'<tr class="fb-row-{f.node_type}" data-label="{html.escape(f.label)}" '
                f'data-layer="{f.layer}" data-token="{f.token_idx}" '
                f'data-evidence="{f.evidence_kind}" '
                f'data-score="{f.score:.6f}">'
                f"<td>{f.id}</td>"
                f"<td>{html.escape(f.label) or '<em>unlabelled</em>'}</td>"
                f"<td>{f.layer}</td>"
                f"<td>{f.token_idx}</td>"
                f"<td>{f.score:.3f}</td>"
                f'<td><span class="fb-bar" style="width:{bar_w}px"></span></td>'
                f"<td>{f.evidence_kind}</td>"
                f"<td>{f.node_type}</td>"
                f"</tr>"
            )
        if not rows:
            body = '<div class="fb-empty">No features extracted for this trace.</div>'
        else:
            body = (
                '<table class="fb-table" id="fb-table">'
                "<thead><tr>"
                "<th data-key='id'>ID</th>"
                "<th data-key='label'>Label</th>"
                "<th data-key='layer'>Layer</th>"
                "<th data-key='token'>Token</th>"
                "<th data-key='score'>Score</th>"
                "<th>Bar</th>"
                "<th data-key='evidence'>Evidence</th>"
                "<th data-key='type'>Node type</th>"
                "</tr></thead>"
                "<tbody>" + "\n".join(rows) + "</tbody></table>"
            )

        # Layer-band options for the filter dropdown.
        layers = sorted({f.layer for f in self.features})
        layer_opts = "".join(f'<option value="{lyr}">layer {lyr}</option>' for lyr in layers)

        controls = (
            '<div class="fb-controls">'
            '<input id="fb-search" type="search" placeholder="Search labels (regex supported)" '
            'style="flex:1"></input>'
            '<select id="fb-layer-filter"><option value="">all layers</option>'
            + layer_opts
            + "</select>"
            '<select id="fb-evidence-filter">'
            '<option value="">all evidence</option>'
            '<option value="white_box">white_box</option>'
            '<option value="black_box">black_box</option>'
            "</select>"
            '<select id="fb-type-filter">'
            '<option value="">all node types</option>'
            '<option value="feature">feature</option>'
            '<option value="input_token">input_token</option>'
            '<option value="output_token">output_token</option>'
            '<option value="supernode">supernode</option>'
            '<option value="error">error</option>'
            "</select>"
            "</div>"
        )

        return f'<style>{_TABLE_CSS}</style><div class="fb-wrap">{controls}{body}</div>'

    # ------------------------------------------------------------------
    # JS that drives search / filter / sort — kept here so the orchestrator
    # can stitch it into the report once globally.
    # ------------------------------------------------------------------

    @staticmethod
    def js() -> str:
        return r"""
function tlsFeatureBrowser() {
  const tbl = document.getElementById('fb-table');
  if (!tbl) return;
  const search = document.getElementById('fb-search');
  const layerSel = document.getElementById('fb-layer-filter');
  const evSel = document.getElementById('fb-evidence-filter');
  const typeSel = document.getElementById('fb-type-filter');
  const rows = Array.from(tbl.querySelectorAll('tbody tr'));

  function refresh() {
    let pattern = null;
    if (search.value) {
      try { pattern = new RegExp(search.value, 'i'); } catch (_) { pattern = null; }
    }
    const layerFilter = layerSel.value;
    const evFilter = evSel.value;
    const typeFilter = typeSel.value;
    for (const r of rows) {
      let show = true;
      if (pattern && !pattern.test(r.dataset.label)) show = false;
      if (layerFilter && r.dataset.layer !== layerFilter) show = false;
      if (evFilter && r.dataset.evidence !== evFilter) show = false;
      if (typeFilter) {
        const tt = r.className.replace(/^.*fb-row-/, '');
        if (tt !== typeFilter) show = false;
      }
      r.style.display = show ? '' : 'none';
    }
  }
  search.addEventListener('input', refresh);
  layerSel.addEventListener('change', refresh);
  evSel.addEventListener('change', refresh);
  typeSel.addEventListener('change', refresh);

  let sortKey = null;
  let sortAsc = true;
  tbl.querySelectorAll('th[data-key]').forEach(th => {
    th.addEventListener('click', () => {
      const key = th.dataset.key;
      if (sortKey === key) sortAsc = !sortAsc; else { sortKey = key; sortAsc = true; }
      const tbody = tbl.querySelector('tbody');
      const sorted = rows.slice().sort((a, b) => {
        let av, bv;
        if (key === 'score' || key === 'layer' || key === 'token') {
          av = parseFloat(a.dataset[key === 'score' ? 'score' : key]);
          bv = parseFloat(b.dataset[key === 'score' ? 'score' : key]);
        } else {
          av = a.dataset[key] || a.children[0].textContent;
          bv = b.dataset[key] || b.children[0].textContent;
        }
        if (av < bv) return sortAsc ? -1 : 1;
        if (av > bv) return sortAsc ? 1 : -1;
        return 0;
      });
      sorted.forEach(r => tbody.appendChild(r));
    });
  });
}
"""
