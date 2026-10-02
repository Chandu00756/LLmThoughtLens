"""Scorecards for benchmark records: Markdown and a self-contained HTML page.

* :func:`render_markdown` — summary, probe × model score table, metrics,
  probe definitions, errors and environment, as GitHub-flavoured Markdown.
* :func:`render_html` — the same content as one HTML file.  The score
  heatmap is a plain table (sequential single-hue scale, every cell also
  printed as a number and PASS/FAIL, readable without JavaScript).  One
  bar chart (mean score per model) is added: ``plotlyjs="inline"``
  (default) embeds plotly.js so the file is fully self-contained (~5 MB),
  ``"cdn"`` links the bundled plotly.js version from cdn.plot.ly, ``"svg"``
  draws a static inline SVG chart (self-contained, a few KB, no
  JavaScript) and ``"none"`` omits the chart.
* :func:`write_reports` — writes ``<stem>.json``, ``<stem>.md`` and
  ``<stem>.html`` in one call.

Everything taken from model output is HTML-escaped.  Synthetic results are
marked in every view.
"""

from __future__ import annotations

import html
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from LLmThoughtLens.bench.schema import BenchResult

__all__ = ["render_html", "render_markdown", "write_reports"]

PlotlyMode = Literal["inline", "cdn", "svg", "none"]
_PLOTLY_MODES = ("inline", "cdn", "svg", "none")
_N_STEPS = 13  # sequential ramp steps 100 … 700


# ---------------------------------------------------------------------------
# Shared formatting
# ---------------------------------------------------------------------------


def _as_result(result: BenchResult | Mapping[str, Any]) -> BenchResult:
    return result if isinstance(result, BenchResult) else BenchResult(dict(result))


def _fmt(v: Any, nd: int = 2) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _verdict(row: Mapping[str, Any]) -> str:
    if row.get("passed") is None:
        return "ERROR" if row.get("n_error") else "n/a"
    return "PASS" if row["passed"] else "FAIL"


def _score_cell_text(row: Mapping[str, Any] | None, repeats: int) -> str:
    if row is None:
        return "n/a"
    if row.get("mean_score") is None:
        return "error" if row.get("n_error") else "n/a"
    text = f"{row['mean_score']:.2f} {_verdict(row)}"
    if repeats > 1:
        text += f" ±{(row.get('std_score') or 0.0):.2f} ({_fmt(row.get('pass_rate'))} pass)"
    if row.get("n_error"):
        text += f" [{row['n_error']} err]"
    return text


def _tokens(usage: Mapping[str, Any] | None) -> str:
    if not usage or not usage.get("calls"):
        return "n/a"
    pt, ct = usage.get("prompt_tokens"), usage.get("completion_tokens")
    return f"{_fmt(pt)} / {_fmt(ct)}"


#: Keys shown first (in this order) when a metric's mean is a mapping.
_HEADLINE_KEYS = ("spearman", "pearson", "sign_agreement")
#: Bookkeeping keys never shown in the summary cell (still in the JSON record).
_BOOKKEEPING_KEYS = frozenset({"n", "runtime_s", "clean_metric"})


def _mapping_text(mean: Mapping[str, Any], n_items: int | None) -> str:
    keys = [k for k in _HEADLINE_KEYS if k in mean]
    keys += [k for k in sorted(mean) if k not in keys and k not in _BOOKKEEPING_KEYS]
    shown = [f"{k} {_fmt(mean[k], 3)}" for k in keys[:4]]
    text = ", ".join(shown) if shown else "ok"
    return text + (f" (mean of {n_items})" if n_items else "")


def _metric_text(rec: Mapping[str, Any] | None) -> str:
    if not rec:
        return "n/a"
    status = rec.get("status")
    if status == "ok":
        v = rec.get("value")
        if isinstance(v, Mapping) and isinstance(v.get("mean"), Mapping):
            per = v.get("per_prompt")
            return _mapping_text(v["mean"], len(per) if isinstance(per, list) else None)
        if isinstance(v, Mapping) and "mean" in v:
            extra = f" (n={v['n']}/{v['of']})" if "n" in v and "of" in v else ""
            return f"{_fmt(v['mean'], 3)}{extra}"
        return _fmt(v, 3) if not isinstance(v, (dict, list)) else json.dumps(v)[:80]
    return f"{status}: {rec.get('reason') or ''}".strip()


def _model_metric_summary(
    result: BenchResult, name: str, level: str, model: Mapping[str, Any]
) -> str:
    if level == "model":
        return _metric_text(model.get("metrics", {}).get(name))
    recs = [
        c.get("metrics", {}).get(name)
        for c in result.cells
        if c["model"] == model["label"] and c.get("metrics", {}).get(name)
    ]
    means = [
        float(r["value"]["mean"])
        for r in recs
        if r
        and r.get("status") == "ok"
        and isinstance(r.get("value"), Mapping)
        and isinstance(r["value"].get("mean"), (int, float))
    ]
    if means:
        return f"{sum(means) / len(means):.3f} (mean of {len(means)} cells)"
    reasons = sorted({str(r.get("reason")) for r in recs if r and r.get("reason")})
    return f"skipped: {reasons[0]}" if reasons else "n/a"


def _env_lines(env: Mapping[str, Any]) -> list[tuple[str, str]]:
    if not env:
        return [("environment", "not captured")]
    git = env.get("git") or {}
    commit = git.get("commit")
    commit_text = (
        "n/a" if not commit else commit[:12] + (" (dirty tree)" if git.get("dirty") else "")
    )
    devices = env.get("devices") or {}
    dev_text = (
        ", ".join(f"{k}={v}" for k, v in devices.items() if k != "error") if devices else "n/a"
    )
    pkgs = env.get("packages") or {}
    pkg_text = ", ".join(f"{k} {v}" for k, v in pkgs.items() if v) or "n/a"
    return [
        ("captured", str(env.get("captured_utc"))),
        ("python", f"{env.get('python')} ({env.get('python_implementation')})"),
        ("platform", f"{env.get('platform')} / {env.get('machine')}"),
        ("cpu_count", str(env.get("cpu_count"))),
        ("devices", dev_text),
        ("git commit", commit_text),
        ("HF_HUB_OFFLINE", str(env.get("hf_hub_offline"))),
        ("packages", pkg_text),
    ]


def _md_escape(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def render_markdown(result: BenchResult | Mapping[str, Any]) -> str:
    """Render a benchmark record as a Markdown scorecard."""
    res = _as_result(result)
    d = res.data
    cfg = d.get("config", {})
    repeats = int(cfg.get("repeats", 1))
    models = res.models
    labels = [m["label"] for m in models]
    by_model = {r["model"]: r for r in res.aggregates.get("by_model", [])}
    git = (d.get("environment") or {}).get("git") or {}
    out: list[str] = [f"# {d.get('title', 'Benchmark scorecard')}", ""]
    meta = [
        f"Run `{d.get('run_id')}`",
        str(d.get("created_utc")),
        f"schema `{d.get('schema')}` v{d.get('schema_version')}",
        f"{repeats} repeat(s), seeds {cfg.get('seeds')}",
        f"temperature {cfg.get('temperature')}",
        f"total {_fmt(d.get('duration_s'), 1)} s",
    ]
    if git.get("commit"):
        meta.append(
            f"commit `{git['commit'][:12]}`" + (" (dirty tree)" if git.get("dirty") else "")
        )
    out += [" · ".join(meta), ""]
    for note in d.get("notes", []):
        out.append(f"> {note}")
    if d.get("notes"):
        out.append("")

    out += ["## Models", ""]
    out.append(
        "| Model | Evidence | Framing | Probes passed | Mean score | Error cells "
        "| Load (s) | Run (s) | Tokens in / out | Cost (USD) |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for m in models:
        agg = by_model.get(m["label"], {})
        name = m["label"] + (" (synthetic)" if m.get("synthetic") else "")
        if m.get("status") == "error":
            name += " (failed to load)"
        framing = m.get("framing")
        framing_text = ", ".join(framing) if isinstance(framing, list) else _fmt(framing)
        out.append(
            "| "
            + " | ".join(
                _md_escape(x)
                for x in (
                    name,
                    _fmt(m.get("evidence_kind")),
                    framing_text,
                    f"{agg.get('n_passed', 0)} / {agg.get('n_scored', 0)}",
                    _fmt(agg.get("mean_score")),
                    _fmt(agg.get("n_error_cells", 0)),
                    _fmt(m.get("load_s"), 1),
                    _fmt(m.get("duration_s"), 1),
                    _tokens(m.get("usage")),
                    _fmt(m.get("cost_usd"), 4),
                )
            )
            + " |"
        )
    out.append("")

    out += ["## Scores by probe", ""]
    out.append(
        "Mean score over successful repeats; PASS / FAIL = passed in a majority of repeats"
        + ("; ± = standard deviation across repeats." if repeats > 1 else ".")
    )
    out.append("")
    synthetic_labels = {m["label"] for m in models if m.get("synthetic")}
    heads = [x + (" (synthetic)" if x in synthetic_labels else "") for x in labels]
    out.append("| Probe | Style | " + " | ".join(_md_escape(x) for x in heads) + " |")
    out.append("|---|---|" + "---|" * len(labels))
    for p in res.probes:
        row_cells = [_score_cell_text(res.aggregate(lbl, p["name"]), repeats) for lbl in labels]
        out.append(
            f"| `{p['name']}` | {p.get('style', '')} | "
            + " | ".join(_md_escape(x) for x in row_cells)
            + " |"
        )
    out.append("")

    levels = (d.get("metrics") or {}).get("levels", {})
    if levels:
        out += ["## Metrics", ""]
        out.append("| Metric | Level | " + " | ".join(_md_escape(x) for x in labels) + " |")
        out.append("|---|---|" + "---|" * len(labels))
        for name, level in levels.items():
            vals = [_model_metric_summary(res, name, level, m) for m in models]
            out.append(f"| `{name}` | {level} | " + " | ".join(_md_escape(v) for v in vals) + " |")
        out.append("")

    out += ["## Probe definitions", "", "| Probe | Pass rule | Source |", "|---|---|---|"]
    for p in res.probes:
        out.append(
            f"| `{p['name']}` | {_md_escape(p.get('threshold') or p.get('description', ''))} "
            f"| {_md_escape(p.get('citation', ''))} |"
        )
    out.append("")

    errors = [c for c in res.cells if c["status"] == "error"]
    failed_models = [m for m in models if m.get("status") == "error"]
    if errors or failed_models:
        out += ["## Errors", ""]
        for m in failed_models:
            err = m.get("error") or {}
            out.append(
                f"- **{_md_escape(m['label'])}** failed to load: "
                f"`{err.get('type')}`: {_md_escape(err.get('message', ''))}"
            )
        for c in errors:
            if any(m["label"] == c["model"] for m in failed_models):
                continue
            err = c.get("error") or {}
            out.append(
                f"- `{_md_escape(c['model'])}` / `{c['probe']}` (repeat {c['repeat']}): "
                f"`{err.get('type')}`: {_md_escape(err.get('message', ''))}"
            )
        out.append("")

    out += ["## Environment", ""]
    for k, v in _env_lines(d.get("environment") or {}):
        out.append(f"- **{k}**: {_md_escape(v)}")
    for m in models:
        if m.get("server_info"):
            si = m["server_info"]
            out.append(
                f"- **{_md_escape(m['label'])} server**: Ollama {si.get('version')} "
                f"(logprobs supported: {_fmt(si.get('logprobs_supported'))}, "
                f"thinking model: {_fmt(si.get('thinking_model'))})"
            )
        if m.get("model_info"):
            mi = m["model_info"]
            out.append(
                f"- **{_md_escape(m['label'])} model**: family {mi.get('family')}, "
                f"{mi.get('n_layers')} layers, d_model {mi.get('d_model')}, "
                f"{_fmt(mi.get('n_params'))} params, {mi.get('dtype')} on {mi.get('device')}, "
                f"chat template: {_fmt(mi.get('chat_template'))}"
            )
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

_CSS = """
:root {
  color-scheme: light;
  --surface-0: #f5f5f3; --surface-1: #fcfcfb; --border: #dddcd7;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #6b6a66;
  --series-1: #2a78d6; --callout: #fff7e0; --callout-border: #eda100;
  --error-bg: #f0efec;
  --s0:#cde2fb; --s1:#b7d3f6; --s2:#9ec5f4; --s3:#86b6ef; --s4:#6da7ec; --s5:#5598e7;
  --s6:#3987e5; --s7:#2a78d6; --s8:#256abf; --s9:#1c5cab; --s10:#184f95; --s11:#104281;
  --s12:#0d366b; --ink-light:#0b0b0b; --ink-dark:#ffffff;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface-0: #121211; --surface-1: #1a1a19; --border: #383835;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #a3a29a;
    --series-1: #3987e5; --callout: #2b2615; --callout-border: #c98500; --error-bg: #383835;
    --s0:#0d366b; --s1:#104281; --s2:#184f95; --s3:#1c5cab; --s4:#256abf; --s5:#2a78d6;
    --s6:#3987e5; --s7:#5598e7; --s8:#6da7ec; --s9:#86b6ef; --s10:#9ec5f4; --s11:#b7d3f6;
    --s12:#cde2fb; --ink-light:#ffffff; --ink-dark:#0b0b0b;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-0: #121211; --surface-1: #1a1a19; --border: #383835;
  --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #a3a29a;
  --series-1: #3987e5; --callout: #2b2615; --callout-border: #c98500; --error-bg: #383835;
  --s0:#0d366b; --s1:#104281; --s2:#184f95; --s3:#1c5cab; --s4:#256abf; --s5:#2a78d6;
  --s6:#3987e5; --s7:#5598e7; --s8:#6da7ec; --s9:#86b6ef; --s10:#9ec5f4; --s11:#b7d3f6;
  --s12:#cde2fb; --ink-light:#ffffff; --ink-dark:#0b0b0b;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface-0); color: var(--text-primary);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 1.6rem; margin: 0 0 4px; }
h2 { font-size: 1.15rem; margin: 32px 0 8px; }
.meta { color: var(--text-secondary); font-size: 0.9rem; }
.callout { background: var(--callout); border-left: 4px solid var(--callout-border);
  padding: 8px 12px; margin: 16px 0; border-radius: 4px; }
.callout p { margin: 4px 0; }
.card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px;
  padding: 12px; overflow-x: auto; }
table { border-collapse: separate; border-spacing: 2px; width: 100%; font-size: 0.88rem; }
th, td { padding: 6px 8px; text-align: left; vertical-align: top; }
th { color: var(--text-secondary); font-weight: 600; border-bottom: 1px solid var(--border); }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.heat td.cell { text-align: center; font-variant-numeric: tabular-nums; border-radius: 4px;
  min-width: 92px; }
.heat td.err { background: var(--error-bg); color: var(--text-secondary); }
.tag { display: inline-block; font-size: 0.75rem; padding: 0 6px; border-radius: 4px;
  border: 1px solid var(--border); color: var(--text-secondary); }
.legend { display: flex; align-items: center; gap: 8px; color: var(--text-secondary);
  font-size: 0.8rem; margin-top: 8px; }
.legend .ramp { display: flex; }
.legend .ramp span { width: 18px; height: 12px; }
details { margin: 8px 0; }
summary { cursor: pointer; color: var(--text-secondary); }
code { font-size: 0.85em; }
dl { display: grid; grid-template-columns: max-content 1fr; gap: 4px 16px; margin: 0; }
dt { color: var(--text-secondary); }
dd { margin: 0; overflow-wrap: anywhere; }
#bar { width: 100%; height: 320px; }
footer { color: var(--text-muted); font-size: 0.8rem; margin-top: 32px; }
"""


def _step(score: float) -> int:
    return max(0, min(_N_STEPS - 1, round(float(score) * (_N_STEPS - 1))))


def _heat_cell(row: Mapping[str, Any] | None, repeats: int) -> str:
    if row is None or row.get("mean_score") is None:
        label = "error" if row and row.get("n_error") else "n/a"
        return f'<td class="cell err">{label}</td>'
    k = _step(row["mean_score"])
    ink = "var(--ink-dark)" if k >= 7 else "var(--ink-light)"
    tip = (
        f"{row['model']} · {row['probe']}: mean {row['mean_score']:.3f}, "
        f"pass rate {_fmt(row.get('pass_rate'))}, n_ok {row['n_ok']}/{row['n']}"
    )
    if row.get("synthetic"):
        tip += " (synthetic)"
    body = f"{row['mean_score']:.2f}<br><small>{_verdict(row)}"
    if repeats > 1:
        body += f" ±{(row.get('std_score') or 0.0):.2f}"
    body += "</small>"
    return (
        f'<td class="cell" style="background:var(--s{k});color:{ink}" '
        f'title="{html.escape(tip)}">{body}</td>'
    )


def _plotly_script(mode: PlotlyMode) -> str:
    if mode == "inline":
        import plotly.offline

        return f"<script>{plotly.offline.get_plotlyjs()}</script>"
    if mode == "cdn":
        import plotly.offline

        version = plotly.offline.get_plotlyjs_version()
        return f'<script src="https://cdn.plot.ly/plotly-{version}.min.js"></script>'
    return ""


def _svg_bar_chart(labels: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    """Static horizontal bar chart of mean score per model (no JavaScript).

    Colours come from the page's CSS tokens, so it follows light / dark mode.
    """
    bar_h, gap, label_w, plot_w, pad = 22, 14, 220, 520, 8
    height = pad * 2 + len(labels) * (bar_h + gap) + 28
    width = label_w + plot_w + 140
    out = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Mean probe score per model" style="width:100%;height:auto;max-width:'
        f'{width}px">'
    ]
    axis_y = pad + len(labels) * (bar_h + gap)
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = label_w + tick * plot_w
        out.append(
            f'<line x1="{x:.1f}" y1="{pad}" x2="{x:.1f}" y2="{axis_y}" '
            'style="stroke:var(--border)" stroke-width="1"/>'
            f'<text x="{x:.1f}" y="{axis_y + 16}" text-anchor="middle" font-size="12" '
            f'style="fill:var(--text-secondary)">{tick:.2f}</text>'
        )
    for i, (label, row) in enumerate(zip(labels, rows, strict=True)):
        y = pad + i * (bar_h + gap)
        mean = row.get("mean_score")
        out.append(
            f'<text x="{label_w - 8}" y="{y + bar_h / 2 + 4:.1f}" text-anchor="end" '
            f'font-size="13" style="fill:var(--text-primary)">{html.escape(str(label))}</text>'
        )
        if mean is None:
            out.append(
                f'<text x="{label_w + 4}" y="{y + bar_h / 2 + 4:.1f}" font-size="12" '
                'style="fill:var(--text-muted)">no successful cells</text>'
            )
            continue
        w = max(0.0, min(1.0, float(mean))) * plot_w
        note = f"{float(mean):.2f} · {row.get('n_passed', 0)}/{row.get('n_scored', 0)} passed"
        out.append(
            f'<rect x="{label_w}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="3" '
            f'style="fill:var(--series-1)"><title>{html.escape(str(label))}: '
            f"{html.escape(note)}</title></rect>"
            f'<text x="{label_w + w + 6:.1f}" y="{y + bar_h / 2 + 4:.1f}" font-size="12" '
            f'style="fill:var(--text-secondary)">{html.escape(note)}</text>'
        )
    out.append("</svg>")
    return "".join(out)


def render_html(result: BenchResult | Mapping[str, Any], *, plotlyjs: PlotlyMode = "inline") -> str:
    """Render a benchmark record as a self-contained HTML scorecard."""
    if plotlyjs not in _PLOTLY_MODES:
        raise ValueError(f"plotlyjs must be one of {_PLOTLY_MODES}, got {plotlyjs!r}")
    res = _as_result(result)
    d = res.data
    e = html.escape
    cfg = d.get("config", {})
    repeats = int(cfg.get("repeats", 1))
    models = res.models
    labels = [m["label"] for m in models]
    by_model = {r["model"]: r for r in res.aggregates.get("by_model", [])}
    git = (d.get("environment") or {}).get("git") or {}
    parts: list[str] = []

    meta = [
        f"run <code>{e(str(d.get('run_id')))}</code>",
        e(str(d.get("created_utc"))),
        f"schema <code>{e(str(d.get('schema')))}</code> v{e(str(d.get('schema_version')))}",
        f"{repeats} repeat(s), temperature {e(str(cfg.get('temperature')))}",
        f"{_fmt(d.get('duration_s'), 1)} s",
    ]
    if git.get("commit"):
        meta.append(
            f"commit <code>{e(git['commit'][:12])}</code>"
            + (" (dirty tree)" if git.get("dirty") else "")
        )
    parts.append(f"<h1>{e(str(d.get('title', 'Benchmark scorecard')))}</h1>")
    parts.append(f'<p class="meta">{" · ".join(meta)}</p>')
    if d.get("notes"):
        parts.append(
            '<div class="callout">' + "".join(f"<p>{e(str(n))}</p>" for n in d["notes"]) + "</div>"
        )

    # Models table
    rows = []
    for m in models:
        agg = by_model.get(m["label"], {})
        tags = ""
        if m.get("synthetic"):
            tags += ' <span class="tag">synthetic</span>'
        if m.get("status") == "error":
            tags += ' <span class="tag">failed to load</span>'
        framing = m.get("framing")
        framing_text = ", ".join(framing) if isinstance(framing, list) else _fmt(framing)
        rows.append(
            "<tr>"
            f"<td>{e(m['label'])}{tags}</td>"
            f"<td>{e(_fmt(m.get('evidence_kind')))}</td>"
            f"<td>{e(framing_text)}</td>"
            f'<td class="num">{agg.get("n_passed", 0)} / {agg.get("n_scored", 0)}</td>'
            f'<td class="num">{_fmt(agg.get("mean_score"))}</td>'
            f'<td class="num">{agg.get("n_error_cells", 0)}</td>'
            f'<td class="num">{_fmt(m.get("load_s"), 1)}</td>'
            f'<td class="num">{_fmt(m.get("duration_s"), 1)}</td>'
            f'<td class="num">{e(_tokens(m.get("usage")))}</td>'
            f'<td class="num">{_fmt(m.get("cost_usd"), 4)}</td>'
            "</tr>"
        )
    parts.append(
        '<h2>Models</h2><div class="card"><table><thead><tr><th>Model</th><th>Evidence</th>'
        "<th>Framing</th><th>Probes passed</th><th>Mean score</th><th>Error cells</th>"
        "<th>Load (s)</th><th>Run (s)</th><th>Tokens in / out</th><th>Cost (USD)</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )

    # Heatmap table
    synthetic_labels = {m["label"] for m in models if m.get("synthetic")}
    head = "".join(
        f"<th>{e(x)}"
        + ('<br><span class="tag">synthetic</span>' if x in synthetic_labels else "")
        + "</th>"
        for x in labels
    )
    body = []
    for p in res.probes:
        cells = "".join(_heat_cell(res.aggregate(lbl, p["name"]), repeats) for lbl in labels)
        body.append(
            f"<tr><th><code>{e(p['name'])}</code><br>"
            f'<span class="tag">{e(str(p.get("style", "")))}</span></th>{cells}</tr>'
        )
    ramp = "".join(f'<span style="background:var(--s{k})"></span>' for k in range(_N_STEPS))
    parts.append(
        "<h2>Scores by probe</h2>"
        '<div class="card"><table class="heat"><thead><tr><th>Probe</th>'
        + head
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
        + f'<div class="legend">score 0<span class="ramp">{ramp}</span>1 · PASS / FAIL = '
        "passed in a majority of repeats · hover a cell for details</div></div>"
    )

    # Bar chart (single series: mean score per model)
    if plotlyjs == "svg":
        parts.append(
            '<h2>Mean score per model</h2><div class="card">'
            + _svg_bar_chart(labels, [by_model.get(lbl, {}) for lbl in labels])
            + "</div>"
        )
    elif plotlyjs != "none":
        agg_rows = [by_model.get(lbl, {}) for lbl in labels]
        fig = {
            "data": [
                {
                    "type": "bar",
                    "orientation": "h",
                    "y": labels,
                    "x": [r.get("mean_score") for r in agg_rows],
                    "text": [
                        f"{r.get('n_passed', 0)}/{r.get('n_scored', 0)} passed" for r in agg_rows
                    ],
                    "textposition": "outside",
                    "hovertemplate": "%{y}<br>mean score %{x:.3f}<br>%{text}<extra></extra>",
                }
            ],
            "layout": {
                "xaxis": {"range": [0, 1.15], "title": {"text": "mean probe score"}},
                "yaxis": {"autorange": "reversed", "automargin": True},
                "margin": {"l": 8, "r": 16, "t": 8, "b": 48},
                "bargap": 0.45,
                "showlegend": False,
            },
        }
        parts.append(
            '<h2>Mean score per model</h2><div class="card"><div id="bar"></div></div>'
            + _plotly_script(plotlyjs)
            + "<script>(function(){var fig="
            + json.dumps(fig).replace("</", "<\\/")
            + ";if(!window.Plotly){document.getElementById('bar').textContent="
            "'Chart unavailable (plotly.js not loaded); see the tables.';return;}"
            "var cs=getComputedStyle(document.documentElement);"
            "function v(n){return cs.getPropertyValue(n).trim();}"
            "fig.data[0].marker={color:v('--series-1')};"
            "fig.data[0].textfont={color:v('--text-secondary')};"
            "var ax={color:v('--text-secondary'),gridcolor:v('--border'),zerolinecolor:v('--border')};"
            "Object.assign(fig.layout.xaxis,ax);Object.assign(fig.layout.yaxis,ax);"
            "fig.layout.paper_bgcolor='rgba(0,0,0,0)';fig.layout.plot_bgcolor='rgba(0,0,0,0)';"
            "fig.layout.font={color:v('--text-primary')};"
            "Plotly.newPlot('bar',fig.data,fig.layout,{displayModeBar:false,responsive:true});"
            "})();</script>"
        )

    # Metrics
    levels = (d.get("metrics") or {}).get("levels", {})
    if levels:
        mrows = []
        for name, level in levels.items():
            vals = "".join(
                f"<td>{e(_model_metric_summary(res, name, level, m))}</td>" for m in models
            )
            mrows.append(f"<tr><td><code>{e(name)}</code></td><td>{e(level)}</td>{vals}</tr>")
        parts.append(
            '<h2>Metrics</h2><div class="card"><table><thead><tr><th>Metric</th><th>Level</th>'
            + head
            + "</tr></thead><tbody>"
            + "".join(mrows)
            + "</tbody></table></div>"
        )

    # Probe definitions
    prows = "".join(
        f"<tr><td><code>{e(p['name'])}</code></td><td>{e(str(p.get('description', '')))}</td>"
        f"<td>{e(str(p.get('threshold', '')))}</td><td>{e(str(p.get('citation', '')))}</td></tr>"
        for p in res.probes
    )
    parts.append(
        '<h2>Probe definitions</h2><div class="card"><table><thead><tr><th>Probe</th>'
        "<th>Measures</th><th>Pass rule</th><th>Source</th></tr></thead><tbody>"
        + prows
        + "</tbody></table></div>"
    )

    # Errors
    failed = [m for m in models if m.get("status") == "error"]
    errs = [
        c
        for c in res.cells
        if c["status"] == "error" and not any(m["label"] == c["model"] for m in failed)
    ]
    if failed or errs:
        items = [
            f"<li><b>{e(m['label'])}</b> failed to load: <code>"
            f"{e(str((m.get('error') or {}).get('type')))}</code> "
            f"{e(str((m.get('error') or {}).get('message', '')))}</li>"
            for m in failed
        ] + [
            f"<li><code>{e(c['model'])}</code> / <code>{e(c['probe'])}</code> "
            f"(repeat {c['repeat']}): <code>{e(str((c.get('error') or {}).get('type')))}</code> "
            f"{e(str((c.get('error') or {}).get('message', '')))}</li>"
            for c in errs
        ]
        parts.append(f'<h2>Errors</h2><div class="card"><ul>{"".join(items)}</ul></div>')

    # Cell details
    crow = "".join(
        "<tr>"
        f"<td>{e(c['model'])}</td><td><code>{e(c['probe'])}</code></td>"
        f'<td class="num">{c["repeat"]}</td><td class="num">{_fmt(c.get("score"))}</td>'
        f"<td>{'n/a' if c.get('passed') is None else ('PASS' if c['passed'] else 'FAIL')}</td>"
        f"<td>{e(str(c.get('summary', '')))}</td>"
        f'<td class="num">{_fmt(c.get("duration_s"), 2)}</td>'
        f'<td class="num">{e(_tokens(c.get("usage")))}</td>'
        "</tr>"
        for c in res.cells
    )
    parts.append(
        "<h2>All cells</h2><details><summary>Show every (model, probe, repeat) cell</summary>"
        '<div class="card"><table><thead><tr><th>Model</th><th>Probe</th><th>Repeat</th>'
        "<th>Score</th><th>Verdict</th><th>Summary</th><th>Time (s)</th>"
        "<th>Tokens in / out</th></tr></thead><tbody>" + crow + "</tbody></table></div></details>"
    )

    # Environment
    env_items = "".join(
        f"<dt>{e(k)}</dt><dd>{e(v)}</dd>" for k, v in _env_lines(d.get("environment") or {})
    )
    for m in models:
        if m.get("server_info"):
            env_items += (
                f"<dt>{e(m['label'])} server</dt><dd>"
                f"{e(json.dumps(m['server_info'], ensure_ascii=False))}</dd>"
            )
        if m.get("model_info"):
            env_items += (
                f"<dt>{e(m['label'])} model</dt><dd>"
                f"{e(json.dumps(m['model_info'], ensure_ascii=False))}</dd>"
            )
    parts.append(f'<h2>Environment</h2><div class="card"><dl>{env_items}</dl></div>')
    parts.append(
        "<footer>Generated by LLmThoughtLens.bench. Probe methods, metrics, thresholds and "
        "limitations: docs/benchmarks/methodology.md. Behavioural probes measure outputs, "
        "not mechanisms.</footer>"
    )
    title = e(str(d.get("title", "Benchmark scorecard")))
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{title}</title><style>{_CSS}</style></head><body><main>"
        + "".join(parts)
        + "</main></body></html>"
    )


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def write_reports(
    result: BenchResult | Mapping[str, Any],
    out_dir: str | Path,
    *,
    stem: str = "scorecard",
    plotlyjs: PlotlyMode = "inline",
    formats: Sequence[str] = ("json", "md", "html"),
) -> dict[str, Path]:
    """Write ``<stem>.json`` / ``.md`` / ``.html`` into *out_dir*; return the paths."""
    res = _as_result(result)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    unknown = set(formats) - {"json", "md", "html"}
    if unknown:
        raise ValueError(f"unknown formats {sorted(unknown)}")
    paths: dict[str, Path] = {}
    if "json" in formats:
        paths["json"] = out / f"{stem}.json"
        res.to_json(paths["json"])
    if "md" in formats:
        paths["md"] = out / f"{stem}.md"
        paths["md"].write_text(render_markdown(res), encoding="utf-8")
    if "html" in formats:
        paths["html"] = out / f"{stem}.html"
        paths["html"].write_text(render_html(res, plotlyjs=plotlyjs), encoding="utf-8")
    return paths
