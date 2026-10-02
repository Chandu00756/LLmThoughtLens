/* ThoughtLens Live — dashboard client. Real WebSocket + REST, no framework. */
"use strict";

const $ = (id) => document.getElementById(id);
const COLORS = {
  accent: "#01a0aa", accentDeep: "#01696f", gold: "#d19900",
  input: "#4f98a3", suppress: "#a12c7b", muted: "#9aa595",
};
let lastPayload = null;
let wbSteps = [];

/* ---------------- WebSocket ---------------- */
function connectWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen = () => { $("ws-dot").classList.add("live"); $("ws-label").textContent = "live"; };
  ws.onclose = () => {
    $("ws-dot").classList.remove("live"); $("ws-label").textContent = "reconnecting…";
    setTimeout(connectWS, 1500);
  };
  ws.onmessage = (e) => {
    let ev; try { ev = JSON.parse(e.data); } catch { return; }
    dispatch(ev);
  };
}

function logEvent(ev) {
  const log = $("event-log");
  const row = document.createElement("div");
  row.className = "evrow";
  const t = new Date((ev.ts || Date.now() / 1000) * 1000).toLocaleTimeString();
  row.innerHTML = `<span class="k">${ev.kind}</span> <span class="muted">${t}</span>`;
  log.prepend(row);
  while (log.children.length > 80) log.removeChild(log.lastChild);
}

function dispatch(ev) {
  logEvent(ev);
  switch (ev.kind) {
    case "trace_started": setBadges([{ t: "tracing…", c: "" }], true); break;
    case "trace_complete": renderTrace(ev.data); break;
    case "trace_error": setBadges([{ t: "error: " + (ev.data.error || ""), c: "black_box" }]); break;
    case "proxy_request": appendProxy(ev.data, true); break;
    case "proxy_exchange": appendProxy(ev.data, false); break;
    case "whitebox_started": startWhitebox(ev.data); break;
    case "whitebox_step": stepWhitebox(ev.data); break;
    case "whitebox_complete": finishWhitebox(ev.data); break;
    case "whitebox_error": setBadges([{ t: "white-box error: " + (ev.data.error || ""), c: "black_box" }]); break;
    case "xray_started": startXray(ev.data); break;
    case "xray_step": stepXray(ev.data); break;
    case "xray_complete": finishXray(ev.data); break;
    case "xray_error": setBadges([{ t: "x-ray error: " + (ev.data.error || ""), c: "black_box" }]); break;
    case "steer_started": setBadges([{ t: "steering…", c: "white_box" }], true); break;
    case "steer_complete": renderSteer(ev.data); break;
    case "steer_error": renderSteerError(ev.data.error || "steering failed"); break;
  }
}

/* ---------------- LLM X-ray (logit lens) ---------------- */
let xrayPrev = [];
function startXray(d) {
  switchTab("xray");
  xrayPrev = [];
  $("xray-tokens").innerHTML = "";
  $("xray-lens").innerHTML = '<div class="center-empty"><span class="spinner"></span> reading hidden states…</div>';
  const banner = $("xray-banner");
  banner.style.display = "block";
  banner.innerHTML = `🔬 <b>${escapeHtml(d.model)}</b> on <b>${escapeHtml(d.device)}</b> — logit lens `
    + (d.has_logit_lens ? "active. Watch the prediction climb the layers." : "unavailable for this model.");
  setBadges([{ t: "x-ray: " + (d.model || ""), c: "white_box" }, { t: "thinking…", c: "" }], true);
}
function stepXray(d) {
  // Token stream
  const t = document.createElement("span");
  t.className = "tok"; t.textContent = d.token;
  $("xray-tokens").appendChild(t);

  // Logit-lens column: layer-by-layer top prediction (rendered bottom→top via CSS).
  const lens = $("xray-lens");
  lens.innerHTML = "";
  const finalTok = d.logit_lens.length ? d.logit_lens[d.logit_lens.length - 1].top[0][0] : "";
  d.logit_lens.forEach((entry) => {
    const top = entry.top[0];
    const changed = xrayPrev[entry.layer] !== undefined && xrayPrev[entry.layer] !== top[0];
    const isFinalToken = top[0] === d.token;
    const row = document.createElement("div");
    row.className = "xray-row" + (isFinalToken ? " final" : "") + (changed ? " changed" : "");
    row.innerHTML = `<span class="ly">L${entry.layer}</span>`
      + `<span class="tk">${escapeHtml(top[0])}</span>`
      + `<span class="pb"><span class="pf" style="width:${Math.round(100 * top[1])}%"></span></span>`;
    lens.appendChild(row);
    xrayPrev[entry.layer] = top[0];
  });

  // Activation grid heatmap (layers × tokens). Raw ‖h‖: cap the colour scale
  // at the largest non-sink cell so an attention-sink column cannot flatten
  // every other token into one colour (its hover keeps the true value).
  // Numeric x (token index) + tick labels: Plotly merges categorical axes, so
  // a repeated token (e.g. " the" twice) would otherwise share one column.
  const cap = sinkCap(d.grid);
  const toks = d.tokens || [];
  const idx = toks.map((_, i) => i);
  const tokAxis = { tickmode: "array", tickvals: idx, ticktext: toks.map(String) };
  Plotly.react("xray-grid-plot", [{
    z: d.grid, x: idx, y: d.grid.map((_, i) => "L" + i), type: "heatmap",
    colorscale: [[0, "#0d1b1c"], [0.5, COLORS.accentDeep], [1, COLORS.gold]],
    zmin: cap.zmin, zmax: cap.zmax,
    text: d.grid.map((row) => row.map((_, t) => escapeHtml(toks[t] !== undefined ? toks[t] : "tok" + t) + (cap.sinks.has(t) ? "<br>attention-sink outlier — colour capped" : ""))),
    hovertemplate: "layer %{y}, token %{x} %{text}: ‖h‖=%{z:.1f}<extra></extra>",
  }], plotLayout({ height: 280, margin: { t: 10, b: 60, l: 40, r: 10 }, xaxis: tokAxis }), { displayModeBar: false, responsive: true });

  // Attention heatmap (last layer)
  if (d.attention && d.attention.length) {
    const n = d.attention.length, aIdx = [...Array(n).keys()];
    const lab = (i) => escapeHtml(toks[i] !== undefined ? toks[i] : "tok" + i);
    Plotly.react("xray-attn-plot", [{
      z: d.attention, x: aIdx, y: aIdx, type: "heatmap",
      colorscale: [[0, "#0d1b1c"], [1, COLORS.accent]],
      text: d.attention.map((row, i) => row.map((_, j) => `${i}:${lab(i)} → ${j}:${lab(j)}`)),
      hovertemplate: "%{text}: %{z:.3f}<extra></extra>",
    }], plotLayout({
      height: 280, margin: { t: 10, b: 60, l: 60, r: 10 },
      xaxis: { tickmode: "array", tickvals: aIdx, ticktext: aIdx.map((i) => String(toks[i] !== undefined ? toks[i] : i)) },
      yaxis: { tickmode: "array", tickvals: aIdx, ticktext: aIdx.map((i) => String(toks[i] !== undefined ? toks[i] : i)), autorange: "reversed" },
    }), { displayModeBar: false, responsive: true });
  }
  setBadges([{ t: "x-ray step " + d.step, c: "white_box" }, { t: "→ " + d.token, c: "out" }], true);
}
// Columns whose ‖h‖ exceeds 6× the other tokens' median in ≥30% of layers
// (the extractor's massive-activation rule) → excluded from the colour range.
function sinkCap(grid) {
  const none = { zmin: undefined, zmax: undefined, sinks: new Set() };
  const L = grid.length, T = L ? grid[0].length : 0;
  if (!L || T < 2) return none;
  const median = (a) => { const s = a.slice().sort((x, y) => x - y), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
  const hits = new Array(T).fill(0);
  grid.forEach((row) => row.forEach((v, t) => { const ref = median(row.filter((_, j) => j !== t)); if (ref > 0 && v > 6 * ref) hits[t]++; }));
  const sinks = new Set(hits.map((c, t) => (c >= 0.3 * L ? t : -1)).filter((t) => t >= 0));
  if (!sinks.size || sinks.size === T) return none;
  const rest = grid.flatMap((row) => row.filter((_, t) => !sinks.has(t)));
  return { zmin: Math.min(...rest), zmax: Math.max(...rest), sinks };
}
function finishXray(d) {
  setBadges([{ t: "x-ray complete", c: "white_box" }, { t: "answer: " + (d.completion || "").slice(0, 50), c: "out" }]);
}
async function runXray() {
  const prompt = $("prompt-input").value.trim(); if (!prompt) return;
  const provider = $("provider-select").value;
  if (provider !== "huggingface" && provider !== "mock") {
    // Honest: an API model cannot be opened up — switch to a local one.
    const banner = $("xray-banner");
    banner.style.display = "block";
    banner.innerHTML = "⚠️ The X-ray reads a model's <b>internal weights</b>. "
      + `“${escapeHtml(provider)}” is an API model — it only returns text, so it cannot be opened up. `
      + "Switch the provider to <b>HuggingFace</b> (e.g. <code>gpt2</code>, or your own model) to X-ray it.";
    switchTab("xray");
    return;
  }
  const model = $("model-input").value || "gpt2";
  switchTab("xray");
  $("xray-lens").innerHTML = '<div class="center-empty"><span class="spinner"></span> loading model…</div>';
  await fetch("/api/xray/stream", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_name: model, prompt, max_new_tokens: 12 }),
  });
}
function updateXrayHint() {
  const provider = $("provider-select").value;
  const banner = $("xray-banner");
  if (!banner) return;
  if (provider !== "huggingface" && provider !== "mock") {
    banner.style.display = "block";
    banner.innerHTML = "ℹ️ X-ray needs a <b>local model</b> (its weights). "
      + `“${escapeHtml(provider)}” is an API model — pick <b>HuggingFace</b> to open a model up.`;
  } else {
    banner.style.display = "none";
  }
}

/* ---------------- Badges ---------------- */
function setBadges(items, spinning) {
  const el = $("badges");
  el.innerHTML = "";
  if (spinning) { const s = document.createElement("span"); s.className = "spinner"; el.appendChild(s); }
  for (const it of items) {
    const b = document.createElement("span");
    b.className = "badge " + (it.c || "");
    b.textContent = it.t;
    el.appendChild(b);
  }
}

/* ---------------- Trace rendering ---------------- */
function renderTrace(p) {
  lastPayload = p;
  const ev = p.evidence_kind || "";
  setBadges([
    { t: "output: " + (p.output_token || "—"), c: "out" },
    { t: ev || "—", c: ev },
    { t: (p.model || p.provider || "model"), c: "" },
    { t: (p.features ? p.features.length : 0) + " features", c: "" },
  ].concat(attributionBadges(p)));
  renderAttribution(p);
  renderHeatmap(p);
  renderGraph(p);
  renderFeatures(p);
  renderProbes(p);
}

function plotLayout(extra) {
  const css = getComputedStyle(document.body);
  const paper = css.getPropertyValue("--surface").trim();
  const text = css.getPropertyValue("--text").trim();
  return Object.assign({
    paper_bgcolor: paper, plot_bgcolor: paper,
    font: { color: text, size: 12 },
    margin: { t: 40, b: 50, l: 50, r: 20 }, height: 360,
  }, extra || {});
}

// Positions the extractor left out of the feature ranking (attention sinks).
function excludedOf(p) {
  const meta = (p.graph && p.graph.meta) || {};
  return new Set((p.excluded_positions || meta.excluded_positions || []).map(Number));
}

function renderHeatmap(p) {
  const feats = p.features || [];
  if (!feats.length) return;
  $("heatmap-empty").style.display = "none";
  // input_tokens = what feature.token_idx indexes (summary.tokens is the completion for API models).
  const tokens = p.input_tokens || (p.summary && p.summary.tokens) || [];
  const excluded = excludedOf(p), reasons = p.exclusion_reasons || {};
  // Colour = Σ feature score per token only (never raw activation norms);
  // excluded positions are greyed with a hover note, not silently hot or zero.
  const agg = {};
  for (const f of feats) if (!excluded.has(f.token_idx)) agg[f.token_idx] = (agg[f.token_idx] || 0) + Math.max(0, f.score);
  const n = Math.max(tokens.length, ...Object.keys(agg).map((k) => +k + 1));
  const max = Math.max(1e-9, ...Object.values(agg));
  const xs = [], labels = [], z = [], zEx = [], hover = [];
  for (let i = 0; i < n; i++) {
    const label = tokens[i] !== undefined ? tokens[i] : "tok" + i;
    xs.push(i); labels.push(label);
    const ex = excluded.has(i);
    z.push(ex ? null : (agg[i] || 0) / max); zEx.push(ex ? 1 : null);
    hover.push(ex ? `${escapeHtml(label)}<br>excluded: ${escapeHtml(reasons[i] || "left out of feature ranking")}`
      : `${escapeHtml(label)}: Σ feature score ${(agg[i] || 0).toFixed(3)}`);
  }
  const cell = { x: xs, y: ["evidence"], type: "heatmap", zmin: 0, zmax: 1, xgap: 2, text: [hover], hovertemplate: "%{text}<extra></extra>", hoverongaps: false };
  const traces = [Object.assign({ z: [z], colorscale: [[0, "#0d1b1c"], [0.5, COLORS.accentDeep], [1, COLORS.gold]] }, cell)];
  if (excluded.size) traces.push(Object.assign({ z: [zEx], colorscale: [[0, COLORS.muted], [1, COLORS.muted]], showscale: false }, cell));
  Plotly.react("heatmap-plot", traces, plotLayout({
    height: 200, title: "Per-token feature evidence (Σ feature score" + (excluded.size ? "; grey = excluded attention sink" : "") + ")",
    xaxis: { tickmode: "array", tickvals: xs, ticktext: labels },
  }), { displayModeBar: false, responsive: true });
}

/* ---------------- Attribution method + faithfulness ---------------- */
const SEMANTICS = {
  causal_linearised: { label: "causal · linearised estimate", c: "white_box" },
  correlational: { label: "correlational (not causal)", c: "" },
  causal_input_masking: { label: "causal · input masking", c: "black_box" },
};
const FAITH_CAVEAT = "Predicted = linearised (gradient × activation) estimate; measured = real ablation. "
  + "Nonlinear components (e.g. GPT-2's large layer-0 writes) can make them disagree, so read all four numbers together.";
function fmtCorr(v) { return v === null || v === undefined || !isFinite(v) ? "undefined" : (v >= 0 ? "+" : "") + (+v).toFixed(2); }
function faithText(f) {
  const sign = f.sign_agreement === null || f.sign_agreement === undefined ? "undefined" : Math.round(100 * f.sign_agreement) + "%";
  return `Spearman ${fmtCorr(f.spearman)} · Pearson ${fmtCorr(f.pearson)} · n=${f.n} · sign agreement ${sign}`;
}
function attributionBadges(p) {
  const a = p.attribution || {};
  const out = [];
  if (a.edge_semantics) {
    const s = SEMANTICS[a.edge_semantics] || { label: a.edge_semantics, c: "" };
    out.push({ t: "edges: " + s.label, c: s.c });
  }
  if (a.faithfulness) out.push({ t: "faithfulness: " + faithText(a.faithfulness), c: "" });
  return out;
}
function renderAttribution(p) {
  const box = $("graph-attr"); if (!box) return;
  const a = p.attribution || {};
  if (!a.edge_semantics) { box.style.display = "none"; return; }
  const s = SEMANTICS[a.edge_semantics] || { label: a.edge_semantics, c: "" };
  const rows = [`<span class="badge ${s.c}">edges: ${escapeHtml(s.label)}</span> `
    + `<span class="muted">method ${escapeHtml(a.method || "?")}`
    + (a.metric ? ` · metric ${escapeHtml(a.metric)}` + (a.target_token !== null && a.target_token !== undefined ? ` of ${escapeHtml(JSON.stringify(a.target_token))}` : "") : "")
    + "</span>"];
  if (a.semantics_explanation) rows.push(`<div class="note">${escapeHtml(a.semantics_explanation)}</div>`);
  if (a.method_fallback) rows.push(`<div class="note"><b>Fallback:</b> gradient attribution not used — ${escapeHtml(a.method_fallback)}</div>`);
  if (typeof a.unexplained_fraction === "number") {
    rows.push(`<div class="note"><b>Error node:</b> ${(100 * a.unexplained_fraction).toFixed(1)}% `
      + (a.error_kind === "attribution_mass" ? "of attribution mass not covered by the graph's nodes." : "of residual activation energy not covered by the graph's nodes.") + "</div>");
  }
  if (a.faithfulness) {
    rows.push(`<div><span class="badge faith">faithfulness: ${escapeHtml(faithText(a.faithfulness))}</span></div>`);
    rows.push(`<div class="note">${escapeHtml(a.faithfulness.caveat || FAITH_CAVEAT)}</div>`);
  } else if (a.faithfulness_skipped) {
    rows.push(`<div class="note"><b>Faithfulness:</b> not computed — ${escapeHtml(a.faithfulness_skipped)}</div>`);
  } else if (a.edge_semantics === "causal_linearised") {
    rows.push('<div class="note"><b>Faithfulness:</b> not validated — set “Validate (k ablations)” to compare the estimates with real ablations.</div>');
  }
  (p.sae_warnings || []).forEach((w) => rows.push(`<div class="note"><b>SAE input:</b> ${escapeHtml(w)}</div>`));
  (p.notes || []).forEach((w) => rows.push(`<div class="note"><b>Note:</b> ${escapeHtml(w)}</div>`));
  box.innerHTML = rows.join("");
  box.style.display = "block";
}

function renderGraph(p) {
  const g = p.graph; if (!g || !g.nodes || !g.nodes.length) return;
  $("graph-empty").style.display = "none";
  const byId = {}; g.nodes.forEach((n) => (byId[n.id] = n));
  const maxLayer = Math.max(0, ...g.nodes.filter((n) => n.node_type === "feature").map((n) => n.layer));
  const col = {}, posOf = {};
  function xOf(n) {
    if (n.node_type === "input_token") return -1;
    if (n.node_type === "output_token" || n.node_type === "error") return maxLayer + 1;
    return Math.max(0, Math.min(maxLayer, n.layer));
  }
  g.nodes.forEach((n) => { const x = xOf(n); (col[x] = col[x] || []).push(n); });
  // Spread each column over a fixed height so labels never pile up.
  Object.keys(col).forEach((x) => {
    const members = col[x]; members.sort((a, b) => a.token_idx - b.token_idx);
    const span = 10; // fixed vertical span per column
    members.forEach((m, i) => {
      const y = members.length > 1 ? (i / (members.length - 1) - 0.5) * span : 0;
      posOf[m.id] = [parseFloat(x), y];
    });
  });
  // Only label the most important nodes so the left column of input tokens
  // doesn't become an unreadable stack. Scores are NOT comparable across node
  // types (input tokens are fixed at 1.0, features are unitless centred scores,
  // the error node is raw residual energy ~1e6), so rank within a type: every
  // output/error/supernode, then the top features by |score|, then the input
  // tokens that feed a labelled node (by |edge weight|) or host a labelled feature.
  const labelSet = new Set(), internal = [];
  g.nodes.forEach((n) => {
    if (n.node_type === "output_token" || n.node_type === "error" || n.node_type === "supernode") labelSet.add(n.id);
    else if (n.node_type !== "input_token") internal.push(n);
  });
  const topFeats = internal.sort((a, b) => Math.abs(b.score) - Math.abs(a.score)).slice(0, 10);
  topFeats.forEach((n) => labelSet.add(n.id));
  const hostTok = new Set(topFeats.map((n) => n.token_idx)), flow = {};
  (g.edges || []).forEach((e) => {
    const s = byId[e.src];
    if (s && s.node_type === "input_token" && labelSet.has(e.dst)) flow[e.src] = (flow[e.src] || 0) + Math.abs(e.weight);
  });
  g.nodes.filter((n) => n.node_type === "input_token" && (flow[n.id] || hostTok.has(n.token_idx)))
    .sort((a, b) => (flow[b.id] || 0) - (flow[a.id] || 0) || a.token_idx - b.token_idx)
    .slice(0, 6).forEach((n) => labelSet.add(n.id));
  const excluded = excludedOf(p);
  const promoteX = [], promoteY = [], supX = [], supY = [];
  (g.edges || []).forEach((e) => {
    const a = posOf[e.src], b = posOf[e.dst]; if (!a || !b) return;
    const arr = e.polarity === "suppress" ? [supX, supY] : [promoteX, promoteY];
    arr[0].push(a[0], (a[0] + b[0]) / 2, b[0], null);
    arr[1].push(a[1], (a[1] + b[1]) / 2 + 0.25, b[1], null);
  });
  const typeColor = { input_token: COLORS.input, feature: COLORS.accent, output_token: COLORS.gold, error: COLORS.suppress, supernode: COLORS.gold };
  const traces = [
    { x: promoteX, y: promoteY, mode: "lines", line: { color: COLORS.accent, width: 1.3 }, hoverinfo: "skip", name: "promote" },
    { x: supX, y: supY, mode: "lines", line: { color: COLORS.suppress, width: 1.3, dash: "dot" }, hoverinfo: "skip", name: "suppress" },
  ];
  const types = {};
  g.nodes.forEach((n) => { if (posOf[n.id]) (types[n.node_type] = types[n.node_type] || []).push(n); });
  Object.entries(types).forEach(([t, members]) => {
    traces.push({
      x: members.map((n) => posOf[n.id][0]), y: members.map((n) => posOf[n.id][1]),
      mode: "markers+text",
      text: members.map((n) => (labelSet.has(n.id) ? (n.label || n.id) : "")),
      textposition: "middle right", textfont: { size: 10 }, name: t,
      // Fixed size + per-type colour: never scale markers by score across types.
      marker: {
        size: 13, line: { width: 1, color: "#0008" },
        color: t === "input_token" ? members.map((n) => (excluded.has(n.token_idx) ? COLORS.muted : COLORS.input)) : (typeColor[t] || COLORS.muted),
      },
      hovertext: members.map((n) => `${escapeHtml(n.label)}<br>${n.node_type} · layer ${n.layer} · `
        + (n.node_type === "error"
          ? (n.error_kind === "attribution_mass"
            ? `uncovered attribution mass ${(+n.score).toPrecision(3)} (metric units)`
            : `residual energy ${(+n.score).toExponential(2)}`)
            + (n.unexplained_fraction !== undefined ? ` · unexplained ${(100 * n.unexplained_fraction).toFixed(1)}%` : "")
          : `score ${(+n.score).toFixed(3)}`
            + (typeof n.attribution === "number" ? `<br>attribution ${(n.attribution >= 0 ? "+" : "") + n.attribution.toPrecision(3)}` : "")
            + (typeof n.patched_effect === "number" ? `<br>measured ablation effect ${(n.patched_effect >= 0 ? "+" : "") + n.patched_effect.toPrecision(3)}` : "")
            + (n.selected_by === "attribution" ? "<br>added by attribution (not an extracted feature)" : ""))
        + (n.node_type === "input_token" && excluded.has(n.token_idx)
          ? "<br>excluded: " + escapeHtml((p.exclusion_reasons || {})[n.token_idx] || "left out of feature ranking") : "")),
      hoverinfo: "text",
    });
  });
  const blackbox = (p.evidence_kind || "") === "black_box";
  const sem = (p.attribution && p.attribution.edge_semantics) || (g.meta && g.meta.edge_semantics) || "";
  const title = blackbox || sem === "causal_input_masking"
    ? "Input→output attribution — causal input masking (API model: internals not observable; use 🔬 X-ray on a local model)"
    : sem === "causal_linearised"
      ? "Attribution graph — causal, linearised estimate (gradient × activation)"
      : sem === "correlational"
        ? "Attribution graph — correlational (activation flow, not causal)"
        : "Attribution graph";
  Plotly.react("graph-plot", traces, plotLayout({
    height: 480, title: { text: title, font: { size: 13 } }, showlegend: true,
    xaxis: { visible: false }, yaxis: { visible: false, range: [-6, 6] },
  }), { displayModeBar: false, responsive: true });
}

function renderFeatures(p) {
  const feats = (p.features || []).slice().sort((a, b) => b.score - a.score);
  if (!feats.length) return;
  $("features-empty").style.display = "none";
  const max = Math.max(...feats.map((f) => Math.abs(f.score)), 1e-9);
  const q = ($("feature-search").value || "").toLowerCase();
  const rows = feats.filter((f) => !q || (f.label || "").toLowerCase().includes(q)).map((f) => `
    <tr><td>${f.id}</td><td>${escapeHtml(f.label || "—")}</td><td>${f.layer}</td><td>${f.token_idx}</td>
    <td>${(+f.score).toFixed(3)}</td><td><div class="fbar" style="width:${Math.round(120 * Math.abs(f.score) / max)}px"></div></td>
    <td>${f.evidence_kind}</td></tr>`).join("");
  $("features-table").innerHTML = `<table class="ftable"><thead><tr>
    <th>ID</th><th>Label</th><th>Layer</th><th>Tok</th><th>Score</th><th></th><th>Evidence</th>
    </tr></thead><tbody>${rows}</tbody></table>`;
}

function renderProbes(p) {
  const probes = p.probes || [];
  if (!probes.length) return;
  $("probes-empty").style.display = "none";
  $("probes-list").innerHTML = probes.map((r) => `
    <div class="probe">
      <span class="pill ${r.passed ? "pass" : "fail"}">${r.passed ? "PASS" : "FAIL"}</span>
      <b style="min-width:170px">${escapeHtml(r.probe_name)}</b>
      <div class="pgrow"><div class="pgfill" style="width:${Math.round(100 * Math.max(0, Math.min(1, r.score)))}%"></div></div>
      <span class="muted" style="flex:2">${escapeHtml(r.summary || "")}</span>
    </div>`).join("");
}

/* ---------------- White-box live stream ---------------- */
function startWhitebox(d) {
  wbSteps = [];
  $("stream-empty").style.display = "none";
  $("tok-stream").innerHTML = "";
  setBadges([{ t: "white-box: " + (d.model || ""), c: "white_box" }, { t: "thinking…", c: "" }], true);
}
function stepWhitebox(d) {
  wbSteps.push(d);
  const tok = document.createElement("span");
  tok.className = "tok"; tok.textContent = d.token;
  $("tok-stream").appendChild(tok);
  // Live residual-stream chart: per-layer norm, one line per step.
  const traces = wbSteps.map((s, i) => ({
    x: s.layer_norms.map((_, l) => l), y: s.layer_norms, mode: "lines",
    line: { width: 1.5, color: i === wbSteps.length - 1 ? COLORS.gold : COLORS.accent },
    opacity: 0.35 + 0.65 * (i / Math.max(1, wbSteps.length - 1)), name: "t" + i, showlegend: false,
  }));
  Plotly.react("stream-plot", traces, plotLayout({
    height: 320, title: "Residual-stream norm per layer (live, real activations)",
    xaxis: { title: "layer" }, yaxis: { title: "‖h‖" },
  }), { displayModeBar: false, responsive: true });
}
function finishWhitebox(d) {
  setBadges([{ t: "white-box complete", c: "white_box" }, { t: "completion: " + (d.completion || "").slice(0, 40), c: "out" }]);
}

/* ---------------- Live proxy ---------------- */
function appendProxy(d, pending) {
  const log = $("proxy-log");
  if (log.querySelector(".center-empty")) log.innerHTML = "";
  const row = document.createElement("div");
  row.className = "evrow";
  let dist = "";
  if (d.next_token_distribution && d.next_token_distribution.length) {
    const max = Math.max(...d.next_token_distribution.map((x) => x[1]), 1e-9);
    dist = d.next_token_distribution.slice(0, 5).map((x) =>
      `<div class="dist-bar"><span class="lab">${escapeHtml(String(x[0]))}</span>
       <span class="track"><span class="fill" style="width:${Math.round(100 * x[1] / max)}%"></span></span>
       <span class="muted">${(x[1]).toFixed(3)}</span></div>`).join("");
  }
  row.innerHTML = `<div><span class="k">${pending ? "→ request" : "✓ exchange"}</span>
    <span class="muted">${escapeHtml((d.model || ""))}</span></div>
    <div class="muted" style="margin:3px 0">${escapeHtml((d.prompt || "").slice(0, 140))}</div>
    ${d.completion ? `<div><b>${escapeHtml((d.completion || "").slice(0, 160))}</b></div>` : ""}
    ${dist}`;
  log.prepend(row);
  while (log.children.length > 40) log.removeChild(log.lastChild);
}

/* ---------------- Config ---------------- */
async function loadConfig() {
  const cfg = await (await fetch("/api/config")).json();
  window._cfg = cfg;
  $("provider-select").value = cfg.active_provider || "ollama";
  applyProviderFields();
  $("topk-input").value = cfg.top_k_features;
  $("thr-input").value = cfg.attribution_threshold;
}
function applyProviderFields() {
  const name = $("provider-select").value;
  const s = (window._cfg && window._cfg.providers && window._cfg.providers[name]) || {};
  $("model-input").value = s.model || "";
  $("url-input").value = s.base_url || "";
  $("key-status").textContent = s.api_key_set ? `key set (${s.api_key_masked})` : "no key stored";
  const needsKey = (name === "openai" || name === "anthropic");
  const needsUrl = (name === "ollama");
  $("key-label").style.display = needsKey ? "block" : "none";
  $("key-input").style.display = needsKey ? "block" : "none";
  $("key-status").style.display = needsKey ? "block" : "none";
  $("url-label").style.display = needsUrl ? "block" : "none";
  $("url-input").style.display = needsUrl ? "block" : "none";
  const notes = {
    ollama: "Local serving, but black-box: Ollama's API returns text + logprobs only, "
      + "never layer activations — so it can't be X-rayed. Load the same weights as HuggingFace to open them up.",
    huggingface: "Local, white-box. Model box accepts a HF id (gpt2) OR a local weights folder/path "
      + "(safetensors/PyTorch). Full X-ray: real activations, attention, logit lens. (GGUF-only weights need conversion.)",
    openai: "Black-box. Uses real logprobs for attribution (chat models). Reasoning models (o1/o3/o4, gpt-5 family) return no logprobs: "
      + "their top-token probability is a 1.0 placeholder and attribution is coarse.",
    anthropic: "Black-box. No token logprobs — attribution via sampled token + masking.",
    mock: "Offline synthetic provider for demos and tests.",
  };
  $("provider-note").textContent = notes[name] || "";
  if (typeof updateXrayHint === "function") updateXrayHint();
}
async function saveProvider() {
  const name = $("provider-select").value;
  const body = { provider: name, model: $("model-input").value, base_url: $("url-input").value, make_active: true };
  const key = $("key-input").value;
  if (key) body.api_key = key;
  window._cfg = await (await fetch("/api/config/provider", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  })).json();
  $("key-input").value = "";
  applyProviderFields();
  flash($("save-btn"), "Saved ✓");
}
async function testProvider() {
  const r = $("test-result"); r.className = "test-result show"; r.textContent = "testing…";
  const name = $("provider-select").value;
  const res = await (await fetch("/api/provider/test", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ provider: name }),
  })).json();
  r.className = "test-result show " + (res.ok ? "ok" : "err");
  r.textContent = (res.ok ? "✓ " : "✗ ") + res.detail;
}
async function saveDefaults() {
  await fetch("/api/config/defaults", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ top_k_features: +$("topk-input").value, attribution_threshold: +$("thr-input").value }),
  });
  flash($("defaults-btn"), "Saved ✓");
}

/* ---------------- Actions ---------------- */
function traceOptions() {
  const num = (id, dflt) => { const v = parseInt($(id).value, 10); return Number.isFinite(v) && v >= 0 ? v : dflt; };
  return {
    attribution: $("attr-method").value,
    metric: $("attr-metric").value,
    scoring: $("attr-scoring").value,
    validate: num("attr-validate", 0),
    attribution_nodes: num("attr-nodes", 10),
  };
}
function saveTraceOptions() {
  try { localStorage.setItem("tl-trace-options", JSON.stringify(traceOptions())); } catch (e) { /* storage unavailable */ }
}
function restoreTraceOptions() {
  let o = null;
  try { o = JSON.parse(localStorage.getItem("tl-trace-options") || "null"); } catch (e) { o = null; }
  if (!o) return;
  if (o.attribution) $("attr-method").value = o.attribution;
  if (o.metric) $("attr-metric").value = o.metric;
  if (o.scoring) $("attr-scoring").value = o.scoring;
  if (o.validate !== undefined) $("attr-validate").value = o.validate;
  if (o.attribution_nodes !== undefined) $("attr-nodes").value = o.attribution_nodes;
}
function errorText(data) {
  if (!data) return "request failed";
  if (data.error) return data.error;
  if (Array.isArray(data.detail)) return data.detail.map((d) => (d.loc || []).slice(-1)[0] + ": " + d.msg).join("; ");
  return data.detail ? String(data.detail) : "request failed";
}
async function runTrace() {
  const prompt = $("prompt-input").value.trim(); if (!prompt) return;
  setBadges([{ t: "tracing…", c: "" }], true);
  saveTraceOptions();
  const res = await fetch("/api/trace", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(Object.assign({ prompt, provider: $("provider-select").value, run_probes: false }, traceOptions())),
  });
  const data = await res.json();
  if (!res.ok || data.error) setBadges([{ t: "error: " + errorText(data), c: "black_box" }]);
}

/* ---------------- SAE picker ---------------- */
function renderSaeState(st) {
  const sel = $("sae-release");
  if (st.releases && !sel.options.length) {
    Object.entries(st.releases).forEach(([name, info]) => {
      const o = document.createElement("option");
      o.value = name; o.textContent = `${name} (${info.model})`; o.dataset.example = info.example_sae_id;
      sel.appendChild(o);
    });
    if (sel.options.length && !$("sae-id").value) $("sae-id").placeholder = sel.options[0].dataset.example || "";
  }
  const active = st.active || [];
  const loaded = (st.loaded || []).filter((l) => active.includes(l.key));
  $("sae-status").innerHTML = active.length
    ? "Attached to HuggingFace traces: " + loaded.map((l) => `<b>${escapeHtml(l.key)}</b> (${escapeHtml(l.hook_name || "")}, ${l.d_sae} features`
      + (l.hf_model ? `, trained on ${escapeHtml(l.hf_model)}` : "") + ")").join(", ")
    : "No SAE attached — traces use residual-site features.";
}
async function loadSaeState() {
  try { renderSaeState(await (await fetch("/api/sae")).json()); } catch (e) { /* server without the SAE API */ }
}
async function loadSae() {
  const release = $("sae-release").value, saeId = $("sae-id").value.trim() || $("sae-id").placeholder;
  if (!release || !saeId) return;
  $("sae-status").innerHTML = '<span class="spinner"></span> loading SAE…';
  const res = await fetch("/api/sae/load", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ release, sae_id: saeId, allow_download: $("sae-download").checked, attach: true }),
  });
  const data = await res.json();
  if (!res.ok || data.error) { $("sae-status").textContent = "✗ " + errorText(data); return; }
  renderSaeState(data);
}
async function detachSae() {
  renderSaeState(await (await fetch("/api/sae/detach", { method: "POST" })).json());
}

/* ---------------- Steering ---------------- */
async function runSteer() {
  const prompt = $("prompt-input").value.trim(); if (!prompt) return;
  const lines = (id) => $(id).value.split("\n").map((x) => x.trim()).filter(Boolean);
  const provider = $("provider-select").value;
  const body = {
    prompt, provider,
    positive: lines("steer-pos"), negative: lines("steer-neg"),
    layer: parseInt($("steer-layer").value, 10), coeff: parseFloat($("steer-coeff").value),
    site: $("steer-site").value, positions: $("steer-positions").value,
    max_new_tokens: parseInt($("steer-tokens").value, 10) || 20,
  };
  if (provider === "huggingface" && $("model-input").value) body.model_name = $("model-input").value;
  switchTab("steering");
  $("steer-out").innerHTML = '<div class="center-empty"><span class="spinner"></span> generating baseline and steered completions…</div>';
  const res = await fetch("/api/steer", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await res.json();
  if (!res.ok || data.error) renderSteerError(errorText(data));
}
function renderSteerError(msg) {
  $("steer-out").innerHTML = `<div class="xray-banner">⚠️ ${escapeHtml(msg)}</div>`;
  setBadges([{ t: "steering error", c: "black_box" }]);
}
function tokenSpans(trace, diverged) {
  return (trace.tokens || []).map((t, i) => `<span class="${i === diverged ? "diverge" : ""}">${escapeHtml(t)}</span>`).join("");
}
function shiftRows(rows) {
  return (rows || []).slice(0, 6).map((r) => `<tr><td><code>${escapeHtml(JSON.stringify(r.token))}</code></td>`
    + `<td>${(+r.p_baseline).toFixed(4)}</td><td>${(+r.p_steered).toFixed(4)}</td></tr>`).join("") || '<tr><td colspan="3">—</td></tr>';
}
function renderSteer(d) {
  switchTab("steering");
  const v = d.vector || (d.vectors && d.vectors[0]) || {};
  const div = d.diverged_at === null || d.diverged_at === undefined ? null : d.diverged_at;
  $("steer-out").innerHTML = `
    <div class="badges" style="margin:10px 0">
      <span class="badge white_box">${escapeHtml(d.evidence_kind || "white_box")}</span>
      <span class="badge">${escapeHtml(d.method || "")}</span>
      <span class="badge">effect: ${escapeHtml(d.effect_semantics || "")}</span>
      <span class="badge">${escapeHtml(d.model || "")} · layer ${v.layer} ${escapeHtml(v.site || "")} · coeff ${(d.vectors && d.vectors[0] ? d.vectors[0].coeff : v.coeff)}</span>
    </div>
    <div class="note">${escapeHtml(d.note || "")}</div>
    <div class="steer-cols">
      <div class="steer-col"><h4 class="xray-h">Baseline</h4><div class="steer-text">${tokenSpans(d.baseline || {}, div)}</div></div>
      <div class="steer-col"><h4 class="xray-h">Steered</h4><div class="steer-text">${tokenSpans(d.steered || {}, div)}</div></div>
    </div>
    <div class="note">KL first step ${(+d.first_step_kl).toFixed(4)} nats · mean ${(+d.mean_kl).toFixed(4)} nats · `
      + (div === null ? "identical completions" : `first differs at generated token ${div}`)
      + ` · KL is teacher-forced along the baseline completion.</div>
    <div id="steer-kl-plot"></div>
    <div class="steer-cols">
      <div class="steer-col"><h4 class="xray-h">Promoted (first step)</h4><table class="ftable"><thead><tr><th>token</th><th>p base</th><th>p steered</th></tr></thead><tbody>${shiftRows(d.promoted)}</tbody></table></div>
      <div class="steer-col"><h4 class="xray-h">Suppressed (first step)</h4><table class="ftable"><thead><tr><th>token</th><th>p base</th><th>p steered</th></tr></thead><tbody>${shiftRows(d.suppressed)}</tbody></table></div>
    </div>`;
  const kl = d.kl_per_step || [], toks = (d.baseline && d.baseline.tokens) || [];
  const xs = kl.map((_, i) => i);
  Plotly.react("steer-kl-plot", [{
    x: xs, y: kl, type: "bar", marker: { color: COLORS.accent },
    text: xs.map((i) => escapeHtml(toks[i] !== undefined ? toks[i] : "")),
    hovertemplate: "step %{x} (%{text}): KL %{y:.4f} nats<extra></extra>",
  }], plotLayout({
    height: 240, title: { text: "KL(steered ‖ baseline) per step", font: { size: 12 } },
    xaxis: { tickmode: "array", tickvals: xs, ticktext: xs.map((i) => String(toks[i] !== undefined ? toks[i] : i)) },
    yaxis: { title: "nats" },
  }), { displayModeBar: false, responsive: true });
  setBadges([{ t: "steering complete", c: "white_box" }, { t: "mean KL " + (+d.mean_kl).toFixed(3), c: "out" }]);
}

/* ---------------- Benchmarks ---------------- */
async function loadBenchList() {
  try {
    const data = await (await fetch("/api/bench/results")).json();
    const sel = $("bench-list");
    sel.innerHTML = '<option value="">— records in ' + escapeHtml(data.dir || "the bench folder") + " —</option>";
    (data.results || []).forEach((r) => {
      const o = document.createElement("option"); o.value = r.name; o.textContent = r.name; sel.appendChild(o);
    });
  } catch (e) { /* server without the bench API */ }
}
async function viewBenchByName(name) {
  if (!name) return;
  const res = await fetch("/api/bench/results/" + encodeURIComponent(name).replace(/%2F/g, "/"));
  const data = await res.json();
  if (!res.ok || data.error) { $("bench-out").innerHTML = `<div class="xray-banner">⚠️ ${escapeHtml(errorText(data))}</div>`; return; }
  renderBench(data);
}
async function viewBenchFile(file) {
  if (!file) return;
  let record;
  try { record = JSON.parse(await file.text()); } catch (e) {
    $("bench-out").innerHTML = '<div class="xray-banner">⚠️ not a JSON file</div>'; return;
  }
  const res = await fetch("/api/bench/view", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(record) });
  const data = await res.json();
  if (!res.ok || data.error) { $("bench-out").innerHTML = `<div class="xray-banner">⚠️ ${escapeHtml(errorText(data))}</div>`; return; }
  renderBench(data);
}
function renderBench(b) {
  const probes = b.probes || [];
  const cell = (m, pr) => (b.by_model_probe || []).find((r) => r.model === m && r.probe === pr) || {};
  const head = "<tr><th>model</th><th>status</th><th>passed</th><th>mean</th>" + probes.map((pr) => `<th>${escapeHtml(pr)}</th>`).join("") + "</tr>";
  const rows = (b.by_model || []).map((m) => {
    const cells = probes.map((pr) => {
      const c = cell(m.model, pr);
      if (c.mean_score === null || c.mean_score === undefined) return `<td class="muted">${c.n_error ? "error" : "—"}</td>`;
      return `<td><span class="pill ${c.passed ? "pass" : "fail"}">${(+c.mean_score).toFixed(2)}</span></td>`;
    }).join("");
    return `<tr><td><b>${escapeHtml(m.model)}</b>${m.synthetic ? ' <span class="badge">synthetic</span>' : ""}</td>`
      + `<td>${escapeHtml(m.status || "")}</td><td>${m.n_passed} / ${m.n_scored}</td>`
      + `<td>${m.mean_score === null || m.mean_score === undefined ? "—" : (+m.mean_score).toFixed(2)}</td>${cells}</tr>`;
  }).join("");
  $("bench-out").innerHTML = `<div class="note"><b>${escapeHtml(b.title || "benchmark")}</b> · ${escapeHtml(b.created_utc || "")}`
    + ` · schema ${escapeHtml(String(b.schema || ""))} v${escapeHtml(String(b.schema_version || ""))}</div>`
    + (b.notes || []).map((n) => `<div class="note">${escapeHtml(n)}</div>`).join("")
    + `<div style="overflow-x:auto"><table class="ftable">${head}${rows}</table></div>`;
}
async function runWhitebox() {
  const prompt = $("prompt-input").value.trim(); if (!prompt) return;
  switchTab("stream");
  await fetch("/api/whitebox/stream", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_name: $("model-input").value || "gpt2", prompt, max_new_tokens: 20 }),
  });
}

/* ---------------- UI plumbing ---------------- */
function switchTab(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === "panel-" + name));
  if (lastPayload) { if (name === "graph") renderGraph(lastPayload); if (name === "heatmap") renderHeatmap(lastPayload); }
  if (name === "bench") loadBenchList();
}
function flash(btn, msg) { const old = btn.textContent; btn.textContent = msg; setTimeout(() => (btn.textContent = old), 1200); }
function escapeHtml(s) { return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

function init() {
  connectWS();
  loadConfig();
  $("proxy-url").textContent = `${location.origin}/v1`;
  document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => switchTab(t.dataset.tab)));
  $("provider-select").addEventListener("change", applyProviderFields);
  $("test-btn").addEventListener("click", testProvider);
  $("save-btn").addEventListener("click", saveProvider);
  $("defaults-btn").addEventListener("click", saveDefaults);
  $("trace-btn").addEventListener("click", runTrace);
  $("whitebox-btn").addEventListener("click", runWhitebox);
  $("xray-btn").addEventListener("click", runXray);
  $("steer-btn").addEventListener("click", runSteer);
  $("sae-load-btn").addEventListener("click", loadSae);
  $("sae-detach-btn").addEventListener("click", detachSae);
  $("sae-release").addEventListener("change", () => { const o = $("sae-release").selectedOptions[0]; if (o) $("sae-id").placeholder = o.dataset.example || ""; });
  $("bench-list").addEventListener("change", () => viewBenchByName($("bench-list").value));
  $("bench-file").addEventListener("change", () => viewBenchFile($("bench-file").files[0]));
  ["attr-method", "attr-metric", "attr-scoring", "attr-validate", "attr-nodes"].forEach((id) => $(id).addEventListener("change", saveTraceOptions));
  restoreTraceOptions();
  loadSaeState();
  loadBenchList();
  updateXrayHint();
  $("feature-search").addEventListener("input", () => lastPayload && renderFeatures(lastPayload));
  $("copy-proxy").addEventListener("click", () => navigator.clipboard.writeText(`${location.origin}/v1`).then(() => flash($("copy-proxy"), "copied ✓")));
  $("theme-btn").addEventListener("click", () => {
    const h = document.documentElement;
    h.setAttribute("data-theme", h.getAttribute("data-theme") === "dark" ? "light" : "dark");
    if (lastPayload) renderTrace(lastPayload);
  });
  $("prompt-input").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) runTrace(); });
}
document.addEventListener("DOMContentLoaded", init);
