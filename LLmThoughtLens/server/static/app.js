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

  // Activation grid heatmap (layers × tokens)
  Plotly.react("xray-grid-plot", [{
    z: d.grid, x: d.tokens, y: d.grid.map((_, i) => "L" + i), type: "heatmap",
    colorscale: [[0, "#0d1b1c"], [0.5, COLORS.accentDeep], [1, COLORS.gold]],
    hovertemplate: "layer %{y}, %{x}: ‖h‖=%{z:.1f}<extra></extra>",
  }], plotLayout({ height: 280, margin: { t: 10, b: 60, l: 40, r: 10 } }), { displayModeBar: false, responsive: true });

  // Attention heatmap (last layer)
  if (d.attention && d.attention.length) {
    Plotly.react("xray-attn-plot", [{
      z: d.attention, x: d.tokens, y: d.tokens, type: "heatmap",
      colorscale: [[0, "#0d1b1c"], [1, COLORS.accent]],
      hovertemplate: "%{y} → %{x}: %{z:.3f}<extra></extra>",
    }], plotLayout({ height: 280, margin: { t: 10, b: 60, l: 60, r: 10 } }), { displayModeBar: false, responsive: true });
  }
  setBadges([{ t: "x-ray step " + d.step, c: "white_box" }, { t: "→ " + d.token, c: "out" }], true);
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
  ]);
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

function renderHeatmap(p) {
  const feats = p.features || [];
  if (!feats.length) return;
  $("heatmap-empty").style.display = "none";
  const tokens = (p.summary && p.summary.tokens) || [];
  const agg = {};
  for (const f of feats) agg[f.token_idx] = (agg[f.token_idx] || 0) + Math.max(0, f.score);
  const n = Math.max(tokens.length, Object.keys(agg).length);
  const labels = [], z = [];
  for (let i = 0; i < n; i++) { labels.push(tokens[i] || ("tok" + i)); z.push(agg[i] || 0); }
  const max = Math.max(...z, 1e-9);
  Plotly.react("heatmap-plot", [{
    z: [z.map((v) => v / max)], x: labels, y: ["activation"], type: "heatmap",
    colorscale: [[0, "#0d1b1c"], [0.5, COLORS.accentDeep], [1, COLORS.gold]],
    xgap: 2, hovertemplate: "%{x}: %{z:.3f}<extra></extra>",
  }], plotLayout({ height: 200, title: "Per-token activation (real)" }), { displayModeBar: false, responsive: true });
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
  // Only label the most important nodes (+ every output/error/supernode) so the
  // left column of input tokens doesn't become an unreadable stack.
  const ranked = g.nodes.slice().sort((a, b) => Math.abs(b.score) - Math.abs(a.score));
  const labelSet = new Set(ranked.slice(0, 12).map((n) => n.id));
  g.nodes.forEach((n) => {
    if (n.node_type === "output_token" || n.node_type === "error" || n.node_type === "supernode") {
      labelSet.add(n.id);
    }
  });
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
      marker: { size: 13, color: typeColor[t] || COLORS.muted, line: { width: 1, color: "#0008" } },
      hovertext: members.map((n) => `${n.label}<br>${n.node_type} · layer ${n.layer} · score ${(+n.score).toFixed(3)}`),
      hoverinfo: "text",
    });
  });
  const blackbox = (p.evidence_kind || "") === "black_box";
  const title = blackbox
    ? "Input→output attribution (API model — internals not observable; use 🔬 X-ray on a local model)"
    : "Attribution graph (real causal flow)";
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
    openai: "Black-box. Uses real logprobs for attribution.",
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
async function runTrace() {
  const prompt = $("prompt-input").value.trim(); if (!prompt) return;
  setBadges([{ t: "tracing…", c: "" }], true);
  const res = await fetch("/api/trace", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, provider: $("provider-select").value, run_probes: false }),
  });
  const data = await res.json();
  if (data.error) setBadges([{ t: "error: " + data.error, c: "black_box" }]);
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
