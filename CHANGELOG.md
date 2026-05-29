# Changelog

All notable changes to **LLmThoughtLens** are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-05-29

First public release. Implements the full Phase 0 → Phase 10 plan from the
build-tracking document.

### Added — Providers (Phase 1+2)

- `BaseProvider` ABC and `ProviderOutput` dataclass with `evidence_kind`
  tagging (`"white_box"` / `"black_box"`), `n_tokens` / `n_layers` /
  `d_model` accessors, and `to_summary()` for JSON serialisation.
- Deterministic seeded `MockProvider` with real-shape activations
  `(L, T, D)`, attentions `(L, H, T, T)` with a true causal mask, last-token
  logits `(V,)`, sorted top-k probabilities. Implements
  `run_with_intervention` by applying each intervention's `.apply_numpy`
  to the synthesised activations.
- `OpenAIProvider` — Chat Completions adapter with real top-k logprobs
  (`logprobs=True, top_logprobs=k`) and an explicit evidence note.
- `AnthropicProvider` — Messages API adapter that honestly reports
  there are no per-token logprobs.
- `HuggingFaceProvider` — loads any HF causal LM, runs forward with
  `output_hidden_states=True, output_attentions=True`, packs real
  `(L, T, D)` / `(L, H, T, T)` tensors, supports `run_with_intervention`
  via real PyTorch `forward_pre_hook` on each block's MLP submodule.
- `OllamaProvider` — local Ollama HTTP adapter with `ping()` health check.
- Lazy provider registry: `list_providers()`, `available_providers()`,
  `get_provider()`, `register_provider()`.

### Added — Sparse Autoencoder (Phase 5)

- TopK `SparseAutoencoder` in real PyTorch with autograd training, Adam
  optimisation, decoder-column unit-norm renormalisation after every step,
  dead-feature resampling, and exact `TopK(h, k)` structural sparsity.
- Real diagnostics: reconstruction MSE, explained variance, L0 sparsity
  (mean / std), dead-feature fraction, feature-density histogram.
- `.save()` / `.load()` to a single `.pt` file with config + labels.
- `feature_direction(feature_id)` returns the unit decoder column for
  intervention targeting.
- `ActivationCache` — collects per-token residual-stream activations from
  any white-box provider, supports `max_tokens` cap, persists to a single
  compressed `.npz` with embedded provenance metadata.
- `FeatureLabeler` — auto-labels SAE features by querying an LLM labelling
  provider with the top-N activating contexts per feature; sanitised
  output (max 5 words, 60 chars).

### Added — Feature extraction (Phase 4+5)

- `FeatureExtractor` with white-box (SAE codes or L2 norm) and black-box
  (real token-masking) paths. Black-box path issues real provider calls,
  records real `prob_baseline − prob_masked` deltas, supports a budget
  guard, and caches masked-prompt probabilities.
- Real pairwise interaction scoring:
  `interaction(i, j) = P_full + P_mask_ij − P_mask_i − P_mask_j`.

### Added — Interventions (Phase 9)

- `FeatureIntervention` with three modes (`amplify`, `inhibit`, `clamp`),
  acting along a unit direction (SAE decoder column when an SAE is
  attached; raw hidden-dim coordinate otherwise).
- `apply_numpy` for offline (L, T, D) arrays and `apply_torch` for live
  hidden-state tensors inside hooks.
- `intervention_context(blocks, interventions)` — context manager that
  installs `forward_pre_hook`s on each block's MLP submodule and
  guarantees their removal on exit, even on exception.

### Added — Attribution graph (Phase 6)

- Typed `AttributionGraph` with `CircuitNode` (`input_token`, `feature`,
  `supernode`, `output_token`, `error`, `safety`, `suppressor`) and
  `CircuitEdge` with signed weights and polarity (`promote` /
  `suppress`).
- Real attribution: white-box edges combine source / destination L2
  norms, cosine alignment, and the attention share from
  `dst_token → src_token` across heads. Black-box edges are real
  prob-delta perturbation weights. Error-residual node summarises
  activation mass unexplained by the captured features.
- `prune(threshold, keep_isolated=...)`, `top_paths`, JSON + CSV export.
- `SupernodeGrouper` — cosine-similarity clustering using SAE-decoder
  directions when an SAE is attached, raw activations otherwise, with a
  label-only fallback.
- `top_causal_paths(graph, n)` — Dijkstra on `-log|w|` returning
  `CausalPath` objects with per-edge weights and a cumulative log-score.
- `GraphDiff.compute(a, b)` produces added / removed / changed sets and
  renders itself to JSON (`to_json`) or to an embeddable HTML fragment
  (`to_html`) used by the report builder.

### Added — Probes (Phase 8)

- Ten built-in probes mirroring the Anthropic Biology case studies:
  `MultiHopProbe`, `CapitalsProbe`, `RhymePlanningProbe`,
  `PersonaConsistencyProbe`, `MultilingualAbstractionProbe`,
  `HallucinationProbe`, `CoTFaithfulnessProbe`, `RefusalProbe`,
  `SuppressorProbe`, `MotivatedReasoningProbe`.
- Each probe runs real prompts against the provider, returns a
  `ProbeResult` with `score ∈ [0, 1]`, a `passed` boolean, an `evidence`
  dict, and a one-paragraph `summary`.
- `ProbeRunner` with per-probe progress callbacks and exception capture.
- `ProbeReport` with mean score, JSON export, and CLI scorecard
  rendering.

### Added — Self-contained HTML report (Phase 7)

- `ReportBuilder.from_trace_result(...)` produces a single
  CDN-Plotly-only `.html` file with five real tabs:
  - Token Heatmap (`Plotly Heatmap` over real activations),
  - Attribution Graph (layered Plotly DAG with typed nodes + signed edges),
  - Residual Stream (per-token PCA trajectory across layers),
  - Feature Browser (searchable, regex-filterable, click-sortable HTML
    table),
  - Probe Dashboard (per-probe scorecard rows + Scatterpolar radar +
    collapsible evidence).
- `add_graph_diff(diff)` embeds a side-by-side GraphDiff tab.

### Added — Textual TUI / CLI (Phase 3 + advanced)

- `LLmThoughtLens` console-script with subcommands `tui`, `trace`,
  `probe`, `benchmark`, `cache-activations`, `train-sae`,
  `label-features`, `providers`, `version`.
- Full Textual `LLmThoughtLensApp` with `HomeScreen`, `ConnectScreen`
  (masked API key + live connection test), `TraceScreen` (live mini-trace
  with token heatmap + features table), `FeatureBrowserScreen` (fuzzy
  search via rapidfuzz), `ProbeScreen` (spacebar checkboxes + per-probe
  progress widgets), `GraphSummaryScreen` (ASCII top-causal-paths), and
  `ExportScreen`.
- Vim navigation (`j` / `k`), arrow / Tab / Enter / Escape / Q
  shortcuts, per-screen `BINDINGS`.
- Persistent config + last-20 session history in
  `~/.LLmThoughtLens/config.json`.

### Added — Tests (Phase 11)

- 95 unit + integration tests covering: providers contract, SAE training
  convergence + sparsity + persistence, feature extractor (white-box L2 +
  real black-box masking + pairwise interactions), interventions
  (NumPy + torch + context-manager + three causal claims), graph
  primitives (prune + paths + JSON/CSV + diff), tracer end-to-end,
  10-probe runner, report HTML structural assertions, smoke tests.

### Added — Release engineering (Phase 10)

- `make release-check` runs ruff, mypy, pytest, pip-audit, and
  `python -m build --no-isolation`, then verifies that both `.tar.gz`
  and `.whl` are produced in `dist/`.
- `make publish-test` and `make publish` wrap twine.
- `LLmThoughtLens benchmark` subcommand writes
  `benchmark_results.json` and prints a Rich scorecard.

### Documentation

- `docs/quickstart.md` — runnable examples for every provider.
- `docs/probe-reference.md` — one section per built-in probe.

### Cross-cutting guardrails (always-on)

- Every UI surface and `evidence_note` tags traces as observed
  (white-box) or approximated (black-box). The package never synthesises
  activations / attentions / logits for providers that cannot supply
  them.
- All HF forward-pre hooks are installed and removed via
  `intervention_context`, exception-safe.
