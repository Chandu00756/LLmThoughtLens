"""All Textual screens used by the LLmThoughtLens TUI."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.table import Table
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Header,
    Input,
    Label,
    LoadingIndicator,
    Select,
    Static,
)

from LLmThoughtLens.tui.config import SessionEntry, TUIConfig, save_config
from LLmThoughtLens.tui.widgets import AsciiAttributionGraph, FuzzyList, ProbeProgress

if TYPE_CHECKING:
    from LLmThoughtLens.scope import TraceResult


# =====================================================================
# Provider connect screen
# =====================================================================


class ConnectScreen(Screen):
    """Select provider, enter model + API key, live-test connection."""

    BINDINGS = [
        Binding("escape", "app.pop_screen", "back", show=True),
        Binding("q", "app.quit", "quit", show=True),
        Binding("ctrl+t", "test", "test connection", show=True),
        Binding("ctrl+s", "save", "save & continue", show=True),
    ]

    def __init__(self, cfg: TUIConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="connect-body"):
            yield Static("[b]Connect a model[/b]", id="title")
            yield Label("Provider")
            yield Select(
                options=[
                    ("mock", "mock"),
                    ("openai", "openai"),
                    ("anthropic", "anthropic"),
                    ("huggingface", "huggingface"),
                    ("ollama", "ollama"),
                ],
                value=self.cfg.provider,
                id="provider-select",
            )
            yield Label("Model id")
            yield Input(
                value=self.cfg.model,
                placeholder="gpt-4o-mini / claude-3-5-haiku / gpt2 / …",
                id="model-input",
            )
            yield Label("API key (masked)")
            yield Input(
                value=self.cfg.api_key,
                password=True,
                placeholder="sk-… (or leave blank to use env var)",
                id="key-input",
            )
            yield Label("Base URL (Ollama / Azure / proxy)")
            yield Input(
                value=self.cfg.base_url, placeholder="http://localhost:11434", id="url-input"
            )
            yield Checkbox(
                "save key to ~/.LLmThoughtLens/config.json",
                value=self.cfg.save_api_key,
                id="save-key",
            )
            with Horizontal(id="connect-actions"):
                yield Button("Test (Ctrl-T)", id="btn-test", variant="primary")
                yield Button("Save & continue (Ctrl-S)", id="btn-save", variant="success")
            yield Static("", id="connect-status")
        yield Footer()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _collect(self) -> None:
        self.cfg.provider = self.query_one("#provider-select", Select).value or "mock"
        self.cfg.model = self.query_one("#model-input", Input).value
        self.cfg.api_key = self.query_one("#key-input", Input).value
        self.cfg.base_url = self.query_one("#url-input", Input).value
        self.cfg.save_api_key = self.query_one("#save-key", Checkbox).value

    def action_test(self) -> None:
        self._collect()
        status = self.query_one("#connect-status", Static)
        status.update("[yellow]testing connection…[/yellow]")
        provider = build_provider_from_config(self.cfg)
        if provider is None:
            status.update("[red]could not instantiate provider — missing extras?[/red]")
            return
        try:
            out = provider.run("ping")
            status.update(
                f"[green]OK[/green] — provider {provider.name} returned "
                f"{len(out.tokens)} tokens; evidence={out.evidence_kind}"
            )
        except Exception as exc:  # noqa: BLE001
            status.update(f"[red]connection failed:[/red] {exc!r}")

    def action_save(self) -> None:
        self._collect()
        save_config(self.cfg)
        self.app.cfg = self.cfg
        self.app.pop_screen()
        self.app.push_screen(TraceScreen(self.cfg))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-test":
            self.action_test()
        elif event.button.id == "btn-save":
            self.action_save()


# =====================================================================
# Trace screen — run a prompt, show heatmap line + features inline
# =====================================================================


class TraceScreen(Screen):
    """Type a prompt, hit Enter, get a live mini-trace."""

    BINDINGS = [
        Binding("escape", "app.pop_screen", "back", show=True),
        Binding("q", "app.quit", "quit", show=True),
        Binding("ctrl+r", "run", "run trace", show=True),
        Binding("ctrl+p", "probes", "probes", show=True),
        Binding("ctrl+f", "features", "features", show=True),
        Binding("ctrl+g", "graph", "graph", show=True),
        Binding("ctrl+e", "export", "export", show=True),
    ]

    def __init__(self, cfg: TUIConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.last_result: TraceResult | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical():
            yield Static("[b]Trace a prompt[/b]", id="title")
            yield Input(
                value=self.cfg.last_prompt,
                placeholder="Enter prompt then ctrl-R",
                id="prompt-input",
            )
            with Horizontal(id="trace-actions"):
                yield Button("Run (Ctrl-R)", id="btn-run", variant="primary")
                yield Button("Features (Ctrl-F)", id="btn-feats")
                yield Button("Probes (Ctrl-P)", id="btn-probes")
                yield Button("Graph (Ctrl-G)", id="btn-graph")
                yield Button("Export (Ctrl-E)", id="btn-export")
            yield LoadingIndicator(id="loading")
            with VerticalScroll(id="trace-output"):
                yield Static("[dim]No trace yet.[/dim]", id="heatmap-render")
                yield Static("", id="features-render")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#loading", LoadingIndicator).display = False

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_run(self) -> None:
        prompt = self.query_one("#prompt-input", Input).value or self.cfg.last_prompt
        self.cfg.last_prompt = prompt
        save_config(self.cfg)
        spinner = self.query_one("#loading", LoadingIndicator)
        spinner.display = True
        self.run_worker(self._run_trace(prompt), exclusive=True)

    async def _run_trace(self, prompt: str) -> None:
        from LLmThoughtLens.scope import Scope

        provider = build_provider_from_config(self.cfg)
        if provider is None:
            self._render_error("Could not instantiate provider — check Connect screen.")
            return
        scope = Scope(
            provider,
            top_k_features=self.cfg.top_k_features,
            attribution_threshold=self.cfg.attribution_threshold,
            blackbox_budget=self.cfg.blackbox_budget,
        )
        try:
            result = scope.trace_full(prompt)
        except Exception as exc:  # noqa: BLE001
            self._render_error(f"trace failed: {exc!r}")
            return
        self.last_result = result
        self.app.last_result = result
        self.cfg.push_history(
            SessionEntry(
                when=_now_short(),
                provider=provider.name,
                model=provider.model_id,
                prompt=prompt,
                output_token=result.output_token,
            )
        )
        save_config(self.cfg)
        self._render_result(result)

    def _render_result(self, result: TraceResult) -> None:
        # NOTE: must NOT be named ``_render`` — that collides with Textual's
        # internal ``Widget._render()`` and crashes the screen on draw.
        self.query_one("#loading", LoadingIndicator).display = False
        self.query_one("#heatmap-render", Static).update(_ascii_heatmap(result))
        self.query_one("#features-render", Static).update(_ascii_features(result))

    def _render_error(self, msg: str) -> None:
        self.query_one("#loading", LoadingIndicator).display = False
        self.query_one("#heatmap-render", Static).update(f"[red]{msg}[/red]")

    def action_probes(self) -> None:
        self.app.push_screen(ProbeScreen(self.cfg))

    def action_features(self) -> None:
        if self.last_result is not None:
            self.app.push_screen(FeatureBrowserScreen(self.last_result))

    def action_graph(self) -> None:
        if self.last_result is not None:
            self.app.push_screen(GraphSummaryScreen(self.last_result))

    def action_export(self) -> None:
        if self.last_result is not None:
            self.app.push_screen(ExportScreen(self.last_result))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        mapping = {
            "btn-run": self.action_run,
            "btn-feats": self.action_features,
            "btn-probes": self.action_probes,
            "btn-graph": self.action_graph,
            "btn-export": self.action_export,
        }
        fn = mapping.get(event.button.id or "")
        if fn:
            fn()


# =====================================================================
# Feature browser screen
# =====================================================================


class FeatureBrowserScreen(Screen):
    BINDINGS = [
        Binding("escape", "app.pop_screen", "back", show=True),
        Binding("q", "app.quit", "quit", show=True),
        Binding("j", "cursor_down", "down", show=True),
        Binding("k", "cursor_up", "up", show=True),
    ]

    def __init__(self, result: TraceResult) -> None:
        super().__init__()
        self.result = result

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical():
            yield Static(
                f"[b]Features ({len(self.result.features)})[/b] — start typing to fuzzy-filter"
            )
            yield FuzzyList(
                [
                    f"{f.id:>6}  layer {f.layer:>2}  tok {f.token_idx:>2}  "
                    f"score {f.score:7.3f}  {f.label}"
                    for f in self.result.features
                ],
                placeholder="filter by label / id",
            )
            yield Static("", id="feature-detail")
        yield Footer()

    def on_fuzzy_list_selected(self, event: FuzzyList.Selected) -> None:
        self.query_one("#feature-detail", Static).update(f"[b]Selected:[/b] {event.value}")

    def action_cursor_down(self) -> None:
        self.query_one(FuzzyList).query_one("ListView").action_cursor_down()

    def action_cursor_up(self) -> None:
        self.query_one(FuzzyList).query_one("ListView").action_cursor_up()


# =====================================================================
# Probe runner screen
# =====================================================================


class ProbeScreen(Screen):
    """Pick probes with space + run with Ctrl-R; live per-probe progress."""

    BINDINGS = [
        Binding("escape", "app.pop_screen", "back", show=True),
        Binding("q", "app.quit", "quit", show=True),
        Binding("ctrl+r", "run", "run", show=True),
        Binding("ctrl+a", "select_all", "select all", show=True),
        Binding("ctrl+n", "select_none", "select none", show=True),
    ]

    def __init__(self, cfg: TUIConfig) -> None:
        super().__init__()
        self.cfg = cfg
        from LLmThoughtLens.probes.builtin import all_probes

        self._probes = all_probes()
        self._progress: dict[str, ProbeProgress] = {}

    def compose(self) -> ComposeResult:
        from LLmThoughtLens.probes.builtin import all_probes

        yield Header(show_clock=True)
        yield Static("[b]Probe runner[/b] — space toggles, Ctrl-R runs", id="title")
        with VerticalScroll(id="probe-list"):
            for p in all_probes():
                yield Checkbox(f"{p.name} — {p.description}", value=True, id=f"ch-{p.name}")
            for p in all_probes():
                widget = ProbeProgress(p.name)
                self._progress[p.name] = widget
                yield widget
        yield Static("", id="probe-summary")
        yield Footer()

    def action_select_all(self) -> None:
        for ch in self.query("Checkbox").results():
            ch.value = True

    def action_select_none(self) -> None:
        for ch in self.query("Checkbox").results():
            ch.value = False

    def action_run(self) -> None:
        selected = []
        for p in self._probes:
            ch = self.query_one(f"#ch-{p.name}", Checkbox)
            if ch.value:
                selected.append(p)
        self.run_worker(self._run_probes(selected), exclusive=True)

    async def _run_probes(self, probes: list) -> None:
        provider = build_provider_from_config(self.cfg)
        if provider is None:
            self.query_one("#probe-summary", Static).update("[red]Provider not configured.[/red]")
            return
        n_passed = 0
        for p in probes:
            widget = self._progress.get(p.name)
            if widget is not None:
                widget.set_running()
            try:
                res = p.run(provider)
            except Exception:  # noqa: BLE001
                widget.set_done(0.0) if widget else None
                continue
            if widget is not None:
                widget.set_done(res.score)
            if res.passed:
                n_passed += 1
        self.query_one("#probe-summary", Static).update(
            f"[b]{n_passed} / {len(probes)} passed.[/b]"
        )


# =====================================================================
# Graph summary screen — ASCII top-paths
# =====================================================================


class GraphSummaryScreen(Screen):
    BINDINGS = [
        Binding("escape", "app.pop_screen", "back", show=True),
        Binding("q", "app.quit", "quit", show=True),
    ]

    def __init__(self, result: TraceResult) -> None:
        super().__init__()
        self.result = result

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(
            f"[b]Attribution graph[/b]  ({self.result.graph.num_nodes} nodes, "
            f"{self.result.graph.num_edges} edges, evidence={self.result.evidence_kind})"
        )
        graph_widget = AsciiAttributionGraph()
        yield graph_widget
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(AsciiAttributionGraph).show(self.result.graph)


# =====================================================================
# Export screen
# =====================================================================


class ExportScreen(Screen):
    BINDINGS = [
        Binding("escape", "app.pop_screen", "back", show=True),
        Binding("q", "app.quit", "quit", show=True),
    ]

    def __init__(self, result: TraceResult) -> None:
        super().__init__()
        self.result = result

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("[b]Export trace artefacts[/b]")
        yield Label("Output prefix (path without extension)")
        yield Input(value="trace", id="prefix-input", placeholder="e.g. /tmp/trace")
        with Horizontal():
            yield Button("HTML report", id="btn-html", variant="primary")
            yield Button("Graph JSON", id="btn-gjson")
            yield Button("Graph CSV", id="btn-gcsv")
            yield Button("Features CSV", id="btn-fcsv")
        yield Static("", id="export-status")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        prefix = self.query_one("#prefix-input", Input).value or "trace"
        status = self.query_one("#export-status", Static)
        try:
            if event.button.id == "btn-html":
                path = f"{prefix}.html"
                self.result.save(path)
                status.update(f"[green]wrote[/green] {path}")
            elif event.button.id == "btn-gjson":
                path = f"{prefix}.graph.json"
                self.result.save_graph_json(path)
                status.update(f"[green]wrote[/green] {path}")
            elif event.button.id == "btn-gcsv":
                path = f"{prefix}.graph.csv"
                self.result.save_graph_csv(path)
                status.update(f"[green]wrote[/green] {path}")
            elif event.button.id == "btn-fcsv":
                path = f"{prefix}.features.csv"
                self.result.save_features_csv(path)
                status.update(f"[green]wrote[/green] {path}")
        except Exception as exc:  # noqa: BLE001
            status.update(f"[red]export failed:[/red] {exc!r}")


# =====================================================================
# Home / navigation tree
# =====================================================================


class HomeScreen(Screen):
    """Landing screen — provider info + nav buttons."""

    BINDINGS = [
        Binding("q", "app.quit", "quit", show=True),
        Binding("c", "connect", "connect", show=True),
        Binding("t", "trace", "trace", show=True),
        Binding("p", "probes", "probes", show=True),
        Binding("h", "history", "history", show=True),
    ]

    def __init__(self, cfg: TUIConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical():
            yield Static("[b]LLmThoughtLens[/b] — interpretability TUI", id="title")
            yield Static(self._summary(), id="summary")
            with Horizontal():
                yield Button("Connect (c)", id="btn-connect", variant="primary")
                yield Button("Trace (t)", id="btn-trace", variant="success")
                yield Button("Probes (p)", id="btn-probes")
                yield Button("History (h)", id="btn-history")
            yield Static("[b]Recent traces[/b]")
            with VerticalScroll(id="history-scroll"):
                yield FuzzyList(
                    [entry.label() for entry in self.cfg.history] or ["(no history)"],
                    placeholder="filter history",
                )
        yield Footer()

    def _summary(self) -> str:
        return (
            f"provider: [b]{self.cfg.provider}[/b]   "
            f"model: [b]{self.cfg.model or '(none)'}[/b]   "
            f"top-k: {self.cfg.top_k_features}   "
            f"threshold: {self.cfg.attribution_threshold}   "
            f"$ spent: {self.cfg.api_cost_usd:.2f}"
        )

    def action_connect(self) -> None:
        self.app.push_screen(ConnectScreen(self.cfg))

    def action_trace(self) -> None:
        self.app.push_screen(TraceScreen(self.cfg))

    def action_probes(self) -> None:
        self.app.push_screen(ProbeScreen(self.cfg))

    def action_history(self) -> None:
        # No standalone history screen — focus the fuzzy list.
        self.query_one(FuzzyList).query_one("Input").focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        mapping = {
            "btn-connect": self.action_connect,
            "btn-trace": self.action_trace,
            "btn-probes": self.action_probes,
            "btn-history": self.action_history,
        }
        fn = mapping.get(event.button.id or "")
        if fn:
            fn()


# =====================================================================
# Helpers
# =====================================================================


def build_provider_from_config(cfg: TUIConfig):
    from LLmThoughtLens.providers.registry import get_provider

    kwargs: dict = {}
    if cfg.provider == "openai":
        kwargs = {"model": cfg.model or "gpt-4o-mini", "api_key": cfg.api_key or None}
    elif cfg.provider == "anthropic":
        kwargs = {"model": cfg.model or "claude-3-5-haiku-20241022", "api_key": cfg.api_key or None}
    elif cfg.provider == "huggingface":
        kwargs = {"model_name": cfg.model or "gpt2"}
    elif cfg.provider == "ollama":
        kwargs = {
            "model": cfg.model or "llama3.2",
            "base_url": cfg.base_url or "http://localhost:11434",
        }
    elif cfg.provider == "mock":
        kwargs = {}
    try:
        return get_provider(cfg.provider, **kwargs)
    except Exception:  # noqa: BLE001
        return None


def _now_short() -> str:
    import datetime

    return datetime.datetime.now().strftime("%m-%d %H:%M")


def _ascii_heatmap(result: TraceResult) -> str:
    tokens = result.output.tokens or ["<empty>"]
    # Aggregate score per token
    agg = dict.fromkeys(range(len(tokens)), 0.0)
    for f in result.features:
        if f.token_idx < len(tokens):
            agg[f.token_idx] += max(0.0, f.score)
    max_s = max(agg.values()) if agg else 1.0
    bars = "▁▂▃▄▅▆▇█"
    lines: list[str] = []
    line1 = "  ".join(tokens)
    lines.append(f"output → [b]{result.output_token}[/b]  ({result.output.output_prob:.2f})")
    lines.append("")
    lines.append(line1)
    bar_row = ""
    for i, _t in enumerate(tokens):
        frac = agg[i] / (max_s + 1e-9)
        idx = min(len(bars) - 1, int(frac * (len(bars) - 1)))
        bar_row += bars[idx] + " " * (len(tokens[i]) + 1)
    lines.append(bar_row)
    return "\n".join(lines)


def _ascii_features(result: TraceResult) -> str:
    top = result.top_features(8)
    if not top:
        return ""
    tbl = Table(title="Top features", show_header=True, header_style="bold")
    tbl.add_column("ID", justify="right")
    tbl.add_column("Layer", justify="right")
    tbl.add_column("Token", justify="right")
    tbl.add_column("Score", justify="right")
    tbl.add_column("Label")
    for f in top:
        tbl.add_row(str(f.id), str(f.layer), str(f.token_idx), f"{f.score:.3f}", f.label)
    from io import StringIO

    from rich.console import Console

    buf = StringIO()
    Console(file=buf, force_terminal=False).print(tbl)
    return buf.getvalue()
