"""Headless behavioural tests for the Textual TUI screens, widgets and config.

Each test mounts the real :class:`LLmThoughtLensApp` with Textual's ``run_test``
pilot (inside ``asyncio.run`` — no async pytest plugin needed), drives screen
actions directly, waits for workers, and asserts on the resulting widget
content, screen stack, persisted config and exported files.  The config
directory is always redirected to ``tmp_path``; the provider is the
deterministic mock.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

pytest.importorskip("textual")

from LLmThoughtLens.probes.base import ProbeResult  # noqa: E402
from LLmThoughtLens.providers.base import BaseProvider, ProviderOutput  # noqa: E402
from LLmThoughtLens.providers.mock_provider import MockProvider  # noqa: E402
from LLmThoughtLens.scope import Scope  # noqa: E402
from LLmThoughtLens.tui import app as tui_app  # noqa: E402
from LLmThoughtLens.tui import config as tui_config  # noqa: E402
from LLmThoughtLens.tui import screens  # noqa: E402
from LLmThoughtLens.tui.app import LLmThoughtLensApp  # noqa: E402
from LLmThoughtLens.tui.config import (  # noqa: E402
    HISTORY_LIMIT,
    SessionEntry,
    TUIConfig,
    load_config,
    save_config,
)
from LLmThoughtLens.tui.widgets import (  # noqa: E402
    AsciiAttributionGraph,
    FuzzyList,
    ProbeProgress,
    static_text,
)
from textual.app import App  # noqa: E402
from textual.widgets import Button, Checkbox, Input, ListItem, ListView, Static  # noqa: E402

PROMPT = "the capital of France is"


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def config_path(tmp_path, monkeypatch):
    """Never touch the real ~/.LLmThoughtLens — redirect the TUI config dir."""
    cfg_dir = tmp_path / "cfg"
    monkeypatch.setattr(tui_config, "CONFIG_DIR", cfg_dir)
    monkeypatch.setattr(tui_config, "CONFIG_PATH", cfg_dir / "config.json")
    return cfg_dir / "config.json"


@pytest.fixture(scope="module")
def result():
    return Scope.from_mock(n_layers=3, n_heads=2, d_model=16, seed=1).trace_full(PROMPT)


def _text(widget: Static) -> str:
    """Plain text of a Static across Textual versions (``content`` / ``renderable``)."""
    value = getattr(widget, "content", None)
    if value is None:
        value = getattr(widget, "renderable", "")
    return str(value)


def _drive(scenario: Callable[[LLmThoughtLensApp, Any], Awaitable[None]], cfg=None) -> None:
    async def _run() -> None:
        app = LLmThoughtLensApp(cfg or TUIConfig())
        async with app.run_test(size=(160, 60)) as pilot:
            await pilot.pause()
            await scenario(app, pilot)

    asyncio.run(_run())


class _FailingProvider(BaseProvider):
    evidence_kind = "black_box"

    def run(self, prompt: str, **kwargs: Any) -> ProviderOutput:
        raise ConnectionError("upstream unreachable")


# ===========================================================================
# ConnectScreen
# ===========================================================================


class TestConnectScreen:
    def test_test_connection_with_mock_reports_ok(self):
        async def scenario(app, pilot):
            screen = screens.ConnectScreen(app.cfg)
            await app.push_screen(screen)
            await pilot.pause()
            screen.on_button_pressed(Button.Pressed(screen.query_one("#btn-test", Button)))
            status = _text(screen.query_one("#connect-status", Static))
            assert "OK" in status
            assert "provider mock" in status
            assert "evidence=white_box" in status
            assert "tokens" in status

        _drive(scenario)

    def test_test_connection_reports_missing_provider(self, monkeypatch):
        monkeypatch.setattr(screens, "build_provider_from_config", lambda cfg: None)

        async def scenario(app, pilot):
            screen = screens.ConnectScreen(app.cfg)
            await app.push_screen(screen)
            await pilot.pause()
            screen.action_test()
            assert "could not instantiate provider" in _text(
                screen.query_one("#connect-status", Static)
            )

        _drive(scenario)

    def test_test_connection_reports_run_failure(self, monkeypatch):
        monkeypatch.setattr(screens, "build_provider_from_config", lambda cfg: _FailingProvider())

        async def scenario(app, pilot):
            screen = screens.ConnectScreen(app.cfg)
            await app.push_screen(screen)
            await pilot.pause()
            screen.action_test()
            status = _text(screen.query_one("#connect-status", Static))
            assert "connection failed" in status
            assert "upstream unreachable" in status

        _drive(scenario)

    def test_save_persists_form_and_opens_trace_screen(self, config_path):
        async def scenario(app, pilot):
            screen = screens.ConnectScreen(app.cfg)
            await app.push_screen(screen)
            await pilot.pause()
            screen.query_one("#model-input", Input).value = "my-model"
            screen.query_one("#key-input", Input).value = "sk-secret-value"
            screen.query_one("#url-input", Input).value = "http://box:1"
            screen.on_button_pressed(Button.Pressed(screen.query_one("#btn-save", Button)))
            await pilot.pause()
            assert isinstance(app.screen, screens.TraceScreen)
            assert app.cfg.model == "my-model"
            assert app.cfg.api_key == "sk-secret-value"

        _drive(scenario)
        saved = json.loads(config_path.read_text())
        assert saved["provider"] == "mock"
        assert saved["model"] == "my-model"
        assert saved["base_url"] == "http://box:1"
        # The key is NOT persisted because the user did not opt in.
        assert saved["api_key"] == ""
        assert saved["save_api_key"] is False

    def test_save_persists_key_when_opted_in(self, config_path):
        async def scenario(app, pilot):
            screen = screens.ConnectScreen(app.cfg)
            await app.push_screen(screen)
            await pilot.pause()
            screen.query_one("#key-input", Input).value = "sk-keep-me"
            screen.query_one("#save-key", Checkbox).value = True
            screen.action_save()
            await pilot.pause()

        _drive(scenario)
        assert json.loads(config_path.read_text())["api_key"] == "sk-keep-me"


# ===========================================================================
# TraceScreen
# ===========================================================================


class TestTraceScreen:
    def test_run_traces_prompt_records_history_and_renders(self, config_path):
        async def scenario(app, pilot):
            screen = screens.TraceScreen(app.cfg)
            await app.push_screen(screen)
            await pilot.pause()
            screen.query_one("#prompt-input", Input).value = PROMPT
            screen.on_button_pressed(Button.Pressed(screen.query_one("#btn-run", Button)))
            await app.workers.wait_for_complete()
            await pilot.pause()

            assert screen.last_result is not None
            assert app.last_result is screen.last_result
            assert screen.last_result.prompt == PROMPT
            heat = _text(screen.query_one("#heatmap-render", Static))
            assert f"output → [b]{screen.last_result.output_token}[/b]" in heat
            assert "the  capital  of  France  is" in heat
            feats = _text(screen.query_one("#features-render", Static))
            assert "Top features" in feats
            top = screen.last_result.top_features(1)[0]
            assert str(top.id) in feats
            assert screen.query_one("#loading").display is False

            entry = app.cfg.history[0]
            assert entry.prompt == PROMPT
            assert entry.provider == "mock"
            assert entry.output_token == screen.last_result.output_token

        _drive(scenario)
        saved = json.loads(config_path.read_text())
        assert saved["last_prompt"] == PROMPT
        assert saved["history"][0]["prompt"] == PROMPT

    def test_empty_prompt_falls_back_to_last_prompt(self):
        cfg = TUIConfig(last_prompt="hello brave world")

        async def scenario(app, pilot):
            screen = screens.TraceScreen(app.cfg)
            await app.push_screen(screen)
            await pilot.pause()
            screen.query_one("#prompt-input", Input).value = ""
            screen.action_run()
            await app.workers.wait_for_complete()
            assert screen.last_result.prompt == "hello brave world"

        _drive(scenario, cfg)

    def test_missing_provider_renders_error(self, monkeypatch):
        monkeypatch.setattr(screens, "build_provider_from_config", lambda cfg: None)

        async def scenario(app, pilot):
            screen = screens.TraceScreen(app.cfg)
            await app.push_screen(screen)
            await pilot.pause()
            screen.action_run()
            await app.workers.wait_for_complete()
            assert "Could not instantiate provider" in _text(
                screen.query_one("#heatmap-render", Static)
            )
            assert screen.last_result is None

        _drive(scenario)

    def test_trace_exception_renders_error(self, monkeypatch):
        def _boom(self, *a, **k):
            raise RuntimeError("tracer exploded")

        monkeypatch.setattr(Scope, "trace_full", _boom)

        async def scenario(app, pilot):
            screen = screens.TraceScreen(app.cfg)
            await app.push_screen(screen)
            await pilot.pause()
            screen.action_run()
            await app.workers.wait_for_complete()
            text = _text(screen.query_one("#heatmap-render", Static))
            assert "trace failed" in text and "tracer exploded" in text
            assert app.cfg.history == []

        _drive(scenario)

    def test_navigation_requires_a_result(self, result):
        async def scenario(app, pilot):
            screen = screens.TraceScreen(app.cfg)
            await app.push_screen(screen)
            await pilot.pause()
            depth = len(app.screen_stack)
            for btn in ("#btn-feats", "#btn-graph", "#btn-export"):
                screen.on_button_pressed(Button.Pressed(screen.query_one(btn, Button)))
            await pilot.pause()
            assert len(app.screen_stack) == depth  # nothing to show yet

            screen.last_result = result
            expected = {
                "#btn-feats": screens.FeatureBrowserScreen,
                "#btn-graph": screens.GraphSummaryScreen,
                "#btn-export": screens.ExportScreen,
                "#btn-probes": screens.ProbeScreen,
            }
            for btn, cls in expected.items():
                screen.on_button_pressed(Button.Pressed(screen.query_one(btn, Button)))
                await pilot.pause()
                assert isinstance(app.screen, cls)
                app.pop_screen()
                await pilot.pause()

        _drive(scenario)


# ===========================================================================
# ProbeScreen
# ===========================================================================


class TestProbeScreen:
    def test_select_none_all_and_run_subset(self):
        async def scenario(app, pilot):
            screen = screens.ProbeScreen(app.cfg)
            await app.push_screen(screen)
            await pilot.pause()
            boxes = list(screen.query(Checkbox).results())
            assert len(boxes) == 10 and all(b.value for b in boxes)
            rows = list(screen.query(ProbeProgress).results())
            assert [r.name for r in rows] == [p.name for p in screen._probes]
            assert all(type(r) is ProbeProgress for r in rows)  # the real widget, no shim
            screen.action_select_none()
            assert not any(b.value for b in boxes)
            screen.action_select_all()
            assert all(b.value for b in boxes)
            screen.action_select_none()

            chosen = screen._probes[:2]
            for p in chosen:
                screen.query_one(f"#ch-{p.name}", Checkbox).value = True
            screen.action_run()
            await app.workers.wait_for_complete()
            await pilot.pause()

            provider = MockProvider()
            expected_pass = sum(1 for p in chosen if p.run(provider).passed)
            summary = _text(screen.query_one("#probe-summary", Static))
            assert f"{expected_pass} / 2 passed." in summary
            for p in chosen:
                assert screen._progress[p.name].state == "done"
                assert screen._progress[p.name].score == pytest.approx(p.run(provider).score)
            for p in screen._probes[2:]:
                assert screen._progress[p.name].state == "pending"

        _drive(scenario)

    def test_crashing_probe_is_scored_zero_and_skipped(self):
        async def scenario(app, pilot):
            screen = screens.ProbeScreen(app.cfg)
            await app.push_screen(screen)
            await pilot.pause()
            bad, good = screen._probes[0], screen._probes[1]

            def _explode(provider):
                raise ValueError("probe bug")

            bad.run = _explode  # type: ignore[method-assign]
            good.run = lambda provider: ProbeResult(good.name, score=0.9, passed=True)  # type: ignore[method-assign]
            await screen._run_probes([bad, good])
            assert screen._progress[bad.name].state == "done"
            assert screen._progress[bad.name].score == 0.0
            assert screen._progress[good.name].score == pytest.approx(0.9)
            assert "1 / 2 passed." in _text(screen.query_one("#probe-summary", Static))

        _drive(scenario)

    def test_missing_provider(self, monkeypatch):
        monkeypatch.setattr(screens, "build_provider_from_config", lambda cfg: None)

        async def scenario(app, pilot):
            screen = screens.ProbeScreen(app.cfg)
            await app.push_screen(screen)
            await pilot.pause()
            await screen._run_probes(screen._probes)
            assert "Provider not configured" in _text(screen.query_one("#probe-summary", Static))

        _drive(scenario)


# ===========================================================================
# Feature browser / graph summary / export
# ===========================================================================


class TestResultScreens:
    def test_feature_browser_lists_features_and_shows_selection(self, result):
        async def scenario(app, pilot):
            screen = screens.FeatureBrowserScreen(result)
            await app.push_screen(screen)
            await pilot.pause()
            lv = screen.query_one(ListView)
            assert len(lv.children) == len(result.features)
            screen.action_cursor_down()
            screen.action_cursor_down()
            screen.action_cursor_up()
            screen.on_fuzzy_list_selected(FuzzyList.Selected("feature 42"))
            assert "feature 42" in _text(screen.query_one("#feature-detail", Static))

        _drive(scenario)

    def test_graph_summary_renders_top_paths(self, result):
        async def scenario(app, pilot):
            screen = screens.GraphSummaryScreen(result)
            await app.push_screen(screen)
            await pilot.pause()
            rendered = str(screen.query_one(AsciiAttributionGraph).render())
            paths = result.graph.top_paths(n=5)
            if paths:
                assert rendered.startswith("1. ")
                assert len(rendered.splitlines()) == len(paths)
            else:
                assert rendered == "(graph has no paths)"

        _drive(scenario)

    def test_export_writes_every_artifact(self, result, tmp_path):
        prefix = tmp_path / "out" / "trace"
        prefix.parent.mkdir()

        async def scenario(app, pilot):
            screen = screens.ExportScreen(result)
            await app.push_screen(screen)
            await pilot.pause()
            screen.query_one("#prefix-input", Input).value = str(prefix)
            for btn in ("#btn-html", "#btn-gjson", "#btn-gcsv", "#btn-fcsv"):
                screen.on_button_pressed(Button.Pressed(screen.query_one(btn, Button)))
                assert "wrote" in _text(screen.query_one("#export-status", Static))

        _drive(scenario)
        assert (
            (tmp_path / "out" / "trace.html").read_text().lstrip().lower().startswith("<!doctype")
        )
        graph = json.loads((tmp_path / "out" / "trace.graph.json").read_text())
        assert len(graph["nodes"]) == result.graph.num_nodes
        assert (tmp_path / "out" / "trace.graph.csv").read_text().strip()
        csv_lines = (tmp_path / "out" / "trace.features.csv").read_text().strip().splitlines()
        assert csv_lines[0].startswith("id,label,layer,token_idx,score")
        assert len(csv_lines) == len(result.features) + 1

    def test_export_failure_is_reported(self, result, tmp_path):
        async def scenario(app, pilot):
            screen = screens.ExportScreen(result)
            await app.push_screen(screen)
            await pilot.pause()
            screen.query_one("#prefix-input", Input).value = str(tmp_path / "missing" / "x")
            screen.on_button_pressed(Button.Pressed(screen.query_one("#btn-fcsv", Button)))
            assert "export failed" in _text(screen.query_one("#export-status", Static))

        _drive(scenario)


# ===========================================================================
# HomeScreen + app-level actions
# ===========================================================================


class TestHomeAndApp:
    def test_home_summary_history_and_navigation(self):
        cfg = TUIConfig(model="tiny", top_k_features=7, api_cost_usd=1.5)
        cfg.push_history(SessionEntry("01-01 00:00", "mock", "mock-L", "hello", "world"))

        async def scenario(app, pilot):
            home = app.screen
            assert isinstance(home, screens.HomeScreen)
            summary = _text(home.query_one("#summary", Static))
            assert "provider: [b]mock[/b]" in summary
            assert "model: [b]tiny[/b]" in summary
            assert "top-k: 7" in summary
            assert "$ spent: 1.50" in summary
            items = home.query_one(ListView).children
            assert len(items) == 1

            home.on_button_pressed(Button.Pressed(home.query_one("#btn-history", Button)))
            await pilot.pause()
            assert isinstance(app.focused, Input)

            for btn, cls in (
                ("#btn-connect", screens.ConnectScreen),
                ("#btn-trace", screens.TraceScreen),
                ("#btn-probes", screens.ProbeScreen),
            ):
                home.on_button_pressed(Button.Pressed(home.query_one(btn, Button)))
                await pilot.pause()
                assert isinstance(app.screen, cls)
                app.pop_screen()
                await pilot.pause()

        _drive(scenario, cfg)

    def test_home_without_history_shows_placeholder(self):
        async def scenario(app, pilot):
            assert "(no model)" not in _text(app.screen.query_one("#summary", Static))
            assert "model: [b](none)[/b]" in _text(app.screen.query_one("#summary", Static))

        _drive(scenario)

    def test_app_home_and_connect_actions(self):
        async def scenario(app, pilot):
            app.action_connect()
            await pilot.pause()
            assert isinstance(app.screen, screens.ConnectScreen)
            app.push_screen(screens.TraceScreen(app.cfg))
            await pilot.pause()
            app.action_home()
            await pilot.pause()
            assert isinstance(app.screen, screens.HomeScreen)
            assert len(app.screen_stack) == 2  # default screen + fresh home

        _drive(scenario)

    def test_app_loads_persisted_config_by_default(self, config_path):
        save_config(TUIConfig(provider="mock", model="persisted-model"))
        assert LLmThoughtLensApp().cfg.model == "persisted-model"

    def test_run_tui_constructs_and_runs_app(self, monkeypatch):
        ran: list[LLmThoughtLensApp] = []
        monkeypatch.setattr(LLmThoughtLensApp, "run", lambda self, *a, **k: ran.append(self))
        cfg = TUIConfig(model="x")
        tui_app.run_tui(cfg)
        assert len(ran) == 1 and ran[0].cfg is cfg


# ===========================================================================
# Pure helpers
# ===========================================================================


class TestHelpers:
    def test_build_provider_from_config_mock(self):
        assert isinstance(screens.build_provider_from_config(TUIConfig()), MockProvider)

    def test_build_provider_from_config_unknown_returns_none(self):
        assert screens.build_provider_from_config(TUIConfig(provider="does-not-exist")) is None

    def test_build_provider_from_config_forwards_model(self):
        p = screens.build_provider_from_config(
            TUIConfig(provider="ollama", model="tiny", base_url="http://box:1/")
        )
        assert p is not None
        assert p.model == "tiny"
        assert p.base_url == "http://box:1"

    def test_ascii_heatmap_bars_follow_feature_scores(self, result):
        text = screens._ascii_heatmap(result)
        lines = text.splitlines()
        assert lines[0].startswith(f"output → [b]{result.output_token}[/b]")
        assert lines[2] == "  ".join(result.output.tokens)
        bars = [c for c in lines[3] if c in "▁▂▃▄▅▆▇█"]
        assert len(bars) == len(result.output.tokens)
        agg = [0.0] * len(result.output.tokens)
        for f in result.features:
            agg[f.token_idx] += max(0.0, f.score)
        # Bar height is monotone in the aggregate positive feature score.
        glyph = "▁▂▃▄▅▆▇█"
        heights = [glyph.index(b) for b in bars]
        order = sorted(range(len(agg)), key=lambda i: agg[i])
        assert [heights[i] for i in order] == sorted(heights)
        assert heights[agg.index(max(agg))] == max(heights) == glyph.index("█")

    @pytest.mark.parametrize(
        ("value", "max_value", "expected"),
        [
            (0.0, 0.0, 0),  # all-zero row -> bottom bar, no ZeroDivisionError
            (0.0, 2.0, 0),
            (2.0, 2.0, 7),  # the maximum always gets the full bar
            (1e9, 1e9, 7),
            (1.0, 2.0, 4),  # 0.5 * 7 = 3.5 -> rounds to 4
            (2.0 / 7.0, 2.0, 1),
            (0.01, 2.0, 0),
            (3.0, 2.0, 7),  # clamped
            (-1.0, 2.0, 0),
        ],
    )
    def test_heat_bucket(self, value, max_value, expected):
        assert screens._heat_bucket(value, max_value) == expected

    def test_ascii_heatmap_max_token_gets_full_bar(self, result):
        """Regression: ``agg / (max + 1e-9)`` floored the maximum into the ▇ bucket."""
        import copy

        from LLmThoughtLens.features.feature import Feature

        tokens = result.output.tokens
        fake = copy.copy(result)
        fake.features = [
            Feature(id=0, label="a", layer=0, token_idx=1, score=0.9),
            Feature(id=1, label="b", layer=0, token_idx=1, score=0.3),
            Feature(id=2, label="c", layer=0, token_idx=0, score=0.6),
            Feature(id=3, label="d", layer=0, token_idx=-1, score=5.0),  # ignored, no crash
            Feature(id=4, label="e", layer=0, token_idx=len(tokens), score=5.0),  # ignored
        ]
        bars = [c for c in screens._ascii_heatmap(fake).splitlines()[3] if c in "▁▂▃▄▅▆▇█"]
        assert bars[1] == "█"
        assert bars[0] == "▅"  # 0.6 / 1.2 = 0.5 -> 3.5 rounds to bucket 4
        assert set(bars[2:]) == {"▁"}

    def test_ascii_heatmap_escapes_markup_in_tokens(self, result):
        import copy

        from rich.text import Text

        fake = copy.copy(result)
        fake.output = copy.copy(result.output)
        fake.output.tokens = ["[INST]", "[/b]", "x"]
        fake.output.top_tokens = [("[/i]", 0.5)]
        fake.features = []
        rendered = Text.from_markup(screens._ascii_heatmap(fake)).plain  # must not raise
        assert "[INST]  [/b]  x" in rendered
        assert "output → [/i]" in rendered

    def test_ascii_features_table(self, result):
        text = screens._ascii_features(result)
        assert "Top features" in text
        for f in result.top_features(8):
            assert f"{f.score:.3f}" in text

    def test_ascii_features_empty(self, result):
        import copy

        empty = copy.copy(result)
        empty.features = []
        assert screens._ascii_features(empty) == ""

    def test_now_short_format(self):
        import re

        assert re.fullmatch(r"\d\d-\d\d \d\d:\d\d", screens._now_short())


# ===========================================================================
# TUI config persistence
# ===========================================================================


class TestTUIConfig:
    def test_session_entry_label_truncates_long_prompts(self):
        short = SessionEntry("t", "mock", "m", "hi", "there").label()
        assert "'hi'" in short and "'there'" in short
        long_prompt = "x" * 60
        label = SessionEntry("t", "mock", "m", long_prompt, "y").label()
        assert "x" * 45 + "…" in label
        assert "x" * 46 not in label

    def test_history_is_capped(self):
        cfg = TUIConfig()
        for i in range(HISTORY_LIMIT + 5):
            cfg.push_history(SessionEntry(str(i), "mock", "m", f"p{i}", "o"))
        assert len(cfg.history) == HISTORY_LIMIT
        assert cfg.history[0].prompt == f"p{HISTORY_LIMIT + 4}"  # newest first

    def test_json_roundtrip_ignores_unknown_keys(self):
        cfg = TUIConfig(provider="mock", model="m", top_k_features=3)
        cfg.push_history(SessionEntry("t", "mock", "m", "p", "o", score_mean=0.5))
        data = cfg.to_json()
        data["some_future_field"] = 1
        back = TUIConfig.from_json(data)
        assert back.model == "m" and back.top_k_features == 3
        assert back.history == cfg.history

    def test_load_missing_or_corrupt_returns_defaults(self, config_path):
        assert load_config() == TUIConfig()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("{not json")
        assert load_config() == TUIConfig()
        config_path.write_text(json.dumps({"history": [{"bogus": 1}]}))
        assert load_config() == TUIConfig()

    def test_save_then_load_roundtrip_strips_unopted_key(self, config_path):
        cfg = TUIConfig(model="m", api_key="sk-temp", save_api_key=False)
        save_config(cfg)
        loaded = load_config()
        assert loaded.model == "m"
        assert loaded.api_key == ""
        assert cfg.api_key == "sk-temp"  # in-memory config untouched


# ===========================================================================
# Widgets
# ===========================================================================


class TestWidgets:
    def test_probe_progress_constructs(self):
        """Regression: ``self.name = name`` hit Textual's read-only ``Widget.name``."""
        w = ProbeProgress("my_probe")
        assert w.name == "my_probe"
        assert w.probe_name == "my_probe"
        assert w.state == "pending"
        assert w.score is None

    def test_probe_progress_set_running_and_done_on_a_mounted_widget(self):
        class _Host(App):
            def compose(self):
                yield ProbeProgress("mounted_probe")

        async def _run():
            app = _Host()
            async with app.run_test() as pilot:
                await pilot.pause()
                w = app.query_one(ProbeProgress)
                w.set_running()
                await pilot.pause()
                assert w.state == "running" and "running" in str(w.render())
                w.set_done(0.75)
                await pilot.pause()
                assert (w.state, w.score) == ("done", 0.75)
                assert "mounted_probe" in str(w.render()) and "0.75" in str(w.render())

        asyncio.run(_run())

    def test_probe_progress_states(self):
        w = ProbeProgress("my_probe")
        assert w.name == "my_probe"
        assert str(w.render()).strip().startswith("·")
        w.state = "running"
        assert "running" in str(w.render())
        w.state, w.score = "done", 0.8
        assert str(w.render()).strip().startswith("✓") and "0.80" in str(w.render())
        w.score = 0.2
        assert str(w.render()).strip().startswith("✗")

    def test_ascii_graph_without_graph(self):
        assert str(AsciiAttributionGraph().render()) == "(no graph)"

    def test_ascii_graph_handles_unknown_nodes_and_empty_paths(self):
        class _G:
            def __init__(self, paths):
                self._paths = paths

            def top_paths(self, n=5):
                return self._paths

            def node(self, nid):
                return None if nid == 99 else type("N", (), {"label": f"n{nid}", "id": nid})()

        g = AsciiAttributionGraph()
        g._graph = _G([[1, 99, 2]])
        assert str(g.render()) == "1. n1 → ?99 → n2"
        g._graph = _G([])
        assert str(g.render()) == "(graph has no paths)"

    def test_fuzzy_list_filters_and_updates(self):
        class _Host(App):
            def compose(self):
                yield FuzzyList(["alpha feature", "beta feature", "gamma"], placeholder="f")

        async def _run():
            app = _Host()
            async with app.run_test() as pilot:
                await pilot.pause()
                fl = app.query_one(FuzzyList)
                lv = fl.query_one(ListView)
                assert len(lv.children) == 3
                fl.query_one(Input).value = "zzzzzz"
                await pilot.pause()
                assert len(lv.children) == 0
                fl.query_one(Input).value = ""
                await pilot.pause()
                assert len(lv.children) == 3
                fl.update_items(["only one"])
                await pilot.pause()
                assert len(lv.children) == 1

        asyncio.run(_run())

    def test_fuzzy_list_selection_posts_selected_value(self):
        """Regression: the handler read ``Static.renderable`` (gone in Textual >= 2)."""
        selected: list[str] = []

        class _Host(App):
            def compose(self):
                yield FuzzyList(["alpha feature"])

            def on_fuzzy_list_selected(self, event: FuzzyList.Selected) -> None:
                selected.append(event.value)

        async def _run():
            app = _Host()
            async with app.run_test() as pilot:
                await pilot.pause()
                fl = app.query_one(FuzzyList)
                lv = fl.query_one(ListView)
                item = lv.children[0]
                assert isinstance(item, ListItem)
                fl.on_list_view_selected(ListView.Selected(lv, item, 0))
                await pilot.pause()
            assert selected == ["alpha feature"]

        asyncio.run(_run())

    def test_fuzzy_list_selection_keeps_markup_like_text_verbatim(self):
        """Labels with ``[`` are shown literally and selected exactly as given."""
        raw = "  7  layer  1  tok  0  score   0.500  [b]bold[/b] [/oops]"
        selected: list[str] = []

        class _Host(App):
            def compose(self):
                yield FuzzyList([raw])

            def on_fuzzy_list_selected(self, event: FuzzyList.Selected) -> None:
                selected.append(event.value)

        async def _run():
            app = _Host()
            async with app.run_test() as pilot:
                await pilot.pause()
                lv = app.query_one(FuzzyList).query_one(ListView)
                item = lv.children[0]
                assert static_text(item.query_one(Static)) == raw  # rendered literally
                lv.index = 0
                lv.action_select_cursor()  # the real keyboard path (Enter)
                await pilot.pause()
            assert selected == [raw]

        asyncio.run(_run())

    def test_fuzzy_list_selection_of_a_foreign_list_item_reads_its_static(self):
        selected: list[str] = []

        class _Host(App):
            def compose(self):
                yield FuzzyList([])

            def on_fuzzy_list_selected(self, event: FuzzyList.Selected) -> None:
                selected.append(event.value)

        async def _run():
            app = _Host()
            async with app.run_test() as pilot:
                await pilot.pause()
                fl = app.query_one(FuzzyList)
                lv = fl.query_one(ListView)
                foreign = ListItem(Static("plain row"))
                empty = ListItem()
                await lv.append(foreign)
                await lv.append(empty)
                fl.on_list_view_selected(ListView.Selected(lv, foreign, 0))
                fl.on_list_view_selected(ListView.Selected(lv, empty, 1))  # ignored
                await pilot.pause()
            assert selected == ["plain row"]

        asyncio.run(_run())

    def test_static_text_accessor_across_textual_versions(self):
        from rich.text import Text

        assert static_text(Static("hello")) == "hello"
        assert static_text(Static(Text("rich text"))) == "rich text"

        class _OldStatic:  # Textual < 2 exposed ``renderable`` instead of ``content``
            renderable = "legacy"

        class _Bare:
            pass

        assert static_text(_OldStatic()) == "legacy"  # type: ignore[arg-type]
        assert static_text(_Bare()) == ""  # type: ignore[arg-type]
