"""Headless TUI tests — mount the Textual app and actually render screens.

These guard against regressions like a method named ``_render`` colliding with
Textual's internal ``Widget._render`` (which crashes the screen on draw).  We
drive the app with Textual's ``run_test`` pilot inside ``asyncio.run`` so no
async-pytest plugin is required.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("textual")

from LLmThoughtLens.scope import Scope  # noqa: E402
from LLmThoughtLens.tui.app import LLmThoughtLensApp  # noqa: E402
from LLmThoughtLens.tui.config import TUIConfig  # noqa: E402
from LLmThoughtLens.tui.screens import (  # noqa: E402
    ConnectScreen,
    FeatureBrowserScreen,
    GraphSummaryScreen,
    TraceScreen,
)


def test_app_mounts_home_with_enhanced_css():
    async def _run():
        app = LLmThoughtLensApp(TUIConfig())
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.screen is not None
            assert type(app.screen).__name__ == "HomeScreen"

    asyncio.run(_run())


def test_trace_screen_renders_real_result():
    async def _run():
        app = LLmThoughtLensApp(TUIConfig())
        async with app.run_test() as pilot:
            await pilot.pause()
            ts = TraceScreen(app.cfg)
            app.push_screen(ts)
            await pilot.pause()
            result = Scope.from_mock(n_layers=3, n_heads=2, d_model=16).trace_full(
                "the capital of France is"
            )
            # Must not raise — exercises _render_result + a real draw pass.
            ts._render_result(result)
            await pilot.pause()

    asyncio.run(_run())


def test_graph_and_feature_screens_draw():
    async def _run():
        app = LLmThoughtLensApp(TUIConfig())
        result = Scope.from_mock(seed=2).trace_full("hello world")
        async with app.run_test() as pilot:
            await pilot.pause()
            app.push_screen(GraphSummaryScreen(result))
            await pilot.pause()
            app.push_screen(FeatureBrowserScreen(result))
            await pilot.pause()
            app.push_screen(ConnectScreen(app.cfg))
            await pilot.pause()

    asyncio.run(_run())
