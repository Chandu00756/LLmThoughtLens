"""LLmThoughtLensApp — top-level Textual application class."""

from __future__ import annotations

from textual.app import App
from textual.binding import Binding

from LLmThoughtLens.tui.config import TUIConfig, load_config
from LLmThoughtLens.tui.screens import (
    ConnectScreen,
    HomeScreen,
    ProbeScreen,
    TraceScreen,
)


class LLmThoughtLensApp(App):
    """Top-level Textual app — pushes the home screen, holds session state."""

    TITLE = "LLmThoughtLens"
    SUB_TITLE = "interpretability TUI"
    CSS = """
    Screen { layout: vertical; }
    #title { padding: 1 2; }
    #connect-body { padding: 1 2; }
    #connect-actions { padding: 1 0; }
    #trace-actions { padding: 1 0; }
    #loading { dock: bottom; height: 1; }
    #trace-output { padding: 1 2; }
    #probe-list { height: 1fr; padding: 1 2; }
    #probe-summary { padding: 1 2; }
    #history-scroll { height: 1fr; padding: 0 2; }
    Button { margin: 0 1; }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "quit", show=True),
        Binding("ctrl+h", "home", "home", show=True),
        Binding("ctrl+k", "connect", "connect", show=True),
    ]

    def __init__(self, cfg: TUIConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or load_config()
        self.last_result = None

    def on_mount(self) -> None:
        self.push_screen(HomeScreen(self.cfg))

    # ------------------------------------------------------------------
    # Global actions
    # ------------------------------------------------------------------

    def action_home(self) -> None:
        while len(self.screen_stack) > 1:
            self.pop_screen()
        self.push_screen(HomeScreen(self.cfg))

    def action_connect(self) -> None:
        self.push_screen(ConnectScreen(self.cfg))


def run_tui(cfg: TUIConfig | None = None) -> None:
    """Launch the interactive Textual TUI."""
    try:
        import textual  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The TUI needs the `tui` extra. Install with: pip install 'LLmThoughtLens[tui]'"
        ) from exc
    app = LLmThoughtLensApp(cfg)
    app.run()


# Silence unused-import linting — these classes are intentionally re-exported
# at TUI-package level so external callers can patch them.
_EXPORTED = (ProbeScreen, TraceScreen, ConnectScreen)
