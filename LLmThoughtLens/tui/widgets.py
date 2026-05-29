"""Reusable Textual widgets — fuzzy list, ASCII attribution graph, probe progress."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from rich.text import Text
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Input, ListItem, ListView, Static

try:
    from rapidfuzz import fuzz

    def _score(query: str, candidate: str) -> int:
        return int(fuzz.partial_ratio(query.lower(), candidate.lower()))
except ImportError:  # pragma: no cover — rapidfuzz is in [tui] extras

    def _score(query: str, candidate: str) -> int:
        q = query.lower()
        c = candidate.lower()
        if not q:
            return 100
        return 100 if q in c else 0


# ---------------------------------------------------------------------------
# Fuzzy-search list
# ---------------------------------------------------------------------------


class FuzzyList(Vertical):
    """Input + ListView pair — typing filters the list in real time."""

    DEFAULT_CSS = """
    FuzzyList { height: 1fr; }
    FuzzyList Input { dock: top; }
    FuzzyList ListView { height: 1fr; }
    """

    class Selected(Message):
        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    def __init__(self, items: Iterable[str], placeholder: str = "search…") -> None:
        super().__init__()
        self._items = list(items)
        self._placeholder = placeholder

    def compose(self):
        yield Input(placeholder=self._placeholder, id="fuzzy-input")
        yield ListView(id="fuzzy-list")

    def on_mount(self) -> None:
        self._refresh("")

    def update_items(self, items: Iterable[str]) -> None:
        self._items = list(items)
        self._refresh(self.query_one("#fuzzy-input", Input).value)

    def _refresh(self, query: str) -> None:
        listview = self.query_one("#fuzzy-list", ListView)
        listview.clear()
        scored = [(it, _score(query, it)) for it in self._items]
        scored.sort(key=lambda t: -t[1])
        for it, score in scored:
            if score > 0 or not query:
                listview.append(ListItem(Static(it)))

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "fuzzy-input":
            self._refresh(event.value)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.item is None:
            return
        first = event.item.children[0] if event.item.children else None
        if isinstance(first, Static):
            value = str(first.renderable)
            self.post_message(self.Selected(value))


# ---------------------------------------------------------------------------
# ASCII attribution graph — compact terminal rendering
# ---------------------------------------------------------------------------


class AsciiAttributionGraph(Static):
    """Render top causal paths as compact ASCII."""

    DEFAULT_CSS = """
    AsciiAttributionGraph { padding: 1 2; }
    """

    def render(self) -> Text:
        graph = getattr(self, "_graph", None)
        if graph is None:
            return Text("(no graph)", style="dim")
        paths = graph.top_paths(n=5)
        if not paths:
            return Text("(graph has no paths)", style="dim")
        lines: list[Text] = []
        for i, path in enumerate(paths, 1):
            chunks: list[str] = []
            for nid in path:
                n = graph.node(nid)
                if n is None:
                    chunks.append(f"?{nid}")
                else:
                    chunks.append(n.label or str(n.id))
            line = Text(f"{i}. " + " → ".join(chunks))
            lines.append(line)
        out = Text("\n").join(lines)
        return out

    def show(self, graph: Any) -> None:
        self._graph = graph
        self.refresh(layout=True)


# ---------------------------------------------------------------------------
# Probe progress bar list
# ---------------------------------------------------------------------------


class ProbeProgress(Static):
    """A single-line probe status row used by the probe runner screen."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self.state = "pending"
        self.score: float | None = None

    def render(self) -> Text:
        if self.state == "running":
            mark = "⟳"
            style = "yellow"
            tail = "…running"
        elif self.state == "done":
            mark = "✓" if self.score and self.score >= 0.5 else "✗"
            style = "green" if self.score and self.score >= 0.5 else "red"
            tail = f"{self.score:.2f}" if self.score is not None else ""
        else:
            mark = "·"
            style = "dim"
            tail = ""
        return Text(f" {mark} {self.name:<26} {tail}", style=style)

    def set_running(self) -> None:
        self.state = "running"
        self.refresh()

    def set_done(self, score: float) -> None:
        self.state = "done"
        self.score = float(score)
        self.refresh()
