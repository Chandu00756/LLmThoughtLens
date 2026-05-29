"""ReportBuilder — assembles interpretability results into an HTML report."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReportSection:
    title: str
    content: str  # HTML fragment
    meta: dict[str, Any] = field(default_factory=dict)


class ReportBuilder:
    """Incrementally build an HTML interpretability report.

    Parameters
    ----------
    title:
        Top-level report title.
    """

    def __init__(self, title: str = "llmscope Report") -> None:
        self.title = title
        self._sections: list[ReportSection] = []

    def add_section(self, title: str, content: str, **meta: Any) -> "ReportBuilder":
        """Append a new section to the report.

        Parameters
        ----------
        title:
            Section heading.
        content:
            HTML content for this section.
        **meta:
            Optional key-value metadata stored on the section.
        """
        self._sections.append(ReportSection(title=title, content=content, meta=meta))
        return self

    def render(self) -> str:
        """Return the complete HTML report as a string."""
        sections_html = "\n".join(
            f"<section><h2>{s.title}</h2>{s.content}</section>"
            for s in self._sections
        )
        return (
            "<!DOCTYPE html><html><head>"
            f"<title>{self.title}</title>"
            "<style>body{{font-family:sans-serif;max-width:960px;margin:auto;}}"
            "section{{margin-bottom:2em;}}</style>"
            f"</head><body><h1>{self.title}</h1>{sections_html}</body></html>"
        )

    def save(self, path: str) -> None:
        """Write the HTML report to *path*."""
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.render())

    def __repr__(self) -> str:
        return f"ReportBuilder(title={self.title!r}, sections={len(self._sections)})"
