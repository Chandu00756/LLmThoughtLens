"""BaseProbe — abstract base class for every LLmThoughtLens probe.

A probe takes a :class:`~LLmThoughtLens.providers.base.BaseProvider`,
runs a small set of prompts against it, and returns a
:class:`ProbeResult` whose ``score`` is in ``[0, 1]`` and whose
``evidence`` dict contains the raw model outputs that motivated the score.

Concrete probes live in :mod:`LLmThoughtLens.probes.builtin`.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from LLmThoughtLens.providers.base import BaseProvider


@dataclass
class ProbeResult:
    """Structured result returned by every probe."""

    probe_name: str
    score: float = 0.0
    passed: bool = False
    evidence: dict[str, Any] = field(default_factory=dict)
    summary: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "probe_name": self.probe_name,
            "score": float(self.score),
            "passed": bool(self.passed),
            "summary": self.summary,
            "evidence": _scrub(self.evidence),
        }

    def __repr__(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return f"ProbeResult({verdict} {self.probe_name!r} score={self.score:.2f})"


def _scrub(d: Any) -> Any:
    """Drop non-JSON-safe entries so ``as_dict`` always serialises cleanly."""
    if isinstance(d, dict):
        return {
            k: _scrub(v) for k, v in d.items() if _is_json_safe(v) or isinstance(v, (dict, list))
        }
    if isinstance(d, list):
        return [_scrub(v) for v in d if _is_json_safe(v) or isinstance(v, (dict, list))]
    return d


def _is_json_safe(v: Any) -> bool:
    return isinstance(v, (str, int, float, bool)) or v is None


class BaseProbe(abc.ABC):
    """Every probe must subclass this and implement :meth:`run`."""

    #: Short, machine-readable name (snake_case).
    name: str = "base_probe"
    #: One-line human description for the report.
    description: str = ""
    #: Optional citation back to the paper case study.
    citation: str = ""

    @abc.abstractmethod
    def run(
        self,
        provider: BaseProvider,
        prompt: str | None = None,
    ) -> ProbeResult:
        """Execute the probe and return its :class:`ProbeResult`.

        Parameters
        ----------
        provider:
            Backend that will receive the probe's prompts.
        prompt:
            Optional override of the probe's built-in prompt(s).  When
            ``None`` the probe uses its hard-coded prompt set.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
