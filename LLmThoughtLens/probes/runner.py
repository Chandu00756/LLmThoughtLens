"""ProbeRunner — execute a battery of probes against a provider.

For multi-model / multi-seed benchmark matrices with environment capture and
scorecards, use :func:`LLmThoughtLens.bench.run_benchmark`.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from LLmThoughtLens.probes.base import BaseProbe, GenerationConfig, ProbeResult

if TYPE_CHECKING:
    from LLmThoughtLens.providers.base import BaseProvider


@dataclass
class ProbeReport:
    """Aggregated results from a probe battery run."""

    provider: str
    model: str
    results: list[ProbeResult] = field(default_factory=list)

    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def n_total(self) -> int:
        return len(self.results)

    @property
    def synthetic(self) -> bool:
        """``True`` when any result came from a synthetic provider (not a model finding)."""
        return any(r.synthetic for r in self.results)

    @property
    def mean_score(self) -> float:
        if not self.results:
            return 0.0
        return float(sum(r.score for r in self.results) / len(self.results))

    def as_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "n_total": self.n_total,
            "n_passed": self.n_passed,
            "mean_score": self.mean_score,
            "synthetic": self.synthetic,
            "results": [r.as_dict() for r in self.results],
        }

    def to_json(self, path: str | Path | None = None, indent: int = 2) -> str | None:
        text = json.dumps(self.as_dict(), indent=indent)
        if path is None:
            return text
        Path(path).write_text(text)
        return None


class ProbeRunner:
    """Run a list of probes against a single provider and collect their results."""

    def __init__(self, probes: Iterable[BaseProbe] | None = None) -> None:
        self._probes: list[BaseProbe] = list(probes or [])

    def add(self, probe: BaseProbe) -> ProbeRunner:
        self._probes.append(probe)
        return self

    @property
    def probes(self) -> list[BaseProbe]:
        return list(self._probes)

    def run_all(
        self,
        provider: BaseProvider,
        progress_callback: Callable[[int, int, BaseProbe], None] | None = None,
        generation: GenerationConfig | None = None,
    ) -> ProbeReport:
        """Run every registered probe.  *progress_callback(i, n, probe)* is called
        before each probe so the TUI can render progress bars.  *generation*,
        when given, overrides every probe's decoding settings for this run."""
        results: list[ProbeResult] = []
        for i, probe in enumerate(self._probes):
            if generation is not None:
                probe = probe.with_generation(generation)
            if progress_callback is not None:
                progress_callback(i, len(self._probes), probe)
            try:
                result = probe.run(provider)
            except Exception as exc:  # noqa: BLE001
                result = ProbeResult(
                    probe_name=probe.name,
                    score=0.0,
                    passed=False,
                    evidence={"error": repr(exc)},
                    summary=f"probe raised: {exc}",
                )
            results.append(result)
        return ProbeReport(
            provider=provider.name,
            model=provider.model_id,
            results=results,
        )

    def __repr__(self) -> str:
        names = [p.name for p in self._probes]
        return f"ProbeRunner(probes={names})"
