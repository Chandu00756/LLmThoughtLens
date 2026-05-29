"""ProbeResult and ProbeRunner — result container and batch executor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from llmscope.probes.base import BaseProbe


@dataclass
class ProbeResult:
    """Structured result returned by a :class:`~llmscope.probes.base.BaseProbe`.

    Attributes
    ----------
    probe_name:
        Name of the probe that produced this result.
    score:
        Scalar summary score (interpretation depends on probe type).
    activations:
        Optionally the probed activation slice.
    labels:
        Token-level labels or scores.
    meta:
        Freeform probe metadata.
    """

    probe_name: str
    score: float = 0.0
    activations: "np.ndarray | None" = None
    labels: list[Any] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"ProbeResult(probe={self.probe_name!r}, score={self.score:.4f})"


class ProbeRunner:
    """Run a collection of probes against a set of activation tensors.

    Parameters
    ----------
    probes:
        Iterable of :class:`~llmscope.probes.base.BaseProbe` instances.
    """

    def __init__(self, probes: list["BaseProbe"] | None = None) -> None:
        self._probes: list["BaseProbe"] = list(probes or [])

    def add(self, probe: "BaseProbe") -> None:
        """Register an additional probe."""
        self._probes.append(probe)

    def run_all(self, activations: "np.ndarray") -> list[ProbeResult]:
        """Run all registered probes and return their results.

        Parameters
        ----------
        activations:
            Shape ``(n_layers, n_tokens, d_model)``.
        """
        results: list[ProbeResult] = []
        for probe in self._probes:
            result = probe.run(activations)
            results.append(result)
        return results

    def __repr__(self) -> str:
        names = [p.name for p in self._probes]
        return f"ProbeRunner(probes={names})"
