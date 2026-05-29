"""BaseProbe — abstract base class for all llmscope probes."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from llmscope.probes.runner import ProbeResult


class BaseProbe(abc.ABC):
    """Abstract base class that every probe must extend.

    A probe takes activation tensors from a :class:`~llmscope.providers.base.ProviderOutput`
    and returns a :class:`~llmscope.probes.runner.ProbeResult` that encodes
    some interpretability-relevant property (e.g. direction linearity, feature
    clustering, …).

    Parameters
    ----------
    name:
        Human-readable identifier for this probe.
    """

    def __init__(self, name: str = "") -> None:
        self.name = name or type(self).__name__

    @abc.abstractmethod
    def run(self, activations: "np.ndarray") -> "ProbeResult":
        """Execute the probe on the given activation tensor.

        Parameters
        ----------
        activations:
            Shape ``(n_layers, n_tokens, d_model)``.

        Returns
        -------
        ProbeResult
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
