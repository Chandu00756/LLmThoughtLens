"""Metrics registry — the plugin hook for extra per-cell and per-model numbers.

A benchmark record stores every metric under its name, so adding a metric
never changes the result schema:

* cell metrics → ``cells[i]["metrics"][name]``
* model metrics → ``models[j]["metrics"][name]``

each as ``{"status": "ok" | "skipped" | "error", "value": …, "reason": str}``.
A metric that cannot apply (wrong provider kind, missing implementation)
is recorded as ``skipped`` with the reason — it never fails the run.

Built-in metrics
----------------
``first_token_confidence`` (cell)
    Mean *real* first-token probability over the probe's completions.
    Skipped for providers that expose no probabilities and for synthetic
    providers; the 1.0 placeholder of black-box APIs is never used.
``attribution_faithfulness`` (model)
    Optional.  Delegates to ``fn(provider, prompt) -> float | Mapping[str, float]``,
    by default ``LLmThoughtLens.circuits.patching.attribution_faithfulness``
    when that module exposes it.  Skipped when no implementation is
    available or the provider is not a white-box model with gradients.

Adding a metric::

    from LLmThoughtLens.bench.metrics import Metric, register_metric

    class MyMetric(Metric):
        name = "my_metric"
        level = "model"

        def compute(self, ctx):
            return 0.5

    register_metric(MyMetric())
    # or swap the faithfulness implementation without touching the schema:
    register_metric(AttributionFaithfulness(fn=my_fn), replace=True)
"""

from __future__ import annotations

import abc
import importlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from LLmThoughtLens.probes.base import is_synthetic_provider

if TYPE_CHECKING:
    from LLmThoughtLens.bench.spec import ModelSpec
    from LLmThoughtLens.probes.base import BaseProbe, ProbeResult, UsageMeter
    from LLmThoughtLens.providers.base import BaseProvider

__all__ = [
    "DEFAULT_METRICS",
    "AttributionFaithfulness",
    "FirstTokenConfidence",
    "Metric",
    "MetricContext",
    "MetricSkipped",
    "evaluate_metric",
    "get_metric",
    "list_metrics",
    "register_metric",
    "unregister_metric",
]

MetricLevel = Literal["cell", "model"]


class MetricSkipped(Exception):  # noqa: N818 — a control-flow signal, not an error
    """Raise from :meth:`Metric.compute` to record ``status="skipped"`` with a reason."""


@dataclass
class MetricContext:
    """Everything a metric may read.

    ``probe`` / ``result`` / ``usage`` are set for cell metrics; ``cells``
    (the model's finished cell records) for model metrics.
    """

    level: MetricLevel
    spec: ModelSpec
    provider: BaseProvider | None
    config: Any = None
    probe: BaseProbe | None = None
    result: ProbeResult | None = None
    usage: UsageMeter | None = None
    cells: list[dict[str, Any]] = field(default_factory=list)


class Metric(abc.ABC):
    """Base class for registry metrics."""

    name: ClassVar[str] = ""
    level: ClassVar[MetricLevel] = "cell"
    description: ClassVar[str] = ""

    def available(self, ctx: MetricContext) -> tuple[bool, str]:
        """``(True, "")`` when :meth:`compute` can run for *ctx*, else ``(False, reason)``."""
        return True, ""

    @abc.abstractmethod
    def compute(self, ctx: MetricContext) -> Any:
        """Return a JSON-safe value (number, string, list or dict)."""


# ---------------------------------------------------------------------------
# Built-in metrics
# ---------------------------------------------------------------------------


class FirstTokenConfidence(Metric):
    """Mean real first-token probability over a cell's completions."""

    name = "first_token_confidence"
    level: ClassVar[MetricLevel] = "cell"
    description = "Mean real probability of the first completion token (None if unavailable)."

    def available(self, ctx: MetricContext) -> tuple[bool, str]:
        if ctx.provider is not None and is_synthetic_provider(ctx.provider):
            return False, "synthetic provider: no real probabilities"
        if ctx.usage is None or not ctx.usage.completions:
            return False, "probe made no metered completion calls"
        return True, ""

    def compute(self, ctx: MetricContext) -> Any:
        assert ctx.usage is not None
        probs = [c.first_token_prob for c in ctx.usage.completions]
        real = [float(p) for p in probs if p is not None]
        if not real:
            raise MetricSkipped("provider exposed no real token probabilities")
        return {"mean": sum(real) / len(real), "n": len(real), "of": len(probs)}


#: Fallback prompts for :class:`AttributionFaithfulness`.
FAITHFULNESS_PROMPTS: tuple[str, ...] = (
    "The capital of the state containing Dallas is",
    "The capital of France is",
)


class AttributionFaithfulness(Metric):
    """Optional attribution-faithfulness score for white-box models.

    Parameters
    ----------
    fn:
        ``fn(provider, prompt) -> float | Mapping[str, float]``.  When ``None``,
        ``LLmThoughtLens.circuits.patching.attribution_faithfulness`` is used
        if it exists.
    prompts:
        Prompts to evaluate; values are averaged (per key for mappings).
    """

    name = "attribution_faithfulness"
    level: ClassVar[MetricLevel] = "model"
    description = "Agreement between attribution scores and causal patching effects."

    MODULE = "LLmThoughtLens.circuits.patching"
    FUNCTION = "attribution_faithfulness"

    def __init__(
        self,
        fn: Callable[[Any, str], Any] | None = None,
        prompts: Sequence[str] = FAITHFULNESS_PROMPTS,
    ) -> None:
        self.fn = fn
        self.prompts = tuple(prompts)

    def _resolve(self) -> tuple[Callable[[Any, str], Any] | None, str]:
        if self.fn is not None:
            return self.fn, ""
        try:
            mod = importlib.import_module(self.MODULE)
        except Exception as exc:  # noqa: BLE001 — absent / broken optional module
            return None, f"{self.MODULE} not importable ({type(exc).__name__})"
        fn = getattr(mod, self.FUNCTION, None)
        if not callable(fn):
            return None, (
                f"{self.MODULE}.{self.FUNCTION} not found; register an adapter with "
                "register_metric(AttributionFaithfulness(fn=...), replace=True)"
            )
        return fn, ""

    def available(self, ctx: MetricContext) -> tuple[bool, str]:
        if ctx.provider is None:
            return False, "provider unavailable"
        if is_synthetic_provider(ctx.provider):
            return False, "synthetic provider"
        if not getattr(ctx.provider, "supports_gradients", False):
            return False, "requires a white-box provider with gradient support (HuggingFace)"
        fn, reason = self._resolve()
        return (fn is not None), reason

    def compute(self, ctx: MetricContext) -> Any:
        fn, reason = self._resolve()
        if fn is None:
            raise MetricSkipped(reason)
        values = [fn(ctx.provider, p) for p in self.prompts]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            nums = [float(v) for v in values]
            return {"mean": sum(nums) / len(nums), "per_prompt": nums}
        if all(isinstance(v, Mapping) for v in values):
            keys = sorted({k for v in values for k in v})
            means: dict[str, float | None] = {}
            for k in keys:
                xs = [float(v[k]) for v in values if isinstance(v.get(k), (int, float))]
                means[k] = sum(xs) / len(xs) if xs else None
            return {"mean": means, "per_prompt": [dict(v) for v in values]}
        raise TypeError("attribution faithfulness fn must return a float or a mapping of floats")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Metric] = {}

#: Metrics computed when a run does not name its own.
DEFAULT_METRICS: tuple[str, ...] = ("first_token_confidence", "attribution_faithfulness")


def register_metric(metric: Metric, *, replace: bool = False) -> Metric:
    """Add *metric* to the registry (``replace=True`` swaps an existing one)."""
    if not metric.name:
        raise ValueError("metric.name must be a non-empty string")
    if metric.level not in ("cell", "model"):
        raise ValueError(f"metric.level must be 'cell' or 'model', got {metric.level!r}")
    if metric.name in _REGISTRY and not replace:
        raise ValueError(f"metric {metric.name!r} already registered (pass replace=True)")
    _REGISTRY[metric.name] = metric
    return metric


def unregister_metric(name: str) -> None:
    """Remove *name* from the registry (no error if absent)."""
    _REGISTRY.pop(name, None)


def get_metric(name: str) -> Metric:
    """Return the registered metric *name* (``KeyError`` if unknown)."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown metric {name!r}; registered: {list_metrics()}") from None


def list_metrics() -> list[str]:
    """Registered metric names, sorted."""
    return sorted(_REGISTRY)


def _json_safe(value: Any) -> Any:
    """Return *value* if it serialises to strict JSON, else raise ``TypeError``."""

    def _finite(v: Any) -> Any:
        if isinstance(v, float) and not math.isfinite(v):
            return None
        if isinstance(v, dict):
            return {str(k): _finite(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [_finite(x) for x in v]
        return v

    cleaned = _finite(value)
    json.dumps(cleaned, allow_nan=False)
    return cleaned


def evaluate_metric(metric: Metric, ctx: MetricContext) -> dict[str, Any]:
    """Run *metric* on *ctx* and return its status record (never raises)."""
    try:
        ok, reason = metric.available(ctx)
        if not ok:
            return {"status": "skipped", "value": None, "reason": reason}
        value = metric.compute(ctx)
        if value is None:
            return {"status": "skipped", "value": None, "reason": "metric returned no value"}
        return {"status": "ok", "value": _json_safe(value), "reason": ""}
    except MetricSkipped as exc:
        return {"status": "skipped", "value": None, "reason": str(exc)}
    except Exception as exc:  # noqa: BLE001 — metric failures are recorded, not raised
        return {"status": "error", "value": None, "reason": f"{type(exc).__name__}: {exc}"}


register_metric(FirstTokenConfidence())
register_metric(AttributionFaithfulness())
