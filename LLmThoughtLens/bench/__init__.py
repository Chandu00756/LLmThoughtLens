"""Benchmarks — run the probe suite over a matrix of models and write scorecards.

Everything a CLI needs is one call::

    from LLmThoughtLens.bench import run_and_write

    result, paths = run_and_write(
        ["hf:gpt2", "ollama:qwen3:1.7b", "mock"], "bench-out", repeats=2
    )

or, in two steps, :func:`run_benchmark` (returns a :class:`BenchResult`, a
versioned JSON record — see :mod:`LLmThoughtLens.bench.schema`) and
:func:`write_reports` (JSON + Markdown + self-contained HTML scorecard).

* :mod:`~LLmThoughtLens.bench.spec` — ``"provider:model"`` specs.
* :mod:`~LLmThoughtLens.bench.runner` — execution, failure isolation, timing,
  usage / cost, aggregation.
* :mod:`~LLmThoughtLens.bench.metrics` — the metric registry (plugin hook for
  extra numbers such as attribution faithfulness).
* :mod:`~LLmThoughtLens.bench.environment` — reproducibility record.
* :mod:`~LLmThoughtLens.bench.scorecard` — Markdown / HTML rendering.

Importing this package needs only the core dependencies; optional extras
(torch, transformers, httpx, …) are imported when a model that needs them is
built.  Results from synthetic providers (the mock) are flagged
``synthetic`` everywhere and are never model findings.  Probe methods and
limitations: ``docs/benchmarks/methodology.md``.
"""

from LLmThoughtLens.bench.environment import capture_environment, git_info
from LLmThoughtLens.bench.metrics import (
    DEFAULT_METRICS,
    AttributionFaithfulness,
    FirstTokenConfidence,
    Metric,
    MetricContext,
    MetricSkipped,
    evaluate_metric,
    get_metric,
    list_metrics,
    register_metric,
    unregister_metric,
)
from LLmThoughtLens.bench.runner import (
    BenchConfig,
    ProgressFn,
    aggregate,
    run_and_write,
    run_benchmark,
)
from LLmThoughtLens.bench.schema import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    BenchResult,
    validate_result,
)
from LLmThoughtLens.bench.scorecard import render_html, render_markdown, write_reports
from LLmThoughtLens.bench.spec import ModelSpec, parse_spec, parse_specs

__all__ = [
    "DEFAULT_METRICS",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "AttributionFaithfulness",
    "BenchConfig",
    "BenchResult",
    "FirstTokenConfidence",
    "Metric",
    "MetricContext",
    "MetricSkipped",
    "ModelSpec",
    "ProgressFn",
    "aggregate",
    "capture_environment",
    "evaluate_metric",
    "get_metric",
    "git_info",
    "list_metrics",
    "parse_spec",
    "parse_specs",
    "register_metric",
    "render_html",
    "render_markdown",
    "run_and_write",
    "run_benchmark",
    "unregister_metric",
    "validate_result",
    "write_reports",
]
