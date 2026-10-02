"""Benchmark runner — a probe suite over a matrix of (provider, model) specs.

One call runs everything and returns a versioned :class:`BenchResult`::

    from LLmThoughtLens.bench import run_benchmark, write_reports

    result = run_benchmark(["hf:gpt2", "ollama:qwen3:1.7b"], repeats=1)
    write_reports(result, "bench-out")      # scorecard.json / .md / .html

Semantics
---------
* **Cells.**  Every (model, probe, repeat) is one cell.  Repeat *i* decodes
  with seed ``seeds[i]`` (default ``base_seed + i``).  At ``temperature=0``
  decoding is greedy, so repeats only re-check determinism.
* **Failure isolation.**  A model whose provider cannot be built (missing
  extra, unknown model, server down) is recorded with ``status="error"`` and
  so are its cells; a probe that raises fails only its own cell.  The run
  always completes.
* **Timing and usage.**  ``load_s`` covers provider construction and, for
  local providers, a warm-up (model load) so cell timings exclude it.  Cell
  ``usage`` sums every :func:`~LLmThoughtLens.probes.base.complete` call made
  by the probe (token counts are ``None`` when a backend does not report
  them).  ``cost_usd`` comes from provider-reported cost or from
  ``BenchConfig.prices`` (USD per million input / output tokens); otherwise
  it is ``None`` — local models have no API cost, not a zero-cost estimate.
* **Honesty.**  Cells from synthetic providers carry ``synthetic=True`` and
  their summaries say so; probabilities are only ever real ones.
* **Metrics.**  Extra numbers come from the registry in
  :mod:`LLmThoughtLens.bench.metrics`; adding one never changes the schema.
"""

from __future__ import annotations

import contextlib
import dataclasses
import gc
import math
import sys
import time
import traceback
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from LLmThoughtLens.bench.environment import capture_environment
from LLmThoughtLens.bench.metrics import (
    DEFAULT_METRICS,
    Metric,
    MetricContext,
    evaluate_metric,
    get_metric,
)
from LLmThoughtLens.bench.schema import SCHEMA_NAME, SCHEMA_VERSION, BenchResult
from LLmThoughtLens.bench.spec import ModelSpec, parse_specs
from LLmThoughtLens.probes.base import (
    BaseProbe,
    GenerationConfig,
    ProbeResult,
    UsageMeter,
    complete,
    is_synthetic_provider,
    record_usage,
)

if TYPE_CHECKING:
    from LLmThoughtLens.providers.base import BaseProvider

__all__ = ["BenchConfig", "ProgressFn", "aggregate", "run_and_write", "run_benchmark"]

#: ``progress(event)`` receives dicts such as
#: ``{"event": "cell_done", "model": "hf/gpt2", "probe": "multi_hop", "repeat": 0,
#: "index": 3, "total": 20, "status": "ok", "score": 1.0}``.
ProgressFn = Callable[[dict[str, Any]], None]


@dataclass
class BenchConfig:
    """Settings for :func:`run_benchmark`.

    Parameters
    ----------
    repeats:
        Number of repeats per (model, probe); ignored when *seeds* is given.
    seeds:
        Explicit per-repeat seeds.
    base_seed:
        First seed when *seeds* is not given.
    temperature:
        Decoding temperature for every probe call (``0`` = greedy).
    chat:
        HuggingFace chat-template switch (``None`` = use it when present).
    metrics:
        Registry metric names; ``None`` = :data:`~LLmThoughtLens.bench.metrics.DEFAULT_METRICS`.
    include_evidence:
        Store each probe's evidence (prompts, responses) in the cell records.
    warmup:
        Load local models before timing cells (no warm-up call is sent to paid APIs).
    capture_env, include_devices:
        Record the environment (and torch devices).
    prices:
        ``{model label or model_id: (usd_per_mtok_in, usd_per_mtok_out)}``.
    title, notes:
        Free text copied into the record.
    """

    repeats: int = 1
    seeds: Sequence[int] | None = None
    base_seed: int = 0
    temperature: float = 0.0
    chat: bool | None = None
    metrics: Sequence[str] | None = None
    include_evidence: bool = True
    warmup: bool = True
    capture_env: bool = True
    include_devices: bool = True
    prices: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    title: str = "LLmThoughtLens probe benchmark"
    notes: Sequence[str] = ()

    def resolved_seeds(self) -> list[int]:
        if self.seeds is not None:
            seeds = [int(s) for s in self.seeds]
            if not seeds:
                raise ValueError("seeds must not be empty")
            return seeds
        if int(self.repeats) < 1:
            raise ValueError(f"repeats must be >= 1, got {self.repeats}")
        return [int(self.base_seed) + i for i in range(int(self.repeats))]

    def as_dict(self) -> dict[str, Any]:
        return {
            "repeats": len(self.resolved_seeds()),
            "seeds": self.resolved_seeds(),
            "temperature": float(self.temperature),
            "deterministic_decoding": float(self.temperature) <= 0.0,
            "chat": self.chat,
            "metrics": list(self.metrics) if self.metrics is not None else list(DEFAULT_METRICS),
            "include_evidence": self.include_evidence,
            "warmup": self.warmup,
            "prices": {k: list(v) for k, v in self.prices.items()},
        }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_benchmark(
    models: Sequence[str | ModelSpec] | str,
    probes: Sequence[str | BaseProbe] | None = None,
    config: BenchConfig | None = None,
    *,
    progress: ProgressFn | None = None,
    **overrides: Any,
) -> BenchResult:
    """Run *probes* over every model in *models* and return the record.

    Parameters
    ----------
    models:
        Specs (``"hf:gpt2"``, ``"ollama:qwen3:1.7b"``, ``"mock"`` or
        :class:`ModelSpec`), or one comma-separated string.
    probes:
        Probe names or instances; ``None`` runs every built-in probe.
    config:
        :class:`BenchConfig`; keyword *overrides* replace its fields
        (``run_benchmark(specs, repeats=3, temperature=0.7)``).
    progress:
        Optional callback receiving progress events (exceptions it raises are ignored).

    Raises
    ------
    ValueError / KeyError
        Only for invalid arguments (bad spec, unknown probe or metric name,
        duplicate labels) — never for a failing model or probe.
    """
    cfg = dataclasses.replace(config or BenchConfig(), **overrides)
    specs = parse_specs(models)
    probe_list = _resolve_probes(probes)
    seeds = cfg.resolved_seeds()
    metric_names = list(cfg.metrics) if cfg.metrics is not None else list(DEFAULT_METRICS)
    metric_objs = [get_metric(n) for n in metric_names]
    cell_metrics = [m for m in metric_objs if m.level == "cell"]
    model_metrics = [m for m in metric_objs if m.level == "model"]

    t_run = time.perf_counter()
    created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    env = capture_environment(include_devices=cfg.include_devices) if cfg.capture_env else {}
    total = len(specs) * len(seeds) * len(probe_list)
    counter = [0]
    models_out: list[dict[str, Any]] = []
    cells_out: list[dict[str, Any]] = []
    for spec in specs:
        rec, cells = _run_model(
            spec, probe_list, seeds, cfg, cell_metrics, model_metrics, progress, counter, total
        )
        models_out.append(rec)
        cells_out.extend(cells)

    probes_meta = [
        {
            "name": p.name,
            "description": p.description,
            "citation": p.citation,
            "style": getattr(p, "style", "instruction"),
            "threshold": getattr(p, "threshold", ""),
        }
        for p in probe_list
    ]
    data: dict[str, Any] = {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "run_id": uuid.uuid4().hex[:12],
        "title": cfg.title,
        "created_utc": created,
        "duration_s": round(time.perf_counter() - t_run, 3),
        "config": cfg.as_dict(),
        "environment": env,
        "probes": probes_meta,
        "models": models_out,
        "cells": cells_out,
        "aggregates": aggregate(models_out, cells_out, [p.name for p in probe_list]),
        "metrics": {
            "requested": metric_names,
            "descriptions": {m.name: m.description for m in metric_objs},
            "levels": {m.name: m.level for m in metric_objs},
        },
        "notes": _notes(cfg, models_out),
    }
    return BenchResult(data)


def run_and_write(
    models: Sequence[str | ModelSpec] | str,
    out_dir: str | Path,
    probes: Sequence[str | BaseProbe] | None = None,
    *,
    stem: str = "scorecard",
    plotlyjs: str = "inline",
    formats: Sequence[str] = ("json", "md", "html"),
    progress: ProgressFn | None = None,
    **config: Any,
) -> tuple[BenchResult, dict[str, Path]]:
    """:func:`run_benchmark` + :func:`~LLmThoughtLens.bench.scorecard.write_reports` in one call.

    The CLI entry point::

        result, paths = run_and_write(["hf:gpt2", "mock"], "bench-out", repeats=2)

    *config* keywords are :class:`BenchConfig` fields; *stem*, *plotlyjs*
    (``"inline"`` / ``"cdn"`` / ``"svg"`` / ``"none"``) and *formats* go to
    :func:`~LLmThoughtLens.bench.scorecard.write_reports`.  Returns the record
    and ``{"json" | "md" | "html": path}``.
    """
    from LLmThoughtLens.bench.scorecard import write_reports

    result = run_benchmark(models, probes, progress=progress, **config)
    paths = write_reports(
        result,
        out_dir,
        stem=stem,
        plotlyjs=plotlyjs,  # type: ignore[arg-type]  # validated by render_html
        formats=formats,
    )
    return result, paths


# ---------------------------------------------------------------------------
# Per-model execution
# ---------------------------------------------------------------------------


def _resolve_probes(probes: Sequence[str | BaseProbe] | None) -> list[BaseProbe]:
    from LLmThoughtLens.probes.builtin import all_probes, probe_by_name

    if probes is None:
        return all_probes()
    out: list[BaseProbe] = []
    for p in probes:
        if isinstance(p, BaseProbe):
            out.append(p)
            continue
        inst = probe_by_name(str(p))
        if inst is None:
            raise KeyError(f"unknown probe {p!r}")
        out.append(inst)
    names = [p.name for p in out]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate probe names in {names}")
    if not out:
        raise ValueError("at least one probe is required")
    return out


def _emit(progress: ProgressFn | None, event: dict[str, Any]) -> None:
    if progress is None:
        return
    # A broken progress UI must not stop the run.
    with contextlib.suppress(Exception):
        progress(event)


def _error_record(exc: BaseException, stage: str) -> dict[str, Any]:
    home = str(Path.home())
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)[-6:])
    return {
        "stage": stage,
        "type": type(exc).__name__,
        "message": str(exc)[:500].replace(home, "~"),
        "traceback": tb[-2000:].replace(home, "~"),
    }


def _model_info(provider: BaseProvider) -> dict[str, Any] | None:
    """Architecture facts for white-box models (loads the model)."""
    if not (
        getattr(provider, "supports_gradients", False)
        and isinstance(getattr(type(provider), "hooked", None), property)
    ):
        return None
    hm = provider.hooked  # type: ignore[attr-defined]
    info: dict[str, Any] = {
        "family": hm.family,
        "n_layers": hm.n_layers,
        "d_model": hm.d_model,
        "n_heads": hm.n_heads,
        "vocab_size": hm.vocab_size,
        "device": str(hm.device),
        "chat_template": bool(getattr(hm.tokenizer, "chat_template", None)),
        "attn_implementation": hm.attn_implementation,
    }
    try:
        params = list(hm.model.parameters())
        info["dtype"] = str(params[0].dtype).replace("torch.", "") if params else None
        info["n_params"] = int(sum(p.numel() for p in params))
    except Exception:  # noqa: BLE001
        pass
    return info


def _warmup(provider: BaseProvider) -> None:
    """Load the model / server weights so cell timings exclude loading (not metered)."""
    complete(provider, "Hello", GenerationConfig(max_new_tokens=1))


def _price_for(
    spec: ModelSpec, model_id: str | None, cfg: BenchConfig
) -> tuple[float, float] | None:
    for key in (spec.label, model_id or "", spec.model):
        if key and key in cfg.prices:
            pin, pout = cfg.prices[key]
            return float(pin), float(pout)
    return None


def _cell_cost(meter: UsageMeter, price: tuple[float, float] | None) -> float | None:
    if meter.cost_usd is not None:
        return float(meter.cost_usd)
    if price is None or not meter.prompt_tokens_known or not meter.completion_tokens_known:
        return None
    return (meter.prompt_tokens * price[0] + meter.completion_tokens * price[1]) / 1e6


def _blank_cell(spec: ModelSpec, probe: BaseProbe, r: int, seed: int) -> dict[str, Any]:
    return {
        "model": spec.label,
        "probe": probe.name,
        "repeat": r,
        "seed": seed,
        "status": "ok",
        "score": None,
        "passed": None,
        "summary": "",
        "synthetic": False,
        "framing": None,
        "completion_source": None,
        "duration_s": 0.0,
        "usage": UsageMeter().as_dict(),
        "cost_usd": None,
        "error": None,
        "metrics": {},
        "evidence": None,
    }


def _run_model(
    spec: ModelSpec,
    probes: list[BaseProbe],
    seeds: list[int],
    cfg: BenchConfig,
    cell_metrics: list[Metric],
    model_metrics: list[Metric],
    progress: ProgressFn | None,
    counter: list[int],
    total: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rec: dict[str, Any] = {
        "label": spec.label,
        "provider": spec.provider,
        "model": spec.model,
        "spec": spec.as_dict(),
        "model_id": None,
        "status": "ok",
        "error": None,
        "evidence_kind": None,
        "synthetic": False,
        "supports_gradients": None,
        "framing": None,
        "server_info": None,
        "model_info": None,
        "load_s": None,
        "duration_s": None,
        "usage": UsageMeter().as_dict(),
        "cost_usd": None,
        "metrics": {},
    }
    cells: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    _emit(progress, {"event": "model_start", "model": spec.label})

    provider: BaseProvider | None = None
    try:
        provider = spec.build()
        rec["model_id"] = provider.model_id
        rec["evidence_kind"] = provider.evidence_kind
        rec["synthetic"] = is_synthetic_provider(provider)
        rec["supports_gradients"] = bool(getattr(provider, "supports_gradients", False))
        server_info = getattr(provider, "server_info", None)
        if callable(server_info):
            rec["server_info"] = server_info()
        if cfg.warmup and spec.is_local:
            rec["model_info"] = _model_info(provider)
            _warmup(provider)
        rec["load_s"] = round(time.perf_counter() - t0, 3)
    except Exception as exc:  # noqa: BLE001 — recorded per model
        err = _error_record(exc, "provider_init")
        rec.update(status="error", error=err, load_s=round(time.perf_counter() - t0, 3))
        for r, seed in enumerate(seeds):
            for probe in probes:
                cell = _blank_cell(spec, probe, r, seed)
                cell.update(status="error", error=err, summary=f"provider failed: {err['message']}")
                cells.append(cell)
                counter[0] += 1
        _emit(progress, {"event": "model_error", "model": spec.label, "error": err["message"]})
        rec["duration_s"] = round(time.perf_counter() - t0, 3)
        return rec, cells

    _emit(progress, {"event": "model_ready", "model": spec.label, "load_s": rec["load_s"]})
    price = _price_for(spec, rec["model_id"], cfg)
    model_meter = UsageMeter()
    for r, seed in enumerate(seeds):
        gen = GenerationConfig(temperature=float(cfg.temperature), seed=seed, chat=cfg.chat)
        for probe in probes:
            counter[0] += 1
            cell = _blank_cell(spec, probe, r, seed)
            _emit(
                progress,
                {
                    "event": "cell_start",
                    "model": spec.label,
                    "probe": probe.name,
                    "repeat": r,
                    "index": counter[0],
                    "total": total,
                },
            )
            runnable = probe.with_generation(gen)
            result: ProbeResult | None = None
            tc = time.perf_counter()
            with record_usage() as meter:
                try:
                    result = runnable.run(provider)
                    _check_result(result)
                except Exception as exc:  # noqa: BLE001 — recorded per cell
                    cell.update(status="error", error=_error_record(exc, "probe"))
                    cell["summary"] = f"probe raised {type(exc).__name__}: {exc}"[:300]
                    result = None
            cell["duration_s"] = round(time.perf_counter() - tc, 4)
            cell["usage"] = meter.as_dict()
            cell["cost_usd"] = _cell_cost(meter, price)
            for c in meter.completions:
                model_meter.add(c)
            if result is not None:
                synthetic = result.synthetic or bool(rec["synthetic"])
                if synthetic and not result.synthetic:
                    # Custom probe on a synthetic provider: label it here.
                    result.evidence["synthetic"] = True
                    if not result.summary.startswith("[synthetic"):
                        result.summary = "[synthetic provider - not a model finding] " + (
                            result.summary
                        )
                cell.update(
                    score=float(result.score),
                    passed=bool(result.passed),
                    summary=result.summary,
                    synthetic=synthetic,
                    framing=_jsonable(result.evidence.get("framing")),
                    completion_source=_jsonable(result.evidence.get("completion_source")),
                )
                if cfg.include_evidence:
                    cell["evidence"] = result.as_dict()["evidence"]
            else:
                cell["synthetic"] = bool(rec["synthetic"])
            for metric in cell_metrics:
                ctx = MetricContext(
                    level="cell",
                    spec=spec,
                    provider=provider,
                    config=cfg,
                    probe=runnable,
                    result=result,
                    usage=meter,
                )
                cell["metrics"][metric.name] = evaluate_metric(metric, ctx)
            cells.append(cell)
            _emit(
                progress,
                {
                    "event": "cell_done",
                    "model": spec.label,
                    "probe": probe.name,
                    "repeat": r,
                    "index": counter[0],
                    "total": total,
                    "status": cell["status"],
                    "score": cell["score"],
                    "passed": cell["passed"],
                    "duration_s": cell["duration_s"],
                },
            )

    for metric in model_metrics:
        ctx = MetricContext(level="model", spec=spec, provider=provider, config=cfg, cells=cells)
        rec["metrics"][metric.name] = evaluate_metric(metric, ctx)

    rec["usage"] = model_meter.as_dict()
    costs = [c["cost_usd"] for c in cells]
    rec["cost_usd"] = (
        float(sum(c for c in costs if c is not None)) if any(c is not None for c in costs) else None
    )
    framings = sorted({str(c["framing"]) for c in cells if isinstance(c.get("framing"), str)})
    rec["framing"] = framings[0] if len(framings) == 1 else (framings or None)
    rec["duration_s"] = round(time.perf_counter() - t0, 3)
    _emit(progress, {"event": "model_done", "model": spec.label, "duration_s": rec["duration_s"]})
    del provider
    _release_memory()
    return rec, cells


def _check_result(result: Any) -> None:
    if not isinstance(result, ProbeResult):
        raise TypeError(f"probe returned {type(result).__name__}, expected ProbeResult")
    score = float(result.score)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"probe score {result.score!r} is outside [0, 1]")


def _jsonable(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [x for x in v if isinstance(x, (str, int, float, bool))]
    return str(v)


def _release_memory() -> None:
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mps = getattr(torch, "mps", None)
        if mps is not None and torch.backends.mps.is_available():
            mps.empty_cache()
    except Exception:  # noqa: BLE001, S110 — best effort
        pass


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _mean(xs: Sequence[float]) -> float | None:
    return float(sum(xs) / len(xs)) if xs else None


def _pstdev(xs: Sequence[float]) -> float | None:
    if not xs:
        return None
    m = sum(xs) / len(xs)
    return float(math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs)))


def aggregate(
    models: Sequence[Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]],
    probe_names: Sequence[str],
) -> dict[str, Any]:
    """Per-(model, probe) and per-model summaries of *cells*.

    A (model, probe) is ``passed`` when it passed in a strict majority of its
    successful repeats; ``mean_score`` averages successful repeats only.
    """
    by_mp: list[dict[str, Any]] = []
    for m in models:
        label = m["label"]
        for probe in probe_names:
            mine = [c for c in cells if c["model"] == label and c["probe"] == probe]
            ok = [c for c in mine if c["status"] == "ok"]
            scores = [float(c["score"]) for c in ok]
            passes = [1.0 if c["passed"] else 0.0 for c in ok]
            pass_rate = _mean(passes)
            by_mp.append(
                {
                    "model": label,
                    "probe": probe,
                    "n": len(mine),
                    "n_ok": len(ok),
                    "n_error": len(mine) - len(ok),
                    "mean_score": _mean(scores),
                    "std_score": _pstdev(scores),
                    "min_score": min(scores) if scores else None,
                    "max_score": max(scores) if scores else None,
                    "pass_rate": pass_rate,
                    "passed": None if pass_rate is None else pass_rate > 0.5,
                    "synthetic": any(bool(c.get("synthetic")) for c in mine),
                    "consistent": len({(c["score"], c["passed"]) for c in ok}) <= 1,
                }
            )
    by_model: list[dict[str, Any]] = []
    for m in models:
        rows = [r for r in by_mp if r["model"] == m["label"]]
        scored = [r for r in rows if r["mean_score"] is not None]
        n_passed = sum(1 for r in scored if r["passed"])
        by_model.append(
            {
                "model": m["label"],
                "status": m["status"],
                "n_probes": len(rows),
                "n_scored": len(scored),
                "n_passed": n_passed,
                "pass_rate": (n_passed / len(scored)) if scored else None,
                "mean_score": _mean([float(r["mean_score"]) for r in scored]),
                "n_error_cells": int(sum(r["n_error"] for r in rows)),
                "synthetic": bool(m.get("synthetic")),
                "duration_s": m.get("duration_s"),
            }
        )
    return {"by_model_probe": by_mp, "by_model": by_model}


def _notes(cfg: BenchConfig, models: Sequence[Mapping[str, Any]]) -> list[str]:
    notes = [str(n) for n in cfg.notes]
    if float(cfg.temperature) <= 0.0:
        notes.append(
            "temperature=0: decoding is greedy, so repeats beyond the first only re-check "
            "determinism."
        )
    synthetic = [m["label"] for m in models if m.get("synthetic")]
    if synthetic:
        notes.append(
            f"Synthetic providers ({', '.join(synthetic)}): scores are produced by random "
            "logits, not by a model, and are not findings."
        )
    failed = [m["label"] for m in models if m.get("status") == "error"]
    if failed:
        notes.append(f"Models that failed to load: {', '.join(failed)} (see models[].error).")
    return notes
