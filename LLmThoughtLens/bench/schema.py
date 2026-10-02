"""Versioned JSON schema for benchmark records.

A record (``schema == "llmthoughtlens.bench"``, ``schema_version == 1``) is a
plain JSON object::

    {
      "schema": "llmthoughtlens.bench", "schema_version": 1,
      "run_id": str, "title": str, "created_utc": str, "duration_s": float,
      "config": {...},            # BenchConfig.as_dict()
      "environment": {...},       # capture_environment()
      "probes": [{"name", "description", "citation", "style", "threshold"}],
      "models": [{"label", "provider", "model", "model_id", "status", "error",
                  "evidence_kind", "synthetic", "supports_gradients", "framing",
                  "server_info", "model_info", "load_s", "duration_s",
                  "usage", "cost_usd", "metrics"}],
      "cells": [{"model", "probe", "repeat", "seed", "status", "score", "passed",
                 "summary", "synthetic", "framing", "completion_source",
                 "duration_s", "usage", "cost_usd", "error", "metrics", "evidence"}],
      "aggregates": {"by_model_probe": [...], "by_model": [...]},
      "metrics": {"requested": [...], "descriptions": {...}},
      "notes": [str]
    }

``status`` is ``"ok"`` or ``"error"`` (a failing model or probe is recorded,
never fatal); metric entries are ``{"status": "ok" | "skipped" | "error",
"value", "reason"}``.  New metrics only add keys under ``metrics`` — the
schema version changes only for incompatible layout changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["SCHEMA_NAME", "SCHEMA_VERSION", "BenchResult", "validate_result"]

SCHEMA_NAME = "llmthoughtlens.bench"
SCHEMA_VERSION = 1

_TOP_KEYS: dict[str, type | tuple[type, ...]] = {
    "schema": str,
    "schema_version": int,
    "run_id": str,
    "title": str,
    "created_utc": str,
    "duration_s": (int, float),
    "config": dict,
    "environment": dict,
    "probes": list,
    "models": list,
    "cells": list,
    "aggregates": dict,
    "metrics": dict,
    "notes": list,
}
_MODEL_KEYS = ("label", "provider", "model", "status", "synthetic", "metrics", "usage")
_CELL_KEYS = (
    "model",
    "probe",
    "repeat",
    "seed",
    "status",
    "score",
    "passed",
    "synthetic",
    "duration_s",
    "usage",
    "error",
    "metrics",
)
_STATUSES = {"ok", "error"}
_METRIC_STATUSES = {"ok", "skipped", "error"}


def validate_result(data: Any) -> list[str]:
    """Return a list of schema problems (empty when *data* is a valid v1 record)."""
    problems: list[str] = []
    if not isinstance(data, dict):
        return ["record is not a JSON object"]
    for key, typ in _TOP_KEYS.items():
        if key not in data:
            problems.append(f"missing top-level key {key!r}")
        elif not isinstance(data[key], typ):
            problems.append(f"top-level key {key!r} has type {type(data[key]).__name__}")
    if data.get("schema") != SCHEMA_NAME:
        problems.append(f"schema is {data.get('schema')!r}, expected {SCHEMA_NAME!r}")
    if data.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version is {data.get('schema_version')!r}")
    labels = set()
    for i, m in enumerate(data.get("models") or []):
        missing = [k for k in _MODEL_KEYS if k not in m]
        if missing:
            problems.append(f"models[{i}] missing {missing}")
        if m.get("status") not in _STATUSES:
            problems.append(f"models[{i}].status = {m.get('status')!r}")
        labels.add(m.get("label"))
        problems.extend(_check_metrics(f"models[{i}]", m.get("metrics")))
    probes = {p.get("name") for p in data.get("probes") or [] if isinstance(p, dict)}
    for i, c in enumerate(data.get("cells") or []):
        missing = [k for k in _CELL_KEYS if k not in c]
        if missing:
            problems.append(f"cells[{i}] missing {missing}")
            continue
        if c["status"] not in _STATUSES:
            problems.append(f"cells[{i}].status = {c['status']!r}")
        if c["model"] not in labels:
            problems.append(f"cells[{i}].model {c['model']!r} not in models")
        if c["probe"] not in probes:
            problems.append(f"cells[{i}].probe {c['probe']!r} not in probes")
        if c["status"] == "ok":
            score = c["score"]
            if not isinstance(score, (int, float)) or not 0.0 <= float(score) <= 1.0:
                problems.append(f"cells[{i}].score {score!r} not in [0, 1]")
            if not isinstance(c["passed"], bool):
                problems.append(f"cells[{i}].passed is not a bool")
        elif not isinstance(c["error"], dict):
            problems.append(f"cells[{i}] has status 'error' but no error record")
        problems.extend(_check_metrics(f"cells[{i}]", c.get("metrics")))
    return problems


def _check_metrics(where: str, metrics: Any) -> list[str]:
    if not isinstance(metrics, dict):
        return [f"{where}.metrics is not an object"]
    out = []
    for name, rec in metrics.items():
        if not isinstance(rec, dict) or rec.get("status") not in _METRIC_STATUSES:
            out.append(f"{where}.metrics[{name!r}] has no valid status")
    return out


@dataclass
class BenchResult:
    """A benchmark record (thin wrapper around the JSON object in :attr:`data`)."""

    data: dict[str, Any]

    @property
    def models(self) -> list[dict[str, Any]]:
        return list(self.data.get("models", []))

    @property
    def cells(self) -> list[dict[str, Any]]:
        return list(self.data.get("cells", []))

    @property
    def probes(self) -> list[dict[str, Any]]:
        return list(self.data.get("probes", []))

    @property
    def environment(self) -> dict[str, Any]:
        return dict(self.data.get("environment", {}))

    @property
    def aggregates(self) -> dict[str, Any]:
        return dict(self.data.get("aggregates", {}))

    def model_labels(self) -> list[str]:
        return [str(m["label"]) for m in self.models]

    def probe_names(self) -> list[str]:
        return [str(p["name"]) for p in self.probes]

    def aggregate(self, model: str, probe: str) -> dict[str, Any] | None:
        """The ``by_model_probe`` aggregate for (*model*, *probe*), if any."""
        for row in self.aggregates.get("by_model_probe", []):
            if row["model"] == model and row["probe"] == probe:
                return dict(row)
        return None

    def validate(self) -> list[str]:
        return validate_result(self.data)

    def to_json(self, path: str | Path | None = None, indent: int = 2) -> str:
        """Serialise (strict JSON); also writes to *path* when given."""
        text = json.dumps(self.data, indent=indent, allow_nan=False, ensure_ascii=False)
        if path is not None:
            Path(path).write_text(text + "\n", encoding="utf-8")
        return text

    @classmethod
    def from_json(cls, source: str | Path) -> BenchResult:
        """Load from a path or a JSON string; raises ``ValueError`` on an invalid record."""
        text = str(source)
        if not text.lstrip().startswith("{"):
            text = Path(source).read_text(encoding="utf-8")
        data = json.loads(text)
        problems = validate_result(data)
        if problems:
            raise ValueError("invalid benchmark record: " + "; ".join(problems[:5]))
        return cls(data)
