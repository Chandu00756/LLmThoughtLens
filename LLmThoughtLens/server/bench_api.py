"""Benchmark results API — view the JSON records written by ``LLmThoughtLens benchmark``.

``GET  /api/bench/results``         list ``*.json`` records under the config home's
                                     ``bench/`` directory (``~/.LLmThoughtLens/bench``,
                                     or ``$LLMTHOUGHTLENS_HOME/bench``).
``GET  /api/bench/results/{name}``  validate one of them and return a summary.
``POST /api/bench/view``            validate a record the browser uploaded (file
                                     picker) and return the same summary.

Records are validated with :func:`LLmThoughtLens.bench.validate_result` (schema
``llmthoughtlens.bench`` v1); an invalid file is reported, never half-rendered.
Only files inside the bench directory can be read (no path traversal).  The
summary keeps the record's honesty flags: per-model ``status`` and
``synthetic`` (mock results are never model findings) and per-cell error counts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse

__all__ = ["bench_dir", "bench_summary", "build_router"]


def bench_dir() -> Path:
    """``<config home>/bench`` (read at call time so tests can redirect the config home)."""
    from LLmThoughtLens.server import config_api

    return Path(config_api.CONFIG_DIR) / "bench"


def bench_summary(data: dict[str, Any]) -> dict[str, Any]:
    """Validated, compact view of one benchmark record (raises ``ValueError`` if invalid)."""
    from LLmThoughtLens.bench import BenchResult, validate_result

    problems = validate_result(data)
    if problems:
        raise ValueError("invalid benchmark record: " + "; ".join(problems[:5]))
    result = BenchResult(data)
    env = result.environment
    return {
        "schema": data.get("schema"),
        "schema_version": data.get("schema_version"),
        "run_id": data.get("run_id"),
        "title": data.get("title"),
        "created_utc": data.get("created_utc"),
        "duration_s": data.get("duration_s"),
        "notes": list(data.get("notes", [])),
        "probes": result.probe_names(),
        "models": [
            {
                k: m.get(k)
                for k in (
                    "label",
                    "provider",
                    "model",
                    "model_id",
                    "status",
                    "error",
                    "evidence_kind",
                    "synthetic",
                    "duration_s",
                    "cost_usd",
                )
            }
            for m in result.models
        ],
        "by_model": list(result.aggregates.get("by_model", [])),
        "by_model_probe": list(result.aggregates.get("by_model_probe", [])),
        "environment": {
            k: env.get(k) for k in ("python", "platform", "packages", "git") if k in env
        },
    }


def _within(base: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(base.resolve())
    except ValueError:
        return False
    return True


def build_router() -> APIRouter:
    router = APIRouter(prefix="/api/bench", tags=["bench"])

    @router.get("/results")
    def list_results() -> dict[str, Any]:
        base = bench_dir()
        files: list[dict[str, Any]] = []
        if base.is_dir():
            for path in sorted(base.rglob("*.json")):
                if not _within(base, path) or not path.is_file():
                    continue
                stat = path.stat()
                files.append(
                    {
                        "name": path.relative_to(base).as_posix(),
                        "size": stat.st_size,
                        "mtime": stat.st_mtime,
                    }
                )
        return {"dir": str(base), "results": files}

    @router.get("/results/{name:path}")
    def get_result(name: str) -> Any:
        import json

        base = bench_dir()
        path = base / name
        if not name.endswith(".json") or not _within(base, path) or not path.is_file():
            return JSONResponse({"error": f"no benchmark record {name!r} in {base}"}, 404)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("not a JSON object")
            return {"name": name, **bench_summary(data)}
        except ValueError as exc:
            return JSONResponse({"error": f"{name}: {exc}"}, 422)

    @router.post("/view")
    def post_view(data: dict[str, Any] = Body(...)) -> Any:  # noqa: B008 — FastAPI idiom
        try:
            return bench_summary(data)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, 422)

    return router
