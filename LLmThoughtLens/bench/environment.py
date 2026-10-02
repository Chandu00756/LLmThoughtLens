"""Environment capture for reproducible benchmark records.

:func:`capture_environment` records the interpreter, OS, package versions,
compute devices and the git commit of the source tree.  It never fails:
anything that cannot be determined is ``None``.

* Package versions come from installed distribution metadata
  (:mod:`importlib.metadata`), so optional extras are never imported for it.
* Devices are probed only when torch is importable, and only inside the
  function (core installs never import torch).
* The commit comes from the read-only ``git rev-parse HEAD`` and
  ``git status --porcelain --untracked-files=no``; nothing is written.
"""

from __future__ import annotations

import datetime as _dt
import importlib.metadata as _md
import importlib.util
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

__all__ = ["PACKAGES", "capture_environment", "git_info"]

#: Distributions whose versions are recorded.
PACKAGES: tuple[str, ...] = (
    "LLmThoughtLens",
    "numpy",
    "torch",
    "transformers",
    "tokenizers",
    "safetensors",
    "huggingface_hub",
    "accelerate",
    "httpx",
    "plotly",
    "openai",
    "anthropic",
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _version(dist: str) -> str | None:
    try:
        return _md.version(dist)
    except _md.PackageNotFoundError:
        return None
    except Exception:  # noqa: BLE001 — broken metadata must not break a run
        return None


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        proc = subprocess.run(  # noqa: S603 — fixed read-only git invocations
            ["git", *args],  # noqa: S607
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout if proc.returncode == 0 else None


def git_info(repo_root: str | Path | None = None) -> dict[str, Any]:
    """``{"commit", "dirty"}`` of the source tree (``None`` values outside a git checkout)."""
    root = Path(repo_root) if repo_root is not None else _REPO_ROOT
    commit = _git(["rev-parse", "HEAD"], root)
    status = _git(["status", "--porcelain", "--untracked-files=no"], root) if commit else None
    return {
        "commit": commit.strip() if commit else None,
        "dirty": None if status is None else bool(status.strip()),
    }


def _devices() -> dict[str, Any]:
    info: dict[str, Any] = {"torch_available": False, "cuda": None, "mps": None}
    try:
        if importlib.util.find_spec("torch") is None:
            return info
    except (ImportError, ValueError):
        return info
    try:
        import torch

        info["torch_available"] = True
        info["cuda"] = bool(torch.cuda.is_available())
        if info["cuda"]:
            info["cuda_device"] = str(torch.cuda.get_device_name(0))
        mps = getattr(torch.backends, "mps", None)
        info["mps"] = bool(mps is not None and mps.is_available())
        info["default_device"] = "cuda" if info["cuda"] else ("mps" if info["mps"] else "cpu")
        info["torch_threads"] = int(torch.get_num_threads())
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def capture_environment(
    *, include_devices: bool = True, repo_root: str | Path | None = None
) -> dict[str, Any]:
    """Return a JSON-safe description of the machine and software running a benchmark.

    Parameters
    ----------
    include_devices:
        Probe torch for CUDA / MPS (imports torch when it is installed).
    repo_root:
        Git checkout to read the commit from; defaults to this source tree.
    """
    env: dict[str, Any] = {
        "captured_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "packages": {name: _version(name) for name in PACKAGES},
        "git": git_info(repo_root),
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
        "argv0": Path(sys.argv[0]).name if sys.argv and sys.argv[0] else None,
    }
    try:
        from LLmThoughtLens import __version__

        env["llmthoughtlens_version"] = __version__
    except Exception:  # noqa: BLE001
        env["llmthoughtlens_version"] = None
    if include_devices:
        env["devices"] = _devices()
    return env
