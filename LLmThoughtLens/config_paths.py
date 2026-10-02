"""Config-home paths and owner-only file helpers shared by the TUI and the server.

This module deliberately depends only on the standard library, so both
:mod:`LLmThoughtLens.tui.config` (``tui`` extra) and
:mod:`LLmThoughtLens.server.config_api` (``server`` extra) can use it under
any install — ``pip install 'LLmThoughtLens[server]'`` must not need Textual.

Permissions model
-----------------
Files under the config home may hold API keys, so they are written with
:func:`write_private_text`:

* every file is written atomically (temp file in the same directory +
  :func:`os.replace`) with ``0o600`` permissions, which also tightens a
  pre-existing world-readable file the first time it is rewritten;
* a missing directory is created ``0o700``;
* the config home itself (``~/.LLmThoughtLens`` or ``$LLMTHOUGHTLENS_HOME``) is
  tightened to ``0o700`` on every config write when it already exists with
  looser permissions — callers opt in with ``tighten_dir=True``.  The user's
  home directory and filesystem roots are never chmod-ed, and neither is a
  directory owned by someone else;
* a config file that is a symlink is *followed*: the link is kept and its
  target is replaced atomically (dotfile-manager setups keep working).  A
  dangling link whose target directory does not exist raises
  :class:`FileNotFoundError` instead of creating directories elsewhere.
"""

from __future__ import annotations

import contextlib
import os
import stat
import tempfile
from pathlib import Path

#: Directory holding every persisted LLmThoughtLens config file.
CONFIG_DIR = Path(os.environ.get("LLMTHOUGHTLENS_HOME", "~/.LLmThoughtLens")).expanduser()
#: TUI preferences + session history.
CONFIG_PATH = CONFIG_DIR / "config.json"
#: Dashboard / server provider settings (API keys live here).
SERVER_CONFIG_PATH = CONFIG_DIR / "server.json"

#: Owner-only permissions for the config dir and the files inside it.
PRIVATE_DIR_MODE = 0o700
PRIVATE_FILE_MODE = 0o600


def _never_tighten(path: Path) -> bool:
    """True for directories we must not chmod even when asked to tighten."""
    try:
        resolved = path.resolve()
    except OSError:  # pragma: no cover — unresolvable path; be conservative
        return True
    if resolved == Path(resolved.anchor):
        return True
    with contextlib.suppress(RuntimeError, KeyError, OSError):  # no resolvable $HOME
        if resolved == Path.home().resolve():
            return True
    return False


def ensure_private_dir(path: Path, *, tighten_existing: bool = False) -> None:
    """Make sure directory *path* exists; a newly created leaf is ``0o700``.

    Parameters
    ----------
    path:
        Directory to create (parents included) if missing.
    tighten_existing:
        When ``True`` and *path* already exists with group/other permission
        bits set, chmod it to ``0o700``.  Only meant for the LLmThoughtLens
        config home; the user's home directory, filesystem roots and
        directories owned by another user are never touched.  Failures to
        chmod are ignored (the files themselves are still ``0o600``).
    """
    if path.is_dir():
        if tighten_existing:
            _tighten_dir(path)
        return
    path.mkdir(mode=PRIVATE_DIR_MODE, parents=True, exist_ok=True)
    with contextlib.suppress(OSError):  # umask-proof; a no-op on Windows
        os.chmod(path, PRIVATE_DIR_MODE)


def _tighten_dir(path: Path) -> None:
    if os.name == "nt" or _never_tighten(path):
        return
    with contextlib.suppress(OSError):
        st = path.stat()
        getuid = getattr(os, "getuid", None)
        if getuid is not None and st.st_uid != getuid():
            return
        if stat.S_IMODE(st.st_mode) & 0o077:
            os.chmod(path, PRIVATE_DIR_MODE)


def _write_target(path: Path) -> Path:
    """Resolve the file :func:`write_private_text` should actually replace."""
    if not path.is_symlink():
        return path
    target = Path(os.path.realpath(path))
    if not target.parent.is_dir():
        raise FileNotFoundError(
            f"{path} is a symlink to {target}, whose directory does not exist; "
            "fix or remove the link (LLmThoughtLens will not create directories "
            "outside the config home)."
        )
    return target


def write_private_text(path: Path, text: str, *, tighten_dir: bool = False) -> None:
    """Atomically write *text* to *path* with owner-only (``0o600``) permissions.

    The content goes to a temp file in the destination directory (created
    ``0o600`` by :func:`tempfile.mkstemp`), is flushed and fsynced, then moved
    over the destination with :func:`os.replace`.  Readers therefore never
    observe a partial file, a crash leaves the previous version intact, and a
    pre-existing file with looser permissions is replaced by a ``0o600`` one.
    The temp file is removed on any failure.

    If *path* is a symlink, its final target is replaced and the link itself
    is left in place (see the module docstring).

    Parameters
    ----------
    path:
        Destination file.
    text:
        UTF-8 text to write.
    tighten_dir:
        Also tighten an existing parent directory to ``0o700`` (see
        :func:`ensure_private_dir`).  Pass ``True`` only when *path* lives
        directly in the LLmThoughtLens config home.
    """
    ensure_private_dir(path.parent, tighten_existing=tighten_dir)
    target = _write_target(path)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        with contextlib.suppress(OSError):  # mkstemp is already 0o600; Windows ignores modes
            os.chmod(tmp_name, PRIVATE_FILE_MODE)
        os.replace(tmp_name, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


__all__ = [
    "CONFIG_DIR",
    "CONFIG_PATH",
    "PRIVATE_DIR_MODE",
    "PRIVATE_FILE_MODE",
    "SERVER_CONFIG_PATH",
    "ensure_private_dir",
    "write_private_text",
]
