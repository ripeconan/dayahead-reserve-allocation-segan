"""Fail-closed firewall for confirmatory and R77 scientific payloads."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterable

from .constants import ACCESS_FLAGS, FORBIDDEN_PATH_SUBSTRINGS


class FirewallViolation(RuntimeError):
    """Raised when a forbidden path or year is requested."""


def _normalize_path_text(path: str | os.PathLike[str]) -> str:
    text = str(path)
    # Accept both Windows and POSIX separators in substring checks.
    return text.replace("\\", "/").lower()


def path_is_forbidden(path: str | os.PathLike[str]) -> bool:
    normalized = _normalize_path_text(path)
    return any(token.lower() in normalized for token in FORBIDDEN_PATH_SUBSTRINGS)


def assert_path_allowed(path: str | os.PathLike[str], *, context: str = "") -> Path:
    """Fail closed before any open/hash/list of a forbidden path."""
    if path_is_forbidden(path):
        # Do not touch the path further; report only the request context.
        detail = f" context={context}" if context else ""
        raise FirewallViolation(
            f"FORBIDDEN_PATH_ACCESS_ATTEMPT path_request={path!s}{detail}"
        )
    return Path(path)


def assert_paths_allowed(paths: Iterable[str | os.PathLike[str]], *, context: str = "") -> None:
    for path in paths:
        assert_path_allowed(path, context=context)


def assert_no_year_2026(timestamps) -> None:
    """Fail closed if any timestamp falls in calendar year 2026 (local or UTC)."""
    import pandas as pd

    index = pd.DatetimeIndex(timestamps)
    if len(index) == 0:
        return
    years = index.year
    if (years == 2026).any():
        raise FirewallViolation("YEAR_2026_ROW_PRESENT fail-closed before scientific use")


def assert_frame_no_2026(frame) -> None:
    import pandas as pd

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("assert_frame_no_2026 expects a DataFrame")
    assert_no_year_2026(frame.index)
    for column in frame.columns:
        name = str(column).lower()
        if "2026" in name:
            raise FirewallViolation(f"YEAR_2026_COLUMN_PRESENT column={column}")


def access_flag_snapshot() -> dict[str, bool]:
    """Return a copy of access flags; all must remain false in R78."""
    return dict(ACCESS_FLAGS)


def assert_access_flags_clean(flags: dict[str, bool] | None = None) -> None:
    current = access_flag_snapshot() if flags is None else flags
    dirty = {k: v for k, v in current.items() if v is True and k != "sota_claimed"}
    # sota_claimed is also required false, include it.
    dirty = {k: v for k, v in current.items() if v is True}
    if dirty:
        raise FirewallViolation(f"ACCESS_FLAGS_DIRTY {dirty}")


def safe_resolve_existing(path: str | os.PathLike[str], *, context: str = "") -> Path:
    """Resolve an existing path only after firewall allow-list check."""
    allowed = assert_path_allowed(path, context=context)
    # Avoid resolving through forbidden parents when possible: check as_posix first.
    resolved = allowed.expanduser()
    assert_path_allowed(resolved, context=context + ":expanduser")
    if resolved.exists():
        resolved = resolved.resolve()
        assert_path_allowed(resolved, context=context + ":resolve")
    return resolved


def path_mentions_r77_d2(path: str | os.PathLike[str]) -> bool:
    return "gefs_revision_r77_d2" in _normalize_path_text(path) or (
        "r77" in _normalize_path_text(path) and "d2" in _normalize_path_text(path)
    )


def pure_path_parts_look_like_2026_cohort(path: str | os.PathLike[str]) -> bool:
    text = _normalize_path_text(path)
    return "2026h1" in text or "nyiso_2026" in text
