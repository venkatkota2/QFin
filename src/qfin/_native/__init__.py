"""Private loader for QFin's optional C++20 financial extension."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

from qfin.exceptions import NativeBackendUnavailableError

_extension: ModuleType | None
_load_error: BaseException | None

try:
    _extension = import_module("qfin._qfin_native")
    _load_error = None
except (ImportError, OSError) as exc:  # pragma: no cover - exercised in source-only installs
    _extension = None
    _load_error = exc


def available() -> bool:
    """Return whether the compiled QFin financial extension loaded."""

    return _extension is not None


def load_error() -> BaseException | None:
    """Return the extension import error, if any."""

    return _load_error


def require() -> ModuleType:
    """Return the extension or raise a user-facing QFin exception."""

    if _extension is None:
        detail = "" if _load_error is None else f" ({_load_error})"
        raise NativeBackendUnavailableError(
            "QFin's native C++ extension is unavailable. Reinstall a compatible wheel "
            "or use engine='numpy'." + detail
        )
    return _extension


__all__ = ["available", "load_error", "require"]
