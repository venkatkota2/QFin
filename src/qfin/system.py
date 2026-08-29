"""Runtime capability reporting without exposing implementation details in normal APIs."""

from __future__ import annotations

from importlib.util import find_spec
from typing import TypedDict

from qfin import _native


class SystemInfo(TypedDict):
    qfin_version: str
    native_extension: bool
    native_backend: str | None
    native_cpp_standard: str | None
    native_compiler: str | None
    pennylane: bool
    pennylane_lightning: bool
    preferred_quantum_device: str | None


def system_info() -> SystemInfo:
    """Return installed classical and quantum execution capabilities."""

    from qfin import __version__

    native = _native.available()
    extension = _native.require() if native else None
    pennylane = find_spec("pennylane") is not None
    lightning = find_spec("pennylane_lightning") is not None
    preferred = "lightning.qubit" if lightning else ("default.qubit" if pennylane else None)
    return {
        "qfin_version": __version__,
        "native_extension": native,
        "native_backend": None if extension is None else str(extension.backend_name),
        "native_cpp_standard": None if extension is None else str(extension.cpp_standard),
        "native_compiler": None if extension is None else str(extension.compiler),
        "pennylane": pennylane,
        "pennylane_lightning": lightning,
        "preferred_quantum_device": preferred,
    }


__all__ = ["SystemInfo", "system_info"]
