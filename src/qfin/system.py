"""Runtime capability reporting without exposing implementation details in normal APIs."""

from __future__ import annotations

from importlib.util import find_spec
from typing import TypedDict

from qfin import _native
from qfin.backends.devices import available_tested_devices


class SystemInfo(TypedDict):
    qfin_version: str
    native_extension: bool
    native_backend: str | None
    native_cpp_standard: str | None
    native_compiler: str | None
    pennylane: bool
    pennylane_lightning: bool
    qiskit: bool
    preferred_quantum_device: str | None
    tested_quantum_devices: tuple[str, ...]
    noise_simulator: str | None
    openqasm_export: bool
    factorized_state_preparation: bool
    structured_arithmetic_oracles: bool
    factorized_tail_risk: bool
    portfolio_optimization: str
    block_encoding_implemented: bool
    qsvt_implemented: bool


def system_info() -> SystemInfo:
    """Return installed classical and quantum execution capabilities."""

    from qfin import __version__

    native = _native.available()
    extension = _native.require() if native else None
    pennylane = find_spec("pennylane") is not None
    lightning = find_spec("pennylane_lightning") is not None
    qiskit = find_spec("qiskit") is not None
    preferred = "lightning.qubit" if lightning else ("default.qubit" if pennylane else None)
    devices = available_tested_devices()
    return {
        "qfin_version": __version__,
        "native_extension": native,
        "native_backend": None if extension is None else str(extension.backend_name),
        "native_cpp_standard": None if extension is None else str(extension.cpp_standard),
        "native_compiler": None if extension is None else str(extension.compiler),
        "pennylane": pennylane,
        "pennylane_lightning": lightning,
        "qiskit": qiskit,
        "preferred_quantum_device": preferred,
        "tested_quantum_devices": devices,
        "noise_simulator": "default.mixed" if pennylane else None,
        "openqasm_export": pennylane,
        "factorized_state_preparation": True,
        "structured_arithmetic_oracles": True,
        "factorized_tail_risk": True,
        "portfolio_optimization": "classical-scipy",
        "block_encoding_implemented": False,
        "qsvt_implemented": False,
    }


__all__ = ["SystemInfo", "system_info"]
