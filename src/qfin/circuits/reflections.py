"""Gate-level reflections used by amplitude estimation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from qfin.exceptions import BackendUnavailableError


def _qml() -> Any:
    try:
        import pennylane as qml
    except ImportError as exc:
        raise BackendUnavailableError(
            "PennyLane is required to construct quantum circuits. "
            "Install QFin with `python -m pip install -e '.[quantum]'`."
        ) from exc
    return qml


def apply_zero_reflection(register_wires: Sequence[int], *, work_wire: int) -> None:
    """Flip the phase of ``|0...0>`` using X, H, and multi-controlled X gates."""
    register = tuple(register_wires)
    if len(register) < 2:
        raise ValueError("zero reflection requires at least two register wires")
    if work_wire in register:
        raise ValueError("work_wire must be outside the reflected register")
    qml = _qml()
    for wire in register:
        qml.PauliX(wires=wire)
    target = register[-1]
    qml.Hadamard(wires=target)
    qml.MultiControlledX(
        wires=register,
        work_wires=(work_wire,),
        work_wire_type="borrowed",
    )
    qml.Hadamard(wires=target)
    for wire in register:
        qml.PauliX(wires=wire)


def zero_reflection_operation_count(register_qubits: int) -> int:
    """Return high-level operations queued by ``apply_zero_reflection``."""
    if register_qubits < 2:
        raise ValueError("register_qubits must be at least two")
    return 2 * register_qubits + 3

