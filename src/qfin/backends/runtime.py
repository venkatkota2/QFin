"""Shared PennyLane runtime helpers.

The financial and compiler layers deliberately do not import PennyLane.  This
module is the single optional-runtime boundary used by all simulator adapters.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from qfin.exceptions import BackendUnavailableError

ComplexPrecision = Literal["complex64", "complex128"]
DEFAULT_DEVICE = "lightning.qubit"


def load_pennylane() -> Any:
    """Import PennyLane only when a quantum circuit is requested."""
    try:
        import pennylane as qml
    except ImportError as exc:
        raise BackendUnavailableError(
            "PennyLane is required to execute quantum circuits. "
            "Install QFin with `python -m pip install -e '.[quantum]'`."
        ) from exc
    return qml


def create_device(
    qml: Any,
    device_name: str,
    *,
    wires: int,
    seed: int | None,
    precision: ComplexPrecision,
) -> tuple[Any, str]:
    """Create one device for a complete circuit batch.

    ``auto`` prefers the compiled C++ Lightning simulator and falls back to
    PennyLane's portable Python simulator.  An explicit device name never
    falls back silently.  Lightning alone receives ``c_dtype`` because that is
    a plugin-specific option.
    """
    if wires < 1:
        raise ValueError("wires must be positive")
    if precision not in ("complex64", "complex128"):
        raise ValueError("precision must be 'complex64' or 'complex128'")
    candidates = (DEFAULT_DEVICE, "default.qubit") if device_name == "auto" else (device_name,)
    last_error: Exception | None = None
    for candidate in candidates:
        kwargs: dict[str, object] = {"wires": wires, "seed": seed}
        if candidate.startswith("lightning."):
            kwargs["c_dtype"] = (
                np.complex64 if precision == "complex64" else np.complex128
            )
        try:
            return qml.device(candidate, **kwargs), candidate
        except Exception as exc:  # PennyLane plugins expose different error classes.
            last_error = exc
            if device_name != "auto":
                break
    message = f"PennyLane device {device_name!r} is unavailable"
    if device_name == DEFAULT_DEVICE:
        message += "; install QFin with the 'quantum' extra for PennyLane Lightning"
    raise BackendUnavailableError(message) from last_error


def execute_probability_circuits(
    circuits: Sequence[Any],
) -> tuple[NDArray[np.float64], ...]:
    """Execute several QNodes on their shared, already-created device.

    PennyLane's generic tape-batch path has a substantial cold-start cost for
    these short, heterogeneous Grover circuits.  Reusing one Lightning device
    while invoking the QNodes sequentially is faster in measured QFin runs and
    still avoids repeated plugin/device initialization.
    """
    return tuple(np.asarray(circuit(), dtype=np.float64) for circuit in circuits)
