"""Gate-decomposable payoff loading."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

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


@dataclass(frozen=True, slots=True)
class PayoffRotation:
    """Multiplexed objective-qubit rotations for a normalized payoff."""

    angles: NDArray[np.float64]

    @classmethod
    def from_normalized_payoff(
        cls, normalized_payoff: NDArray[np.float64]
    ) -> PayoffRotation:
        payoff = np.asarray(normalized_payoff, dtype=np.float64).reshape(-1)
        if payoff.size < 2 or payoff.size & (payoff.size - 1):
            raise ValueError("normalized_payoff must have power-of-two length")
        if np.any((payoff < 0) | (payoff > 1)) or not np.all(np.isfinite(payoff)):
            raise ValueError("normalized_payoff values must lie in [0, 1]")
        angles = 2.0 * np.arcsin(np.sqrt(payoff))
        angles.setflags(write=False)
        return cls(angles=angles)

    @property
    def rotation_count(self) -> int:
        return int(self.angles.size)

    @property
    def parameter_count(self) -> int:
        return self.rotation_count

    def apply(self, control_wires: Sequence[int], target_wire: int) -> None:
        """Queue the payoff multiplexer on the active PennyLane tape."""
        controls = tuple(control_wires)
        if 2 ** len(controls) != self.angles.size:
            raise ValueError("control wires do not address every payoff angle")
        qml = _qml()
        qml.SelectPauliRot(
            self.angles,
            control_wires=controls,
            target_wire=target_wire,
            rot_axis="Y",
        )

