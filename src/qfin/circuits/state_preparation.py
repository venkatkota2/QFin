"""Structured probability-tree state preparation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import log2
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


def probability_tree_angles(
    probabilities: NDArray[np.float64],
) -> tuple[NDArray[np.float64], ...]:
    """Return conditional ``RY`` angles for a binary probability tree.

    At level ``l`` there are ``2**l`` angles, one for each prefix already
    encoded on the more-significant qubits. The rotations map ``|0...0>`` to
    amplitudes ``sqrt(probabilities)`` without constructing a dense unitary.
    """

    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if values.size < 2 or values.size & (values.size - 1):
        raise ValueError("probabilities must contain a power-of-two number of values")
    if np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError("probabilities must be finite and non-negative")
    total = float(np.sum(values))
    if total <= 0:
        raise ValueError("probabilities must have positive total mass")
    values = values / total
    qubits = int(log2(values.size))
    levels: list[NDArray[np.float64]] = []

    for level in range(qubits):
        block_size = 2 ** (qubits - level)
        level_angles = np.zeros(2**level, dtype=np.float64)
        for prefix in range(2**level):
            start = prefix * block_size
            midpoint = start + block_size // 2
            end = start + block_size
            left_mass = float(np.sum(values[start:midpoint]))
            right_mass = float(np.sum(values[midpoint:end]))
            branch_mass = left_mass + right_mass
            if branch_mass > 0:
                level_angles[prefix] = 2.0 * np.arctan2(
                    np.sqrt(right_mass), np.sqrt(left_mass)
                )
        level_angles.setflags(write=False)
        levels.append(level_angles)
    return tuple(levels)


@dataclass(frozen=True, slots=True)
class ProbabilityTreePreparation:
    """Decomposable distribution loader made from multiplexed ``RY`` gates."""

    levels: tuple[NDArray[np.float64], ...]

    @classmethod
    def from_probabilities(
        cls, probabilities: NDArray[np.float64]
    ) -> ProbabilityTreePreparation:
        return cls(levels=probability_tree_angles(probabilities))

    @property
    def qubits(self) -> int:
        return len(self.levels)

    @property
    def rotation_count(self) -> int:
        return sum(level.size for level in self.levels)

    @property
    def parameter_count(self) -> int:
        return self.rotation_count

    def apply(self, wires: Sequence[int]) -> None:
        """Queue the probability-tree circuit on the active PennyLane tape."""
        wire_tuple = tuple(wires)
        if len(wire_tuple) != self.qubits:
            raise ValueError("one data wire is required per probability-tree level")
        qml = _qml()
        qml.RY(float(self.levels[0][0]), wires=wire_tuple[0])
        for level in range(1, self.qubits):
            qml.SelectPauliRot(
                self.levels[level],
                control_wires=wire_tuple[:level],
                target_wire=wire_tuple[level],
                rot_axis="Y",
            )


@dataclass(frozen=True, slots=True)
class UniformQuantilePreparation:
    """Parameter-free loader for an inverse-CDF quantile representation."""

    qubits: int

    def __post_init__(self) -> None:
        if self.qubits < 1:
            raise ValueError("qubits must be positive")

    @property
    def gate_count(self) -> int:
        return self.qubits

    @property
    def parameter_count(self) -> int:
        return 0

    def apply(self, wires: Sequence[int]) -> None:
        """Prepare the uniform superposition labeling midpoint quantiles."""
        wire_tuple = tuple(wires)
        if len(wire_tuple) != self.qubits:
            raise ValueError("one data wire is required per quantile bit")
        qml = _qml()
        for wire in wire_tuple:
            qml.Hadamard(wires=wire)
