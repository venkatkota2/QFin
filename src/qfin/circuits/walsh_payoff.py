"""Sparse Walsh/Pauli payoff approximation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite, log2
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


def _fast_walsh_hadamard(values: NDArray[np.float64]) -> NDArray[np.float64]:
    transformed = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    if transformed.size < 2 or transformed.size & (transformed.size - 1):
        raise ValueError("Walsh transform input must have power-of-two length")
    width = 1
    while width < transformed.size:
        blocks = transformed.reshape(-1, 2 * width)
        left = blocks[:, :width].copy()
        right = blocks[:, width:].copy()
        blocks[:, :width] = left + right
        blocks[:, width:] = left - right
        width *= 2
    return transformed


def _walsh_character(mask: int, points: int) -> NDArray[np.float64]:
    indices = np.arange(points, dtype=np.uint64)
    character = np.ones(points, dtype=np.float64)
    bit = 0
    remaining = mask
    while remaining:
        if remaining & 1:
            character[indices & np.uint64(1 << bit) != 0] *= -1.0
        remaining >>= 1
        bit += 1
    return character


@dataclass(frozen=True, slots=True)
class WalshTerm:
    """One coefficient multiplying a parity character of the data bits."""

    mask: int
    coefficient: float

    def __post_init__(self) -> None:
        if self.mask < 0:
            raise ValueError("Walsh mask must be non-negative")
        if not isfinite(self.coefficient):
            raise ValueError("Walsh coefficient must be finite")


@dataclass(frozen=True, slots=True)
class WalshPayoffApproximation:
    """Tolerance-controlled objective rotation represented by Pauli strings.

    A term with mask ``s`` applies ``exp(-i c_s Z_s Y / 2)``. On a data basis
    state ``|x>`` this is an objective-qubit ``RY`` rotation by
    ``c_s (-1)**popcount(s & x)``. The commuting terms therefore synthesize a
    sparse Walsh approximation to ``2 asin(sqrt(payoff(x)))`` without a
    uniformly controlled table of one angle per grid point.
    """

    qubits: int
    terms: tuple[WalshTerm, ...]
    full_term_count: int
    target_price_error: float
    price_error: float
    angle_rmse: float
    max_angle_error: float
    payoff_rmse: float
    max_payoff_error: float
    exact_amplitude: float
    approximate_amplitude: float
    met_tolerance: bool

    @classmethod
    def fit(
        cls,
        normalized_payoff: NDArray[np.float64],
        *,
        financial_multiplier: float,
        target_price_error: float,
        max_angle_rmse: float = 0.1,
        max_terms: int | None = None,
    ) -> WalshPayoffApproximation:
        """Fit the smallest magnitude-ordered expansion meeting both errors."""
        payoff = np.asarray(normalized_payoff, dtype=np.float64).reshape(-1)
        if payoff.size < 2 or payoff.size & (payoff.size - 1):
            raise ValueError("normalized_payoff must have power-of-two length")
        if np.any((payoff < 0) | (payoff > 1)) or not np.all(np.isfinite(payoff)):
            raise ValueError("normalized_payoff values must lie in [0, 1]")
        if not isfinite(financial_multiplier) or financial_multiplier < 0:
            raise ValueError("financial_multiplier must be finite and non-negative")
        if not isfinite(target_price_error) or target_price_error <= 0:
            raise ValueError("target_price_error must be finite and positive")
        if not isfinite(max_angle_rmse) or max_angle_rmse <= 0:
            raise ValueError("max_angle_rmse must be finite and positive")
        limit = payoff.size if max_terms is None else max_terms
        if not 1 <= limit <= payoff.size:
            raise ValueError("max_terms must lie between one and the payoff size")

        target_angles = 2.0 * np.arcsin(np.sqrt(payoff))
        coefficients = _fast_walsh_hadamard(target_angles) / payoff.size
        order = np.argsort(-np.abs(coefficients), kind="stable")
        approximate_angles = np.zeros_like(target_angles)
        exact_amplitude = float(np.mean(payoff))
        terms: list[WalshTerm] = []

        def diagnostics() -> tuple[float, float, float, float, float, float]:
            approximate_payoff = np.sin(approximate_angles / 2.0) ** 2
            approximate_amplitude = float(np.mean(approximate_payoff))
            price_error = financial_multiplier * abs(
                approximate_amplitude - exact_amplitude
            )
            angle_difference = approximate_angles - target_angles
            payoff_difference = approximate_payoff - payoff
            return (
                approximate_amplitude,
                price_error,
                float(np.sqrt(np.mean(angle_difference**2))),
                float(np.max(np.abs(angle_difference))),
                float(np.sqrt(np.mean(payoff_difference**2))),
                float(np.max(np.abs(payoff_difference))),
            )

        metrics = diagnostics()
        met_tolerance = metrics[1] <= target_price_error and metrics[2] <= max_angle_rmse
        for mask_value in order[:limit]:
            if met_tolerance:
                break
            mask = int(mask_value)
            coefficient = float(coefficients[mask])
            if abs(coefficient) <= 1e-15:
                continue
            terms.append(WalshTerm(mask=mask, coefficient=coefficient))
            approximate_angles += coefficient * _walsh_character(mask, payoff.size)
            metrics = diagnostics()
            met_tolerance = (
                metrics[1] <= target_price_error and metrics[2] <= max_angle_rmse
            )

        return cls(
            qubits=int(log2(payoff.size)),
            terms=tuple(terms),
            full_term_count=payoff.size,
            target_price_error=target_price_error,
            price_error=metrics[1],
            angle_rmse=metrics[2],
            max_angle_error=metrics[3],
            payoff_rmse=metrics[4],
            max_payoff_error=metrics[5],
            exact_amplitude=exact_amplitude,
            approximate_amplitude=metrics[0],
            met_tolerance=met_tolerance,
        )

    @property
    def parameter_count(self) -> int:
        return len(self.terms)

    @property
    def rotation_count(self) -> int:
        return self.parameter_count

    @property
    def compression_ratio(self) -> float:
        return self.parameter_count / self.full_term_count

    def to_dict(self) -> dict[str, float | int | bool]:
        """Return summary diagnostics without expanding the coefficient table."""
        return {
            "retained_terms": self.parameter_count,
            "full_term_count": self.full_term_count,
            "compression_ratio": self.compression_ratio,
            "target_price_error": self.target_price_error,
            "price_error": self.price_error,
            "angle_rmse": self.angle_rmse,
            "max_angle_error": self.max_angle_error,
            "payoff_rmse": self.payoff_rmse,
            "max_payoff_error": self.max_payoff_error,
            "exact_amplitude": self.exact_amplitude,
            "approximate_amplitude": self.approximate_amplitude,
            "met_tolerance": self.met_tolerance,
        }

    def approximate_angles(self) -> NDArray[np.float64]:
        """Reconstruct the rotation function for diagnostics and tests."""
        angles = np.zeros(self.full_term_count, dtype=np.float64)
        for term in self.terms:
            angles += term.coefficient * _walsh_character(
                term.mask, self.full_term_count
            )
        return angles

    def apply(self, control_wires: Sequence[int], target_wire: int) -> None:
        """Queue the sparse commuting Pauli rotations on an active tape."""
        controls = tuple(control_wires)
        if len(controls) != self.qubits:
            raise ValueError("one control wire is required per Walsh input bit")
        if target_wire in controls:
            raise ValueError("target_wire must be outside the data register")
        qml = _qml()
        for term in self.terms:
            if term.mask == 0:
                qml.RY(term.coefficient, wires=target_wire)
                continue
            selected = tuple(
                controls[self.qubits - 1 - bit]
                for bit in range(self.qubits)
                if term.mask & (1 << bit)
            )
            qml.PauliRot(
                term.coefficient,
                pauli_word="Z" * len(selected) + "Y",
                wires=(*selected, target_wire),
            )
