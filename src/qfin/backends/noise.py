"""Repeatable synthetic noise and zero-noise extrapolation experiments."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from math import isfinite
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qfin.exceptions import BackendUnavailableError
from qfin.resources.device import CircuitRuntime


@dataclass(frozen=True, slots=True)
class NoiseModel:
    """Local channel assumptions for ``default.mixed`` research experiments."""

    depolarizing_probability: float
    readout_bit_flip_probability: float = 0.0
    label: str = "local_depolarizing_plus_readout"

    def __post_init__(self) -> None:
        for name, value in (
            ("depolarizing_probability", self.depolarizing_probability),
            ("readout_bit_flip_probability", self.readout_bit_flip_probability),
        ):
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and lie in [0, 1]")
        if not self.label.strip():
            raise ValueError("noise model label must not be empty")

    def scaled(self, factor: float) -> NoiseModel:
        """Scale explicit channel probabilities for sensitivity experiments."""

        if not isfinite(factor) or factor < 0:
            raise ValueError("noise scale factor must be finite and non-negative")
        return NoiseModel(
            depolarizing_probability=min(1.0, factor * self.depolarizing_probability),
            readout_bit_flip_probability=min(1.0, factor * self.readout_bit_flip_probability),
            label=f"{self.label}@{factor:g}x",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "depolarizing_probability": self.depolarizing_probability,
            "readout_bit_flip_probability": self.readout_bit_flip_probability,
            "semantics": (
                "A single-qubit depolarizing channel is inserted on every affected "
                "wire after each queued gate; bit-flip channels are inserted before readout."
            ),
        }


@dataclass(frozen=True, slots=True)
class NoiseMitigationReport:
    """Ideal, noisy, and zero-noise-extrapolated objective probabilities."""

    power: int
    noise_model: NoiseModel
    shots: int | None
    seed: int | None
    scale_factors: tuple[float, ...]
    scaled_probabilities: tuple[float, ...]
    ideal_probability: float
    noisy_probability: float
    extrapolated_probability: float
    mitigated_probability: float
    extrapolation_order: int
    noisy_absolute_error: float
    mitigated_absolute_error: float
    mitigation_improved: bool
    mitigation_clipped: bool
    simulator: str = "default.mixed"

    def to_dict(self) -> dict[str, object]:
        return {
            "power": self.power,
            "noise_model": self.noise_model.to_dict(),
            "shots": self.shots,
            "seed": self.seed,
            "scale_factors": list(self.scale_factors),
            "scaled_probabilities": list(self.scaled_probabilities),
            "ideal_probability": self.ideal_probability,
            "noisy_probability": self.noisy_probability,
            "extrapolated_probability": self.extrapolated_probability,
            "mitigated_probability": self.mitigated_probability,
            "extrapolation_order": self.extrapolation_order,
            "noisy_absolute_error": self.noisy_absolute_error,
            "mitigated_absolute_error": self.mitigated_absolute_error,
            "mitigation_improved": self.mitigation_improved,
            "mitigation_clipped": self.mitigation_clipped,
            "simulator": self.simulator,
            "caveat": (
                "Synthetic local-channel simulator experiment. Results are not a "
                "prediction for a named quantum processor or a mitigation guarantee."
            ),
        }


def _qml() -> Any:
    try:
        import pennylane as qml
    except ImportError as exc:
        raise BackendUnavailableError(
            "PennyLane is required for noise experiments. Install QFin with "
            "`python -m pip install -e '.[quantum]'`."
        ) from exc
    return qml


def _noisy_probability(
    runtime: CircuitRuntime,
    *,
    power: int,
    noise_model: NoiseModel,
    scale_factor: float,
    shots: int | None,
    seed: int | None,
) -> float:
    qml = _qml()
    device = qml.device(
        "default.mixed",
        wires=runtime.total_wires,
        seed=seed,
    )

    @qml.qnode(device)  # type: ignore[untyped-decorator]
    def expectation_circuit() -> Any:
        runtime.queue_circuit(power)
        return qml.expval(qml.PauliZ(wires=runtime.objective_wire))

    folded = qml.noise.fold_global(expectation_circuit, scale_factor=scale_factor)
    noisy = folded
    if noise_model.depolarizing_probability > 0:
        noisy = qml.noise.insert(
            noisy,
            qml.DepolarizingChannel,
            noise_model.depolarizing_probability,
            position="all",
        )
    if noise_model.readout_bit_flip_probability > 0:
        noisy = qml.noise.insert(
            noisy,
            qml.BitFlip,
            noise_model.readout_bit_flip_probability,
            position="end",
        )
    if shots is not None:
        noisy = qml.set_shots(noisy, shots=shots)
    expectation = float(noisy())
    return float(np.clip((1.0 - expectation) / 2.0, 0.0, 1.0))


def analyze_noise(
    runtime: CircuitRuntime,
    noise_model: NoiseModel,
    *,
    power: int = 0,
    shots: int | None = None,
    seed: int | None = 0,
    scale_factors: tuple[float, ...] = (1.0, 3.0, 5.0),
    extrapolation_order: int = 1,
) -> NoiseMitigationReport:
    """Run deterministic or shot-based folding and polynomial ZNE."""

    if power < 0:
        raise ValueError("power must be non-negative")
    if shots is not None and shots <= 0:
        raise ValueError("shots must be positive or None")
    if len(scale_factors) < 2:
        raise ValueError("at least two scale factors are required")
    if any(not isfinite(factor) or factor < 1 for factor in scale_factors):
        raise ValueError("scale factors must be finite and at least one")
    if any(right <= left for left, right in pairwise(scale_factors)):
        raise ValueError("scale factors must be strictly increasing")
    if not 1 <= extrapolation_order < len(scale_factors):
        raise ValueError("extrapolation_order must be between one and factors minus one")

    ideal_probability = float(runtime.probability(power=power, shots=None, seed=seed))
    values = tuple(
        _noisy_probability(
            runtime,
            power=power,
            noise_model=noise_model,
            scale_factor=factor,
            shots=shots,
            seed=None if seed is None else seed + index,
        )
        for index, factor in enumerate(scale_factors)
    )
    coefficients: NDArray[np.float64] = np.asarray(
        np.polynomial.polynomial.polyfit(
            np.asarray(scale_factors, dtype=np.float64),
            np.asarray(values, dtype=np.float64),
            extrapolation_order,
        ),
        dtype=np.float64,
    )
    extrapolated = float(coefficients[0])
    mitigated = float(np.clip(extrapolated, 0.0, 1.0))
    noisy_error = abs(values[0] - ideal_probability)
    mitigated_error = abs(mitigated - ideal_probability)
    return NoiseMitigationReport(
        power=power,
        noise_model=noise_model,
        shots=shots,
        seed=seed,
        scale_factors=scale_factors,
        scaled_probabilities=values,
        ideal_probability=ideal_probability,
        noisy_probability=values[0],
        extrapolated_probability=extrapolated,
        mitigated_probability=mitigated,
        extrapolation_order=extrapolation_order,
        noisy_absolute_error=noisy_error,
        mitigated_absolute_error=mitigated_error,
        mitigation_improved=mitigated_error < noisy_error,
        mitigation_clipped=mitigated != extrapolated,
    )


__all__ = ["NoiseMitigationReport", "NoiseModel", "analyze_noise"]
