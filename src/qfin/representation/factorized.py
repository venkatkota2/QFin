"""Factorized financial representations that avoid joint probability tables."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from qfin.finance.distributions import Distribution, Normal
from qfin.finance.factors import GaussianFactorModel
from qfin.representation.encoding import DistributionEncoding, encode, encode_quantiles

FloatArray = NDArray[np.float64]
FactorEncodingMethod = Literal["quantile", "probability"]


@dataclass(frozen=True, slots=True)
class LinearFactorTransform:
    """Classical interpretation of independent latent quantum registers.

    The transform is metadata used to interpret basis-state values. QFin 0.8
    does not claim to implement the affine map as reversible quantum
    arithmetic.
    """

    matrix: FloatArray
    offset: FloatArray
    output_names: Sequence[str]

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.float64)
        offset = np.asarray(self.offset, dtype=np.float64).reshape(-1)
        names = tuple(self.output_names)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("matrix must be a non-empty output-by-latent array")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("matrix must be finite")
        if offset.shape != (matrix.shape[0],) or not np.all(np.isfinite(offset)):
            raise ValueError("offset must contain one finite value per output")
        if len(names) != matrix.shape[0] or not all(names):
            raise ValueError("output_names must contain one non-empty name per output")
        if len(set(names)) != len(names):
            raise ValueError("output_names must be unique")
        matrix = np.ascontiguousarray(matrix)
        offset = np.ascontiguousarray(offset)
        matrix.setflags(write=False)
        offset.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "output_names", names)

    @property
    def latent_factors(self) -> int:
        return int(self.matrix.shape[1])

    @property
    def outputs(self) -> int:
        return int(self.matrix.shape[0])

    def apply(self, latent_values: FloatArray) -> FloatArray:
        values = np.asarray(latent_values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.latent_factors:
            raise ValueError("latent_values must be point-by-latent-factor")
        return np.asarray(values @ self.matrix.T + self.offset, dtype=np.float64)

    def to_dict(self) -> dict[str, object]:
        return {
            "latent_factors": self.latent_factors,
            "outputs": self.outputs,
            "output_names": list(self.output_names),
            "quantum_arithmetic_implemented": False,
            "interpretation": "classical affine interpretation of latent basis states",
        }


@dataclass(frozen=True, slots=True)
class MaterializedFactorGrid:
    """Small validation-only Cartesian product of a factorized encoding."""

    values: FloatArray
    probabilities: FloatArray
    value_names: Sequence[str]

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        probabilities = np.asarray(self.probabilities, dtype=np.float64).reshape(-1)
        names = tuple(self.value_names)
        if values.ndim != 2 or values.shape[0] != probabilities.size:
            raise ValueError("values and probabilities must contain the same points")
        if values.shape[1] != len(names):
            raise ValueError("value_names must contain one name per value column")
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(probabilities)):
            raise ValueError("materialized values and probabilities must be finite")
        if np.any(probabilities < 0) or not np.isclose(np.sum(probabilities), 1.0):
            raise ValueError("probabilities must be non-negative and sum to one")
        values = np.ascontiguousarray(values)
        probabilities = np.ascontiguousarray(probabilities)
        values.setflags(write=False)
        probabilities.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "value_names", names)


@dataclass(frozen=True, slots=True)
class FactorizedDistributionEncoding:
    """Independent latent registers stored without a full Cartesian product."""

    factors: Sequence[DistributionEncoding]
    factor_names: Sequence[str]
    dependence_assumption: str = "independent factors"
    transform: LinearFactorTransform | None = None

    def __post_init__(self) -> None:
        factors = tuple(self.factors)
        names = tuple(self.factor_names)
        if not factors:
            raise ValueError("at least one factor encoding is required")
        if len(names) != len(factors) or not all(names):
            raise ValueError("factor_names must contain one non-empty name per factor")
        if len(set(names)) != len(names):
            raise ValueError("factor_names must be unique")
        if not self.dependence_assumption.strip():
            raise ValueError("dependence_assumption must not be empty")
        if self.transform is not None and self.transform.latent_factors != len(factors):
            raise ValueError("transform must consume one column per latent factor")
        object.__setattr__(self, "factors", factors)
        object.__setattr__(self, "factor_names", names)

    @property
    def factor_count(self) -> int:
        return len(self.factors)

    @property
    def total_qubits(self) -> int:
        return sum(factor.qubits for factor in self.factors)

    @property
    def qubits_per_factor(self) -> tuple[int, ...]:
        return tuple(factor.qubits for factor in self.factors)

    @property
    def joint_grid_points(self) -> int:
        return int(2**self.total_qubits)

    @property
    def stored_marginal_points(self) -> int:
        return sum(factor.grid_points for factor in self.factors)

    @property
    def avoids_joint_probability_table(self) -> bool:
        return True

    @property
    def value_names(self) -> tuple[str, ...]:
        if self.transform is None:
            return tuple(self.factor_names)
        return tuple(self.transform.output_names)

    def materialize(self, *, max_points: int = 65_536) -> MaterializedFactorGrid:
        """Materialize a small Cartesian product only for validation.

        Production loaders consume the marginal encodings directly. The guard
        prevents an accidental exponential allocation.
        """

        if max_points < 1:
            raise ValueError("max_points must be positive")
        if self.joint_grid_points > max_points:
            raise ValueError(
                f"joint grid has {self.joint_grid_points} points, above max_points={max_points}; "
                "use marginal metadata or increase the validation limit explicitly"
            )
        value_meshes = np.meshgrid(
            *(factor.grid for factor in self.factors),
            indexing="ij",
        )
        latent_values = np.column_stack([mesh.reshape(-1) for mesh in value_meshes])
        probability_meshes = np.meshgrid(
            *(factor.probabilities for factor in self.factors),
            indexing="ij",
        )
        probabilities = np.ones(self.joint_grid_points, dtype=np.float64)
        for mesh in probability_meshes:
            probabilities *= mesh.reshape(-1)
        values = latent_values if self.transform is None else self.transform.apply(latent_values)
        return MaterializedFactorGrid(values, probabilities, self.value_names)

    def expectation(
        self,
        objective: Callable[[FloatArray], FloatArray],
        *,
        max_points: int = 65_536,
    ) -> float:
        grid = self.materialize(max_points=max_points)
        values = np.asarray(objective(grid.values), dtype=np.float64).reshape(-1)
        if values.shape != grid.probabilities.shape or not np.all(np.isfinite(values)):
            raise ValueError("objective must return one finite value per materialized point")
        return float(np.dot(grid.probabilities, values))

    def to_dict(self) -> dict[str, object]:
        return {
            "encoding_method": "factorized_marginal_amplitudes",
            "state_preparation_method": "factorized_marginal_loader",
            "factor_names": list(self.factor_names),
            "factor_count": self.factor_count,
            "qubits_per_factor": list(self.qubits_per_factor),
            "total_qubits": self.total_qubits,
            "joint_grid_points": self.joint_grid_points,
            "stored_marginal_points": self.stored_marginal_points,
            "dependence_assumption": self.dependence_assumption,
            "avoids_joint_probability_table": True,
            "transform": None if self.transform is None else self.transform.to_dict(),
            "caveat": (
                "The loader prepares independent latent registers. Any listed affine "
                "transform is a classical basis-state interpretation; reversible quantum "
                "arithmetic for that transform is not implemented."
            ),
        }


def _qubit_allocation(
    factor_count: int,
    qubits_per_factor: int | Sequence[int],
) -> tuple[int, ...]:
    if isinstance(qubits_per_factor, int):
        allocation = (qubits_per_factor,) * factor_count
    else:
        allocation = tuple(int(value) for value in qubits_per_factor)
    if len(allocation) != factor_count or any(value < 1 for value in allocation):
        raise ValueError("qubits_per_factor must provide one positive value per factor")
    return allocation


def encode_independent_factors(
    distributions: Sequence[Distribution],
    *,
    qubits_per_factor: int | Sequence[int] = 3,
    factor_names: Sequence[str] | None = None,
    method: FactorEncodingMethod = "quantile",
    target_error: float = 1e-3,
    tail_probability: float = 1e-6,
    transform: LinearFactorTransform | None = None,
    dependence_assumption: str = "independent factors",
) -> FactorizedDistributionEncoding:
    """Encode marginal factors without constructing their joint table."""

    factors = tuple(distributions)
    if not factors:
        raise ValueError("distributions must not be empty")
    if not isfinite(target_error) or target_error <= 0:
        raise ValueError("target_error must be finite and positive")
    if method not in ("quantile", "probability"):
        raise ValueError("method must be 'quantile' or 'probability'")
    allocation = _qubit_allocation(len(factors), qubits_per_factor)
    names = (
        tuple(f"factor_{index}" for index in range(len(factors)))
        if factor_names is None
        else tuple(factor_names)
    )
    encoder = encode_quantiles if method == "quantile" else encode
    encodings = tuple(
        encoder(
            distribution,
            target_error=target_error,
            qubits=qubits,
            min_qubits=qubits,
            max_qubits=qubits,
            tail_probability=tail_probability,
        )
        for distribution, qubits in zip(factors, allocation, strict=True)
    )
    return FactorizedDistributionEncoding(
        factors=encodings,
        factor_names=names,
        dependence_assumption=dependence_assumption,
        transform=transform,
    )


def encode_gaussian_factors(
    model: GaussianFactorModel,
    *,
    qubits_per_factor: int | Sequence[int] = 3,
    target_error: float = 1e-3,
    tail_probability: float = 1e-6,
) -> FactorizedDistributionEncoding:
    """Encode independent Gaussian drivers plus a classical correlation map."""

    eigenvalues, eigenvectors = np.linalg.eigh(model.correlation)
    correlation_root = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))
    assert model.means is not None
    assert model.standard_deviations is not None
    transform = LinearFactorTransform(
        matrix=np.diag(model.standard_deviations) @ correlation_root,
        offset=model.means,
        output_names=model.factor_names,
    )
    latent_names = tuple(f"latent_{index}" for index in range(model.factor_count))
    return encode_independent_factors(
        tuple(Normal(0.0, 1.0) for _ in range(model.factor_count)),
        qubits_per_factor=qubits_per_factor,
        factor_names=latent_names,
        method="quantile",
        target_error=target_error,
        tail_probability=tail_probability,
        transform=transform,
        dependence_assumption=(
            "independent latent Gaussian registers with a classical affine correlation map"
        ),
    )


__all__ = [
    "FactorEncodingMethod",
    "FactorizedDistributionEncoding",
    "LinearFactorTransform",
    "MaterializedFactorGrid",
    "encode_gaussian_factors",
    "encode_independent_factors",
]
