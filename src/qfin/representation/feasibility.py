"""Feasibility metadata for future block-encoding and QSVT research."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True, slots=True)
class BlockEncodingFeasibility:
    """Mathematical and implementation status for one explicit matrix."""

    rows: int
    columns: int
    square: bool
    hermitian: bool
    positive_semidefinite: bool
    operator_norm: float
    normalization_factor: float
    normalized_operator_norm: float
    condition_number: float
    nonzero_entries: int
    maximum_nonzeros_per_row: int
    density: float
    padded_dimension: int
    data_qubits: int
    classical_storage_bytes: int
    mathematical_qsvt_candidate: bool
    qfin_block_encoding_implemented: bool = False
    qfin_qsvt_implemented: bool = False
    implementation_reasons: tuple[str, ...] = (
        "no tested block-encoding oracle is implemented",
        "no polynomial synthesis or QSVT execution path is implemented",
    )

    def to_dict(self) -> dict[str, object]:
        return {
            "rows": self.rows,
            "columns": self.columns,
            "square": self.square,
            "hermitian": self.hermitian,
            "positive_semidefinite": self.positive_semidefinite,
            "operator_norm": self.operator_norm,
            "normalization_factor": self.normalization_factor,
            "normalized_operator_norm": self.normalized_operator_norm,
            "condition_number": self.condition_number,
            "nonzero_entries": self.nonzero_entries,
            "maximum_nonzeros_per_row": self.maximum_nonzeros_per_row,
            "density": self.density,
            "padded_dimension": self.padded_dimension,
            "data_qubits": self.data_qubits,
            "classical_storage_bytes": self.classical_storage_bytes,
            "mathematical_qsvt_candidate": self.mathematical_qsvt_candidate,
            "qfin_block_encoding_implemented": self.qfin_block_encoding_implemented,
            "qfin_qsvt_implemented": self.qfin_qsvt_implemented,
            "implementation_reasons": list(self.implementation_reasons),
            "caveat": (
                "A matrix satisfying mathematical preconditions is not an efficient oracle. "
                "This report does not claim a block encoding, QSVT circuit, or speedup."
            ),
        }


def analyze_block_encoding(
    matrix: ArrayLike,
    *,
    tolerance: float = 1e-12,
    max_dimension: int = 2_048,
) -> BlockEncodingFeasibility:
    """Inspect an explicit matrix without constructing a quantum circuit."""

    values = np.asarray(matrix, dtype=np.complex128)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("matrix must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("matrix must be finite")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    rows, columns = (int(value) for value in values.shape)
    if max(rows, columns) > max_dimension:
        raise ValueError(
            f"explicit analysis is limited to dimension {max_dimension}; "
            "provide structural oracle metadata for larger problems"
        )

    square = rows == columns
    hermitian = square and bool(np.allclose(values, values.conj().T, atol=tolerance, rtol=0.0))
    singular_values = np.linalg.svd(values, compute_uv=False)
    operator_norm = float(singular_values[0])
    positive_singular_values = singular_values[singular_values > tolerance]
    condition_number = (
        float("inf")
        if positive_singular_values.size < min(rows, columns)
        else float(positive_singular_values[0] / positive_singular_values[-1])
    )
    positive_semidefinite = False
    if hermitian:
        positive_semidefinite = bool(np.min(np.linalg.eigvalsh(values).real) >= -tolerance)
    nonzero = np.abs(values) > tolerance
    nonzero_entries = int(np.count_nonzero(nonzero))
    maximum_nonzeros = int(np.max(np.count_nonzero(nonzero, axis=1)))
    padded_dimension = 1 << ceil(log2(max(rows, columns))) if max(rows, columns) > 1 else 1
    data_qubits = int(log2(padded_dimension))
    normalization = max(operator_norm, 1.0)
    normalized_norm = operator_norm / normalization
    reasons = [
        "no tested block-encoding oracle is implemented",
        "no polynomial synthesis or QSVT execution path is implemented",
    ]
    if not square:
        reasons.append("the current research policy considers square operators only")
    if not hermitian:
        reasons.append("the current research policy requires a Hermitian operator")
    if operator_norm <= tolerance:
        reasons.append("the operator norm is numerically zero")
    mathematical_candidate = square and hermitian and operator_norm > tolerance
    return BlockEncodingFeasibility(
        rows=rows,
        columns=columns,
        square=square,
        hermitian=hermitian,
        positive_semidefinite=positive_semidefinite,
        operator_norm=operator_norm,
        normalization_factor=normalization,
        normalized_operator_norm=normalized_norm,
        condition_number=condition_number,
        nonzero_entries=nonzero_entries,
        maximum_nonzeros_per_row=maximum_nonzeros,
        density=nonzero_entries / (rows * columns),
        padded_dimension=padded_dimension,
        data_qubits=data_qubits,
        classical_storage_bytes=int(values.nbytes),
        mathematical_qsvt_candidate=mathematical_candidate,
        implementation_reasons=tuple(reasons),
    )


__all__ = ["BlockEncodingFeasibility", "analyze_block_encoding"]
