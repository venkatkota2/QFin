"""Reversible fixed-point arithmetic plans for structured factor objectives."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, isfinite, log2, sqrt
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from qfin.exceptions import BackendUnavailableError, ResourceLimitError
from qfin.representation.factorized import FactorizedDistributionEncoding

if TYPE_CHECKING:
    from qfin.finance.exposures import FactorTailProbability, SparseExposureObjective

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _qml() -> Any:
    try:
        import pennylane as qml
    except ImportError as exc:
        raise BackendUnavailableError(
            "PennyLane is required for reversible arithmetic circuits. "
            "Install QFin with `python -m pip install -e '.[quantum]'`."
        ) from exc
    return qml


def _zero_polynomial(*_values: int) -> int:
    """Placeholder callable; QFin supplies PennyLane's bit coefficients directly."""

    return 0


def _required_qubits(maximum: int) -> int:
    if maximum < 0:
        raise ValueError("maximum must be non-negative")
    return max(1, ceil(log2(maximum + 1))) if maximum else 1


@dataclass(frozen=True, slots=True)
class IntegerQuadraticTerm:
    """One integer coefficient multiplying two unsigned input registers."""

    left: int
    right: int
    coefficient: int

    def __post_init__(self) -> None:
        if self.left < 0 or self.right < self.left:
            raise ValueError("quadratic indices must satisfy 0 <= left <= right")
        if self.coefficient == 0:
            raise ValueError("quadratic coefficient must be non-zero")


@dataclass(frozen=True, slots=True)
class IntegerPolynomialPlan:
    """Sparse integer quadratic mapped to PennyLane ``OutPoly`` coefficients."""

    input_qubits: tuple[int, ...]
    output_qubits: int
    constant: int
    linear: tuple[int, ...]
    quadratic: tuple[IntegerQuadraticTerm, ...] = ()
    range_policy: Literal["bounded_unsigned", "modular_addend"] = "bounded_unsigned"

    def __post_init__(self) -> None:
        if not self.input_qubits or any(qubits < 1 for qubits in self.input_qubits):
            raise ValueError("input_qubits must contain positive register widths")
        if self.output_qubits < 1:
            raise ValueError("output_qubits must be positive")
        if len(self.linear) != len(self.input_qubits):
            raise ValueError("linear must contain one coefficient per input register")
        for term in self.quadratic:
            if term.right >= len(self.input_qubits):
                raise ValueError("quadratic term references an unavailable register")
        if self.range_policy not in ("bounded_unsigned", "modular_addend"):
            raise ValueError("invalid integer polynomial range policy")
        lower, upper = self.range_bounds()
        if self.range_policy == "bounded_unsigned" and (
            lower < 0 or upper >= 2**self.output_qubits
        ):
            raise ValueError("integer polynomial range does not fit the output register")

    @property
    def input_bits(self) -> int:
        return sum(self.input_qubits)

    @property
    def bit_coefficients(self) -> tuple[tuple[tuple[int, ...], int], ...]:
        """Return the sparse Boolean polynomial consumed by ``qml.OutPoly``."""

        coefficients: defaultdict[tuple[int, ...], int] = defaultdict(int)
        zero = (0,) * self.input_bits
        coefficients[zero] += self.constant
        starts = np.cumsum((0, *self.input_qubits[:-1]), dtype=np.int64)

        def add_mask(positions: tuple[int, ...], coefficient: int) -> None:
            if coefficient == 0:
                return
            mask = [0] * self.input_bits
            for position in positions:
                mask[position] = 1
            coefficients[tuple(mask)] += coefficient

        for register, coefficient in enumerate(self.linear):
            start = int(starts[register])
            qubits = self.input_qubits[register]
            for bit in range(qubits):
                weight = 1 << (qubits - 1 - bit)
                add_mask((start + bit,), coefficient * weight)

        for term in self.quadratic:
            left_start = int(starts[term.left])
            right_start = int(starts[term.right])
            left_qubits = self.input_qubits[term.left]
            right_qubits = self.input_qubits[term.right]
            if term.left != term.right:
                for left_bit in range(left_qubits):
                    left_weight = 1 << (left_qubits - 1 - left_bit)
                    for right_bit in range(right_qubits):
                        right_weight = 1 << (right_qubits - 1 - right_bit)
                        add_mask(
                            (left_start + left_bit, right_start + right_bit),
                            term.coefficient * left_weight * right_weight,
                        )
                continue

            for left_bit in range(left_qubits):
                left_weight = 1 << (left_qubits - 1 - left_bit)
                add_mask((left_start + left_bit,), term.coefficient * left_weight**2)
                for right_bit in range(left_bit + 1, left_qubits):
                    right_weight = 1 << (left_qubits - 1 - right_bit)
                    add_mask(
                        (left_start + left_bit, left_start + right_bit),
                        2 * term.coefficient * left_weight * right_weight,
                    )

        return tuple(
            (mask, coefficient)
            for mask, coefficient in sorted(coefficients.items())
            if coefficient != 0
        )

    @property
    def monomial_count(self) -> int:
        return len(self.bit_coefficients)

    def range_bounds(self) -> tuple[int, int]:
        lower = self.constant
        upper = self.constant
        maxima = tuple(2**qubits - 1 for qubits in self.input_qubits)
        for register, coefficient in enumerate(self.linear):
            contribution = coefficient * maxima[register]
            lower += min(0, contribution)
            upper += max(0, contribution)
        for term in self.quadratic:
            contribution = term.coefficient * maxima[term.left] * maxima[term.right]
            lower += min(0, contribution)
            upper += max(0, contribution)
        return lower, upper

    def evaluate(self, indices: IntArray) -> IntArray:
        values = np.asarray(indices, dtype=np.int64)
        if values.ndim != 2 or values.shape[1] != len(self.input_qubits):
            raise ValueError("indices must be point-by-input-register")
        result = np.full(values.shape[0], self.constant, dtype=np.int64)
        for register, coefficient in enumerate(self.linear):
            result += coefficient * values[:, register]
        for term in self.quadratic:
            result += (
                term.coefficient * values[:, term.left] * values[:, term.right]
            )
        return result

    def apply(
        self,
        input_registers: Sequence[Sequence[int]],
        output_wires: Sequence[int],
    ) -> None:
        registers = tuple(tuple(wires) for wires in input_registers)
        output = tuple(output_wires)
        if tuple(len(wires) for wires in registers) != self.input_qubits:
            raise ValueError("input register widths do not match the arithmetic plan")
        if len(output) != self.output_qubits:
            raise ValueError("output register width does not match the arithmetic plan")
        if len(set((*sum(registers, ()), *output))) != self.input_bits + len(output):
            raise ValueError("input and output arithmetic wires must be disjoint")
        qml = _qml()
        qml.OutPoly(
            _zero_polynomial,
            input_registers=registers,
            output_wires=output,
            coeffs_list=self.bit_coefficients,
        )


@dataclass(frozen=True, slots=True)
class AffineOutputPlan:
    """One fixed-point affine output stored in an unsigned work register."""

    name: str
    scale: float
    shift_ticks: int
    minimum_code: int
    maximum_code: int
    maximum_abs_error_bound: float
    polynomial: IntegerPolynomialPlan

    def decode(self, codes: IntArray) -> FloatArray:
        values = np.asarray(codes, dtype=np.float64)
        return np.asarray((values - self.shift_ticks) / self.scale, dtype=np.float64)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "scale": self.scale,
            "shift_ticks": self.shift_ticks,
            "minimum_code": self.minimum_code,
            "maximum_code": self.maximum_code,
            "output_qubits": self.polynomial.output_qubits,
            "integer_monomials": self.polynomial.monomial_count,
            "maximum_abs_error_bound": self.maximum_abs_error_bound,
        }


@dataclass(frozen=True, slots=True)
class ReversibleAffineTransformPlan:
    """Out-of-place fixed-point affine map implemented with ``qml.OutPoly``."""

    input_qubits: tuple[int, ...]
    latent_factor_names: tuple[str, ...]
    outputs: tuple[AffineOutputPlan, ...]
    input_grid_lower: FloatArray
    input_grid_step: FloatArray
    real_base: FloatArray
    real_coefficients: FloatArray

    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(output.name for output in self.outputs)

    @property
    def output_qubits(self) -> tuple[int, ...]:
        return tuple(output.polynomial.output_qubits for output in self.outputs)

    @property
    def total_output_qubits(self) -> int:
        return sum(self.output_qubits)

    @property
    def integer_monomials(self) -> int:
        return sum(output.polynomial.monomial_count for output in self.outputs)

    def output(self, name: str) -> AffineOutputPlan:
        try:
            return self.outputs[self.output_names.index(name)]
        except ValueError as exc:
            raise KeyError(name) from exc

    def evaluate_codes(self, indices: IntArray) -> IntArray:
        columns = [output.polynomial.evaluate(indices) for output in self.outputs]
        return np.column_stack(columns).astype(np.int64, copy=False)

    def decode(self, codes: IntArray) -> FloatArray:
        values = np.asarray(codes, dtype=np.int64)
        if values.ndim != 2 or values.shape[1] != len(self.outputs):
            raise ValueError("codes must be point-by-affine-output")
        return np.column_stack(
            [output.decode(values[:, index]) for index, output in enumerate(self.outputs)]
        )

    def exact_values(self, indices: IntArray) -> FloatArray:
        values = np.asarray(indices, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.input_qubits):
            raise ValueError("indices must be point-by-input-register")
        return np.asarray(self.real_base + values @ self.real_coefficients.T, dtype=np.float64)

    def apply(
        self,
        input_registers: Sequence[Sequence[int]],
        output_registers: Sequence[Sequence[int]],
    ) -> None:
        registers = tuple(tuple(wires) for wires in input_registers)
        outputs = tuple(tuple(wires) for wires in output_registers)
        if len(outputs) != len(self.outputs):
            raise ValueError("one output register is required per affine output")
        for output, wires in zip(self.outputs, outputs, strict=True):
            output.polynomial.apply(registers, wires)

    def to_dict(self) -> dict[str, object]:
        return {
            "implementation": "PennyLane OutPoly fixed-point affine arithmetic",
            "input_qubits": list(self.input_qubits),
            "latent_factor_names": list(self.latent_factor_names),
            "outputs": [output.to_dict() for output in self.outputs],
            "total_output_qubits": self.total_output_qubits,
            "integer_monomials": self.integer_monomials,
            "joint_table_materialized": False,
            "reversible_quantum_arithmetic_implemented": True,
            "caveat": (
                "The map uses unsigned modular registers sized to prevent wrap on the "
                "validated domain. It is fixed-point arithmetic, not arbitrary precision."
            ),
        }


@dataclass(frozen=True, slots=True)
class AffineTransformValidation:
    """Numerical parity between the real affine map and fixed-point registers."""

    evaluated_points: int
    chunks: int
    maximum_abs_error: float
    root_mean_square_error: float
    error_bound: float
    joint_table_materialized: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "evaluated_points": self.evaluated_points,
            "chunks": self.chunks,
            "maximum_abs_error": self.maximum_abs_error,
            "root_mean_square_error": self.root_mean_square_error,
            "error_bound": self.error_bound,
            "joint_table_materialized": self.joint_table_materialized,
        }


def _grid_affine_metadata(
    encoding: FactorizedDistributionEncoding,
    *,
    tolerance: float,
) -> tuple[FloatArray, FloatArray]:
    lower = np.empty(encoding.factor_count, dtype=np.float64)
    step = np.empty(encoding.factor_count, dtype=np.float64)
    for index, factor in enumerate(encoding.factors):
        differences = np.diff(factor.grid)
        candidate = float(differences[0])
        if candidate <= 0 or not np.allclose(differences, candidate, atol=tolerance, rtol=1e-10):
            raise ValueError(
                "reversible arithmetic requires affine factor grids; use "
                "encode_independent_factors(..., method='probability')"
            )
        lower[index] = float(factor.grid[0])
        step[index] = candidate
    return lower, step


def _real_affine_map(
    encoding: FactorizedDistributionEncoding,
    *,
    tolerance: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, tuple[str, ...]]:
    lower, step = _grid_affine_metadata(encoding, tolerance=tolerance)
    if encoding.transform is None:
        matrix = np.eye(encoding.factor_count, dtype=np.float64)
        offset = np.zeros(encoding.factor_count, dtype=np.float64)
        names = tuple(encoding.factor_names)
    else:
        matrix = encoding.transform.matrix
        offset = encoding.transform.offset
        names = tuple(encoding.transform.output_names)
    base = np.asarray(offset + matrix @ lower, dtype=np.float64)
    coefficients = np.asarray(matrix * step[np.newaxis, :], dtype=np.float64)
    return lower, step, base, coefficients, names


def compile_affine_transform(
    encoding: FactorizedDistributionEncoding,
    *,
    scale: float = 1_024.0,
    output_names: Sequence[str] | None = None,
    max_output_qubits: int = 16,
    grid_tolerance: float = 1e-12,
) -> ReversibleAffineTransformPlan:
    """Compile an affine financial factor map into fixed-point output registers."""

    if not isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    if max_output_qubits < 1:
        raise ValueError("max_output_qubits must be positive")
    lower, step, base, coefficients, names = _real_affine_map(
        encoding, tolerance=grid_tolerance
    )
    selected = names if output_names is None else tuple(output_names)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("output_names must be non-empty and unique")
    missing = set(selected) - set(names)
    if missing:
        raise ValueError("unknown affine outputs: " + ", ".join(sorted(missing)))
    maxima = np.asarray([2**qubits - 1 for qubits in encoding.qubits_per_factor], dtype=np.int64)
    plans: list[AffineOutputPlan] = []
    real_rows: list[FloatArray] = []
    real_bases: list[float] = []
    for name in selected:
        row_index = names.index(name)
        base_ticks = round(scale * float(base[row_index]))
        coefficient_ticks = tuple(
            round(scale * float(value)) for value in coefficients[row_index]
        )
        raw_min = base_ticks
        raw_max = base_ticks
        for coefficient, maximum in zip(coefficient_ticks, maxima, strict=True):
            contribution = coefficient * int(maximum)
            raw_min += min(0, contribution)
            raw_max += max(0, contribution)
        shift = max(0, -raw_min)
        minimum_code = raw_min + shift
        maximum_code = raw_max + shift
        output_qubits = _required_qubits(maximum_code)
        if output_qubits > max_output_qubits:
            raise ResourceLimitError(
                f"affine output {name!r} requires {output_qubits} qubits at scale={scale:g}, "
                f"above max_output_qubits={max_output_qubits}"
            )
        rounding_bound = 0.5 * (1.0 + float(np.sum(maxima))) / scale
        polynomial = IntegerPolynomialPlan(
            input_qubits=encoding.qubits_per_factor,
            output_qubits=output_qubits,
            constant=base_ticks + shift,
            linear=coefficient_ticks,
        )
        plans.append(
            AffineOutputPlan(
                name=name,
                scale=scale,
                shift_ticks=shift,
                minimum_code=minimum_code,
                maximum_code=maximum_code,
                maximum_abs_error_bound=rounding_bound,
                polynomial=polynomial,
            )
        )
        real_bases.append(float(base[row_index]))
        real_rows.append(np.asarray(coefficients[row_index], dtype=np.float64))
    return ReversibleAffineTransformPlan(
        input_qubits=encoding.qubits_per_factor,
        latent_factor_names=tuple(encoding.factor_names),
        outputs=tuple(plans),
        input_grid_lower=lower,
        input_grid_step=step,
        real_base=np.asarray(real_bases, dtype=np.float64),
        real_coefficients=np.vstack(real_rows),
    )


def validate_affine_transform(
    encoding: FactorizedDistributionEncoding,
    plan: ReversibleAffineTransformPlan,
    *,
    chunk_size: int = 65_536,
    max_points: int = 1_048_576,
) -> AffineTransformValidation:
    """Stream every encoded basis state and validate fixed-point affine outputs."""

    if chunk_size < 1 or max_points < 1:
        raise ValueError("chunk_size and max_points must be positive")
    points = encoding.joint_grid_points
    if points > max_points:
        raise ValueError(
            f"affine validation requires {points} streamed points, above max_points={max_points}"
        )
    maximum = 0.0
    squared = 0.0
    chunks = 0
    maxima = tuple(factor.grid_points for factor in encoding.factors)
    for start in range(0, points, chunk_size):
        stop = min(start + chunk_size, points)
        flat = np.arange(start, stop, dtype=np.int64)
        residual = flat.copy()
        indices = np.empty((flat.size, encoding.factor_count), dtype=np.int64)
        for factor_index in range(encoding.factor_count - 1, -1, -1):
            indices[:, factor_index] = residual % maxima[factor_index]
            residual //= maxima[factor_index]
        exact = plan.exact_values(indices)
        approximate = plan.decode(plan.evaluate_codes(indices))
        difference = approximate - exact
        maximum = max(maximum, float(np.max(np.abs(difference))))
        squared += float(np.sum(difference**2))
        chunks += 1
    return AffineTransformValidation(
        evaluated_points=points,
        chunks=chunks,
        maximum_abs_error=maximum,
        root_mean_square_error=sqrt(squared / (points * len(plan.outputs))),
        error_bound=max(output.maximum_abs_error_bound for output in plan.outputs),
    )


@dataclass(frozen=True, slots=True)
class IntegerHingePlan:
    """A conditional fixed-point positive-part addition."""

    factor: str
    affine_output_index: int
    threshold_code: int
    coefficient: int
    mode: Literal["inactive", "always", "conditional"]
    polynomial: IntegerPolynomialPlan
    minimum_contribution: int
    maximum_contribution: int

    def to_dict(self) -> dict[str, object]:
        return {
            "factor": self.factor,
            "threshold_code": self.threshold_code,
            "coefficient": self.coefficient,
            "mode": self.mode,
            "integer_monomials": self.polynomial.monomial_count,
            "minimum_contribution": self.minimum_contribution,
            "maximum_contribution": self.maximum_contribution,
        }


@dataclass(frozen=True, slots=True)
class StructuredLossOraclePlan:
    """Reversible sparse loss arithmetic with no joint payoff lookup table."""

    input_qubits: tuple[int, ...]
    loss_scale: float
    loss_shift_ticks: int
    polynomial: IntegerPolynomialPlan
    affine: ReversibleAffineTransformPlan | None
    hinges: tuple[IntegerHingePlan, ...]
    coefficient_rounding_bound: float

    @property
    def loss_qubits(self) -> int:
        return self.polynomial.output_qubits

    @property
    def affine_qubits(self) -> tuple[int, ...]:
        return () if self.affine is None else self.affine.output_qubits

    @property
    def total_affine_qubits(self) -> int:
        return sum(self.affine_qubits)

    @property
    def piecewise_work_qubits(self) -> int:
        return int(any(hinge.mode == "conditional" for hinge in self.hinges))

    @property
    def arithmetic_qubits(self) -> int:
        return self.loss_qubits + self.total_affine_qubits + self.piecewise_work_qubits

    @property
    def integer_monomials(self) -> int:
        affine = 0 if self.affine is None else self.affine.integer_monomials
        return self.polynomial.monomial_count + affine + sum(
            hinge.polynomial.monomial_count
            for hinge in self.hinges
            if hinge.mode != "inactive"
        )

    def decode_loss(self, codes: IntArray) -> FloatArray:
        values = np.asarray(codes, dtype=np.float64)
        return np.asarray((values - self.loss_shift_ticks) / self.loss_scale, dtype=np.float64)

    def evaluate_codes(self, indices: IntArray) -> IntArray:
        values = self.polynomial.evaluate(indices)
        if self.affine is None:
            return values
        affine_codes = self.affine.evaluate_codes(indices)
        for hinge in self.hinges:
            if hinge.mode == "inactive":
                continue
            code = affine_codes[:, hinge.affine_output_index]
            values += hinge.coefficient * np.maximum(code - hinge.threshold_code, 0)
        if np.any(values < 0) or np.any(values >= 2**self.loss_qubits):
            raise RuntimeError("compiled loss arithmetic exceeded its validated register range")
        return values

    def threshold_code(self, threshold: float, *, inclusive: bool) -> int:
        scaled = self.loss_scale * threshold + self.loss_shift_ticks
        return ceil(scaled) if inclusive else int(np.floor(scaled)) + 1

    def apply(
        self,
        input_registers: Sequence[Sequence[int]],
        loss_wires: Sequence[int],
        affine_output_registers: Sequence[Sequence[int]] = (),
        piecewise_work_wire: int | None = None,
    ) -> None:
        registers = tuple(tuple(wires) for wires in input_registers)
        loss = tuple(loss_wires)
        self.polynomial.apply(registers, loss)
        if self.affine is None:
            return
        affine_registers = tuple(tuple(wires) for wires in affine_output_registers)
        self.affine.apply(registers, affine_registers)
        qml = _qml()
        for hinge in self.hinges:
            if hinge.mode == "inactive":
                continue
            source = affine_registers[hinge.affine_output_index]
            operation = qml.OutPoly(
                _zero_polynomial,
                input_registers=[source],
                output_wires=loss,
                coeffs_list=hinge.polynomial.bit_coefficients,
            )
            if hinge.mode == "always":
                continue
            if piecewise_work_wire is None:
                raise ValueError("conditional hinge arithmetic requires a work wire")
            qml.IntegerComparator(
                hinge.threshold_code,
                geq=True,
                wires=(*source, piecewise_work_wire),
            )
            qml.ctrl(operation, control=piecewise_work_wire)
            qml.IntegerComparator(
                hinge.threshold_code,
                geq=True,
                wires=(*source, piecewise_work_wire),
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "implementation": "reversible sparse fixed-point OutPoly arithmetic",
            "input_qubits": list(self.input_qubits),
            "loss_scale": self.loss_scale,
            "loss_shift_ticks": self.loss_shift_ticks,
            "loss_qubits": self.loss_qubits,
            "affine_qubits": list(self.affine_qubits),
            "piecewise_work_qubits": self.piecewise_work_qubits,
            "arithmetic_qubits": self.arithmetic_qubits,
            "integer_monomials": self.integer_monomials,
            "coefficient_rounding_bound": self.coefficient_rounding_bound,
            "hinges": [hinge.to_dict() for hinge in self.hinges],
            "joint_payoff_table_materialized": False,
            "reversible_quantum_arithmetic_implemented": True,
        }


def _objective_polynomial_in_indices(
    encoding: FactorizedDistributionEncoding,
    objective: SparseExposureObjective,
) -> tuple[float, FloatArray, FloatArray]:
    _, _, base, coefficients, names = _real_affine_map(encoding, tolerance=1e-12)
    count = len(names)
    linear = np.zeros(count, dtype=np.float64)
    quadratic = np.zeros((count, count), dtype=np.float64)
    name_to_index = {name: index for index, name in enumerate(names)}
    for name, coefficient in objective.linear.items():
        linear[name_to_index[name]] += coefficient
    for (left, right), coefficient in objective.quadratic.items():
        left_index = name_to_index[left]
        right_index = name_to_index[right]
        if left_index == right_index:
            quadratic[left_index, right_index] += coefficient
        else:
            quadratic[left_index, right_index] += 0.5 * coefficient
            quadratic[right_index, left_index] += 0.5 * coefficient
    constant_index = float(
        objective.constant + linear @ base + base @ quadratic @ base
    )
    linear_index = np.asarray(
        coefficients.T @ (linear + 2.0 * quadratic @ base), dtype=np.float64
    )
    quadratic_index = np.asarray(coefficients.T @ quadratic @ coefficients, dtype=np.float64)
    return constant_index, linear_index, quadratic_index


def compile_structured_loss_oracle(
    encoding: FactorizedDistributionEncoding,
    objective: SparseExposureObjective,
    *,
    loss_scale: float = 1_024.0,
    factor_scale: float | None = None,
    max_loss_qubits: int = 16,
    max_affine_output_qubits: int = 16,
) -> StructuredLossOraclePlan:
    """Compile a sparse financial objective into reversible integer arithmetic."""

    if not isfinite(loss_scale) or loss_scale <= 0:
        raise ValueError("loss_scale must be finite and positive")
    if factor_scale is None:
        minimum_hinge_slope = min(
            (abs(term.slope) for term in objective.piecewise),
            default=1.0,
        )
        factor_scale = loss_scale * min(1.0, minimum_hinge_slope)
    if not isfinite(factor_scale) or factor_scale <= 0:
        raise ValueError("factor_scale must be finite and positive")
    if max_loss_qubits < 1:
        raise ValueError("max_loss_qubits must be positive")

    constant_real, linear_real, quadratic_real = _objective_polynomial_in_indices(
        encoding, objective
    )
    constant_ticks = round(loss_scale * constant_real)
    linear_ticks = tuple(round(loss_scale * value) for value in linear_real)
    quadratic_terms: list[IntegerQuadraticTerm] = []
    coefficient_rounding_bound = abs(constant_ticks / loss_scale - constant_real)
    maxima = tuple(2**qubits - 1 for qubits in encoding.qubits_per_factor)
    coefficient_rounding_bound += sum(
        abs(integer / loss_scale - real) * maximum
        for integer, real, maximum in zip(
            linear_ticks, linear_real, maxima, strict=True
        )
    )
    for left in range(encoding.factor_count):
        for right in range(left, encoding.factor_count):
            real_value = (
                float(quadratic_real[left, left])
                if left == right
                else 2.0 * float(quadratic_real[left, right])
            )
            coefficient = round(loss_scale * real_value)
            if coefficient != 0:
                quadratic_terms.append(IntegerQuadraticTerm(left, right, coefficient))
            coefficient_rounding_bound += (
                abs(coefficient / loss_scale - real_value)
                * maxima[left]
                * maxima[right]
            )

    piecewise_names = tuple(dict.fromkeys(term.factor for term in objective.piecewise))
    affine = (
        None
        if not piecewise_names
        else compile_affine_transform(
            encoding,
            scale=factor_scale,
            output_names=piecewise_names,
            max_output_qubits=max_affine_output_qubits,
        )
    )
    preliminary = IntegerPolynomialPlan(
        input_qubits=encoding.qubits_per_factor,
        output_qubits=1,
        constant=constant_ticks,
        linear=linear_ticks,
        quadratic=tuple(quadratic_terms),
        range_policy="modular_addend",
    )
    raw_min, raw_max = preliminary.range_bounds()

    hinges_metadata: list[
        tuple[
            str,
            int,
            int,
            int,
            Literal["inactive", "always", "conditional"],
            int,
            int,
        ]
    ] = []
    if affine is not None:
        for term in objective.piecewise:
            output_index = affine.output_names.index(term.factor)
            output = affine.outputs[output_index]
            threshold_code = ceil(factor_scale * term.threshold + output.shift_ticks)
            coefficient = round(loss_scale * term.slope / factor_scale)
            if coefficient == 0:
                raise ResourceLimitError(
                    f"piecewise slope for {term.factor!r} rounds to zero; increase loss_scale"
                )
            if threshold_code > output.maximum_code:
                mode: Literal["inactive", "always", "conditional"] = "inactive"
                max_delta = 0
            elif threshold_code <= output.minimum_code:
                mode = "always"
                max_delta = output.maximum_code - threshold_code
            else:
                mode = "conditional"
                max_delta = output.maximum_code - threshold_code
            minimum = min(0, coefficient * max_delta)
            maximum = max(0, coefficient * max_delta)
            raw_min += minimum
            raw_max += maximum
            hinges_metadata.append(
                (
                    term.factor,
                    output_index,
                    threshold_code,
                    coefficient,
                    mode,
                    minimum,
                    maximum,
                )
            )

    shift = max(0, -raw_min)
    maximum_code = raw_max + shift
    loss_qubits = _required_qubits(maximum_code)
    if loss_qubits > max_loss_qubits:
        raise ResourceLimitError(
            f"structured loss requires {loss_qubits} output qubits at scale={loss_scale:g}, "
            f"above max_loss_qubits={max_loss_qubits}"
        )
    polynomial = IntegerPolynomialPlan(
        input_qubits=encoding.qubits_per_factor,
        output_qubits=loss_qubits,
        constant=constant_ticks + shift,
        linear=linear_ticks,
        quadratic=tuple(quadratic_terms),
    )
    hinges: list[IntegerHingePlan] = []
    if affine is not None:
        for factor, output_index, threshold, coefficient, mode, minimum, maximum in hinges_metadata:
            source_qubits = affine.outputs[output_index].polynomial.output_qubits
            hinge_polynomial = IntegerPolynomialPlan(
                input_qubits=(source_qubits,),
                output_qubits=loss_qubits,
                constant=-coefficient * threshold,
                linear=(coefficient,),
                range_policy="modular_addend",
            )
            hinges.append(
                IntegerHingePlan(
                    factor=factor,
                    affine_output_index=output_index,
                    threshold_code=threshold,
                    coefficient=coefficient,
                    mode=mode,
                    polynomial=hinge_polynomial,
                    minimum_contribution=minimum,
                    maximum_contribution=maximum,
                )
            )
    return StructuredLossOraclePlan(
        input_qubits=encoding.qubits_per_factor,
        loss_scale=loss_scale,
        loss_shift_ticks=shift,
        polynomial=polynomial,
        affine=affine,
        hinges=tuple(hinges),
        coefficient_rounding_bound=float(coefficient_rounding_bound),
    )


@dataclass(frozen=True, slots=True)
class StructuredTailOracleValidation:
    """Streaming numerical comparison of exact and fixed-point tail objectives."""

    exact_probability: float
    oracle_probability: float
    oracle_error: float
    disagreement_probability: float
    maximum_loss_error: float
    root_mean_square_loss_error: float
    evaluated_points: int
    chunks: int
    joint_table_materialized: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "exact_probability": self.exact_probability,
            "oracle_probability": self.oracle_probability,
            "oracle_error": self.oracle_error,
            "disagreement_probability": self.disagreement_probability,
            "maximum_loss_error": self.maximum_loss_error,
            "root_mean_square_loss_error": self.root_mean_square_loss_error,
            "evaluated_points": self.evaluated_points,
            "chunks": self.chunks,
            "joint_table_materialized": self.joint_table_materialized,
        }


def validate_structured_tail_oracle(
    problem: FactorTailProbability,
    plan: StructuredLossOraclePlan,
    *,
    chunk_size: int = 65_536,
    max_points: int = 1_048_576,
) -> StructuredTailOracleValidation:
    """Compare exact encoded losses with the reversible fixed-point comparator."""

    if chunk_size < 1 or max_points < 1:
        raise ValueError("chunk_size and max_points must be positive")
    points = problem.model.joint_grid_points
    if points > max_points:
        raise ValueError(
            f"oracle validation requires {points} streamed points, above max_points={max_points}"
        )
    threshold_code = plan.threshold_code(problem.threshold, inclusive=problem.inclusive)
    exact_probability = 0.0
    oracle_probability = 0.0
    disagreement = 0.0
    maximum_error = 0.0
    weighted_squared_error = 0.0
    chunks = 0
    for start in range(0, points, chunk_size):
        stop = min(start + chunk_size, points)
        indices, exact_losses, probabilities = problem.model.chunk(start, stop)
        codes = plan.evaluate_codes(indices)
        approximate_losses = plan.decode_loss(codes)
        exact_selected = (
            exact_losses >= problem.threshold
            if problem.inclusive
            else exact_losses > problem.threshold
        )
        oracle_selected = codes >= threshold_code
        exact_probability += float(np.sum(probabilities[exact_selected]))
        oracle_probability += float(np.sum(probabilities[oracle_selected]))
        disagreement += float(np.sum(probabilities[exact_selected != oracle_selected]))
        differences = approximate_losses - exact_losses
        maximum_error = max(maximum_error, float(np.max(np.abs(differences))))
        weighted_squared_error += float(np.sum(probabilities * differences**2))
        chunks += 1
    exact_probability = float(np.clip(exact_probability, 0.0, 1.0))
    oracle_probability = float(np.clip(oracle_probability, 0.0, 1.0))
    return StructuredTailOracleValidation(
        exact_probability=exact_probability,
        oracle_probability=oracle_probability,
        oracle_error=abs(oracle_probability - exact_probability),
        disagreement_probability=float(np.clip(disagreement, 0.0, 1.0)),
        maximum_loss_error=maximum_error,
        root_mean_square_loss_error=sqrt(max(weighted_squared_error, 0.0)),
        evaluated_points=points,
        chunks=chunks,
    )


__all__ = [
    "AffineOutputPlan",
    "AffineTransformValidation",
    "IntegerHingePlan",
    "IntegerPolynomialPlan",
    "IntegerQuadraticTerm",
    "ReversibleAffineTransformPlan",
    "StructuredLossOraclePlan",
    "StructuredTailOracleValidation",
    "compile_affine_transform",
    "compile_structured_loss_oracle",
    "validate_affine_transform",
    "validate_structured_tail_oracle",
]
