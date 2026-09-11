"""Typed internal compiler normalization; public compile signature stays stable."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal

from qfin._validation import require_integer
from qfin.exceptions import QFinValidationError
from qfin.representation.strategies import RepresentationTarget


@dataclass(frozen=True, slots=True)
class CompileConfig:
    target_error: float = 0.01
    backend: str = "auto"
    min_qubits: int = 3
    max_qubits: int = 12
    tail_probability: float | None = None
    representation_method: Literal["auto", "quantile", "probability"] = "auto"
    payoff_angle_tolerance: float = 0.1
    payoff_max_terms: int | None = None
    representation_target: RepresentationTarget | None = None
    max_state_preparation_parameters: int = 32767
    max_state_preparation_memory_bytes: int = 256 * 1024 * 1024
    arithmetic_scale: float | None = None
    max_arithmetic_qubits: int = 16
    max_affine_output_qubits: int = 16
    max_factor_validation_points: int = 1048576
    factor_validation_chunk_size: int = 65536
    max_integer_monomials: int = 4096
    max_factorized_wires: int = 28

    def normalized(self) -> CompileConfig:
        target_error = self.target_error
        backend = self.backend
        min_qubits = self.min_qubits
        max_qubits = self.max_qubits
        tail_probability = self.tail_probability
        representation_method = self.representation_method
        payoff_angle_tolerance = self.payoff_angle_tolerance
        payoff_max_terms = self.payoff_max_terms
        representation_target = self.representation_target
        max_state_preparation_parameters = self.max_state_preparation_parameters
        max_state_preparation_memory_bytes = self.max_state_preparation_memory_bytes
        arithmetic_scale = self.arithmetic_scale
        max_arithmetic_qubits = self.max_arithmetic_qubits
        max_affine_output_qubits = self.max_affine_output_qubits
        max_factor_validation_points = self.max_factor_validation_points
        factor_validation_chunk_size = self.factor_validation_chunk_size
        max_integer_monomials = self.max_integer_monomials
        max_factorized_wires = self.max_factorized_wires
        if not isfinite(target_error) or target_error <= 0:
            raise QFinValidationError("target_error must be finite and greater than zero")
        min_qubits = require_integer(min_qubits, "min_qubits", minimum=1)
        max_qubits = require_integer(max_qubits, "max_qubits", minimum=min_qubits)
        payoff_max_terms = (
            None
            if payoff_max_terms is None
            else require_integer(payoff_max_terms, "payoff_max_terms", minimum=1)
        )
        max_state_preparation_parameters = require_integer(
            max_state_preparation_parameters,
            "max_state_preparation_parameters",
            minimum=0,
        )
        max_state_preparation_memory_bytes = require_integer(
            max_state_preparation_memory_bytes,
            "max_state_preparation_memory_bytes",
            minimum=1,
        )
        max_arithmetic_qubits = require_integer(
            max_arithmetic_qubits,
            "max_arithmetic_qubits",
            minimum=1,
        )
        max_affine_output_qubits = require_integer(
            max_affine_output_qubits,
            "max_affine_output_qubits",
            minimum=1,
        )
        max_factor_validation_points = require_integer(
            max_factor_validation_points,
            "max_factor_validation_points",
            minimum=1,
        )
        factor_validation_chunk_size = require_integer(
            factor_validation_chunk_size,
            "factor_validation_chunk_size",
            minimum=1,
        )
        max_integer_monomials = require_integer(
            max_integer_monomials,
            "max_integer_monomials",
            minimum=1,
        )
        max_factorized_wires = require_integer(
            max_factorized_wires,
            "max_factorized_wires",
            minimum=1,
        )
        if representation_method not in ("auto", "quantile", "probability"):
            raise QFinValidationError(
                "representation_method must be 'auto', 'quantile', or 'probability'"
            )
        if not isfinite(payoff_angle_tolerance) or payoff_angle_tolerance <= 0:
            raise QFinValidationError("payoff_angle_tolerance must be finite and positive")
        if arithmetic_scale is not None and (
            not isfinite(arithmetic_scale) or arithmetic_scale <= 0.0
        ):
            raise QFinValidationError("arithmetic_scale must be finite and positive")
        if tail_probability is not None and (
            not isfinite(tail_probability) or not 0.0 < tail_probability < 1.0
        ):
            raise QFinValidationError("tail_probability must lie strictly between zero and one")

        return CompileConfig(
            target_error=target_error,
            backend=backend,
            min_qubits=min_qubits,
            max_qubits=max_qubits,
            tail_probability=tail_probability,
            representation_method=representation_method,
            payoff_angle_tolerance=payoff_angle_tolerance,
            payoff_max_terms=payoff_max_terms,
            representation_target=representation_target,
            max_state_preparation_parameters=max_state_preparation_parameters,
            max_state_preparation_memory_bytes=max_state_preparation_memory_bytes,
            arithmetic_scale=arithmetic_scale,
            max_arithmetic_qubits=max_arithmetic_qubits,
            max_affine_output_qubits=max_affine_output_qubits,
            max_factor_validation_points=max_factor_validation_points,
            factor_validation_chunk_size=factor_validation_chunk_size,
            max_integer_monomials=max_integer_monomials,
            max_factorized_wires=max_factorized_wires,
        )
