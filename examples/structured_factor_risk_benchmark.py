"""Reproducible structured factor VaR/CVaR benchmark (no claimed speedup)."""

from __future__ import annotations

import platform
from collections.abc import Callable
from dataclasses import dataclass
from math import log2
from statistics import median
from time import perf_counter
from typing import TypeVar

import numpy as np

import qfin

T = TypeVar("T")


@dataclass(frozen=True)
class BenchmarkRow:
    factors: int
    points_per_factor: int
    joint_points: int
    generic_seconds: float
    streamed_seconds: float
    generic_stored_values: int
    structured_input_values: int
    var_difference: float
    cvar_difference: float


def timed(call: Callable[[], T], repeats: int) -> tuple[float, T]:
    measurements: list[float] = []
    result: T | None = None
    for _ in range(repeats):
        start = perf_counter()
        result = call()
        measurements.append(perf_counter() - start)
    assert result is not None
    return median(measurements), result


def build_model(factors: int, points_per_factor: int) -> qfin.FactorizedLossModel:
    qubits = int(log2(points_per_factor))
    grid = np.linspace(-2.0, 2.0, points_per_factor)
    raw = np.exp(-0.5 * grid**2)
    probabilities = raw / np.sum(raw)
    marginals = tuple(
        qfin.DistributionEncoding(
            grid=grid,
            probabilities=probabilities,
            qubits=qubits,
            lower_bound=-2.0,
            upper_bound=2.0,
            tail_probability=0.0,
            discretization_error=0.0,
            mean_error=0.0,
            objective="benchmark",
        )
        for _ in range(factors)
    )
    names = tuple(f"factor_{index}" for index in range(factors))
    objective = qfin.SparseExposureObjective(
        constant=0.25,
        linear={name: 1.0 + 0.1 * index for index, name in enumerate(names)},
        quadratic={(names[0], names[-1]): 0.05},
    )
    return qfin.FactorizedLossModel(
        qfin.FactorizedDistributionEncoding(marginals, names),
        objective,
    )


def generic_reference(
    model: qfin.FactorizedLossModel,
    confidence: float,
) -> qfin.RiskSummary:
    _, losses, probabilities = model.chunk(0, model.joint_grid_points)
    return qfin.aggregate_risk(
        qfin.LossDistribution(losses, probabilities),
        confidence=confidence,
        engine="numpy",
    )


def run_classical_rows() -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []
    for factors, points_per_factor in ((3, 2), (3, 4), (3, 8), (3, 16), (4, 16)):
        model = build_model(factors, points_per_factor)
        problem = qfin.FactorCVaR(model, confidence=0.95)
        repeats = 3 if model.joint_grid_points <= 4_096 else 1
        generic_seconds, generic = timed(
            lambda selected_model=model, confidence=problem.confidence: generic_reference(
                selected_model, confidence
            ),
            repeats,
        )
        streamed_seconds, streamed = timed(
            lambda selected_problem=problem, selected_model=model: qfin.evaluate_factor_risk(
                selected_problem,
                chunk_size=65_536,
                max_points=selected_model.joint_grid_points,
            ),
            repeats,
        )
        rows.append(
            BenchmarkRow(
                factors=factors,
                points_per_factor=points_per_factor,
                joint_points=model.joint_grid_points,
                generic_seconds=generic_seconds,
                streamed_seconds=streamed_seconds,
                generic_stored_values=2 * model.joint_grid_points,
                structured_input_values=2 * factors * points_per_factor,
                var_difference=abs(streamed.var - generic.var),
                cvar_difference=abs(streamed.cvar - generic.cvar),
            )
        )
    return rows


def quantum_microbenchmark() -> dict[str, float | int | str]:
    model = build_model(1, 2)
    problem = qfin.FactorCVaR(model, confidence=0.75)
    start = perf_counter()
    compiled = qfin.compile(
        problem,
        backend="pennylane",
        target_error=0.2,
        arithmetic_scale=2.0,
        max_factor_validation_points=2,
        max_factorized_wires=12,
    )
    compile_seconds = perf_counter() - start
    threshold_code = compiled.validation.occupied_codes[0]
    tail = compiled.tail_runtime(threshold_code + 1, max_total_wires=12)
    excess = compiled.excess_runtime(threshold_code, 0, max_total_wires=12)
    tail.probability()
    excess.probability()
    tail_seconds, tail_probability = timed(tail.probability, repeats=5)
    excess_seconds, excess_probability = timed(excess.probability, repeats=5)
    return {
        "compile_seconds": compile_seconds,
        "tail_seconds": tail_seconds,
        "excess_seconds": excess_seconds,
        "tail_probability_error": abs(tail_probability - tail.theoretical_amplitude()),
        "excess_probability_error": abs(excess_probability - excess.theoretical_amplitude()),
        "loss_qubits": compiled.oracle.loss_qubits,
        "maximum_runtime_qubits": compiled.resources().maximum_runtime_qubits,
        "device": tail.device_name,
    }


def main() -> None:
    rows = run_classical_rows()
    quantum = quantum_microbenchmark()
    info = qfin.system_info()
    print("# QFin 1.0 structured factor-risk benchmark")
    print()
    print(f"- OS: {platform.platform()}")
    print(f"- CPU: {platform.processor() or 'not reported by platform'}")
    print(f"- Python: {platform.python_version()}")
    print(f"- NumPy: {np.__version__}")
    print(f"- QFin: {qfin.__version__}")
    print(f"- Native compiler: {info['native_compiler']}")
    print(f"- Quantum device: {quantum['device']}")
    print()
    print("## Classical exact-reference scaling")
    print()
    print(
        "| Factors x marginal points | Joint points | Generic NumPy (s) | "
        "Streamed exact (s) | Generic stored values | Structured input values | "
        "|VaR diff| | |CVaR diff| |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row.factors} x {row.points_per_factor} | {row.joint_points:,} | "
            f"{row.generic_seconds:.6f} | {row.streamed_seconds:.6f} | "
            f"{row.generic_stored_values:,} | {row.structured_input_values:,} | "
            f"{row.var_difference:.3e} | {row.cvar_difference:.3e} |"
        )
    print()
    print("The streamed calculation is the bounded-memory correctness oracle; it performs")
    print("repeated passes and is not presented as a speed optimization.")
    print()
    print("## Structured circuit microbenchmark")
    print()
    print("Circuit timings are medians of five runs after one warm-up execution.")
    print()
    for key, value in quantum.items():
        rendered = f"{value:.6g}" if isinstance(value, float) else str(value)
        print(f"- {key}: {rendered}")
    print()
    print("These are simulator wall-clock measurements, not hardware or advantage claims.")


if __name__ == "__main__":
    main()
