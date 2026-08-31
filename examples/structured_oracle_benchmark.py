"""Reproducible QFin 0.9 structured-oracle benchmark report generator."""

from __future__ import annotations

import argparse
import platform
import statistics
import time
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TypeVar

import numpy as np

import qfin

T = TypeVar("T")


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def _median_call(function: Callable[[], T], repeats: int) -> tuple[float, T]:
    timings: list[float] = []
    result: T | None = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = function()
        timings.append(time.perf_counter() - started)
    assert result is not None
    return statistics.median(timings), result


def _problem(qubits: int) -> qfin.FactorTailProbability:
    encoding = qfin.encode_independent_factors(
        [qfin.Normal(), qfin.Normal()],
        qubits_per_factor=qubits,
        factor_names=("rates", "equity"),
        method="probability",
        tail_probability=1e-4,
    )
    objective = qfin.SparseExposureObjective(
        constant=0.1,
        linear={"rates": 0.75, "equity": -0.4},
        quadratic={("rates", "equity"): 0.125},
        piecewise=(qfin.HingeExposure("equity", threshold=0.0, slope=0.5),),
    )
    return qfin.FactorTailProbability(
        qfin.FactorizedLossModel(encoding, objective),
        threshold=0.25,
    )


def _compile(problem: qfin.FactorTailProbability) -> qfin.CompiledFactorTailModel:
    return qfin.compile(
        problem,
        backend="classical",
        target_error=0.05,
        max_arithmetic_qubits=24,
        max_affine_output_qubits=24,
        max_factor_validation_points=problem.model.joint_grid_points,
        factor_validation_chunk_size=256,
        max_factorized_wires=40,
    )


def _construction_rows(qubits: tuple[int, ...], repeats: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for count in qubits:
        problem = _problem(count)
        seconds, compiled = _median_call(
            lambda problem=problem: _compile(problem),
            repeats,
        )
        selected = compiled.state_preparation_strategy.require_selected()
        structured_values = selected.stored_values + compiled.oracle.integer_monomials
        generic_values = 3 * problem.model.joint_grid_points - 1
        rows.append(
            {
                "qubits_per_factor": count,
                "joint_points": problem.model.joint_grid_points,
                "marginal_points": problem.model.encoding.stored_marginal_points,
                "seconds": seconds,
                "structured_values": structured_values,
                "generic_values": generic_values,
                "storage_ratio": generic_values / structured_values,
                "oracle_error": compiled.validation.oracle_error,
                "maximum_loss_error": compiled.validation.maximum_loss_error,
            }
        )
    return rows


def _target_rows(repeats: int) -> list[dict[str, object]]:
    problem = _problem(1)
    compiled = qfin.compile(
        problem,
        backend="pennylane",
        target_error=0.05,
        max_arithmetic_qubits=24,
        max_affine_output_qubits=24,
        max_factorized_wires=40,
    )
    rows: list[dict[str, object]] = []
    for topology in ("all_to_all", "linear"):
        seconds, comparison = _median_call(
            lambda topology=topology: compiled.target_comparison(
                schedule=(0,),
                shots=1_000,
                target=topology,  # type: ignore[arg-type]
                max_joint_points=4,
                max_total_wires=40,
            ),
            repeats,
        )
        rows.append(
            {
                "topology": topology,
                "seconds": seconds,
                "structured_gates": comparison.structured.total_routed_gates_per_objective,
                "generic_gates": comparison.generic.total_routed_gates_per_objective,
                "structured_depth": comparison.structured.maximum_routed_depth,
                "generic_depth": comparison.generic.maximum_routed_depth,
                "structured_swaps": comparison.structured.circuits[0].routing_swaps,
                "generic_swaps": comparison.generic.circuits[0].routing_swaps,
                "gate_ratio": comparison.routed_gate_ratio,
            }
        )
    return rows


def _simulator_rows(repeats: int) -> list[dict[str, object]]:
    compiled = qfin.compile(
        _problem(1),
        backend="pennylane",
        target_error=0.05,
        max_arithmetic_qubits=24,
        max_affine_output_qubits=24,
        max_factorized_wires=40,
    )
    rows: list[dict[str, object]] = []
    for device in ("default.qubit", "lightning.qubit"):
        runtime = compiled.to_pennylane(device_name=device, max_total_wires=40)
        runtime.probability(0)
        seconds, probability = _median_call(
            lambda runtime=runtime: runtime.probability(0),
            repeats,
        )
        rows.append(
            {
                "device": device,
                "seconds": seconds,
                "probability": probability,
                "absolute_error": abs(probability - compiled.validation.oracle_probability),
            }
        )
    return rows


def _markdown(
    construction: list[dict[str, object]],
    targets: list[dict[str, object]],
    simulators: list[dict[str, object]],
    repeats: int,
) -> str:
    lines = [
        "# QFin 0.9 structured-oracle performance",
        "",
        "All values below were produced by `examples/structured_oracle_benchmark.py`.",
        "They are medians of end-to-end public compiler/runtime calls; no result is fabricated.",
        "",
        "## Environment",
        "",
        f"- CPU: {platform.processor() or platform.machine()}",
        f"- OS: {platform.platform()}",
        f"- Python: {platform.python_version()}",
        f"- QFin: {qfin.__version__}",
        f"- NumPy: {np.__version__}",
        f"- PennyLane: {_package_version('pennylane')}",
        f"- PennyLane-Lightning: {_package_version('pennylane-lightning')}",
        f"- Repeats: {repeats}",
        "",
        "## Streaming construction and validation",
        "",
        "| Qubits/factor | Joint points | Marginal points | Compile + validate (s) | "
        "Structured values | Generic values | Generic / structured | Oracle p error | "
        "Max loss error |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in construction:
        lines.append(
            f"| {row['qubits_per_factor']} | {row['joint_points']:,} | "
            f"{row['marginal_points']:,} | {row['seconds']:.6f} | "
            f"{row['structured_values']:,} | {row['generic_values']:,} | "
            f"{row['storage_ratio']:.2f}x | {row['oracle_error']:.3e} | "
            f"{row['maximum_loss_error']:.3e} |"
        )
    lines.extend(
        [
            "",
            "Validation streams every encoded point in bounded chunks. Memory is bounded, but "
            "validation time remains exponential in total factor qubits.",
            "",
            "## Portable target comparison (two 1-qubit factors, power 0)",
            "",
            "| Topology | Analysis (s) | Structured gates | Generic gates | Structured depth | "
            "Generic depth | Structured swaps | Generic swaps | Gate ratio |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in targets:
        lines.append(
            f"| {row['topology']} | {row['seconds']:.6f} | "
            f"{row['structured_gates']:,} | {row['generic_gates']:,} | "
            f"{row['structured_depth']:,} | {row['generic_depth']:,} | "
            f"{row['structured_swaps']:,} | {row['generic_swaps']:,} | "
            f"{row['gate_ratio']:.2f}x |"
        )
    lines.extend(
        [
            "",
            "The generic comparison is deliberately limited to four joint points. At this tiny "
            "size reversible arithmetic can use more gates than a lookup-style loader; the "
            "structured benefit is avoiding exponentially stored joint probability/payoff data, "
            "not a promise of lower gate count or quantum advantage.",
            "",
            "## Power-0 simulator execution",
            "",
            "| Device | Median (s) | Probability | Absolute difference |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in simulators:
        lines.append(
            f"| {row['device']} | {row['seconds']:.6f} | {row['probability']:.12f} | "
            f"{row['absolute_error']:.3e} |"
        )
    lines.extend(
        [
            "",
            "QFin constructs the finance-specific arithmetic. PennyLane-Lightning performs the "
            "compiled state-vector simulation. These timings are simulator measurements, not "
            "hardware or fault-tolerant runtime estimates.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    qubits = (1, 2, 3, 4, 5) if args.full else (1, 2, 3)
    report = _markdown(
        _construction_rows(qubits, args.repeats),
        _target_rows(args.repeats),
        _simulator_rows(args.repeats),
        args.repeats,
    )
    if args.output is None:
        print(report)
    else:
        args.output.write_text(report, encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
