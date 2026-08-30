"""Reproduce QFin 0.8 representation and optimization measurements."""

from __future__ import annotations

import argparse
import platform
import statistics
import time
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import TypeVar

import numpy as np

import qfin

ResultT = TypeVar("ResultT")


def _median_call(function: Callable[[], ResultT], repeats: int) -> tuple[float, ResultT]:
    function()
    timings: list[float] = []
    result: ResultT | None = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = function()
        timings.append(time.perf_counter() - started)
    assert result is not None
    return statistics.median(timings), result


def _cpu_name() -> str:
    path = Path("/proc/cpuinfo")
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _factor_encoding(factors: int, qubits: int) -> qfin.FactorizedDistributionEncoding:
    return qfin.encode_independent_factors(
        tuple(qfin.Normal() for _ in range(factors)),
        qubits_per_factor=qubits,
        method="probability",
    )


def _flattened_angles(
    encoding: qfin.FactorizedDistributionEncoding,
) -> qfin.ProbabilityTreePreparation:
    probabilities = encoding.materialize(max_points=65_536).probabilities
    return qfin.ProbabilityTreePreparation.from_probabilities(probabilities)


def _optimization_problem(assets: int) -> qfin.MeanVarianceProblem:
    generator = np.random.default_rng(17 + assets)
    factors = generator.normal(size=(assets, min(8, assets)))
    covariance = factors @ factors.T / factors.shape[1]
    covariance *= 0.015 / float(np.mean(np.diag(covariance)))
    covariance += np.diag(np.linspace(0.015, 0.045, assets))
    expected_returns = np.linspace(0.035, 0.12, assets)
    return qfin.MeanVarianceProblem(
        expected_returns,
        covariance,
        risk_aversion=3.0,
        upper_bounds=np.full(assets, min(0.35, 5.0 / assets)),
    )


def benchmark(repeats: int) -> str:
    representation_rows: list[tuple[int, int, int, int, int, int, float, float | None]] = []
    for factors, qubits in ((2, 3), (3, 4), (4, 5)):
        factorized_time, encoding = _median_call(
            partial(_factor_encoding, factors, qubits),
            repeats,
        )
        report = qfin.compare_state_preparation_strategies(encoding)
        selected = report.require_selected()
        flattened = next(
            candidate
            for candidate in report.candidates
            if candidate.strategy == "flattened_probability_tree"
        )
        flattened_time: float | None = None
        if encoding.joint_grid_points <= 65_536:
            flattened_time, _ = _median_call(
                partial(_flattened_angles, encoding),
                repeats,
            )
        representation_rows.append(
            (
                factors,
                qubits,
                encoding.joint_grid_points,
                encoding.stored_marginal_points,
                selected.classical_parameters,
                flattened.classical_parameters,
                factorized_time,
                flattened_time,
            )
        )

    optimization_rows: list[tuple[int, float, float, float, int]] = []
    for assets in (10, 25, 50):
        problem = _optimization_problem(assets)
        elapsed, result = _median_call(problem.solve, repeats)
        optimization_rows.append(
            (
                assets,
                elapsed,
                abs(result.budget_residual),
                result.utility_improvement,
                result.iterations,
            )
        )

    covariance = _optimization_problem(50).covariance
    feasibility_time, feasibility = _median_call(
        lambda: qfin.analyze_block_encoding(covariance),
        repeats,
    )
    lines = [
        "# QFin 0.8 scalable-representation performance",
        "",
        "All timings below come from public QFin APIs and are medians of "
        f"{repeats} runs after one warm-up.",
        "",
        "## Environment",
        "",
        f"- OS: {platform.platform()}",
        f"- CPU: {_cpu_name()}",
        f"- Python: {platform.python_version()}",
        f"- QFin: {qfin.__version__}",
        f"- NumPy: {np.__version__}",
        "- Factor marginals: standard normal probability encodings",
        "- Optimization solver: SciPy SLSQP with analytical gradient",
        "",
        "## Factorized construction",
        "",
        "| Factors | Qubits/factor | Joint points | Stored marginal points | "
        "Factorized angles | Flattened angles | Factorized build (s) | "
        "Flattened build (s) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in representation_rows:
        factors, qubits, joint, stored, factor_angles, flat_angles, factor_time, flat_time = row
        flat_label = "not materialized" if flat_time is None else f"{flat_time:.6f}"
        lines.append(
            f"| {factors} | {qubits} | {joint:,} | {stored:,} | "
            f"{factor_angles:,} | {flat_angles:,} | {factor_time:.6f} | {flat_label} |"
        )
    lines.extend(
        [
            "",
            "The largest flattened case is deliberately not allocated. Its angle and "
            "memory counts are analytical properties of the represented dimensions, "
            "not fabricated timings. Factorized construction stores and prepares each "
            "marginal independently.",
            "",
            "## Classical mean-variance baseline",
            "",
            "| Assets | Solve time (s) | Budget residual | Utility improvement vs "
            "feasible start | Iterations |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    lines.extend(
        f"| {assets} | {elapsed:.6f} | {residual:.3e} | {improvement:.6e} | {iterations} |"
        for assets, elapsed, residual, improvement, iterations in optimization_rows
    )
    lines.extend(
        [
            "",
            "## Block-encoding feasibility analysis",
            "",
            f"A 50x50 covariance analysis took {feasibility_time:.6f} s. "
            f"Hermitian={feasibility.hermitian}, PSD={feasibility.positive_semidefinite}, "
            f"condition number={feasibility.condition_number:.6g}.",
            "",
            "QFin does not construct a block-encoding oracle or execute QSVT. This timing "
            "covers classical feasibility metadata only.",
            "",
            "## Interpretation",
            "",
            "The factorized loader removes joint probability-table construction where "
            "independence or latent-factor structure permits. It does not remove the cost "
            "of a general multivariate payoff oracle, and it is not evidence of quantum "
            "advantage. Optimization remains classical because QFin has no validated "
            "quantum portfolio optimizer.",
            "",
            "Reproduce with:",
            "",
            "```bash",
            "python examples/scalable_representation_benchmark.py --repeats 5 \\",
            "  --output docs/scalable-representation-performance.md",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    if arguments.repeats < 1:
        raise ValueError("repeats must be positive")
    report = benchmark(arguments.repeats)
    if arguments.output is not None:
        arguments.output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
