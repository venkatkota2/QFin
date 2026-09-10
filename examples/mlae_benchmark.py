"""Benchmark and seeded coverage study for QFin's existing MLAE algorithm.

The benchmark compares the hardened deterministic coarse-to-fine estimator with
the previous 131,073-point dense search. The coverage study reports both the raw
Wilks likelihood-ratio region and QFin's finite-sample guarded region.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from importlib.metadata import PackageNotFoundError, version
from math import asin, sin, sqrt
from pathlib import Path

import numpy as np

import qfin
from qfin.algorithms import CircuitObservation, maximum_likelihood_amplitude_estimate
from qfin.algorithms.amplitude_estimation import _dense_reference_maximum_likelihood

_AMPLITUDES = (0.001, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 0.999)
_CONFIGURATIONS = (
    ((0,), 20),
    ((0, 1, 2, 4), 20),
    ((0, 1, 2, 4), 100),
    ((0, 1, 2, 4), 1_000),
    ((0, 2, 5), 100),
)


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def _cpu_description() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", maxsplit=1)[1].strip()
    return platform.processor() or platform.machine() or "not reported"


def _observations(
    amplitude: float,
    schedule: Sequence[int],
    shots: int,
    generator: np.random.Generator,
) -> tuple[CircuitObservation, ...]:
    theta = asin(sqrt(amplitude))
    return tuple(
        CircuitObservation(
            power=power,
            successes=int(
                generator.binomial(shots, sin((2 * power + 1) * theta) ** 2)
            ),
            shots=shots,
        )
        for power in schedule
    )


def _median_seconds(function: Callable[[], object], repeats: int) -> float:
    function()
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def _optimizer_benchmark(repeats: int) -> dict[str, object]:
    generator = np.random.default_rng(20_260_905)
    schedule = (0, 1, 2, 4)
    observations = tuple(
        _observations(amplitude, schedule, 1_000, generator)
        for amplitude in np.linspace(0.025, 0.975, 31)
    )

    def dense() -> tuple[tuple[float, float], ...]:
        return tuple(
            _dense_reference_maximum_likelihood(items) for items in observations
        )

    def hardened() -> tuple[object, ...]:
        return tuple(maximum_likelihood_amplitude_estimate(items) for items in observations)

    dense_results = dense()
    hardened_results = hardened()
    maximum_difference = max(
        abs(sin(dense_result[0]) ** 2 - hardened_result.amplitude)
        for dense_result, hardened_result in zip(
            dense_results, hardened_results, strict=True
        )
    )
    reference_seconds = _median_seconds(dense, repeats)
    qfin_seconds = _median_seconds(hardened, repeats)
    return {
        "benchmark": "MLAE repeated VaR-threshold fits",
        "problem_size": "31 thresholds; schedule=(0,1,2,4); 1000 shots/circuit",
        "reference_seconds": reference_seconds,
        "qfin_seconds": qfin_seconds,
        "speedup": reference_seconds / qfin_seconds,
        "max_difference": maximum_difference,
        "reference": "legacy 131073-point dense global grid",
        "qfin": "coarse-to-fine global search plus guarded confidence region",
    }


def _coverage_study(repetitions: int) -> list[dict[str, object]]:
    generator = np.random.default_rng(20_260_905)
    rows: list[dict[str, object]] = []
    for schedule, shots in _CONFIGURATIONS:
        print(f"Coverage: schedule={schedule}, shots={shots}", file=sys.stderr, flush=True)
        for amplitude in _AMPLITUDES:
            guarded_covered = 0
            likelihood_ratio_covered = 0
            for _ in range(repetitions):
                estimate = maximum_likelihood_amplitude_estimate(
                    _observations(amplitude, schedule, shots, generator)
                )
                guarded_covered += any(
                    lower <= amplitude <= upper
                    for lower, upper in estimate.confidence_regions_95
                )
                likelihood_ratio_covered += any(
                    lower <= amplitude <= upper
                    for lower, upper in estimate.likelihood_ratio_regions_95
                )
            rows.append(
                {
                    "amplitude": amplitude,
                    "schedule": list(schedule),
                    "shots_per_circuit": shots,
                    "repetitions": repetitions,
                    "guarded_coverage": guarded_covered / repetitions,
                    "raw_likelihood_ratio_coverage": (
                        likelihood_ratio_covered / repetitions
                    ),
                }
            )
    return rows


def _environment(repeats: int, coverage_repetitions: int) -> dict[str, object]:
    native = qfin.system_info()
    return {
        "os": platform.platform(),
        "architecture": platform.machine(),
        "cpu": _cpu_description(),
        "python": sys.version.split()[0],
        "qfin": qfin.__version__,
        "numpy": np.__version__,
        "scipy": _package_version("scipy"),
        "pennylane": _package_version("pennylane"),
        "pennylane_lightning": _package_version("pennylane-lightning"),
        "compiler": native["native_compiler"],
        "compiler_flags": "release defaults; see cpp/CMakeLists.txt",
        "thread_count": int(os.environ.get("OMP_NUM_THREADS", "1")),
        "timing_repetitions": repeats,
        "coverage_repetitions_per_case": coverage_repetitions,
        "seed": 20_260_905,
    }


def _markdown(report: dict[str, object]) -> str:
    environment = report["environment"]
    optimizer = report["optimizer"]
    coverage = report["coverage"]
    assert isinstance(environment, dict)
    assert isinstance(optimizer, dict)
    assert isinstance(coverage, list)
    lines = [
        "# QFin MLAE validation",
        "",
        "This validates QFin's existing maximum-likelihood amplitude-estimation "
        "algorithm; it does not introduce a new quantum algorithm or claim quantum advantage.",
        "",
        "## Environment",
        "",
        *[f"- {key.replace('_', ' ').title()}: {value}" for key, value in environment.items()],
        "",
        "## Optimizer",
        "",
        "| Workload | Dense reference (s) | Hardened QFin (s) | Speedup | "
        "Max amplitude difference |",
        "| --- | ---: | ---: | ---: | ---: |",
        (
            f"| {optimizer['problem_size']} | {optimizer['reference_seconds']:.6f} | "
            f"{optimizer['qfin_seconds']:.6f} | {optimizer['speedup']:.2f}x | "
            f"{optimizer['max_difference']:.3e} |"
        ),
        "",
        "The hardened timing includes the disjoint likelihood-ratio calculation and "
        "finite-sample confidence guard; the dense reference performs only the legacy "
        "global point search. Thus the comparison does not omit new uncertainty work.",
        "",
        "## Seeded empirical 95% coverage",
        "",
        "| Schedule | Shots | Amplitude | Guarded region | Raw LR region |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in coverage:
        assert isinstance(row, dict)
        lines.append(
            f"| `{tuple(row['schedule'])}` | {row['shots_per_circuit']} | "
            f"{row['amplitude']:.3f} | {row['guarded_coverage']:.3f} | "
            f"{row['raw_likelihood_ratio_coverage']:.3f} |"
        )
    lines.extend(
        [
            "",
            "The raw region uses the one-parameter Wilks likelihood-ratio cutoff and is "
            "reported as diagnostic metadata. The main interval is its union with a "
            "simultaneous exact Clopper-Pearson set, which is deliberately conservative "
            "near boundaries and at low shot counts. VaR/CVaR searches remain adaptive, "
            "so their aggregate intervals are not simultaneous-coverage guarantees.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--coverage-repetitions", type=int, default=200)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json-output", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.repeats < 1 or arguments.coverage_repetitions < 1:
        parser.error("repeats and coverage repetitions must be positive")

    environment = _environment(arguments.repeats, arguments.coverage_repetitions)
    optimizer = _optimizer_benchmark(arguments.repeats)
    optimizer["environment"] = environment
    report = {
        "environment": environment,
        "optimizer": optimizer,
        "coverage": _coverage_study(arguments.coverage_repetitions),
    }
    markdown = _markdown(report)
    if arguments.output is not None:
        arguments.output.write_text(markdown + "\n", encoding="utf-8")
    if arguments.json_output is not None:
        arguments.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
