"""Reproducible end-to-end QFin native benchmarks.

Run a short pass with ``python examples/native_benchmark.py``. Add ``--full``
for the requested 100k-instrument/policy and 10k-scenario cases. Timings include
Python public-API conversion and the Python/C++ boundary, not only kernel time.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from math import exp
from pathlib import Path

import numpy as np

import qfin


@dataclass(frozen=True, slots=True)
class Measurement:
    section: str
    problem: str
    reference_name: str
    accelerated_name: str
    reference_seconds: float
    native_seconds: float
    maximum_difference: float

    @property
    def speedup(self) -> float:
        return self.reference_seconds / self.native_seconds


def _seconds(
    function: Callable[[], object], repeats: int, *, warmup: bool = True
) -> float:
    if warmup:
        function()
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def _once_with_result(function: Callable[[], np.ndarray]) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    result = function()
    return result, time.perf_counter() - start


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def _cpu_description() -> str:
    if platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", maxsplit=1)[1].strip()
        except (OSError, IndexError):
            pass
    return platform.processor() or platform.machine() or "not reported"


def _python_bond_metrics(
    bonds: Sequence[qfin.FixedRateBond], curve: qfin.YieldCurve
) -> np.ndarray:
    output = np.empty((len(bonds), 4), dtype=np.float64)
    for instrument, bond in enumerate(bonds):
        times, amounts = bond.cashflows()
        price = first = second = price_down = price_up = 0.0
        for cashflow_time, amount in zip(times, amounts, strict=True):
            rate = curve.zero_rate(float(cashflow_time))
            present_value = float(amount) * exp(-rate * float(cashflow_time))
            price += present_value
            first += float(cashflow_time) * present_value
            second += float(cashflow_time) ** 2 * present_value
            price_down += present_value * exp(1.0e-4 * float(cashflow_time))
            price_up += present_value * exp(-1.0e-4 * float(cashflow_time))
        output[instrument] = (
            price,
            0.0 if price == 0 else first / price,
            0.0 if price == 0 else second / price,
            0.5 * (price_down - price_up),
        )
    return output


def _bond_measurements(full: bool, repeats: int) -> list[Measurement]:
    sizes = [1, 100, 10_000, 100_000] if full else [1, 100, 10_000]
    curve = qfin.YieldCurve(
        [0, 1, 5, 10, 30], [0.02, 0.025, 0.03, 0.035, 0.04]
    )
    measurements: list[Measurement] = []
    for size in sizes:
        problem = "1 bond" if size == 1 else f"{size:,} bonds"
        bonds = [
            qfin.FixedRateBond(
                1 + index % 30,
                0.01 + 0.001 * (index % 50),
                frequency=(1, 2, 4)[index % 3],
            )
            for index in range(size)
        ]
        numpy_result = qfin.price_bonds(bonds, curve, engine="numpy")
        native_result = qfin.price_bonds(bonds, curve, engine="native")
        numpy_seconds = _seconds(
            lambda bonds=bonds: qfin.price_bonds(bonds, curve, engine="numpy"),
            repeats,
        )
        native_seconds = _seconds(
            lambda bonds=bonds: qfin.price_bonds(bonds, curve, engine="native"),
            repeats,
        )
        difference = float(
            np.max(
                np.abs(numpy_result.dirty_prices - native_result.dirty_prices),
                initial=0.0,
            )
        )
        measurements.append(
            Measurement(
                "Fixed-income public API",
                problem,
                "NumPy",
                "QFin C++",
                numpy_seconds,
                native_seconds,
                difference,
            )
        )
        if size <= (100_000 if full else 100):
            python_result, python_seconds = _once_with_result(
                lambda bonds=bonds: _python_bond_metrics(bonds, curve)
            )
            native_metrics = np.column_stack(
                (
                    native_result.dirty_prices,
                    native_result.macaulay_duration,
                    native_result.convexity,
                    native_result.dv01,
                )
            )
            measurements.append(
                Measurement(
                    "Fixed-income pure Python",
                    problem,
                    "Python",
                    "QFin C++",
                    python_seconds,
                    native_seconds,
                    float(np.max(np.abs(python_result - native_metrics), initial=0.0)),
                )
            )
    return measurements


def _life_measurements(full: bool, repeats: int) -> list[Measurement]:
    sizes = [1_000, 10_000, 100_000] if full else [1_000, 10_000]
    ages = np.arange(20, 121, dtype=np.float64)
    mortality = qfin.MortalityTable(
        ages, np.minimum(0.0002 * np.exp(0.075 * (ages - 20)), 1.0)
    )
    curve = qfin.YieldCurve(
        [0, 1, 5, 10, 30, 50], [0.02, 0.022, 0.027, 0.03, 0.035, 0.037]
    )
    assumptions = qfin.ProjectionAssumptions(
        mortality, curve, lapse_rate=0.04, expense_per_policy=25
    )
    measurements: list[Measurement] = []
    for size in sizes:
        policies = [
            qfin.LifePolicy(
                30 + index % 40,
                50_000 + 1_000 * (index % 20),
                200 + index % 100,
                10 + index % 20,
            )
            for index in range(size)
        ]
        reference: qfin.LifeProjectionResult
        if size >= 10_000:
            start = time.perf_counter()
            reference = qfin.project_liabilities(policies, assumptions, engine="numpy")
            reference_seconds = time.perf_counter() - start
        else:
            reference = qfin.project_liabilities(policies, assumptions, engine="numpy")
            reference_seconds = _seconds(
                lambda policies=policies: qfin.project_liabilities(
                    policies, assumptions, engine="numpy"
                ),
                repeats,
            )
        native = qfin.project_liabilities(policies, assumptions, engine="native")
        native_seconds = _seconds(
            lambda policies=policies: qfin.project_liabilities(
                policies, assumptions, engine="native"
            ),
            repeats,
        )
        measurements.append(
            Measurement(
                "Life projection",
                f"{size:,} policies",
                "Python/NumPy",
                "QFin C++",
                reference_seconds,
                native_seconds,
                abs(reference.present_value - native.present_value),
            )
        )
    return measurements


def _yield_measurements(full: bool, repeats: int) -> list[Measurement]:
    sizes = [100, 10_000] if full else [100]
    measurements: list[Measurement] = []
    for size in sizes:
        bonds = [
            qfin.FixedRateBond(
                1 + index % 30,
                0.01 + 0.001 * (index % 50),
                frequency=(1, 2, 4)[index % 3],
            )
            for index in range(size)
        ]
        yields = np.linspace(-0.01, 0.15, size)
        prices = qfin.price_bonds_from_yield(
            bonds, yields, engine="native"
        ).dirty_prices
        reference = qfin.yield_from_prices(bonds, prices, engine="numpy")
        native = qfin.yield_from_prices(bonds, prices, engine="native")
        if not np.all(reference.converged) or not np.all(native.converged):
            raise RuntimeError("yield benchmark did not converge")
        measurements.append(
            Measurement(
                "Yield solving",
                f"{size:,} bonds",
                "Python/NumPy",
                "QFin C++",
                _seconds(
                    lambda bonds=bonds, prices=prices: qfin.yield_from_prices(
                        bonds, prices, engine="numpy"
                    ),
                    repeats,
                ),
                _seconds(
                    lambda bonds=bonds, prices=prices: qfin.yield_from_prices(
                        bonds, prices, engine="native"
                    ),
                    repeats,
                ),
                float(np.max(np.abs(reference.yields - native.yields), initial=0.0)),
            )
        )
    return measurements


def _alm_measurements(full: bool, repeats: int) -> list[Measurement]:
    sizes = [100, 1_000, 10_000] if full else [100, 1_000]
    curve = qfin.YieldCurve(
        [0, 1, 5, 10, 30], [0.02, 0.025, 0.03, 0.035, 0.04]
    )
    measurements: list[Measurement] = []
    for size in sizes:
        bonds = [qfin.FixedRateBond(1 + index % 30, 0.04) for index in range(size)]
        model = qfin.ALMModel(
            qfin.AssetPortfolio(bonds, np.linspace(1.0, 2.0, size)),
            qfin.LiabilityPortfolio.from_arrays(
                np.arange(1, 31, dtype=np.float64), np.linspace(100, 500, 30)
            ),
            curve,
        )
        reference = model.evaluate(engine="numpy")
        native = model.evaluate(engine="native")
        reference_values = np.array(
            [
                reference.asset_pv,
                reference.liability_pv,
                reference.surplus,
                reference.asset_duration,
                reference.liability_duration,
                reference.asset_convexity,
                reference.liability_convexity,
            ]
        )
        native_values = np.array(
            [
                native.asset_pv,
                native.liability_pv,
                native.surplus,
                native.asset_duration,
                native.liability_duration,
                native.asset_convexity,
                native.liability_convexity,
            ]
        )
        measurements.append(
            Measurement(
                "ALM base valuation",
                f"{size:,} assets",
                "NumPy",
                "QFin C++",
                _seconds(lambda model=model: model.evaluate(engine="numpy"), repeats),
                _seconds(lambda model=model: model.evaluate(engine="native"), repeats),
                float(np.max(np.abs(reference_values - native_values), initial=0.0)),
            )
        )
    return measurements


def _risk_measurements(full: bool, repeats: int) -> list[Measurement]:
    sizes = [1_000, 10_000, 100_000] if full else [1_000, 10_000]
    rng = np.random.default_rng(7)
    measurements: list[Measurement] = []
    for size in sizes:
        distribution = qfin.LossDistribution(rng.normal(size=size), rng.random(size))
        reference = qfin.aggregate_risk(distribution, confidence=0.995, engine="numpy")
        native = qfin.aggregate_risk(distribution, confidence=0.995, engine="native")
        reference_values = np.array(
            [
                reference.mean,
                reference.standard_deviation,
                reference.minimum,
                reference.maximum,
                reference.var,
                reference.cvar,
            ]
        )
        native_values = np.array(
            [
                native.mean,
                native.standard_deviation,
                native.minimum,
                native.maximum,
                native.var,
                native.cvar,
            ]
        )
        measurements.append(
            Measurement(
                "Risk aggregation",
                f"{size:,} weighted losses",
                "NumPy",
                "QFin C++",
                _seconds(
                    lambda distribution=distribution: qfin.aggregate_risk(
                        distribution, confidence=0.995, engine="numpy"
                    ),
                    repeats,
                ),
                _seconds(
                    lambda distribution=distribution: qfin.aggregate_risk(
                        distribution, confidence=0.995, engine="native"
                    ),
                    repeats,
                ),
                float(np.max(np.abs(reference_values - native_values), initial=0.0)),
            )
        )
    return measurements


def _scenario_measurements(full: bool, repeats: int) -> list[Measurement]:
    sizes = [1_000, 10_000] if full else [1_000]
    curve = qfin.YieldCurve(
        [0, 1, 5, 10, 30], [0.02, 0.025, 0.03, 0.035, 0.04]
    )
    bonds = [qfin.FixedRateBond(1 + index % 20, 0.04) for index in range(1_000)]
    model = qfin.ALMModel(
        qfin.AssetPortfolio(bonds, np.ones(len(bonds))),
        qfin.LiabilityPortfolio.from_arrays(
            np.arange(1, 31, dtype=float), np.linspace(100, 500, 30)
        ),
        curve,
    )
    measurements: list[Measurement] = []
    for size in sizes:
        scenarios = qfin.RateScenarioSet.parallel(
            curve, np.linspace(-0.02, 0.02, size)
        )
        reference = model.run_scenarios(scenarios, engine="numpy")
        native = model.run_scenarios(scenarios, engine="native")
        reference_seconds = _seconds(
            lambda scenarios=scenarios: model.run_scenarios(scenarios, engine="numpy"),
            repeats,
        )
        native_seconds = _seconds(
            lambda scenarios=scenarios: model.run_scenarios(scenarios, engine="native"),
            repeats,
        )
        measurements.append(
            Measurement(
                "ALM scenarios",
                f"1,000 bonds x {size:,} scenarios",
                "NumPy",
                "QFin C++",
                reference_seconds,
                native_seconds,
                float(np.max(np.abs(reference.surplus - native.surplus))),
            )
        )
    return measurements


def _quantum_measurements(repeats: int) -> list[Measurement]:
    info = qfin.system_info()
    if not (info["pennylane"] and info["pennylane_lightning"]):
        return []
    market = qfin.BlackScholes(spot=100, rate=0.04, volatility=0.20)
    option = qfin.EuropeanCall(strike=105, maturity=1.0)
    compiled = qfin.compile(
        option,
        market,
        target_error=1.0,
        min_qubits=5,
        max_qubits=5,
    )
    if not isinstance(compiled, qfin.CompiledPricingModel):
        raise RuntimeError("option benchmark did not produce a pricing model")
    default_backend = compiled.to_pennylane(device_name="default.qubit")
    lightning_backend = compiled.to_pennylane(device_name="lightning.qubit")
    default_probability = default_backend.probability(0)
    lightning_probability = lightning_backend.probability(0)
    quantum_repeats = max(repeats, 5)
    return [
        Measurement(
            "Quantum simulation",
            "5 data qubits, power 0",
            "default.qubit",
            "PennyLane-Lightning C++",
            _seconds(lambda: default_backend.probability(0), quantum_repeats),
            _seconds(lambda: lightning_backend.probability(0), quantum_repeats),
            abs(default_probability - lightning_probability),
        )
    ]


def _markdown(measurements: list[Measurement], repeats: int) -> str:
    info = qfin.system_info()
    fixed_public = {
        measurement.problem: measurement
        for measurement in measurements
        if measurement.section == "Fixed-income public API"
    }
    life = [
        measurement
        for measurement in measurements
        if measurement.section == "Life projection"
    ]
    scenarios = [
        measurement
        for measurement in measurements
        if measurement.section == "ALM scenarios"
    ]
    risk = [
        measurement
        for measurement in measurements
        if measurement.section == "Risk aggregation"
    ]
    lines = [
        "# QFin native performance",
        "",
        "These are measured end-to-end public-API timings; no result is fabricated. "
        "Object-to-buffer conversion and Python/C++ boundary costs are included.",
        "",
        "## Environment",
        "",
        f"- Measurement date (UTC): {datetime.now(UTC).date().isoformat()}",
        f"- OS: {platform.platform()}",
        f"- CPU: {_cpu_description()}",
        f"- Architecture: {platform.machine()}",
        f"- Python: {sys.version.split()[0]}",
        f"- QFin: {qfin.__version__}",
        f"- NumPy: {np.__version__}",
        f"- PennyLane: {_package_version('pennylane')}",
        f"- PennyLane-Lightning: {_package_version('pennylane-lightning')}",
        f"- QFin native: {info['native_backend']} ({info['native_cpp_standard']})",
        f"- C++ compiler: {info['native_compiler']}",
        f"- Requested repetitions: median of {repeats}",
        "- Large Python/NumPy references: one timed run",
        "- Quantum device rows: median of at least 5 runs",
        "- Native threading: deterministic single-threaded execution (no OpenMP)",
        "",
        "## Results",
        "",
        "| Workload | Problem | Reference | Accelerated | Reference (s) | "
        "Accelerated (s) | Speedup | Max difference |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in measurements:
        lines.append(
            f"| {item.section} | {item.problem} | {item.reference_name} | "
            f"{item.accelerated_name} | "
            f"{item.reference_seconds:.6f} | {item.native_seconds:.6f} | "
            f"{item.speedup:.2f}x | {item.maximum_difference:.3e} |"
        )
    lines.extend(
        [
            "",
            "## Numerical acceptance",
            "",
            "`Max difference` is the largest absolute difference over the compared "
            "outputs. Parity tests use relative tolerance `1e-13` for fixed-income, "
            "ALM, scenario, and life outputs (`1e-11` for finite-difference DV01). "
            "Weighted expected shortfall uses an absolute `1e-10` large-batch bound; "
            "analytical cases and all other risk statistics use tighter tolerances.",
            "",
            "## Interpretation",
            "",
            "Native dispatch is useful only when the eliminated inner loop exceeds "
            "buffer-conversion cost. The benchmark deliberately exposes crossover cases; "
            "a speedup below 1.0x means the reference was faster for that measured size.",
        ]
    )
    hundred = fixed_public.get("100 bonds")
    ten_thousand = fixed_public.get("10,000 bonds")
    if hundred is not None and ten_thousand is not None:
        if hundred.speedup < 1.0 <= ten_thousand.speedup:
            lines.append(
                "The measured NumPy/native fixed-income crossover is between 100 and "
                "10,000 bonds for this mixed-maturity workload."
            )
        else:
            lines.append(
                "The 100- and 10,000-bond rows do not bracket a stable NumPy/native "
                "crossover in this run; use the displayed measurements directly."
            )
    if life:
        lines.append(
            "Life projection removes the policy-by-year Python loop; observed speedups "
            f"range from {min(item.speedup for item in life):.2f}x to "
            f"{max(item.speedup for item in life):.2f}x."
        )
    if scenarios:
        lines.append(
            "Chunked ALM scenario valuation observed speedups from "
            f"{min(item.speedup for item in scenarios):.2f}x to "
            f"{max(item.speedup for item in scenarios):.2f}x."
        )
    if risk:
        largest_risk_case = risk[-1]
        if largest_risk_case.speedup < 1.0:
            lines.append(
                "Native tail-risk aggregation was slower in the largest measured case; "
                "automatic risk dispatch therefore remains on NumPy."
            )
        else:
            lines.append(
                "Tail-risk timings remain close enough that automatic risk dispatch "
                "stays on NumPy pending a stable crossover across environments."
            )
    lines.extend(
        [
            "The quantum row compares PennyLane devices only; Lightning C++ remains the "
            "quantum simulator and is independent of QFin's finance C++ extension.",
            "SciPy has no separate row because these reference cases use vectorized "
            "NumPy or an explicit batch bisection; no SciPy primitive is used.",
            "Timings are environment-specific measurements, not performance guarantees.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if not qfin.system_info()["native_extension"]:
        parser.error("the QFin native extension must be installed")
    measurements = [
        *_bond_measurements(args.full, args.repeats),
        *_yield_measurements(args.full, args.repeats),
        *_alm_measurements(args.full, args.repeats),
        *_life_measurements(args.full, args.repeats),
        *_scenario_measurements(args.full, args.repeats),
        *_risk_measurements(args.full, args.repeats),
        *_quantum_measurements(args.repeats),
    ]
    report = _markdown(measurements, args.repeats)
    if args.output is not None:
        args.output.write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
