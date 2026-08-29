"""Measure the QFin 0.6 multi-period ALM and life-scenario kernels.

Run ``python examples/alm_life_benchmark.py`` for a short pass. Add ``--full``
for the 10,000-scenario ALM case and a life book representing 100,000 policies.
Timings include public-API validation, buffer conversion, chunking, and the
Python/C++ boundary.
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
from pathlib import Path

import numpy as np

import qfin


@dataclass(frozen=True, slots=True)
class Measurement:
    workload: str
    reference_seconds: float
    native_seconds: float
    maximum_difference: float
    result_bytes: int
    working_set_bytes: int | None = None

    @property
    def speedup(self) -> float:
        return self.reference_seconds / self.native_seconds


def _seconds(function: Callable[[], object], repeats: int) -> float:
    function()
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def _timed_once(function: Callable[[], object]) -> float:
    start = time.perf_counter()
    function()
    return time.perf_counter() - start


def _cpu_description() -> str:
    if platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", maxsplit=1)[1].strip()
        except (OSError, IndexError):
            pass
    return platform.processor() or platform.machine() or "not reported"


def _economic_scenarios(
    curve: qfin.YieldCurve,
    scenario_count: int,
    periods: int,
    *,
    seed: int,
) -> qfin.EconomicScenarioSet:
    rng = np.random.default_rng(seed)
    short = rng.normal(0.0, 0.0075, size=(scenario_count, periods, 1))
    slope = rng.normal(0.0, 0.0025, size=(scenario_count, periods, 1))
    maturity_scale = curve.times / max(float(curve.times[-1]), 1.0)
    rate_shocks = short + slope * maturity_scale[None, None, :]
    return qfin.EconomicScenarioSet(
        rate_shocks,
        credit_spread_shocks=rng.normal(0.0, 0.003, size=(scenario_count, periods)),
        equity_returns=np.clip(rng.normal(0.06, 0.18, size=(scenario_count, periods)), -0.95, None),
        inflation_rates=np.clip(
            rng.normal(0.025, 0.012, size=(scenario_count, periods)), -0.50, None
        ),
        mortality_multipliers=np.exp(rng.normal(0.0, 0.08, size=(scenario_count, periods))),
        lapse_multipliers=np.exp(rng.normal(0.0, 0.12, size=(scenario_count, periods))),
        dependence_assumption="independent benchmark draws; not a calibrated ESG",
    )


def _alm_measurement(
    scenario_count: int,
    periods: int,
    repeats: int,
) -> Measurement:
    curve = qfin.YieldCurve([0, 1, 3, 7, 15, 30], [0.025, 0.027, 0.029, 0.032, 0.035, 0.037])
    bonds = [
        qfin.FixedRateBond(2 + index % 29, 0.02 + 0.001 * (index % 30), frequency=2)
        for index in range(100)
    ]
    model = qfin.ALMModel(
        qfin.AssetPortfolio(
            bonds,
            np.linspace(4.0, 8.0, len(bonds)),
            equity_value=20_000,
            cash_value=2_500,
        ),
        qfin.LiabilityPortfolio.from_arrays(
            np.arange(1, 31, dtype=np.float64),
            np.linspace(1_000, 2_500, 30),
            inflation_linkage=np.linspace(0.25, 1.0, 30),
        ),
        curve,
    )
    scenarios = _economic_scenarios(curve, scenario_count, periods, seed=17)
    strategy = qfin.RebalancingStrategy(
        target_equity_weight=0.30,
        rebalance_frequency=1,
        transaction_cost_rate=0.001,
    )
    reference = model.project_paths(
        scenarios, strategy=strategy, engine="numpy", scenario_chunk_size=256
    )
    native = model.project_paths(
        scenarios, strategy=strategy, engine="native", scenario_chunk_size=256
    )
    maximum_difference = max(
        float(np.max(np.abs(getattr(reference, name) - getattr(native, name)), initial=0.0))
        for name in (
            "asset_values",
            "cash_values",
            "liability_values",
            "surplus",
            "funding_ratio",
            "transaction_costs",
        )
    )
    result_bytes = sum(
        getattr(native, name).nbytes
        for name in (
            "asset_values",
            "bond_values",
            "cash_values",
            "equity_values",
            "liability_values",
            "liability_payments",
            "surplus",
            "funding_ratio",
            "transaction_costs",
        )
    )

    def numpy_call() -> qfin.ALMPathResult:
        return model.project_paths(
            scenarios, strategy=strategy, engine="numpy", scenario_chunk_size=256
        )

    def native_call() -> qfin.ALMPathResult:
        return model.project_paths(
            scenarios, strategy=strategy, engine="native", scenario_chunk_size=256
        )

    reference_seconds = (
        _timed_once(numpy_call) if scenario_count >= 10_000 else _seconds(numpy_call, repeats)
    )
    return Measurement(
        workload=f"100 bonds x {scenario_count:,} scenarios x {periods} years",
        reference_seconds=reference_seconds,
        native_seconds=_seconds(native_call, repeats),
        maximum_difference=maximum_difference,
        result_bytes=result_bytes,
    )


def _life_book(
    model_points: int, represented_policies: int, years: int
) -> qfin.PolicyModelPointSet:
    product_types = ("term_life", "participating_life", "universal_life", "annuity")
    policies: list[qfin.LifePolicy] = []
    for index in range(model_points):
        product = product_types[index % len(product_types)]
        policies.append(
            qfin.LifePolicy(
                age=35 + index % 35,
                sum_assured=0.0 if product == "annuity" else 75_000 + 500 * (index % 50),
                annual_premium=0.0 if product == "annuity" else 450 + index % 150,
                term=years,
                product_type=product,
                annual_benefit=8_000 if product == "annuity" else 0.0,
                account_value=12_500 if product == "universal_life" else 0.0,
                annual_charge=60 if product == "universal_life" else 0.0,
                bonus_rate=0.01 if product == "participating_life" else 0.0,
                disability_benefit=2_000,
                benefit_inflation_linkage=0.5,
            )
        )
    count = represented_policies / model_points
    return qfin.PolicyModelPointSet(policies, np.full(model_points, count))


def _life_measurement(
    model_points: int,
    represented_policies: int,
    scenario_count: int,
    years: int,
    repeats: int,
) -> Measurement:
    curve = qfin.YieldCurve([0, 1, 5, 10, 20, 40], [0.02, 0.022, 0.026, 0.029, 0.033, 0.035])
    ages = np.arange(20, 121, dtype=np.float64)
    mortality = qfin.MortalityTable(ages, np.minimum(0.00025 * np.exp(0.078 * (ages - 20)), 1.0))
    assumptions = qfin.ProjectionAssumptions(
        mortality,
        curve,
        lapse_rate=0.04,
        expense_per_policy=35,
        disability_rate=0.005,
        recovery_rate=0.20,
        disabled_mortality_multiplier=1.5,
        crediting_rate=0.025,
        expense_inflation_rate=0.025,
    )
    book = _life_book(model_points, represented_policies, years)
    scenarios = _economic_scenarios(curve, scenario_count, years, seed=23)
    reference = qfin.project_liability_scenarios(
        book,
        assumptions,
        scenarios,
        engine="numpy",
        scenario_chunk_size=64,
        policy_chunk_size=256,
    )
    native = qfin.project_liability_scenarios(
        book,
        assumptions,
        scenarios,
        engine="native",
        scenario_chunk_size=64,
        policy_chunk_size=256,
    )
    maximum_difference = max(
        float(np.max(np.abs(getattr(reference, name) - getattr(native, name)), initial=0.0))
        for name in (
            "present_values",
            "expected_premiums",
            "expected_benefits",
            "expected_expenses",
            "expected_surrenders",
        )
    )
    result_bytes = sum(
        getattr(native, name).nbytes
        for name in (
            "present_values",
            "expected_premiums",
            "expected_benefits",
            "expected_expenses",
            "expected_surrenders",
        )
    )

    def numpy_call() -> qfin.LifeScenarioResult:
        return qfin.project_liability_scenarios(
            book,
            assumptions,
            scenarios,
            engine="numpy",
            scenario_chunk_size=64,
            policy_chunk_size=256,
        )

    def native_call() -> qfin.LifeScenarioResult:
        return qfin.project_liability_scenarios(
            book,
            assumptions,
            scenarios,
            engine="native",
            scenario_chunk_size=64,
            policy_chunk_size=256,
        )

    workload = model_points * scenario_count * years
    reference_seconds = (
        _timed_once(numpy_call) if workload >= 1_000_000 else _seconds(numpy_call, repeats)
    )
    return Measurement(
        workload=(
            f"{model_points:,} {'model point' if model_points == 1 else 'model points'} / "
            f"{represented_policies:,} policies x {scenario_count:,} "
            f"{'scenario' if scenario_count == 1 else 'scenarios'} x {years} "
            f"{'year' if years == 1 else 'years'}"
        ),
        reference_seconds=reference_seconds,
        native_seconds=_seconds(native_call, repeats),
        maximum_difference=maximum_difference,
        result_bytes=result_bytes,
        working_set_bytes=native.working_set_estimate_bytes,
    )


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value / (1024 * 1024):.3f} MiB"


def _markdown(measurements: Sequence[Measurement], repeats: int) -> str:
    info = qfin.system_info()
    alm_measurements = [item for item in measurements if item.working_set_bytes is None]
    life_measurements = [item for item in measurements if item.working_set_bytes is not None]
    lines = [
        "# QFin 0.6 ALM and life performance",
        "",
        "These are measured end-to-end public-API timings. Validation, conversion, "
        "chunk dispatch, and result construction are included; no number is fabricated.",
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
        f"- QFin native: {info['native_backend']} ({info['native_cpp_standard']})",
        f"- C++ compiler: {info['native_compiler']}",
        f"- Native timings: median of {repeats} runs after one warm-up",
        "- Large NumPy references: one timed run",
        "- Native threading: deterministic single-threaded execution (no OpenMP)",
        "",
        "## Results",
        "",
        "| Workload | NumPy (s) | QFin C++ (s) | Speedup | Max difference | "
        "Returned arrays | Peak chunk estimate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in measurements:
        lines.append(
            f"| {item.workload} | {item.reference_seconds:.6f} | "
            f"{item.native_seconds:.6f} | {item.speedup:.2f}x | "
            f"{item.maximum_difference:.3e} | {_format_bytes(item.result_bytes)} | "
            f"{_format_bytes(item.working_set_bytes)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The ALM kernel returns scenario-by-period portfolio aggregates; it never "
            "returns a scenario-by-instrument cube. The life kernel returns five "
            "scenario-level aggregates and chunks both scenarios and model points, so "
            "represented policy counts do not multiply the result shape after grouping.",
            "",
            "`Max difference` is the largest absolute difference across the compared "
            "aggregate outputs. Tests enforce relative parity tolerances, so absolute "
            "differences must be interpreted against portfolio-scale monetary values.",
            "",
            "Multi-period ALM speedups range from "
            f"{min(item.speedup for item in alm_measurements):.2f}x to "
            f"{max(item.speedup for item in alm_measurements):.2f}x in this run. "
            "Because some rows remain close to 1.0x, the gain is not stable enough "
            'for automatic dispatch: `engine="auto"` conservatively stays on NumPy '
            "and native remains an explicit profiling override.",
            "",
            "The native life kernel is faster at the smallest measured non-empty "
            f"workload ({life_measurements[0].speedup:.2f}x) and the benefit grows for "
            "larger grouped books. Non-empty life and life-scenario workloads therefore "
            "select native execution automatically when the extension is available.",
            "",
            "The generated factors are synthetic independent benchmark draws, not a "
            "calibrated economic-scenario model. Timings are environment-specific and "
            "are not performance guarantees.",
            "",
            "Reproduce this report with:",
            "",
            "```bash",
            "python examples/alm_life_benchmark.py --full --repeats 3 \\",
            "  --output docs/alm-life-performance.md",
            "```",
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
        _alm_measurement(100, 10, args.repeats),
        _alm_measurement(1_000, 10, args.repeats),
    ]
    if args.full:
        measurements.append(_alm_measurement(10_000, 10, args.repeats))
    measurements.extend(
        [
            _life_measurement(1, 100, 1, 1, args.repeats),
            _life_measurement(1, 100, 1, 20, args.repeats),
            _life_measurement(10, 1_000, 10, 20, args.repeats),
            _life_measurement(25, 2_500, 20, 20, args.repeats),
            _life_measurement(100, 10_000, 250, 20, args.repeats),
        ]
    )
    if args.full:
        measurements.append(_life_measurement(1_000, 100_000, 100, 20, args.repeats))
    report = _markdown(measurements, args.repeats)
    if args.output is not None:
        args.output.write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
