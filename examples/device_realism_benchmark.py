"""Reproduce QFin 0.7 backend, routing, mitigation, and export evidence."""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
import statistics
import time
from collections.abc import Callable, Sequence
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


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _cpu_name() -> str:
    path = Path("/proc/cpuinfo")
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _build_model() -> qfin.CompiledPricingModel:
    model = qfin.compile(
        qfin.EuropeanCall(strike=105, maturity=1.0),
        qfin.BlackScholes(spot=100, rate=0.04, volatility=0.20),
        target_error=1.0,
        min_qubits=3,
        max_qubits=3,
    )
    if not isinstance(model, qfin.CompiledPricingModel):
        raise RuntimeError("option compilation did not return a pricing model")
    return model


def benchmark(repeats: int) -> str:
    model = _build_model()
    default = model.to_pennylane(device_name="default.qubit")
    lightning = model.to_pennylane(device_name="lightning.qubit")
    backend_rows: list[tuple[int, float, float, float, float]] = []
    for power in (0, 1):
        default_time, default_probability = _median_call(
            lambda power=power: default.probability(power), repeats
        )
        lightning_time, lightning_probability = _median_call(
            lambda power=power: lightning.probability(power), repeats
        )
        backend_rows.append(
            (
                power,
                default_time,
                lightning_time,
                default_time / lightning_time,
                abs(default_probability - lightning_probability),
            )
        )

    target_rows: list[tuple[str, float, int, int, int, int]] = []
    for target in ("all_to_all", "linear"):
        elapsed, report = _median_call(
            lambda target=target: model.device_resources(
                schedule=(0, 1), shots=1_000, target=target
            ),
            repeats,
        )
        target_rows.append(
            (
                target,
                elapsed,
                report.total_routed_gates_per_objective,
                report.maximum_routed_depth,
                sum(circuit.routing_swaps for circuit in report.circuits),
                report.maximum_two_qubit_gates,
            )
        )

    noise_model = qfin.NoiseModel(0.001, 0.002)
    noise_time, noise = _median_call(lambda: model.noise_analysis(noise_model, power=0), repeats)
    qasm_time, qasm = _median_call(lambda: model.to_openqasm(power=1, target="linear"), repeats)
    qiskit_time: float | None = None
    qiskit_operations: int | None = None
    if qfin.system_info()["qiskit"]:
        qiskit_time, circuit = _median_call(
            lambda: model.to_qiskit(power=1, target="linear"), repeats
        )
        qiskit_operations = int(circuit.size())

    lines = [
        "# QFin 0.7 device-realism performance",
        "",
        "All values below were measured by the public QFin API. Timings are medians "
        f"of {repeats} runs after one warm-up.",
        "",
        "## Environment",
        "",
        f"- OS: {platform.platform()}",
        f"- CPU: {_cpu_name()}",
        f"- Python: {platform.python_version()}",
        f"- QFin: {qfin.__version__}",
        f"- NumPy: {np.__version__}",
        f"- PennyLane: {_version('pennylane')}",
        f"- PennyLane-Lightning: {_version('pennylane-lightning')}",
        f"- Qiskit: {_version('qiskit')}",
        "- Circuit: compressed three-data-qubit European-call objective",
        "- Native gate set: RX, RY, RZ, CNOT",
        "",
        "## Ideal simulator execution",
        "",
        "| Grover power | default.qubit (s) | lightning.qubit (s) | "
        "Lightning speedup | Probability difference |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {power} | {default_time:.6f} | {lightning_time:.6f} | "
        f"{speedup:.2f}x | {difference:.3e} |"
        for power, default_time, lightning_time, speedup, difference in backend_rows
    )
    lines.extend(
        [
            "",
            "## Decomposition and routing",
            "",
            "| Target | Analysis time (s) | Routed gates, powers 0+1 | Max depth | "
            "Inserted SWAPs | Max two-qubit gates |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    lines.extend(
        f"| {target} | {elapsed:.6f} | {gates} | {depth} | {swaps} | {two_qubit} |"
        for target, elapsed, gates, depth, swaps, two_qubit in target_rows
    )
    lines.extend(
        [
            "",
            "The targets are synthetic research topologies, not vendor devices. Analysis "
            "time is compiler preprocessing, not quantum execution time.",
            "",
            "## Synthetic noise and mitigation",
            "",
            "| Experiment | Probability | Absolute error vs ideal |",
            "| --- | ---: | ---: |",
            f"| Ideal | {noise.ideal_probability:.9f} | 0.000e+00 |",
            f"| Local noise | {noise.noisy_probability:.9f} | {noise.noisy_absolute_error:.3e} |",
            f"| Linear ZNE | {noise.mitigated_probability:.9f} | "
            f"{noise.mitigated_absolute_error:.3e} |",
            "",
            f"End-to-end three-scale noise-analysis time: {noise_time:.6f} s.",
            "",
            "This uses analytic `default.mixed`, local per-wire depolarizing probability "
            "0.001 after each gate, readout bit-flip probability 0.002, global folding "
            "at 1x/3x/5x, and first-order extrapolation. It is not a hardware prediction.",
            "",
            "## Interoperability",
            "",
            f"- Linear-target OpenQASM export: {qasm_time:.6f} s, "
            f"{qasm.resources.routed_gates} gates, SHA-256 `{qasm.sha256}`.",
        ]
    )
    if qiskit_time is not None and qiskit_operations is not None:
        lines.append(
            f"- Qiskit parse: {qiskit_time:.6f} s, {qiskit_operations} operations "
            "including terminal measurements."
        )
    else:
        lines.append("- Qiskit parse: not measured because the optional extra was absent.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "These measurements establish numerical parity and expose topology/noise "
            "costs. They do not establish quantum advantage, hardware feasibility, or "
            "a stable simulator speedup at every small circuit size.",
            "",
            "Reproduce with:",
            "",
            "```bash",
            "python examples/device_realism_benchmark.py --repeats 5 \\",
            "  --output docs/device-realism-performance.md",
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
    args = _parser().parse_args(argv)
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    report = benchmark(args.repeats)
    if args.output is not None:
        args.output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
