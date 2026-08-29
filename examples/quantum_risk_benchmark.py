"""Reproduce QFin's small quantum-risk simulator benchmark table."""

from __future__ import annotations

import argparse
import platform
from importlib.metadata import version
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import pennylane

import qfin


def _losses() -> qfin.LossDistribution:
    generator = np.random.default_rng(19)
    values = np.concatenate((generator.normal(0.0, 1.0, 28), [2.5, 3.0, 4.0, 6.0]))
    return qfin.LossDistribution(values)


def _models() -> list[tuple[str, qfin.CompiledRiskModel]]:
    losses = _losses()
    problems = [
        ("Tail probability", qfin.TailProbability(losses, threshold=1.0)),
        ("VaR", qfin.VaR(losses, confidence=0.80)),
        ("CVaR", qfin.CVaR(losses, confidence=0.80)),
    ]
    result: list[tuple[str, qfin.CompiledRiskModel]] = []
    for name, problem in problems:
        compiled = qfin.compile(
            problem,
            target_error=0.10,
            min_qubits=5,
            max_qubits=5,
        )
        assert isinstance(compiled, qfin.CompiledRiskModel)
        result.append((name, compiled))
    return result


def _cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", maxsplit=1)[1].strip()
    return platform.processor() or platform.machine()


def _benchmark(repeats: int, shots: int) -> str:
    devices = ["default.qubit"]
    if qfin.system_info()["pennylane_lightning"]:
        devices.append("lightning.qubit")
    rows: list[tuple[str, str, float, float, float, int, float]] = []
    for device in devices:
        for name, model in _models():
            timings: list[float] = []
            result = None
            for _ in range(repeats):
                started = perf_counter()
                result = model.run_quantum(
                    shots=shots,
                    schedule=(0, 1, 2),
                    seed=23,
                    likelihood_grid_size=32_769,
                    device_name=device,
                )
                timings.append(perf_counter() - started)
            assert result is not None
            state = model.to_pennylane(
                device_name=device
            ).distribution_probabilities()
            state_error = float(
                np.max(np.abs(state - model.representation.probabilities))
            )
            rows.append(
                (
                    name,
                    device,
                    median(timings),
                    result.value,
                    result.classical_value,
                    result.resources.total_circuits,
                    state_error,
                )
            )

    lines = [
        "# QFin Quantum-Risk Simulator Performance",
        "",
        "These are measured simulator timings, not a hardware or quantum-advantage claim.",
        "",
        "## Environment",
        "",
        f"- CPU/platform: {_cpu_name()}",
        f"- OS: {platform.platform()}",
        f"- Python: {platform.python_version()}",
        f"- QFin: {qfin.__version__}",
        f"- NumPy: {np.__version__}",
        f"- PennyLane: {pennylane.__version__}",
        f"- PennyLane-Lightning: {version('pennylane-lightning')}",
        f"- Repeats: {repeats} (median reported)",
        f"- Shots per circuit: {shots}",
        "- MLAE schedule: `(0, 1, 2)`",
        "- Data qubits: 5 (32 encoded grid points)",
        "",
        "## Results",
        "",
        "| Problem | Device | Wall time (s) | Speedup | Quantum estimate | "
        "Classical reference | Circuits | Max state error |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    default_times = {
        name: timing
        for name, device, timing, *_remaining in rows
        if device == "default.qubit"
    }
    for name, device, timing, value, reference, circuits, state_error in rows:
        speedup = default_times[name] / timing
        lines.append(
            f"| {name} | {device} | {timing:.6f} | {speedup:.2f}x | {value:.8f} | "
            f"{reference:.8f} | {circuits} | {state_error:.3e} |"
        )
    lines.extend(
        [
            "",
            "The VaR/CVaR workflow uses hybrid binary search. Its circuit count depends on",
            "the number of occupied encoded loss points. CVaR adds one normalized tail-excess",
            "objective after threshold search.",
            "",
            "The generic empirical probability tree and objective multiplexers require "
            "`O(2**data_qubits)` rotations. The benchmark demonstrates correctness and backend "
            "integration; it does not establish an asymptotic speedup.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--shots", type=int, default=1_000)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.repeats < 1 or arguments.shots < 1:
        parser.error("repeats and shots must be positive")
    report = _benchmark(arguments.repeats, arguments.shots)
    if arguments.output is None:
        print(report)
    else:
        arguments.output.write_text(report, encoding="utf-8")
        print(f"Wrote {arguments.output}")


if __name__ == "__main__":
    main()
