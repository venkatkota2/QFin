"""Emit the checked-in QFin v0.3 numerical demonstration as Markdown."""

from __future__ import annotations

import qfin

CASES = [
    (
        "ATM call",
        qfin.EuropeanCall(strike=100.0, maturity=1.0),
        qfin.BlackScholes(spot=100.0, rate=0.03, volatility=0.20),
    ),
    (
        "ATM put",
        qfin.EuropeanPut(strike=100.0, maturity=1.0),
        qfin.BlackScholes(spot=100.0, rate=0.03, volatility=0.20),
    ),
    (
        "OTM call",
        qfin.EuropeanCall(strike=110.0, maturity=1.5),
        qfin.BlackScholes(spot=100.0, rate=0.04, volatility=0.25),
    ),
    (
        "ITM put",
        qfin.EuropeanPut(strike=110.0, maturity=1.5),
        qfin.BlackScholes(spot=100.0, rate=0.04, volatility=0.25),
    ),
]


def main() -> None:
    print(
        "| Case | Qubits | Black-Scholes | QFin | Abs. error | Shots | "
        "Walsh terms | Retained | Logical qubits |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for seed, (name, option, market) in enumerate(CASES, start=17):
        model = qfin.compile(option, market, target_error=0.25, max_qubits=6)
        result = model.run(shots=1_000, schedule=(0, 1, 2), seed=seed)
        approximation = result.payoff_approximation
        assert approximation is not None
        print(
            f"| {name} | {model.representation.qubits} | {result.classical_value:.4f} "
            f"| {result.value:.4f} | {result.absolute_error:.4f} "
            f"| {result.resources.total_shots:,} | {approximation.parameter_count} "
            f"| {approximation.compression_ratio:.1%} "
            f"| {result.resources.total_logical_qubits} |"
        )


if __name__ == "__main__":
    main()
