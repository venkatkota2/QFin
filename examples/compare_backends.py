"""Compare the v0.3 compressed circuit with both exact references."""

import qfin

market = qfin.BlackScholes(spot=100, rate=0.04, volatility=0.20)
option = qfin.EuropeanCall(strike=105, maturity=1.0)
model = qfin.compile(option, market, target_error=0.10, max_qubits=8)

compressed = model.to_pennylane(mode="compressed")
structured = model.to_pennylane(mode="structured")
dense = model.to_pennylane(mode="dense")

for grover_power in (0, 1, 2, 4):
    structured_probability = structured.probability(grover_power)
    dense_probability = dense.probability(grover_power)
    compressed_probability = compressed.probability(grover_power)
    print(
        f"k={grover_power}: compressed={compressed_probability:.12f}, "
        f"structured={structured_probability:.12f}, "
        f"dense={dense_probability:.12f}, "
        f"exact_difference={abs(structured_probability - dense_probability):.3e}"
    )

print(compressed.circuit_specs(power=1))
