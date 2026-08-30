"""Inspect decomposition, routing, noise, mitigation, and Qiskit export."""

from __future__ import annotations

import json

import qfin


def main() -> None:
    model = qfin.compile(
        qfin.EuropeanCall(strike=105, maturity=1.0),
        qfin.BlackScholes(spot=100, rate=0.04, volatility=0.20),
        target_error=1.0,
        min_qubits=3,
        max_qubits=3,
    )
    if not isinstance(model, qfin.CompiledPricingModel):
        raise RuntimeError("option compilation did not return a pricing model")

    all_to_all = model.device_resources(schedule=(0, 1), shots=1_000)
    linear = model.device_resources(schedule=(0, 1), shots=1_000, target="linear")
    noise = model.noise_analysis(
        qfin.NoiseModel(
            depolarizing_probability=0.001,
            readout_bit_flip_probability=0.002,
        ),
        power=0,
    )
    qasm = model.to_openqasm(power=1, target="linear")
    result = {
        "system": qfin.system_info(),
        "all_to_all": all_to_all.to_dict(),
        "linear": linear.to_dict(),
        "noise": noise.to_dict(),
        "openqasm": qasm.to_dict(),
    }
    if qfin.system_info()["qiskit"]:
        circuit = model.to_qiskit(power=1, target="linear")
        result["qiskit"] = {
            "qubits": int(circuit.num_qubits),
            "operations": int(circuit.size()),
            "depth": int(circuit.depth()),
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
