"""Structured factorized VaR/CVaR without a joint probability/payoff table."""

from __future__ import annotations

import numpy as np

import qfin


def marginal(
    grid: tuple[float, float],
    probabilities: tuple[float, float],
    name: str,
) -> qfin.DistributionEncoding:
    return qfin.DistributionEncoding(
        grid=np.asarray(grid),
        probabilities=np.asarray(probabilities),
        qubits=1,
        lower_bound=min(grid),
        upper_bound=max(grid),
        tail_probability=0.0,
        discretization_error=0.0,
        mean_error=0.0,
        objective=name,
    )


encoding = qfin.FactorizedDistributionEncoding(
    factors=(
        marginal((0.0, 1.0), (0.8, 0.2), "rate loss"),
        marginal((0.0, 2.0), (0.9, 0.1), "equity loss"),
    ),
    factor_names=("rates", "equity"),
)
loss_model = qfin.FactorizedLossModel(
    encoding,
    qfin.SparseExposureObjective(
        linear={"rates": 0.5, "equity": 1.0},
    ),
)
problem = qfin.FactorCVaR(loss_model, confidence=0.65)
compiled = qfin.compile(
    problem,
    target_error=0.15,
    backend="auto",
    arithmetic_scale=2.0,
    max_factor_validation_points=4,
)

reference = compiled.run()
resources = compiled.resources(schedule=(0, 1, 2), shots=2_000)
print("backend:", compiled.backend_name)
print("streamed VaR:", reference.var)
print("streamed CVaR:", reference.cvar)
print("fixed-point CVaR:", compiled.validation.oracle_expected_shortfall)
print("occupied loss codes:", len(compiled.validation.occupied_codes))
print("hybrid objectives:", resources.objective_evaluations)
print("maximum runtime qubits:", resources.maximum_runtime_qubits)
print("joint table materialized:", reference.joint_table_materialized)

if compiled.backend_name == "pennylane":
    quantum = compiled.run_quantum(
        schedule=(0, 1, 2),
        shots=2_000,
        seed=11,
    )
    print("quantum CVaR estimate:", quantum.value)
    print("quantum 95% interval (conditional on VaR):", quantum.confidence_interval_95)
