"""Compile a correlated factor loss into reversible fixed-point arithmetic."""

import numpy as np

import qfin

factor_model = qfin.GaussianFactorModel(
    factor_names=("rates", "equity"),
    correlation=np.array([[1.0, -0.25], [-0.25, 1.0]]),
    means=np.array([0.02, 0.05]),
    standard_deviations=np.array([0.015, 0.12]),
)
encoding = qfin.encode_gaussian_factors(
    factor_model,
    qubits_per_factor=1,
    method="probability",
    tail_probability=0.01,
)
objective = qfin.SparseExposureObjective(
    linear={"rates": 100.0, "equity": 1.5},
    quadratic={("rates", "equity"): 10.0},
    piecewise=(qfin.HingeExposure("equity", threshold=0.05, slope=0.75),),
)
problem = qfin.FactorTailProbability(
    qfin.FactorizedLossModel(encoding, objective),
    threshold=2.0,
)
compiled = qfin.compile(
    problem,
    target_error=0.10,
    backend="auto",
    max_factorized_wires=24,
)

reference = compiled.run()
print(compiled.explain())
print("streamed reference:", reference.to_dict())
print("logical resources:", compiled.resources().to_dict())

if compiled.backend_name == "pennylane":
    runtime = compiled.to_pennylane(max_total_wires=24)
    print("Lightning power-0 probability:", runtime.probability(0))

