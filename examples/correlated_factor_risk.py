"""Generate explicit correlated factors and compile a tail-risk problem."""

import numpy as np

import qfin

factor_model = qfin.GaussianFactorModel(
    factor_names=("rates", "equity", "credit"),
    correlation=np.array(
        [
            [1.00, -0.25, 0.20],
            [-0.25, 1.00, -0.35],
            [0.20, -0.35, 1.00],
        ]
    ),
    standard_deviations=np.array([0.01, 0.15, 0.02]),
)
scenarios = factor_model.simulate(10_000, seed=11, antithetic=True)
losses = scenarios.linear_loss_distribution(
    # Positive values define loss exposure to each factor shock.
    exposures=np.array([20_000.0, -4_000.0, 35_000.0])
)

problem = qfin.TailProbability(losses, threshold=1_500.0)
compiled = qfin.compile(problem, target_error=0.01, min_qubits=4, max_qubits=8)

print("Dependence:", scenarios.dependence_assumption)
print("Classical tail probability:", compiled.run().probability)
print(compiled.explain())
print(compiled.resources(shots=2_000).to_dict())
