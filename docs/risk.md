# Finite loss risk

`LossDistribution` stores finite losses and normalized non-negative scenario
probabilities. The loss convention treats larger values as worse outcomes.
Scaling all input weights by a common positive factor leaves weighted statistics
unchanged. Very large finite weights are normalized without overflowing their sum.

VaR is the first ordered loss whose cumulative probability reaches confidence
`alpha`. CVaR/expected shortfall uses `v + E[max(L-v, 0)]/(1-alpha)`, allocating
boundary mass consistently for a discrete distribution. Strict tail probability
is `P(L > threshold)`; `inclusive=True` selects `>=`. Repeated losses, tiny weights
and confidence levels through 0.9999 are covered by differential tests.

`aggregate_risk` reports mean, standard deviation, VaR and CVaR with an explicit
engine. Local seeded percentile bootstrap intervals describe variation under the
supplied empirical distribution. They do not estimate omitted model risk or prove
coverage for a calibrated real-world population.

`compile(TailProbability(...))` accepts probability-unit target error;
`compile(VaR(...))` and `compile(CVaR(...))` accept loss-unit target error. Classical
`run()` remains available independently of optional PennyLane dependencies.
Quantum execution is explicitly requested with `run_quantum()` and retains
[statistical limitations](quantum-risk.md).

Mean-variance optimization remains classical. Positive-semidefinite covariance
is accepted, but an unconstrained zero-variance, budget-neutral direction with
nonzero expected return makes the objective unbounded and raises
`OptimizationError`. A pseudoinverse must not turn such a direction into a finite
portfolio. Well-posed singular cases, duplicated assets and return-neutral null
spaces retain a constrained/KKT solution. This is not a quantum optimizer.
