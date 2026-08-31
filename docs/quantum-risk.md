# Quantum Tail-Risk Workflow

QFin 0.5 implements an experimental, simulator-tested path from a finite
financial loss distribution to tail probability, value-at-risk, and
conditional value-at-risk estimates. It reuses QFin's distribution compiler,
gate-decomposable state preparation, maximum-likelihood amplitude estimation
(MLAE), PennyLane, and PennyLane-Lightning.

This milestone demonstrates a correct financial-to-quantum abstraction. It
does not claim hardware readiness or quantum advantage.

QFin 1.0 additionally implements VaR/CVaR directly over the reversible
factorized loss register, avoiding the generic joint objective table for its
supported sparse exposure algebra. See
[`structured-factor-risk-1.0.md`](structured-factor-risk-1.0.md); the generic
finite-distribution path documented here remains supported and unchanged.

## Public problems

```python
tail = qfin.TailProbability(losses, threshold=1_000_000)
var = qfin.VaR(losses, confidence=0.995)
cvar = qfin.CVaR(losses, confidence=0.995)

compiled = qfin.compile(cvar, target_error=100_000)
classical = compiled.run()
quantum = compiled.run_quantum(shots=4_000, seed=7)
```

`compiled.run()` remains the stable NumPy/QFin-native classical oracle.
`compiled.run_quantum()` explicitly selects the experimental quantum path.
This preserves backward compatibility and makes the execution choice visible.
`target_error` is expressed in probability units for `TailProbability` and in
loss/currency units for `VaR` and `CVaR`.

## Representation and state preparation

For encoded loss points `L_i` and probabilities `p_i`, QFin prepares

```text
sum_i sqrt(p_i) |i>.
```

The first implementation uses the existing binary probability-tree loader and
multiplexed `RY` objective rotations. Empty bins are permitted, all supplied
scenario mass is retained, and the circuit's measured data-register
probabilities are tested against the compiled representation.

`QuantumObjectiveEncoding` separates the distribution from the normalized
objective. If the objective-qubit success probability is `a`, it maps back to
financial units as

```text
financial value = offset + scale * a.
```

The same interface supports CDF indicators, upper-tail indicators, and
normalized tail excess without a parallel simulator implementation.

## Tail probability

For threshold `K`, the strict-tail objective is

```text
f_i = 1[L_i > K].
```

The objective amplitude is therefore `P(L > K)`. `inclusive=True` changes the
event to `L >= K`. MLAE executes the requested Grover-power schedule and
returns a point estimate and local Fisher-information 95% interval.

## Value-at-risk

For confidence `alpha`, QFin uses CDF objectives

```text
f_i(K) = 1[L_i <= K]
```

inside a classical binary search over occupied encoded grid points. Each
comparison is driven by an MLAE estimate of `P(L <= K)`. The point estimate is
the first tested loss threshold whose estimated CDF reaches `alpha`.

The reported VaR interval combines each local MLAE interval with CDF
monotonicity. It is useful diagnostic information, but it is not a
simultaneous-coverage theorem across every adaptive threshold test.

## Conditional value-at-risk

After selecting a VaR threshold `v`, QFin encodes the normalized tail excess

```text
g_i = max(L_i - v, 0) / M,
M = max_i max(L_i - v, 0).
```

For a finite loss distribution, expected shortfall is evaluated through the
Rockafellar-Uryasev identity

```text
CVaR_alpha = v + E[max(L-v, 0)] / (1-alpha).
```

MLAE estimates the normalized excess amplitude and QFin converts it back to
financial units. The CVaR interval is conditional on the selected VaR grid
point; VaR-search uncertainty is reported separately.

## Error and interval reporting

`RiskErrorBudget` separates:

- distribution discretization error;
- oracle approximation error, currently zero for exact grid-point objectives;
- amplitude-estimation error; and
- the requested interval level.

Compilation compares the encoded result with the original finite-distribution
reference at every candidate qubit count. If the representation allocation is
not met by `max_qubits`, the model remains inspectable but reports
`representation_converged=False`.

`bootstrap_risk_interval()` additionally provides reproducible weighted
empirical percentile-bootstrap intervals for classical VaR/CVaR. Those
intervals measure sampling variation under the supplied empirical
distribution; they do not include model, parameter, or scenario-design risk.

## Resource reporting

`RiskResourceReport` includes:

- logical data, objective, and work qubits;
- circuits, shots, and oracle queries;
- threshold and tail-excess objective counts;
- state-preparation and payoff rotations;
- input scenarios and encoded points;
- estimated sorting comparisons and preprocessing memory; and
- explicit complexity and pre-transpilation caveats.

The generic empirical loader and objective multiplexer both require
`O(2**data_qubits)` rotations. VaR adds `O(log(occupied_grid_points))` hybrid
threshold evaluations. These are honest first-milestone costs, not efficient
QRAM assumptions hidden from users.

## Dependence assumptions

`GaussianFactorModel` creates named correlated factor scenarios from a
validated positive-semidefinite correlation matrix. `FactorScenarios` records
the dependence assumption and can map factors into a finite loss distribution
with a vectorized linear exposure model.

This makes correlation explicit and reproducible. It is not a claim that
financial or insurance tails are Gaussian; copulas, heavy-tailed marginals,
nonlinear revaluation, and calibrated scenario models remain future work.

## Backend responsibility

QFin builds probability and objective circuits. PennyLane owns the circuit and
device abstraction. PennyLane-Lightning's compiled C++ simulator applies gates,
evolves the state vector, and samples measurements. No simulator functionality
was added to QFin C++.

See [quantum-risk-performance.md](quantum-risk-performance.md) for measured
`default.qubit` and `lightning.qubit` timings produced by
`examples/quantum_risk_benchmark.py`.
