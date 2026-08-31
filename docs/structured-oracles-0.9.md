# Structured multivariate payoff oracles (0.9)

QFin 0.9 adds a bounded, executable bridge from factorized financial models to
tail-risk amplitude estimation. Python continues to own financial objects,
validation, compiler policy, circuit construction, and reporting. QFin-owned
C++ continues to accelerate classical finance kernels. PennyLane-Lightning
continues to perform compiled quantum simulation.

## Public model

`SparseExposureObjective` supports:

- one constant;
- named sparse linear terms;
- named sparse quadratic terms, counted once per factor pair; and
- explicit `slope * max(factor - threshold, 0)` terms.

`FactorizedLossModel` combines that objective with a
`FactorizedDistributionEncoding`. `FactorTailProbability` adds a strict or
inclusive threshold and is accepted by `qfin.compile`.

The model intentionally excludes arbitrary Python callables. A callable would
usually require evaluating and storing every joint payoff, defeating the
structured representation boundary.

## Reversible arithmetic

Each probability-grid marginal has evenly spaced labels, so a factor value is
affine in its basis integer. An optional `LinearFactorTransform` preserves that
property. `compile_affine_transform` rounds the affine base and coefficients to
fixed point, shifts the output into an unsigned range, proves the range over
all encoded inputs, and emits an out-of-place `OutPoly` plan.

Polynomial exposures are pulled back analytically to the latent integer
registers. Positive-part terms use an affine output register, a reversible
`IntegerComparator`, and a controlled polynomial addition. The loss register
is shifted and sized over its complete encoded range before circuit execution.

The runtime then applies:

1. marginal state preparation on disjoint factor registers;
2. reversible fixed-point loss arithmetic;
3. a loss-threshold comparator on the objective qubit; and
4. the existing QFin MLAE Grover iterate.

QFin does not add a state-vector simulator. `lightning.qubit` remains the
preferred tested device and executes the resulting PennyLane circuit.

## Error accounting

`StructuredOracleErrorBudget` allocates the requested probability error as:

| Source | Allocation |
| --- | ---: |
| Affine transform quantization | 20% |
| Payoff synthesis and comparator classification | 20% |
| MLAE estimation | 60% |

The transform and payoff allocations share one conservative acceptance test:
the encoded probability mass whose exact and quantized tail classifications
disagree must not exceed their combined 40% budget. The report also retains
maximum and probability-weighted RMS loss error. Continuous-distribution tail
truncation and discretization remain in each marginal encoding's metadata and
are not relabelled as arithmetic error.

Automatic compilation tries power-of-two fixed-point scales from 1 through
4096. It stops at the first scale meeting the disagreement budget, subject to
loss-register, affine-register, integer-monomial, target, and total-wire caps.
An explicit scale is available for reproducible experiments.

## Memory and validation

The production factor loader stores marginal data and never creates a joint
angle table. Classical parity uses mixed-radix index decoding in bounded
chunks, evaluating factor values, losses, and probabilities for one chunk at a
time. This avoids a joint allocation, but it still visits every encoded basis
state. Memory is bounded; validation time is exponential in total factor
qubits.

The generic target comparison is different: it deliberately materializes a
small joint reference so both paths can be decomposed and routed. It is guarded
by `max_joint_points` and reports that materialization explicitly.

## Backend policy

`backend="auto"` selects the experimental PennyLane path only when:

- the factorized loader fits construction and target limits;
- fixed-point classification meets its error allocation;
- arithmetic stays under the monomial cap; and
- the complete circuit fits the configured wire cap.

Otherwise it selects the streamed classical result. `backend="pennylane"`
raises a resource error instead of silently ignoring an unmet condition.
`problem_capabilities` distinguishes this structured quantum algorithm from
the generic empirical tail-risk path and from QFin-native C++ finance kernels.

## Supported and unsupported boundaries

Implemented:

- independent marginal probability-tree preparation;
- affine observed-factor transforms;
- sparse constant/linear/quadratic exposures;
- univariate positive-part exposures;
- strict or inclusive tail probability;
- power-of-two precision search and streamed numerical parity;
- logical and target-routed resource reports; and
- `default.qubit` and `lightning.qubit` execution.

Not implemented:

- inverse-CDF quantile-grid arithmetic;
- arbitrary nonlinear multivariate payoff functions;
- direct structured VaR/CVaR threshold search;
- copula or heavy-tailed dependence arithmetic;
- calibrated hardware execution;
- pulse-level or fault-tolerant synthesis; or
- a quantum advantage claim.

Use `evaluate_factor_tail_probability` for the direct streamed reference and
`compiled.run_quantum(...)` only for small simulator experiments. Reproduce
the measured evidence with:

```bash
python examples/structured_oracle_benchmark.py --full --repeats 3 \
  --output docs/structured-oracle-performance.md
```
