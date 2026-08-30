# QFin 0.8: scalable representation research

QFin 0.8 addresses one specific bottleneck: a generic multivariate empirical
distribution requires a number of stored probabilities and preparation angles
that grows with the full Cartesian product. The release adds working
factorized loaders where the financial structure permits and reports the
remaining costs instead of implying that every multivariate problem is now
efficient.

## Responsibility boundary

| Layer | Implemented in 0.8 |
| --- | --- |
| Financial API | Independent/Gaussian factor encodings and continuous mean-variance problems |
| Representation | Marginal registers, guarded validation materialization, strategy costs, block/QSVT feasibility |
| Compiler | Target-aware loader limits and explicit classical optimization selection |
| Quantum circuit | Executable factorized marginal state preparation |
| Classical solver | SciPy SLSQP and an equality-constrained closed form |
| Not implemented | Reversible affine transforms, general multivariate payoff arithmetic, block encoding, QSVT, QUBO, quantum optimization |

QFin-owned C++ finance kernels and PennyLane-Lightning retain their existing
roles. This milestone does not add quantum kernels to the native extension.

## Factorized marginal encoding

For marginal encodings with `q_j` qubits, a flattened joint distribution has

```text
joint points = product_j 2**q_j = 2**sum_j(q_j).
```

`FactorizedDistributionEncoding` stores each marginal independently:

```text
stored marginal points = sum_j 2**q_j.
```

`FactorizedPreparation` composes one existing QFin loader per marginal. A
quantile marginal uses one Hadamard per wire. A nonuniform probability
marginal uses its own probability tree. It never constructs a joint angle
table.

```python
encoding = qfin.encode_independent_factors(
    [qfin.Normal(), qfin.LogNormal(mu=0.0, sigma=0.2)],
    qubits_per_factor=(4, 4),
    factor_names=("rate_driver", "equity_level"),
    method="probability",
)
loader = qfin.FactorizedPreparation.from_encoding(encoding)
```

For tests and small classical checks, `encoding.materialize(max_points=...)`
constructs the Cartesian product behind a mandatory limit. Exceeding the
limit raises before allocation. This validation method is not used by the
production loader.

## Correlated Gaussian interpretation

`encode_gaussian_factors(...)` diagonalizes a validated correlation matrix and
encodes independent standard-normal latent registers. `LinearFactorTransform`
records the affine map from latent basis-state values to named financial
factors.

The transform is currently classical interpretation metadata. QFin does not
queue reversible multiply/add operations for it. Consequently the release
provides a scalable distribution loader foundation, not a complete scalable
correlated-payoff oracle.

## Strategy comparison

`compare_state_preparation_strategies(...)` reports, for each candidate:

- data, ancilla, and total wires;
- classical construction parameters and stored values;
- estimated classical memory;
- high-level preparation gates and a depth upper bound;
- whether joint materialization is required;
- portable-decomposition and implementation status;
- target width compatibility;
- explicit parameter and memory limit checks; and
- asymptotic scaling and caveats.

For a one-dimensional quantile encoding, QFin compares uniform Hadamards,
the generic probability tree, and the dense simulator reference. For a
factorized encoding, it compares marginal preparation, a flattened tree, and
the rejected dense-joint reference.

Only implemented, portable, within-limit candidates can be selected. Dense
references remain visible because their rejected memory cost is informative.

## Compiler target feedback

`qfin.compile(...)` accepts:

```python
representation_target=qfin.DeviceTarget.linear(8)
max_state_preparation_parameters=32_767
max_state_preparation_memory_bytes=256 * 1024 * 1024
```

An option compilation reserves the objective and work wires, caps data-qubit
search to the remaining width, and rejects a target that cannot fit the
selected loader. Risk compilation applies the same resource policy to the
generic empirical tree. If `backend="auto"` cannot satisfy the target, it
selects the stable classical risk path. If `backend="pennylane"` was explicit,
QFin raises `ResourceLimitError` instead of silently ignoring the target.

The compiled pricing and risk objects retain their
`state_preparation_strategy` report for auditability.

## Block-encoding and QSVT feasibility

`analyze_block_encoding(matrix)` measures explicit classical properties:

- shape and power-of-two padding;
- Hermitian and positive-semidefinite status;
- operator norm and normalization factor;
- condition number;
- density and row sparsity; and
- data-qubit and explicit-storage counts.

`mathematical_qsvt_candidate` says only whether the current research policy's
matrix preconditions hold. The independent fields
`qfin_block_encoding_implemented` and `qfin_qsvt_implemented` remain `False`.
The report explains that satisfying matrix preconditions does not provide an
efficient data-access oracle.

## Mean-variance portfolio baseline

`MeanVarianceProblem` supports:

- named expected returns and a validated PSD covariance matrix;
- risk aversion;
- full-investment budget;
- long-only or unbounded weights;
- per-asset lower and upper bounds; and
- an optional minimum target return.

The implemented objective is

```text
utility(w) = expected_returns.T @ w
             - 0.5 * risk_aversion * w.T @ covariance @ w.
```

Bounded problems use SciPy SLSQP with an analytical gradient. An unbounded,
budget-only problem can use the equality-constrained closed form. Results
retain the feasible starting allocation, utility improvement, budget and
target residuals, iterations, and solver status.

```python
problem = qfin.MeanVarianceProblem(
    expected_returns,
    covariance,
    risk_aversion=3.0,
    target_return=0.07,
)
compiled = qfin.compile(problem)
result = compiled.run()
```

The compiler supports `backend="auto"` and `backend="classical"` only. It
rejects `backend="pennylane"` because no quantum portfolio algorithm is
implemented. Covariance block-encoding analysis is available from the
compiled model but does not change that selection.

## Validation and performance

Tests cover marginal/joint parity on small grids, PennyLane circuit
probabilities, Gaussian correlation interpretation, allocation guards,
target rejection and fallback, matrix feasibility, optimization constraints,
closed-form first-order conditions, capabilities, and compiler boundaries.

See [scalable-representation-performance.md](scalable-representation-performance.md)
for measured construction and optimization results. Reproduce it with:

```bash
python examples/scalable_representation_benchmark.py --repeats 5 \
  --output docs/scalable-representation-performance.md
```

The benchmark does not time a flattened allocation when its configured guard
would reject it. It reports the analytical dimension-derived angle count and
labels the timing as not materialized.

## Next bottleneck

Factorized distribution loading does not make a general multivariate payoff
compact. The next milestone is a tested library of reversible affine factor
maps and sparse linear, quadratic, and piecewise financial objectives, with
error allocation and target-transpiled comparison against the generic
empirical path.
