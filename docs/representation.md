# Representation

`DistributionEncoding` owns immutable grid and probability arrays. Probabilities
must be finite, non-negative and sum to one within the documented floating-point
normalization tolerance. `state_vector()` returns their square roots; squared
amplitudes reproduce probabilities.

| `discretization_error` | Meaning |
| --- | --- |
| `None` | No refinement comparison was performed |
| Positive finite value | Measured refinement estimate |
| `0.0` | A refinement comparison measured zero difference |
| `inf` | No finite convergence estimate is available from the attempted search |

Fixed-qubit `encode` and `encode_quantiles` return `None`, including in `to_dict()`.
Code consuming this field must handle an unknown value. This is a necessary
correctness change from the old misleading zero. A refinement estimate is not a
rigorous upper bound on every financial objective; explicit compiler reference
checks supply additional evidence.

Probability bins retain weighted empirical mass and report omitted domain mass.
Quantile loading uses midpoint inverse CDF samples. Tail truncation, mean error,
objective refinement and payoff approximation are separate error sources.
Non-finite objective values and empty selected domains fail validation.

`QuantumObjectiveEncoding` maps an amplitude `a` to
`financial_offset + financial_scale*a`. CDF, strict/inclusive tail and tail-excess
objectives have exact discrete classical oracles. Circuit backends must preserve
the same encoded objective within their declared approximation tolerance.

Existing factorized loaders store marginals and affine dependence metadata without
materializing the Cartesian product. Small-grid materialization is guarded and
used only for validation. Structured reversible arithmetic retains wire, precision
and validation-work limits. Internal plan types belong under
`qfin.representation`; see the [public API migration](public-api.md).
