# Compiler contract

`compile()` accepts the existing option, finite-risk, structured-factor-risk and
mean-variance problems. It reports which objective, representation, algorithm and
backend are available. No new quantum algorithm or backend is introduced.

| Problem | `target_error` unit |
| --- | --- |
| European call / put | Currency / price units |
| VaR / CVaR, including factorized forms | Loss units |
| Tail probability, including factorized form | Probability units |
| Mean-variance optimization | No quantum error allocation; numerical solver controls apply |

Budgets validate their own finite positive totals and consistent allocations.
`target_error` is a requested accuracy allocation, not an unconditional guarantee
on a finite-shot run. Statistical uncertainty and failure to converge remain
visible even if a reference calculation happens to be close.

Compiled `to_dict()` contracts include problem category, financial objective,
backend, representation, algorithm, target and units, representation error,
algorithm error, sampling/statistical error, convergence, quantum execution
availability, resource estimate type and limitations. Before sampling, statistical
error is `None`. After execution, interval-based statistical error and observed
reference error are distinct fields. `explain()` describes major decisions and
failure to meet budgets. Classical fallback is labelled classical.

Distribution encoding error may be `None` when refinement was not performed.
Compiler-specific comparison against an independent or exact discrete objective
can still measure representation error. These are different measurements; unknown
encoding convergence is not a measured zero. Factorized validation references the
encoded factor grid and does not claim all continuous marginal error has vanished.

European options use Black-Scholes and the existing quantile/probability loaders.
Finite risk uses generic empirical loading; structured factor risk uses only the
existing supported affine grid and sparse exposure algebra. Explicit unsupported
backend requests fail. Resource limits do not silently enable exponential fallback
or an unavailable quantum implementation.

Logical reports precede device decomposition. Synthetic topology, noise and
export analyses are research diagnostics, not production hardware runtime or
fault-tolerant estimates. Read [limitations](limitations.md) alongside results.

## Reproducible execution in 1.1.2

Compilation options are normalized once by an internal typed configuration. No
new user configuration object or required argument is introduced. Quantum result
serialization adds execution provenance and explicit interval semantics while
preserving old keys and positional result fields. [Reproducibility](reproducibility.md)
lists the retained settings; [statistical validation](statistical-validation.md)
distinguishes deterministic encoding error from local or conditional shot intervals.
