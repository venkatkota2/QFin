# Roadmap

## 0.1 to 0.3 — European-option quantum vertical slice (complete)

- calls and puts under Black-Scholes;
- probability and inverse-CDF quantile representations;
- structured and sparse Walsh/Pauli payoff circuits;
- PennyLane execution with `lightning.qubit` by default;
- MLAE, error decomposition, validation, and logical resources.

## 0.4 — native finance and ALM foundation (complete)

- scikit-build-core, CMake, pybind11, and a wheel-compatible C++20 extension;
- NumPy fallback and explicit `auto`/`numpy`/`native` execution modes;
- yield curves, fixed/zero-coupon cash flows, batch pricing, yield solving,
  duration, convexity, DV01, accrued interest, and clean/dirty prices;
- asset/liability portfolios, funding and immunization measures;
- chunked parallel, twist, and key-rate scenarios;
- mortality tables and annual term-life projection;
- weighted loss aggregation, VaR, and expected shortfall;
- ALM scenario losses connected to the existing distribution representation;
- capability metadata that distinguishes classical, native, representation,
  and quantum-algorithm availability;
- Python/native parity tests and reproducible end-to-end benchmarks.

## 0.5 — risk algorithm and compiler policy (complete)

- tail-probability indicator objectives with strict/inclusive boundary policy;
- hybrid VaR binary search driven by MLAE CDF estimates;
- CVaR tail-excess amplitude estimation conditional on the selected VaR;
- distribution/oracle/estimation error budgets and reproducible classical
  percentile-bootstrap intervals;
- compiler selection across option pricing, tail probability, VaR, CVaR, and
  explicit classical-only execution;
- logical state-preparation, circuit, shot, oracle-query, sorting, and
  preprocessing-memory reports;
- validated Gaussian correlation assumptions and vectorized linear-factor
  loss generation;
- measured `default.qubit` versus `lightning.qubit` risk benchmarks;
- explicit documentation of `O(2**qubits)` generic loading and the absence of
  a quantum-advantage claim.

## 0.6 — richer ALM and life products (complete)

- credit-spread, equity, inflation, mortality, and lapse scenario factors;
- stochastic rates and multi-period reinvestment/rebalancing;
- participating, universal-life, annuity, and multi-state policy foundations;
- assumption sets, model-point grouping, attribution, and sensitivity reports;
- chunked scenario × policy execution without full-cube allocation.

The initial implementation deliberately uses annual life steps, simple
product foundations, synthetic scenario generation, deterministic portable
single-threaded kernels, and aggregate equity exposure. Product cash-flow
semantics and scenario dependence are explicit so calibrated models can
replace these foundations without changing the public loss-distribution
bridge.

The factor roadmap includes heavy-tailed marginals, copulas, nonlinear
revaluation, and calibrated economic-scenario models beyond the 0.5 Gaussian
foundation.

## 0.7 — device realism (complete)

- device gate-set decomposition and topology-aware resource reports;
- noise models, error mitigation, and repeatable backend benchmarks;
- additional PennyLane devices only when implemented and tested;
- Qiskit export and hardware-provider capability checks.

The built-in coupling targets are synthetic and the provider interface is
read-only. QFin does not claim hardware execution merely because an exported
circuit can be parsed or a backend exposes enough static primitives.

## 0.8 — scalable representation research (complete)

- factorized and problem-structured loaders that avoid generic empirical
  angle tables where financial structure permits;
- state-preparation strategy comparison with explicit construction costs;
- block-encoding and QSVT feasibility metadata before algorithm claims;
- hardware-target feedback in representation and algorithm selection;
- portfolio-optimization problem objects with strong classical baselines.

The factorized loader is implemented for independent latent registers and can
attach a classical affine interpretation for correlated Gaussian outputs. It
does not yet implement reversible arithmetic for that map or a general
multivariate payoff oracle. Block-encoding/QSVT reports expose mathematical
preconditions while keeping both implementation flags false. Mean-variance
optimization deliberately selects SciPy because no quantum optimizer exists.

## 0.9 — structured multivariate payoff oracles (complete)

- reversible affine factor transforms with numerical and resource validation;
- sparse linear, quadratic, and piecewise financial exposure objectives;
- tail-risk compilation from factorized registers without joint-table
  materialization;
- approximation-error allocation for factor transforms and payoff synthesis;
- target-transpiled comparisons against the generic empirical loader.

The first compiler accepts affine probability grids and sparse constant,
linear, quadratic, and univariate positive-part objectives. It uses fixed-point
out-of-place arithmetic and a reversible comparator. Exact validation streams
the encoded Cartesian domain without allocating it, so memory is bounded while
validation time remains exponential. Quantile inverse-CDF grids, arbitrary
multivariate functions, calibrated non-Gaussian dependence, and compact
fault-tolerant synthesis remain later research.

## 1.0 — structured factorized VaR/CVaR (complete)

- memory-bounded exact encoded-grid VaR/CVaR without a joint loss table;
- fixed-point VaR/CVaR parity and precision selection in financial units;
- hybrid MLAE VaR search across occupied reversible loss codes;
- reversible positive tail-excess subtraction and bitwise CVaR estimation;
- complete threshold/bit/circuit/shot/oracle-query resource accounting; and
- public examples, measured benchmarks, backend guards, and Lightning parity.

The classical reference uses repeated streamed CDF passes: it bounds memory
but is deliberately slower than materialized NumPy on every measured small and
medium case. The quantum path remains hybrid. VaR intervals combine local MLAE
intervals and CVaR intervals are conditional on the selected VaR; neither is a
simultaneous statistical guarantee. No QRAM, hardware runtime, or quantum
advantage claim is made.

## 1.1 — financial accuracy foundation (in progress)

The first conventions slice is implemented:

- valuation dates backed by `datetime.date`;
- user-supplied holiday calendars and explicit business-day adjustment;
- regular schedules with month-end rules and explicit stub boundaries;
- ACT/365 Fixed, ACT/360, ACT/ACT ISDA, 30/360 US/NASD, and 30E/360;
- continuous, periodic, and simple rate quote conversions;
- zero-rate, discount-factor, forward-rate, and direct market-node curve inputs;
- validated interpolation and explicit extrapolation choices; and
- curve diagnostics, construction metadata, and accuracy-led native fallback.

The remaining 1.1 work is deliberately separate: dated bond settlement and
accrual semantics, instrument curve bootstrapping with residual repricing
reports, precisely named curve/YTM sensitivities, key-rate risk, financial-unit
validation reports, and optional independent QuantLib comparisons.

## Later research

- QUBO/variational portfolio research only after exact formulation and strong
  continuous/discrete classical baselines;
- capital allocation and production portfolio constraints;
- quantum PDE/SDE and linear-system methods;
- calibration, stochastic volatility, and path-dependent products;
- OpenMP/SIMD/Kokkos/CUDA only after portable-kernel profiling;
- production actuarial governance, validation, and audit workflows.

Every phase must retain classical validation, explicit error accounting, and a
clear distinction between simulator feasibility and quantum advantage.
