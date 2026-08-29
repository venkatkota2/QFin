# Changelog

## Unreleased

## 0.5.0 — 2026-08-29

- Added public `TailProbability` and `VaR` problems alongside `CVaR`.
- Added `QuantumObjectiveEncoding` for CDF indicators, upper-tail indicators,
  and normalized tail-excess objectives over finite loss representations.
- Added an experimental PennyLane risk runtime that reuses QFin's structured
  probability loader and PennyLane-Lightning rather than implementing another
  simulator.
- Added MLAE tail-probability estimation, hybrid MLAE-driven VaR binary search,
  and CVaR tail-excess estimation through the Rockafellar-Uryasev identity.
- Added separate distribution, exact-oracle, and estimation error budgets;
  local quantum intervals; and reproducible weighted empirical bootstrap
  intervals for classical VaR/CVaR.
- Added logical risk-resource reports covering circuits, shots, oracle queries,
  state preparation, objective evaluations, sorting work, and preprocessing
  memory with explicit pre-transpilation caveats.
- Added validated Gaussian multi-factor correlation scenarios and vectorized
  linear-exposure loss mapping with an explicit dependence-assumption label.
- Added ALM-to-quantum-risk integration tests, examples, and measured
  `default.qubit` versus `lightning.qubit` benchmarks.
- Documented that generic empirical state/objective loading remains
  `O(2**data_qubits)` and that the workflow does not claim quantum advantage.

## 0.4.0 — 2026-08-29

- Added a wheel-compatible C++20 financial extension using CMake,
  scikit-build-core, and pybind11; the public API remains Python-first.
- Added reusable zero curves, fixed/zero-coupon cash flows, batch curve/yield
  pricing, yield solving, duration, convexity, DV01, accrued interest, and
  clean/dirty prices with NumPy/native parity.
- Added fixed-income asset portfolios, deterministic liability portfolios,
  ALM funding/immunization measures, and chunked rate-scenario valuation.
- Added mortality tables, annual term-life model points, lapse/expense
  assumptions, native policy projection, aggregate cash flows, and per-policy
  PVs.
- Added weighted finite loss distributions, VaR/CVaR aggregation, and an honest
  classical-to-quantum representation bridge for ALM scenario losses.
- Added `qfin.system_info()` and problem capability metadata that distinguishes
  classical models, native kernels, quantum representations, and quantum
  algorithms.
- Preserved PennyLane-Lightning as the default quantum simulator and documented
  the separation between QFin-owned finance C++ and Lightning simulator C++.
- Added cross-platform native CI, extensive analytical/edge/parity tests,
  runnable examples, and measured end-to-end performance benchmarks.
- Prevented false representation convergence when coarse grids miss a non-zero
  tail payoff by validating MVP qubit selection against Black–Scholes.
- Preferred PennyLane Lightning's compiled C++ backend through automatic device
  selection, with `default.qubit` fallback and an explicit device override.
- Corrected empirical encodings so custom bounds report and exclude omitted
  mass rather than folding it into boundary bins.
- Renamed the installable distribution to `qfin-quantum` because `qfin` is
  already occupied on PyPI; the import package remains `qfin`.
- Bounded NumPy and SciPy to their current major versions and pinned development
  checks to mypy 1.x after mypy 2.x failed in the supported local environment.
- Aligned package and README licensing metadata with the repository's current
  no-license state.
- Added a reproducible multi-case numerical demonstration and a CI coverage
  regression floor.

## 0.3.0 — 2026-08-24

- Added inverse-CDF quantile encoding, preparing the terminal lognormal
  register with one Hadamard per data qubit and no distribution-angle table.
- Added a sparse Walsh/Pauli payoff compiler controlled by financial price
  error, angle RMSE, and an optional hard term cap.
- Added the compressed PennyLane backend and retained the v0.2 structured and
  v0.1 dense implementations as exact references.
- Separated representation, payoff-approximation, circuit-estimation, and
  total benchmark errors in compiler and runtime results.
- Expanded resource reports with distribution gate counts and payoff
  compression ratios.

## 0.2.0 — 2026-08-24

- Replaced the default dense Householder circuit with a binary probability-tree
  distribution loader made from multiplexed Y rotations.
- Added uniformly controlled payoff rotations and gate-level Grover
  reflections with one reusable work qubit.
- Retained the v0.1 dense backend as an explicit numerical reference.
- Added structured-versus-dense equivalence tests and PennyLane circuit specs.
- Expanded resource reporting to distinguish parameters, work qubits, and
  avoided dense-unitary entries.

## 0.1.0 — 2026-08-24

- Added the European call/put compiler, lognormal discretization, MLAE,
  Black-Scholes validation, the PennyLane simulator backend, CLI, and tests.
