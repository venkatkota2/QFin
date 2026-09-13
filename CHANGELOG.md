# Changelog

## 1.1.3 — 2026-09-13 (repository milestone; unpublished)

- Budgeted exact-binomial uncertainty across all adaptive VaR and CVaR objectives,
  propagated quantile-selection error into expected-shortfall bounds, and made
  inconsistent sampling bounds fall back to loss support instead of false precision.
  Kept all circuits, schedules, point estimators, public signatures and result fields.
- Raised the SciPy dependency floor and minimum CI pin to its 1.11.1 licensing fix.
- Made Windows portable wheel repair select and verify a compiler-compatible
  installed MSVC runtime, fail on repair warnings, and retain runtime hashes.
- Reused ALM period closing valuations as the next period's opening valuations,
  eliminating duplicate work while retaining finite-result and memory checks.
- Added analytical, exhaustive binomial, real-circuit, packaging and performance
  regressions, 9600 sampled workflow reruns and paired ALM timings with identical
  output hashes. Licensing and repository administration remain owner decisions;
  no tag, release, publication or license grant is created.

## 1.1.2 — 2026-09-11 (repository milestone; unpublished)

- Corrected forward/backward coupon schedule month drift after February, near-zero
  and extreme rate-conversion precision, weighted VaR atom selection, CVaR tail
  cancellation, and tiny-maturity DV01/CS01 cancellation in Python and C++.
- Rejected unrepresentable discount factors and financial outputs; checked native
  dimension arithmetic and allocations; removed signed integer overflow in yield
  convexity and long policy-duration arithmetic. Kept strict floating-point flags.
- Added QuantLib/70-digit Decimal references, generated financial properties,
  malformed native buffers, GCC and Clang ASan/UBSan, branch-coverage gates and a
  limited numerical mutation campaign.
- Recorded real empirical and factorized adaptive VaR/CVaR experiments with exact
  discrete references and Monte Carlo uncertainty. Public result metadata separates
  local/conditional intervals from unestablished simultaneous workflow coverage.
- Preserved the 1.x signatures, defaults, positional field order, enum values and
  existing serialization keys; added optional keyword-only provenance and a
  compatible common exception hierarchy. Compiler validation now uses an internal
  typed configuration; classical engine policy is centralized and deterministic.
- Added fresh-process peak RSS/timing dispersion and before/after public API
  measurements. Revalidated the liability crossover and kept automatic standalone
  liability valuation on NumPy; retained explicit engine overrides and modelling scope.
- Added nine portable CPython/platform wheel builds, installed artifact tests,
  checked SHA-256 collection and a manual release-candidate attestation workflow.
  Recorded the legacy SciPy 1.11.0 minimum-test licensing/yank warning for
  maintainer review before any publication; dependency support is unchanged.
  No tag, GitHub Release, PyPI publication, license grant or branch-protection
  administration is performed by this milestone.

## 1.1.1 hardening (repository milestone)

- Corrected scenario interpolation to match shifted curves across supported
  methodologies, centralized dated ALM settlement, rejected unbounded singular
  optimizations, and made unestimated encoding errors explicit.
- Strengthened interpolation diagnostics, log-discount bootstrapping, curve
  provenance, target-error units and compiler/result contracts.
- Reused scenario and key-rate buffers, retained tiny sensitivities with stable
  arithmetic, reused par-yield schedules, reduced large temporary reductions and
  native output copying, and improved cancellation-sensitive aggregates.
- Added guarded likelihood confidence regions and deterministic all-mode MLAE
  optimization with dense-reference and seeded coverage validation.
- Standardized immutable array ownership, integer/factory validation, metadata
  version lookup, finance-first API deprecations and the intentionally small CLI.
- Expanded mathematical properties, randomized native differential tests,
  financial tolerances, cross-platform installed-wheel/minimum-dependency CI,
  strict C++ warnings, ASan/UBSan and structured performance evidence.
- Preserved existing modelling scope and research limitations. No license grant
  or new model/algorithm/backend is introduced.

## 1.1.0 — 2026-09-04

- Added a standard-library-backed financial date layer with valuation dates,
  explicit holiday calendars, five business-day conventions, month-end-aware
  schedule generation, and explicit first/final stub boundaries.
- Added ACT/365 Fixed, ACT/360, ACT/ACT ISDA, 30/360 US/NASD, and 30E/360
  day-count implementations with analytical leap-year and month-end tests.
- Added continuous, annual, semiannual, quarterly, monthly, and simple rate
  quotes plus explicit, discount-factor-preserving conversion utilities.
- Generalized `YieldCurve` to accept zero rates, discount factors, adjacent
  forward rates, and direct market-node quotes; added linear-zero,
  linear-discount, log-linear-discount, and monotone-zero interpolation,
  explicit extrapolation, dated discounting, diagnostics, and `explain()`
  metadata.
- Preserved canonical continuous-rate buffers for the existing C++ kernels and
  made unsupported interpolation/native combinations fail or fall back to the
  NumPy accuracy oracle instead of silently changing curve semantics.
- Upgraded `FixedRateBond` with issue, maturity, settlement, adjusted payment,
  end-of-month, first/next-to-last stub, convention accrual, clean/dirty price,
  spread price, yield round-trip, and par-yield semantics while preserving the
  floating-time API.
- Added curve bootstrapping for deposits, zero-coupon instruments, fixed-rate
  bond clean prices, and simple single-curve swaps, with solved nodes, discount
  factors, zero/forward rates, model quotes, and strict residual diagnostics.
- Distinguished YTM Macaulay/modified duration and convexity from parallel-zero,
  effective, spread, key-rate, DV01/PV01, CS01, and effective-convexity measures.
- Added analytical and immutable golden bond cases, a scalar reference, explicit
  absolute/relative/financial-unit reports, and optional QuantLib schedule,
  price, accrual, yield, duration, convexity, and DV01 validation.
- Added 1.1 examples and methodology documentation, cross-platform regression
  coverage, a Linux independent-validation job, and updated release metadata.

## 1.0.0 — 2026-08-31

- Added `FactorVaR` and `FactorCVaR` public problems over structured
  `FactorizedLossModel` inputs.
- Added an exact memory-bounded VaR/CVaR reference using streamed CDF and
  tail-excess passes, with explicit point-visit and chunk accounting.
- Added fixed-point VaR/CVaR parity in financial units using a bounded
  loss-code histogram, automatic precision selection, and separate
  quantization/estimation error allocation.
- Added hybrid MLAE VaR search over occupied loss codes while reusing the 0.9
  reversible loss register and threshold comparator construction.
- Added reversible positive tail-excess subtraction and bitwise MLAE
  reconstruction of discrete CVaR without a joint payoff lookup table.
- Added full hybrid resource counts for threshold objectives, excess bits,
  circuits, shots, oracle queries, and maximum runtime width, plus conservative
  target-decomposed estimates.
- Added classical/materialized parity, no-materialization, circuit-register,
  quantum VaR/CVaR, fallback, resource, validation, and capability tests.
- Added runnable examples, reproducible measured benchmarks, architecture and
  circuit documentation, and explicit statistical/simulator limitations.

## 0.9.0 — 2026-08-31

- Added sparse constant, linear, quadratic, and positive-part financial
  exposure objects over factorized encodings.
- Added out-of-place reversible fixed-point affine transforms and sparse loss
  arithmetic using PennyLane `OutPoly` and `IntegerComparator`, with unsigned
  register sizing that prevents modular wrap on the validated domain.
- Added streamed factorized loss and tail-probability references that bound
  working memory and never construct a joint probability or payoff table.
- Added `FactorTailProbability` compiler support, explicit transform/payoff/
  estimation probability-error allocations, automatic fixed-point precision
  selection, target and wire guards, and truthful classical fallback policy.
- Added a PennyLane MLAE runtime for factorized preparation, arithmetic loss
  evaluation, comparator marking, and Grover amplification. PennyLane-Lightning
  remains the quantum simulator; QFin does not implement simulation kernels.
- Added logical arithmetic reports and guarded target-transpiled comparisons
  against the generic empirical loader, including decomposition depth, routed
  gates, and SWAPs.
- Added basis-state circuit parity, streamed reference, resource-guard,
  compiler-policy, backend, example, and benchmark coverage.
- Preserved explicit limitations: affine probability grids only, bounded
  validation time, no arbitrary payoff callable, and no quantum-advantage,
  hardware-runtime, or fault-tolerant resource claim.

## 0.8.0 — 2026-08-30

- Added factorized marginal encodings that store and prepare independent
  latent registers without allocating a Cartesian-product probability table.
- Added an executable `FactorizedPreparation` circuit composed from uniform
  quantile or probability-tree marginal loaders, plus guarded small-grid
  materialization for classical validation only.
- Added structured Gaussian-factor encodings with explicit classical affine
  correlation metadata; reversible quantum arithmetic is deliberately not
  claimed.
- Added state-preparation candidate reports covering parameters, stored values,
  memory, high-level gates/depth, scaling, portability, and target compatibility.
- Fed device wire, construction-parameter, and memory limits into option/risk
  compiler policy, including an explicit classical fallback for infeasible
  `backend="auto"` risk workflows.
- Added block-encoding and QSVT mathematical-feasibility reports that always
  distinguish preconditions from QFin implementation availability.
- Added validated continuous mean-variance portfolio problems, bounded and
  target-return constraints, analytical gradients, SciPy SLSQP and closed-form
  classical baselines, compiler integration, and resource metadata.
- Added examples, measured construction/optimization benchmarks, strict typing,
  numerical tests, documentation, and cross-platform CI coverage.
- Preserved the research-prototype positioning: no reversible factor transform,
  block-encoding oracle, QSVT circuit, QUBO, or quantum optimizer is claimed.

## 0.7.0 — 2026-08-30

- Added a tested device registry that prefers `lightning.qubit`, retains
  `default.qubit` and `default.mixed`, and rejects unverified device claims.
- Added synthetic all-to-all, linear, and validated custom `DeviceTarget`
  topologies using a portable RX/RY/RZ/CNOT gate basis.
- Added recursive gate-set decomposition, topology routing, SWAP accounting,
  final-edge validation, logical-to-physical wire maps, and workflow-level
  shot/gate/depth reports for pricing and risk objectives.
- Added explicit local depolarizing/readout noise assumptions, deterministic or
  finite-shot `default.mixed` experiments, global folding, and polynomial
  zero-noise extrapolation with honest before/after error reporting.
- Implemented OpenQASM 2 and optional Qiskit circuit export, including
  reproducible digests and routed objective-wire metadata.
- Added static Qiskit BackendV2-style capability inspection without provider
  authentication, calibration claims, or job submission.
- Added numerical routed-circuit equivalence tests, Qiskit parsing tests,
  repeatable device/noise examples, and measured 0.7 backend evidence.
- Preserved QFin C++ for finance kernels and PennyLane-Lightning for quantum
  simulation; no QFin state-vector or mixed-state simulator was introduced.

## 0.6.0 — 2026-08-29

- Added validated multi-period `EconomicScenarioSet` paths for zero rates,
  credit spreads, equity returns, inflation, mortality, and lapse, including a
  transparent correlated-Gaussian generator with explicit dependence metadata.
- Extended assets with aggregate equity and cash allocations and liabilities
  with cash-flow-level inflation linkage.
- Added one-period multi-factor ALM revaluation, exact residual attribution,
  bump-and-revalue sensitivities, probability-aware loss distributions, and a
  native multi-period roll-forward for reinvestment, liability payments,
  allocation rebalancing, and transaction costs.
- Added term, participating, universal-life, and annuity model points plus
  active/disabled/dead transitions, recovery, inflation-linked benefits,
  credited account values, bonuses, surrenders, and product-level PV reporting.
- Added exact model-point grouping and named life assumption sets while
  retaining the 0.4 term-life defaults and results.
- Added a C++20 life scenario kernel that chunks both scenarios and model
  points and returns scenario aggregates without allocating a full
  scenario-by-policy-by-time result cube.
- Connected multi-period ALM and life-scenario loss distributions to the
  existing TailProbability/VaR/CVaR representation and compiler boundary
  without claiming direct quantum implementations for the financial models.
- Added Python/native parity, analytical product, chunk invariance,
  attribution, sensitivity, validation, integration, example, benchmark, and
  cross-platform CI coverage for the 0.6 vertical slice.
- Preserved PennyLane-Lightning as the compiled quantum simulator; all new C++
  kernels are QFin-owned financial and actuarial preprocessing only.

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
