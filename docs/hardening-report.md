> Historical 1.1.1 report. Current findings and verification are in
> [the 1.1.2 report](hardening-1.1.2.md). The measurements below retain their original milestone.

# QFin numerical integrity and release hardening

This report accompanies [PR #14](https://github.com/venkatkota2/QFin/pull/14),
branch `hardening/qfin-1.1.1`, against audited main
`3dba3b3803f54891d12ad2c138b4dcc276de9d42`. The package metadata version is 1.1.1.
The PR records the final tested head, checks and merge commit. This is an
engineering report, not a claim of production actuarial equivalence or a PyPI
publication announcement.

## A. Executive summary

The hardening work preserves QFin's modelling scope while repairing inconsistent
financial calculations and making uncertainty, validation and execution choices
explicit. The highest-priority repairs make scenarios reprice the actual shifted
curve, carry dated settlement through ALM, reject unbounded singular optimization,
and distinguish unknown encoding error from measured zero.

Performance work removes repeated schedules and interpolation setup before adding
any native machinery. Key-rate and par-yield execution retain their financial
definitions. Stable arithmetic preserves very small sensitivities and improves
severe cancellation. NumPy remains the automatic choice for ordinary bond pricing
where current end-to-end measurements do not support a broad native advantage.
No threading framework, product, factor, stochastic model, quantum algorithm,
quantum backend, QSVT or block-encoding implementation was added.

The financial Python API remains the public interface. Internal aliases have a
documented deprecation cycle; input arrays retain caller ownership; integer
controls no longer silently truncate floats. Installed wheels, optional-dependency
isolation, native sanitizers and targeted numerical coverage now form part of CI.
Current documentation is separated from preserved historical designs and timings.

## B. Numerical correctness changes

| Issue and old behavior | Why it was unsafe | Repaired behavior and regression evidence |
| --- | --- | --- |
| Scenario zero shocks interpolated separately from the base curve | This differs from shifting zero nodes and rebuilding discount/PCHIP interpolation | Prepared NumPy valuation reproduces `curve.shifted(shock)` for all four interpolation methods and all three extrapolation policies. A deterministic randomized slow oracle checks boundaries, singleton curves, rates of both signs, parallel/nonparallel shocks, order and chunk invariance in `test_scenario_semantics.py`. Native support stays restricted to linear-zero/flat-zero. |
| ALM asset settlement only represented numeric time | Dated bonds could be interpreted on the wrong clock or fail downstream | `AssetPortfolio` shares fixed-income `Settlement` and normalization. Omitted dated settlement uses the curve date; incompatible dates, mixed clocks and mixed batches fail explicitly. Dated price/accrual/duration, ALM, scenario and loss identities are exercised end to end. |
| Closed-form covariance pseudoinverse always produced finite-looking weights | A budget-neutral covariance-null-space direction with nonzero expected return makes the objective unbounded | Rank and null-space analysis detect the unbounded direction and raise `OptimizationError`. Well-posed singular problems retain a valid solution. Tests cover positive-definite inputs, duplicates, zero-risk combinations, return-neutral null directions and constrained analytical optima. |
| Fixed-qubit encoding claimed zero discretization error without refinement | An unmeasured approximation appeared exact | `discretization_error=None` means unestimated; infinity represents an unsuccessful refinement comparison; finite estimates, including genuinely measured zero, remain distinct. Encoding, compiler, serialization and explanation tests cover these states. Fixed-qubit objectives still defend shape, finiteness and labels. |
| Curve diagnostics largely inspected nodes | PCHIP zero interpolation can produce undesirable between-node discount and forward behavior | Diagnostics inspect actual interpolants, interpolation-aware derivatives and extrapolation samples, separating node/interpolation/extrapolation warnings. Negative rates and forwards remain permitted; mathematical invalidity is distinct from economic unusualness. |
| Bootstrap discount roots used an arbitrary fixed positive bracket | Extreme negative-rate or long-maturity valid roots could lie outside it | Solve in log-discount coordinates with adaptive bracketing and strict quote repricing. Ordinary, negative-rate, long-maturity, dated swap and failure cases are tested; no calibration model is added. |
| Target error described as price units across problem types | Risk currency and probability accuracy had ambiguous meaning | Metadata, budgets, compiler explanations and results report price/currency, loss or probability units by objective. Factories validate finite positive budgets and consistent allocations independently. |
| Yield solver could converge from a small price residual alone | A short bond can have a small price error and material yield error | Exact residual or yield-bracket-width convergence is required. Randomized near-boundary yields, short/long maturities and explicit failure-to-converge tests verify both price and yield. |
| Key-rate DV01 subtracted nearly equal total bond prices | Trillion-unit principals could erase tiny real node exposure | Prepared linear-zero kernels compute cash-flow changes with `expm1`; interpolation basis weights retain the small complementary weight directly. A 70-digit Decimal oracle tests a cash flow one floating-point step from a node, with roughly `4e-8` currency exposure. Other interpolation methods retain their exact stressed-discount methodology. No tolerance was relaxed to fix macOS parity. |
| Near-zero denominators treated small nonzero PVs as zero | Scaling could change duration or make a valid par yield undefined | Only exact zero receives zero-denominator handling. Tiny-face par-yield, bond analytics and ALM scale tests preserve nonzero quantities. |
| Ordinary sums could erase unit adjustments amid large opposite notionals | Financially meaningful residuals disappeared despite finite inputs | Pairwise sums remain normal; severe cancellation escalates to `math.fsum` in selected Python aggregates and compensated native reductions. Stress references, portfolio/liability/scenario moments and parity tests bound the tradeoff. |
| Large finite probability weights could overflow during normalization | Equivalent weight scaling produced invalid statistics | Normalize after scaling by the largest weight, including loss distributions, empirical/economic scenarios and attribution. `1e308` weight tests verify equivalence and caller-array ownership. |
| Native scenarios could accept an invalid unused node or return a nonfinite PV | Native and shifted-curve semantics disagreed, with plausible partial results | Native validation checks all stressed nodes and queried discounts plus finite outputs; NumPy rejects nonrepresentable scenario totals too. Empty buffers, unused invalid nodes, overflow/underflow and indexed scenarios have regressions. |
| Enum aliases eagerly evaluated invalid fallback constructors | Existing documented interpolation/calendar aliases unexpectedly raised | Alias lookup precedes canonical construction. Every repaired alias is compared with canonical financial output. |
| Calendars with all seven weekend days could never terminate; maximum dates overflowed | Validations or date loops could hang or leak obscure exceptions | Reject calendars without a possible business weekday; handle supported date limits explicitly. Stub consistency, coupon-period collapse, leap-year, endpoint and day-count identities have tests. |
| Local Fisher uncertainty treated MLAE likelihood as a single regular peak | Low shots, boundary amplitudes and ambiguous schedules can under-cover | The primary result carries a likelihood-ratio region with a simultaneous exact-binomial guard, preserving disjoint modes. Fisher curvature is diagnostic only. Dense-reference, boundary, narrow-mode and Monte Carlo tests validate the existing MLAE algorithm. |

The MLAE likelihood is partitioned at its sinusoidal zeroes. Each finite interval
has a concave log likelihood, allowing deterministic bounded refinement of every
candidate mode before choosing the global maximum. This avoids replacing the
dense search with an unverified single local optimum. Confidence regions include
all accepted modes; a single displayed interval is their envelope.

The [MLAE study](mlae-validation.md) contains **9,000 seeded fits** across 45
amplitude/schedule/shot combinations, with amplitudes 0.001, 0.01, 0.05, 0.25, 0.5,
0.75, 0.95, 0.99 and 0.999. Each cell has 200 repetitions. Guarded empirical 95%
coverage ranges from **94.5% to 100%**, compared with **87.5% to 100%** for the
unguarded LR region. These finite Monte Carlo proportions have sampling
uncertainty. The guard can be conservative and does not establish simultaneous
coverage for an adaptive sequence of VaR thresholds or conditional CVaR estimates.

## C. Performance and memory report

All times below are seconds. Public API measurements include Python conversion,
schedule/buffer construction, allocation and boundary crossings. Inputs are
prepared outside the timing as documented by each benchmark. The detailed JSON
records repetition counts, problem shape, numerical differences and environment.

The current runner is Linux x86_64, AMD EPYC 9V74, Python 3.12.14, QFin 1.1.1,
NumPy 2.5.2, SciPy 1.18.1, GCC 13.3.0, PennyLane 0.45.1 and Lightning 0.45.0.
OMP, OpenBLAS and MKL threads were set to one. Release/strict C++ flags, including
`-O3`, C++20 and LTO, are recorded in [build-configuration.json](build-configuration.json).
Earlier evidence retains its original environment instead of being relabelled.

### Changes to execution architecture

| Calculation | Previous execution | Hardened execution | Repetitions | Maximum difference |
| --- | ---: | ---: | ---: | ---: |
| Floating par yield, two derived bonds → one schedule | 0.000190362 | 0.000025087 | 5 | 0 |
| Dated par yield, two derived bonds → one schedule | 0.000616231 | 0.000073578 | 5 | 0 |
| Key rate, 1 bond × 5 nodes, brute force → prepared native | 0.001510993 | 0.000062082 | 3 | 6.70e-15 |
| Key rate, 100 bonds × 5 nodes | 0.018422321 | 0.002370113 | 3 | 4.26e-14 |
| Key rate, 1,000 bonds × 17 nodes | 0.532763344 | 0.031016821 | 3 | 5.68e-14 |
| Key rate, 10,000 bonds × 33 nodes | 8.273657942 | 0.436920111 | 3 | 5.68e-14 |
| 10,000-bond pricing, repeat/bincount → selected reduction | 0.137317111 | 0.119005335 | 5 | 1.42e-14 |
| 31 MLAE fits, dense reference → all-mode refinement and regions | 0.517087500 | 0.052362911 | 5 | 5.66e-6 amplitude |

These references reproduce the earlier execution architecture against the same
financial calculation; they are not a claim that every full release workload
improves by those factors. In particular, most key-rate improvement comes from
preparing once. Prepared NumPy/native timings at 5,000 bonds × 17 nodes are
0.251386252/0.126739772; at 10,000 × 33 they are 0.936766602/0.436920111.
The automatic threshold is therefore conservative at eight million cash-flow
scenario visits; smaller cases vary from near parity to a modest native gain.

### Current engine comparisons

| Workload | NumPy/Python | Native | Interpretation |
| --- | ---: | ---: | --- |
| 10,000-bond ordinary pricing | 0.114580 | 0.116571 | NumPy wins this run |
| 100,000-bond ordinary pricing | 1.234544 | 1.142911 | No stable broad crossover across the matrix |
| 100,000-bond yield inversion | 12.506504 | 2.943251 | Native removes a repeated solver loop |
| ALM base, 10,000 assets | 0.098413 | 0.103154 | NumPy wins this run |
| Scenarios, 1,000 assets × 10,000 scenarios | 11.435073 | 7.792020 | One timed run per engine; max surplus difference 7.28e-10 |
| Life, 100,000 model points | 87.413358 | 0.178059 | Existing policy/year Python loop versus existing native kernel; one timed Python run |
| Weighted risk, 100,000 losses | 0.011494 | 0.012049 | Near parity; NumPy wins this run |
| Weighted risk, 1,000,000 losses | 0.149734 | 0.109280 | Environment-specific native gain |

The representative five-data-qubit circuit measured 0.003596 seconds on
`default.qubit` and 0.002110 on `lightning.qubit`, with objective-probability
difference `1.11e-16` and at least five repetitions. This is a simulator comparison,
not a quantum-advantage claim. The [full current matrix](native-performance-current.md)
also covers 1/100/1,000/10,000/100,000 bonds, multiple curve-node counts, all four
interpolation modes, dated/floating schedules and rates of both signs.

Ordinary pricing and multi-period ALM paths remain NumPy by default. Batch yield
inversion, compatible scenarios and life use native where the observed repeated
loops justify it. Weighted risk retains its measured policy with near-ties
disclosed. Mortality interpolation stays NumPy. Results use `mixed` when substantive
analytics combine engines. [Performance](performance.md) records exact policies
and their environmental limits.

### Rejected or accuracy-driven tradeoffs

- Prefix-difference segmentation lost a one-unit segment after a trillion-unit
  prefix, so it was rejected despite attractive timings in some cases.
- `reduceat` avoids the repeated instrument-index array for large batches, but
  total traced peak allocation for the measured 10,000-bond public call remained
  approximately 27.54 MB either way. No large overall peak-memory gain is claimed.
- In the cancellation stress case, naive and Kahan sums missed about 0.203451
  currency units and pairwise NumPy missed 0.193929 against `math.fsum`. Neumaier
  and the selective fsum path matched the reference. Selective fsum took 0.001641
  seconds versus 0.000007 for pairwise reduction, so it is reserved for severe
  cancellation. These are reduction microbenchmarks.
- Direct NumPy-owned native scenario output removes one vector and copy, or
  8 MB for one million doubles. The private before/after extension investigation
  measured 0.011196/0.012009 seconds at that size; stronger validation is also
  included in the new kernel, so no output-copy speedup is claimed. Small result
  copies are retained. No capsule lifetime machinery was needed.
- Scenario intermediates have a 2,097,152-element cap per NumPy matrix (16 MiB),
  with several such matrices potentially live. Key-rate batches use a separate
  bounded chunk target. Life/ALM retain chunking and aggregate outputs; no new
  scenario × policy × year or scenario × instrument × period cube is allocated.
- No OpenMP, TBB or other threading framework was introduced. Correctness and
  measured single-thread behavior remain the basis for dispatch.

Evidence: [par yield/reductions](hardening-performance.md),
[final key-rate/scenarios](scenario-final-performance.md),
[native output investigation](native-output-performance.md), and
[MLAE](mlae-validation.md), each with accompanying JSON. Scheduled/manual benchmark
CI preserves comparable structured output without making ordinary CI fail on
small noisy timing movements.

## D. Native parity and mathematical tests

`tests/native/test_randomized_differential.py` uses local seeded generators and
includes 12 mixed bond batches, dated settlement batches, weighted loss tails at
confidence 0.90/0.95/0.995/0.9999, eight varying life batches and several ALM scenario
shapes. Bond inputs include zero coupons, negative/zero/high rates, negative yields
close to the frequency lower boundary, annual/semiannual/quarterly coupons,
0.01-year and 55+-year maturities, and face values from tiny to trillion scale.
All reported bond analytics and solver convergence/iterations are compared.

The existing and expanded native suites add malformed dimensions and offsets,
empty and singleton segments, zero/huge cash flows, tiny weights, repeated losses,
retained output ownership, scenario chunking and reproducibility. Native checks
run under ASan/UBSan as well as strict warning builds. NumPy is the differential
reference; analytical, Decimal and optional QuantLib references reduce the risk
of matching implementations sharing a financial mistake.

Metamorphic tests cover cash-flow/notional scaling, monotone yield pricing, yield
round trips, analytical zero coupons, clean plus accrued equals dirty, numerical
duration/convexity, key-rate reconciliation, rate conversion round trips, node/zero
shift identities, scenario permutation/chunk invariance, ALM funding and duration
gap identities, base surplus minus stressed surplus, policy grouping/count scaling,
zero mortality/lapse, and normalized/permuted weighted risk.

Small encoded objectives compare dense, structured and compressed existing circuits
on `default.qubit` and `lightning.qubit`, including seeded shots, tiny angle
coefficients and multiple Grover powers. Factorized arithmetic retains encoded-grid
and resource-limit oracles. Probability normalization, squared amplitudes and
classical discrete objective equivalence remain strict checks.

`FinancialTolerance` requires both floating-point and financial-materiality bounds
when supplied, with explainable price/PV/DV01/loss currency, years, convexity and
probability units. A single relative epsilon is not treated as universal financial
materiality.

## E. API and compatibility

| Classification | Contract |
| --- | --- |
| Stable financial top level | Curves, bonds/cash flows, ALM portfolios/models/results, mortality/life assumptions, loss distributions and VaR/CVaR/tail objectives, mean-variance problems, `compile`, `system_info`, core financial utilities |
| Deprecated top-level aliases | `IntegerHingePlan`, `IntegerPolynomialPlan`, `IntegerQuadraticTerm`, `ReversibleAffineTransformPlan`, `StatePreparationCost`, `StructuredLossOraclePlan`, `ProbabilityTreePreparation`, `WalshTerm` |
| Canonical namespaces retained | `qfin.finance`, `qfin.representation`, `qfin.circuits`, `qfin.compiler`, `qfin.resources`, `qfin.backends` |
| Internal only | Prepared scenario/bond buffers, segmented/summation helpers, native batch kernels and `_qfin_native` |

Deprecated aliases emit `DeprecationWarning` and remain throughout 1.x, with removal
no earlier than 2.0. No canonical import path was removed. The [API contract](public-api.md)
lists exact replacements. Correctness changes intentionally reject previously
ambiguous settlements, silently truncated integer controls, invalid factories and
unbounded optimization; they are documented behavior corrections.

Immutable model arrays use owned dtype-specific copies before read-only flags are
set. Constructor tests verify caller values, shape and writeability do not change.
Shared `operator.index` validation rejects booleans and floats for integer controls,
while accepting genuine integer scalars. Version lookup uses package metadata with
a source-tree fallback. The CLI remains the existing option demo with version,
help, compile-only and seeded JSON execution tests; no CLI feature family was added.

## F. CI and release engineering

There are 16 CI jobs after matrix expansion:

- Ruff and strict mypy.
- Full tests on Linux Python 3.11/3.12/3.13, macOS 3.12 and Windows 3.12.
- Strict native parity build with `-Werror -Wshadow -Wconversion -Wsign-conversion`.
- Linux GCC AddressSanitizer and UndefinedBehaviorSanitizer with halt-on-error.
- Minimum dependencies: Python 3.11, NumPy 1.26.0, SciPy 1.11.0, without PennyLane.
- Nine representative examples, including financial validation and current quantum paths.
- Clean sdist installation and installed-package smoke calculations.
- Actual wheel build/clean install/smoke on the same five OS/Python combinations.

Wheel smoke checks verify installed import paths/native loading, NumPy/native
financial calculations, classical risk/optimization, factorized compilation and
`system_info`. PennyLane, Qiskit and QuantLib remain absent in core-only smoke
environments. Linux Python 3.12 also installs optional Qiskit and QuantLib for
interoperability and independent financial validation. SHA-256 checksum files
accompany wheel/sdist artifacts.

ASan/UBSan exercise native buffer validation, ownership and parity. Leak detection
is disabled for unrelated Python-runtime allocations; address and undefined
behavior checks remain enabled. Aggressive warnings are opt-in for verification,
not imposed on all end-user builds. C++ remains C++20 under the existing
scikit-build-core/CMake/pybind11 architecture.

Workflows have explicit `contents: read`, disabled checkout credential persistence,
and immutable official Node-24 action pins. Build jobs have no publishing secrets.
No PyPI upload automation was present or introduced. A future publication should
use separately scoped trusted publishing after licensing is resolved.

**Administration limitation:** GitHub reports main unprotected and the connected
App lacks administration access. [Exact ruleset settings and all required check
names](repository-settings.md) are supplied for the maintainer. This report does
not claim branch protection was applied. The user-authorized merge is still
conditioned on successful checks and absence of unresolved correctness issues.

## G. Test statistics and audit trail

The mandatory initial audit covered every requested Python layer, native loader,
C++ headers/kernels/bindings, tests/examples/docs, packaging, exports, CLI, workflows
and benchmarks before implementation. The actual CMake entry point is
`cpp/CMakeLists.txt`. Baseline tests, static checks, build/install/native checks and
representative examples preceded H1; H1 regressions were resolved before proceeding
to measured H2 changes, then H3 and H4.

A runner restart removed some original temporary logs. The untouched audited-main
installation was rechecked on 2026-09-09 with an isolated import path and original
native extension: **217 passed, 2 optional skips, 87.71% line coverage**, Ruff clean,
strict mypy clean on 65 source files. This recovered baseline is not a reconstructed
original timing measurement. Current wheel/sdist and CI evidence independently
verify the hardened artifact.

The final local counts and per-module percentages are recorded in
[validation](validation.md) and [hardening-validation.json](hardening-validation.json).
Counts differ with optional dependencies; Linux CI includes two additional
optional-library tests. Global coverage rises from the original required floor
of 78% to 90%, with targeted gates finance 95%, compiler 92%, representation 92%,
resources 90%, backends 85%, and Python native loader 95%.

A clean coverage run exposed that an earlier local report included stale coverage
data: CI correctly rejected finance at 94.84% even though all tests passed. The
gate was retained and missing schedule/factor boundary tests were added. Final
numbers use a fresh complete run. These are Python line-coverage statistics, not
C++ line or branch coverage. Native confidence comes from independent references,
differential tests and sanitizers, not a fabricated C++ coverage percentage.

## H. Remaining limitations

- The repository has no license grant. No license was invented. This remains an
  adoption/reuse blocker where an explicit grant is needed; MIT, Apache-2.0 or
  proprietary terms require a deliberate maintainer decision.
- Main protection requires maintainer administration. CI is implemented but a
  repository ruleset must enforce it against future direct writes and force pushes.
- Quantum experiments remain simulator based, research grade and subject to
  encoding/statistical error and resource restrictions. No quantum advantage,
  production hardware-runtime or fault-tolerant cost is established.
- Native curves support linear-zero/flat-zero only; other supported methods use
  NumPy. Dated settlement must share the curve valuation date; mixed dated/floating
  batches are rejected. Calendars are caller-supplied holiday sets.
- Life/ALM models remain the existing simplified annual/periodic models and are
  not replacements for commercial actuarial platforms. Scenario assumptions are
  not newly calibrated or validated against every market regime.
- Confidence-region coverage does not automatically extend to an entire adaptive
  quantum risk workflow. Finite Monte Carlo coverage measurements are uncertain.
- Timings are hardware/environment dependent, with explicitly identified single
  repetitions for some large workloads. No claim of universal dispatch thresholds
  or uniform speedup is made. Some accuracy/validation changes cost time.
- CI verifies the practical matrix above, not every possible Python/platform/
  architecture combination. Packaging verification does not mean artifacts have
  been released on PyPI or that a new release tag has been published.

The [limitations page](limitations.md) remains part of current documentation.

## I. Major changed files

| Area | Principal files |
| --- | --- |
| Financial semantics and preparation | `finance/curves.py`, `scenarios.py`, `fixed_income.py`, `alm.py`, `calibration.py`, `optimization.py`, `dates.py`, `daycount.py` |
| Aggregation, ownership and validation | `_numerics.py`, `_validation.py`, financial distribution/risk/life/path modules and array-bearing representation/result models |
| Encoding and compilation contracts | `representation/encoding.py`, `factorized.py`, `compiler/compile.py`, compiled pricing/risk/factorized result models |
| MLAE and backend checks | `algorithms/amplitude_estimation.py`, backend/circuit control validation, statistical/backend parity tests |
| Native buffers and arithmetic | `cpp/include/qfin`, `cpp/src`, `cpp/bindings/python.cpp`, `cpp/CMakeLists.txt`, native parity/ownership tests |
| Public contract | `__init__.py`, `__main__.py`, `pyproject.toml`, `CHANGELOG.md`, `README.md`, `docs/public-api.md` |
| Verification and benchmarks | Numerical/property/differential tests, `.github/workflows/ci.yml`, `performance.yml`, coverage/wheel smoke scripts, native/hardening/MLAE/output benchmark scripts |
| Current evidence and history | Current topic docs, numerical methodology, validation, performance JSON/Markdown, repository settings and `docs/history` |

Paths in the first five rows are relative to `src/qfin` except explicit C++,
package and top-level files. The PR diff is the complete file-level change record.
Build outputs, caches and temporary environments are excluded from the commit.

## J. Branch and merge record

- Working branch: `hardening/qfin-1.1.1`.
- Audited main: `3dba3b3803f54891d12ad2c138b4dcc276de9d42`.
- Pull request: [#14](https://github.com/venkatkota2/QFin/pull/14).
- Merge acceptance: all 16 final-head CI jobs must succeed, with no unresolved
  correctness, packaging or compatibility failure or merge conflict.
- Exact tested head, check runs and eventual merge SHA are retained on the PR.
  The final task response records the verified post-merge main SHA; a commit cannot
  embed its own future merge identifier in this document.
- Source/package version: 1.1.1. A source merge does not create a tag or publish a
  PyPI artifact, and neither is claimed here.
