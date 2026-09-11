# QFin 1.1.2 hardening report

This milestone hardens existing behavior and its evidence. No financial model,
quantum algorithm, backend, provider, threading framework or CLI feature is added.
Version 1.1.2 is an unpublished repository milestone. The delivery is
[PR #15](https://github.com/venkatkota2/QFin/pull/15), branch
`hardening/1.1.2-verification`; its timeline records the final tested head and merge.

## A. Starting state and verification provenance

Audited main: `f9d4be69432ee813fccdde609420cdee9eb15580`, version 1.1.1.
The untouched baseline reproduced 654 passing tests and two optional Qiskit skips,
92.9599% Python line coverage and 81.0391% branch coverage. Baseline CI had all
16 jobs green, including optional dependencies, installed wheels and sdist.
See [baseline environment](history/hardening-1.1.2/baseline.json) and
[previous milestone](hardening-report.md).

The rebuilt local 1.1.2 installation passed **2232 tests, two optional Qiskit
skips, in 27.30 seconds**, with **93.5136% line coverage (8160/8726)** and
**82.2451% branch coverage (1839/2236)**. Ruff is clean and strict mypy passes
71 source files. All numerical line and critical branch gates pass. The final
500-example Hypothesis stress profile passed all nine properties in 9.56 seconds.

Implementation checkpoint `46ef9f83d8536e85afbbfaa4fe821524b368f274` passed all
20 CI jobs in [run 34576612885](https://github.com/venkatkota2/QFin/actions/runs/34576612885).
Its Linux 3.12 job passed **2234 tests**, including both locally skipped Qiskit
checks, with **93.5488% lines and 82.2898% branches**. The final local result above
also includes the evidence-based liability dispatch correction following that
checkpoint. Final-head CI is verified separately before merging; this report
does not relabel checkpoint artifacts or development experiments as later builds.
The PR records the final CI run and immutable SHA.

[Raw evidence](history/hardening-1.1.2/README.md) includes per-module coverage,
actual benchmark samples, reference errors, compiler commands, statistical cells,
artifact digests, and a SHA-256 snapshot of the final financial source files.

## B. Defects discovered and disposition

Priorities below describe correctness/operational impact in the affected domain;
they are not claims of regulatory materiality. No known unresolved P0 numerical
defect remains within the supported classical scope. Testing cannot prove that
all possible defects are absent.

| Defect and risk | Cause and old behavior | Correction and independent regression evidence |
| --- | --- | --- |
| P1: coupon schedule drift | Repeatedly adding months to the previously clamped date allowed February to move later coupon dates, changing accrual/cashflows. | Derive regular forward/backward dates from the original anchor. QuantLib EOM, stub, direction and holiday fixtures test dates exactly; dated prices test the resulting cashflows. |
| P1: near-zero/extreme rate conversion | Direct `log(1+x)`/subtraction and conversions through an intermediate discount lost small rates or overflowed even when a quote conversion was representable. | Use `log1p`/`expm1` and continuous-rate conversion; reject nonrepresentable discounts and quotes that round outside their domains. 144 Decimal cases plus inverse, extreme and boundary regressions preserve signs and precision. |
| P1: weighted VaR at atoms | Always scaling probabilities by their maximum could move a 9:1 cumulative probability across the 90% atom in floating point. | Normalize by the direct finite sum and use maximum scaling only for overflow. Independent inverse-CDF cases and weighted scaling/permutation properties retain the specified atom. |
| P1: CVaR cancellation | Subtracting cumulative probabilities near one degraded a small weighted tail. | Python/C++ compute VaR plus the normalized positive excess above VaR, preserving fractional atom semantics. Decimal finite-distribution references cover rare tails, repeated losses and extreme confidence. |
| P1: tiny real DV01/CS01 erased | Subtracting nearly equal large prices lost a sensitivity for tiny maturities and trillion-unit faces. | Use the stable sensitivity numerator already computed with `expm1`, then apply the bump. 70-digit zero-coupon references cover maturity 1e-6 through 60 years and face 1e12. |
| P1: native arithmetic/allocation safety | Frequency squared or duration plus year could overflow before conversion; dimension products and some large buffers lacked complete preflight. | Promote arithmetic before multiplication/addition; check `size_t` addition/multiplication, `ptrdiff_t` and a 512 MiB guarded-group budget before allocation. Direct malformed/logically huge buffers and C++ size tests run under GCC/Clang sanitizers. |
| P1: finite inputs yielding invalid financial outputs | Extreme finite cashflows/rates could return nonfinite bond, life or ALM outputs without a coherent failure. | Python/native finite-result validation now rejects unrepresentable results. Funding ratio infinity is intentional only for zero liabilities. Regression cases exercise overflow and large dimensions without giant allocations. |
| P2: fragmented operational exceptions | Public validation/operational paths did not consistently share a QFin root or native translation. | Compatible QFin subclasses retain ValueError/TypeError/AssertionError catch behavior where applicable. Native translation and `python -O` regressions verify explicit failures. |
| P2: unsupported liability crossover | The old 4096-flow auto threshold was not supported by the new end-to-end measurements. | Rechecks from 2048 to 262144 liabilities favor NumPy or are near ties. Auto now uses NumPy; explicit native is retained and differential tests preserve financial results. |
| P1 research limitation: conditional CVaR interval can miss badly | An ambiguous amplified schedule can choose the wrong VaR; the subsequent conditional interval omits selection uncertainty. | Interval scope and execution provenance are explicit and full workflows are measured. The `(1,)` rare-tail case still has 0/100 conditional CVaR coverage. A rigorous adaptive workflow interval cannot safely be claimed or substituted within this scope; see section F. |

Validation also found weaknesses in the test apparatus. A proposed independent
DV01 oracle initially used a one-sided change; it was corrected to the specified
central 1 bp quantity. API snapshots initially included Python's changing Enum
machinery, so they were narrowed to actual exported QFin contracts while keeping
all enum values and conversion behavior. Windows interpreted an unquoted
Hypothesis version range as shell redirection; the portable-wheel test dependency
now uses an exact version. These failures were understood and fixed before the
green checkpoint; they were not accepted through unexplained reruns.

## C. Independent financial validation

The checked-in corpus has **1073 independent configurations**: 144 rate, 60 day
count, 52 schedule, 162 bond/yield, 432 curve, four bootstrap, 30 weighted-risk,
nine ALM, 144 life and 36 dated-bond cases. Python/native parameterization produces
1454 execution paths; it does not increase the number of independent cases.
Another **26 inline independent configurations** cover nonlinear PCHIP, dated
bootstrapping and zero-coupon sensitivities: **1099 configurations in total**.

`generate_references.py` imports neither QFin nor SciPy. Its expected values use
70-digit Decimal arithmetic and QuantLib 1.43. The separate dated generator uses
QuantLib cashflows with 30E/360 coupon accrual, ACT/365F discounting, explicit
settlement and unadjusted EOM dates. Stored data needs no QuantLib installation.
[Corpus provenance](../tests/reference_data/README.md) records intentional
differences: QFin's signed reversed US 30/360 convention and calendar-month-end
semantics are not silently compared against differing QuantLib conventions.

All four interpolation and both extrapolation families are tested. Affine-zero
PCHIP identities are complemented by independent turning-point Hermite slopes,
endpoint behavior, shifted-curve reconstruction and diagnostics. Bootstrap cases
retain strict residual repricing; a 1e12-face dated quote explicitly requests four
ULPs of absolute tolerance when the default is below representable price spacing.
The default is not loosened, and independent discount-factor tolerance still applies.

Observed maxima are from `tools/reference_summary.py`, which reruns the stored
assertions and records units/engine-specific errors and tolerance fractions:

| Metric | Maximum absolute difference observed | Units and independent gate |
| --- | ---: | --- |
| Compounding discount factor | 2.22e-16 | Dimensionless; relative 2e-14, no absolute allowance |
| Continuous rate | 1.39e-17 | Annual rate; relative 2e-13 plus absolute 1e-25 |
| Day-count year fraction | 4.16e-17 | Years; exact integer days and schedule dates also required |
| Curve discount factor | 2.22e-16 | Dimensionless; relative 2e-14 plus absolute 1e-15 |
| Mixed-bootstrap discount factor | 1.13e-14 | Dimensionless; strict individual quote repricing also required |
| Bond dirty price / NumPy | 0.06640625 | Currency, at extreme 1e12 face; relative 2e-12 and monetary max(1e-10, face × 1e-12) both pass |
| Bond dirty price / native | 0.0078125 | Same currency/materiality contract |
| Bond DV01 / NumPy | 0.000343323 | Currency per 1 bp; numerical and separate monetary gates |
| Solved yield | 8.29e-13 | Annual rate; absolute 2e-10 and convergence/repricing required |
| Weighted VaR | 0 | Exact discrete loss atom |
| Weighted CVaR | 2.84e-14 | Loss units; separate absolute materiality ceiling 1e-8 |
| ALM liability PV | 0.000122070313 | Currency at large scale; scale-aware numerical/materiality gates |
| Scalar life PV | 1.16e-10 | Currency; separate 1e-7 materiality ceiling |

The full [53 metric/engine summaries](history/hardening-1.1.2/reference-errors.json)
include dated clean/dirty/accrued errors and the worst case identifiers. The
largest currency error must not be interpreted as the error on a 100-face bond.
Reference agreement and Python/native agreement are distinct checks.

## D. Properties and mutation sensitivity

Nine Hypothesis properties cover positive-cashflow price monotonicity,
yield/price inversion, face scaling, normalized sensitivities, key-rate
reconciliation with a finite-difference remainder, scenario permutation/chunking,
zero-shock ALM identities, probability scaling/translation/order, life grouping,
dated settlement, rate conversion and optimization budget feasibility. Several
properties check more than one invariant. The default profile uses 50 examples;
the executed stress profile uses 500, with deterministic generation and no timing
deadline. Two native buffer properties each use 100 generated examples.

Generator assumptions were corrected when business-day adjustments collapsed
short stub dates; these are intentionally invalid schedules, not price failures.
Generated valid settlement schedules use unadjusted conventions, while explicit
independent fixtures cover supported adjustments. Economic key-rate reconciliation
uses the nonzero-bump truncation bound rather than claiming exact equality for
separate central differences. Significant numerical failures remain deterministic
rate, atom, variance, solver, sensitivity and schedule regressions.

The bounded mutation campaign passed its isolated unmodified-source control and
killed **9/9 curated mutants**: discount sign, simple accrual horizon, continuous
conversion sign, VaR tie, CVaR normalization, variance centering, forward/backward
month anchoring and bootstrap repricing. Three initially surviving operators drove
stronger inverse-rate, weighted-variance and wrong-root/repricing tests. This is
a regression-sensitivity sample, not an exhaustive mutation coverage percentage.

## E. Native safety and differential verification

The green implementation CI includes native parity/strict warnings (**85 tests**),
GCC ASan/UBSan (**94 tests**) and Clang ASan/UBSan (**94 tests**), plus standalone
checked-size arithmetic executables. Compiler warning sets include conversion,
sign-conversion and shadow checks. No sanitizer finding was accepted or suppressed
to make a run pass. Leak detection is disabled for the Python sanitizer process;
this is not a leak-freedom or exhaustive C++ branch-coverage claim.

Malformed-input checks cover offsets, dimensions, missing/empty segments,
strides, dtypes, readonly arrays, nonfinite values/results and invalid logical
sizes without physical giant allocations. Input ownership remains with callers;
native returned arrays own storage and outlive temporary inputs. Preflight guards
are per buffer group, not a process RSS promise. External simulator allocation
and Python/third-party MemoryError behavior remain outside a universal wrapper.

GCC/Clang keep `-fno-fast-math -ffp-contract=off`; MSVC keeps `/fp:strict`.
Build type owns optimization, with sanitizer `-O1 -g` and IPO disabled.
[Actual compiler commands](history/hardening-1.1.2/compile-commands.json) preserve
the measured builds rather than guessing flags from compiler names.

## F. MLAE, adaptive VaR and CVaR statistics

[Statistical validation](statistical-validation.md) reports **9600 complete
sampled workflow executions in 96 cells**, 100 seeds each, using actual PennyLane
circuits and independent finite-loss Decimal references. Six empirical and two
structured-factor fixtures span confidence 90–99.9%, 100/500/2000 shots and
power schedules `(0,)`, `(0,1,2)`, `(0,1,2,4)` and ambiguous `(1,)`.

The 80 main cells observed 95–100% marginal interval coverage. The additional
ambiguity study exposes **0/100 CVaR coverage** for a 999999:1 rare tail: exact
CVaR 0.02, reported conditional interval near 1000 with zero width. Its Wilson
upper coverage bound is 3.70%. This result is retained prominently, not hidden by
an average. A separate 9000-fit fixed-binomial MLAE study found guarded coverage
94.5–100%, while raw likelihood-ratio coverage reached 87.5%.

Intervals are local/conditional, not simultaneous workflow guarantees and not
inclusive of deterministic encoding error. Finite toy fixtures do not establish
continuous-tail or production validity. No new algorithm is added to disguise
these limits; rigorous adaptive uncertainty remains future research. Raw records
retain their actual development parent SHAs, dirty flags and dependency versions.

## G. Stable API, serialization and failures

The reviewed 1.1.1 manifest covers **312 exported objects** across `qfin`,
`qfin.finance`, `qfin.compiler` and `qfin.validation`. Tests preserve old imports,
parameter names/kinds/defaults, public methods, positional dataclass fields and
enum values. Six representative nested serialization schemas retain all old key
paths. New provenance is optional and keyword-only; no required positional field
is inserted. Compiler options normalize through an internal frozen typed object
without changing the public `compile` signature.

The eight existing deprecated top-level representation/circuit aliases continue
to warn at the caller's filename and remain through 1.x; no removal is before
2.0. [Public API](public-api.md) lists canonical replacements. Snapshots validate
QFin's contract rather than unstable Python standard-library implementation
signatures. Schema tests are representative key-path guarantees, not a promise
that every future optional diagnostic is byte-for-byte identical.

`QFinError` is the common root. Compatible validation/type/calibration/pricing,
resource, backend, optimization and compilation categories are documented in
[exceptions](exceptions.md), with native translation and explicit validation
under optimized Python. External library exceptions need not all be wrapped.

## H. Reproducibility and ownership

[The reproducibility contract](reproducibility.md) distinguishes repeatable
deterministic calculations from seeded backend-dependent sampling. Quantum
results record versions, resolved device, seed, shots, MLAE schedule,
representation, data qubits, likelihood precision and interval meaning. Metadata
is copied on serialization; modifying a returned report does not modify the
result. Public immutable arrays do not freeze or mutate caller-owned buffers.

Cross-platform floating-point results are compared in numerical and financial
tolerances, not promised bitwise identical. Seeds repeat experiments only within
compatible RNG/device/version/call-sequence conditions. Automatic dispatch uses
cheap static policy and never benchmarks at import or runtime.

## I. Performance and memory

[Performance](performance.md) preserves 33-row before/after matrices, a larger
key-rate sweep, alternating before/after rechecks and 17 fresh-process RSS rows
per version. The initial shared-host twofold timing anomalies did not reproduce
in the controlled-order recheck. The NumPy ALM-path median was about 10% slower;
extra finite/dimension checks remain and timing dispersion is reported. This
does not establish the precise cost of an individual guard.

Representative current times: 1000-bond yield inversion, 0.14132 s NumPy versus
0.02914 s native; 10000-bond/33-node key rates, 0.83591 s versus 0.39072 s. The
standalone-liability native crossover failed revalidation through 262144 flows,
so that automatic choice returns to NumPy. No threading framework or universal
speedup claim is introduced.

Current fresh-process peaks for selected chunk-256 workloads were 131.7 MiB for
NumPy bond batches, 107.5 MiB for NumPy rate scenarios, 110.4 MiB for NumPy ALM
paths and 100.1 MiB for native life scenarios. Most peaks and import floors rose
about 4 MiB; no broad RSS reduction is claimed. Raw rows contain min/median/max,
standard deviation and all timings. [Memory](memory.md) documents asymptotic
storage, chunk controls, full returned aggregates and the 512 MiB preflights.

## J. Packaging, governance, license and publication

CI passed editable source, minimum NumPy/SciPy dependencies, representative
examples, clean-installed sdist, five ordinary wheel jobs, and **nine repaired
portable wheels**: CPython 3.11/3.12/3.13 on manylinux x86_64, macOS arm64 and
Windows AMD64. Portable wheels install in isolation, load the native extension,
run the public smoke example, malformed-buffer tests and independent references.
Those environments prove PennyLane, Qiskit and QuantLib are not core requirements.

The legacy SciPy 1.11.0 minimum-dependency test passed with an inspected PyPI
yank warning: **License Violation**. That job deliberately exercises the declared
compatibility floor; it is not licensing clearance or a recommended installation.
QFin's wheels do not vendor SciPy. Review the supported dependency floor before
publication; the existing dependency contract is preserved in this milestone.

The checkpoint's nine portable wheels and sdist were actually downloaded; archive
and per-package SHA-256 values matched. `release_artifacts.py --collect` verified
and assembled the exact already-tested bytes without rebuilding. The
[recorded digests](history/hardening-1.1.2/artifact-SHA256SUMS-46ef9.txt) and
[build contexts](history/hardening-1.1.2/artifact-contexts-46ef9.json) explicitly
belong to checkpoint `46ef9f8`, not an unpublished later build.

The separate manual release-candidate workflow checks version/tag consistency,
runs reusable CI, verifies these artifact manifests, and is configured to attest
the exact collected bytes. **That manual workflow was not dispatched and no
attestation was produced by this task.** Ordinary CI is read-only with pinned
actions and no persisted checkout credentials; attestation has separately scoped
permissions. [Release engineering](release-engineering.md) documents the process.

**Main protection was not applied.** GitHub reported `protected=false`; the
available connector lacks administration mutations. [Repository settings](repository-settings.md)
lists all 20 exact required checks and the administrator actions to enforce them.
The repository still has **no license file or declared license grant**; the
owner/legal reviewer must select licensing. **No PyPI publication, GitHub Release
or tag was created.** Merge authorization does not imply publication authorization.

## K. Remaining limits and owner actions

- Conditional/local quantum intervals remain unsuitable as workflow-wide
  certainty; the observed ambiguous CVaR failure is an explicit research limit.
- Numerical references validate supported conventions and selected stress grids,
  not general calendars, ex-coupon rules, new products or production actuarial
  equivalence. Native curve restrictions are unchanged.
- Property/mutation/sanitizer campaigns are bounded evidence, not exhaustive
  proofs. C++ branch coverage and Python-process leak freedom are not claimed.
- Classical buffer guards do not bound total RSS, caller inputs or device-managed
  exponentially growing quantum states. RSS was measured on Linux only.
- Portable packaging does not cover macOS Intel, Linux ARM, musllinux,
  free-threaded CPython or unlisted interpreters. Timing conclusions are host and
  workload dependent.
- An administrator must enforce main protection. The owner/legal reviewer must
  decide licensing and review the yanked SciPy 1.11.0 legacy dependency floor.
  Any release/tag/PyPI upload and actual candidate attestation
  require separate execution; they were not performed here.

## L. Commands and verified outcomes

Commands ran in separate baseline/hardened installations, with explicit native
builds and one-thread environment settings for measured work. Paths below are
portable output examples; committed records contain the actual observations.

| Command or check | Observed outcome |
| --- | --- |
| Baseline `pytest --cov=qfin`, then branch coverage | 654 passed, 2 optional skips; 92.9599% lines, 81.0391% branches |
| Strict editable native build via pip/CMake | Success; GCC 13.3.0, C++20, finite-math policy and strict warnings |
| `python -m ruff check .` | Clean |
| `python -m mypy src/qfin` | Success, 71 source files |
| `python -m pytest --cov=qfin --cov-branch --cov-report=json:coverage.json --cov-fail-under=90` | 2232 passed, 2 Qiskit skips; 93.5136% lines, 82.2451% branches locally |
| `python examples/check_coverage.py coverage.json` | All global, numerical line and critical branch gates passed |
| `python -m pytest tests/property --hypothesis-profile=stress` | 9 passed; 500-example profile |
| `python tools/reference_summary.py --output reference-errors.json` | 1454 stored-corpus executions passed; 1073 configurations, 53 metric/engine summaries |
| `python tools/mutation_check.py --output mutations.json` | Unmodified control passed; 9/9 mutants killed |
| API snapshots, serialization, deprecation/ownership, optimized validation tests | Passed in full suite and cross-version CI |
| GCC ASan/UBSan and Clang ASan/UBSan CI | 94 tests each plus checked-size executable, all passed |
| Native parity/strict-warnings CI | 85 tests passed |
| Full sampled statistical study + ambiguity study | 8000 + 1600 actual-circuit runs; known 0/100 conditional failure retained |
| `examples/mlae_benchmark.py --coverage-repetitions 200 --repeats 5` | 9000 fixed-experiment fits and optimizer comparison completed |
| `examples/native_benchmark.py --repeats 3`, plus `--full --section key-rate` | Current and baseline matrix, large key-rate evidence completed |
| `tools/memory_benchmark.py --repeats 3` | 17 fresh-process rows per version, samples and peak RSS recorded |
| `examples/hardening_benchmark.py --repeats 5` | Par-yield/reduction alternatives completed; no unsafe reduction selected |
| `examples/alm_life_benchmark.py --repeats 3` | Completed; supplemental timing excluded from comparison claims because some work overlapped |
| CI test matrix: Linux 3.11/3.12/3.13, macOS/Windows 3.12 | All passed; Linux 3.12 includes Qiskit and QuantLib |
| Minimum dependency / examples / sdist / ordinary and portable wheels | All passed in 20-job checkpoint CI |
| Artifact download, SHA-256 verification and exact collection | Nine portable wheels plus one sdist matched tested manifests |
| `EXPECTED_VERSION=1.1.2 python tools/release_artifacts.py --check-version` | Version accepted; no publication |
| PR review-thread check | No unresolved review threads at the pre-merge audit |

The final-head checks and expected-head merge are completed through GitHub; the
PR and post-merge Actions runs provide the final immutable merge/main provenance.
