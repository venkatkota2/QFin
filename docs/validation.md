# Validation and release engineering

The audited main commit is `3dba3b3803f54891d12ad2c138b4dcc276de9d42`.
The initial audit preceded code changes and covered finance, compiler,
representation, backends, resources, validation, native interfaces, C++ headers,
kernels and bindings, tests, examples, documentation, package metadata, exports,
CLI, benchmarks, CMake and GitHub Actions. The actual build entry point is
`cpp/CMakeLists.txt`; there is no independent top-level CMake project.

After a runner restart removed temporary logs, the untouched main installation
was rechecked on 2026-09-09 in an isolated import path. It reports QFin 1.1.0 from
the baseline installation, with its original native extension: **217 passed,
2 optional tests skipped**, Ruff clean and strict mypy clean on 65 source files.
This repeat is labelled a recovered baseline, not reconstructed original timings.
Baseline line coverage is 87.71% in the current dependency environment.

The hardened local suite reports **654 passed, 2 optional tests skipped**,
**92.96%** line coverage, Ruff clean and strict mypy clean on 67 source files.
Native code builds cleanly with `-Werror -Wshadow -Wconversion -Wsign-conversion`.
These numbers describe the verified Linux Python 3.12 run; optional dependency
availability changes collected/skipped counts on other jobs.

| Area | Measured line coverage | Required Linux numerical gate |
| --- | ---: | ---: |
| Finance | 95.48% | 95% |
| Compiler | 92.57% | 92% |
| Representation | 92.42% | 92% |
| Resources | 90.33% | 90% |
| Backends | 87.15% | 85% |
| Python native loader | 100% | 95% |

The global CI floor rises from the original 78% to 90% after adding meaningful
properties, failure paths and numerical boundary tests. The table is Python line
coverage, not a claim of C++ line or branch coverage. Native numerical coverage is
supported by differential tests and sanitizers; no artificial 100% claim is made.

## Numerical regression barriers

- Every supported scenario interpolation/extrapolation is checked against slow
  shifted-curve repricing, including negative/zero rates, boundaries, singleton
  curves, cash-flow positions, shock order and chunk invariance.
- Dated settlement, accrual and clean/dirty prices pass through fixed income, ALM,
  scenarios and loss conversion. Scale, duration-gap and funding identities are
  tested, including tiny nonzero portfolios.
- Yield round trips, analytical zero coupons, numerical duration/convexity,
  key-rate/parallel reconciliation and 70-digit Decimal tiny-exposure checks
  complement NumPy/native parity.
- Singular covariance, budget-neutral zero-risk return directions, duplicated
  assets, bounded allocation and KKT conditions have independent regressions.
- Exact life grouping, count scaling, zero mortality/lapse, scenario chunking and
  cash-flow aggregation are tested.
- Weighted loss normalization, permutations, high-confidence tails, cancellation
  and financial-materiality tolerances test risk calculations.
- Array ownership, integer controls, direct factories, zero/unknown error metadata,
  units, aliases, date endpoints and invalid probabilities have explicit tests.
- Dense, structured and compressed circuits are compared on default.qubit and
  lightning.qubit for seeded existing option objectives. Repeated seeded shot
  schedules reproduce results. Existing structured factor circuits retain their
  encoded-grid oracles and resource guards.
- MLAE validation includes a dense likelihood reference, ambiguous modes,
  boundaries, narrow peaks and a 9,000-fit [coverage study](mlae-validation.md).

`FinancialTolerance` requires numerical and financial bounds simultaneously.
Named units cover price, PV, DV01, duration, convexity, loss statistics and
probabilities. Independent QuantLib tests run in the Linux Python 3.12 CI job;
QuantLib remains optional at runtime.

## CI matrix

The workflow runs Python 3.11/3.12/3.13 on Linux plus Python 3.12 on macOS and
Windows. Dedicated jobs enforce Ruff/strict mypy, strict native warnings/parity,
ASan+UBSan, minimum dependencies, representative examples, clean sdist installation
and actual wheel installation on the same five OS/Python combinations.

The minimum environment installs **NumPy 1.26.0 and SciPy 1.11.0 on Python 3.11**
with no PennyLane dependency. Latest-supported environments install bounded current
extras. Every wheel smoke run imports the installed package, verifies its native
extension, prices bonds natively and with NumPy, runs classical risk and optimizer
calculations, exercises factorized compilation and `system_info`, and verifies
PennyLane/Qiskit/QuantLib remain absent. Checksums accompany the artifacts.

Sanitizers use GCC on Linux with `-fsanitize=address,undefined`, frame pointers and
`halt_on_error=1`. Leak detection is disabled for third-party Python-runtime
allocations; this does not disable address/undefined-behavior checks. Tests include
malformed buffers, empty segments, random parity, retained NumPy-owned outputs,
metamorphic identities and cancellation.

All 16 jobs passed on checkpoint `d657075dba5cedd0d7164883fd8c29a0ffe9f4f0`.
The final numerical gates pass locally. Final PR checks must also pass for the complete report/verification commit; inspect
[PR #14](https://github.com/venkatkota2/QFin/pull/14) for the exact final head and
merge state. [Repository protection settings](repository-settings.md) require
maintainer administration access and are not falsely reported as configured.

The [hardening report](hardening-report.md) records repaired behavior, measured tradeoffs, API changes and limitations. [Machine-readable validation](hardening-validation.json) records the fresh local coverage run; earlier accumulated coverage is not used for final acceptance.
