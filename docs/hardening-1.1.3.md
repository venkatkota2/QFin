# QFin 1.1.3 — follow-up issue fixes

Starting main: `f14de6598523ecefc79e925214a926cc720d3e79` (PR #15).
This patch fixes the technical issues identified in the 1.1.2 report. No new
financial model, quantum circuit/algorithm, backend, release or license is added.

## Changes

| Issue | Correction | Verification |
| --- | --- | --- |
| CVaR could report zero uncertainty after selecting the wrong VaR | Budget exact-binomial bounds over every possible adaptive CDF/excess query, then propagate quantile-selection uncertainty into expected shortfall | Exhaustive finite-count coverage, analytical finite-loss bounds, contradictory-region fallback, real empirical/factorized rare-tail circuits |
| Legacy minimum CI installed yanked SciPy 1.11.0 | Metadata and minimum-version CI now require 1.11.1, upstream's licensing-fix patch | Dependency contract test and minimum-version CI |
| Windows wheel repair selected a 14.40 C++ runtime for a 14.51-linked extension | Select an installed compatible Visual Studio CRT, fail on repair warnings, inspect bundled PE versions and archive runtime hashes | Synthetic old/missing/wrong-architecture DLL regressions and installed portable wheel CI |
| NumPy ALM paths repeated each closing valuation at the next opening | Reuse the preceding closing bond index and short rate | Public path parity, chunk invariance, exact valuation-call regression and alternating baseline/candidate benchmark |

The [statistical derivation](statistical-validation.md) states the assumptions:
one encoded workflow, ideal independent binomial shots conditional on earlier
queries, no encoding/model/device-noise guarantee. The point estimator is
unchanged: ambiguous schedules can still have large point errors and must fail
`meets_target_error` when the classical discrepancy is excessive. Wide intervals
are deliberate when the data cannot distinguish possible amplitudes.

Public signatures, positional fields and existing serialized keys remain intact.
Result provenance additionally records the maximum objective budget, allocated
failure probability and sampling-interval method. Local `AmplitudeEstimate`
regions retain their old meaning; corrected top-level intervals are explicitly
labelled. No exact reference risk value is used to manufacture an interval.

## Verification checkpoint

The 2026-09-13 local run passed **2299 tests**, with two optional Qiskit
skips. Line coverage was **93.61%**, branch coverage **82.44%**; all existing
module/branch gates passed. The new risk-uncertainty helper has 100% line and
branch coverage. Ruff and strict mypy passed (72 source files). The 500-example
Hypothesis stress profile plus deterministic metamorphic regressions passed
all 26 selected tests. These are local observations at implementation checkpoint
`f1afad079dcf3e67ff5ea540e6a8d3ebe1f61e3c`, not claims about a later CI head.
The [PR #16 timeline](https://github.com/venkatkota2/QFin/pull/16) records final
tested commits, all 20 cross-platform CI conclusions and artifact verification.

**Statistical rerun:** 80 main cells and 16 ambiguity cells, 100 repetitions each,
total **9600 actual sampled circuit workflows**. Every main cell covered in
96–100 of 100 runs; every ambiguity cell covered in 100/100. The previously
failing rare-tail CVaR case changed from 0/100 to 100/100 coverage by preserving
quantile uncertainty, not by improving the still-ambiguous point estimate.
The [detailed table](statistical-validation.md#follow-up-measurements-in-113)
and [raw records](history/hardening-1.1.3/README.md) distinguish the clean main
study from development captures and retain per-cell Monte Carlo uncertainty.

**Performance:** same-interpreter alternating baseline/candidate blocks showed
39.5% and 49.6% lower median NumPy ALM-path times at chunks 16 and 256 respectively
for 1000 scenarios × 20 periods × 20 bonds. All nine output arrays were bitwise
identical across versions, chunks and blocks. This is workload-specific evidence,
not a universal speedup; see [timings, dispersion and provenance](performance.md).

Reproduction:

```bash
python -m pytest --cov=qfin --cov-branch --cov-report=json:coverage.json
python examples/check_coverage.py coverage.json
python -m ruff check .
python -m mypy src/qfin
HYPOTHESIS_PROFILE=stress python -m pytest tests/property tests/finance/test_metamorphic_properties.py
python tools/statistical_validation.py --ambiguity-only --repetitions 100 --output ambiguity.json
python tools/statistical_validation.py --repetitions 100 --output statistical.json
python tools/benchmark_alm_reuse.py --baseline /path/to/f14de65-checkout --output alm.json
```

## Owner-controlled issues

The repository still has no license file or declared license. Choosing MIT,
Apache-2.0, proprietary terms, or another license requires the owner's decision;
the SciPy fix does not grant rights to QFin. `main` was still unprotected on the
2026-09-13 read-only check. The connected GitHub tools do not expose repository
administration; the [required checks and rules](repository-settings.md) remain
instructions for an administrator, not enforced settings.

No tag, GitHub Release, PyPI upload or manual attestation was created. Windows
runtime version checks do not replace redistribution review or certify every
host/DLL load order. The library remains research-grade.
