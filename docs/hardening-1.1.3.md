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

The first complete local run passed **2298 tests**, with two optional Qiskit
skips. Line coverage was **93.61%**, branch coverage **82.44%**; all existing
module/branch gates passed. The new risk-uncertainty helper has 100% line and
branch coverage. Ruff and strict mypy passed (72 source files). These are local
development observations, not claims about a later CI head. The PR records the
final tested commit and CI conclusions. Large real-circuit reruns and paired
performance evidence are recorded as they complete.

Reproduction:

```bash
python -m pytest --cov=qfin --cov-branch --cov-report=json:coverage.json
python examples/check_coverage.py coverage.json
python -m ruff check .
python -m mypy src/qfin
python tools/statistical_validation.py --ambiguity-only --repetitions 100 --output ambiguity.json
python tools/statistical_validation.py --repetitions 100 --output statistical.json
python tools/benchmark_alm_reuse.py --baseline /path/to/f14de65-checkout --output alm.json
```

## Owner-controlled issues

The repository still has no license file or declared license. Choosing MIT,
Apache-2.0, proprietary terms, or another license requires the owner's decision;
the SciPy fix does not grant rights to QFin. `main` was still unprotected on the
2026-09-12 read-only check. The connected GitHub tools do not expose repository
administration; the [required checks and rules](repository-settings.md) remain
instructions for an administrator, not enforced settings.

No tag, GitHub Release, PyPI upload or manual attestation was created. Windows
runtime version checks do not replace redistribution review or certify every
host/DLL load order. The library remains research-grade.
