# QFin 1.1.1 hardening checkpoint

This is an in-progress engineering record, not a release-readiness declaration.
No modelling features are added during this work.

## Branch and audited base

- Branch: `hardening/qfin-1.1.1`.
- Audited main: `3dba3b3803f54891d12ad2c138b4dcc276de9d42`.
- Main was checked again on 2026-09-08 and remains at that commit.
- H1 numerical integrity: `db0a2194f0956bd711a8ae97e1517c60f3c0966e`.
- H2 measured kernel changes: `5320913008095e6107a7f2e23182ae17aff544bb`.
- H3 ownership and API validation: `963197572622e0c73889a3c6f5e9060851f662e6`.

The initial audit and baseline runs preceded H1. After a runner outage, the three
commits and early H4 work survived, while later uncommitted release work and some
temporary logs did not. Verification below records runs from the recovered
workspace; unavailable raw baseline logs must not be reconstructed as measurements.

## H4 verified locally on 2026-09-08

- Python 3.12.13, GNU C++ 13.3.0, Linux x86_64.
- Ruff passes; strict mypy passes all 67 source files.
- Full suite: 434 passed, 2 optional tests skipped; line coverage 89.82%.
- Strict native warning build passes with `-Werror -Wshadow -Wconversion
  -Wsign-conversion`.
- ASan/UBSan installed-wheel native and metamorphic tests: 39 passed, with
  `halt_on_error=1`; leak detection disabled because third-party Python runtime
  allocations are outside this check. The imported extension path was verified.
- sdist and native wheel builds pass. Clean wheel smoke validation is being
  completed separately from editable-install tests.
- MLAE: 9,000 seeded fits, 45 amplitude/schedule/shot configurations. Guarded
  empirical 95% region coverage ranges from 94.5% to 100%; raw LR coverage from
  87.5% to 100%. Each row has 200 replications, so coverage itself has sampling
  uncertainty. See `mlae-validation.md` and the accompanying JSON.
- The 31-fit MLAE benchmark measures 0.517088 s for the dense reference and
  0.052363 s for the hardened estimate including confidence regions. These are
  environment-specific medians of five repetitions.

## Work still required before merge

- Complete and review the full current-version performance matrix and dispatch
  evidence; preserve numerical differences and per-case repetition counts.
- Finish current documentation, compiler-contract tests, targeted boundary
  coverage, and release report.
- Exercise actual wheel/sdist installations and minimum dependency environment.
- Push the hardening branch and create a draft PR; resolve every required CI
  failure before marking ready or merging.
- Record exact PR/check/merge SHAs after verification.

## Repository administration and licensing

Main currently has no branch protection. The connected GitHub App excludes
administration access, so an administrator must configure a main ruleset requiring
pull requests and successful CI, disallowing force pushes and deletion, and
preventing ordinary administrator bypass. Exact check names will be recorded after
the workflow matrix runs.

The repository has no license grant. No license is invented or added. This is an
adoption/reuse blocker for parties who need an explicit grant of rights; the
maintainer must choose and authorize any license before publishing such a grant.
