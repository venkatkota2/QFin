# Explicit limitations

QFin remains experimental and research grade. Hardening numerical contracts does
not establish regulatory approval, market calibration, financial suitability or
production equivalence to commercial pricing/actuarial platforms.

- Native curve kernels support linear-zero interpolation with flat-zero
  extrapolation. Unsupported explicit native requests fail; automatic dispatch
  uses NumPy where supported.
- Dated curve settlement must equal valuation date. Mixed dated/floating batches
  and inconsistent settlement clocks are unsupported.
- Calendars are explicit supplied holiday sets, not maintained market calendars.
  Bootstrapping retains the existing simple single-curve instrument conventions.
- Life projection uses annual steps and simplified existing products and
  transitions. It is not a Prophet/AXIS/PathWise replacement.
- Gaussian economic/factor scenarios are transparent assumptions, not calibrated
  tail models. Bootstrap intervals do not include scenario-design or model risk.
- Quantum workflows run on tested PennyLane simulators. There is no claim of
  quantum advantage, production hardware runtime, or fault-tolerant resource cost.
- Generic empirical quantum loading requires exponential grid resources. Existing
  structured factor arithmetic supports restricted affine grids and sparse
  exposures with explicit validation and resource limits.
- MLAE reports a guarded likelihood-based confidence region for each fixed
  experiment. Separate 1.1.3 workflow bounds account for adaptive VaR selection
  and all excess queries under ideal binomial shots. They do not cover encoding,
  device noise, model risk or simultaneous comparisons across independent runs.
  Ambiguous schedules may still yield inaccurate point estimates and wide bounds.
- No QSVT, block-encoding implementation, new stochastic models, new derivatives,
  new quantum backends or new hardware-provider execution is added here.
- Native execution is single threaded. Benchmarks depend on the CPU, compiler,
  dependency versions and thread environment. Tiny timing wins are not portable
  acceleration promises.

The repository currently contains no license grant. This is an adoption/reuse
blocker where an explicit grant is needed. The maintainer must authorize a license;
none is invented by the hardening work. Apache-2.0, MIT or proprietary terms are
possible choices with different obligations; selecting one is a maintainer/legal
decision, not a technical default.

## Operational boundaries

The sampling guarantee follows its stated mathematical assumptions, not a
generalization from finite empirical studies.
Memory guards are per operation/group, not a process RSS guarantee. Tested portable
wheels cover Linux x86_64, macOS arm64 and Windows AMD64 on CPython 3.11–3.13.
Manual release attestation is configured separately from normal CI; no publication
is implied. Main protection requires repository administration, and licensing
remains unresolved. See [the follow-up report](hardening-1.1.3.md).
