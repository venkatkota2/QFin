# Statistical validation of the existing workflows

## Corrected workflow bounds in 1.1.3

VaR/CVaR now budget sampling uncertainty across the full adaptive workflow and
propagate VaR-selection uncertainty into CVaR. The point estimator, schedules,
circuits, seeds, and shot counts are unchanged. Individual `AmplitudeEstimate`
regions remain fixed-experiment diagnostics; the top-level risk interval uses
separate, more conservative bounds derived from the same observations.

For `m` occupied candidate losses, binary search visits at most
`ceil(log2(m)) + 1` objectives, including a selected-point check. Set a fixed
budget `B` to that bound plus one empirical excess objective, or all `r`
structured excess-bit objectives. Each objective gets failure probability
`0.05/B`, split over both tails of every Grover-power binomial observation.
QFin inverts these exact Clopper–Pearson constraints through every sine branch,
preserving amplitude alias modes. Empty intersections revert to full support.
CDF monotonicity converts their simultaneous bounds into a quantile interval;
contradictory bounds also revert to full support, never a selected point.

This is a finite union-bound argument, not a claim inferred from Monte Carlo:
conditional on previous queries, each fresh objective has failure probability
at most `0.05/B` under ideal independent binomial shots. Summing those
conditional failure bounds over at most `B` executed objectives gives at most
0.05 workflow failure probability. Independence *between* the resulting
interval events is not required. See [exact binomial intervals](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html)
and the [Bonferroni principle](https://www.itl.nist.gov/div898/handbook/prc/section4/prc473.htm).
The allocation must use the pre-determined maximum query count, not the
data-dependent number of queries actually executed.

For CVaR at risk confidence `alpha`, define
`g(t) = t + E[(L-t)+]/(1-alpha)`. Discrete expected shortfall, including fractional
mass at a quantile atom, is `min_t g(t)` and a true VaR is a minimizer. This uses
the [Rockafellar–Uryasev objective](https://sites.math.washington.edu/~rtr/papers/rtr179-CVaR1.pdf);
the following finite-support propagation bound is the implementation's derivation.
Every subgradient of `g` lies in `[-alpha/(1-alpha), 1]`. If the VaR interval is
`[a,b]`, the excess-based interval for `g(t_selected)` is `[g_lo,g_hi]`, and

```text
penalty = max(0, t_selected-a, alpha/(1-alpha)*(b-t_selected)),
```

then `[g_lo-penalty, g_hi]` contains CVaR whenever all input bounds cover. Intersect
with the known encoded loss support and the VaR lower bound. An inconsistent
intersection falls back to the entire loss support. This does not use the exact
classical or encoded CVaR to manufacture coverage. Structured excess bits share
the same workflow budget, rather than combining unadjusted marginal intervals.

`interval_semantics.simultaneous_workflow_coverage` is true for these corrected
results **under the stated sampling model**. It excludes deterministic encoding,
quantization, calibration, device noise, model misspecification, and simultaneous
coverage across separate runs/problems. It is not a hardware or regulatory
guarantee. A schedule without power zero can still have non-identifiable modes,
large point error and very wide intervals. The classical error comparison and
`meets_target_error` remain essential; a confidence bound does not repair a bad
point estimate. No new quantum algorithm is introduced.

Regression tests enumerate finite binomial count outcomes, compare power-zero
bounds with SciPy's independent exact test, check discrete CVaR propagation,
exercise contradictory bounds and run the real rare-tail circuits on both
empirical and structured paths. Follow-up measurements are recorded in the
[1.1.3 report](hardening-1.1.3.md).

## Follow-up measurements in 1.1.3

The same eight fixtures, five main configurations and 100 seeds per cell were
rerun through the actual public compiler and PennyLane circuits: **8000 main
executions plus 1600 ambiguity executions**. Minimum per-cell coverage against
the independent Decimal reference was 96% for VaR and 100% for CVaR. Every
ambiguity cell covered in 100/100 executions, including the rare-tail CVaR case
that covered in 0/100 with the old conditional interval.

| Fixture | Main VaR minimum | Main CVaR minimum | Ambiguous VaR | Ambiguous CVaR |
| --- | ---: | ---: | ---: | ---: |
| uniform | 100% | 100% | 100% | 100% |
| two_point | 97% | 100% | 100% | 100% |
| repeated_atom | 96% | 100% | 100% | 100% |
| rare_tail | 100% | 100% | 100% | 100% |
| near_degenerate | 100% | 100% | 100% | 100% |
| symmetric | 100% | 100% | 100% | 100% |
| factor_two_point | 99% | 100% | 100% | 100% |
| factor_uniform | 100% | 100% | 100% | 100% |

These are separate finite Monte Carlo observations, not a pooled or universal
coverage guarantee. The Wilson intervals in each raw row quantify their sampling
uncertainty. The rare-tail ambiguity point estimate remains wrong by about
999.98 loss units; its interval now spans the encoded support, approximately
`[0,1000]`, instead of claiming zero uncertainty. The fix honestly represents
non-identifiability; it does not make an ambiguous schedule identify the truth.

The [main record](history/hardening-1.1.3/statistical-workflows.json) was captured
at clean commit `061c4ce9fd68ac7487380c8628b88e2cf9a45043`, version 1.1.3.
The [ambiguity record](history/hardening-1.1.3/statistical-ambiguity.json) retains
its actual development provenance: parent `f14de6598523ecefc79e925214a926cc720d3e79`,
dirty working tree, version 1.1.3. Both used the corrected risk implementation
committed in `061c4ce`; subsequent Windows-repair and documentation changes do not
change that implementation. Python was 3.12.14, NumPy 2.5.3, SciPy 1.18.1,
PennyLane 0.45.1 and Lightning 0.45.0, with `lightning.qubit`, single-thread
settings and likelihood grid 4097. Per-row elapsed times sum to 520.13 s and
40.84 s respectively; these are not benchmark comparisons.

## Archived 1.1.2 evidence

The experiments below used the **old local/conditional bounds**. They are kept
as the reproducible failure baseline, not relabelled as 1.1.3 results.

## Experiment and provenance

`tools/statistical_validation.py` invokes the public compiler and actual PennyLane
shot circuits. Independent Decimal summation of the finite distribution defines
VaR and expected shortfall, including fractional probability at the VaR atom.
There are six empirical fixtures (uniform, two point, repeated atom, rare tail,
near-degenerate and symmetric), plus structured two-point and uniform factor
fixtures. The rare-tail weights are 999999:1; confidence levels reach 99.9%.

| Risk confidence | Shots per circuit | MLAE powers | Purpose |
| --- | ---: | --- | --- |
| 90% | 100 | 0 | Small-shot boundary/atom behavior |
| 95% | 500 | 0, 1, 2 | Moderate sampling |
| 99% | 2000 | 0, 1, 2, 4 | Tail and multi-power behavior |
| 99.5% | 100 | 0, 1, 2 | Deliberately small-shot extreme quantile |
| 99.9% | 2000 | 0, 1, 2, 4 | Extreme quantile |
| 95% | 100 | 1 only | Aliasing/ambiguous schedule without power zero |

Each distribution/configuration runs both VaR and CVaR for 100 seeds,
71833 through 71932: **80 main cells/8000 executions plus 16 ambiguity cells/1600
executions**, 9600 complete sampled executions. Every cell records error, interval
width, coverage against the true and encoded references, VaR selection frequency,
query counts, wall time and a Wilson 95% interval for the empirical coverage.

These development observations retain their actual provenance. The main study
records parent `7f683ddf8851e0c46278a68a95ba78f040385a38`, a dirty working tree and
package metadata 1.1.1 before the version bump; the ambiguity study records parent
`ef23f99872d0a1e6a16ff01d04b95bdc1b61e1aa`, dirty tree and version 1.1.2. They tested
the new workflow/provenance implementation subsequently committed in this PR;
they are not relabelled as clean final-head runs. Final CI separately exercises
small real-circuit regressions and seeded serialization contracts.

The environment was Python 3.12.14, NumPy 2.5.2, SciPy 1.18.1, PennyLane 0.45.1,
Lightning 0.45.0, `lightning.qubit`, likelihood grid 4097, single-thread settings.
The main campaign took 665.69 seconds; ambiguity took 40.74 seconds. Different
devices need not reproduce the same draws from the same seed.

## Observations

The table shows the minimum empirical interval coverage across the five main
configurations for each family, followed by the one ambiguity configuration.
These are separate cells, not pooled probabilities or universal lower bounds.

| Fixture | Main VaR minimum | Main CVaR minimum | Ambiguous VaR | Ambiguous CVaR |
| --- | ---: | ---: | ---: | ---: |
| uniform | 100% | 100% | 100% | 100% |
| two_point | 97% | 97% | 100% | 100% |
| repeated_atom | 95% | 97% | 100% | 100% |
| rare_tail | 100% | 99% | 100% | 0% |
| near_degenerate | 100% | 100% | 100% | 100% |
| symmetric | 100% | 100% | 100% | 100% |
| factor_two_point | 97% | 99% | 100% | 100% |
| factor_uniform | 100% | 100% | 100% | 99% |

All main cells covered their exact references in 95–100 of 100 executions. This
sample size is modest: 95/100 has a Wilson 95% interval of approximately
88.8–97.8%; even 100/100 only gives a lower bound of 96.3%. Shared seeds and
related fixtures are not treated as independent samples for a pooled guarantee.
Small-shot atom cases can still have large point error even with broad intervals.

**The ambiguity study exposes a serious conditional-interval limitation.** For
losses `[0, 1000]`, weights `[999999, 1]`, confidence 95%, 100 shots and schedule
`(1,)`, the exact VaR is 0 and exact CVaR is 0.02. All 100 runs selected VaR near
1000. The VaR interval spanned the support, covering the truth, but the conditional
CVaR interval collapsed near `[1000, 1000]`: **0/100 coverage**, about 999.98 loss
units of point error, and zero interval width. The Wilson upper bound on observed
coverage is 3.70%. The discrepancy is not encoding error (about 1.7e-17).

Thus a narrow conditional CVaR interval cannot certify overall risk accuracy.
Check classical/encoded comparisons and error budgets as well as interval scope.
The existing `meets_target_error` is based on the classical comparison, not merely
interval width. No workflow-wide interval correction or new quantum algorithm is
introduced; solving adaptive selection and aliasing rigorously remains research
work. Supplying power zero helps identify amplitudes in the tested configurations,
but is not by itself a simultaneous-coverage theorem.

## Fixed-experiment MLAE study

`examples/mlae_benchmark.py` separately exercises 45 fixed amplitude/shot/schedule
cells with 200 repetitions each: **9000 estimator fits**. Guarded coverage ranges
from 94.5% to 100%; raw likelihood-ratio coverage reaches as low as 87.5%. This
study draws the fixed binomial observations directly and must not be described as
9600 additional complete circuit workflows. Guarded regions do not turn adaptive
queries into a simultaneous statement. The 31-threshold optimizer comparison
measured 0.05317 s versus 0.52241 s for a dense reference search (five repetitions),
with maximum amplitude difference 5.66e-6.

## Reproduction and evidence

```bash
python tools/statistical_validation.py --repetitions 100 --output statistical.json
python tools/statistical_validation.py --ambiguity-only --repetitions 100 --output ambiguity.json
python examples/mlae_benchmark.py --repeats 5 --coverage-repetitions 200 --output mlae.md --json-output mlae.json
python -m pytest tests/test_statistical_workflows.py
```

The weekly/manual verification workflow runs both full campaigns. No hardware
jobs, noise-provider changes or new execution backends are included. Finite toy
fixtures do not establish continuous-tail, production actuarial or regulatory
validity, nor quantum advantage.

Raw evidence: [complete main cells](history/hardening-1.1.2/statistical-workflows.json),
[ambiguity cells](history/hardening-1.1.2/statistical-ambiguity.json), and
[fixed-experiment MLAE results](history/hardening-1.1.2/mlae.json).
