# Statistical validation of the existing workflows

This is empirical evidence for research workflows. A reported 95% interval is
not a demonstrated simultaneous 95% guarantee for adaptive VaR/CVaR. CVaR
intervals condition on the selected VaR; deterministic encoding error is separate.
The 1.1.2 result serializers state these meanings explicitly.

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
