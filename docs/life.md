# Life projection

The existing annual-step projection supports term, participating, universal-life
and annuity model points. This hardening milestone adds no insurance products,
mortality models or stochastic families.

`MortalityTable`, `LifePolicy`, `ProjectionAssumptions`/`LifeAssumptionSet` and
`PolicyModelPointSet` describe the supported model. Exact grouping of identical
policies preserves aggregate expected cash flows. Model-point counts scale
exposure and cash flows; they do not change per-policy economics.

Premiums and expenses occur at the start of each policy year and benefits at year
end. Mortality precedes disability/recovery and lapse. Active, disabled and dead
states, credited account values, bonuses, surrender and inflation-linked benefits
retain the [existing explicit ordering](alm-life-0.6.md).

`project_liabilities` returns aggregate and per-policy results.
`project_liability_scenarios` chunks scenarios and model points and returns
scenario aggregates that can feed `LossDistribution`. No full
scenario-by-model-point-by-year tensor is required. Zero mortality, zero lapse,
identical-policy grouping and exposure scaling have analytical regression tests.

Native loops accelerate policy-by-year arithmetic when compatible. Standalone
mortality interpolation stays in NumPy. Substantive NumPy analytics added to a
native cash-flow projection are reported as `mixed`; inspect each result's engine
metadata rather than inferring execution from installation status.

This is an experimental annual model with simplified products, transitions and
assumptions. It is not calibrated by default and is not a replacement for
commercial actuarial platforms such as Prophet, AXIS or PathWise. Published
speedups describe measured model workloads, not production actuarial equivalence.

## Reliability and memory

Projection horizons and major output groups are checked before allocation.
Unrepresentable benefits, cashflows or present values raise explicit validation
errors in both execution paths. Model-point aggregation retains the existing
annual survival/lapse timing; no product mechanics are added. Independent scalar
recursion and grouping properties support the [1.1.2 verification](validation-1.1.2-spec.md).
