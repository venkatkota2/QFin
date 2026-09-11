# Asset-liability modelling

`AssetPortfolio` holds existing fixed-rate bonds, quantities, aggregate equity
and cash, and the same `Settlement` information used by fixed income.
`LiabilityPortfolio` holds deterministic cash-flow streams and existing inflation
linkage. Cash-flow times share the curve's valuation origin.

`ALMModel.evaluate()` calculates asset and liability PV, surplus, funding ratio,
duration and convexity gaps. `run_scenarios()` stresses zero-rate nodes with the
[exact curve contract](curves.md). `loss_distribution()` converts scenario surplus
to loss as `base_surplus - stressed_surplus`. Zero shocks reproduce base valuation.

For nonzero denominators, funding ratio is `A/L` and duration gap is
`D_assets - (L/A) * D_liabilities`. Scaling both sides leaves the ratio and gap
unchanged and scales surplus. Empty/zero-value portfolios use explicitly handled
zero-denominator outcomes; tiny nonzero PVs are not rounded into zero portfolios.

Dated bonds can flow through asset construction, base evaluation, rate scenarios
and loss conversion with omitted settlement when the curve has a valuation date.
Unsupported mixtures or inconsistent dates fail at normalization rather than
being interpreted as numeric time.

Existing economic scenarios include rates, credit spread, equity returns,
inflation, mortality and lapse. One-period factor attribution and multi-period
roll-forward retain their existing model ordering. Multi-period paths remain a
simplified research model; they are not a complete enterprise ALM engine. See
[the original model ordering](alm-life-0.6.md).

Scenario chunks bound temporary valuation matrices. Outputs are aggregates, not
scenario-by-instrument-by-period cubes. Local seeded Gaussian generators and
probability-aware loss conversion are reproducible. Native support remains
restricted to the curve methodology it implements exactly; NumPy remains the
reference for all supported interpolation modes.

## Reliability and memory

Valuation and path results reject overflowing finite-input monetary calculations.
An infinite funding ratio remains intentional for zero liabilities. Public
scenario chunk sizes do not bound the size of the full returned path matrix; see
[the memory contract](memory.md) and [exception compatibility](exceptions.md).
