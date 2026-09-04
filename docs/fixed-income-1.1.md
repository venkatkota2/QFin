# Fixed-income accuracy in QFin 1.1

QFin 1.1 adds dated bond semantics and independent validation while preserving
the floating-year APIs and C++ kernels from earlier releases. Accuracy takes
priority over backend selection: unsupported curve/native combinations fall
back under `engine="auto"` and fail explicitly under `engine="native"`.

## Dated fixed-rate bonds

`FixedRateBond` accepts either the original floating maturity or an issue and
maturity date:

```python
bond = qfin.FixedRateBond(
    issue_date="2026-01-31",
    maturity_date="2031-01-31",
    coupon_rate=0.045,
    frequency=2,
    day_count="30/360",
    business_day_convention="modified_following",
    termination_convention="modified_following",
    date_generation="backward",
    end_of_month=True,
    first_coupon_date=None,
    next_to_last_coupon_date=None,
)
```

Coupon amounts use the bond day count between unadjusted accrual boundaries.
Payments use adjusted dates. Settlement excludes payments on or before the
settlement date. Accrued interest is the coupon-rate accrual from the preceding
unadjusted boundary and resets on a boundary. Clean price equals dirty price
minus accrued interest.

The optional first and next-to-last coupon dates define short or long stubs
without QFin inferring contract intent. Ex-coupon periods are intentionally not
part of 1.1.

A dated curve price requires settlement to equal the curve valuation date. If
settlement is omitted, QFin uses that valuation date. This explicit restriction
prevents a curve anchored on one date from being silently applied as though it
were anchored on another.

## Yield and curve pricing

`price_bonds_from_yield` uses nominal annual yield compounded at the bond coupon
frequency. It supports negative yields while `1 + yield / frequency > 0`.
`yield_from_prices` solves the same convention, so price/yield round trips do
not cross conventions.

`price_bonds` discounts deterministic cash flows from `YieldCurve`. An optional
`z_spread` is additive to the continuously compounded zero rate. `par_yield`
returns the coupon rate that makes clean price equal face value, or another
explicit target clean price.

## Sensitivity definitions

`BondBatchAnalytics.methodology` distinguishes curve and YTM calculations.
The explicitly named outputs are:

| Output | Definition |
| --- | --- |
| `ytm_macaulay_duration` | present-value-weighted time under nominal YTM |
| `ytm_modified_duration` | Macaulay duration divided by `1 + y/frequency` |
| `ytm_convexity` | analytical second-order sensitivity to nominal YTM |
| `parallel_zero_duration` | analytical sensitivity to an additive continuous parallel zero-rate shift |
| `effective_duration` | central bump-and-revalue duration using a one-basis-point bump |
| `effective_convexity` | central bump-and-revalue convexity using a one-basis-point bump |
| `spread_duration` | analytical sensitivity to additive continuous z-spread |
| `dv01` / `pv01` | currency price increase for a one-basis-point rate decrease, centrally measured |
| `cs01` | currency price increase for a one-basis-point continuous spread decrease |

For deterministic cash flows, a parallel continuous zero shift and additive
continuous z-spread have the same local mathematics. They remain separate
fields because they represent different risk-factor interpretations.

Legacy `macaulay_duration`, `modified_duration`, and `convexity` fields remain
for compatibility. For curve pricing the legacy duration fields contain the
parallel-zero duration. New code should use the explicit fields.

`key_rate_risk` applies a central shock to one continuously compounded curve
node at a time while preserving interpolation and extrapolation. It reports a
bond-by-node matrix of DV01 and duration plus a separately computed parallel
DV01 reconciliation value.

## Instrument bootstrapping

`bootstrap_curve` orders instruments by maturity and solves one positive
discount-factor node at each maturity. Deposits and zero coupons imply nodes
directly. Fixed-rate bonds and simple fixed-for-floating swaps use a bounded
Brent root. The initial swap model is spot-starting and single-curve, with
floating-leg PV equal to `1 - DF(maturity)`.

The returned `CurveBootstrapReport` contains:

- instrument type, identifier, input quote, model quote, and residual;
- node times, discount factors, continuous zero rates, and adjacent forwards;
- interpolation, extrapolation, valuation date, and tolerance metadata; and
- an overall success flag that is returned only after all residuals pass.

Duplicate maturity nodes are rejected. Non-positive directly implied discount
factors, unbracketed roots, and residual failures raise `CurveBootstrapError`.

## Independent and financial-unit validation

The normal test suite includes analytical formulas, immutable golden bond
cases, invariants, yield round trips, and an independently structured scalar
reference. The optional `validation` dependency installs QuantLib and checks a
controlled dated bond's independently generated schedule, clean/dirty price,
accrued interest, solved yield, Macaulay and modified duration, convexity, and
DV01.

```bash
python -m pip install -e ".[validation]"
```

QuantLib is a development validator only and is never used by QFin runtime
pricing. `FinancialTolerance` and `validate_financial_values` combine absolute,
relative, and named financial-unit thresholds. Failed reports include actual,
expected, difference, unit, and allowed error rather than only a machine-scale
assertion.

## Current boundaries

- Holiday sets are supplied by users; QFin does not imply a jurisdiction.
- `ACT/ACT` is ISDA, not ICMA or AFB.
- Ex-coupon logic is deferred.
- The bootstrap is single-curve and does not claim OIS, credit, or inflation
  calibration.
- Key rates are curve-node shocks, not user-defined tenor buckets.
- QuantLib remains optional and is exercised in one Linux CI job.
