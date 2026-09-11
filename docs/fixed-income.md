# Fixed income

`FixedRateBond` supports the existing floating and dated schedule models.
`price_bonds` returns clean/dirty prices, accrued interest and named risk measures.
`price_bonds_from_yield` and `yield_from_prices` share settlement and coupon
frequency conventions. [Settlement rules](financial-conventions.md) also apply
when bonds enter `AssetPortfolio` and ALM scenarios.

| Measure | Meaning |
| --- | --- |
| YTM Macaulay/modified duration | Time-weighted PV / first YTM derivative under coupon-frequency compounding |
| Parallel-zero duration | Sensitivity to a continuous parallel zero-rate shift |
| Effective duration/convexity | Explicit finite central-bump sensitivity |
| DV01/PV01 | Currency sensitivity normalized to a one-basis-point rate move |
| Spread duration / CS01 | Continuous spread sensitivity / one-basis-point spread move |
| Key-rate duration / DV01 | One zero-rate node bumped at a time, retaining interpolation |

`key_rate_risk` prepares the bond batch once and computes all up/down node and
parallel shocks from those buffers. The public result and finite-bump meaning
remain unchanged. For linear-zero/flat-zero curves, `expm1` computes each cash-flow
PV change without subtracting large nearly equal bond prices. Interpolation basis
weights are evaluated directly to retain tiny exposures near curve nodes.

Summed key-rate sensitivities reconcile with parallel sensitivity to finite-bump
accuracy for linear-zero methodology. Nonlinear interpolation and finite bumps
need not produce exact additivity; the report includes separately computed
parallel DV01 so the difference can be assessed.

`par_yield` uses a single schedule. If `P0` is principal PV, `C1` is unit-coupon PV
and `AI1` is unit-coupon accrued interest, the coupon is
`(target_clean_price - P0) / (C1 - AI1)`. It is an affine solution, not a root
solver, and is independent of face-value scaling. A zero coupon coefficient is
undefined and raises an error.

Yield inversion uses a bracketed solver and stops on an exact price residual or a
yield-space bracket-width condition. It does not turn a small currency residual
into a false claim of highly accurate yield for a very short bond. Solver results
retain convergence information. Negative yields are accepted only above the
periodic compounding lower boundary.

Analytical, finite-difference, Decimal and independent QuantLib tests validate
these measures. See [validation](validation.md) for tolerances and
[historical convention details](fixed-income-1.1.md) for schedule examples.

## 1.1.2 convention and precision corrections

Forward/backward schedules derive each regular boundary from the original anchor;
February clamping no longer shifts all subsequent coupon dates. Adjustments that
collapse adjacent dates still fail explicitly. DV01/CS01 and effective sensitivities
use stable `sinh`/`expm1` differences, including short maturities at large notionals.
Dated price/accrual references use QuantLib only with matching clocks, coupon
conventions, adjustment and settlement. See [independent validation](validation-1.1.2-spec.md).
