# Financial conventions foundation

This document defines the convention layer introduced for QFin 1.1. It is the
first financial-accuracy slice; it does not claim that dated bond settlement,
curve bootstrapping, or independent QuantLib validation are complete.

## Dates and calendars

`ValuationDate`, `Calendar`, and `Schedule` use Python's `datetime.date` as the
underlying representation. APIs accept `date`, `datetime`, or ISO `YYYY-MM-DD`
strings and normalize them once at construction.

A `Calendar` contains an explicit holiday set and weekend weekday numbers.
QFin ships no implicit jurisdictional holiday database. The supported
business-day conventions are:

- unadjusted;
- following;
- modified following;
- preceding; and
- modified preceding.

`Schedule` takes a payment frequency per year that divides 12. It supports
forward/backward generation, month-end preservation, a separately configurable
termination convention, and explicit first and next-to-last coupon dates. The
explicit boundaries represent short or long stubs without guessing contract
intent.

## Day counts

The implemented conventions are:

| QFin name | Definition |
| --- | --- |
| `ACT/365 Fixed` | actual calendar days divided by 365 |
| `ACT/360` | actual calendar days divided by 360 |
| `ACT/ACT` | ISDA split-year calculation using 365 or 366 per calendar-year segment |
| `30/360` | US/NASD month-end and February rules, divided by 360 |
| `30E/360` | each day-of-month capped at 30, divided by 360 |

`day_count` returns the signed numerator and `year_fraction` returns the signed
fraction. Reversing dates reverses the sign.

## Rate quotes

`RateQuote` always carries its compounding convention. Supported conventions
are continuous, annual, semiannual, quarterly, monthly, and simple. Conversion
preserves the discount factor over the explicitly supplied horizon. The horizon
matters for simple-rate conversion.

Periodic quotes require `1 + r/m > 0`; simple quotes require `1 + r*t > 0`.
Negative rates within those mathematical domains are supported.

## Yield curves

`YieldCurve` can be constructed from:

- zero-rate nodes;
- positive discount-factor nodes;
- adjacent-interval forward-rate quotes; or
- homogeneous direct `CurveMarketQuote` zero-rate/discount-factor nodes.

Direct market nodes are not bootstrapped instruments. A deposit, bond, or swap
bootstrap must solve nodes and report residual repricing errors; that is a
separate 1.1 deliverable.

QFin stores canonical continuously compounded zero-rate nodes because existing
scenario and C++ kernels define additive rate shocks in that convention. The
source quote convention remains available as `quote_compounding`, and
`quoted_zero_rate` performs explicit output conversion. `explain()` reports
both conventions.

The validated interpolation choices are:

- linear continuously compounded zero rates;
- linear discount factors;
- log-linear discount factors; and
- monotone PCHIP continuously compounded zero rates.

Extrapolation is separately configured as flat zero, flat adjacent forward, or
error. All methods recover their supplied nodes in the analytical tests.

Current native curve kernels implement linear continuous-zero interpolation
with flat-zero extrapolation. `engine="auto"` therefore selects the NumPy
reference for other curve methods, while `engine="native"` raises a clear
error. This is intentional: backend selection may not change financial
semantics.

`diagnostics()` reports adjacent continuously compounded forwards, increasing
discount-factor intervals, negative-forward intervals, and warnings. Negative
rates are supported, so a warning is diagnostic information rather than an
automatic calibration failure.

## Limitations and next work

- Holiday data must be supplied by the caller.
- `ACT/ACT` currently means ISDA; ICMA and AFB variants are not implemented.
- Market-node input is not an instrument bootstrap.
- Existing `FixedRateBond` remains a floating-time instrument until the dated
  bond work is completed.
- Native kernels do not yet implement discount-factor or monotone interpolation.
