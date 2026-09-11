# Curves and scenario semantics

`YieldCurve` stores canonical continuously compounded zero-rate nodes. It accepts
existing zero, discount-factor, adjacent-forward and market-quote inputs, with
explicit compounding, interpolation, extrapolation, day count and valuation date.

A scenario is an additive continuously compounded shock to each zero-rate node.
The stressed curve retains the base interpolation and extrapolation. Its value
is defined by `curve.shifted(shock).discount(cashflow_times)`. Interpolating shocks
separately is not the general contract.

| Interpolation | Quantity interpolated |
| --- | --- |
| `linear_zero` | Continuously compounded zero rates |
| `monotone_zero` | Shape-preserving PCHIP zero rates |
| `linear_discount` | Discount factors |
| `log_linear_discount` | Log discount factors |

NumPy scenarios reproduce all four methods with `flat_zero`, `flat_forward`, or
`error` extrapolation. The last rejects cash flows outside the node domain.
Single-node curves retain the underlying curve's boundary behavior. Native
valuation supports only `linear_zero` with `flat_zero`; explicitly requesting an
unsupported native combination raises an error. The automatic engine falls back
to NumPy. Scenario order only changes output order; chunk size does not change the
financial method.

Node shocks must imply finite positive discounts. Negative zero rates and negative
forwards are economically unusual in some contexts, but are not globally invalid.
Diagnostics distinguish node, between-node and extrapolation findings. The actual
interpolant is inspected for finite positive discounts, increases in discount,
negative forwards and extreme local forwards. Where available, derivatives give
`f(t) = -d log D(t)/dt`. Monotone zero rates do not imply monotone discounts.
Diagnostics sample the interpolants and use interpolation-aware derivatives; they are not a proof of the
absence of every possible numerical pathology at every real-valued time.

Bootstrapping solves for log discounts with adaptive brackets, guaranteeing
positive candidates. Strict quote repricing remains the acceptance condition.
Unsupported or rootless calibration fails clearly. Single-curve deposit,
zero-coupon, bond and simple swap conventions remain unchanged; no new market
model or instrument is introduced.

`curve.explain()` preserves the base input convention, valuation date, day count,
interpolation/extrapolation and immutable provenance for applied shifts and
originating quotes where present. Shifting does not relabel an originally periodic
quote as if the user had entered a continuous quote.

## Independent 1.1.2 checks

The checked-in corpus covers every existing interpolator and extrapolator,
including negative and nearly zero rates and 60-year horizons. Independent
Hermite-polynomial tests exercise turning-point PCHIP slopes; exact node and
shifted-curve scenario semantics remain separately tested. Rate conversions now
avoid a rounded intermediate discount factor and reject unrepresentable results.
Bootstrap success still requires input-quote repricing after root termination.
At trillion-scale prices, an absolute default tolerance smaller than one ULP can
correctly fail; choose an explicit financial-unit tolerance and independently
check the recovered curve. The default has not been loosened.
