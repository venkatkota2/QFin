# Financial conventions

Curve time is measured from the curve valuation date using its day count. Bond
coupon accrual uses the bond's own day count and unadjusted schedule boundaries;
payment dates follow its calendar and business-day convention. These clocks can
differ intentionally. Dates use Python `datetime.date` or ISO date strings.

| Instrument type | Accepted settlement |
| --- | --- |
| Floating-time bond | Non-negative finite numeric time; omitted means zero |
| Dated bond valued on a dated curve | Date equal to the curve valuation date; omitted uses that date |
| Dated bond without a curve, such as YTM pricing | Explicit date; omitted uses issue date |

A dated bond on a curve without a valuation date cannot be interpreted by dated
curve pricing. Mixed dated/floating batches, numeric settlement for dated bonds,
date settlement for floating bonds and a settlement date different from the
curve date are rejected. `AssetPortfolio` reuses the fixed-income `Settlement`
type and the same normalization. There is no second ALM settlement model.

Only payments after settlement are included. Floating-time schedule filtering
uses the existing `1e-12` year boundary tolerance; coupon-date settlement excludes
the coupon on that date. Clean price plus accrued interest equals dirty price.
Prices and accrued interest are currency amounts for the supplied face value,
not implicitly prices per 100. Rate inputs use decimal fractions, so `0.01` is
one percent and `1e-4` is one basis point.

Supported day counts are ACT/365 Fixed, ACT/360, ACT/ACT ISDA, 30/360 US/NASD and
30E/360. Calendars contain explicit holidays and weekends; they are not a market
holiday feed. Rates may be continuous, periodic or simple, but conversion must
preserve a finite positive discount factor. Negative rates are allowed within
each compounding convention's mathematical domain.

Liability cash-flow times are relative to the same valuation origin as the curve.
Scenario shocks always act on canonical continuously compounded zero-rate nodes,
regardless of the convention used to enter the original quotes.

See [curves](curves.md), [fixed income](fixed-income.md) and the preserved
[dated financial foundation](financial-conventions-1.1.md) for detailed schedule
examples and the originally supported convention set.
