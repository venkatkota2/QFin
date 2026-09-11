"""Regenerate independent fixtures. Intentionally imports neither QFin nor SciPy.

Decimal arithmetic uses 70 digits. QuantLib supplies dated convention references.
Changing this generator requires review of the financial formulas and provenance.
"""

from __future__ import annotations

import itertools
import json
from datetime import date
from decimal import Decimal, getcontext
from pathlib import Path

import QuantLib as ql

getcontext().prec = 70
D = Decimal
ROOT = Path(__file__).resolve().parents[1] / "tests/reference_data"


def save(name, cases):
    for index, case in enumerate(cases):
        case["id"] = f"{name}-{index:03d}"
    (ROOT / f"{name}.json").write_text(json.dumps(cases, indent=2) + "\n")
    print(name, len(cases))


def qdate(text):
    value = date.fromisoformat(text)
    return ql.Date(value.day, value.month, value.year)


def discount(rate, time, compounding):
    r, t = D(str(rate)), D(str(time))
    if compounding == "continuous":
        return (-r * t).exp()
    if compounding == "simple":
        return 1 / (1 + r * t)
    f = D({"annual": 1, "semiannual": 2, "quarterly": 4, "monthly": 12}[compounding])
    return (-(f * t) * (1 + r / f).ln()).exp()


def main():
    ROOT.mkdir(exist_ok=True)
    cases = []
    for comp, rate, time in itertools.product(
        ["continuous", "simple", "annual", "semiannual", "quarterly", "monthly"],
        [-0.01, -1e-12, 0, 1e-12, 0.07, 1.5],
        [1e-6, 0.25, 1, 60],
    ):
        if comp == "simple" and 1 + rate * time <= 0:
            continue
        df = discount(rate, time, comp)
        cases.append(
            {
                "source": "Decimal(70): compounding discount identity",
                "rate": rate,
                "time": time,
                "compounding": comp,
                "discount": float(df),
                "continuous": float(-df.ln() / D(str(time))),
            }
        )
    save("rate_conversions", cases)
    conventions = {
        "ACT/365 Fixed": ql.Actual365Fixed(),
        "ACT/360": ql.Actual360(),
        "ACT/ACT": ql.ActualActual(ql.ActualActual.ISDA),
        "30/360": ql.Thirty360(ql.Thirty360.USA),
        "30E/360": ql.Thirty360(ql.Thirty360.European),
    }
    cases = []
    for start, end in [
        ("2024-02-29", "2024-03-31"),
        ("2023-02-28", "2024-02-29"),
        ("2023-12-15", "2025-03-01"),
        ("2000-01-01", "2060-12-31"),
        ("2024-01-30", "2024-02-29"),
        ("2024-02-29", "2024-02-29"),
    ]:
        for name, dc in conventions.items():
            for reverse in [False, True]:
                a, b = (end, start) if reverse else (start, end)
                sign = -1 if reverse else 1
                cases.append(
                    {
                        "source": f"QuantLib {ql.__version__}; ordered signed day count",
                        "start": a,
                        "end": b,
                        "convention": name,
                        "days": sign * dc.dayCount(qdate(start), qdate(end)),
                        "year_fraction": sign * dc.yearFraction(qdate(start), qdate(end)),
                    }
                )
    save("daycounts", cases)
    calendar = ql.BespokeCalendar("reference")
    calendar.addWeekend(ql.Saturday)
    calendar.addWeekend(ql.Sunday)
    holidays = ["2024-07-31", "2025-01-31"]
    for holiday in holidays:
        calendar.addHoliday(qdate(holiday))
    bdcs = {
        "unadjusted": ql.Unadjusted,
        "following": ql.Following,
        "modified_following": ql.ModifiedFollowing,
        "preceding": ql.Preceding,
        "modified_preceding": ql.ModifiedPreceding,
    }
    cases = []
    inputs = [
        ("2024-01-30", "2025-07-30", 12, False, None, None),
        ("2024-01-31", "2026-01-31", 2, True, None, None),
        ("2024-01-15", "2026-01-31", 2, False, "2024-04-30", None),
        ("2024-01-15", "2026-01-31", 2, False, "2024-10-31", None),
        ("2024-01-31", "2026-01-15", 2, False, None, "2025-10-31"),
        ("2024-01-31", "2026-01-15", 2, False, None, "2025-04-30"),
    ]
    for inp, (generation, rule), (bdc, adjustment) in itertools.product(
        inputs,
        [("forward", ql.DateGeneration.Forward), ("backward", ql.DateGeneration.Backward)],
        bdcs.items(),
    ):
        start, end, freq, eom, first, penultimate = inp
        # QFin EOM is explicitly calendar month-end; QuantLib can instead use
        # business EOM. Compare unadjusted generation; adjustment is tested separately.
        if eom and bdc != "unadjusted":
            continue
        schedule = ql.Schedule(
            qdate(start),
            qdate(end),
            ql.Period(12 // freq, ql.Months),
            calendar,
            adjustment,
            ql.Unadjusted,
            rule,
            eom,
            qdate(first) if first else ql.Date(),
            qdate(penultimate) if penultimate else ql.Date(),
        )
        cases.append(
            {
                "source": f"QuantLib {ql.__version__} Schedule",
                "start_date": start,
                "end_date": end,
                "frequency": freq,
                "end_of_month": eom,
                "first_coupon_date": first,
                "next_to_last_coupon_date": penultimate,
                "date_generation": generation,
                "business_day_convention": bdc,
                "termination_convention": "unadjusted",
                "holidays": holidays,
                "dates": [d.ISO() for d in schedule],
            }
        )
    save("schedules", cases)
    cases = []
    for years, freq, coupon, ytm, face in itertools.product(
        [1, 10, 60], [1, 2, 12], [0, 0.07], [-0.01, 0, 0.11], [0.001, 100, 1e12]
    ):
        f = D(freq)
        y = D(str(ytm))
        nominal = D(str(face))
        base = 1 + y / f
        cash = [
            (D(i) / f, nominal * D(str(coupon)) / f + (nominal if i == years * freq else 0))
            for i in range(1, years * freq + 1)
        ]
        pv = [(t, c * base ** (-f * t)) for t, c in cash]
        price = sum(v for _, v in pv)
        mac = sum(t * v for t, v in pv) / price
        conv = sum(t * (t + 1 / f) * v / base**2 for t, v in pv) / price
        bumped = sum(c * (base + D("0.0001") / f) ** (-f * t) for t, c in cash)
        down = sum(c * (base - D("0.0001") / f) ** (-f * t) for t, c in cash)
        cases.append(
            {
                "source": "Decimal(70): coupon DCF and yield derivatives",
                "maturity": years,
                "frequency": freq,
                "coupon_rate": coupon,
                "yield": ytm,
                "face_value": face,
                "price": float(price),
                "macaulay": float(mac),
                "modified": float(mac / base),
                "convexity": float(conv),
                "dv01": float((down - bumped) / 2),
            }
        )
    save("fixed_income", cases)
    cases = []
    # Affine zero rates are exactly reproduced by all monotone Hermite schemes.
    for interpolation, extrapolation, intercept, slope in itertools.product(
        ["linear_zero", "linear_discount", "log_linear_discount", "monotone_zero"],
        ["flat_zero", "flat_forward", "error"],
        [-0.01, 0.02],
        [0, 0.003],
    ):
        times = [0.5, 1, 3, 10]
        rates = [float(D(str(intercept)) + D(str(slope)) * D(str(t))) for t in times]
        for query in [0, 0.25, 0.5, 0.75, 1, 2, 3, 7, 10, 60]:
            t = D(str(query))
            nodes = [D(str(x)) for x in times]
            zeros = [D(str(x)) for x in rates]
            logs = [-a * b for a, b in zip(nodes, zeros, strict=True)]
            outside = query < times[0] or query > times[-1]
            if outside and extrapolation == "error":
                continue
            if t == 0:
                df = D(1)
            elif outside and extrapolation == "flat_zero":
                df = (-t * (zeros[0] if query < times[0] else zeros[-1])).exp()
            elif outside:
                j = 0 if query < times[0] else len(times) - 2
                df = (
                    logs[j] + (t - nodes[j]) * (logs[j + 1] - logs[j]) / (nodes[j + 1] - nodes[j])
                ).exp()
            else:
                j = min(
                    next(
                        (i for i in range(len(times) - 1) if query <= times[i + 1]), len(times) - 2
                    ),
                    len(times) - 2,
                )
                w = (t - nodes[j]) / (nodes[j + 1] - nodes[j])
                if interpolation == "linear_discount":
                    df = (1 - w) * logs[j].exp() + w * logs[j + 1].exp()
                elif interpolation == "log_linear_discount":
                    df = ((1 - w) * logs[j] + w * logs[j + 1]).exp()
                else:
                    df = (-t * ((1 - w) * zeros[j] + w * zeros[j + 1])).exp()
            cases.append(
                {
                    "source": "Decimal(70): affine zero/Hermite or scalar discount interpolation",
                    "times": times,
                    "rates": rates,
                    "interpolation": interpolation,
                    "extrapolation": extrapolation,
                    "query": query,
                    "discount": float(df),
                }
            )
    save("curves", cases)
    cases = []
    for rate in [-0.01, 1e-8, 0.04, 0.5]:
        r = D(str(rate))
        times = [0.5, 1, 2, 5, 60]
        cases.append(
            {
                "source": "Decimal(70): flat curve; deposit, zero, bond, swap identities",
                "rate": rate,
                "times": times,
                "discounts": [float((-r * D(str(t))).exp()) for t in times],
                "deposit_rate": float(((r / D(2)).exp() - 1) * 2),
                "zero_price": float(D(100) * (-r).exp()),
                "bond_clean_price": float(
                    sum(D(3) * (-r * D(i) / 2).exp() for i in range(1, 5)) + D(100) * (-r * 2).exp()
                ),
                "swap_rate": float(2 * ((r / 2).exp() - 1)),
            }
        )
    save("bootstrapping", cases)
    cases = []
    for losses, weights in [
        ([0, 1], [9, 1]),
        ([-1, 0, 0, 100], [1, 3, 5, 1]),
        ([1, 2, 3, 4], [1, 1, 1, 1]),
        ([0, 1000], [999999, 1]),
        ([7, 7], [1, 1]),
        ([-4, -1, 1, 4], [1, 2, 2, 1]),
    ]:
        for level in ["0.9", "0.95", "0.99", "0.995", "0.999"]:
            total = sum(map(D, weights))
            pairs = sorted((D(x), D(w) / total) for x, w in zip(losses, weights, strict=True))
            cumulative = D(0)
            var = None
            tail = D(0)
            alpha = D(level)
            for loss, weight in pairs:
                previous = cumulative
                cumulative += weight
                if var is None and cumulative >= alpha:
                    var = loss
                tail += loss * max(D(0), cumulative - max(previous, alpha))
            cases.append(
                {
                    "source": "Decimal(70): inverse CDF and fractional VaR-atom tail integral",
                    "losses": losses,
                    "weights": weights,
                    "confidence": float(alpha),
                    "var": float(var),
                    "cvar": float(tail / (1 - alpha)),
                    "mean": float(sum(x * w for x, w in pairs)),
                }
            )
    save("risk", cases)
    cases = []
    for rate, scale in itertools.product([-0.02, 0, 0.06], [1e-6, 1, 1e10]):
        r = D(str(rate))
        k = D(str(scale))
        asset = D(100) * k * (-r * 5).exp()
        liability = D(90) * k * (-r * 3).exp()
        cases.append(
            {
                "source": "Decimal(70): zero-coupon asset/liability DCF identities",
                "rate": rate,
                "scale": scale,
                "asset_pv": float(asset),
                "liability_pv": float(liability),
                "surplus": float(asset - liability),
                "deficit": float(max(liability - asset, D(0))),
                "funding_ratio": float(asset / liability),
            }
        )
    save("alm", cases)
    cases = []
    for qx, lapse, term, premium, expense, count in itertools.product(
        [0, 0.02, 1], [0, 0.1, 1], [1, 5], [0, 100], [0, 10], [1, 1000]
    ):
        n = term + 1
        premiums = [D(0)] * n
        benefits = [D(0)] * n
        expenses = [D(0)] * n
        net = [D(0)] * n
        survival = D(1)
        q = D(str(qx))
        lapse_probability = D(str(lapse))
        c = D(count)
        for year in range(term):
            premiums[year] = c * survival * premium
            expenses[year] = c * survival * expense
            benefits[year + 1] = c * survival * q * 10000
            net[year] += expenses[year] - premiums[year]
            net[year + 1] += benefits[year + 1]
            survival *= (1 - q) * (1 - lapse_probability)
        cases.append(
            {
                "source": "Decimal(70): term recursion: start premium/expense, end deaths",
                "qx": qx,
                "lapse": lapse,
                "term": term,
                "premium": premium,
                "expense": expense,
                "count": count,
                "premiums": list(map(float, premiums)),
                "benefits": list(map(float, benefits)),
                "expenses": list(map(float, expenses)),
                "net": list(map(float, net)),
                "pv": float(sum(v * (-D("0.03") * D(i)).exp() for i, v in enumerate(net))),
            }
        )
    save("life", cases)


if __name__ == "__main__":
    main()
