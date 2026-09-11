"""QuantLib dated bond and bootstrap market quotes, independent of QFin."""

import itertools
import json
from pathlib import Path

import QuantLib as ql


def main():
    issue = ql.Date(31, 1, 2024)
    settlement = ql.Date(30, 4, 2024)
    rows = []
    for years, rate, coupon, face in itertools.product(
        [2, 10, 60], [-0.01, 1e-12, 0.15], [0, 0.08], [100, 1e12]
    ):
        maturity = ql.Date(31, 1, 2024 + years)
        ql.Settings.instance().evaluationDate = settlement
        curve = ql.FlatForward(settlement, rate, ql.Actual365Fixed(), ql.Continuous)
        schedule = ql.Schedule(
            issue,
            maturity,
            ql.Period(6, ql.Months),
            ql.NullCalendar(),
            ql.Unadjusted,
            ql.Unadjusted,
            ql.DateGeneration.Backward,
            True,
        )
        bond = ql.FixedRateBond(
            0, face, schedule, [coupon], ql.Thirty360(ql.Thirty360.European), ql.Unadjusted
        )
        bond.setPricingEngine(ql.DiscountingBondEngine(ql.YieldTermStructureHandle(curve)))
        rows.append(
            {
                "id": f"dated-bond-{len(rows):03d}",
                "source": f"QuantLib {ql.__version__}",
                "issue": "2024-01-31",
                "maturity": f"{2024 + years}-01-31",
                "settlement": "2024-04-30",
                "rate": rate,
                "coupon": coupon,
                "face": face,
                "dirty_price": bond.dirtyPrice() * face / 100,
                "clean_price": bond.cleanPrice() * face / 100,
                "accrued_interest": bond.accruedAmount(settlement) * face / 100,
                "curve_day_count": "ACT/365 Fixed",
                "coupon_day_count": "30E/360",
            }
        )
    root = Path(__file__).resolve().parents[1] / "tests/reference_data"
    (root / "dated_bonds.json").write_text(
        "[\n" + ",\n".join("  " + json.dumps(row, separators=(",", ":")) for row in rows) + "\n]\n"
    )
    print(f"{len(rows)} QuantLib dated bond references")


if __name__ == "__main__":
    main()
