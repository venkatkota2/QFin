"""Price a fixed-rate portfolio through QFin's public Python API."""

import qfin

curve = qfin.YieldCurve(
    times=[0.0, 1.0, 5.0, 10.0, 30.0],
    zero_rates=[0.025, 0.027, 0.031, 0.034, 0.038],
)
bonds = [
    qfin.FixedRateBond(maturity=2.0, coupon_rate=0.03, frequency=2),
    qfin.FixedRateBond(maturity=7.0, coupon_rate=0.04, frequency=2),
    qfin.FixedRateBond(maturity=15.0, coupon_rate=0.05, frequency=2),
]

analytics = qfin.price_bonds(bonds, curve)
for index, price in enumerate(analytics.dirty_prices):
    print(
        f"Bond {index + 1}: price={price:.4f}, "
        f"duration={analytics.macaulay_duration[index]:.4f}, "
        f"convexity={analytics.convexity[index]:.4f}, "
        f"DV01={analytics.dv01[index]:.4f}"
    )
