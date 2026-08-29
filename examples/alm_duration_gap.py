"""Evaluate a fixed-income asset/liability duration gap."""

import qfin

curve = qfin.YieldCurve([0, 1, 5, 10, 20], [0.02, 0.024, 0.03, 0.034, 0.037])
assets = qfin.AssetPortfolio(
    bonds=[
        qfin.FixedRateBond(3, 0.03),
        qfin.FixedRateBond(8, 0.04),
        qfin.FixedRateBond(15, 0.045),
    ],
    quantities=[20, 15, 10],
)
liabilities = qfin.LiabilityPortfolio.from_arrays(
    times=[2, 5, 10, 15],
    amounts=[1_000, 1_500, 2_000, 2_500],
)

result = qfin.ALMModel(assets, liabilities, curve).evaluate()
print(f"Asset PV:          {result.asset_pv:,.2f}")
print(f"Liability PV:      {result.liability_pv:,.2f}")
print(f"Funding ratio:     {result.funding_ratio:.4f}")
print(f"Surplus:           {result.surplus:,.2f}")
print(f"Duration gap:      {result.duration_gap:.4f}")
print(f"Convexity gap:     {result.convexity_gap:.4f}")
