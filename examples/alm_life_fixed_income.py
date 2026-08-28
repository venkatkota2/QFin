"""Fixed-income/life ALM example with a PennyLane Lightning risk circuit."""

import numpy as np

import qfin

mortality = qfin.MortalityTable.illustrative_gompertz_makeham(
    min_age=20,
    max_age=120,
)

assets = qfin.FixedIncomePortfolio(
    (
        qfin.BondPosition(qfin.FixedRateBond(1_000, 0.04, 10, 2), 600),
        qfin.BondPosition(qfin.FixedRateBond(1_000, 0.045, 20, 2), 450),
    )
)
liabilities = qfin.LifePolicyPortfolio(
    (
        qfin.PolicyPosition(
            qfin.TermLifePolicy(
                issue_age=40,
                term=25,
                face_amount=100_000,
                annual_expense=25,
            ),
            80,
        ),
        qfin.PolicyPosition(
            qfin.WholeLifePolicy(
                issue_age=50,
                face_amount=50_000,
                annual_expense=20,
            ),
            40,
        ),
    )
)

alm = qfin.AssetLiabilityModel(
    assets=assets,
    liabilities=liabilities,
    discount_curve=qfin.DiscountCurve.flat(0.04),
    mortality=mortality,
)
print(alm.evaluate().to_dict())

shocks = np.array([-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04])
scenarios = alm.run_parallel_shocks(shocks)
print(
    {
        "expected_surplus": scenarios.expected_surplus,
        "shortfall_probability": scenarios.shortfall_probability,
        "expected_shortfall": scenarios.expected_shortfall,
    }
)

compiled = qfin.compile_alm(
    alm,
    shocks,
    metric="expected_shortfall",
    target_error=25_000,
)
print(compiled.explain())
result = compiled.run(shots=2_000, schedule=(0, 1, 2, 4), seed=7)
print(result.to_dict())
