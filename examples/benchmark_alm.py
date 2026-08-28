"""Reproducible throughput benchmark for the vectorized ALM scenario engine."""

from time import perf_counter

import numpy as np

import qfin

mortality = qfin.MortalityTable.illustrative_gompertz_makeham()
assets = qfin.FixedIncomePortfolio(
    tuple(
        qfin.BondPosition(
            qfin.FixedRateBond(1_000, 0.025 + maturity / 1_000, maturity, 2),
            100,
        )
        for maturity in range(2, 42, 2)
    )
)
liabilities = qfin.LifePolicyPortfolio(
    tuple(
        qfin.PolicyPosition(
            qfin.TermLifePolicy(
                issue_age=30 + index,
                term=20 + index,
                face_amount=100_000,
                annual_premium=500,
            ),
            20,
        )
        for index in range(10)
    )
)
model = qfin.AssetLiabilityModel(
    assets,
    liabilities,
    qfin.DiscountCurve.flat(0.04),
    mortality,
)

shocks = np.linspace(-0.05, 0.05, 100_000)
started = perf_counter()
result = model.run_parallel_shocks(shocks)
elapsed = perf_counter() - started
print(f"scenarios={shocks.size:,}")
print(f"seconds={elapsed:.6f}")
print(f"scenarios_per_second={shocks.size / elapsed:,.0f}")
print(f"expected_shortfall={result.expected_shortfall:.6f}")
