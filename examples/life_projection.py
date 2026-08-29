"""Project expected term-life cash flows without exposing C++ internals."""

import numpy as np

import qfin

ages = np.arange(20, 121, dtype=float)
qx = np.minimum(0.0002 * np.exp(0.075 * (ages - 20)), 1.0)
mortality = qfin.MortalityTable(ages, qx)
curve = qfin.YieldCurve([0, 1, 5, 10, 30, 50], [0.02, 0.022, 0.027, 0.03, 0.035, 0.037])
policies = [
    qfin.LifePolicy(age=35, sum_assured=250_000, annual_premium=700, term=20),
    qfin.LifePolicy(age=50, sum_assured=100_000, annual_premium=900, term=10),
]
assumptions = qfin.ProjectionAssumptions(
    mortality=mortality,
    curve=curve,
    lapse_rate=0.04,
    expense_per_policy=30,
)

projection = qfin.project_liabilities(policies, assumptions)
print(f"Net liability PV: {projection.present_value:,.2f}")
print(f"Liability duration: {projection.duration:.4f}")
print("First five net cash flows:", projection.net_liability_cashflows[:5])
