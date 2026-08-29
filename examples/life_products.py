"""Project grouped term, participating, universal-life, and annuity points."""

import numpy as np

import qfin

ages = np.arange(20, 121, dtype=float)
mortality = qfin.MortalityTable(
    ages,
    np.minimum(0.00025 * np.exp(0.078 * (ages - 20)), 1.0),
)
curve = qfin.YieldCurve(
    [0, 1, 5, 10, 20, 40],
    [0.02, 0.022, 0.026, 0.029, 0.033, 0.035],
)
policies = [
    qfin.LifePolicy(40, 150_000, 650, 20, product_type="term_life"),
    qfin.LifePolicy(
        45,
        100_000,
        900,
        20,
        product_type="participating_life",
        bonus_rate=0.01,
    ),
    qfin.LifePolicy(
        50,
        100_000,
        1_200,
        20,
        product_type="universal_life",
        account_value=25_000,
        annual_charge=80,
        crediting_spread=0.005,
    ),
    qfin.LifePolicy(
        65,
        0,
        0,
        20,
        product_type="annuity",
        annual_benefit=12_000,
        benefit_inflation_linkage=1.0,
    ),
]
model_points = qfin.PolicyModelPointSet(policies, counts=[2_500, 2_000, 1_500, 1_000])
assumptions = qfin.LifeAssumptionSet(
    mortality=mortality,
    curve=curve,
    lapse_rate=0.04,
    expense_per_policy=35,
    disability_rate=0.005,
    recovery_rate=0.20,
    disabled_mortality_multiplier=1.5,
    crediting_rate=0.025,
    expense_inflation_rate=0.025,
)

projection = qfin.project_liabilities(model_points, assumptions)
sensitivity = qfin.life_sensitivities(model_points, assumptions)

print(f"Model points: {model_points.model_point_count}")
print(f"Policies represented: {model_points.policy_count:,.0f}")
print(f"Net liability PV: {projection.present_value:,.2f}")
print(f"Liability duration: {projection.duration:.4f}")
print("PV by product:", projection.product_present_values)
print("Sensitivity impacts:", sensitivity.to_dict())
