"""Project a multi-factor ALM portfolio through stochastic economic paths."""

import numpy as np

import qfin

curve = qfin.YieldCurve(
    [0, 1, 3, 7, 15, 30],
    [0.025, 0.027, 0.029, 0.032, 0.035, 0.037],
)
alm = qfin.ALMModel(
    assets=qfin.AssetPortfolio(
        [qfin.FixedRateBond(7, 0.032), qfin.FixedRateBond(20, 0.042)],
        [120, 80],
        equity_value=12_000,
        cash_value=1_500,
    ),
    liabilities=qfin.LiabilityPortfolio.from_arrays(
        [2, 5, 10, 20],
        [4_000, 7_000, 10_000, 14_000],
        inflation_linkage=[0.25, 0.50, 0.75, 1.0],
    ),
    curve=curve,
)

scenarios = qfin.EconomicScenarioSet.correlated_gaussian(
    curve,
    scenario_count=512,
    periods=10,
    correlation=np.eye(6),
    standard_deviations=[0.0075, 0.003, 0.18, 0.012, 0.08, 0.12],
    means=[0.0, 0.0, 0.06, 0.025, 0.0, 0.0],
    seed=11,
    antithetic=True,
)
strategy = qfin.RebalancingStrategy(
    target_equity_weight=0.30,
    rebalance_frequency=1,
    transaction_cost_rate=0.001,
)

paths = alm.project_paths(scenarios, strategy=strategy, scenario_chunk_size=128)
risk = qfin.aggregate_risk(paths.loss_distribution(), confidence=0.995)
mean_funding = np.average(paths.funding_ratio[:, -1], weights=paths.probabilities)
mean_costs = np.average(np.sum(paths.transaction_costs, axis=1), weights=paths.probabilities)

print(f"Execution engine: {paths.engine}")
print(f"Mean horizon funding ratio: {mean_funding:.4f}")
print(f"Horizon VaR (99.5%): {risk.var:,.2f}")
print(f"Horizon CVaR (99.5%): {risk.cvar:,.2f}")
print(f"Mean transaction costs: {mean_costs:,.2f}")
