"""Run a parallel-rate ALM scenario set in bounded memory."""

import numpy as np

import qfin

curve = qfin.YieldCurve([0, 1, 5, 10, 20], [0.02, 0.024, 0.03, 0.034, 0.037])
model = qfin.ALMModel(
    assets=qfin.AssetPortfolio(
        [qfin.FixedRateBond(5, 0.03), qfin.FixedRateBond(12, 0.045)],
        [30, 20],
    ),
    liabilities=qfin.LiabilityPortfolio.from_arrays(
        [3, 8, 15], [1_000, 2_000, 3_000]
    ),
    curve=curve,
)
scenarios = qfin.RateScenarioSet.parallel(curve, np.linspace(-0.02, 0.02, 101))
result = model.run_scenarios(scenarios, chunk_size=32)

print(f"Scenarios: {len(result.labels)}")
print(f"Minimum funding ratio: {np.min(result.funding_ratio):.4f}")
print(f"Maximum scenario loss: {np.max(result.loss_distribution().losses):,.2f}")
