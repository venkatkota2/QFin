"""Bridge native ALM scenarios into QFin's quantum representation layer."""

import numpy as np

import qfin

curve = qfin.YieldCurve([0, 1, 5, 10], [0.02, 0.025, 0.03, 0.035])
alm = qfin.ALMModel(
    qfin.AssetPortfolio([qfin.FixedRateBond(5, 0.03)], [10]),
    qfin.LiabilityPortfolio.from_arrays([3, 8], [400, 600]),
    curve,
)
scenario_result = alm.run_scenarios(
    qfin.RateScenarioSet.parallel(curve, np.linspace(-0.02, 0.02, 64))
)
risk = qfin.CVaR(scenario_result.loss_distribution(), confidence=0.995)
compiled = qfin.compile(risk, target_error=1.0, max_qubits=8)

print(compiled.explain())
print("Classical result:", compiled.run())
print("Capabilities:", qfin.problem_capabilities(risk).to_dict())
# The loss representation is ready. QFin deliberately raises if to_pennylane()
# is requested because a quantum CVaR oracle/estimator is not yet implemented.
