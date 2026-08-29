"""Run ALM scenarios through QFin's experimental quantum CVaR workflow."""

import numpy as np

import qfin

curve = qfin.YieldCurve([0, 1, 5, 10], [0.02, 0.025, 0.03, 0.035])
alm = qfin.ALMModel(
    qfin.AssetPortfolio([qfin.FixedRateBond(5, 0.03)], [10]),
    qfin.LiabilityPortfolio.from_arrays([3, 8], [400, 600]),
    curve,
)
scenario_result = alm.run_scenarios(
    qfin.RateScenarioSet.parallel(curve, np.linspace(-0.02, 0.02, 32))
)
risk = qfin.CVaR(scenario_result.loss_distribution(), confidence=0.90)
compiled = qfin.compile(risk, target_error=1.0, min_qubits=4, max_qubits=5)

print(compiled.explain())
classical = compiled.run()
quantum = compiled.run_quantum(
    shots=2_000,
    schedule=(0, 1, 2),
    seed=7,
    likelihood_grid_size=32_769,
    bootstrap_resamples=100,
)

print("Classical result:", classical)
print("Quantum CVaR estimate:", quantum.expected_shortfall)
print("Quantum conditional 95% interval:", quantum.confidence_interval_95)
print("Quantum VaR grid estimate:", quantum.value_at_risk)
print("Resources:", quantum.resources.to_dict())
print("Capabilities:", qfin.problem_capabilities(risk).to_dict())
# This is a simulator research workflow. The first empirical state/oracle loader
# scales as O(2**data_qubits), and QFin does not claim quantum advantage.
