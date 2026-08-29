"""Feed chunked life-scenario liability losses into QFin's risk compiler."""

import numpy as np

import qfin

ages = np.arange(20, 121, dtype=float)
mortality = qfin.MortalityTable(
    ages,
    np.minimum(0.0002 * np.exp(0.08 * (ages - 20)), 1.0),
)
curve = qfin.YieldCurve([0, 1, 5, 10, 20], [0.02, 0.022, 0.026, 0.03, 0.034])
book = qfin.PolicyModelPointSet(
    [
        qfin.LifePolicy(40, 100_000, 550, 10),
        qfin.LifePolicy(65, 0, 0, 10, product_type="annuity", annual_benefit=8_000),
    ],
    counts=[5_000, 2_000],
)
assumptions = qfin.LifeAssumptionSet(
    mortality,
    curve,
    lapse_rate=0.04,
    expense_per_policy=30,
)
scenarios = qfin.EconomicScenarioSet.correlated_gaussian(
    curve,
    scenario_count=64,
    periods=10,
    correlation=np.eye(6),
    standard_deviations=[0.006, 0.002, 0.15, 0.01, 0.10, 0.15],
    means=[0.0, 0.0, 0.05, 0.025, 0.0, 0.0],
    seed=7,
    antithetic=True,
)

scenario_result = qfin.project_liability_scenarios(
    book,
    assumptions,
    scenarios,
    scenario_chunk_size=16,
    policy_chunk_size=2,
)
problem = qfin.CVaR(scenario_result.loss_distribution(), confidence=0.95)
compiled = qfin.compile(
    problem,
    backend="classical",
    target_error=10_000,
    min_qubits=4,
    max_qubits=8,
)

print(f"Projection engine: {scenario_result.engine}")
print(f"Peak chunk estimate: {scenario_result.working_set_estimate_bytes / 1024:.1f} KiB")
print(f"Classical scenario CVaR: {compiled.run().cvar:,.2f}")
print("Compiler capabilities:", qfin.problem_capabilities(problem).to_dict())
# The same finite loss representation can be executed with run_quantum() when
# PennyLane is installed. PennyLane-Lightning remains the quantum simulator.
