"""Use QFin 0.8 factorized representations and classical optimization policy."""

from __future__ import annotations

import json

import numpy as np

import qfin


def main() -> None:
    factor_model = qfin.GaussianFactorModel(
        ("rates", "equity", "inflation"),
        np.array(
            [
                [1.0, -0.25, 0.35],
                [-0.25, 1.0, 0.10],
                [0.35, 0.10, 1.0],
            ]
        ),
        means=np.array([0.02, 0.07, 0.025]),
        standard_deviations=np.array([0.01, 0.18, 0.015]),
    )
    factorized = qfin.encode_gaussian_factors(factor_model, qubits_per_factor=3)
    preparation = qfin.FactorizedPreparation.from_encoding(factorized)
    strategies = qfin.compare_state_preparation_strategies(
        factorized,
        target=qfin.DeviceTarget.linear(factorized.total_qubits),
    )

    optimization = qfin.MeanVarianceProblem(
        expected_returns=np.array([0.045, 0.075, 0.11]),
        covariance=np.array(
            [
                [0.025, 0.004, 0.002],
                [0.004, 0.065, 0.009],
                [0.002, 0.009, 0.145],
            ]
        ),
        risk_aversion=3.0,
        asset_names=("fixed_income", "balanced", "equity"),
    )
    compiled = qfin.compile(optimization)
    if not isinstance(compiled, qfin.CompiledOptimizationModel):
        raise RuntimeError("optimization compilation did not return the expected model")
    result = {
        "factorized_encoding": factorized.to_dict(),
        "factorized_preparation": preparation.to_dict(),
        "strategy_selection": strategies.to_dict(),
        "optimization": compiled.run().to_dict(),
        "optimization_resources": compiled.resources().to_dict(),
        "covariance_feasibility": compiled.block_encoding_feasibility().to_dict(),
        "compiler_explanation": compiled.explain(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
