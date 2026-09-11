"""Representative documented result schemas; expectations captured from 1.1.1."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import qfin
from qfin.algorithms import CircuitObservation, maximum_likelihood_amplitude_estimate


def key_paths(value, prefix=""):
    if isinstance(value, dict):
        return sorted(
            path
            for key, item in value.items()
            for path in [prefix + key, *key_paths(item, prefix + key + ".")]
        )
    if isinstance(value, list) and value:
        return key_paths(value[0], prefix + "[].")
    return []


def contracts():
    curve = qfin.YieldCurve([0, 1, 10], [0.02, 0.03, 0.04])
    model = qfin.ALMModel(
        qfin.AssetPortfolio([qfin.FixedRateBond(5, 0.05)]),
        qfin.LiabilityPortfolio([qfin.CashFlow(3, 80)]),
        curve,
    )
    distribution = qfin.LossDistribution([0, 1, 2, 3])
    risk = qfin.compile(
        qfin.CVaR(distribution), backend="classical", target_error=0.1, min_qubits=2, max_qubits=2
    )
    estimate = maximum_likelihood_amplitude_estimate(
        (CircuitObservation(0, 20, 100),), grid_size=4097
    )
    objects = {
        "curve.explain": curve.explain(),
        "alm.dataclass": dataclasses.asdict(model.evaluate(engine="numpy")),
        "compiled_risk.to_dict": risk.to_dict(),
        "risk_resources.to_dict": risk.resources().to_dict(),
        "bootstrap.to_dict": qfin.bootstrap_risk_interval(distribution, resamples=10).to_dict(),
        "amplitude.to_dict": estimate.to_dict(),
    }
    return {name: key_paths(item) for name, item in objects.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(contracts(), indent=2, sort_keys=True) + "\n")
