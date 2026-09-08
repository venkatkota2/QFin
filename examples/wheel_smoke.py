"""Exercise the installed core package without optional extras."""

from importlib.metadata import version
from importlib.util import find_spec

import numpy as np

import qfin
from qfin.representation import (
    compare_state_preparation_strategies,
    encode_independent_factors,
)


def main() -> None:
    curve = qfin.YieldCurve([0, 1], [0.02, 0.02])
    bonds = [qfin.FixedRateBond(1, 0.02)]
    reference = qfin.price_bonds(bonds, curve, engine="numpy")
    result = qfin.price_bonds(bonds, curve, engine="native")
    np.testing.assert_allclose(result.dirty_prices, reference.dirty_prices, rtol=1e-13)
    risk = qfin.compile(
        qfin.TailProbability(qfin.LossDistribution([0, 1]), 0.5),
        backend="classical", target_error=0.1, min_qubits=2, max_qubits=2,
    )
    factors = encode_independent_factors(
        [qfin.Normal(), qfin.Normal()], qubits_per_factor=2, method="probability",
    )
    strategy = compare_state_preparation_strategies(factors).require_selected()
    model = qfin.FactorizedLossModel(
        factors, qfin.SparseExposureObjective(linear={"factor_0": 1.0}),
    )
    factor_tail = qfin.compile(
        qfin.FactorTailProbability(model, threshold=0.0),
        backend="classical", arithmetic_scale=1.0,
    )
    factor_cvar = qfin.compile(
        qfin.FactorCVaR(model, confidence=0.75),
        backend="classical", target_error=0.2, arithmetic_scale=1.0,
    ).run()
    optimization = qfin.compile(qfin.MeanVarianceProblem(
        [0.04, 0.08], [[0.02, 0.0], [0.0, 0.05]],
    ))
    info = qfin.system_info()
    assert info["native_extension"]
    assert qfin.__version__ == version("qfin-quantum")
    assert result.dirty_prices.shape == (1,)
    assert risk.run().probability == 0.5
    assert strategy.strategy == "factorized_marginal_loader"
    assert 0 <= factor_tail.run().probability <= 1
    assert factor_cvar.cvar >= factor_cvar.var
    assert optimization.run().success
    assert find_spec("pennylane") is None
    assert find_spec("qiskit") is None
    assert find_spec("QuantLib") is None
    print(info)


if __name__ == "__main__":
    main()
