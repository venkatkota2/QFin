"""Exercise an installed native or pure-Python package without optional extras."""

import argparse
import importlib
import pkgutil
import subprocess
import sys
import sysconfig
from importlib.metadata import version
from importlib.resources import files
from importlib.util import find_spec
from pathlib import Path

import numpy as np

import qfin
from qfin.representation import (
    compare_state_preparation_strategies,
    encode_independent_factors,
)


def main(*, pure_python: bool = False) -> None:
    # An editable install or source checkout must not mask missing wheel files.
    assert Path(qfin.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    assert files("qfin").joinpath("py.typed").is_file()
    for module in pkgutil.walk_packages(qfin.__path__, prefix="qfin."):
        importlib.import_module(module.name)
    for optional in ("pennylane", "pennylane_lightning", "qiskit", "QuantLib"):
        assert find_spec(optional) is None
        assert optional not in sys.modules

    curve = qfin.YieldCurve([0, 1], [0.02, 0.02])
    bonds = [qfin.FixedRateBond(1, 0.02)]
    reference = qfin.price_bonds(bonds, curve, engine="numpy")
    result = qfin.price_bonds(bonds, curve, engine="auto" if pure_python else "native")
    np.testing.assert_allclose(result.dirty_prices, reference.dirty_prices, rtol=1e-13)
    if pure_python:
        assert find_spec("qfin._qfin_native") is None
        try:
            qfin.price_bonds(bonds, curve, engine="native")
        except qfin.NativeBackendUnavailableError:
            pass
        else:
            raise AssertionError("a pure-Python install must reject explicit native execution")
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
    assert info["native_extension"] is (not pure_python)
    assert qfin.__version__ == version("qfin-quantum")
    assert result.dirty_prices.shape == (1,)
    assert risk.run().probability == 0.5
    assert strategy.strategy == "factorized_marginal_loader"
    assert 0 <= factor_tail.run().probability <= 1
    assert factor_cvar.cvar >= factor_cvar.var
    assert optimization.run().success
    module_cli = [sys.executable, "-I", "-m", "qfin"]
    console_cli = [str(Path(sysconfig.get_path("scripts")) / (
        "qfin.exe" if sys.platform == "win32" else "qfin"
    ))]
    for command in (module_cli, console_cli):
        completed = subprocess.run(
            [*command, "--version"], check=True, capture_output=True, text=True,
        )
        assert completed.stdout.strip() == f"qfin {qfin.__version__}"
    completed = subprocess.run(
        [*module_cli, "price", "--spot", "100", "--strike", "105",
         "--maturity", "1", "--rate", "0.04", "--volatility", "0.2",
         "--compile-only"],
        check=True, capture_output=True, text=True,
    )
    assert "European call" in completed.stdout
    print(info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pure-python", action="store_true")
    main(pure_python=parser.parse_args().pure_python)
