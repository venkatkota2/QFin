"""Compatibility and safety behavior that must survive optimized Python."""

import subprocess
import sys

import pytest

import qfin
from qfin._dispatch import POLICIES, decide_engine
from qfin._memory import check_allocation


def test_exception_hierarchy_preserves_existing_catches():
    for error, builtin in [
        (qfin.QFinValidationError, ValueError),
        (qfin.QFinTypeError, TypeError),
        (qfin.CurveBootstrapError, ValueError),
        (qfin.FinancialValidationError, AssertionError),
    ]:
        assert issubclass(error, qfin.QFinError) and issubclass(error, builtin)
    assert issubclass(qfin.NativeBackendUnavailableError, qfin.BackendUnavailableError)
    assert issubclass(qfin.BackendUnavailableError, qfin.BackendError)
    with pytest.raises(qfin.QFinError):
        qfin.FixedRateBond(-1, 0.03)


@pytest.mark.parametrize("engine", ["numpy", "native", "auto"])
def test_huge_public_life_horizon_preflight(engine):
    assumptions = qfin.ProjectionAssumptions(
        qfin.MortalityTable([0, 120], [0, 0]), qfin.YieldCurve([0, 1], [0, 0])
    )
    with pytest.raises(qfin.ResourceLimitError):
        qfin.project_liabilities(
            [qfin.LifePolicy(40, 1000, 0, 2**31 - 1)], assumptions, engine=engine
        )


def test_huge_bond_preflight_and_zero_dimension():
    with pytest.raises(qfin.ResourceLimitError):
        qfin.FixedRateBond(1e308, 0.03).payment_schedule()
    assert check_allocation((2**100, 0)) == 0
    with pytest.raises(qfin.ResourceLimitError):
        check_allocation((-1,))


@pytest.mark.parametrize("policy", POLICIES)
def test_dispatch_threshold_boundaries_and_explanation(policy, monkeypatch):
    monkeypatch.setattr("qfin._native.available", lambda: True)
    threshold = POLICIES[policy]
    if threshold is None:
        assert decide_engine("auto", 10**15, auto_native_threshold=threshold).engine == "numpy"
    else:
        assert decide_engine("auto", threshold, auto_native_threshold=threshold).engine == "native"
        if threshold > 0:
            assert (
                decide_engine("auto", threshold - 1, auto_native_threshold=threshold).engine
                == "numpy"
            )
    assert (
        decide_engine(
            "auto", 10**15, native_compatible=False, auto_native_threshold=threshold
        ).engine
        == "numpy"
    )
    monkeypatch.setattr("qfin._native.available", lambda: False)
    assert decide_engine("auto", 10**15, auto_native_threshold=threshold).engine == "numpy"


def test_validation_is_present_with_python_optimization():
    code = """import qfin
operations = [lambda:qfin.FixedRateBond(-1,0.03),
    lambda:qfin.compile(qfin.VaR(qfin.LossDistribution([1,2])),max_qubits=0),
    lambda:qfin.discount_factor(-2,1,"annual")]
for operation in operations:
    try: operation()
    except qfin.QFinError: pass
    else: raise RuntimeError("optimized mode removed public validation")
"""
    subprocess.run([sys.executable, "-O", "-c", code], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("rate", [-1000.0, 1000.0])
def test_unrepresentable_discount_factor_is_explicit(rate):
    with pytest.raises(qfin.QFinValidationError, match="finite double"):
        qfin.discount_factor(rate, 1000)


def test_rate_conversion_avoids_unrepresentable_intermediate_discount():
    from math import log1p

    assert qfin.convert_rate(0.05, "annual", "continuous", time=1e300) == log1p(0.05)
    assert qfin.convert_rate(1e-20, "annual", "continuous") == pytest.approx(1e-20, rel=1e-15)
