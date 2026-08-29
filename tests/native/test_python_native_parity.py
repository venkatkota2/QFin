import numpy as np
import pytest

import qfin

pytestmark = pytest.mark.skipif(
    not qfin.system_info()["native_extension"], reason="native extension unavailable"
)


def test_native_build_metadata_is_reported() -> None:
    info = qfin.system_info()
    assert info["native_backend"] == "qfin-native"
    assert info["native_cpp_standard"] == "C++20"
    assert info["native_compiler"]


def test_fixed_income_native_matches_numpy_for_large_mixed_batch() -> None:
    curve = qfin.YieldCurve(
        [0.0, 1.0, 5.0, 10.0, 30.0],
        [-0.005, 0.01, 0.025, 0.035, 0.04],
    )
    bonds = [
        qfin.FixedRateBond(
            maturity=1.0 + index % 30,
            coupon_rate=0.001 * (index % 80),
            frequency=(1, 2, 4)[index % 3],
        )
        for index in range(2_000)
    ]
    reference = qfin.price_bonds(bonds, curve, engine="numpy")
    native = qfin.price_bonds(bonds, curve, engine="native")
    np.testing.assert_allclose(native.dirty_prices, reference.dirty_prices, rtol=1e-13)
    np.testing.assert_allclose(native.macaulay_duration, reference.macaulay_duration, rtol=1e-13)
    np.testing.assert_allclose(native.convexity, reference.convexity, rtol=1e-13)
    np.testing.assert_allclose(native.dv01, reference.dv01, rtol=1e-11)


def test_yield_price_and_solver_native_parity() -> None:
    bonds = [qfin.FixedRateBond(1 + index % 20, 0.03, frequency=2) for index in range(200)]
    yields = np.linspace(-0.01, 0.20, len(bonds))
    reference = qfin.price_bonds_from_yield(bonds, yields, engine="numpy")
    native = qfin.price_bonds_from_yield(bonds, yields, engine="native")
    np.testing.assert_allclose(native.dirty_prices, reference.dirty_prices, rtol=1e-13)
    solved = qfin.yield_from_prices(bonds, native.dirty_prices, engine="native")
    assert np.all(solved.converged)
    np.testing.assert_allclose(solved.yields, yields, atol=1e-10)


def test_risk_native_matches_weighted_numpy_reference() -> None:
    rng = np.random.default_rng(7)
    losses = rng.normal(size=20_000)
    probabilities = rng.uniform(size=20_000)
    distribution = qfin.LossDistribution(losses, probabilities)
    reference = qfin.aggregate_risk(distribution, confidence=0.995, engine="numpy")
    native = qfin.aggregate_risk(distribution, confidence=0.995, engine="native")
    assert native.mean == pytest.approx(reference.mean, abs=1e-14)
    assert native.standard_deviation == pytest.approx(reference.standard_deviation, abs=1e-14)
    assert native.var == pytest.approx(reference.var, abs=1e-14)
    assert native.cvar == pytest.approx(reference.cvar, abs=3e-12)


def test_auto_dispatch_falls_back_to_numpy_when_extension_cannot_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import qfin._native as native_loader

    curve = qfin.YieldCurve([0.0, 10.0], [0.03, 0.03])
    bonds = [qfin.FixedRateBond(10.0, 0.03) for _ in range(500)]
    monkeypatch.setattr(native_loader, "_extension", None)
    result = qfin.price_bonds(bonds, curve, engine="auto")
    assert result.engine == "numpy"
    with pytest.raises(qfin.NativeBackendUnavailableError, match="engine='numpy'"):
        qfin.price_bonds(bonds, curve, engine="native")
