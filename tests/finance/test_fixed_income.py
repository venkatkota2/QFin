from math import exp

import numpy as np
import pytest

import qfin


@pytest.fixture
def flat_curve() -> qfin.YieldCurve:
    return qfin.YieldCurve([0.0, 30.0], [0.04, 0.04])


def test_zero_coupon_curve_price_has_analytical_value(flat_curve: qfin.YieldCurve) -> None:
    bond = qfin.FixedRateBond(maturity=5.0, coupon_rate=0.0, face_value=100.0)
    result = qfin.price_bonds(bond, flat_curve, engine="numpy")
    assert result.dirty_prices[0] == pytest.approx(100 * exp(-0.04 * 5))
    assert result.macaulay_duration[0] == pytest.approx(5.0)
    assert result.convexity[0] == pytest.approx(25.0)
    assert result.dv01[0] > 0


def test_par_bond_price_and_yield_round_trip() -> None:
    bond = qfin.FixedRateBond(maturity=10.0, coupon_rate=0.05, frequency=2)
    priced = qfin.price_bonds_from_yield(bond, 0.05, engine="numpy")
    assert priced.dirty_prices[0] == pytest.approx(100.0, abs=1e-11)
    solved = qfin.yield_from_prices(bond, priced.dirty_prices, engine="numpy")
    assert solved.converged[0]
    assert solved.yields[0] == pytest.approx(0.05, abs=1e-11)
    assert priced.modified_duration[0] < priced.macaulay_duration[0]


def test_negative_yield_round_trip_and_extreme_rate() -> None:
    bonds = [
        qfin.FixedRateBond(1.0, 0.02, frequency=4),
        qfin.FixedRateBond(30.0, 0.01, frequency=1),
    ]
    yields = np.array([-0.02, 0.40])
    prices = qfin.price_bonds_from_yield(bonds, yields, engine="numpy").dirty_prices
    solved = qfin.yield_from_prices(bonds, prices, engine="numpy")
    np.testing.assert_allclose(solved.yields, yields, atol=1e-10)
    assert np.all(solved.converged)


def test_clean_dirty_and_accrued_interest() -> None:
    bond = qfin.FixedRateBond(maturity=2.0, coupon_rate=0.04, frequency=2)
    curve = qfin.YieldCurve([0.0, 5.0], [0.0, 0.0])
    result = qfin.price_bonds(bond, curve, settlement=0.25, engine="numpy")
    assert result.accrued_interest[0] == pytest.approx(1.0)
    assert result.clean_prices[0] == pytest.approx(result.dirty_prices[0] - 1.0)
    on_coupon = qfin.price_bonds(bond, curve, settlement=0.5, engine="numpy")
    assert on_coupon.accrued_interest[0] == 0.0


def test_irregular_final_stub_and_maturity_boundary() -> None:
    bond = qfin.FixedRateBond(maturity=1.25, coupon_rate=0.12, frequency=2)
    times, amounts = bond.cashflows()
    np.testing.assert_allclose(times, [0.5, 1.0, 1.25])
    np.testing.assert_allclose(amounts, [6.0, 6.0, 103.0])
    matured_times, matured_amounts = bond.cashflows(settlement=1.25)
    assert matured_times.size == matured_amounts.size == 0


def test_empty_batch_and_invalid_inputs(flat_curve: qfin.YieldCurve) -> None:
    empty = qfin.price_bonds([], flat_curve, engine="numpy")
    assert empty.dirty_prices.size == 0
    with pytest.raises(ValueError, match="finite"):
        qfin.FixedRateBond(1.0, float("nan"))
    with pytest.raises(ValueError, match="positive"):
        qfin.FixedRateBond(0.0, 0.01)
    with pytest.raises(ValueError, match="integer"):
        qfin.FixedRateBond(1.0, 0.01, frequency=2.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="prices"):
        qfin.yield_from_prices(qfin.FixedRateBond(1.0, 0.0), 0.0)


def test_yield_dv01_near_domain_boundary_and_unbracketed_solve() -> None:
    bond = qfin.FixedRateBond(1.0, 0.0, frequency=1)
    reference = qfin.price_bonds_from_yield(bond, -0.99995, engine="numpy")
    native = qfin.price_bonds_from_yield(bond, -0.99995, engine="native")
    assert np.isfinite(reference.dv01[0])
    np.testing.assert_allclose(native.dv01, reference.dv01, rtol=1e-13)
    for engine in ("numpy", "native"):
        solved = qfin.yield_from_prices(bond, 1.0e-100, engine=engine)
        assert not solved.converged[0]
        assert solved.iterations[0] == 0
