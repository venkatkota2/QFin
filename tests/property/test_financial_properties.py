"""Generated valid financial objects with economic, metamorphic reference checks."""

from datetime import date, timedelta

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

import qfin

RATES = st.floats(-0.05, 0.2, allow_nan=False, allow_infinity=False)
AMOUNTS = st.floats(1e-4, 1e12, allow_nan=False, allow_infinity=False)
PROBABILITIES = st.floats(0, 1, allow_nan=False, allow_infinity=False)


@st.composite
def bonds(draw):
    frequency = draw(st.sampled_from([1, 2, 4, 12]))
    periods = draw(st.integers(1, 120))
    return qfin.FixedRateBond(
        periods / frequency, draw(st.floats(0, 0.2)), draw(AMOUNTS), frequency
    )


@st.composite
def curves(draw):
    ticks = draw(st.lists(st.integers(1, 240), min_size=2, max_size=6, unique=True))
    nodes = np.array(sorted(ticks)) / 4
    rates = draw(st.lists(RATES, min_size=len(nodes), max_size=len(nodes)))
    method = draw(
        st.sampled_from(["linear_zero", "linear_discount", "log_linear_discount", "monotone_zero"])
    )
    return qfin.YieldCurve(nodes, rates, interpolation=method)


@st.composite
def distributions(draw):
    size = draw(st.integers(2, 15))
    losses = draw(st.lists(st.integers(-10000, 10000), min_size=size, max_size=size))
    weights = draw(st.lists(st.integers(1, 10000), min_size=size, max_size=size))
    return np.array(losses, dtype=float), np.array(weights, dtype=float)


@given(bonds(), RATES, st.floats(1e-6, 0.05))
def test_positive_bond_monotonicity_and_yield_roundtrip(bond, yield_rate, bump):
    price = qfin.price_bonds_from_yield(bond, yield_rate, engine="numpy")
    higher = qfin.price_bonds_from_yield(bond, yield_rate + bump, engine="numpy")
    assert higher.dirty_prices[0] < price.dirty_prices[0]
    solved = qfin.yield_from_prices(bond, price.dirty_prices, engine="numpy")
    assert solved.converged.all()
    assert solved.yields[0] == pytest.approx(yield_rate, abs=4e-11)
    recovered = qfin.price_bonds_from_yield(bond, solved.yields, engine="numpy")
    np.testing.assert_allclose(recovered.dirty_prices, price.dirty_prices, rtol=1e-10)


@given(bonds(), RATES, st.floats(0.001, 1e6))
def test_cashflow_scaling_and_normalized_sensitivities(bond, rate, scale):
    bigger = qfin.FixedRateBond(
        bond.maturity, bond.coupon_rate, bond.face_value * scale, bond.frequency
    )
    first = qfin.price_bonds_from_yield(bond, rate, engine="numpy")
    second = qfin.price_bonds_from_yield(bigger, rate, engine="numpy")
    np.testing.assert_allclose(second.dirty_prices, scale * first.dirty_prices, rtol=2e-13)
    np.testing.assert_allclose(second.dv01, scale * first.dv01, rtol=2e-10)
    np.testing.assert_allclose(
        second.ytm_modified_duration, first.ytm_modified_duration, rtol=2e-13
    )


@given(bonds(), RATES)
def test_key_rate_reconciliation_with_finite_difference_remainder(bond, rate):
    curve = qfin.YieldCurve([0, 1, 5, 60], [rate] * 4)
    risk = qfin.key_rate_risk(bond, curve, engine="numpy")
    price = qfin.price_bonds(bond, curve, engine="numpy").dirty_prices[0]
    difference = abs(np.sum(risk.key_rate_dv01) - risk.parallel_dv01[0])
    # Partition-of-unity first derivatives reconcile exactly. Separate central
    # differences have O((bump*T)^3) price remainders at nonzero bump size.
    bound = price * (1e-4 * bond.maturity) ** 3 / 3 + price * 1e-12
    assert difference <= bound


@given(curves(), st.integers(1, 7), st.integers(1, 7))
def test_alm_zero_shock_scenario_permutation_chunking(curve, chunk1, chunk2):
    model = qfin.ALMModel(
        qfin.AssetPortfolio([qfin.FixedRateBond(5, 0.04)]),
        qfin.LiabilityPortfolio([qfin.CashFlow(3, 80)]),
        curve,
    )
    shifts = np.linspace(-0.01, 0.01, 7)[:, None] * np.ones((1, len(curve.times)))
    scenarios = qfin.RateScenarioSet(shifts)
    result = model.run_scenarios(scenarios, engine="numpy", chunk_size=chunk1)
    permuted = model.run_scenarios(
        qfin.RateScenarioSet(shifts[::-1]), engine="numpy", chunk_size=chunk2
    )
    np.testing.assert_allclose(result.surplus, permuted.surplus[::-1], rtol=2e-13, atol=1e-12)
    base = model.evaluate(engine="numpy")
    assert result.surplus[3] == pytest.approx(base.surplus, abs=1e-11)
    np.testing.assert_allclose(
        result.loss_distribution().losses, base.surplus - result.surplus, atol=1e-11
    )
    assert base.surplus == pytest.approx(base.asset_pv - base.liability_pv)
    assert base.deficit == max(-base.surplus, 0)
    assert base.funding_ratio == pytest.approx(base.asset_pv / base.liability_pv)


@given(
    distributions(),
    st.sampled_from([0.9, 0.95, 0.99, 0.995, 0.999]),
    st.integers(-250, 250),
    st.integers(-10000, 10000),
)
def test_weight_scale_order_and_loss_translation(data, alpha, exponent, translation):
    losses, weights = data
    first = qfin.aggregate_risk(
        qfin.LossDistribution(losses, weights), confidence=alpha, engine="numpy"
    )
    second = qfin.aggregate_risk(
        qfin.LossDistribution(losses[::-1] + translation, weights[::-1] * 10.0**exponent),
        confidence=alpha,
        engine="numpy",
    )
    assert second.var == pytest.approx(first.var + translation, abs=1e-10)
    assert second.cvar == pytest.approx(first.cvar + translation, abs=1e-8)
    if qfin.system_info()["native_extension"]:
        native = qfin.aggregate_risk(
            qfin.LossDistribution(losses, weights), confidence=alpha, engine="native"
        )
        assert native.var == first.var
        assert native.cvar == pytest.approx(first.cvar, abs=1e-8)


@given(PROBABILITIES, PROBABILITIES, st.integers(1, 30), st.integers(1, 100000))
def test_life_grouping_matches_scalar_survival(qx, lapse, term, count):
    policy = qfin.LifePolicy(40, 10000, 0, term)
    curve = qfin.YieldCurve([0, 60], [0.03, 0.03])
    assumptions = qfin.ProjectionAssumptions(
        qfin.MortalityTable([0, 120], [qx, qx]), curve, lapse_rate=lapse
    )
    grouped = qfin.project_liabilities(
        qfin.PolicyModelPointSet([policy], [count]), assumptions, engine="numpy"
    )
    split = qfin.project_liabilities(
        qfin.PolicyModelPointSet([policy, policy], [count / 2, count / 2]),
        assumptions,
        engine="numpy",
    )
    expected = sum(
        count * 10000 * qx * ((1 - qx) * (1 - lapse)) ** year * np.exp(-0.03 * (year + 1))
        for year in range(term)
    )
    assert grouped.present_value == pytest.approx(expected, rel=2e-12, abs=1e-8)
    np.testing.assert_allclose(
        split.net_liability_cashflows, grouped.net_liability_cashflows, rtol=2e-13
    )


@given(
    st.integers(2000, 2035),
    st.integers(1, 12),
    st.integers(1, 28),
    st.integers(1, 700),
    st.sampled_from(["forward", "backward"]),
)
def test_dated_schedule_and_settlement_identity(year, month, day, days, generation):
    issue = date(year, month, day)
    maturity = issue + timedelta(days=days + 400)
    settlement = issue + timedelta(days=days)
    bond = qfin.FixedRateBond.from_dates(
        issue,
        maturity,
        0.05,
        date_generation=generation,
        business_day_convention="unadjusted",
    )
    curve = qfin.YieldCurve([0, 10], [0.04, 0.04], valuation_date=settlement)
    result = qfin.price_bonds(bond, curve, engine="numpy")
    np.testing.assert_allclose(
        result.clean_prices + result.accrued_interest, result.dirty_prices, rtol=2e-14
    )
    assert all(a < b for a, b in zip(bond.schedule.dates, bond.schedule.dates[1:], strict=False))


@given(
    RATES,
    st.sampled_from(["annual", "semiannual", "quarterly", "monthly", "simple", "continuous"]),
    st.floats(0.01, 10),
)
def test_rate_conversion_preserves_discount(rate, convention, time):
    df = qfin.discount_factor(rate, time, convention)
    continuous = qfin.convert_rate(rate, convention, "continuous", time=time)
    assert qfin.discount_factor(continuous, time) == pytest.approx(df, rel=2e-14)


@given(st.lists(st.floats(0.01, 1), min_size=2, max_size=5), st.floats(0.01, 10))
def test_positive_definite_optimization_budget(variances, aversion):
    n = len(variances)
    problem = qfin.MeanVarianceProblem(
        np.linspace(0.02, 0.1, n), np.diag(variances), risk_aversion=aversion
    )
    result = qfin.compile(problem).run()
    assert np.sum(result.weights) == pytest.approx(1, abs=1e-8)
    assert np.all(result.weights >= -1e-8)
