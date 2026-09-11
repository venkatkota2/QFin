"""Independent expected values: no production reference generators in this test."""

import json
from pathlib import Path

import numpy as np
import pytest

import qfin

DATA = Path(__file__).parent / "reference_data"


def cases(name):
    return [
        pytest.param(case, id=case["id"])
        for case in json.loads((DATA / f"{name}.json").read_text())
    ]


def check(actual, expected, *, rtol=2e-12, atol=1e-12, materiality=None):
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    if materiality is not None:
        assert np.max(np.abs(np.asarray(actual) - np.asarray(expected)), initial=0) <= materiality


@pytest.mark.parametrize("case", cases("rate_conversions"))
def test_reference_rates(case):
    r, t, c = case["rate"], case["time"], case["compounding"]
    check(qfin.discount_factor(r, t, c), case["discount"], rtol=2e-14, atol=0)
    check(qfin.continuous_rate(r, c, time=t), case["continuous"], rtol=2e-13, atol=1e-25)


@pytest.mark.parametrize("case", cases("daycounts"))
def test_reference_daycounts(case):
    args = case["start"], case["end"], case["convention"]
    assert qfin.day_count(*args) == case["days"]
    check(qfin.year_fraction(*args), case["year_fraction"], atol=2e-14)


@pytest.mark.parametrize("case", cases("schedules"))
def test_reference_schedules(case):
    args = {k: v for k, v in case.items() if k not in {"source", "id", "dates", "holidays"}}
    args["calendar"] = qfin.Calendar(holidays=frozenset(case["holidays"]))
    assert [d.isoformat() for d in qfin.Schedule(**args).dates] == case["dates"]


@pytest.mark.parametrize("case", cases("fixed_income"))
@pytest.mark.parametrize("engine", ["numpy", "native"])
def test_reference_bonds(case, engine):
    if engine == "native" and not qfin.system_info()["native_extension"]:
        pytest.skip("native extension unavailable")
    bond = qfin.FixedRateBond(
        **{key: case[key] for key in ("maturity", "coupon_rate", "face_value", "frequency")}
    )
    result = qfin.price_bonds_from_yield([bond], case["yield"], engine=engine)
    materiality = max(1e-10, case["face_value"] * 1e-12)
    check(result.dirty_prices, [case["price"]], materiality=materiality)
    check(result.ytm_macaulay_duration, [case["macaulay"]])
    check(result.ytm_modified_duration, [case["modified"]])
    check(result.ytm_convexity, [case["convexity"]])
    check(result.dv01, [case["dv01"]], atol=materiality, materiality=materiality)
    check(
        result.clean_prices + result.accrued_interest, result.dirty_prices, materiality=materiality
    )
    solved = qfin.yield_from_prices([bond], result.clean_prices, engine=engine)
    assert solved.converged.all()
    check(solved.yields, [case["yield"]], rtol=0, atol=2e-10)


@pytest.mark.parametrize("case", cases("curves"))
def test_reference_curves(case):
    curve = qfin.YieldCurve(
        case["times"], case["rates"], case["interpolation"], case["extrapolation"]
    )
    check(curve.discount(case["query"]), case["discount"], rtol=2e-14, atol=1e-15)
    check(curve.discount(curve.times), curve.node_discount_factors, rtol=2e-14)
    if case["query"] > 0:
        check(np.exp(-curve.zero_rate(case["query"]) * case["query"]), case["discount"], rtol=2e-14)


@pytest.mark.parametrize("case", cases("bootstrapping"))
def test_reference_bootstrap(case):
    bond = qfin.FixedRateBond(2, 0.06)
    instruments = [
        qfin.Deposit(0.5, case["deposit_rate"]),
        qfin.ZeroCouponInstrument(1, case["zero_price"]),
        qfin.BondMarketQuote(bond, case["bond_clean_price"]),
        qfin.SimpleSwap(5, case["swap_rate"]),
        qfin.SimpleSwap(60, case["swap_rate"]),
    ]
    report = qfin.bootstrap_curve(instruments, interpolation="log_linear_discount")
    assert report.success and all(item.passed for item in report.instruments)
    check(report.curve.discount(case["times"]), case["discounts"], rtol=2e-10, atol=1e-12)
    for item in report.instruments:
        assert abs(item.residual) <= item.tolerance
        assert abs(item.residual) <= 1e-8  # price/rate units, alongside solver tolerance


@pytest.mark.parametrize("case", cases("risk"))
@pytest.mark.parametrize("engine", ["numpy", "native"])
def test_reference_risk(case, engine):
    if engine == "native" and not qfin.system_info()["native_extension"]:
        pytest.skip("native extension unavailable")
    result = qfin.aggregate_risk(
        qfin.LossDistribution(case["losses"], case["weights"]),
        confidence=case["confidence"],
        engine=engine,
    )
    check(result.value_at_risk, case["var"], rtol=0, atol=0)
    check(result.expected_shortfall, case["cvar"], atol=1e-8, materiality=1e-8)
    check(result.mean, case["mean"])


@pytest.mark.parametrize("case", cases("alm"))
@pytest.mark.parametrize("engine", ["numpy", "native"])
def test_reference_alm(case, engine):
    if engine == "native" and not qfin.system_info()["native_extension"]:
        pytest.skip("native extension unavailable")
    k = case["scale"]
    model = qfin.ALMModel(
        qfin.AssetPortfolio([qfin.FixedRateBond(5, 0, 100 * k)]),
        qfin.LiabilityPortfolio([qfin.CashFlow(3, 90 * k)]),
        qfin.YieldCurve([1, 60], [case["rate"]] * 2),
    )
    result = model.evaluate(engine=engine)
    for key in ["asset_pv", "liability_pv", "surplus", "deficit", "funding_ratio"]:
        check(getattr(result, key), case[key], atol=max(1e-12, k * 1e-10))


@pytest.mark.parametrize("case", cases("life"))
@pytest.mark.parametrize("engine", ["numpy", "native"])
def test_reference_life(case, engine):
    if engine == "native" and not qfin.system_info()["native_extension"]:
        pytest.skip("native extension unavailable")
    policy = qfin.LifePolicy(40, 10000, case["premium"], case["term"])
    points = qfin.PolicyModelPointSet([policy], [case["count"]])
    assumptions = qfin.ProjectionAssumptions(
        qfin.MortalityTable([0, 120], [case["qx"]] * 2),
        qfin.YieldCurve([0, 60], [0.03, 0.03]),
        lapse_rate=case["lapse"],
        expense_per_policy=case["expense"],
    )
    result = qfin.project_liabilities(points, assumptions, engine=engine)
    for field, key in [
        ("expected_premiums", "premiums"),
        ("expected_benefits", "benefits"),
        ("expected_expenses", "expenses"),
        ("net_liability_cashflows", "net"),
    ]:
        check(getattr(result, field), case[key], materiality=1e-7)
    check(result.present_value, case["pv"], materiality=1e-7)
