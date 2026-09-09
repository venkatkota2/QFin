"""Public numerical boundaries: malformed inputs must not produce plausible results."""

from collections.abc import Callable
from datetime import date, datetime

import numpy as np
import pytest

import qfin


@pytest.mark.parametrize(
    "factory,message",
    [
        (lambda: qfin.YieldCurve([0, 1], [0, 0], interpolation="unsupported"), "interpolation"),
        (lambda: qfin.YieldCurve([0, 1], [0, 0], extrapolation="unsupported"), "extrapolation"),
        (lambda: qfin.CurveMarketQuote(-1, 0), "time"),
        (lambda: qfin.CurveMarketQuote(1, np.nan), "value"),
        (lambda: qfin.CurveMarketQuote(1, 0, quote_type="par"), "quote_type"),
        (lambda: qfin.CurveMarketQuote(1, 0, quote_type="discount_factor"), "positive"),
        (lambda: qfin.CurveMarketQuote(1, 0, instrument=" "), "instrument"),
        (lambda: qfin.YieldCurve.from_discount_factors([], []), "non-zero"),
        (lambda: qfin.YieldCurve.from_discount_factors([-1, 1], [1, 1]), "non-negative"),
        (lambda: qfin.YieldCurve.from_discount_factors([1, 1], [1, 1]), "increasing"),
        (lambda: qfin.YieldCurve.from_discount_factors([0, 1], [1, 0]), "positive"),
        (lambda: qfin.YieldCurve.from_discount_factors([0, 1], [.99, .98]), "time zero"),
        (lambda: qfin.YieldCurve.from_forward_rates([1], []), "adjacent"),
        (lambda: qfin.YieldCurve.from_forward_rates([1, 2], [0]), "start at zero"),
        (lambda: qfin.YieldCurve.from_forward_rates([0, 0], [0]), "increasing"),
        (lambda: qfin.YieldCurve.from_forward_rates([0, 1], [np.nan]), "finite"),
        (lambda: qfin.YieldCurve.from_market_quotes([]), "CurveMarketQuote"),
        (lambda: qfin.YieldCurve.from_market_quotes([
            qfin.CurveMarketQuote(1, .02), qfin.CurveMarketQuote(2, .95, "discount_factor")
        ]), "homogeneous"),
        (lambda: qfin.BondMarketQuote(qfin.FixedRateBond(1, 0), 0), "clean_price"),
        (lambda: qfin.BondMarketQuote(qfin.FixedRateBond(1, 0), 100, identifier=" "),
         "identifier"),
        (lambda: qfin.SimpleSwap(1, .03, frequency=5), "frequency"),
        (lambda: qfin.bootstrap_curve([]), "at least one"),
        (lambda: qfin.bootstrap_curve([qfin.Deposit(1, .01), qfin.Deposit(1, .02)]),
         "unique"),
        (lambda: qfin.FixedRateBond(1, 0).cashflows(settlement=-1), "settlement"),
        (lambda: qfin.par_yield(qfin.FixedRateBond(1, 0), qfin.YieldCurve([0, 1], [0, 0]),
                               target_clean_price=0), "target_clean_price"),
        (lambda: qfin.key_rate_risk(qfin.FixedRateBond(1, 0),
                                  qfin.YieldCurve([0, 1], [0, 0]), bump_size=np.nan),
         "bump_size"),
        (lambda: qfin.LifePolicy(20, 1, 1, 1, crediting_spread=np.inf), "crediting_spread"),
        (lambda: qfin.LifePolicy(20, 1, 1, 1, issue_age=-1), "issue_age"),
        (lambda: qfin.LifePolicy(20, 1, 1, 1, issue_age=19), "age must equal"),
        (lambda: qfin.LifePolicy(20, 1, 1, 1, mortality_category=""), "mortality_category"),
    ],
)
def test_financial_factories_reject_invalid_numerical_contracts(
    factory: Callable[[], object], message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"expected_returns": [np.nan, .1]}, "finite"),
        ({"covariance": [[1]]}, "square"),
        ({"covariance": [[1, np.nan], [np.nan, 1]]}, "finite"),
        ({"covariance": [[1, .2], [.1, 1]]}, "symmetric"),
        ({"risk_aversion": 0}, "risk_aversion"),
        ({"budget": np.inf}, "budget"),
        ({"target_return": np.nan}, "target_return"),
        ({"asset_names": ["a"]}, "asset_names"),
        ({"asset_names": ["a", "a"]}, "unique"),
        ({"lower_bounds": [.7, .7]}, "available budget"),
        ({"upper_bounds": [.4, .4]}, "satisfy the budget"),
        ({"lower_bounds": [.8, 0], "upper_bounds": [.5, 1]}, "cannot exceed"),
        ({"lower_bounds": [-.1, 0]}, "non-negative"),
        ({"lower_bounds": [np.nan, 0]}, "non-NaN"),
        ({"upper_bounds": [1, 1, 1]}, "per asset"),
    ],
)
def test_optimizer_defends_feasibility_before_calling_solver(
    kwargs: dict[str, object], message: str,
) -> None:
    values = {"expected_returns": [.04, .08], "covariance": np.diag([.01, .02])}
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        qfin.MeanVarianceProblem(**values)


def test_constrained_optimizer_matches_one_dimensional_analytical_solution() -> None:
    # w2 = 1-w1; unconstrained stationary point is w1 = 2/3. Bounds
    # force the solution to w1=.8, exercising feasible-start redistribution.
    problem = qfin.MeanVarianceProblem(
        [.04, .04], np.diag([.01, .02]), lower_bounds=[.8, 0], upper_bounds=[1, .2],
    )
    result = problem.solve()
    np.testing.assert_allclose(result.weights, [.8, .2], rtol=0, atol=1e-9)
    assert result.variance == pytest.approx(.8**2 * .01 + .2**2 * .02, abs=1e-12)
    with pytest.raises(ValueError, match="weights"):
        problem.utility([np.nan, 1])
    with pytest.raises(ValueError, match="weights"):
        problem.portfolio_variance([1])


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"factor_names": ("", "b")}, "non-empty"),
        ({"factor_names": ("a", "a")}, "unique"),
        ({"correlation": np.ones((2, 1))}, "square"),
        ({"correlation": [[1, np.nan], [np.nan, 1]]}, "finite"),
        ({"correlation": [[1, .2], [.3, 1]]}, "symmetric"),
        ({"correlation": [[.9, .2], [.2, 1]]}, "diagonal"),
        ({"correlation": [[1, 1.1], [1.1, 1]]}, "correlations"),
        ({"means": [np.nan, 0]}, "means"),
        ({"standard_deviations": [1, 0]}, "positive"),
    ],
)
def test_factor_dependence_inputs_cannot_silently_change_model(
    kwargs: dict[str, object], message: str,
) -> None:
    values = {"factor_names": ("a", "b"), "correlation": np.eye(2)}
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        qfin.GaussianFactorModel(**values)


@pytest.mark.parametrize(
    "alias,expected",
    [("linear", "linear_zero"), ("linear zero rate", "linear_zero"),
     ("linear discount factor", "linear_discount"), ("log-linear", "log_linear_discount"),
     ("loglinear_discount", "log_linear_discount"),
     ("log linear discount factor", "log_linear_discount"),
     ("monotone", "monotone_zero"), ("pchip", "monotone_zero")],
)
def test_existing_curve_aliases_preserve_the_canonical_method(alias: str, expected: str) -> None:
    curve = qfin.YieldCurve([0, 1, 5], [-.01, .02, .04], interpolation=alias)
    reference = qfin.YieldCurve([0, 1, 5], [-.01, .02, .04], interpolation=expected)
    np.testing.assert_array_equal(curve.discount(np.linspace(0, 5, 31)),
                                  reference.discount(np.linspace(0, 5, 31)))


def test_calendar_aliases_and_supported_date_endpoints() -> None:
    assert qfin.CurveExtrapolation.parse("flat") is qfin.CurveExtrapolation.FLAT_ZERO
    assert qfin.CurveExtrapolation.parse("raise") is qfin.CurveExtrapolation.ERROR
    calendar = qfin.Calendar()
    for alias, canonical in [("none", "unadjusted"), ("modifiedfollowing", "modified_following"),
                             ("modifiedpreceding", "modified_preceding")]:
        assert calendar.adjust("2024-06-01", alias) == calendar.adjust("2024-06-01", canonical)
    assert qfin.as_date(datetime(2024, 1, 2, 3)) == date(2024, 1, 2)
    assert qfin.is_month_end(date.max)
    assert qfin.add_months("9999-11-30", 1, end_of_month=True) == date.max
    assert qfin.year_fraction("9999-01-01", date.max, "ACT/ACT ISDA") == pytest.approx(364/365)
    with pytest.raises(ValueError, match="at least one possible business"):
        qfin.Calendar(weekend_days=frozenset(range(7)))
    with pytest.raises(ValueError, match="supported date range"):
        calendar.advance_business_days(date.max, 1)
    with pytest.raises(ValueError, match="supported date range"):
        qfin.Calendar(holidays=frozenset({date.max})).adjust(date.max)
    with pytest.raises(ValueError, match="supported year range"):
        qfin.add_months(date.min, -1)
    with pytest.raises(ValueError, match="convention"):
        calendar.adjust(date.today(), "unsupported")
    with pytest.raises(ValueError, match="ISO"):
        qfin.as_date("2024-02-30")
    with pytest.raises(TypeError, match="date"):
        qfin.as_date(1.5)


@pytest.mark.parametrize("kwargs,message", [
    ({"rate_shocks": np.zeros((0, 1, 2))}, "shape"),
    ({"rate_shocks": np.full((1, 1, 2), np.nan)}, "finite"),
    ({"period_length": 0}, "period_length"),
    ({"dependence_assumption": ""}, "dependence_assumption"),
    ({"inflation_rates": -1}, "inflation"),
    ({"equity_returns": np.ones((2, 4))}, "shape"),
])
def test_economic_scenario_dimensions_and_rates_are_validated(
    kwargs: dict[str, object], message: str,
) -> None:
    values = {"rate_shocks": np.zeros((2, 3, 2))}
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        qfin.EconomicScenarioSet(**values)


def test_one_axis_economic_paths_normalize_without_mutating_the_input() -> None:
    returns = np.array([.01, .02, .03])
    single_period = qfin.EconomicScenarioSet(np.zeros((3, 1, 2)), equity_returns=returns)
    single_scenario = qfin.EconomicScenarioSet(np.zeros((1, 3, 2)), equity_returns=returns)
    np.testing.assert_array_equal(single_period.equity_returns[:, 0], returns)
    np.testing.assert_array_equal(single_scenario.equity_returns[0], returns)
    assert returns.flags.writeable
    with pytest.raises(ValueError, match="horizon"):
        single_period.rate_scenarios(period=1)


@pytest.mark.parametrize("factory,message", [
    (lambda: qfin.RateScenarioSet(np.zeros((0, 2))), "non-empty"),
    (lambda: qfin.RateScenarioSet([[0, np.inf]]), "finite"),
    (lambda: qfin.RateScenarioSet([[0], [1]], labels=("a", "a")), "unique"),
    (lambda: qfin.RateScenarioSet.parallel(qfin.YieldCurve([0], [0]), []), "finite"),
    (lambda: qfin.RateScenarioSet.steepener(qfin.YieldCurve([0], [0]),
                                           short_shift=np.nan, long_shift=0), "finite"),
    (lambda: qfin.RateScenarioSet.key_rate(qfin.YieldCurve([0], [0]),
                                          key_time=0, shift=0, width=0), "width"),
    (lambda: qfin.RateScenarioSet.key_rate(qfin.YieldCurve([0], [0]),
                                          key_time=np.nan, shift=0, width=1), "finite"),
    (lambda: qfin.AssetPortfolio([qfin.FixedRateBond(1, 0)], quantities=[]), "quantities"),
    (lambda: qfin.AssetPortfolio([], equity_value=-1), "equity_value"),
    (lambda: qfin.AssetPortfolio([], cash_value=np.inf), "cash_value"),
    (lambda: qfin.LiabilityPortfolio.from_arrays([1, 2], [1]), "same shape"),
    (lambda: qfin.LiabilityPortfolio.from_arrays([1], [1], inflation_linkage=[-1]), "linkage"),
    (lambda: qfin.CashFlow(-1, 1), "time"),
    (lambda: qfin.CashFlow(1, np.nan), "amount"),
    (lambda: qfin.FixedRateBond(1, 0, face_value=0), "face_value"),
    (lambda: qfin.FixedRateBond(1, 0, first_coupon_date="2024-01-01"), "stub"),
    (lambda: qfin.FixedRateBond(1, 0).schedule, "calendar"),
    (lambda: qfin.MortalityTable([], []), "non-zero"),
    (lambda: qfin.MortalityTable([30, 29], [.01, .02]), "increasing"),
    (lambda: qfin.MortalityTable([30], [.01], category=""), "category"),
    (lambda: qfin.MortalityTable([30], [.01]).qx(np.nan), "ages"),
    (lambda: qfin.MortalityTable([30], [.01]).survival_probability(-1, 1), "age"),
    (lambda: qfin.LifePolicy(-1, 1, 1, 1), "non-negative"),
    (lambda: qfin.LifePolicy(30, 1, 1, 1, policy_duration=2), "policy_duration"),
])
def test_remaining_financial_value_boundaries(
    factory: Callable[[], object], message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


def test_empty_portfolio_and_single_node_risk_have_analytical_results() -> None:
    curve = qfin.YieldCurve([0], [.03])
    empty = qfin.ALMModel(qfin.AssetPortfolio([]), qfin.LiabilityPortfolio([]), curve)
    assert empty.evaluate().surplus == 0
    scenario = qfin.RateScenarioSet.steepener(curve, short_shift=-.01, long_shift=.02)
    np.testing.assert_array_equal(scenario.shocks, [[.02]])
    bond = qfin.FixedRateBond(2, 0)
    risk = qfin.key_rate_risk(bond, curve, engine="numpy")
    assert risk.key_rate_dv01[0, 0] == pytest.approx(risk.parallel_dv01[0], abs=1e-15)
    with pytest.raises(ValueError, match="no coupon"):
        qfin.par_yield(bond, curve, settlement=2)
    matured = qfin.price_bonds(bond, curve, settlement=2, engine="numpy")
    assert matured.dirty_prices[0] == 0
    assert matured.pv01[0] == matured.dv01[0]
    with pytest.raises(ValueError, match="matured"):
        qfin.yield_from_prices(bond, [100], settlement=2, engine="numpy")
    # A one-iteration solve must not claim convergence merely because it returns a value.
    result = qfin.yield_from_prices(bond, [99], max_iterations=1, engine="numpy")
    assert not result.converged[0]
    assert result.iterations[0] == 1
    with pytest.raises(ValueError, match="tolerance"):
        qfin.yield_from_prices(bond, [99], tolerance=0)
    with pytest.raises(ValueError, match="frequency"):
        qfin.price_bonds_from_yield(bond, -2, engine="numpy")
    with pytest.raises(ValueError, match="per bond"):
        qfin.price_bonds_from_yield(bond, [0, 1], engine="numpy")


def test_weighted_attribution_survives_large_weights_and_reconciles() -> None:
    from qfin.finance.alm import ALMFactorAttribution

    attribution = ALMFactorAttribution(("rate", "equity"), np.array([[1., 2.], [3., 4.]]),
                                      np.array([.1, .2]), np.array([3.1, 7.2]))
    reference = attribution.weighted_mean([1, 1])
    assert attribution.weighted_mean([1e308, 1e308]) == reference
    assert reference['total'] == pytest.approx(
        reference['rate'] + reference['equity'] + reference['interaction'], abs=1e-14)
    for weights in ([0, 0], [1], [-1, 2], [np.nan, 1]):
        with pytest.raises(ValueError, match="probabilities"):
            attribution.weighted_mean(weights)


def test_dated_swap_bootstrap_reprices_quoted_negative_rates() -> None:
    instruments = [qfin.Deposit("2025-07-31", -.03),
                   qfin.SimpleSwap("2035-01-31", -.015, end_of_month=True)]
    report = qfin.bootstrap_curve(instruments, valuation_date="2025-01-31",
                                 tolerance=1e-10)
    assert report.success
    assert report.explain()['success']
    assert report.curve.discount(10) > 1
    assert all(abs(item.residual) <= 1e-10 for item in report.instruments)
    bond = qfin.FixedRateBond.from_dates("2025-01-31", "2030-01-31", .03)
    quote = qfin.BondMarketQuote(bond, 100)
    assert quote.maturity == date(2030, 1, 31)
    # Modified following moves Sunday 2027-01-31 back into January.
    expected = np.arange(.5, 5.1, .5)
    expected[3] = 719/360
    np.testing.assert_allclose(bond.payment_schedule(), expected, atol=1e-14)
    assert bond.accrued_interest() == 0
    assert bond.accrued_interest("2025-07-31") == 0


@pytest.mark.parametrize('engine', ['numpy', 'native'])
@pytest.mark.parametrize('shocks,time,amount', [
    ([0., 1e4], 0., 1.),  # Invalid unused curve node must still be rejected.
    ([0., 0.], 1., 1e308),  # Discounting a huge cash flow overflows its PV.
    ([0., 0.], 1e4, 1.),  # Flat-zero extrapolation overflows.
])
def test_scenario_engines_reject_nonrepresentable_curve_or_value(
    engine: str, shocks: list[float], time: float, amount: float,
) -> None:
    from qfin.finance.scenarios import scenario_portfolio_values

    if engine == 'native' and not qfin.system_info()['native_extension']:
        pytest.skip('native extension unavailable')
    curve = qfin.YieldCurve([0., 1.], [-1., -1.])
    with (
        np.errstate(over='ignore', invalid='ignore', under='ignore'),
        pytest.raises(ValueError, match=r'discount|non-finite'),
    ):
        scenario_portfolio_values([time], [amount], [0, 1], [1], curve,
                                  qfin.RateScenarioSet([shocks]), engine=engine)


def test_rate_quote_conversion_boundaries_are_explicit() -> None:
    from qfin.finance.rates import continuous_rate

    assert qfin.discount_factor(-.01, 0) == 1
    assert continuous_rate(-.01, 'simple', time=0) == -.01
    assert qfin.convert_rate(-.01, 'annual', 'annual') == -.01
    for function, args in [(qfin.discount_factor, (np.nan, 1)),
                           (qfin.rate_from_discount_factor, (0, 1)),
                           (qfin.rate_from_discount_factor, (1, 0)),
                           (qfin.RateQuote, (np.inf,)),
                           (continuous_rate, (np.nan, 'continuous')),
                           (continuous_rate, (-1., 'simple'))]:
        with pytest.raises(ValueError, match=r'finite|positive|requires'):
            function(*args)
    with pytest.raises(ValueError, match='compounding'):
        qfin.Compounding.parse('unsupported')
    with pytest.raises(ValueError, match='positive'):
        qfin.convert_rate(.01, 'annual', 'annual', time=0)
    mortality = qfin.MortalityTable([30, 40], [.01, .03])
    np.testing.assert_allclose(mortality.px([30, 35, 40]), [.99, .98, .97], atol=1e-15)
