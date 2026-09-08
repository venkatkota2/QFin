from math import exp

import numpy as np
import pytest

import qfin


def test_bond_valuation_scales_linearly_without_changing_normalized_risk() -> None:
    curve = qfin.YieldCurve(
        [0.0, 0.5, 3.0, 12.0, 40.0],
        [-0.01, 0.005, 0.035, 0.06, 0.025],
        interpolation="monotone_zero",
        extrapolation="flat_forward",
    )
    base = qfin.FixedRateBond(17.25, 0.0475, face_value=100.0, frequency=4)
    scaled = qfin.FixedRateBond(17.25, 0.0475, face_value=1.0e8, frequency=4)
    base_result = qfin.price_bonds(base, curve, settlement=0.37, engine="numpy")
    scaled_result = qfin.price_bonds(scaled, curve, settlement=0.37, engine="numpy")
    scale = scaled.face_value / base.face_value

    for field in ("dirty_prices", "clean_prices", "accrued_interest", "dv01", "cs01"):
        np.testing.assert_allclose(
            getattr(scaled_result, field),
            scale * getattr(base_result, field),
            rtol=2.0e-14,
            atol=1.0e-8,
        )
    for field in (
        "macaulay_duration",
        "modified_duration",
        "convexity",
        "effective_duration",
        "effective_convexity",
    ):
        np.testing.assert_allclose(
            getattr(scaled_result, field),
            getattr(base_result, field),
            rtol=2.0e-13,
            atol=2.0e-13,
        )


def test_positive_cashflow_bond_price_decreases_with_yield_and_round_trips() -> None:
    generator = np.random.default_rng(20_260_905)
    bonds: list[qfin.FixedRateBond] = []
    yields: list[float] = []
    for _ in range(80):
        frequency = int(generator.choice([1, 2, 4]))
        bonds.append(
            qfin.FixedRateBond(
                maturity=float(generator.uniform(0.05, 55.0)),
                coupon_rate=float(generator.uniform(0.0, 0.20)),
                face_value=float(10.0 ** generator.uniform(-2.0, 10.0)),
                frequency=frequency,
            )
        )
        yields.append(float(generator.uniform(-0.8 * frequency, 0.75)))

    prices = qfin.price_bonds_from_yield(bonds, yields, engine="numpy")
    solved = qfin.yield_from_prices(bonds, prices.dirty_prices, engine="numpy")
    assert np.all(solved.converged)
    np.testing.assert_allclose(solved.yields, yields, rtol=2.0e-10, atol=2.0e-10)

    template = qfin.FixedRateBond(30.0, 0.055, frequency=2)
    ordered_yields = np.linspace(-1.5, 0.75, 101)
    ordered_prices = qfin.price_bonds_from_yield(
        [template] * ordered_yields.size,
        ordered_yields,
        engine="numpy",
    ).dirty_prices
    assert np.all(np.diff(ordered_prices) < 0.0)


def test_yield_duration_and_convexity_match_numerical_derivatives() -> None:
    bonds = (
        qfin.FixedRateBond(0.3, 0.0, frequency=4),
        qfin.FixedRateBond(7.4, 0.06, frequency=2),
        qfin.FixedRateBond(50.5, 0.015, frequency=1),
    )
    yields = (-0.02, 0.0475, 0.18)
    bump = 1.0e-4
    for bond, yield_rate in zip(bonds, yields, strict=True):
        base = qfin.price_bonds_from_yield(bond, yield_rate, engine="numpy")
        down = qfin.price_bonds_from_yield(bond, yield_rate - bump, engine="numpy")
        up = qfin.price_bonds_from_yield(bond, yield_rate + bump, engine="numpy")
        price = base.dirty_prices[0]
        numerical_duration = (down.dirty_prices[0] - up.dirty_prices[0]) / (
            2.0 * bump * price
        )
        numerical_convexity = (
            down.dirty_prices[0] - 2.0 * price + up.dirty_prices[0]
        ) / (bump * bump * price)

        assert base.ytm_modified_duration[0] == pytest.approx(
            numerical_duration,
            rel=2.0e-6,
            abs=2.0e-8,
        )
        assert base.ytm_convexity[0] == pytest.approx(
            numerical_convexity,
            rel=2.0e-6,
            abs=2.0e-6,
        )


@pytest.mark.parametrize(
    "interpolation",
    ["linear_zero", "linear_discount", "log_linear_discount", "monotone_zero"],
)
def test_curve_zero_shift_composition_and_node_reproduction(interpolation: str) -> None:
    times = np.array([0.0, 0.25, 2.0, 10.0, 60.0])
    rates = np.array([-0.02, -0.01, 0.04, 0.08, 0.025])
    first = np.array([0.001, -0.002, 0.0, 0.003, -0.001])
    second = np.array([-0.0005, 0.001, 0.002, -0.001, 0.0005])
    curve = qfin.YieldCurve(
        times,
        rates,
        interpolation=interpolation,
        extrapolation="flat_forward",
    )

    np.testing.assert_array_equal(curve.shifted(0.0).zero_rates, curve.zero_rates)
    np.testing.assert_allclose(curve.zero_rate(times), rates, rtol=0.0, atol=2.0e-15)
    composed = curve.shifted(first).shifted(second)
    combined = curve.shifted(first + second)
    np.testing.assert_allclose(composed.zero_rates, combined.zero_rates, rtol=0.0, atol=1e-17)
    sample_times = np.linspace(0.0, 80.0, 257)
    np.testing.assert_allclose(
        composed.discount(sample_times),
        combined.discount(sample_times),
        rtol=3.0e-14,
        atol=3.0e-14,
    )


@pytest.mark.parametrize(
    "compounding",
    ["continuous", "annual", "semiannual", "quarterly", "monthly", "simple"],
)
def test_rate_discount_round_trip_over_valid_random_domains(compounding: str) -> None:
    generator = np.random.default_rng(9_173)
    for _ in range(100):
        time = float(10.0 ** generator.uniform(-4.0, 2.0))
        lower = -0.009 / time if compounding == "simple" else -0.75
        rate = float(generator.uniform(lower, 0.75))
        discount = qfin.discount_factor(rate, time, compounding)
        recovered = qfin.rate_from_discount_factor(discount, time, compounding)
        assert recovered == pytest.approx(rate, rel=2.0e-12, abs=2.0e-12)


def test_alm_joint_scaling_preserves_ratios_and_scales_losses() -> None:
    curve = qfin.YieldCurve([0.0, 1.0, 5.0, 20.0], [0.01, 0.02, 0.035, 0.04])
    bonds = [qfin.FixedRateBond(4.0, 0.03), qfin.FixedRateBond(17.0, 0.05)]
    quantities = np.array([7.0, 11.0])
    liability_times = np.array([1.0, 4.0, 12.0, 25.0])
    liability_amounts = np.array([200.0, 400.0, 900.0, 1_200.0])
    factor = 1.0e6
    base = qfin.ALMModel(
        qfin.AssetPortfolio(bonds, quantities),
        qfin.LiabilityPortfolio.from_arrays(liability_times, liability_amounts),
        curve,
    )
    scaled = qfin.ALMModel(
        qfin.AssetPortfolio(bonds, factor * quantities),
        qfin.LiabilityPortfolio.from_arrays(
            liability_times,
            factor * liability_amounts,
        ),
        curve,
    )
    base_value = base.evaluate(engine="numpy")
    scaled_value = scaled.evaluate(engine="numpy")
    for field in ("asset_pv", "liability_pv", "surplus", "deficit"):
        assert getattr(scaled_value, field) == pytest.approx(
            factor * getattr(base_value, field),
            rel=3.0e-14,
            abs=2.0e-6,
        )
    for field in (
        "funding_ratio",
        "asset_duration",
        "liability_duration",
        "duration_gap",
        "asset_convexity",
        "liability_convexity",
        "convexity_gap",
    ):
        assert getattr(scaled_value, field) == pytest.approx(
            getattr(base_value, field), rel=3.0e-14, abs=3.0e-14
        )

    scenarios = qfin.RateScenarioSet.parallel(curve, [-0.01, 0.0, 0.015])
    base_losses = base.run_scenarios(scenarios, engine="numpy").loss_distribution().losses
    scaled_losses = scaled.run_scenarios(
        scenarios, engine="numpy"
    ).loss_distribution().losses
    np.testing.assert_allclose(
        scaled_losses,
        factor * base_losses,
        rtol=2.0e-13,
        atol=2.0e-6,
    )


def test_life_grouping_and_model_point_scaling_preserve_aggregates() -> None:
    mortality = qfin.MortalityTable([0.0, 40.0, 80.0, 120.0], [0.0, 0.002, 0.04, 1.0])
    assumptions = qfin.ProjectionAssumptions(
        mortality,
        qfin.YieldCurve([0.0, 50.0], [0.025, 0.025]),
        lapse_rate=0.03,
        expense_per_policy=17.0,
    )
    first = qfin.LifePolicy(40, 100_000, 600, 12)
    second = qfin.LifePolicy(55, 75_000, 900, 8)
    expanded_policies = [first, first, first, second, second]
    expanded = qfin.project_liabilities(expanded_policies, assumptions, engine="numpy")
    grouped = qfin.project_liabilities(
        qfin.PolicyModelPointSet.from_policies(expanded_policies),
        assumptions,
        engine="numpy",
    )
    for field in (
        "expected_premiums",
        "expected_benefits",
        "expected_expenses",
        "expected_surrenders",
        "net_liability_cashflows",
        "active",
        "disabled",
        "deaths",
    ):
        np.testing.assert_allclose(
            getattr(grouped, field), getattr(expanded, field), rtol=2.0e-14, atol=2.0e-12
        )
    assert grouped.present_value == pytest.approx(expanded.present_value, rel=2.0e-14)

    one = qfin.project_liabilities(
        qfin.PolicyModelPointSet([first], [1.0]), assumptions, engine="numpy"
    )
    many = qfin.project_liabilities(
        qfin.PolicyModelPointSet([first], [1.0e7]), assumptions, engine="numpy"
    )
    np.testing.assert_allclose(
        many.net_liability_cashflows,
        1.0e7 * one.net_liability_cashflows,
        rtol=3.0e-14,
        atol=2.0e-5,
    )
    assert many.present_value == pytest.approx(1.0e7 * one.present_value, rel=3.0e-14)
    assert many.duration == pytest.approx(one.duration, rel=3.0e-14)


def test_weighted_risk_is_normalization_and_permutation_invariant() -> None:
    generator = np.random.default_rng(81_833)
    losses = generator.normal(1.0e6, 3.0e5, 2_000)
    weights = generator.lognormal(0.0, 3.0, losses.size)
    permutation = generator.permutation(losses.size)
    base = qfin.aggregate_risk(
        qfin.LossDistribution(losses, weights), confidence=0.9975, engine="numpy"
    )
    scaled = qfin.aggregate_risk(
        qfin.LossDistribution(losses, 1.0e150 * weights),
        confidence=0.9975,
        engine="numpy",
    )
    permuted = qfin.aggregate_risk(
        qfin.LossDistribution(losses[permutation], weights[permutation]),
        confidence=0.9975,
        engine="numpy",
    )
    for field in (
        "mean",
        "standard_deviation",
        "minimum",
        "maximum",
        "var",
        "cvar",
    ):
        assert getattr(scaled, field) == pytest.approx(
            getattr(base, field), rel=2.0e-14, abs=2.0e-8
        )
        assert getattr(permuted, field) == pytest.approx(
            getattr(base, field), rel=2.0e-14, abs=2.0e-8
        )
    assert base.cvar >= base.var

    uniform = qfin.aggregate_risk(
        qfin.LossDistribution(losses, np.ones(losses.size)),
        confidence=0.95,
        engine="numpy",
    )
    unweighted = qfin.aggregate_risk(
        qfin.LossDistribution(losses), confidence=0.95, engine="numpy"
    )
    assert uniform == unweighted


def test_zero_coupon_price_matches_direct_discounting_across_curve_shapes() -> None:
    for interpolation in (
        "linear_zero",
        "linear_discount",
        "log_linear_discount",
        "monotone_zero",
    ):
        curve = qfin.YieldCurve(
            [0.0, 0.5, 5.0, 30.0],
            [-0.01, 0.015, 0.07, 0.025],
            interpolation=interpolation,
            extrapolation="flat_forward",
        )
        for maturity in (0.01, 0.5, 3.7, 30.0, 75.0):
            bond = qfin.FixedRateBond(maturity, 0.0, face_value=1.0e12)
            result = qfin.price_bonds(bond, curve, engine="numpy")
            assert result.dirty_prices[0] == pytest.approx(
                1.0e12 * curve.discount(maturity),
                rel=2.0e-14,
                abs=2.0e-4,
            )
            assert result.macaulay_duration[0] == pytest.approx(maturity, abs=2.0e-14)
            assert result.convexity[0] == pytest.approx(maturity**2, abs=2.0e-12)
            if curve.zero_rate(maturity) == pytest.approx(0.0):
                assert result.dirty_prices[0] == pytest.approx(1.0e12)
            else:
                assert np.isfinite(exp(-curve.zero_rate(maturity) * maturity))
