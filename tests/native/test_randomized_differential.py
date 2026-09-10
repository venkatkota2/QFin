from datetime import date, timedelta

import numpy as np
import pytest

import qfin

pytestmark = pytest.mark.skipif(
    not qfin.system_info()["native_extension"], reason="native extension unavailable"
)


def _assert_batch_analytics_close(
    native: qfin.BondBatchAnalytics,
    reference: qfin.BondBatchAnalytics,
) -> None:
    for field, relative, absolute in (
        ("dirty_prices", 4.0e-13, 2.0e-8),
        ("clean_prices", 4.0e-13, 2.0e-8),
        ("accrued_interest", 0.0, 0.0),
        ("macaulay_duration", 4.0e-13, 2.0e-12),
        ("modified_duration", 4.0e-13, 2.0e-12),
        ("convexity", 8.0e-13, 2.0e-10),
        ("dv01", 5.0e-11, 2.0e-8),
        ("effective_duration", 5.0e-12, 2.0e-10),
        ("effective_convexity", 5.0e-9, 2.0e-7),
    ):
        np.testing.assert_allclose(
            getattr(native, field),
            getattr(reference, field),
            rtol=relative,
            atol=absolute,
        )


def test_randomized_fixed_income_and_yield_native_differential() -> None:
    generator = np.random.default_rng(27_182_818)
    for case in range(12):
        size = 1 + 7 * case
        bonds: list[qfin.FixedRateBond] = []
        yields = np.empty(size, dtype=np.float64)
        for index in range(size):
            frequency = (1, 2, 4)[(index + case) % 3]
            maturity = float(10.0 ** generator.uniform(-1.5, np.log10(60.0)))
            if index == 0:
                maturity = 0.01
                yield_rate = -0.9999 * frequency
            elif index == 1 and size > 1:
                maturity = 55.0
                yield_rate = 0.75
            else:
                yield_rate = float(generator.uniform(-0.15, 0.50))
            bonds.append(
                qfin.FixedRateBond(
                    maturity=maturity,
                    coupon_rate=float(generator.choice([0.0, generator.uniform(0.001, 0.25)])),
                    face_value=float(10.0 ** generator.uniform(-3.0, 12.0)),
                    frequency=frequency,
                )
            )
            yields[index] = yield_rate

        reference = qfin.price_bonds_from_yield(bonds, yields, engine="numpy")
        native = qfin.price_bonds_from_yield(bonds, yields, engine="native")
        _assert_batch_analytics_close(native, reference)

        reference_solve = qfin.yield_from_prices(
            bonds, reference.dirty_prices, engine="numpy"
        )
        native_solve = qfin.yield_from_prices(
            bonds, reference.dirty_prices, engine="native"
        )
        assert np.all(reference_solve.converged)
        assert np.all(native_solve.converged)
        np.testing.assert_array_equal(native_solve.iterations, reference_solve.iterations)
        np.testing.assert_allclose(native_solve.yields, reference_solve.yields, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(native_solve.yields, yields, rtol=3.0e-11, atol=4.0e-12)


def test_dated_bond_native_differential_across_settlements() -> None:
    valuation_date = date(2024, 1, 31)
    curve = qfin.YieldCurve(
        [0.0, 0.5, 3.0, 12.0, 60.0],
        [-0.01, 0.0, 0.035, 0.065, 0.025],
        valuation_date=valuation_date,
    )
    bonds = [
        qfin.FixedRateBond.from_dates(
            valuation_date - timedelta(days=365 * (1 + index % 3)),
            valuation_date + timedelta(days=30 + 365 * (1 + index % 55)),
            (0.0, 0.015, 0.12)[index % 3],
            face_value=10.0 ** (index % 9),
            frequency=(1, 2, 4)[index % 3],
            business_day_convention="unadjusted",
        )
        for index in range(30)
    ]
    reference = qfin.price_bonds(bonds, curve, engine="numpy")
    native = qfin.price_bonds(bonds, curve, engine="native")
    _assert_batch_analytics_close(native, reference)
    np.testing.assert_allclose(
        native.clean_prices + native.accrued_interest,
        native.dirty_prices,
        rtol=0.0,
        atol=2.0e-8,
    )


@pytest.mark.parametrize("confidence", [0.90, 0.95, 0.995, 0.9999])
def test_randomized_weighted_risk_native_differential(confidence: float) -> None:
    generator = np.random.default_rng(31_415_926 + round(confidence * 10_000))
    losses = np.concatenate(
        (
            generator.normal(0.0, 1.0e12, 2_000),
            np.repeat([-1.0e12, 0.0, 1.0e12], 20),
            np.array([-1.0, 1.0]),
        )
    )
    probabilities = 10.0 ** generator.uniform(-250.0, 0.0, losses.size)
    permutation = generator.permutation(losses.size)
    distribution = qfin.LossDistribution(
        losses[permutation], probabilities[permutation]
    )

    reference = qfin.aggregate_risk(distribution, confidence=confidence, engine="numpy")
    native = qfin.aggregate_risk(distribution, confidence=confidence, engine="native")
    for field in ("mean", "standard_deviation", "minimum", "maximum", "var", "cvar"):
        assert getattr(native, field) == pytest.approx(
            getattr(reference, field), rel=8.0e-14, abs=2.0e-3
        )


def test_randomized_life_projection_native_differential() -> None:
    generator = np.random.default_rng(16_180_339)
    mortality = qfin.MortalityTable(
        np.arange(0.0, 121.0),
        np.minimum(1.0, 1.0e-5 * np.exp(0.09 * np.arange(0.0, 121.0))),
    )
    assumptions = qfin.ProjectionAssumptions(
        mortality,
        qfin.YieldCurve([0.0, 1.0, 10.0, 60.0], [-0.01, 0.02, 0.05, 0.03]),
        lapse_rate=np.linspace(0.01, 0.08, 60),
        expense_per_policy=35.0,
        disability_rate=0.015,
        recovery_rate=0.12,
        disabled_mortality_multiplier=1.8,
        expense_inflation_rate=0.025,
    )
    for case in range(8):
        policies: list[qfin.LifePolicy] = []
        counts: list[float] = []
        for index in range(1 + case * 3):
            term = int(generator.integers(1, 61))
            age = float(generator.integers(18, 90))
            policies.append(
                qfin.LifePolicy(
                    age,
                    sum_assured=float(10.0 ** generator.uniform(2.0, 9.0)),
                    annual_premium=float(10.0 ** generator.uniform(-1.0, 6.0)),
                    term=term,
                    disability_benefit=float(generator.uniform(0.0, 50_000.0)),
                )
            )
            counts.append(float(10.0 ** ((index + case) % 7)))
        model_points = qfin.PolicyModelPointSet(policies, counts)
        reference = qfin.project_liabilities(model_points, assumptions, engine="numpy")
        native = qfin.project_liabilities(model_points, assumptions, engine="native")
        for field in (
            "expected_premiums",
            "expected_benefits",
            "expected_expenses",
            "expected_surrenders",
            "net_liability_cashflows",
            "active",
            "disabled",
            "deaths",
            "policy_present_values",
        ):
            np.testing.assert_allclose(
                getattr(native, field),
                getattr(reference, field),
                rtol=4.0e-13,
                atol=2.0e-5,
            )
        assert native.present_value == pytest.approx(
            reference.present_value, rel=4.0e-13, abs=2.0e-5
        )
        assert native.duration == pytest.approx(
            reference.duration, rel=4.0e-13, abs=2.0e-11
        )


def test_randomized_alm_scenario_native_differential() -> None:
    generator = np.random.default_rng(14_142_135)
    for node_count, asset_count in ((1, 1), (2, 17), (7, 53), (23, 101)):
        curve_times = np.linspace(0.0, 60.0, node_count)
        curve = qfin.YieldCurve(
            curve_times,
            generator.uniform(-0.03, 0.12, node_count),
        )
        bonds = [
            qfin.FixedRateBond(
                float(generator.uniform(0.01, 75.0)),
                float(generator.uniform(0.0, 0.20)),
                face_value=float(10.0 ** generator.uniform(0.0, 8.0)),
                frequency=(1, 2, 4)[index % 3],
            )
            for index in range(asset_count)
        ]
        model = qfin.ALMModel(
            qfin.AssetPortfolio(bonds, generator.uniform(-5.0, 20.0, asset_count)),
            qfin.LiabilityPortfolio.from_arrays(
                generator.uniform(0.0, 80.0, 31),
                generator.uniform(-1.0e8, 1.0e9, 31),
            ),
            curve,
        )
        scenarios = qfin.RateScenarioSet(
            generator.uniform(-0.08, 0.08, (37, node_count))
        )
        reference = model.run_scenarios(scenarios, engine="numpy", chunk_size=7)
        native = model.run_scenarios(scenarios, engine="native", chunk_size=11)
        for field in ("asset_pv", "liability_pv", "surplus", "funding_ratio"):
            np.testing.assert_allclose(
                getattr(native, field),
                getattr(reference, field),
                rtol=8.0e-13,
                atol=2.0e-3,
            )
        assert native.engine == "mixed"
