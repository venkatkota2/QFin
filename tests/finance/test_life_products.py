import numpy as np
import pytest

import qfin


def _assumptions(**kwargs: object) -> qfin.ProjectionAssumptions:
    mortality = qfin.MortalityTable([0, 120], [0.01, 0.01])
    curve = qfin.YieldCurve([0, 30], [0.02, 0.02])
    return qfin.ProjectionAssumptions(mortality, curve, **kwargs)  # type: ignore[arg-type]


def test_exact_model_point_grouping_preserves_exposure() -> None:
    first = qfin.LifePolicy(40, 100_000, 500, 5)
    second = qfin.LifePolicy(50, 50_000, 300, 4)
    grouped = qfin.PolicyModelPointSet.from_policies([first, first, second], [1, 2, 3])
    assert grouped.model_point_count == 2
    np.testing.assert_allclose(grouped.counts, [3, 3])
    assert grouped.policy_count == 6
    assert grouped.compression_ratio == 3


def test_product_foundations_have_analytical_zero_decrement_cashflows() -> None:
    assumptions = qfin.ProjectionAssumptions(
        qfin.MortalityTable([0, 120], [0, 0]),
        qfin.YieldCurve([0, 30], [0, 0]),
        lapse_rate=0,
        crediting_rate=0,
    )
    annuity = qfin.LifePolicy(65, 0, 0, 3, product_type="annuity", annual_benefit=10_000)
    participating = qfin.LifePolicy(
        40,
        100_000,
        0,
        2,
        product_type="participating_life",
        bonus_rate=0.10,
    )
    universal = qfin.LifePolicy(
        40,
        0,
        0,
        2,
        product_type="universal_life",
        account_value=10_000,
    )
    result = qfin.project_liabilities(
        [annuity, participating, universal], assumptions, engine="numpy"
    )
    assert result.product_present_values["annuity"] == pytest.approx(30_000)
    assert result.product_present_values["participating_life"] == pytest.approx(121_000)
    assert result.product_present_values["universal_life"] == pytest.approx(10_000)


def test_multi_state_and_mixed_product_native_parity() -> None:
    assumptions = _assumptions(
        lapse_rate=0.03,
        expense_per_policy=10,
        disability_rate=0.02,
        recovery_rate=0.10,
        disabled_mortality_multiplier=2.0,
        crediting_rate=0.03,
        expense_inflation_rate=0.02,
    )
    points = qfin.PolicyModelPointSet(
        [
            qfin.LifePolicy(40, 100_000, 500, 5, disability_benefit=1_000),
            qfin.LifePolicy(
                45,
                80_000,
                700,
                5,
                product_type="participating_life",
                bonus_rate=0.01,
            ),
            qfin.LifePolicy(
                50,
                50_000,
                1_000,
                5,
                product_type="universal_life",
                account_value=10_000,
                annual_charge=100,
                crediting_spread=0.005,
            ),
            qfin.LifePolicy(
                65,
                0,
                0,
                5,
                product_type="annuity",
                annual_benefit=10_000,
                benefit_inflation_linkage=1,
            ),
        ],
        [10, 20, 30, 40],
    )
    reference = qfin.project_liabilities(points, assumptions, engine="numpy")
    native = qfin.project_liabilities(points, assumptions, engine="native")
    for name in (
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
        np.testing.assert_allclose(getattr(native, name), getattr(reference, name), rtol=1e-13)
    assert np.any(native.disabled > 0)
    assert native.present_value == pytest.approx(reference.present_value, rel=1e-13)


def test_empty_model_point_book_is_supported_by_both_engines() -> None:
    assumptions = _assumptions()
    reference = qfin.project_liabilities([], assumptions, engine="numpy")
    native = qfin.project_liabilities([], assumptions, engine="native")
    assert reference.present_value == native.present_value == 0
    assert reference.policy_present_values.size == native.policy_present_values.size == 0
    assert all(value == 0 for value in native.product_present_values.values())


def test_non_empty_life_auto_dispatch_uses_measured_native_path() -> None:
    result = qfin.project_liabilities([qfin.LifePolicy(40, 100_000, 500, 1)], _assumptions())
    assert result.engine == "native"


def test_life_model_point_and_assumption_validation() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        qfin.LifePolicy(40, 100_000, 500, 5, product_type="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="counts"):
        qfin.PolicyModelPointSet([qfin.LifePolicy(40, 100_000, 500, 5)], [0])
    with pytest.raises(ValueError, match="probabilities"):
        _assumptions(disability_rate=1.01)
    with pytest.raises(ValueError, match="cover every projection year"):
        qfin.project_liabilities(
            [qfin.LifePolicy(40, 100_000, 500, 5)],
            _assumptions(lapse_rate=[0.01, 0.02]),
        )
