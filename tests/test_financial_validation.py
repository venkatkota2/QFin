import numpy as np
import pytest

import qfin


def test_golden_bond_portfolio_matches_independent_scalar_reference() -> None:
    actual: list[float] = []
    expected: list[float] = []
    labels: list[str] = []
    for case in qfin.GOLDEN_BOND_CASES:
        bond = qfin.FixedRateBond(
            case.maturity,
            case.coupon_rate,
            frequency=case.frequency,
        )
        result = qfin.price_bonds_from_yield(bond, case.yield_rate, engine="numpy")
        reference = qfin.reference_bond_from_yield(bond, case.yield_rate)
        actual.extend(
            [
                result.dirty_prices[0],
                result.ytm_macaulay_duration[0],
                result.ytm_modified_duration[0],
                result.ytm_convexity[0],
                reference.dirty_price,
            ]
        )
        expected.extend(
            [
                case.dirty_price,
                case.macaulay_duration,
                case.modified_duration,
                case.convexity,
                case.dirty_price,
            ]
        )
        labels.extend(f"{case.name}:{name}" for name in ("price", "mac", "mod", "conv", "ref"))
    report = qfin.validate_financial_values(
        "golden bond portfolio",
        actual,
        expected,
        labels=labels,
        tolerance=qfin.FinancialTolerance(
            absolute=1.0e-11,
            relative=1.0e-12,
            financial=1.0e-10,
            unit="price/duration units",
        ),
    )
    assert report.passed
    report.assert_valid()
    assert report.explain()["failed_count"] == 0


def test_financial_tolerance_failure_has_business_unit_diagnostic() -> None:
    report = qfin.validate_financial_values(
        "portfolio PV",
        [1_000_000.51],
        [1_000_000.00],
        labels=["present value"],
        tolerance=qfin.FinancialTolerance(
            absolute=1.0e-12,
            relative=0.0,
            financial=0.50,
            unit="USD",
        ),
    )
    assert not report.passed
    assert report.failed_checks[0].difference == pytest.approx(0.51)
    assert "USD" in report.failed_checks[0].diagnostic
    with pytest.raises(qfin.FinancialValidationError, match="present value"):
        report.assert_valid()


def test_validation_rejects_shape_and_nonfinite_inputs() -> None:
    with pytest.raises(ValueError, match="equal shapes"):
        qfin.validate_financial_values("bad", [1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="finite"):
        qfin.validate_financial_values("bad", [np.nan], [1.0])


def test_optional_quantlib_validation_matches_regular_dated_bond() -> None:
    if not qfin.quantlib_available():
        pytest.skip("optional QuantLib validation dependency is not installed")
    bond = qfin.FixedRateBond.from_dates(
        "2024-01-15",
        "2029-01-15",
        0.04,
        frequency=2,
        business_day_convention="unadjusted",
    )
    assert qfin.quantlib_bond_schedule(bond) == bond.schedule.dates
    qfin_result = qfin.price_bonds_from_yield(
        bond, 0.0475, settlement="2024-04-15", engine="numpy"
    )
    independent = qfin.quantlib_bond_from_yield(
        bond, 0.0475, settlement="2024-04-15"
    )
    np.testing.assert_allclose(
        [
            qfin_result.dirty_prices[0],
            qfin_result.clean_prices[0],
            qfin_result.accrued_interest[0],
            qfin_result.ytm_macaulay_duration[0],
            qfin_result.ytm_modified_duration[0],
            qfin_result.ytm_convexity[0],
            qfin_result.dv01[0],
        ],
        [
            independent.dirty_price,
            independent.clean_price,
            independent.accrued_interest,
            independent.macaulay_duration,
            independent.modified_duration,
            independent.convexity,
            independent.dv01,
        ],
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    solved = qfin.yield_from_prices(
        bond,
        independent.dirty_price,
        settlement="2024-04-15",
        engine="numpy",
    )
    assert solved.converged[0]
    assert solved.yields[0] == pytest.approx(0.0475, abs=1.0e-11)
