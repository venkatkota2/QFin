import numpy as np
import pytest

from qfin.validation import FinancialTolerance, validate_financial_values


def test_financial_materiality_does_not_relax_numerical_parity() -> None:
    result = validate_financial_values(
        "bond", [100.001], [100.0],
        tolerance=FinancialTolerance.for_quantity("price_per_100"),
    )
    assert not result.passed
    assert result.checks[0].financial_passed
    assert not result.checks[0].numerical_passed


def test_large_notional_requires_financial_and_numerical_parity() -> None:
    result = validate_financial_values(
        "portfolio", [1e12 + 1.0], [1e12],
        tolerance=FinancialTolerance(relative=1e-10, financial=0.01, unit="currency"),
    )
    assert not result.passed
    assert result.checks[0].numerical_passed
    assert not result.checks[0].financial_passed
    assert "financial gate=False" in result.checks[0].diagnostic
    with pytest.raises(AssertionError, match="portfolio validation failed"):
        result.assert_valid()


@pytest.mark.parametrize("quantity", [
    "price_per_100", "pv", "dv01", "duration", "convexity", "risk_currency", "probability",
])
def test_financial_profiles_report_units_and_accept_exact_parity(quantity: str) -> None:
    tolerance = FinancialTolerance.for_quantity(quantity, notional=1e6)
    report = validate_financial_values("exact", [1, 2], [1, 2], tolerance=tolerance)
    assert report.passed
    assert report.maximum_absolute_difference == 0.0
    assert report.explain()["check_count"] == 2
    assert all(item.unit == tolerance.unit for item in report.checks)


@pytest.mark.parametrize("notional", [0.0, -1.0, np.nan, np.inf])
def test_financial_profile_rejects_invalid_scale(notional: float) -> None:
    with pytest.raises(ValueError, match="notional"):
        FinancialTolerance.for_quantity("pv", notional=notional)


def test_validation_rejects_unknown_units_and_nonfinite_references() -> None:
    with pytest.raises(ValueError, match="unknown financial quantity"):
        FinancialTolerance.for_quantity("arbitrary")
    with pytest.raises(ValueError, match="finite"):
        FinancialTolerance().allowed_error(np.nan)
    with pytest.raises(ValueError, match="labels"):
        validate_financial_values("labels", [1.0], [1.0], labels=[])
