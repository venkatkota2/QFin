from math import exp

import pytest

import qfin


@pytest.mark.parametrize(
    ("compounding", "expected"),
    [
        ("continuous", exp(-0.05 * 2)),
        ("annual", 1 / 1.05**2),
        ("semiannual", 1 / 1.025**4),
        ("quarterly", 1 / 1.0125**8),
        ("monthly", 1 / (1 + 0.05 / 12) ** 24),
        ("simple", 1 / 1.10),
    ],
)
def test_supported_compounding_discount_factors(
    compounding: str, expected: float
) -> None:
    assert qfin.discount_factor(0.05, 2.0, compounding) == pytest.approx(expected)


def test_rate_conversion_preserves_discount_factor_and_metadata() -> None:
    annual = qfin.RateQuote(0.05, "annual")
    continuous = annual.equivalent("continuous")
    assert continuous.compounding is qfin.Compounding.CONTINUOUS
    assert continuous.discount(7.0) == pytest.approx(annual.discount(7.0))
    assert continuous.equivalent("annual").rate == pytest.approx(annual.rate)


def test_simple_rate_conversion_depends_on_stated_horizon() -> None:
    converted = qfin.convert_rate(0.08, "simple", "continuous", time=2.0)
    assert exp(-converted * 2.0) == pytest.approx(1 / 1.16)


def test_invalid_rate_domains_fail_clearly() -> None:
    with pytest.raises(ValueError, match="negative frequency"):
        qfin.discount_factor(-2.0, 1.0, "annual")
    with pytest.raises(ValueError, match="simple rate"):
        qfin.discount_factor(-0.6, 2.0, "simple")
