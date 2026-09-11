"""Direct binding preflight and malformed buffers: run under both sanitizers."""

import gc

import numpy as np
import pytest

import qfin
from qfin import _native

pytestmark = pytest.mark.skipif(not _native.available(), reason="native extension unavailable")


@pytest.mark.parametrize("offsets", [[], [-1, 2], [0, -1, 2], [0, 3, 2], [0, 1], [1, 2], [0, 2, 3]])
def test_invalid_offsets_are_public_validation_errors(offsets):
    native = _native.require()
    with pytest.raises(qfin.QFinValidationError):
        native.price_cashflow_batches(
            [1.0, 2.0], [10.0, 100.0], np.array(offsets, dtype=np.int64), [0.0, 2.0], [0.0, 0.0]
        )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("position", range(4))
def test_nonfinite_financial_buffers_rejected(bad, position):
    args = [
        np.array([1.0, 2.0]),
        np.array([10.0, 100.0]),
        np.array([0, 2], dtype=np.int64),
        np.array([0.0, 2.0]),
        np.array([0.0, 0.0]),
    ]
    args[[0, 1, 3, 4][position]][0] = bad
    with pytest.raises(qfin.QFinValidationError):
        _native.require().price_cashflow_batches(*args)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
@pytest.mark.parametrize("stride", [1, 2, -1])
def test_strides_dtypes_readonly_and_owned_outputs(dtype, stride):
    native = _native.require()
    times = np.arange(1, 9, dtype=dtype)[::stride]
    amounts = np.arange(1, 9, dtype=dtype)[::stride]
    before = amounts.copy()
    times.setflags(write=False)
    amounts.setflags(write=False)
    out = native.price_cashflow_batches(
        times, amounts, [0, len(times), len(times)], [0, 10], [0, 0]
    )
    np.testing.assert_array_equal(out["prices"], [float(sum(before)), 0])
    del times, amounts
    gc.collect()
    np.testing.assert_array_equal(out["prices"], [float(sum(before)), 0])


def test_huge_logical_shape_rejected_without_materializing():
    # A zero-width ndarray is legal and owns no huge allocation. Its requested
    # scenario output would require terabytes; reject before constructing it.
    shocks = np.empty((2**40, 0))
    with pytest.raises(qfin.ResourceLimitError):
        _native.require().scenario_instrument_present_values(
            [1], [100], [0, 1], [0, 1], [0, 0], shocks
        )


def test_native_life_horizon_is_bounded_before_allocation():
    with pytest.raises(qfin.ResourceLimitError):
        _native.require().project_term_life_policies(
            [40],
            [10000],
            [100],
            np.array([2**31 - 1], dtype=np.int32),
            [0, 120],
            [0.01, 0.01],
            [0.0],
            0.0,
            1.0,
            [0, 60],
            [0.03, 0.03],
        )


def test_overflowing_discount_rejected_as_validation():
    with pytest.raises(qfin.QFinValidationError):
        _native.require().price_cashflow_batches([1e308], [1], [0, 1], [0, 1], [-1, -1])


@pytest.mark.parametrize("frequency", [46341, 2**31 - 1])
def test_large_native_frequency_does_not_overflow_integer_square(frequency):
    result = _native.require().price_cashflow_batches_from_yield(
        [1.0], [100.0], [0, 1], [0.0], np.array([frequency], dtype=np.int32)
    )
    assert result["prices"][0] == 100.0
    # d²P/dy² / P for a zero-coupon bond at y=0 is 1 + 1/f.
    assert result["convexities"][0] == pytest.approx(1 + 1 / frequency, rel=2e-15)


@pytest.mark.parametrize("engine", ["numpy", "native"])
def test_life_output_overflow_fails_explicitly(engine):
    points = qfin.PolicyModelPointSet([qfin.LifePolicy(40, 1e308, 0, 1)], [2])
    assumptions = qfin.ProjectionAssumptions(
        qfin.MortalityTable([0, 120], [1, 1]), qfin.YieldCurve([0, 1], [0, 0])
    )
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(qfin.QFinValidationError, match="finite double"),
    ):
        qfin.project_liabilities(points, assumptions, engine=engine)


def test_direct_term_life_output_overflow_is_public_validation_error():
    with pytest.raises(qfin.QFinValidationError, match="finite double"):
        _native.require().project_term_life_policies(
            [40, 40],
            [1e308, 1e308],
            [0, 0],
            [1, 1],
            [0, 120],
            [1, 1],
            [0],
            0.0,
            1.0,
            [0, 1],
            [0, 0],
        )


@pytest.mark.parametrize("engine", ["numpy", "native"])
def test_alm_output_overflow_is_public_validation_error(engine):
    model = qfin.ALMModel(
        qfin.AssetPortfolio([qfin.FixedRateBond(1, 0, face_value=1e308)], quantities=[2]),
        qfin.LiabilityPortfolio([qfin.CashFlow(1, 100)]),
        qfin.YieldCurve([0, 1], [0, 0]),
    )
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(qfin.QFinValidationError, match="finite double"),
    ):
        model.evaluate(engine=engine)


@pytest.mark.parametrize("engine", ["numpy", "native"])
@pytest.mark.parametrize("tiny_liability", [False, True])
def test_alm_path_result_and_nonzero_funding_denominator_overflow(engine, tiny_liability):
    model = qfin.ALMModel(
        qfin.AssetPortfolio([], cash_value=1e308, equity_value=0 if tiny_liability else 1e308),
        qfin.LiabilityPortfolio([qfin.CashFlow(1, 1e-308 if tiny_liability else 100)]),
        qfin.YieldCurve([0, 1], [0, 0]),
    )
    scenarios = qfin.EconomicScenarioSet(np.zeros((1, 1, 2)))
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(qfin.QFinValidationError, match="finite double"),
    ):
        model.project_paths(scenarios, engine=engine)
