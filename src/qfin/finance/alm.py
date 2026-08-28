"""Asset-liability management for fixed-income assets and life policies."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from qfin.finance.fixed_income import (
    CashFlowSchedule,
    DiscountCurve,
    FixedIncomePortfolio,
)
from qfin.finance.life import LifeCashFlowProjection, LifePolicyPortfolio, MortalityTable

FloatArray = NDArray[np.float64]


def _readonly(values: ArrayLike, *, name: str) -> FloatArray:
    result = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain at least one finite value")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class ALMValuation:
    """Base-curve present values and parallel-shift sensitivities."""

    asset_value: float
    liability_value: float
    surplus: float
    funding_ratio: float
    asset_duration: float
    liability_duration: float
    duration_gap: float
    dollar_duration_gap: float
    asset_convexity: float
    liability_convexity: float

    def to_dict(self) -> dict[str, float]:
        return {
            "asset_value": self.asset_value,
            "liability_value": self.liability_value,
            "surplus": self.surplus,
            "funding_ratio": self.funding_ratio,
            "asset_duration": self.asset_duration,
            "liability_duration": self.liability_duration,
            "duration_gap": self.duration_gap,
            "dollar_duration_gap": self.dollar_duration_gap,
            "asset_convexity": self.asset_convexity,
            "liability_convexity": self.liability_convexity,
        }


@dataclass(frozen=True, slots=True)
class ALMScenarioResult:
    """Vectorized parallel-rate scenario values and derived shortfalls."""

    parallel_shocks: FloatArray
    probabilities: FloatArray
    asset_values: FloatArray
    liability_values: FloatArray
    surplus: FloatArray
    funding_ratios: FloatArray

    def __post_init__(self) -> None:
        funding_ratios = np.asarray(
            self.funding_ratios, dtype=np.float64
        ).reshape(-1).copy()
        if funding_ratios.size == 0 or np.any(np.isnan(funding_ratios)):
            raise ValueError("funding_ratios must not contain NaN")
        funding_ratios.setflags(write=False)
        arrays = (
            _readonly(self.parallel_shocks, name="parallel_shocks"),
            _readonly(self.probabilities, name="probabilities"),
            _readonly(self.asset_values, name="asset_values"),
            _readonly(self.liability_values, name="liability_values"),
            _readonly(self.surplus, name="surplus"),
            funding_ratios,
        )
        if len({array.shape for array in arrays}) != 1:
            raise ValueError("all ALM scenario arrays must have the same shape")
        if np.any(arrays[1] < 0) or not np.isclose(np.sum(arrays[1]), 1.0):
            raise ValueError("scenario probabilities must be non-negative and sum to one")
        for name, array in zip(
            (
                "parallel_shocks",
                "probabilities",
                "asset_values",
                "liability_values",
                "surplus",
                "funding_ratios",
            ),
            arrays,
            strict=True,
        ):
            object.__setattr__(self, name, array)

    @property
    def shortfalls(self) -> FloatArray:
        values = np.maximum(-self.surplus, 0.0)
        values.setflags(write=False)
        return values

    @property
    def shortfall_probability(self) -> float:
        return float(np.dot(self.probabilities, self.surplus < 0))

    @property
    def expected_shortfall(self) -> float:
        return float(np.dot(self.probabilities, self.shortfalls))

    @property
    def expected_surplus(self) -> float:
        return float(np.dot(self.probabilities, self.surplus))

    def to_dict(self) -> dict[str, object]:
        return {
            "parallel_shocks": self.parallel_shocks.tolist(),
            "probabilities": self.probabilities.tolist(),
            "asset_values": self.asset_values.tolist(),
            "liability_values": self.liability_values.tolist(),
            "surplus": self.surplus.tolist(),
            "funding_ratios": self.funding_ratios.tolist(),
            "shortfall_probability": self.shortfall_probability,
            "expected_shortfall": self.expected_shortfall,
            "expected_surplus": self.expected_surplus,
        }


@dataclass(frozen=True, slots=True)
class AssetLiabilityModel:
    """Fixed-income/life ALM model with pre-aggregated valuation arrays.

    The model projects deterministic expected policy cash flows from a supplied
    mortality basis.  Parallel rate scenarios are valued in bounded-memory
    vectorized chunks; no Python loop is performed per scenario.
    """

    assets: FixedIncomePortfolio
    liabilities: LifePolicyPortfolio
    discount_curve: DiscountCurve
    mortality: MortalityTable
    _asset_cashflows: CashFlowSchedule = field(init=False, repr=False)
    _liability_projection: LifeCashFlowProjection = field(init=False, repr=False)
    _liability_cashflows: CashFlowSchedule = field(init=False, repr=False)
    _scenario_times: FloatArray = field(init=False, repr=False)
    _discounted_cashflow_columns: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        asset_cashflows = self.assets.cashflows
        liability_projection = self.liabilities.expected_cashflows(self.mortality)
        liability_cashflows = liability_projection.net_schedule
        times = np.union1d(asset_cashflows.times, liability_cashflows.times)
        columns = np.zeros((times.size, 2), dtype=np.float64)
        columns[np.searchsorted(times, asset_cashflows.times), 0] = asset_cashflows.amounts
        columns[np.searchsorted(times, liability_cashflows.times), 1] = (
            liability_cashflows.amounts
        )
        columns *= self.discount_curve.discount(times)[:, None]
        times.setflags(write=False)
        columns.setflags(write=False)
        object.__setattr__(self, "_asset_cashflows", asset_cashflows)
        object.__setattr__(self, "_liability_projection", liability_projection)
        object.__setattr__(self, "_liability_cashflows", liability_cashflows)
        object.__setattr__(self, "_scenario_times", times)
        object.__setattr__(self, "_discounted_cashflow_columns", columns)

    @property
    def asset_cashflows(self) -> CashFlowSchedule:
        return self._asset_cashflows

    @property
    def liability_projection(self) -> LifeCashFlowProjection:
        return self._liability_projection

    @property
    def liability_cashflows(self) -> CashFlowSchedule:
        return self._liability_cashflows

    def evaluate(self, curve: DiscountCurve | None = None) -> ALMValuation:
        valuation_curve = self.discount_curve if curve is None else curve
        asset_value = self._asset_cashflows.present_value(valuation_curve)
        liability_value = self._liability_cashflows.present_value(valuation_curve)
        asset_duration = self._asset_cashflows.parallel_duration(valuation_curve)
        liability_duration = self._liability_cashflows.parallel_duration(valuation_curve)
        asset_convexity = self._asset_cashflows.parallel_convexity(valuation_curve)
        liability_convexity = self._liability_cashflows.parallel_convexity(
            valuation_curve
        )
        funding_ratio = (
            asset_value / liability_value
            if abs(liability_value) > 1e-15
            else float("inf")
        )
        return ALMValuation(
            asset_value=asset_value,
            liability_value=liability_value,
            surplus=asset_value - liability_value,
            funding_ratio=funding_ratio,
            asset_duration=asset_duration,
            liability_duration=liability_duration,
            duration_gap=asset_duration - liability_duration,
            dollar_duration_gap=(
                asset_value * asset_duration - liability_value * liability_duration
            ),
            asset_convexity=asset_convexity,
            liability_convexity=liability_convexity,
        )

    def run_parallel_shocks(
        self,
        parallel_shocks: ArrayLike,
        *,
        probabilities: ArrayLike | None = None,
        max_working_bytes: int = 64 * 1024 * 1024,
    ) -> ALMScenarioResult:
        """Value many parallel curve shocks with bounded temporary memory."""
        shocks = _readonly(parallel_shocks, name="parallel_shocks")
        if probabilities is None:
            weights = np.full(shocks.size, 1.0 / shocks.size, dtype=np.float64)
        else:
            weights = _readonly(probabilities, name="probabilities").copy()
            if weights.shape != shocks.shape:
                raise ValueError("probabilities must match parallel_shocks")
            if np.any(weights < 0):
                raise ValueError("probabilities must be non-negative")
            total = float(np.sum(weights))
            if total <= 0:
                raise ValueError("probabilities must have positive total mass")
            weights /= total
        if isinstance(max_working_bytes, bool) or max_working_bytes < 8:
            raise ValueError("max_working_bytes must be at least 8")

        bytes_per_scenario = max(8 * self._scenario_times.size, 8)
        chunk_size = max(1, max_working_bytes // bytes_per_scenario)
        values = np.empty((shocks.size, 2), dtype=np.float64)
        for start in range(0, shocks.size, chunk_size):
            stop = min(start + chunk_size, shocks.size)
            discount_shifts = np.exp(
                -shocks[start:stop, None] * self._scenario_times[None, :]
            )
            values[start:stop] = discount_shifts @ self._discounted_cashflow_columns

        asset_values = values[:, 0]
        liability_values = values[:, 1]
        surplus = asset_values - liability_values
        funding_ratios = np.full(shocks.size, np.inf, dtype=np.float64)
        np.divide(
            asset_values,
            liability_values,
            out=funding_ratios,
            where=np.abs(liability_values) > 1e-15,
        )
        return ALMScenarioResult(
            parallel_shocks=shocks,
            probabilities=weights,
            asset_values=asset_values,
            liability_values=liability_values,
            surplus=surplus,
            funding_ratios=funding_ratios,
        )
