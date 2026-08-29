"""Asset-liability portfolios, valuation, and rate-scenario analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from qfin import _native
from qfin.finance.curves import YieldCurve
from qfin.finance.fixed_income import (
    CashFlow,
    Engine,
    FixedRateBond,
    flatten_bond_cashflows,
    price_bonds,
)
from qfin.finance.scenarios import RateScenarioSet, scenario_portfolio_values

if TYPE_CHECKING:
    from qfin.finance.risk import LossDistribution

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class AssetPortfolio:
    """Batch of fixed-rate bonds and position quantities."""

    bonds: tuple[FixedRateBond, ...]
    quantities: FloatArray
    settlement: float = 0.0

    def __init__(
        self,
        bonds: list[FixedRateBond] | tuple[FixedRateBond, ...],
        quantities: ArrayLike | None = None,
        *,
        settlement: float = 0.0,
    ) -> None:
        items = tuple(bonds)
        if any(not isinstance(item, FixedRateBond) for item in items):
            raise TypeError("bonds must contain FixedRateBond objects")
        if quantities is None:
            weights = np.ones(len(items), dtype=np.float64)
        else:
            weights = np.ascontiguousarray(quantities, dtype=np.float64).reshape(-1)
        if weights.shape != (len(items),) or not np.all(np.isfinite(weights)):
            raise ValueError("quantities must be finite with one value per bond")
        if not np.isfinite(settlement) or settlement < 0:
            raise ValueError("settlement must be finite and non-negative")
        weights.setflags(write=False)
        object.__setattr__(self, "bonds", items)
        object.__setattr__(self, "quantities", weights)
        object.__setattr__(self, "settlement", float(settlement))


@dataclass(frozen=True, slots=True)
class LiabilityPortfolio:
    """Deterministic liability outflows, positive from the insurer's perspective."""

    cashflows: tuple[CashFlow, ...]

    def __init__(self, cashflows: list[CashFlow] | tuple[CashFlow, ...]) -> None:
        items = tuple(cashflows)
        if any(not isinstance(item, CashFlow) for item in items):
            raise TypeError("cashflows must contain CashFlow objects")
        object.__setattr__(self, "cashflows", items)

    @classmethod
    def from_arrays(cls, times: ArrayLike, amounts: ArrayLike) -> LiabilityPortfolio:
        time_array = np.asarray(times, dtype=np.float64).reshape(-1)
        amount_array = np.asarray(amounts, dtype=np.float64).reshape(-1)
        if time_array.shape != amount_array.shape:
            raise ValueError("times and amounts must have the same shape")
        return cls(
            [
                CashFlow(time=float(time), amount=float(amount))
                for time, amount in zip(time_array, amount_array, strict=True)
            ]
        )

    def buffers(self) -> tuple[FloatArray, FloatArray]:
        return (
            np.ascontiguousarray([item.time for item in self.cashflows], dtype=np.float64),
            np.ascontiguousarray([item.amount for item in self.cashflows], dtype=np.float64),
        )


@dataclass(frozen=True, slots=True)
class ALMResult:
    """Base-curve ALM valuation and immunization measures."""

    asset_pv: float
    liability_pv: float
    surplus: float
    deficit: float
    funding_ratio: float
    asset_duration: float
    liability_duration: float
    duration_gap: float
    asset_convexity: float
    liability_convexity: float
    convexity_gap: float
    engine: Literal["numpy", "native", "mixed"]


@dataclass(frozen=True, slots=True)
class ALMScenarioResult:
    """Portfolio-level values under a scenario set."""

    labels: tuple[str, ...]
    asset_pv: FloatArray
    liability_pv: FloatArray
    surplus: FloatArray
    funding_ratio: FloatArray
    base_surplus: float
    engine: Literal["numpy", "native", "mixed"]

    def loss_distribution(self, probabilities: ArrayLike | None = None) -> LossDistribution:
        """Map scenario surplus deterioration into QFin's loss-distribution layer."""

        from qfin.finance.risk import LossDistribution

        probability_array = (
            None
            if probabilities is None
            else np.ascontiguousarray(probabilities, dtype=np.float64).reshape(-1)
        )
        return LossDistribution(self.base_surplus - self.surplus, probability_array)


def _liability_metrics(
    portfolio: LiabilityPortfolio,
    curve: YieldCurve,
    engine: Engine,
) -> tuple[float, float, float, Literal["numpy", "native"]]:
    times, amounts = portfolio.buffers()
    if times.size == 0:
        return 0.0, 0.0, 0.0, "numpy"
    if engine not in ("auto", "numpy", "native"):
        raise ValueError("engine must be 'auto', 'numpy', or 'native'")
    selected: Literal["numpy", "native"]
    if engine == "native":
        _native.require()
        selected = "native"
    elif engine == "numpy":
        selected = "numpy"
    else:
        selected = "native" if _native.available() and times.size >= 4_096 else "numpy"
    if selected == "native":
        raw = cast(
            dict[str, object],
            _native.require().price_cashflow_batches(
                times,
                amounts,
                np.array([0, times.size], dtype=np.int64),
                curve.times,
                curve.zero_rates,
                0.0,
            ),
        )
        return (
            float(np.asarray(raw["prices"])[0]),
            float(np.asarray(raw["macaulay_durations"])[0]),
            float(np.asarray(raw["convexities"])[0]),
            selected,
        )
    present_values = amounts * np.asarray(curve.discount(times), dtype=np.float64)
    value = float(np.sum(present_values))
    if abs(value) <= 1.0e-15:
        return value, 0.0, 0.0, selected
    duration = float(np.dot(times, present_values) / value)
    convexity = float(np.dot(times * times, present_values) / value)
    return value, duration, convexity, selected


@dataclass(frozen=True, slots=True)
class ALMModel:
    """Fixed-income assets against deterministic liability cash flows."""

    assets: AssetPortfolio
    liabilities: LiabilityPortfolio
    curve: YieldCurve

    def evaluate(self, *, engine: Engine = "auto") -> ALMResult:
        asset_analytics = price_bonds(
            self.assets.bonds,
            self.curve,
            settlement=self.assets.settlement,
            engine=engine,
        )
        market_values = self.assets.quantities * asset_analytics.dirty_prices
        asset_pv = float(np.sum(market_values))
        asset_duration = (
            0.0
            if abs(asset_pv) <= 1.0e-15
            else float(np.dot(market_values, asset_analytics.macaulay_duration) / asset_pv)
        )
        asset_convexity = (
            0.0
            if abs(asset_pv) <= 1.0e-15
            else float(np.dot(market_values, asset_analytics.convexity) / asset_pv)
        )
        liability_pv, liability_duration, liability_convexity, liability_engine = (
            _liability_metrics(self.liabilities, self.curve, engine)
        )
        surplus = asset_pv - liability_pv
        ratio = liability_pv / asset_pv if abs(asset_pv) > 1.0e-15 else 0.0
        funding = asset_pv / liability_pv if abs(liability_pv) > 1.0e-15 else float("inf")
        combined_engine: Literal["numpy", "native", "mixed"] = (
            asset_analytics.engine
            if asset_analytics.engine == liability_engine
            else "mixed"
        )
        return ALMResult(
            asset_pv=asset_pv,
            liability_pv=liability_pv,
            surplus=surplus,
            deficit=max(-surplus, 0.0),
            funding_ratio=funding,
            asset_duration=asset_duration,
            liability_duration=liability_duration,
            duration_gap=asset_duration - ratio * liability_duration,
            asset_convexity=asset_convexity,
            liability_convexity=liability_convexity,
            convexity_gap=asset_convexity - ratio * liability_convexity,
            engine=combined_engine,
        )

    def run_scenarios(
        self,
        scenarios: RateScenarioSet,
        *,
        engine: Engine = "auto",
        chunk_size: int = 1_024,
    ) -> ALMScenarioResult:
        """Revalue both sides under node-aligned rate shocks in bounded memory."""

        base = self.evaluate(engine=engine)
        asset_times, asset_amounts, asset_offsets = flatten_bond_cashflows(
            self.assets.bonds, settlement=self.assets.settlement
        )
        asset_values, asset_engine = scenario_portfolio_values(
            asset_times,
            asset_amounts,
            asset_offsets,
            self.assets.quantities,
            self.curve,
            scenarios,
            engine=engine,
            chunk_size=chunk_size,
        )
        liability_times, liability_amounts = self.liabilities.buffers()
        liability_values, liability_engine = scenario_portfolio_values(
            liability_times,
            liability_amounts,
            np.array([0, liability_times.size], dtype=np.int64),
            np.array([1.0], dtype=np.float64),
            self.curve,
            scenarios,
            engine=engine,
            chunk_size=chunk_size,
        )
        surplus = asset_values - liability_values
        funding = np.divide(
            asset_values,
            liability_values,
            out=np.full_like(asset_values, np.inf),
            where=liability_values != 0,
        )
        combined_engine: Literal["numpy", "native", "mixed"] = (
            asset_engine if asset_engine == liability_engine else "mixed"
        )
        return ALMScenarioResult(
            scenarios.labels,
            asset_values,
            liability_values,
            surplus,
            funding,
            base.surplus,
            combined_engine,
        )


__all__ = [
    "ALMModel",
    "ALMResult",
    "ALMScenarioResult",
    "AssetPortfolio",
    "LiabilityPortfolio",
]
