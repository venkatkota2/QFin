"""Asset-liability portfolios, valuation, and rate-scenario analysis."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from operator import index as integer_index
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
from qfin.finance.scenarios import (
    EconomicScenarioSet,
    RateScenarioSet,
    scenario_indexed_cashflow_values,
    scenario_portfolio_values,
)

if TYPE_CHECKING:
    from qfin.finance.risk import LossDistribution

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class AssetPortfolio:
    """Fixed-rate bonds plus aggregate equity and cash allocations."""

    bonds: tuple[FixedRateBond, ...]
    quantities: FloatArray
    settlement: float = 0.0
    equity_value: float = 0.0
    cash_value: float = 0.0

    def __init__(
        self,
        bonds: list[FixedRateBond] | tuple[FixedRateBond, ...],
        quantities: ArrayLike | None = None,
        *,
        settlement: float = 0.0,
        equity_value: float = 0.0,
        cash_value: float = 0.0,
    ) -> None:
        items = tuple(bonds)
        if any(not isinstance(item, FixedRateBond) for item in items):
            raise TypeError("bonds must contain FixedRateBond objects")
        if quantities is None:
            weights = np.ones(len(items), dtype=np.float64)
        else:
            weights = np.array(quantities, dtype=np.float64, order="C", copy=True).reshape(-1)
        if weights.shape != (len(items),) or not np.all(np.isfinite(weights)):
            raise ValueError("quantities must be finite with one value per bond")
        if not np.isfinite(settlement) or settlement < 0:
            raise ValueError("settlement must be finite and non-negative")
        if not isfinite(equity_value) or equity_value < 0:
            raise ValueError("equity_value must be finite and non-negative")
        if not isfinite(cash_value) or cash_value < 0:
            raise ValueError("cash_value must be finite and non-negative")
        weights.setflags(write=False)
        object.__setattr__(self, "bonds", items)
        object.__setattr__(self, "quantities", weights)
        object.__setattr__(self, "settlement", float(settlement))
        object.__setattr__(self, "equity_value", float(equity_value))
        object.__setattr__(self, "cash_value", float(cash_value))


@dataclass(frozen=True, slots=True)
class LiabilityPortfolio:
    """Liability outflows with optional cash-flow-specific inflation linkage."""

    cashflows: tuple[CashFlow, ...]
    inflation_linkage: FloatArray

    def __init__(
        self,
        cashflows: list[CashFlow] | tuple[CashFlow, ...],
        inflation_linkage: ArrayLike | None = None,
    ) -> None:
        items = tuple(cashflows)
        if any(not isinstance(item, CashFlow) for item in items):
            raise TypeError("cashflows must contain CashFlow objects")
        if inflation_linkage is None:
            linkage = np.zeros(len(items), dtype=np.float64)
        else:
            linkage = np.array(inflation_linkage, dtype=np.float64, order="C", copy=True).reshape(
                -1
            )
        if (
            linkage.shape != (len(items),)
            or not np.all(np.isfinite(linkage))
            or np.any(linkage < 0.0)
        ):
            raise ValueError(
                "inflation_linkage must contain one finite non-negative value per cash flow"
            )
        linkage.setflags(write=False)
        object.__setattr__(self, "cashflows", items)
        object.__setattr__(self, "inflation_linkage", linkage)

    @classmethod
    def from_arrays(
        cls,
        times: ArrayLike,
        amounts: ArrayLike,
        *,
        inflation_linkage: ArrayLike | None = None,
    ) -> LiabilityPortfolio:
        time_array = np.asarray(times, dtype=np.float64).reshape(-1)
        amount_array = np.asarray(amounts, dtype=np.float64).reshape(-1)
        if time_array.shape != amount_array.shape:
            raise ValueError("times and amounts must have the same shape")
        return cls(
            [
                CashFlow(time=float(time), amount=float(amount))
                for time, amount in zip(time_array, amount_array, strict=True)
            ],
            inflation_linkage,
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


@dataclass(frozen=True, slots=True)
class ALMFactorAttribution:
    """Isolated factor impacts plus the exact residual interaction."""

    factor_names: tuple[str, ...]
    impacts: FloatArray
    interaction: FloatArray
    total_change: FloatArray

    def weighted_mean(self, probabilities: ArrayLike) -> dict[str, float]:
        """Return probability-weighted mean attribution by factor."""

        weights = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        if (
            weights.shape != (self.impacts.shape[0],)
            or not np.all(np.isfinite(weights))
            or np.any(weights < 0.0)
            or float(np.sum(weights)) <= 0.0
        ):
            raise ValueError("probabilities must be non-negative and align to scenarios")
        weights = weights / np.sum(weights)
        result = {
            name: float(weights @ self.impacts[:, index])
            for index, name in enumerate(self.factor_names)
        }
        result["interaction"] = float(weights @ self.interaction)
        result["total"] = float(weights @ self.total_change)
        return result


@dataclass(frozen=True, slots=True)
class ALMFactorScenarioResult:
    """One-period ALM values under rates, spread, equity, and inflation factors."""

    labels: tuple[str, ...]
    probabilities: FloatArray
    asset_pv: FloatArray
    liability_pv: FloatArray
    surplus: FloatArray
    funding_ratio: FloatArray
    base_surplus: float
    attribution: ALMFactorAttribution
    engine: Literal["numpy", "native", "mixed"]

    def loss_distribution(self) -> LossDistribution:
        """Return scenario surplus deterioration with scenario probabilities."""

        from qfin.finance.risk import LossDistribution

        return LossDistribution(self.base_surplus - self.surplus, self.probabilities)


@dataclass(frozen=True, slots=True)
class RebalancingStrategy:
    """Asset roll-forward, liability-payment, and allocation policy."""

    target_equity_weight: float | None = None
    rebalance_frequency: int = 1
    transaction_cost_rate: float = 0.0
    pay_liabilities: bool = True

    def __post_init__(self) -> None:
        if self.target_equity_weight is not None and (
            not isfinite(self.target_equity_weight) or not 0.0 <= self.target_equity_weight <= 1.0
        ):
            raise ValueError("target_equity_weight must lie in [0, 1]")
        try:
            frequency = integer_index(self.rebalance_frequency)
        except TypeError as exc:
            raise ValueError("rebalance_frequency must be a positive integer") from exc
        if isinstance(self.rebalance_frequency, bool) or frequency <= 0:
            raise ValueError("rebalance_frequency must be a positive integer")
        if not isfinite(self.transaction_cost_rate) or not (
            0.0 <= self.transaction_cost_rate < 1.0
        ):
            raise ValueError("transaction_cost_rate must lie in [0, 1)")
        object.__setattr__(self, "rebalance_frequency", frequency)


@dataclass(frozen=True, slots=True)
class ALMPathResult:
    """Memory-bounded multi-period ALM path projection outputs."""

    times: FloatArray
    labels: tuple[str, ...]
    probabilities: FloatArray
    asset_values: FloatArray
    bond_values: FloatArray
    cash_values: FloatArray
    equity_values: FloatArray
    liability_values: FloatArray
    liability_payments: FloatArray
    surplus: FloatArray
    funding_ratio: FloatArray
    transaction_costs: FloatArray
    initial_surplus: float
    engine: Literal["numpy", "native"]

    def loss_distribution(self, period: int = -1) -> LossDistribution:
        """Map deterioration at one horizon into the quantum-risk input layer."""

        from qfin.finance.risk import LossDistribution

        try:
            index = integer_index(period)
        except TypeError as exc:
            raise ValueError("period must be an integer") from exc
        if isinstance(period, bool):
            raise ValueError("period must be an integer")
        if index < 0:
            index += self.surplus.shape[1]
        if not 0 <= index < self.surplus.shape[1]:
            raise ValueError("period is outside the projected horizon")
        return LossDistribution(self.initial_surplus - self.surplus[:, index], self.probabilities)


@dataclass(frozen=True, slots=True)
class ALMSensitivityReport:
    """Forward bump-and-revalue impacts on base surplus."""

    base_surplus: float
    rate_impact: float
    credit_spread_impact: float
    equity_impact: float
    inflation_impact: float
    rate_bump: float
    credit_spread_bump: float
    equity_bump: float
    inflation_bump: float
    interaction: float
    engine: Literal["numpy", "native", "mixed"]

    def to_dict(self) -> dict[str, float | str]:
        return {
            "base_surplus": self.base_surplus,
            "rate_impact": self.rate_impact,
            "credit_spread_impact": self.credit_spread_impact,
            "equity_impact": self.equity_impact,
            "inflation_impact": self.inflation_impact,
            "rate_bump": self.rate_bump,
            "credit_spread_bump": self.credit_spread_bump,
            "equity_bump": self.equity_bump,
            "inflation_bump": self.inflation_bump,
            "interaction": self.interaction,
            "engine": self.engine,
        }


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


def _factor_alm_values(
    model: ALMModel,
    scenarios: EconomicScenarioSet,
    *,
    include_rates: bool,
    include_spreads: bool,
    include_equity: bool,
    include_inflation: bool,
    engine: Engine,
    chunk_size: int,
) -> tuple[
    FloatArray,
    FloatArray,
    Literal["numpy", "native"],
    Literal["numpy", "native"],
]:
    """Evaluate one-period factor subsets for exact residual attribution."""

    scenarios.validate_curve(model.curve)
    asset_times, asset_amounts, asset_offsets = flatten_bond_cashflows(
        model.assets.bonds, settlement=model.assets.settlement
    )
    rate_shocks = (
        scenarios.rate_shocks[:, 0, :]
        if include_rates
        else np.zeros_like(scenarios.rate_shocks[:, 0, :])
    )
    if include_spreads:
        rate_shocks = rate_shocks + scenarios.credit_spread_shocks[:, 0, None]
    asset_values, asset_engine = scenario_portfolio_values(
        asset_times,
        asset_amounts,
        asset_offsets,
        model.assets.quantities,
        model.curve,
        RateScenarioSet(rate_shocks, scenarios.labels),
        engine=engine,
        chunk_size=chunk_size,
    )
    equity_returns = scenarios.equity_returns[:, 0] if include_equity else 0.0
    asset_values = (
        asset_values + model.assets.equity_value * (1.0 + equity_returns) + model.assets.cash_value
    )
    liability_times, liability_amounts = model.liabilities.buffers()
    liability_scenarios = EconomicScenarioSet(
        scenarios.rate_shocks[:, 0:1, :]
        if include_rates
        else np.zeros_like(scenarios.rate_shocks[:, 0:1, :]),
        inflation_rates=(
            scenarios.inflation_rates[:, 0:1]
            if include_inflation
            else np.zeros_like(scenarios.inflation_rates[:, 0:1])
        ),
        probabilities=scenarios.probabilities,
        labels=scenarios.labels,
        period_length=scenarios.period_length,
        dependence_assumption=scenarios.dependence_assumption,
    )
    liability_values, liability_engine = scenario_indexed_cashflow_values(
        liability_times,
        liability_amounts,
        model.liabilities.inflation_linkage,
        model.curve,
        liability_scenarios,
        engine=engine,
        chunk_size=chunk_size,
    )
    return asset_values, liability_values, asset_engine, liability_engine


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
        bond_pv = float(np.sum(market_values))
        asset_pv = bond_pv + self.assets.equity_value + self.assets.cash_value
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
            asset_analytics.engine if asset_analytics.engine == liability_engine else "mixed"
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
        asset_values = asset_values + self.assets.equity_value + self.assets.cash_value
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

    def run_factor_scenarios(
        self,
        scenarios: EconomicScenarioSet,
        *,
        engine: Engine = "auto",
        chunk_size: int = 1_024,
    ) -> ALMFactorScenarioResult:
        """Run one-period rate, spread, equity, and inflation revaluation.

        Mortality and lapse paths belong to the policy projection engine. They
        remain on the shared scenario object but do not alter deterministic
        liability cash flows in this method.
        """

        base = self.evaluate(engine=engine)
        full_asset, full_liability, asset_engine, liability_engine = _factor_alm_values(
            self,
            scenarios,
            include_rates=True,
            include_spreads=True,
            include_equity=True,
            include_inflation=True,
            engine=engine,
            chunk_size=chunk_size,
        )
        full_surplus = full_asset - full_liability
        factor_flags = (
            ("rates", True, False, False, False),
            ("credit_spread", False, True, False, False),
            ("equity", False, False, True, False),
            ("inflation", False, False, False, True),
        )
        impact_columns: list[FloatArray] = []
        for _, rates, spreads, equity, inflation in factor_flags:
            isolated_asset, isolated_liability, _, _ = _factor_alm_values(
                self,
                scenarios,
                include_rates=rates,
                include_spreads=spreads,
                include_equity=equity,
                include_inflation=inflation,
                engine=engine,
                chunk_size=chunk_size,
            )
            impact_columns.append(isolated_asset - isolated_liability - base.surplus)
        impacts = np.ascontiguousarray(np.column_stack(impact_columns), dtype=np.float64)
        total_change = np.ascontiguousarray(full_surplus - base.surplus, dtype=np.float64)
        interaction = np.ascontiguousarray(total_change - np.sum(impacts, axis=1), dtype=np.float64)
        funding = np.divide(
            full_asset,
            full_liability,
            out=np.full_like(full_asset, np.inf),
            where=full_liability != 0.0,
        )
        combined_engine: Literal["numpy", "native", "mixed"] = (
            asset_engine if asset_engine == liability_engine else "mixed"
        )
        return ALMFactorScenarioResult(
            labels=scenarios.labels,
            probabilities=scenarios.probabilities,
            asset_pv=full_asset,
            liability_pv=full_liability,
            surplus=full_surplus,
            funding_ratio=funding,
            base_surplus=base.surplus,
            attribution=ALMFactorAttribution(
                factor_names=tuple(item[0] for item in factor_flags),
                impacts=impacts,
                interaction=interaction,
                total_change=total_change,
            ),
            engine=combined_engine,
        )

    def project_paths(
        self,
        scenarios: EconomicScenarioSet,
        *,
        strategy: RebalancingStrategy | None = None,
        engine: Engine = "auto",
        scenario_chunk_size: int = 512,
    ) -> ALMPathResult:
        """Roll assets and liabilities through multi-period economic paths."""

        from qfin.finance.alm_paths import project_alm_paths

        return project_alm_paths(
            self,
            scenarios,
            strategy=strategy,
            engine=engine,
            scenario_chunk_size=scenario_chunk_size,
        )

    def sensitivities(
        self,
        *,
        rate_bump: float = 0.0001,
        credit_spread_bump: float = 0.0001,
        equity_bump: float = 0.01,
        inflation_bump: float = 0.01,
        engine: Engine = "auto",
    ) -> ALMSensitivityReport:
        """Return isolated rate, spread, equity, and inflation bump impacts."""

        bumps = (rate_bump, credit_spread_bump, equity_bump, inflation_bump)
        if any(not isfinite(value) or value <= 0.0 for value in bumps):
            raise ValueError("sensitivity bumps must be finite and positive")
        scenarios = EconomicScenarioSet(
            np.full((1, 1, self.curve.times.size), rate_bump, dtype=np.float64),
            credit_spread_shocks=credit_spread_bump,
            equity_returns=equity_bump,
            inflation_rates=inflation_bump,
            labels=("sensitivity_bumps",),
        )
        result = self.run_factor_scenarios(scenarios, engine=engine)
        impacts = result.attribution.impacts[0]
        return ALMSensitivityReport(
            base_surplus=result.base_surplus,
            rate_impact=float(impacts[0]),
            credit_spread_impact=float(impacts[1]),
            equity_impact=float(impacts[2]),
            inflation_impact=float(impacts[3]),
            rate_bump=rate_bump,
            credit_spread_bump=credit_spread_bump,
            equity_bump=equity_bump,
            inflation_bump=inflation_bump,
            interaction=float(result.attribution.interaction[0]),
            engine=result.engine,
        )


__all__ = [
    "ALMFactorAttribution",
    "ALMFactorScenarioResult",
    "ALMModel",
    "ALMPathResult",
    "ALMResult",
    "ALMScenarioResult",
    "ALMSensitivityReport",
    "AssetPortfolio",
    "LiabilityPortfolio",
    "RebalancingStrategy",
]
