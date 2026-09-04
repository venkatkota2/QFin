"""Multi-period ALM roll-forward with bounded scenario-chunk memory."""

from __future__ import annotations

from operator import index as integer_index
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from qfin import _native
from qfin.finance.alm import ALMModel, ALMPathResult, RebalancingStrategy
from qfin.finance.curves import YieldCurve
from qfin.finance.fixed_income import Engine, flatten_bond_cashflows
from qfin.finance.scenarios import EconomicScenarioSet

FloatArray = NDArray[np.float64]


def _interpolate_paths(query: FloatArray, nodes: FloatArray, path_values: FloatArray) -> FloatArray:
    """Flat-tail linear interpolation for a row of values per scenario."""

    if query.size == 0:
        return np.empty((path_values.shape[0], 0), dtype=np.float64)
    if nodes.size == 1:
        return np.broadcast_to(path_values[:, :1], (path_values.shape[0], query.size))
    upper = np.searchsorted(nodes, query, side="right")
    upper = np.clip(upper, 1, nodes.size - 1)
    lower = upper - 1
    denominator = nodes[upper] - nodes[lower]
    fractions = np.divide(
        query - nodes[lower],
        denominator,
        out=np.zeros_like(query),
        where=denominator != 0.0,
    )
    before = query <= nodes[0]
    after = query >= nodes[-1]
    lower[before] = 0
    upper[before] = 0
    fractions[before] = 0.0
    lower[after] = nodes.size - 1
    upper[after] = nodes.size - 1
    fractions[after] = 0.0
    return path_values[:, lower] + fractions * (path_values[:, upper] - path_values[:, lower])


def _scenario_present_value(
    cashflow_times: FloatArray,
    cashflow_amounts: FloatArray,
    valuation_time: float,
    curve: YieldCurve,
    rate_shocks: FloatArray,
    spread_shocks: FloatArray,
    *,
    amount_scales: FloatArray | None = None,
) -> FloatArray:
    remaining = cashflow_times > valuation_time + 1.0e-12
    if not np.any(remaining):
        return np.zeros(rate_shocks.shape[0], dtype=np.float64)
    tenors = np.ascontiguousarray(cashflow_times[remaining] - valuation_time)
    amounts = cashflow_amounts[remaining]
    base_rates = np.asarray(curve.zero_rate(tenors), dtype=np.float64)
    shocks = _interpolate_paths(tenors, curve.times, rate_shocks)
    scaled_amounts: FloatArray
    if amount_scales is None:
        scaled_amounts = np.broadcast_to(amounts, shocks.shape)
    else:
        scaled_amounts = amount_scales[:, remaining] * amounts[None, :]
    return cast(
        FloatArray,
        np.sum(
            scaled_amounts
            * np.exp(
                -(base_rates[None, :] + shocks + spread_shocks[:, None])
                * tenors[None, :]
            ),
            axis=1,
            dtype=np.float64,
        ),
    )


def _weighted_asset_cashflows(model: ALMModel) -> tuple[FloatArray, FloatArray]:
    times, amounts, offsets = flatten_bond_cashflows(
        model.assets.bonds, settlement=model.assets.settlement
    )
    counts = np.diff(offsets)
    weights = np.repeat(model.assets.quantities, counts)
    return times, np.ascontiguousarray(amounts * weights, dtype=np.float64)


def _numpy_path_chunk(
    model: ALMModel,
    scenarios: EconomicScenarioSet,
    strategy: RebalancingStrategy,
    start: int,
    stop: int,
) -> dict[str, FloatArray]:
    scenario_count = stop - start
    period_count = scenarios.period_count
    output_width = period_count + 1
    dt = scenarios.period_length
    curve_times = model.curve.times
    rate_paths = scenarios.rate_shocks[start:stop]
    spread_paths = scenarios.credit_spread_shocks[start:stop]
    equity_returns = scenarios.equity_returns[start:stop]
    inflation_rates = scenarios.inflation_rates[start:stop]

    asset_times, asset_amounts = _weighted_asset_cashflows(model)
    liability_times, liability_amounts = model.liabilities.buffers()
    liability_linkage = model.liabilities.inflation_linkage
    zero_shocks = np.zeros((scenario_count, curve_times.size), dtype=np.float64)
    zero_spreads = np.zeros(scenario_count, dtype=np.float64)
    initial_bond = float(
        _scenario_present_value(
            asset_times,
            asset_amounts,
            0.0,
            model.curve,
            zero_shocks[:1],
            zero_spreads[:1],
        )[0]
    )
    initial_liability = float(
        _scenario_present_value(
            liability_times,
            liability_amounts,
            0.0,
            model.curve,
            zero_shocks[:1],
            zero_spreads[:1],
        )[0]
    )

    assets = np.empty((scenario_count, output_width), dtype=np.float64)
    bonds = np.empty_like(assets)
    cash = np.empty_like(assets)
    equities = np.empty_like(assets)
    liabilities = np.empty_like(assets)
    surplus = np.empty_like(assets)
    funding = np.empty_like(assets)
    payments = np.zeros((scenario_count, period_count), dtype=np.float64)
    costs = np.zeros_like(payments)
    bond_balance = np.full(scenario_count, initial_bond, dtype=np.float64)
    cash_balance = np.full(scenario_count, model.assets.cash_value, dtype=np.float64)
    equity_balance = np.full(scenario_count, model.assets.equity_value, dtype=np.float64)
    inflation_index = np.ones(scenario_count, dtype=np.float64)

    bonds[:, 0] = bond_balance
    cash[:, 0] = cash_balance
    equities[:, 0] = equity_balance
    assets[:, 0] = bond_balance + cash_balance + equity_balance
    liabilities[:, 0] = initial_liability
    surplus[:, 0] = assets[:, 0] - liabilities[:, 0]
    funding[:, 0] = np.divide(
        assets[:, 0],
        liabilities[:, 0],
        out=np.full(scenario_count, np.inf),
        where=liabilities[:, 0] != 0.0,
    )

    base_short_rate = model.curve.zero_rate(dt)
    for period in range(period_count):
        period_start = period * dt
        period_end = (period + 1) * dt
        start_rate_shocks = zero_shocks if period == 0 else rate_paths[:, period - 1, :]
        end_rate_shocks = rate_paths[:, period, :]
        start_spreads = zero_spreads if period == 0 else spread_paths[:, period - 1]
        end_spreads = spread_paths[:, period]
        start_index = _scenario_present_value(
            asset_times,
            asset_amounts,
            period_start,
            model.curve,
            start_rate_shocks,
            start_spreads,
        )
        end_index = _scenario_present_value(
            asset_times,
            asset_amounts,
            period_end,
            model.curve,
            end_rate_shocks,
            end_spreads,
        )
        tenor = np.array([dt], dtype=np.float64)
        average_short_shock = 0.5 * (
            _interpolate_paths(tenor, curve_times, start_rate_shocks)[:, 0]
            + _interpolate_paths(tenor, curve_times, end_rate_shocks)[:, 0]
        )
        cash_growth = np.exp((base_short_rate + average_short_shock) * dt)
        received = (asset_times > period_start + 1.0e-12) & (asset_times <= period_end + 1.0e-12)
        if np.any(received):
            received_wealth = np.sum(
                asset_amounts[received][None, :]
                * np.exp(
                    (base_short_rate + average_short_shock[:, None])
                    * (period_end - asset_times[received])[None, :]
                ),
                axis=1,
                dtype=np.float64,
            )
        else:
            received_wealth = np.zeros(scenario_count, dtype=np.float64)
        bond_return = np.divide(
            end_index + received_wealth,
            start_index,
            out=cash_growth.copy(),
            where=np.abs(start_index) > 1.0e-15,
        )
        bond_balance *= bond_return
        cash_balance *= cash_growth
        equity_balance *= 1.0 + equity_returns[:, period]
        inflation_index *= np.power(1.0 + inflation_rates[:, period], dt)

        due = (liability_times > period_start + 1.0e-12) & (liability_times <= period_end + 1.0e-12)
        if np.any(due):
            payments[:, period] = np.sum(
                liability_amounts[due][None, :]
                * np.power(inflation_index[:, None], liability_linkage[due][None, :]),
                axis=1,
                dtype=np.float64,
            )
        if strategy.pay_liabilities:
            total_before_payment = bond_balance + cash_balance + equity_balance
            total_after_payment = total_before_payment - payments[:, period]
            scale = np.divide(
                total_after_payment,
                total_before_payment,
                out=np.ones_like(total_after_payment),
                where=np.abs(total_before_payment) > 1.0e-15,
            )
            bond_balance *= scale
            cash_balance *= scale
            equity_balance *= scale

        should_rebalance = (
            strategy.target_equity_weight is not None
            and (period + 1) % strategy.rebalance_frequency == 0
        )
        if should_rebalance:
            assert strategy.target_equity_weight is not None
            total = bond_balance + cash_balance + equity_balance
            desired_before_cost = strategy.target_equity_weight * np.maximum(total, 0.0)
            costs[:, period] = (
                np.abs(desired_before_cost - equity_balance) * strategy.transaction_cost_rate
            )
            after_cost = total - costs[:, period]
            positive = after_cost > 0.0
            non_equity_before = bond_balance + cash_balance
            cash_share = np.divide(
                cash_balance,
                non_equity_before,
                out=np.zeros_like(cash_balance),
                where=np.abs(non_equity_before) > 1.0e-15,
            )
            cash_share = np.clip(cash_share, 0.0, 1.0)
            rebalanced_non_equity = (1.0 - strategy.target_equity_weight) * after_cost
            equity_balance = np.where(
                positive, strategy.target_equity_weight * after_cost, equity_balance
            )
            cash_balance = np.where(positive, cash_share * rebalanced_non_equity, cash_balance)
            bond_balance = np.where(positive, rebalanced_non_equity - cash_balance, bond_balance)

        amount_scales = np.power(inflation_index[:, None], liability_linkage[None, :])
        liability_value = _scenario_present_value(
            liability_times,
            liability_amounts,
            period_end,
            model.curve,
            end_rate_shocks,
            zero_spreads,
            amount_scales=amount_scales,
        )
        bonds[:, period + 1] = bond_balance
        cash[:, period + 1] = cash_balance
        equities[:, period + 1] = equity_balance
        assets[:, period + 1] = bond_balance + cash_balance + equity_balance
        liabilities[:, period + 1] = liability_value
        surplus[:, period + 1] = assets[:, period + 1] - liability_value
        funding[:, period + 1] = np.divide(
            assets[:, period + 1],
            liability_value,
            out=np.full(scenario_count, np.inf),
            where=liability_value != 0.0,
        )
    return {
        "asset_values": assets,
        "bond_values": bonds,
        "cash_values": cash,
        "equity_values": equities,
        "liability_values": liabilities,
        "liability_payments": payments,
        "surplus": surplus,
        "funding_ratio": funding,
        "transaction_costs": costs,
    }


def project_alm_paths(
    model: ALMModel,
    scenarios: EconomicScenarioSet,
    *,
    strategy: RebalancingStrategy | None = None,
    engine: Engine = "auto",
    scenario_chunk_size: int = 512,
) -> ALMPathResult:
    """Project stochastic rates and asset/liability roll-forwards in chunks."""

    scenarios.validate_curve(model.curve)
    selected_strategy = strategy or RebalancingStrategy()
    try:
        chunk_size = integer_index(scenario_chunk_size)
    except TypeError as exc:
        raise ValueError("scenario_chunk_size must be a positive integer") from exc
    if isinstance(scenario_chunk_size, bool) or chunk_size <= 0:
        raise ValueError("scenario_chunk_size must be a positive integer")
    if engine not in ("auto", "numpy", "native"):
        raise ValueError("engine must be 'auto', 'numpy', or 'native'")
    asset_times, asset_amounts = _weighted_asset_cashflows(model)
    liability_times, liability_amounts = model.liabilities.buffers()
    selected: Literal["numpy", "native"]
    if engine == "native":
        if not model.curve.native_compatible:
            raise ValueError(
                "native engine requires linear-zero interpolation with flat-zero extrapolation"
            )
        _native.require()
        selected = "native"
    else:
        # The 0.6 benchmark found no stable native crossover through 10,000
        # scenarios. Keep auto deterministic and performance-led while retaining
        # the native implementation as an explicit parity-tested profiling path.
        selected = "numpy"

    shape = (scenarios.scenario_count, scenarios.period_count + 1)
    period_shape = (scenarios.scenario_count, scenarios.period_count)
    output = {
        "asset_values": np.empty(shape, dtype=np.float64),
        "bond_values": np.empty(shape, dtype=np.float64),
        "cash_values": np.empty(shape, dtype=np.float64),
        "equity_values": np.empty(shape, dtype=np.float64),
        "liability_values": np.empty(shape, dtype=np.float64),
        "liability_payments": np.empty(period_shape, dtype=np.float64),
        "surplus": np.empty(shape, dtype=np.float64),
        "funding_ratio": np.empty(shape, dtype=np.float64),
        "transaction_costs": np.empty(period_shape, dtype=np.float64),
    }
    for start in range(0, scenarios.scenario_count, chunk_size):
        stop = min(start + chunk_size, scenarios.scenario_count)
        if selected == "numpy":
            chunk = _numpy_path_chunk(model, scenarios, selected_strategy, start, stop)
        else:
            raw = cast(
                dict[str, object],
                _native.require().project_alm_paths(
                    asset_times,
                    asset_amounts,
                    liability_times,
                    liability_amounts,
                    model.liabilities.inflation_linkage,
                    model.curve.times,
                    model.curve.zero_rates,
                    np.ascontiguousarray(scenarios.rate_shocks[start:stop]),
                    np.ascontiguousarray(scenarios.credit_spread_shocks[start:stop]),
                    np.ascontiguousarray(scenarios.equity_returns[start:stop]),
                    np.ascontiguousarray(scenarios.inflation_rates[start:stop]),
                    scenarios.period_length,
                    model.assets.equity_value,
                    model.assets.cash_value,
                    -1.0
                    if selected_strategy.target_equity_weight is None
                    else selected_strategy.target_equity_weight,
                    selected_strategy.rebalance_frequency,
                    selected_strategy.transaction_cost_rate,
                    selected_strategy.pay_liabilities,
                ),
            )
            chunk_width = scenarios.period_count + 1
            chunk_periods = scenarios.period_count
            chunk = {
                "asset_values": np.asarray(raw["asset_values"], dtype=np.float64).reshape(
                    stop - start, chunk_width
                ),
                "bond_values": np.asarray(raw["bond_values"], dtype=np.float64).reshape(
                    stop - start, chunk_width
                ),
                "cash_values": np.asarray(raw["cash_values"], dtype=np.float64).reshape(
                    stop - start, chunk_width
                ),
                "equity_values": np.asarray(raw["equity_values"], dtype=np.float64).reshape(
                    stop - start, chunk_width
                ),
                "liability_values": np.asarray(raw["liability_values"], dtype=np.float64).reshape(
                    stop - start, chunk_width
                ),
                "liability_payments": np.asarray(
                    raw["liability_payments"], dtype=np.float64
                ).reshape(stop - start, chunk_periods),
                "surplus": np.asarray(raw["surplus"], dtype=np.float64).reshape(
                    stop - start, chunk_width
                ),
                "funding_ratio": np.asarray(raw["funding_ratio"], dtype=np.float64).reshape(
                    stop - start, chunk_width
                ),
                "transaction_costs": np.asarray(raw["transaction_costs"], dtype=np.float64).reshape(
                    stop - start, chunk_periods
                ),
            }
        for name, values in chunk.items():
            output[name][start:stop] = values

    times = np.arange(scenarios.period_count + 1, dtype=np.float64) * scenarios.period_length
    initial_surplus = float(output["surplus"][0, 0])
    return ALMPathResult(
        times=times,
        labels=scenarios.labels,
        probabilities=scenarios.probabilities,
        asset_values=output["asset_values"],
        bond_values=output["bond_values"],
        cash_values=output["cash_values"],
        equity_values=output["equity_values"],
        liability_values=output["liability_values"],
        liability_payments=output["liability_payments"],
        surplus=output["surplus"],
        funding_ratio=output["funding_ratio"],
        transaction_costs=output["transaction_costs"],
        initial_surplus=initial_surplus,
        engine=selected,
    )


__all__ = ["project_alm_paths"]
