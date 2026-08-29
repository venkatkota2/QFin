"""Batched zero-curve scenarios with chunked native and NumPy execution."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from operator import index as integer_index
from typing import ClassVar, Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from qfin import _native
from qfin.finance.curves import YieldCurve
from qfin.finance.fixed_income import Engine

FloatArray = NDArray[np.float64]
Int64Array = NDArray[np.int64]


def _scenario_path(
    values: ArrayLike | None,
    *,
    scenario_count: int,
    period_count: int,
    default: float,
    name: str,
) -> FloatArray:
    """Normalize one economic factor to a scenario-by-period buffer."""

    if values is None:
        result = np.full((scenario_count, period_count), default, dtype=np.float64)
    else:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 0:
            result = np.full((scenario_count, period_count), float(array), dtype=np.float64)
        elif array.shape == (scenario_count, period_count):
            result = np.array(array, dtype=np.float64, order="C", copy=True)
        elif period_count == 1 and array.shape == (scenario_count,):
            result = np.array(array[:, None], dtype=np.float64, order="C", copy=True)
        elif scenario_count == 1 and array.shape == (period_count,):
            result = np.array(array[None, :], dtype=np.float64, order="C", copy=True)
        else:
            raise ValueError(
                f"{name} must be scalar or have shape ({scenario_count}, {period_count})"
            )
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True, init=False)
class EconomicScenarioSet:
    """Validated multi-period economic and biometric factor paths.

    Rate shocks are additive continuously-compounded zero-rate shocks with
    shape ``scenario x period x curve-node``. The remaining factors use shape
    ``scenario x period``. Multipliers are applied to base mortality and lapse
    assumptions; returns and inflation rates are decimal one-period rates.
    """

    FACTOR_NAMES: ClassVar[tuple[str, ...]] = (
        "rate_parallel",
        "credit_spread",
        "equity_log_return",
        "inflation_log",
        "mortality_log_multiplier",
        "lapse_log_multiplier",
    )

    rate_shocks: FloatArray
    credit_spread_shocks: FloatArray
    equity_returns: FloatArray
    inflation_rates: FloatArray
    mortality_multipliers: FloatArray
    lapse_multipliers: FloatArray
    probabilities: FloatArray
    labels: tuple[str, ...]
    period_length: float
    dependence_assumption: str

    def __init__(
        self,
        rate_shocks: ArrayLike,
        *,
        credit_spread_shocks: ArrayLike | None = None,
        equity_returns: ArrayLike | None = None,
        inflation_rates: ArrayLike | None = None,
        mortality_multipliers: ArrayLike | None = None,
        lapse_multipliers: ArrayLike | None = None,
        probabilities: ArrayLike | None = None,
        labels: tuple[str, ...] | list[str] = (),
        period_length: float = 1.0,
        dependence_assumption: str = "user-supplied economic factor paths",
    ) -> None:
        rates = np.asarray(rate_shocks, dtype=np.float64)
        if rates.ndim == 2:
            rates = rates[:, None, :]
        if rates.ndim != 3 or min(rates.shape) == 0:
            raise ValueError("rate_shocks must have shape scenario-by-period-by-curve-node")
        if not np.all(np.isfinite(rates)):
            raise ValueError("rate shocks must be finite")
        rates = np.array(rates, dtype=np.float64, order="C", copy=True)
        if not isfinite(period_length) or period_length <= 0:
            raise ValueError("period_length must be finite and positive")
        if not dependence_assumption:
            raise ValueError("dependence_assumption must be non-empty")

        scenario_count, period_count, _ = rates.shape
        spreads = _scenario_path(
            credit_spread_shocks,
            scenario_count=scenario_count,
            period_count=period_count,
            default=0.0,
            name="credit_spread_shocks",
        )
        equities = _scenario_path(
            equity_returns,
            scenario_count=scenario_count,
            period_count=period_count,
            default=0.0,
            name="equity_returns",
        )
        inflation = _scenario_path(
            inflation_rates,
            scenario_count=scenario_count,
            period_count=period_count,
            default=0.0,
            name="inflation_rates",
        )
        mortality = _scenario_path(
            mortality_multipliers,
            scenario_count=scenario_count,
            period_count=period_count,
            default=1.0,
            name="mortality_multipliers",
        )
        lapse = _scenario_path(
            lapse_multipliers,
            scenario_count=scenario_count,
            period_count=period_count,
            default=1.0,
            name="lapse_multipliers",
        )
        if np.any(equities <= -1.0):
            raise ValueError("equity returns must be greater than -1")
        if np.any(inflation <= -1.0):
            raise ValueError("inflation rates must be greater than -1")
        if np.any(mortality < 0.0) or np.any(lapse < 0.0):
            raise ValueError("mortality and lapse multipliers must be non-negative")

        if probabilities is None:
            scenario_probabilities = np.full(scenario_count, 1.0 / scenario_count, dtype=np.float64)
        else:
            scenario_probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
            if (
                scenario_probabilities.shape != (scenario_count,)
                or not np.all(np.isfinite(scenario_probabilities))
                or np.any(scenario_probabilities < 0.0)
                or float(np.sum(scenario_probabilities)) <= 0.0
            ):
                raise ValueError(
                    "probabilities must be finite, non-negative, and have one "
                    "positive-total value per scenario"
                )
            scenario_probabilities = np.ascontiguousarray(
                scenario_probabilities / np.sum(scenario_probabilities),
                dtype=np.float64,
            )
        scenario_labels = tuple(labels) or tuple(
            f"scenario_{index}" for index in range(scenario_count)
        )
        if (
            len(scenario_labels) != scenario_count
            or not all(isinstance(label, str) and label for label in scenario_labels)
            or len(set(scenario_labels)) != len(scenario_labels)
        ):
            raise ValueError("labels must be unique and contain one value per scenario")

        buffers = (rates, spreads, equities, inflation, mortality, lapse)
        for buffer in buffers:
            buffer.setflags(write=False)
        scenario_probabilities.setflags(write=False)
        object.__setattr__(self, "rate_shocks", rates)
        object.__setattr__(self, "credit_spread_shocks", spreads)
        object.__setattr__(self, "equity_returns", equities)
        object.__setattr__(self, "inflation_rates", inflation)
        object.__setattr__(self, "mortality_multipliers", mortality)
        object.__setattr__(self, "lapse_multipliers", lapse)
        object.__setattr__(self, "probabilities", scenario_probabilities)
        object.__setattr__(self, "labels", scenario_labels)
        object.__setattr__(self, "period_length", float(period_length))
        object.__setattr__(self, "dependence_assumption", dependence_assumption)

    @property
    def scenario_count(self) -> int:
        return int(self.rate_shocks.shape[0])

    @property
    def period_count(self) -> int:
        return int(self.rate_shocks.shape[1])

    @property
    def curve_node_count(self) -> int:
        return int(self.rate_shocks.shape[2])

    def validate_curve(self, curve: YieldCurve) -> None:
        """Validate that the rate-path nodes align with ``curve``."""

        if self.curve_node_count != curve.times.size:
            raise ValueError("rate shocks must contain one value per curve node")

    def rate_scenarios(self, period: int = 0) -> RateScenarioSet:
        """Return one path period as the existing one-period rate interface."""

        try:
            period_index = integer_index(period)
        except TypeError as exc:
            raise ValueError("period must be an integer") from exc
        if isinstance(period, bool) or not 0 <= period_index < self.period_count:
            raise ValueError("period is outside the scenario horizon")
        return RateScenarioSet(self.rate_shocks[:, period_index, :], self.labels)

    @classmethod
    def correlated_gaussian(
        cls,
        curve: YieldCurve,
        scenario_count: int,
        periods: int,
        *,
        correlation: ArrayLike,
        standard_deviations: ArrayLike,
        means: ArrayLike | None = None,
        seed: int | None = None,
        antithetic: bool = False,
        period_length: float = 1.0,
    ) -> EconomicScenarioSet:
        """Generate transparent correlated Gaussian factor innovations.

        Equity and inflation factors are mapped from Gaussian log changes;
        mortality and lapse factors are mapped to positive lognormal
        multipliers. Period innovations are independent. This is a research
        scenario foundation, not a calibrated economic-scenario model.
        """

        try:
            scenarios = integer_index(scenario_count)
            period_count = integer_index(periods)
        except TypeError as exc:
            raise ValueError("scenario_count and periods must be integers") from exc
        if (
            isinstance(scenario_count, bool)
            or isinstance(periods, bool)
            or scenarios <= 0
            or period_count <= 0
        ):
            raise ValueError("scenario_count and periods must be positive")
        from qfin.finance.factors import GaussianFactorModel

        model = GaussianFactorModel(
            cls.FACTOR_NAMES,
            np.asarray(correlation, dtype=np.float64),
            None if means is None else np.asarray(means, dtype=np.float64),
            np.asarray(standard_deviations, dtype=np.float64),
        )
        generated = model.simulate(
            scenarios * period_count,
            seed=seed,
            antithetic=antithetic,
        ).values.reshape(scenarios, period_count, len(cls.FACTOR_NAMES))
        parallel_rates = np.repeat(generated[:, :, 0, None], curve.times.size, axis=2)
        return cls(
            parallel_rates,
            credit_spread_shocks=generated[:, :, 1],
            equity_returns=np.expm1(generated[:, :, 2]),
            inflation_rates=np.expm1(generated[:, :, 3]),
            mortality_multipliers=np.exp(generated[:, :, 4]),
            lapse_multipliers=np.exp(generated[:, :, 5]),
            period_length=period_length,
            dependence_assumption=(
                "Gaussian same-period correlation; independent period innovations; "
                "log mappings for equity, inflation, mortality, and lapse"
            ),
        )


@dataclass(frozen=True, slots=True)
class RateScenarioSet:
    """Additive zero-rate shocks aligned to a reusable curve's nodes."""

    shocks: FloatArray
    labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        shocks = np.array(self.shocks, dtype=np.float64, order="C", copy=True)
        if shocks.ndim != 2 or shocks.shape[0] == 0 or shocks.shape[1] == 0:
            raise ValueError("shocks must be a non-empty scenario-by-curve-node matrix")
        if not np.all(np.isfinite(shocks)):
            raise ValueError("scenario shocks must be finite")
        labels = self.labels or tuple(f"scenario_{index}" for index in range(shocks.shape[0]))
        if len(labels) != shocks.shape[0] or len(set(labels)) != len(labels):
            raise ValueError("labels must be unique with one label per scenario")
        shocks.setflags(write=False)
        object.__setattr__(self, "shocks", shocks)
        object.__setattr__(self, "labels", tuple(labels))

    @classmethod
    def parallel(cls, curve: YieldCurve, shifts: ArrayLike) -> RateScenarioSet:
        """Construct parallel zero-rate scenarios from decimal shifts."""

        values = np.asarray(shifts, dtype=np.float64).reshape(-1)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("parallel shifts must contain finite values")
        matrix = np.repeat(values[:, None], curve.times.size, axis=1)
        labels = tuple(
            f"parallel_{index}_{shift:+.4f}bp" for index, shift in enumerate(10_000 * values)
        )
        return cls(matrix, labels)

    @classmethod
    def steepener(
        cls,
        curve: YieldCurve,
        *,
        short_shift: float,
        long_shift: float,
        label: str = "steepener",
    ) -> RateScenarioSet:
        """Construct one linear short-to-long zero-rate twist."""

        if not (isfinite(short_shift) and isfinite(long_shift)):
            raise ValueError("steepener shifts must be finite")
        if curve.times.size == 1:
            shock = np.array([long_shift], dtype=np.float64)
        else:
            scale = (curve.times - curve.times[0]) / (curve.times[-1] - curve.times[0])
            shock = short_shift + scale * (long_shift - short_shift)
        return cls(shock[None, :], (label,))

    @classmethod
    def key_rate(
        cls,
        curve: YieldCurve,
        *,
        key_time: float,
        shift: float,
        width: float,
        label: str | None = None,
    ) -> RateScenarioSet:
        """Construct a triangular key-rate shock at the curve nodes."""

        if not all(isfinite(value) for value in (key_time, shift, width)):
            raise ValueError("key-rate inputs must be finite")
        if key_time < 0 or width <= 0:
            raise ValueError("key_time must be non-negative and width positive")
        weights = np.maximum(1.0 - np.abs(curve.times - key_time) / width, 0.0)
        name = label or f"key_{key_time:g}y_{shift * 10_000:+.1f}bp"
        return cls((shift * weights)[None, :], (name,))


def _validate_flat_portfolio(
    times: FloatArray,
    amounts: FloatArray,
    offsets: Int64Array,
    weights: FloatArray,
    curve: YieldCurve,
    scenarios: RateScenarioSet,
) -> None:
    if times.ndim != 1 or amounts.shape != times.shape:
        raise ValueError("cash-flow times and amounts must be one-dimensional and aligned")
    if offsets.ndim != 1 or offsets.size != weights.size + 1:
        raise ValueError("offsets must delimit one cash-flow stream per position weight")
    if offsets.size == 0 or offsets[0] != 0 or offsets[-1] != times.size:
        raise ValueError("invalid cash-flow offsets")
    if np.any(np.diff(offsets) < 0):
        raise ValueError("cash-flow offsets must be non-decreasing")
    if not np.all(np.isfinite(times)) or np.any(times < 0):
        raise ValueError("cash-flow times must be finite and non-negative")
    if not np.all(np.isfinite(amounts)) or not np.all(np.isfinite(weights)):
        raise ValueError("cash-flow amounts and position weights must be finite")
    if scenarios.shocks.shape[1] != curve.times.size:
        raise ValueError("each scenario must have one shock per curve node")


def _numpy_scenario_values(
    times: FloatArray,
    amounts: FloatArray,
    offsets: Int64Array,
    weights: FloatArray,
    curve: YieldCurve,
    shocks: FloatArray,
) -> FloatArray:
    if times.size == 0:
        return np.zeros(shocks.shape[0], dtype=np.float64)
    if curve.times.size == 1:
        interpolated_shocks = np.broadcast_to(shocks[:, :1], (shocks.shape[0], times.size))
    else:
        upper = np.searchsorted(curve.times, times, side="right")
        upper = np.clip(upper, 1, curve.times.size - 1)
        lower = upper - 1
        denominator = curve.times[upper] - curve.times[lower]
        fractions = np.divide(
            times - curve.times[lower],
            denominator,
            out=np.zeros_like(times),
            where=denominator != 0,
        )
        before = times <= curve.times[0]
        after = times >= curve.times[-1]
        lower[before] = 0
        upper[before] = 0
        fractions[before] = 0.0
        lower[after] = curve.times.size - 1
        upper[after] = curve.times.size - 1
        fractions[after] = 0.0
        interpolated_shocks = shocks[:, lower] + fractions * (shocks[:, upper] - shocks[:, lower])
    base_rates = np.asarray(curve.zero_rate(times), dtype=np.float64)
    discounted = amounts * np.exp(-(base_rates + interpolated_shocks) * times)
    counts = np.diff(offsets)
    position_index = np.repeat(np.arange(weights.size), counts)
    cashflow_weights = weights[position_index]
    return np.asarray(
        np.sum(discounted * cashflow_weights, axis=1, dtype=np.float64),
        dtype=np.float64,
    )


def scenario_portfolio_values(
    cashflow_times: FloatArray,
    cashflow_amounts: FloatArray,
    offsets: Int64Array,
    position_weights: FloatArray,
    curve: YieldCurve,
    scenarios: RateScenarioSet,
    *,
    engine: Engine = "auto",
    chunk_size: int = 1_024,
) -> tuple[FloatArray, Literal["numpy", "native"]]:
    """Value flattened streams under rate scenarios without materializing a full cube."""

    times = np.ascontiguousarray(cashflow_times, dtype=np.float64).reshape(-1)
    amounts = np.ascontiguousarray(cashflow_amounts, dtype=np.float64).reshape(-1)
    stream_offsets = np.ascontiguousarray(offsets, dtype=np.int64).reshape(-1)
    weights = np.ascontiguousarray(position_weights, dtype=np.float64).reshape(-1)
    _validate_flat_portfolio(times, amounts, stream_offsets, weights, curve, scenarios)
    try:
        normalized_chunk_size = integer_index(chunk_size)
    except TypeError as exc:
        raise ValueError("chunk_size must be a positive integer") from exc
    if isinstance(chunk_size, bool) or normalized_chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if engine not in ("auto", "numpy", "native"):
        raise ValueError("engine must be 'auto', 'numpy', or 'native'")
    workload = times.size * scenarios.shocks.shape[0]
    selected: Literal["numpy", "native"]
    if engine == "native":
        _native.require()
        selected = "native"
    elif engine == "numpy":
        selected = "numpy"
    else:
        selected = "native" if _native.available() and workload >= 50_000 else "numpy"

    values = np.empty(scenarios.shocks.shape[0], dtype=np.float64)
    for start in range(0, scenarios.shocks.shape[0], normalized_chunk_size):
        stop = min(start + normalized_chunk_size, scenarios.shocks.shape[0])
        shock_chunk = np.ascontiguousarray(scenarios.shocks[start:stop])
        if selected == "native":
            raw = cast(
                object,
                _native.require().scenario_portfolio_present_values(
                    times,
                    amounts,
                    stream_offsets,
                    weights,
                    curve.times,
                    curve.zero_rates,
                    shock_chunk,
                ),
            )
            values[start:stop] = np.asarray(raw, dtype=np.float64)
        else:
            values[start:stop] = _numpy_scenario_values(
                times, amounts, stream_offsets, weights, curve, shock_chunk
            )
    return values, selected


def scenario_indexed_cashflow_values(
    cashflow_times: ArrayLike,
    cashflow_amounts: ArrayLike,
    inflation_linkage: ArrayLike,
    curve: YieldCurve,
    scenarios: EconomicScenarioSet,
    *,
    period: int = 0,
    engine: Engine = "auto",
    chunk_size: int = 1_024,
) -> tuple[FloatArray, Literal["numpy", "native"]]:
    """Value inflation-linked cash flows under one economic-scenario period."""

    times = np.ascontiguousarray(cashflow_times, dtype=np.float64).reshape(-1)
    amounts = np.ascontiguousarray(cashflow_amounts, dtype=np.float64).reshape(-1)
    linkages = np.ascontiguousarray(inflation_linkage, dtype=np.float64).reshape(-1)
    if times.shape != amounts.shape or times.shape != linkages.shape:
        raise ValueError("cash-flow times, amounts, and inflation linkage must align")
    if (
        not np.all(np.isfinite(times))
        or np.any(times < 0.0)
        or not np.all(np.isfinite(amounts))
        or not np.all(np.isfinite(linkages))
        or np.any(linkages < 0.0)
    ):
        raise ValueError("indexed cash-flow inputs must be finite and non-negative in time/linkage")
    scenarios.validate_curve(curve)
    try:
        period_index = integer_index(period)
        normalized_chunk_size = integer_index(chunk_size)
    except TypeError as exc:
        raise ValueError("period and chunk_size must be integers") from exc
    if (
        isinstance(period, bool)
        or not 0 <= period_index < scenarios.period_count
        or isinstance(chunk_size, bool)
        or normalized_chunk_size <= 0
    ):
        raise ValueError("period must be in range and chunk_size must be positive")
    if engine not in ("auto", "numpy", "native"):
        raise ValueError("engine must be 'auto', 'numpy', or 'native'")
    workload = times.size * scenarios.scenario_count
    selected: Literal["numpy", "native"]
    if engine == "native":
        _native.require()
        selected = "native"
    elif engine == "numpy":
        selected = "numpy"
    else:
        selected = "native" if _native.available() and workload >= 50_000 else "numpy"

    values = np.empty(scenarios.scenario_count, dtype=np.float64)
    for start in range(0, scenarios.scenario_count, normalized_chunk_size):
        stop = min(start + normalized_chunk_size, scenarios.scenario_count)
        rate_shocks = np.ascontiguousarray(scenarios.rate_shocks[start:stop, period_index, :])
        inflation = np.ascontiguousarray(scenarios.inflation_rates[start:stop, period_index])
        if selected == "native":
            raw = cast(
                object,
                _native.require().scenario_indexed_cashflow_present_values(
                    times,
                    amounts,
                    linkages,
                    curve.times,
                    curve.zero_rates,
                    rate_shocks,
                    inflation,
                ),
            )
            values[start:stop] = np.asarray(raw, dtype=np.float64)
            continue
        if times.size == 0:
            values[start:stop] = 0.0
            continue
        if curve.times.size == 1:
            interpolated_shocks = np.broadcast_to(rate_shocks[:, :1], (stop - start, times.size))
        else:
            upper = np.searchsorted(curve.times, times, side="right")
            upper = np.clip(upper, 1, curve.times.size - 1)
            lower = upper - 1
            denominator = curve.times[upper] - curve.times[lower]
            fractions = np.divide(
                times - curve.times[lower],
                denominator,
                out=np.zeros_like(times),
                where=denominator != 0,
            )
            before = times <= curve.times[0]
            after = times >= curve.times[-1]
            lower[before] = 0
            upper[before] = 0
            fractions[before] = 0.0
            lower[after] = curve.times.size - 1
            upper[after] = curve.times.size - 1
            fractions[after] = 0.0
            interpolated_shocks = rate_shocks[:, lower] + fractions * (
                rate_shocks[:, upper] - rate_shocks[:, lower]
            )
        rate = np.asarray(curve.zero_rate(times), dtype=np.float64)
        scale = np.power(1.0 + inflation[:, None], times[None, :] * linkages[None, :])
        values[start:stop] = np.sum(
            amounts[None, :]
            * scale
            * np.exp(-(rate[None, :] + interpolated_shocks) * times[None, :]),
            axis=1,
            dtype=np.float64,
        )
    return values, selected


__all__ = [
    "EconomicScenarioSet",
    "RateScenarioSet",
    "scenario_indexed_cashflow_values",
    "scenario_portfolio_values",
]
