"""Batched zero-curve scenarios with chunked native and NumPy execution."""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isfinite
from typing import ClassVar, Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import PchipInterpolator

from qfin import _native
from qfin._dispatch import POLICIES, resolve_engine
from qfin._memory import check_allocation
from qfin._validation import require_integer
from qfin.exceptions import QFinValidationError
from qfin.finance.curves import CurveExtrapolation, CurveInterpolation, YieldCurve
from qfin.finance.fixed_income import Engine

FloatArray = NDArray[np.float64]
Int64Array = NDArray[np.int64]

# Bound each NumPy scenario matrix to 16 MiB (several can coexist during
# interpolation). The public chunk_size is an upper bound, not a reservation.
_MAX_SCENARIO_MATRIX_ELEMENTS = 2_097_152


def _bounded_numpy_chunk_size(requested: int, times: FloatArray, curve: YieldCurve) -> int:
    width = max(1, times.size, curve.times.size)
    return min(requested, max(1, _MAX_SCENARIO_MATRIX_ELEMENTS // width))


def _scenario_path(
    values: ArrayLike | None,
    *,
    scenario_count: int,
    period_count: int,
    default: float,
    name: str,
) -> FloatArray:
    """Normalize one economic factor to a scenario-by-period buffer."""

    check_allocation((scenario_count, period_count))
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
            raise QFinValidationError(
                f"{name} must be scalar or have shape ({scenario_count}, {period_count})"
            )
    if not np.all(np.isfinite(result)):
        raise QFinValidationError(f"{name} must be finite")
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
            raise QFinValidationError(
                "rate_shocks must have shape scenario-by-period-by-curve-node"
            )
        if not np.all(np.isfinite(rates)):
            raise QFinValidationError("rate shocks must be finite")
        rates = np.array(rates, dtype=np.float64, order="C", copy=True)
        if not isfinite(period_length) or period_length <= 0:
            raise QFinValidationError("period_length must be finite and positive")
        if not dependence_assumption:
            raise QFinValidationError("dependence_assumption must be non-empty")

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
            raise QFinValidationError("equity returns must be greater than -1")
        if np.any(inflation <= -1.0):
            raise QFinValidationError("inflation rates must be greater than -1")
        if np.any(mortality < 0.0) or np.any(lapse < 0.0):
            raise QFinValidationError("mortality and lapse multipliers must be non-negative")

        if probabilities is None:
            scenario_probabilities = np.full(scenario_count, 1.0 / scenario_count, dtype=np.float64)
        else:
            scenario_probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
            if (
                scenario_probabilities.shape != (scenario_count,)
                or not np.all(np.isfinite(scenario_probabilities))
                or np.any(scenario_probabilities < 0.0)
                or float(np.max(scenario_probabilities, initial=0.0)) <= 0.0
            ):
                raise QFinValidationError(
                    "probabilities must be finite, non-negative, and have one "
                    "positive-total value per scenario"
                )
            scaled_probabilities = scenario_probabilities / np.max(scenario_probabilities)
            scenario_probabilities = np.ascontiguousarray(
                scaled_probabilities / np.sum(scaled_probabilities),
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
            raise QFinValidationError("labels must be unique and contain one value per scenario")

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
            raise QFinValidationError("rate shocks must contain one value per curve node")

    def rate_scenarios(self, period: int = 0) -> RateScenarioSet:
        """Return one path period as the existing one-period rate interface."""

        period_index = require_integer(period, "period")
        if not 0 <= period_index < self.period_count:
            raise QFinValidationError("period is outside the scenario horizon")
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

        scenarios = require_integer(scenario_count, "scenario_count", minimum=1)
        period_count = require_integer(periods, "periods", minimum=1)
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
            raise QFinValidationError("shocks must be a non-empty scenario-by-curve-node matrix")
        if not np.all(np.isfinite(shocks)):
            raise QFinValidationError("scenario shocks must be finite")
        labels = self.labels or tuple(f"scenario_{index}" for index in range(shocks.shape[0]))
        if len(labels) != shocks.shape[0] or len(set(labels)) != len(labels):
            raise QFinValidationError("labels must be unique with one label per scenario")
        shocks.setflags(write=False)
        object.__setattr__(self, "shocks", shocks)
        object.__setattr__(self, "labels", tuple(labels))

    @classmethod
    def parallel(cls, curve: YieldCurve, shifts: ArrayLike) -> RateScenarioSet:
        """Construct parallel zero-rate scenarios from decimal shifts."""

        values = np.asarray(shifts, dtype=np.float64).reshape(-1)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise QFinValidationError("parallel shifts must contain finite values")
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
            raise QFinValidationError("steepener shifts must be finite")
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
            raise QFinValidationError("key-rate inputs must be finite")
        if key_time < 0 or width <= 0:
            raise QFinValidationError("key_time must be non-negative and width positive")
        weights = np.maximum(1.0 - np.abs(curve.times - key_time) / width, 0.0)
        name = label or f"key_{key_time:g}y_{shift * 10_000:+.1f}bp"
        return cls((shift * weights)[None, :], (name,))


@dataclass(frozen=True, slots=True)
class _PreparedScenarioValuation:
    """Scenario-independent interpolation state for a fixed cash-flow grid."""

    times: FloatArray
    curve: YieldCurve
    lower: Int64Array
    upper: Int64Array
    fractions: FloatArray
    before: NDArray[np.bool_]
    after: NDArray[np.bool_]

    @classmethod
    def build(cls, times: FloatArray, curve: YieldCurve) -> _PreparedScenarioValuation:
        query = np.ascontiguousarray(times, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(query)) or np.any(query < 0.0):
            raise QFinValidationError("cash-flow times must be finite and non-negative")
        before = query < curve.times[0]
        after = query > curve.times[-1]
        if curve.extrapolation is CurveExtrapolation.ERROR and (np.any(before) or np.any(after)):
            raise QFinValidationError("cash-flow time is outside the curve domain")
        if curve.times.size == 1:
            lower = np.zeros(query.size, dtype=np.int64)
            upper = np.zeros(query.size, dtype=np.int64)
            fractions = np.zeros(query.size, dtype=np.float64)
        else:
            upper = np.searchsorted(curve.times, query, side="right")
            upper = np.clip(upper, 1, curve.times.size - 1).astype(np.int64, copy=False)
            lower = upper - 1
            denominator = curve.times[upper] - curve.times[lower]
            fractions = np.divide(
                query - curve.times[lower],
                denominator,
                out=np.zeros_like(query),
                where=denominator != 0.0,
            )
        return cls(query, curve, lower, upper, fractions, before, after)

    def discount_factors(self, shocks: FloatArray) -> FloatArray:
        """Reproduce ``curve.shifted(shock).discount(times)`` for a shock batch."""

        if shocks.ndim != 2 or shocks.shape[1] != self.curve.times.size:
            raise QFinValidationError("scenario shocks must contain one value per curve node")
        node_rates = self.curve.zero_rates[None, :] + shocks
        node_log_discounts = -node_rates * self.curve.times[None, :]
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            node_discounts = np.exp(node_log_discounts)
        if not np.all(np.isfinite(node_discounts)) or np.any(node_discounts <= 0.0):
            raise QFinValidationError(
                "scenario shocks imply non-finite or non-positive node discount factors"
            )

        if self.curve.interpolation is CurveInterpolation.LINEAR_ZERO:
            interpolated_rates = node_rates[:, self.lower] + self.fractions * (
                node_rates[:, self.upper] - node_rates[:, self.lower]
            )
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                result = np.exp(-interpolated_rates * self.times[None, :])
        elif self.curve.interpolation is CurveInterpolation.MONOTONE_ZERO:
            if self.curve.times.size == 1:
                interpolated_rates = np.broadcast_to(
                    node_rates[:, :1], (shocks.shape[0], self.times.size)
                )
            else:
                clipped = np.clip(self.times, self.curve.times[0], self.curve.times[-1])
                interpolator = PchipInterpolator(
                    self.curve.times,
                    node_rates,
                    axis=1,
                    extrapolate=False,
                )
                interpolated_rates = np.asarray(interpolator(clipped), dtype=np.float64)
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                result = np.exp(-interpolated_rates * self.times[None, :])
        elif self.curve.interpolation is CurveInterpolation.LINEAR_DISCOUNT:
            result = node_discounts[:, self.lower] + self.fractions * (
                node_discounts[:, self.upper] - node_discounts[:, self.lower]
            )
        else:
            interpolated_logs = node_log_discounts[:, self.lower] + self.fractions * (
                node_log_discounts[:, self.upper] - node_log_discounts[:, self.lower]
            )
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                result = np.exp(interpolated_logs)

        if self.curve.extrapolation is CurveExtrapolation.FLAT_ZERO:
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                left = np.exp(-node_rates[:, :1] * self.times[None, :])
                right = np.exp(-node_rates[:, -1:] * self.times[None, :])
            result = np.where(self.before[None, :], left, result)
            result = np.where(self.after[None, :], right, result)
        elif self.curve.extrapolation is CurveExtrapolation.FLAT_FORWARD:
            if self.curve.times.size == 1:
                left_log = right_log = -node_rates[:, :1] * self.times[None, :]
            else:
                left_slope = (node_log_discounts[:, 1] - node_log_discounts[:, 0]) / (
                    self.curve.times[1] - self.curve.times[0]
                )
                right_slope = (node_log_discounts[:, -1] - node_log_discounts[:, -2]) / (
                    self.curve.times[-1] - self.curve.times[-2]
                )
                left_log = node_log_discounts[:, :1] + left_slope[:, None] * (
                    self.times[None, :] - self.curve.times[0]
                )
                right_log = node_log_discounts[:, -1:] + right_slope[:, None] * (
                    self.times[None, :] - self.curve.times[-1]
                )
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                result = np.where(self.before[None, :], np.exp(left_log), result)
                result = np.where(self.after[None, :], np.exp(right_log), result)

        result = np.where(self.times[None, :] == 0.0, 1.0, result)
        if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
            raise QFinValidationError(
                "scenario interpolation produced non-finite or non-positive discount factors"
            )
        return np.asarray(result, dtype=np.float64)


def _segment_sum_rows(values: FloatArray, offsets: Int64Array) -> FloatArray:
    """Reduce rows over prefix-delimited streams, including empty streams."""

    stream_count = offsets.size - 1
    result = np.zeros((values.shape[0], stream_count), dtype=np.float64)
    if values.shape[1] == 0 or stream_count == 0:
        return result
    starts = offsets[:-1]
    valid_starts = starts < values.shape[1]
    if np.any(valid_starts):
        result[:, valid_starts] = np.add.reduceat(values, starts[valid_starts], axis=1)
    result[:, np.diff(offsets) == 0] = 0.0
    return result


def _validate_flat_portfolio(
    times: FloatArray,
    amounts: FloatArray,
    offsets: Int64Array,
    weights: FloatArray,
    curve: YieldCurve,
    scenarios: RateScenarioSet,
) -> None:
    if times.ndim != 1 or amounts.shape != times.shape:
        raise QFinValidationError("cash-flow times and amounts must be one-dimensional and aligned")
    if offsets.ndim != 1 or offsets.size != weights.size + 1:
        raise QFinValidationError("offsets must delimit one cash-flow stream per position weight")
    if offsets.size == 0 or offsets[0] != 0 or offsets[-1] != times.size:
        raise QFinValidationError("invalid cash-flow offsets")
    if np.any(np.diff(offsets) < 0):
        raise QFinValidationError("cash-flow offsets must be non-decreasing")
    if not np.all(np.isfinite(times)) or np.any(times < 0):
        raise QFinValidationError("cash-flow times must be finite and non-negative")
    if not np.all(np.isfinite(amounts)) or not np.all(np.isfinite(weights)):
        raise QFinValidationError("cash-flow amounts and position weights must be finite")
    if scenarios.shocks.shape[1] != curve.times.size:
        raise QFinValidationError("each scenario must have one shock per curve node")


def _weighted_cashflow_amounts(
    amounts: FloatArray,
    offsets: Int64Array,
    weights: FloatArray,
) -> FloatArray:
    """Apply position weights once without allocating a repeated index vector."""

    weighted = np.array(amounts, dtype=np.float64, order="C", copy=True)
    for position, weight in enumerate(weights):
        weighted[offsets[position] : offsets[position + 1]] *= weight
    return weighted


def _numpy_scenario_values(
    prepared: _PreparedScenarioValuation,
    weighted_amounts: FloatArray,
    shocks: FloatArray,
) -> FloatArray:
    discounted = prepared.discount_factors(shocks) * weighted_amounts[None, :]
    values = np.asarray(np.sum(discounted, axis=1, dtype=np.float64), dtype=np.float64)
    if np.any(weighted_amounts < 0.0) and np.any(weighted_amounts > 0.0):
        magnitudes = np.sum(np.abs(discounted), axis=1, dtype=np.float64)
        cancellation = (magnitudes > 0.0) & (np.abs(values) <= 1.0e-10 * magnitudes)
        for row in np.flatnonzero(cancellation):
            values[row] = fsum(float(item) for item in discounted[row])
    return values


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
    normalized_chunk_size = require_integer(chunk_size, "chunk_size", minimum=1)
    workload = times.size * scenarios.shocks.shape[0]
    selected = resolve_engine(
        engine,
        workload,
        native_compatible=curve.native_compatible,
        auto_native_threshold=POLICIES["rate_scenarios"],
    )

    prepared = _PreparedScenarioValuation.build(times, curve)
    weighted_amounts = _weighted_cashflow_amounts(amounts, stream_offsets, weights)
    if selected == "numpy":
        normalized_chunk_size = _bounded_numpy_chunk_size(normalized_chunk_size, times, curve)
    check_allocation((scenarios.shocks.shape[0],))
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
            values[start:stop] = _numpy_scenario_values(prepared, weighted_amounts, shock_chunk)
    if not np.all(np.isfinite(values)):
        raise QFinValidationError("scenario valuation produced non-finite cash-flow values")
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
        raise QFinValidationError("cash-flow times, amounts, and inflation linkage must align")
    if (
        not np.all(np.isfinite(times))
        or np.any(times < 0.0)
        or not np.all(np.isfinite(amounts))
        or not np.all(np.isfinite(linkages))
        or np.any(linkages < 0.0)
    ):
        raise QFinValidationError(
            "indexed cash-flow inputs must be finite and non-negative in time/linkage"
        )
    scenarios.validate_curve(curve)
    period_index = require_integer(period, "period")
    normalized_chunk_size = require_integer(chunk_size, "chunk_size", minimum=1)
    if not 0 <= period_index < scenarios.period_count:
        raise QFinValidationError("period must be in range")
    workload = times.size * scenarios.scenario_count
    selected = resolve_engine(
        engine,
        workload,
        native_compatible=curve.native_compatible,
        auto_native_threshold=POLICIES["rate_scenarios"],
    )

    prepared = _PreparedScenarioValuation.build(times, curve)
    check_allocation((scenarios.scenario_count,))
    values = np.empty(scenarios.scenario_count, dtype=np.float64)
    if selected == "numpy":
        normalized_chunk_size = _bounded_numpy_chunk_size(normalized_chunk_size, times, curve)
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
        discounts = prepared.discount_factors(rate_shocks)
        scale = np.power(1.0 + inflation[:, None], times[None, :] * linkages[None, :])
        values[start:stop] = np.sum(
            amounts[None, :] * scale * discounts,
            axis=1,
            dtype=np.float64,
        )
    if not np.all(np.isfinite(values)):
        raise QFinValidationError("scenario valuation produced non-finite cash-flow values")
    return values, selected


__all__ = [
    "EconomicScenarioSet",
    "RateScenarioSet",
    "scenario_indexed_cashflow_values",
    "scenario_portfolio_values",
]
