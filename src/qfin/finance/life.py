"""Mortality tables and an extensible annual term-life projection foundation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from operator import index as integer_index
from typing import Literal, cast, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from qfin import _native
from qfin.finance.alm import LiabilityPortfolio
from qfin.finance.curves import YieldCurve
from qfin.finance.fixed_income import CashFlow, Engine

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class MortalityTable:
    """Annual mortality probabilities with linear interpolation and flat tails."""

    ages: FloatArray
    rates: FloatArray
    category: str = "unisex"

    def __post_init__(self) -> None:
        ages = np.ascontiguousarray(self.ages, dtype=np.float64).reshape(-1)
        rates = np.ascontiguousarray(self.rates, dtype=np.float64).reshape(-1)
        if ages.size == 0 or ages.shape != rates.shape:
            raise ValueError("ages and rates must have equal non-zero length")
        if not np.all(np.isfinite(ages)) or np.any(ages < 0) or np.any(np.diff(ages) <= 0):
            raise ValueError("mortality ages must be finite, non-negative, and increasing")
        if not np.all(np.isfinite(rates)) or np.any((rates < 0) | (rates > 1)):
            raise ValueError("mortality qx values must lie in [0, 1]")
        if not self.category:
            raise ValueError("mortality category must be non-empty")
        ages.setflags(write=False)
        rates.setflags(write=False)
        object.__setattr__(self, "ages", ages)
        object.__setattr__(self, "rates", rates)

    @overload
    def qx(self, age: float) -> float: ...

    @overload
    def qx(self, age: ArrayLike) -> FloatArray: ...

    def qx(self, age: float | ArrayLike) -> float | FloatArray:
        """Return annual death probabilities for attained ages."""

        query = np.asarray(age, dtype=np.float64)
        if not np.all(np.isfinite(query)) or np.any(query < 0):
            raise ValueError("ages must be finite and non-negative")
        # np.interp is already a compiled, vectorized kernel and benchmarks faster
        # than crossing the QFin extension boundary for this isolated operation.
        values = np.interp(
            query,
            self.ages,
            self.rates,
            left=self.rates[0],
            right=self.rates[-1],
        )
        if query.ndim == 0:
            return float(values)
        return np.asarray(values, dtype=np.float64)

    @overload
    def px(self, age: float) -> float: ...

    @overload
    def px(self, age: ArrayLike) -> FloatArray: ...

    def px(self, age: float | ArrayLike) -> float | FloatArray:
        """Return one-year survival probabilities."""

        value = self.qx(age)
        if isinstance(value, float):
            return 1.0 - value
        return np.asarray(1.0 - value, dtype=np.float64)

    def survival_probability(self, age: float, years: int) -> float:
        """Return annual-step survival ``p_x(years)``."""

        if not isfinite(age) or age < 0:
            raise ValueError("age must be finite and non-negative")
        try:
            year_count = integer_index(years)
        except TypeError as exc:
            raise ValueError("years must be a non-negative integer") from exc
        if isinstance(years, bool) or year_count < 0:
            raise ValueError("years must be non-negative")
        if year_count == 0:
            return 1.0
        qx = np.asarray(self.qx(age + np.arange(year_count, dtype=np.float64)))
        return float(np.prod(1.0 - qx))


@dataclass(frozen=True, slots=True)
class LifePolicy:
    """Annual-step term-life policy record used by the projection engine."""

    age: float
    sum_assured: float
    annual_premium: float
    term: int
    issue_age: float | None = None
    policy_duration: int = 0
    product_type: Literal["term_life"] = "term_life"
    mortality_category: str = "unisex"

    def __post_init__(self) -> None:
        if not isfinite(self.age) or self.age < 0:
            raise ValueError("age must be finite and non-negative")
        if not isfinite(self.sum_assured) or self.sum_assured < 0:
            raise ValueError("sum_assured must be finite and non-negative")
        if not isfinite(self.annual_premium) or self.annual_premium < 0:
            raise ValueError("annual_premium must be finite and non-negative")
        try:
            term = integer_index(self.term)
            duration = integer_index(self.policy_duration)
        except TypeError as exc:
            raise ValueError("term and policy_duration must be integers") from exc
        if (
            isinstance(self.term, bool)
            or isinstance(self.policy_duration, bool)
            or term <= 0
            or duration < 0
            or duration > term
        ):
            raise ValueError("require 0 <= policy_duration <= term with positive term")
        object.__setattr__(self, "term", term)
        object.__setattr__(self, "policy_duration", duration)
        if self.issue_age is not None:
            if not isfinite(self.issue_age) or self.issue_age < 0:
                raise ValueError("issue_age must be finite and non-negative")
            expected_age = self.issue_age + duration
            if not np.isclose(self.age, expected_age, atol=1.0e-9):
                raise ValueError("age must equal issue_age + policy_duration")
        if self.product_type != "term_life":
            raise ValueError("the first projection milestone supports term_life only")
        if not self.mortality_category:
            raise ValueError("mortality_category must be non-empty")

    @property
    def remaining_term(self) -> int:
        return self.term - self.policy_duration


@dataclass(frozen=True, slots=True)
class ProjectionAssumptions:
    """Mortality, lapse, expense, and discount assumptions."""

    mortality: MortalityTable
    curve: YieldCurve
    lapse_rate: float | FloatArray = 0.0
    expense_per_policy: float = 0.0
    mortality_multiplier: float = 1.0

    def __post_init__(self) -> None:
        lapse = np.ascontiguousarray(self.lapse_rate, dtype=np.float64).reshape(-1)
        if lapse.size == 0 or not np.all(np.isfinite(lapse)) or np.any((lapse < 0) | (lapse > 1)):
            raise ValueError("lapse rates must contain probabilities in [0, 1]")
        if not isfinite(self.expense_per_policy) or self.expense_per_policy < 0:
            raise ValueError("expense_per_policy must be finite and non-negative")
        if not isfinite(self.mortality_multiplier) or self.mortality_multiplier < 0:
            raise ValueError("mortality_multiplier must be finite and non-negative")
        lapse.setflags(write=False)
        object.__setattr__(self, "lapse_rate", lapse)


@dataclass(frozen=True, slots=True)
class LifeProjectionResult:
    """Aggregate annual cash flows plus per-policy present values."""

    times: FloatArray
    expected_premiums: FloatArray
    expected_benefits: FloatArray
    expected_expenses: FloatArray
    net_liability_cashflows: FloatArray
    in_force: FloatArray
    policy_present_values: FloatArray
    present_value: float
    duration: float
    engine: Literal["numpy", "native"]

    def to_liability_portfolio(self) -> LiabilityPortfolio:
        """Convert projected net insurer outflows into deterministic ALM cash flows."""

        return LiabilityPortfolio(
            [
                CashFlow(float(time), float(amount))
                for time, amount in zip(
                    self.times, self.net_liability_cashflows, strict=True
                )
                if amount != 0.0
            ]
        )


def _policy_buffers(
    policies: tuple[LifePolicy, ...],
) -> tuple[FloatArray, FloatArray, FloatArray, NDArray[np.int32]]:
    return (
        np.ascontiguousarray([policy.age for policy in policies], dtype=np.float64),
        np.ascontiguousarray([policy.sum_assured for policy in policies], dtype=np.float64),
        np.ascontiguousarray([policy.annual_premium for policy in policies], dtype=np.float64),
        np.ascontiguousarray([policy.remaining_term for policy in policies], dtype=np.int32),
    )


def _numpy_projection(
    policies: tuple[LifePolicy, ...], assumptions: ProjectionAssumptions
) -> dict[str, object]:
    maximum_term = max((policy.remaining_term for policy in policies), default=0)
    premiums = np.zeros(maximum_term + 1, dtype=np.float64)
    benefits = np.zeros(maximum_term + 1, dtype=np.float64)
    expenses = np.zeros(maximum_term + 1, dtype=np.float64)
    net = np.zeros(maximum_term + 1, dtype=np.float64)
    in_force_counts = np.zeros(maximum_term + 1, dtype=np.float64)
    policy_values = np.zeros(len(policies), dtype=np.float64)
    lapse_rates = cast(FloatArray, assumptions.lapse_rate)
    if lapse_rates.size != 1 and lapse_rates.size < maximum_term:
        raise ValueError("lapse_rate must be scalar or cover every projection year")
    for policy_index, policy in enumerate(policies):
        in_force = 1.0
        policy_pv = 0.0
        for year in range(policy.remaining_term):
            in_force_counts[year] += in_force
            premium = in_force * policy.annual_premium
            expense = in_force * assumptions.expense_per_policy
            qx = min(
                max(
                    assumptions.mortality.qx(policy.age + year)
                    * assumptions.mortality_multiplier,
                    0.0,
                ),
                1.0,
            )
            death_benefit = in_force * qx * policy.sum_assured
            lapse = float(lapse_rates[0] if lapse_rates.size == 1 else lapse_rates[year])
            in_force = in_force * (1.0 - qx) * (1.0 - lapse)
            premiums[year] += premium
            expenses[year] += expense
            benefits[year + 1] += death_benefit
            net[year] += expense - premium
            net[year + 1] += death_benefit
            policy_pv += (expense - premium) * assumptions.curve.discount(float(year))
            policy_pv += death_benefit * assumptions.curve.discount(float(year + 1))
        policy_values[policy_index] = policy_pv
    times = np.arange(maximum_term + 1, dtype=np.float64)
    discounted = net * np.asarray(assumptions.curve.discount(times), dtype=np.float64)
    present_value = float(np.sum(discounted))
    duration = (
        0.0
        if abs(present_value) <= 1.0e-15
        else float(np.dot(times, discounted) / present_value)
    )
    return {
        "expected_premiums": premiums,
        "expected_benefits": benefits,
        "expected_expenses": expenses,
        "net_liability_cashflows": net,
        "in_force": in_force_counts,
        "policy_present_values": policy_values,
        "present_value": present_value,
        "duration": duration,
    }


def project_liabilities(
    policies: list[LifePolicy] | tuple[LifePolicy, ...],
    assumptions: ProjectionAssumptions,
    *,
    engine: Engine = "auto",
) -> LifeProjectionResult:
    """Project a policy batch across the Python/C++ boundary once."""

    items = tuple(policies)
    if any(not isinstance(policy, LifePolicy) for policy in items):
        raise TypeError("policies must contain LifePolicy objects")
    mismatched = [
        policy.mortality_category
        for policy in items
        if policy.mortality_category != assumptions.mortality.category
    ]
    if mismatched:
        raise ValueError("all policies must match the supplied mortality-table category")
    maximum_term = max((policy.remaining_term for policy in items), default=0)
    lapse_rates = cast(FloatArray, assumptions.lapse_rate)
    if lapse_rates.size != 1 and lapse_rates.size < maximum_term:
        raise ValueError("lapse_rate must be scalar or cover every projection year")
    if engine not in ("auto", "numpy", "native"):
        raise ValueError("engine must be 'auto', 'numpy', or 'native'")
    workload = len(items) * maximum_term
    selected: Literal["numpy", "native"]
    if engine == "native":
        _native.require()
        selected = "native"
    elif engine == "numpy":
        selected = "numpy"
    else:
        selected = "native" if _native.available() and workload >= 10_000 else "numpy"
    if selected == "native":
        ages, assured, premiums, terms = _policy_buffers(items)
        raw = cast(
            dict[str, object],
            _native.require().project_term_life_policies(
                ages,
                assured,
                premiums,
                terms,
                assumptions.mortality.ages,
                assumptions.mortality.rates,
                lapse_rates,
                assumptions.expense_per_policy,
                assumptions.mortality_multiplier,
                assumptions.curve.times,
                assumptions.curve.zero_rates,
            ),
        )
    else:
        raw = _numpy_projection(items, assumptions)
    times = np.arange(maximum_term + 1, dtype=np.float64)
    return LifeProjectionResult(
        times=times,
        expected_premiums=np.asarray(raw["expected_premiums"], dtype=np.float64),
        expected_benefits=np.asarray(raw["expected_benefits"], dtype=np.float64),
        expected_expenses=np.asarray(raw["expected_expenses"], dtype=np.float64),
        net_liability_cashflows=np.asarray(
            raw["net_liability_cashflows"], dtype=np.float64
        ),
        in_force=np.asarray(raw["in_force"], dtype=np.float64),
        policy_present_values=np.asarray(raw["policy_present_values"], dtype=np.float64),
        present_value=float(cast(float, raw["present_value"])),
        duration=float(cast(float, raw["duration"])),
        engine=selected,
    )


__all__ = [
    "LifePolicy",
    "LifeProjectionResult",
    "MortalityTable",
    "ProjectionAssumptions",
    "project_liabilities",
]
