"""Mortality tables and extensible annual life projection foundations."""

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
        ages = np.array(self.ages, dtype=np.float64, order="C", copy=True).reshape(-1)
        rates = np.array(self.rates, dtype=np.float64, order="C", copy=True).reshape(-1)
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


ProductType = Literal["term_life", "participating_life", "universal_life", "annuity"]

_PRODUCT_CODES: dict[str, int] = {
    "term_life": 0,
    "participating_life": 1,
    "universal_life": 2,
    "annuity": 3,
}


@dataclass(frozen=True, slots=True)
class LifePolicy:
    """Annual model point for supported protection and annuity foundations."""

    age: float
    sum_assured: float
    annual_premium: float
    term: int
    issue_age: float | None = None
    policy_duration: int = 0
    product_type: ProductType = "term_life"
    mortality_category: str = "unisex"
    annual_benefit: float = 0.0
    account_value: float = 0.0
    annual_charge: float = 0.0
    crediting_spread: float = 0.0
    bonus_rate: float = 0.0
    disability_benefit: float = 0.0
    benefit_inflation_linkage: float = 0.0

    def __post_init__(self) -> None:
        non_negative = {
            "age": self.age,
            "sum_assured": self.sum_assured,
            "annual_premium": self.annual_premium,
            "annual_benefit": self.annual_benefit,
            "account_value": self.account_value,
            "annual_charge": self.annual_charge,
            "bonus_rate": self.bonus_rate,
            "disability_benefit": self.disability_benefit,
            "benefit_inflation_linkage": self.benefit_inflation_linkage,
        }
        if any(not isfinite(value) or value < 0.0 for value in non_negative.values()):
            raise ValueError(
                "policy amounts, age, bonus, and inflation linkage must be finite and non-negative"
            )
        if not isfinite(self.crediting_spread):
            raise ValueError("crediting_spread must be finite")
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
            if not np.isclose(self.age, self.issue_age + duration, atol=1.0e-9):
                raise ValueError("age must equal issue_age + policy_duration")
        if self.product_type not in _PRODUCT_CODES:
            raise ValueError(f"unsupported life product type: {self.product_type}")
        if not self.mortality_category:
            raise ValueError("mortality_category must be non-empty")

    @property
    def remaining_term(self) -> int:
        return self.term - self.policy_duration

    @property
    def product_code(self) -> int:
        return _PRODUCT_CODES[self.product_type]


@dataclass(frozen=True, slots=True, init=False)
class PolicyModelPointSet:
    """Unique policy model points and their positive exposure counts."""

    policies: tuple[LifePolicy, ...]
    counts: FloatArray

    def __init__(
        self,
        policies: list[LifePolicy] | tuple[LifePolicy, ...],
        counts: ArrayLike | None = None,
    ) -> None:
        items = tuple(policies)
        if any(not isinstance(policy, LifePolicy) for policy in items):
            raise TypeError("policies must contain LifePolicy objects")
        weights = (
            np.ones(len(items), dtype=np.float64)
            if counts is None
            else np.array(counts, dtype=np.float64, order="C", copy=True).reshape(-1)
        )
        if (
            weights.shape != (len(items),)
            or not np.all(np.isfinite(weights))
            or np.any(weights <= 0.0)
        ):
            raise ValueError("counts must contain one finite positive value per model point")
        weights = np.array(weights, dtype=np.float64, order="C", copy=True)
        weights.setflags(write=False)
        object.__setattr__(self, "policies", items)
        object.__setattr__(self, "counts", weights)

    @classmethod
    def from_policies(
        cls,
        policies: list[LifePolicy] | tuple[LifePolicy, ...],
        counts: ArrayLike | None = None,
    ) -> PolicyModelPointSet:
        """Group exactly equal policies while preserving first-seen order."""

        items = tuple(policies)
        weights = (
            np.ones(len(items), dtype=np.float64)
            if counts is None
            else np.asarray(counts, dtype=np.float64).reshape(-1)
        )
        if weights.shape != (len(items),):
            raise ValueError("counts must contain one value per policy")
        grouped: dict[LifePolicy, float] = {}
        for policy, weight in zip(items, weights, strict=True):
            if not isinstance(policy, LifePolicy):
                raise TypeError("policies must contain LifePolicy objects")
            if not isfinite(float(weight)) or weight <= 0.0:
                raise ValueError("counts must be finite and positive")
            grouped[policy] = grouped.get(policy, 0.0) + float(weight)
        return cls(tuple(grouped), tuple(grouped.values()))

    @property
    def model_point_count(self) -> int:
        return len(self.policies)

    @property
    def policy_count(self) -> float:
        return float(np.sum(self.counts))

    @property
    def compression_ratio(self) -> float:
        return 1.0 if not self.policies else self.policy_count / len(self.policies)


def _assumption_path(
    values: float | ArrayLike,
    name: str,
    *,
    probability: bool = False,
    positive: bool = False,
    greater_than_minus_one: bool = False,
) -> FloatArray:
    path = np.array(values, dtype=np.float64, order="C", copy=True).reshape(-1)
    invalid = path.size == 0 or not np.all(np.isfinite(path))
    if probability:
        invalid = invalid or bool(np.any((path < 0.0) | (path > 1.0)))
    if positive:
        invalid = invalid or bool(np.any(path < 0.0))
    if greater_than_minus_one:
        invalid = invalid or bool(np.any(path <= -1.0))
    if invalid:
        qualifier = (
            "probabilities in [0, 1]" if probability else "finite values in the supported range"
        )
        raise ValueError(f"{name} must contain {qualifier}")
    path.setflags(write=False)
    return path


@dataclass(frozen=True, slots=True)
class ProjectionAssumptions:
    """Named mortality, decrement, expense, crediting, and scaling assumptions."""

    mortality: MortalityTable
    curve: YieldCurve
    lapse_rate: float | FloatArray = 0.0
    expense_per_policy: float = 0.0
    mortality_multiplier: float = 1.0
    disability_rate: float | FloatArray = 0.0
    recovery_rate: float | FloatArray = 0.0
    disabled_mortality_multiplier: float = 1.0
    crediting_rate: float | FloatArray = 0.0
    expense_inflation_rate: float | FloatArray = 0.0
    premium_multiplier: float = 1.0
    benefit_multiplier: float = 1.0

    def __post_init__(self) -> None:
        lapse = _assumption_path(self.lapse_rate, "lapse rates", probability=True)
        disability = _assumption_path(self.disability_rate, "disability rates", probability=True)
        recovery = _assumption_path(self.recovery_rate, "recovery rates", probability=True)
        crediting = _assumption_path(
            self.crediting_rate, "crediting rates", greater_than_minus_one=True
        )
        expense_inflation = _assumption_path(
            self.expense_inflation_rate,
            "expense inflation rates",
            greater_than_minus_one=True,
        )
        non_negative = (
            self.expense_per_policy,
            self.mortality_multiplier,
            self.disabled_mortality_multiplier,
            self.premium_multiplier,
            self.benefit_multiplier,
        )
        if any(not isfinite(value) or value < 0.0 for value in non_negative):
            raise ValueError("scalar projection assumptions must be finite and non-negative")
        object.__setattr__(self, "lapse_rate", lapse)
        object.__setattr__(self, "disability_rate", disability)
        object.__setattr__(self, "recovery_rate", recovery)
        object.__setattr__(self, "crediting_rate", crediting)
        object.__setattr__(self, "expense_inflation_rate", expense_inflation)


LifeAssumptionSet = ProjectionAssumptions


@dataclass(frozen=True, slots=True)
class LifeProjectionResult:
    """Aggregate product/state cash flows plus model-point present values."""

    times: FloatArray
    expected_premiums: FloatArray
    expected_benefits: FloatArray
    expected_expenses: FloatArray
    expected_surrenders: FloatArray
    net_liability_cashflows: FloatArray
    in_force: FloatArray
    active: FloatArray
    disabled: FloatArray
    deaths: FloatArray
    policy_present_values: FloatArray
    model_point_counts: FloatArray
    product_present_values: dict[str, float]
    present_value: float
    duration: float
    engine: Literal["numpy", "native"]

    def to_liability_portfolio(self) -> LiabilityPortfolio:
        """Convert projected net insurer outflows into deterministic ALM cash flows."""

        return LiabilityPortfolio(
            [
                CashFlow(float(time), float(amount))
                for time, amount in zip(self.times, self.net_liability_cashflows, strict=True)
                if amount != 0.0
            ]
        )


def _path_value(path: FloatArray, year: int) -> float:
    return float(path[0] if path.size == 1 else path[year])


def _validate_horizon(assumptions: ProjectionAssumptions, maximum_term: int) -> None:
    named_paths = {
        "lapse_rate": cast(FloatArray, assumptions.lapse_rate),
        "disability_rate": cast(FloatArray, assumptions.disability_rate),
        "recovery_rate": cast(FloatArray, assumptions.recovery_rate),
        "crediting_rate": cast(FloatArray, assumptions.crediting_rate),
        "expense_inflation_rate": cast(FloatArray, assumptions.expense_inflation_rate),
    }
    for name, path in named_paths.items():
        if path.size != 1 and path.size < maximum_term:
            raise ValueError(f"{name} must be scalar or cover every projection year")


def _as_model_points(
    policies: PolicyModelPointSet | list[LifePolicy] | tuple[LifePolicy, ...],
) -> PolicyModelPointSet:
    return policies if isinstance(policies, PolicyModelPointSet) else PolicyModelPointSet(policies)


def _policy_buffers(model_points: PolicyModelPointSet) -> tuple[FloatArray, ...]:
    policies = model_points.policies
    return (
        np.ascontiguousarray([policy.age for policy in policies], dtype=np.float64),
        np.ascontiguousarray([policy.sum_assured for policy in policies], dtype=np.float64),
        np.ascontiguousarray([policy.annual_premium for policy in policies], dtype=np.float64),
        np.ascontiguousarray([policy.remaining_term for policy in policies], dtype=np.int32),
        np.ascontiguousarray([policy.policy_duration for policy in policies], dtype=np.int32),
        np.ascontiguousarray([policy.term for policy in policies], dtype=np.int32),
        np.ascontiguousarray([policy.product_code for policy in policies], dtype=np.int32),
        np.ascontiguousarray(model_points.counts, dtype=np.float64),
        np.ascontiguousarray([policy.annual_benefit for policy in policies], dtype=np.float64),
        np.ascontiguousarray([policy.account_value for policy in policies], dtype=np.float64),
        np.ascontiguousarray([policy.annual_charge for policy in policies], dtype=np.float64),
        np.ascontiguousarray([policy.crediting_spread for policy in policies], dtype=np.float64),
        np.ascontiguousarray([policy.bonus_rate for policy in policies], dtype=np.float64),
        np.ascontiguousarray([policy.disability_benefit for policy in policies], dtype=np.float64),
        np.ascontiguousarray(
            [policy.benefit_inflation_linkage for policy in policies],
            dtype=np.float64,
        ),
    )


def _numpy_projection(
    model_points: PolicyModelPointSet, assumptions: ProjectionAssumptions
) -> dict[str, object]:
    policies = model_points.policies
    maximum_term = max((policy.remaining_term for policy in policies), default=0)
    output_size = maximum_term + 1
    premiums = np.zeros(output_size, dtype=np.float64)
    benefits = np.zeros(output_size, dtype=np.float64)
    expenses = np.zeros(output_size, dtype=np.float64)
    surrenders = np.zeros(output_size, dtype=np.float64)
    net = np.zeros(output_size, dtype=np.float64)
    active_counts = np.zeros(output_size, dtype=np.float64)
    disabled_counts = np.zeros(output_size, dtype=np.float64)
    death_counts = np.zeros(output_size, dtype=np.float64)
    point_values = np.zeros(len(policies), dtype=np.float64)
    lapse_rates = cast(FloatArray, assumptions.lapse_rate)
    disability_rates = cast(FloatArray, assumptions.disability_rate)
    recovery_rates = cast(FloatArray, assumptions.recovery_rate)
    crediting_rates = cast(FloatArray, assumptions.crediting_rate)
    inflation_rates = cast(FloatArray, assumptions.expense_inflation_rate)

    for policy_index, (policy, count) in enumerate(zip(policies, model_points.counts, strict=True)):
        active = 1.0
        disabled = 0.0
        account_value = policy.account_value
        inflation_index = 1.0
        policy_pv = 0.0
        for year in range(policy.remaining_term):
            active_counts[year] += count * active
            disabled_counts[year] += count * disabled
            premium = active * policy.annual_premium * assumptions.premium_multiplier
            expense = (active + disabled) * assumptions.expense_per_policy * inflation_index
            if policy.product_type == "universal_life":
                account_value = max(
                    0.0,
                    (
                        account_value
                        + policy.annual_premium * assumptions.premium_multiplier
                        - policy.annual_charge
                    )
                    * (1.0 + _path_value(crediting_rates, year) + policy.crediting_spread),
                )
            qx_active = float(
                np.clip(
                    assumptions.mortality.qx(policy.age + year) * assumptions.mortality_multiplier,
                    0.0,
                    1.0,
                )
            )
            qx_disabled = float(
                np.clip(
                    qx_active * assumptions.disabled_mortality_multiplier,
                    0.0,
                    1.0,
                )
            )
            active_deaths = active * qx_active
            disabled_deaths = disabled * qx_disabled
            active_survivors = active - active_deaths
            disabled_survivors = disabled - disabled_deaths
            new_disabled = active_survivors * _path_value(disability_rates, year)
            recoveries = disabled_survivors * _path_value(recovery_rates, year)
            active_before_lapse = active_survivors - new_disabled + recoveries
            disabled_before_lapse = disabled_survivors - recoveries + new_disabled
            lapse = _path_value(lapse_rates, year)
            lapse_count = (active_before_lapse + disabled_before_lapse) * lapse
            active = active_before_lapse * (1.0 - lapse)
            disabled = disabled_before_lapse * (1.0 - lapse)
            next_inflation = inflation_index * (1.0 + _path_value(inflation_rates, year))
            benefit_scale = (
                assumptions.benefit_multiplier * next_inflation**policy.benefit_inflation_linkage
            )
            alive_deaths = active_deaths + disabled_deaths
            if policy.product_type == "annuity":
                death_benefit = 0.0
                annuity_benefit = (
                    (active_survivors + disabled_survivors) * policy.annual_benefit * benefit_scale
                )
            else:
                insured_amount = policy.sum_assured
                if policy.product_type == "participating_life":
                    insured_amount *= (1.0 + policy.bonus_rate) ** (
                        policy.policy_duration + year + 1
                    )
                elif policy.product_type == "universal_life":
                    insured_amount = max(insured_amount, account_value)
                death_benefit = alive_deaths * insured_amount * benefit_scale
                annuity_benefit = 0.0
            disability_benefit = disabled_before_lapse * policy.disability_benefit * benefit_scale
            surrender_benefit = (
                lapse_count * account_value if policy.product_type == "universal_life" else 0.0
            )
            maturity_benefit = 0.0
            if year + 1 == policy.remaining_term:
                ending_in_force = active + disabled
                if policy.product_type == "participating_life":
                    maturity_benefit = (
                        ending_in_force
                        * policy.sum_assured
                        * (1.0 + policy.bonus_rate) ** policy.term
                        * benefit_scale
                    )
                elif policy.product_type == "universal_life":
                    maturity_benefit = ending_in_force * account_value * benefit_scale
            end_benefit = death_benefit + annuity_benefit + disability_benefit + maturity_benefit
            start_net = expense - premium
            end_net = end_benefit + surrender_benefit
            premiums[year] += count * premium
            expenses[year] += count * expense
            benefits[year + 1] += count * end_benefit
            surrenders[year + 1] += count * surrender_benefit
            death_counts[year + 1] += count * alive_deaths
            net[year] += count * start_net
            net[year + 1] += count * end_net
            policy_pv += count * start_net * assumptions.curve.discount(float(year))
            policy_pv += count * end_net * assumptions.curve.discount(float(year + 1))
            inflation_index = next_inflation
        point_values[policy_index] = policy_pv
    times = np.arange(output_size, dtype=np.float64)
    discounted = net * np.asarray(assumptions.curve.discount(times), dtype=np.float64)
    present_value = float(np.sum(discounted))
    duration = (
        0.0 if abs(present_value) <= 1.0e-15 else float(np.dot(times, discounted) / present_value)
    )
    return {
        "expected_premiums": premiums,
        "expected_benefits": benefits,
        "expected_expenses": expenses,
        "expected_surrenders": surrenders,
        "net_liability_cashflows": net,
        "active": active_counts,
        "disabled": disabled_counts,
        "deaths": death_counts,
        "policy_present_values": point_values,
        "present_value": present_value,
        "duration": duration,
    }


def project_liabilities(
    policies: PolicyModelPointSet | list[LifePolicy] | tuple[LifePolicy, ...],
    assumptions: ProjectionAssumptions,
    *,
    engine: Engine = "auto",
) -> LifeProjectionResult:
    """Project product and multi-state model points in one batched dispatch."""

    model_points = _as_model_points(policies)
    mismatched = [
        policy.mortality_category
        for policy in model_points.policies
        if policy.mortality_category != assumptions.mortality.category
    ]
    if mismatched:
        raise ValueError("all policies must match the supplied mortality-table category")
    maximum_term = max((policy.remaining_term for policy in model_points.policies), default=0)
    _validate_horizon(assumptions, maximum_term)
    if engine not in ("auto", "numpy", "native"):
        raise ValueError("engine must be 'auto', 'numpy', or 'native'")
    workload = model_points.model_point_count * maximum_term
    selected: Literal["numpy", "native"]
    if engine == "native":
        _native.require()
        selected = "native"
    elif engine == "numpy":
        selected = "numpy"
    else:
        selected = "native" if _native.available() and workload > 0 else "numpy"
    if selected == "native":
        buffers = _policy_buffers(model_points)
        raw = cast(
            dict[str, object],
            _native.require().project_life_model_points(
                *buffers,
                assumptions.mortality.ages,
                assumptions.mortality.rates,
                cast(FloatArray, assumptions.lapse_rate),
                cast(FloatArray, assumptions.disability_rate),
                cast(FloatArray, assumptions.recovery_rate),
                cast(FloatArray, assumptions.crediting_rate),
                cast(FloatArray, assumptions.expense_inflation_rate),
                assumptions.expense_per_policy,
                assumptions.mortality_multiplier,
                assumptions.disabled_mortality_multiplier,
                assumptions.premium_multiplier,
                assumptions.benefit_multiplier,
                assumptions.curve.times,
                assumptions.curve.zero_rates,
            ),
        )
    else:
        raw = _numpy_projection(model_points, assumptions)
    times = np.arange(maximum_term + 1, dtype=np.float64)
    point_values = np.asarray(raw["policy_present_values"], dtype=np.float64)
    product_values = {
        product: float(
            np.sum(
                point_values[
                    np.asarray(
                        [policy.product_type == product for policy in model_points.policies],
                        dtype=np.bool_,
                    )
                ]
            )
        )
        for product in _PRODUCT_CODES
    }
    active = np.asarray(raw["active"], dtype=np.float64)
    disabled = np.asarray(raw["disabled"], dtype=np.float64)
    return LifeProjectionResult(
        times=times,
        expected_premiums=np.asarray(raw["expected_premiums"], dtype=np.float64),
        expected_benefits=np.asarray(raw["expected_benefits"], dtype=np.float64),
        expected_expenses=np.asarray(raw["expected_expenses"], dtype=np.float64),
        expected_surrenders=np.asarray(raw["expected_surrenders"], dtype=np.float64),
        net_liability_cashflows=np.asarray(raw["net_liability_cashflows"], dtype=np.float64),
        in_force=active + disabled,
        active=active,
        disabled=disabled,
        deaths=np.asarray(raw["deaths"], dtype=np.float64),
        policy_present_values=point_values,
        model_point_counts=model_points.counts,
        product_present_values=product_values,
        present_value=float(cast(float, raw["present_value"])),
        duration=float(cast(float, raw["duration"])),
        engine=selected,
    )


__all__ = [
    "LifeAssumptionSet",
    "LifePolicy",
    "LifeProjectionResult",
    "MortalityTable",
    "PolicyModelPointSet",
    "ProjectionAssumptions",
    "project_liabilities",
]
