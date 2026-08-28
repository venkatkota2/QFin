"""Expected life-insurance cash flows under a deterministic mortality basis."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from qfin.finance.fixed_income import CashFlowSchedule

FloatArray = NDArray[np.float64]


def _finite_vector(values: ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain at least one finite value")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class MortalityTable:
    """Select mortality table represented by integer attained ages and qx."""

    ages: NDArray[np.int64]
    qx: FloatArray

    def __post_init__(self) -> None:
        raw_ages = np.asarray(self.ages)
        if raw_ages.ndim != 1 or raw_ages.size == 0:
            raise ValueError("ages must be a non-empty one-dimensional array")
        if not np.all(np.equal(raw_ages, np.floor(raw_ages))):
            raise ValueError("mortality ages must be integers")
        ages = np.asarray(raw_ages, dtype=np.int64).copy()
        qx = _finite_vector(self.qx, name="qx")
        if ages.shape != qx.shape:
            raise ValueError("ages and qx must have the same shape")
        if np.any(np.diff(ages) != 1):
            raise ValueError("mortality ages must be consecutive and increasing")
        if np.any((qx < 0) | (qx > 1)):
            raise ValueError("qx values must lie in [0, 1]")
        ages.setflags(write=False)
        object.__setattr__(self, "ages", ages)
        object.__setattr__(self, "qx", qx)

    @classmethod
    def illustrative_gompertz_makeham(
        cls,
        *,
        min_age: int = 0,
        max_age: int = 120,
        makeham: float = 0.0005,
        scale: float = 0.000075,
        growth: float = 1.075,
    ) -> MortalityTable:
        """Create a synthetic teaching basis, not a regulatory valuation table."""
        if min_age < 0 or max_age <= min_age:
            raise ValueError("require 0 <= min_age < max_age")
        if makeham < 0 or scale < 0 or growth <= 1:
            raise ValueError("invalid Gompertz-Makeham parameters")
        ages = np.arange(min_age, max_age + 1, dtype=np.int64)
        force = makeham + scale * np.power(growth, ages, dtype=np.float64)
        qx = np.clip(1.0 - np.exp(-force), 0.0, 1.0)
        qx[-1] = 1.0
        return cls(ages=ages, qx=qx)

    @property
    def min_age(self) -> int:
        return int(self.ages[0])

    @property
    def max_age(self) -> int:
        return int(self.ages[-1])

    def qx_at(self, ages: ArrayLike) -> FloatArray:
        query = np.asarray(ages)
        if not np.all(np.equal(query, np.floor(query))):
            raise ValueError("attained ages must be integers")
        attained = np.asarray(query, dtype=np.int64)
        indices = attained - self.min_age
        if np.any(indices < 0) or np.any(indices >= self.qx.size):
            raise ValueError("requested attained age is outside the mortality table")
        return np.asarray(self.qx[indices], dtype=np.float64)

    def survival_start_probabilities(self, issue_age: int, years: int) -> FloatArray:
        """Probability alive at the start of policy years 0 through years-1."""
        if years < 1:
            raise ValueError("years must be positive")
        qx = self.qx_at(issue_age + np.arange(years, dtype=np.int64))
        survival = np.ones(years, dtype=np.float64)
        if years > 1:
            with np.errstate(divide="ignore"):
                survival[1:] = np.exp(np.cumsum(np.log1p(-qx[:-1])))
        return survival

    def death_probabilities(self, issue_age: int, years: int) -> FloatArray:
        qx = self.qx_at(issue_age + np.arange(years, dtype=np.int64))
        return self.survival_start_probabilities(issue_age, years) * qx


@dataclass(frozen=True, slots=True)
class LifeCashFlowProjection:
    """Expected annual policy cash flows; premiums offset gross liabilities."""

    times: FloatArray
    benefits: FloatArray
    premiums: FloatArray
    expenses: FloatArray

    def __post_init__(self) -> None:
        times = _finite_vector(self.times, name="projection times")
        benefits = _finite_vector(self.benefits, name="benefits")
        premiums = _finite_vector(self.premiums, name="premiums")
        expenses = _finite_vector(self.expenses, name="expenses")
        if not (times.shape == benefits.shape == premiums.shape == expenses.shape):
            raise ValueError("life cash-flow arrays must have the same shape")
        if np.any(times < 0) or np.any(np.diff(times) <= 0):
            raise ValueError("projection times must be non-negative and increasing")
        if np.any(benefits < 0) or np.any(premiums < 0) or np.any(expenses < 0):
            raise ValueError("projected benefits, premiums, and expenses must be non-negative")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "benefits", benefits)
        object.__setattr__(self, "premiums", premiums)
        object.__setattr__(self, "expenses", expenses)

    @property
    def net_liabilities(self) -> FloatArray:
        values = self.benefits + self.expenses - self.premiums
        values.setflags(write=False)
        return values

    @property
    def net_schedule(self) -> CashFlowSchedule:
        return CashFlowSchedule(self.times, self.net_liabilities)

    def scaled(self, multiplier: float) -> LifeCashFlowProjection:
        if not isfinite(multiplier) or multiplier < 0:
            raise ValueError("projection multiplier must be finite and non-negative")
        return LifeCashFlowProjection(
            self.times,
            self.benefits * multiplier,
            self.premiums * multiplier,
            self.expenses * multiplier,
        )


def _project_policy(
    *,
    issue_age: int,
    years: int,
    face_amount: float,
    annual_premium: float,
    premium_term: int | None,
    annual_expense: float,
    mortality: MortalityTable,
) -> LifeCashFlowProjection:
    survival = mortality.survival_start_probabilities(issue_age, years)
    deaths = mortality.death_probabilities(issue_age, years)
    times = np.arange(years + 1, dtype=np.float64)
    benefits = np.zeros(years + 1, dtype=np.float64)
    premiums = np.zeros(years + 1, dtype=np.float64)
    expenses = np.zeros(years + 1, dtype=np.float64)
    benefits[1:] = face_amount * deaths
    pay_years = years if premium_term is None else min(years, premium_term)
    premiums[:pay_years] = annual_premium * survival[:pay_years]
    expenses[:years] = annual_expense * survival
    return LifeCashFlowProjection(times, benefits, premiums, expenses)


def _validate_policy(
    issue_age: int,
    face_amount: float,
    annual_premium: float,
    premium_term: int | None,
    annual_expense: float,
) -> None:
    if isinstance(issue_age, bool) or issue_age < 0:
        raise ValueError("issue_age must be a non-negative integer")
    values = (face_amount, annual_premium, annual_expense)
    if not all(isfinite(value) for value in values):
        raise ValueError("policy amounts must be finite")
    if face_amount <= 0 or annual_premium < 0 or annual_expense < 0:
        raise ValueError("face_amount must be positive and other amounts non-negative")
    if premium_term is not None and premium_term < 1:
        raise ValueError("premium_term must be positive when supplied")


@dataclass(frozen=True, slots=True)
class TermLifePolicy:
    issue_age: int
    term: int
    face_amount: float
    annual_premium: float = 0.0
    premium_term: int | None = None
    annual_expense: float = 0.0

    def __post_init__(self) -> None:
        _validate_policy(
            self.issue_age,
            self.face_amount,
            self.annual_premium,
            self.premium_term,
            self.annual_expense,
        )
        if isinstance(self.term, bool) or self.term < 1:
            raise ValueError("term must be a positive integer")
        if self.premium_term is not None and self.premium_term > self.term:
            raise ValueError("premium_term cannot exceed the policy term")

    def expected_cashflows(self, mortality: MortalityTable) -> LifeCashFlowProjection:
        return _project_policy(
            issue_age=self.issue_age,
            years=self.term,
            face_amount=self.face_amount,
            annual_premium=self.annual_premium,
            premium_term=self.premium_term,
            annual_expense=self.annual_expense,
            mortality=mortality,
        )


@dataclass(frozen=True, slots=True)
class WholeLifePolicy:
    issue_age: int
    face_amount: float
    annual_premium: float = 0.0
    premium_term: int | None = None
    annual_expense: float = 0.0

    def __post_init__(self) -> None:
        _validate_policy(
            self.issue_age,
            self.face_amount,
            self.annual_premium,
            self.premium_term,
            self.annual_expense,
        )

    def expected_cashflows(self, mortality: MortalityTable) -> LifeCashFlowProjection:
        years = mortality.max_age - self.issue_age + 1
        if years < 1:
            raise ValueError("issue_age is outside the mortality table")
        if self.premium_term is not None and self.premium_term > years:
            raise ValueError("premium_term exceeds the available mortality horizon")
        return _project_policy(
            issue_age=self.issue_age,
            years=years,
            face_amount=self.face_amount,
            annual_premium=self.annual_premium,
            premium_term=self.premium_term,
            annual_expense=self.annual_expense,
            mortality=mortality,
        )


LifePolicy: TypeAlias = TermLifePolicy | WholeLifePolicy


@dataclass(frozen=True, slots=True)
class PolicyPosition:
    policy: LifePolicy
    count: float = 1.0

    def __post_init__(self) -> None:
        if not isfinite(self.count) or self.count < 0:
            raise ValueError("policy count must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class LifePolicyPortfolio:
    positions: tuple[PolicyPosition, ...]

    def __post_init__(self) -> None:
        positions = tuple(self.positions)
        if not positions:
            raise ValueError("life portfolio requires at least one position")
        object.__setattr__(self, "positions", positions)

    def expected_cashflows(self, mortality: MortalityTable) -> LifeCashFlowProjection:
        projections = tuple(
            position.policy.expected_cashflows(mortality).scaled(position.count)
            for position in self.positions
        )
        times = np.unique(np.concatenate([projection.times for projection in projections]))
        benefits = np.zeros(times.size, dtype=np.float64)
        premiums = np.zeros(times.size, dtype=np.float64)
        expenses = np.zeros(times.size, dtype=np.float64)
        for projection in projections:
            indices = np.searchsorted(times, projection.times)
            benefits[indices] += projection.benefits
            premiums[indices] += projection.premiums
            expenses[indices] += projection.expenses
        return LifeCashFlowProjection(times, benefits, premiums, expenses)
