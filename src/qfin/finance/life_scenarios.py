"""Chunked life model-point scenarios and deterministic sensitivity reports."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isclose
from operator import index as integer_index
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from numpy.typing import NDArray

from qfin import _native
from qfin.finance.curves import YieldCurve
from qfin.finance.fixed_income import Engine
from qfin.finance.life import (
    LifePolicy,
    PolicyModelPointSet,
    ProjectionAssumptions,
    _as_model_points,
    _path_value,
    _policy_buffers,
    _validate_horizon,
    project_liabilities,
)
from qfin.finance.scenarios import EconomicScenarioSet

FloatArray = NDArray[np.float64]

if TYPE_CHECKING:
    from qfin.finance.risk import LossDistribution


@dataclass(frozen=True, slots=True)
class LifeScenarioResult:
    """Scenario-level liability PV and aggregate cash-flow components."""

    labels: tuple[str, ...]
    probabilities: FloatArray
    present_values: FloatArray
    expected_premiums: FloatArray
    expected_benefits: FloatArray
    expected_expenses: FloatArray
    expected_surrenders: FloatArray
    base_present_value: float
    model_point_count: int
    policy_count: float
    working_set_estimate_bytes: int
    engine: Literal["numpy", "native"]

    def loss_distribution(self) -> LossDistribution:
        """Return liability PV deterioration for classical or quantum tail risk."""

        from qfin.finance.risk import LossDistribution

        return LossDistribution(self.present_values - self.base_present_value, self.probabilities)


@dataclass(frozen=True, slots=True)
class LifeSensitivityReport:
    """Forward bump impacts on aggregate liability present value."""

    base_present_value: float
    mortality_impact: float
    lapse_impact: float
    rate_impact: float
    expense_impact: float
    mortality_relative_bump: float
    lapse_absolute_bump: float
    rate_absolute_bump: float
    expense_relative_bump: float
    engine: Literal["numpy", "native"]

    def to_dict(self) -> dict[str, float | str]:
        return {
            "base_present_value": self.base_present_value,
            "mortality_impact": self.mortality_impact,
            "lapse_impact": self.lapse_impact,
            "rate_impact": self.rate_impact,
            "expense_impact": self.expense_impact,
            "mortality_relative_bump": self.mortality_relative_bump,
            "lapse_absolute_bump": self.lapse_absolute_bump,
            "rate_absolute_bump": self.rate_absolute_bump,
            "expense_relative_bump": self.expense_relative_bump,
            "engine": self.engine,
        }


def _numpy_scenario_chunk(
    model_points: PolicyModelPointSet,
    assumptions: ProjectionAssumptions,
    scenarios: EconomicScenarioSet,
    scenario_start: int,
    scenario_stop: int,
) -> dict[str, FloatArray]:
    scenario_count = scenario_stop - scenario_start
    present_values = np.zeros(scenario_count, dtype=np.float64)
    premium_totals = np.zeros(scenario_count, dtype=np.float64)
    benefit_totals = np.zeros(scenario_count, dtype=np.float64)
    expense_totals = np.zeros(scenario_count, dtype=np.float64)
    surrender_totals = np.zeros(scenario_count, dtype=np.float64)
    lapse_rates = cast(FloatArray, assumptions.lapse_rate)
    disability_rates = cast(FloatArray, assumptions.disability_rate)
    recovery_rates = cast(FloatArray, assumptions.recovery_rate)
    crediting_rates = cast(FloatArray, assumptions.crediting_rate)
    base_inflation = cast(FloatArray, assumptions.expense_inflation_rate)

    for local_scenario, scenario in enumerate(range(scenario_start, scenario_stop)):
        scenario_pv = 0.0
        for policy, count in zip(model_points.policies, model_points.counts, strict=True):
            active = 1.0
            disabled = 0.0
            account_value = policy.account_value
            inflation_index = 1.0
            discount = 1.0
            for year in range(policy.remaining_term):
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
                        assumptions.mortality.qx(policy.age + year)
                        * assumptions.mortality_multiplier
                        * scenarios.mortality_multipliers[scenario, year],
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
                lapse = float(
                    np.clip(
                        _path_value(lapse_rates, year)
                        * scenarios.lapse_multipliers[scenario, year],
                        0.0,
                        1.0,
                    )
                )
                lapse_count = (active_before_lapse + disabled_before_lapse) * lapse
                active = active_before_lapse * (1.0 - lapse)
                disabled = disabled_before_lapse * (1.0 - lapse)
                inflation_index *= (1.0 + _path_value(base_inflation, year)) * (
                    1.0 + scenarios.inflation_rates[scenario, year]
                )
                benefit_scale = (
                    assumptions.benefit_multiplier
                    * inflation_index**policy.benefit_inflation_linkage
                )
                deaths = active_deaths + disabled_deaths
                if policy.product_type == "annuity":
                    death_benefit = 0.0
                    annuity_benefit = (
                        (active_survivors + disabled_survivors)
                        * policy.annual_benefit
                        * benefit_scale
                    )
                else:
                    insured_amount = policy.sum_assured
                    if policy.product_type == "participating_life":
                        insured_amount *= (1.0 + policy.bonus_rate) ** (
                            policy.policy_duration + year + 1
                        )
                    elif policy.product_type == "universal_life":
                        insured_amount = max(insured_amount, account_value)
                    death_benefit = deaths * insured_amount * benefit_scale
                    annuity_benefit = 0.0
                disability_benefit = (
                    disabled_before_lapse * policy.disability_benefit * benefit_scale
                )
                surrender = (
                    lapse_count * account_value if policy.product_type == "universal_life" else 0.0
                )
                maturity = 0.0
                if year + 1 == policy.remaining_term:
                    ending_in_force = active + disabled
                    if policy.product_type == "participating_life":
                        maturity = (
                            ending_in_force
                            * policy.sum_assured
                            * (1.0 + policy.bonus_rate) ** policy.term
                            * benefit_scale
                        )
                    elif policy.product_type == "universal_life":
                        maturity = ending_in_force * account_value * benefit_scale
                end_benefit = death_benefit + annuity_benefit + disability_benefit + maturity
                start_net = expense - premium
                base_forward = assumptions.curve.forward_rate(float(year), float(year + 1))
                short_shock = float(
                    np.interp(
                        1.0,
                        assumptions.curve.times,
                        scenarios.rate_shocks[scenario, year],
                        left=scenarios.rate_shocks[scenario, year, 0],
                        right=scenarios.rate_shocks[scenario, year, -1],
                    )
                )
                end_discount = discount * np.exp(-(base_forward + short_shock))
                end_net = end_benefit + surrender
                scenario_pv += count * (start_net * discount + end_net * end_discount)
                premium_totals[local_scenario] += count * premium
                benefit_totals[local_scenario] += count * end_benefit
                expense_totals[local_scenario] += count * expense
                surrender_totals[local_scenario] += count * surrender
                discount = end_discount
        present_values[local_scenario] = scenario_pv
    return {
        "present_values": present_values,
        "expected_premiums": premium_totals,
        "expected_benefits": benefit_totals,
        "expected_expenses": expense_totals,
        "expected_surrenders": surrender_totals,
    }


def project_liability_scenarios(
    policies: PolicyModelPointSet | list[LifePolicy] | tuple[LifePolicy, ...],
    assumptions: ProjectionAssumptions,
    scenarios: EconomicScenarioSet,
    *,
    engine: Engine = "auto",
    scenario_chunk_size: int = 256,
    policy_chunk_size: int = 4_096,
) -> LifeScenarioResult:
    """Project scenario-by-model-point workloads without a full result cube."""

    model_points = _as_model_points(policies)
    scenarios.validate_curve(assumptions.curve)
    if not isclose(scenarios.period_length, 1.0, abs_tol=1.0e-12):
        raise ValueError("the annual life engine requires period_length=1")
    maximum_term = max((policy.remaining_term for policy in model_points.policies), default=0)
    _validate_horizon(assumptions, maximum_term)
    if maximum_term > scenarios.period_count:
        raise ValueError("economic scenarios must cover the full policy horizon")
    mismatched = [
        policy.mortality_category
        for policy in model_points.policies
        if policy.mortality_category != assumptions.mortality.category
    ]
    if mismatched:
        raise ValueError("all policies must match the supplied mortality-table category")
    try:
        scenario_chunk = integer_index(scenario_chunk_size)
        policy_chunk = integer_index(policy_chunk_size)
    except TypeError as exc:
        raise ValueError("chunk sizes must be positive integers") from exc
    if (
        isinstance(scenario_chunk_size, bool)
        or isinstance(policy_chunk_size, bool)
        or scenario_chunk <= 0
        or policy_chunk <= 0
    ):
        raise ValueError("chunk sizes must be positive integers")
    if engine not in ("auto", "numpy", "native"):
        raise ValueError("engine must be 'auto', 'numpy', or 'native'")
    workload = scenarios.scenario_count * model_points.model_point_count * maximum_term
    selected: Literal["numpy", "native"]
    if engine == "native":
        _native.require()
        selected = "native"
    elif engine == "numpy":
        selected = "numpy"
    else:
        selected = "native" if _native.available() and workload > 0 else "numpy"
    output = {
        name: np.zeros(scenarios.scenario_count, dtype=np.float64)
        for name in (
            "present_values",
            "expected_premiums",
            "expected_benefits",
            "expected_expenses",
            "expected_surrenders",
        )
    }
    for scenario_start in range(0, scenarios.scenario_count, scenario_chunk):
        scenario_stop = min(scenario_start + scenario_chunk, scenarios.scenario_count)
        for policy_start in range(0, model_points.model_point_count, policy_chunk):
            policy_stop = min(policy_start + policy_chunk, model_points.model_point_count)
            point_chunk = PolicyModelPointSet(
                model_points.policies[policy_start:policy_stop],
                model_points.counts[policy_start:policy_stop],
            )
            if selected == "numpy":
                values = _numpy_scenario_chunk(
                    point_chunk,
                    assumptions,
                    scenarios,
                    scenario_start,
                    scenario_stop,
                )
            else:
                raw = cast(
                    dict[str, object],
                    _native.require().project_life_scenarios(
                        *_policy_buffers(point_chunk),
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
                        np.ascontiguousarray(scenarios.rate_shocks[scenario_start:scenario_stop]),
                        np.ascontiguousarray(
                            scenarios.mortality_multipliers[scenario_start:scenario_stop]
                        ),
                        np.ascontiguousarray(
                            scenarios.lapse_multipliers[scenario_start:scenario_stop]
                        ),
                        np.ascontiguousarray(
                            scenarios.inflation_rates[scenario_start:scenario_stop]
                        ),
                    ),
                )
                values = {name: np.asarray(raw[name], dtype=np.float64) for name in output}
            for name in output:
                output[name][scenario_start:scenario_stop] += values[name]

    base = project_liabilities(model_points, assumptions, engine=selected)
    peak_scenarios = min(scenario_chunk, scenarios.scenario_count)
    peak_points = min(policy_chunk, max(model_points.model_point_count, 1))
    working_set = 8 * (
        peak_scenarios * (scenarios.period_count * (scenarios.curve_node_count + 3) + 5)
        + peak_points * 16
    )
    return LifeScenarioResult(
        labels=scenarios.labels,
        probabilities=scenarios.probabilities,
        present_values=output["present_values"],
        expected_premiums=output["expected_premiums"],
        expected_benefits=output["expected_benefits"],
        expected_expenses=output["expected_expenses"],
        expected_surrenders=output["expected_surrenders"],
        base_present_value=base.present_value,
        model_point_count=model_points.model_point_count,
        policy_count=model_points.policy_count,
        working_set_estimate_bytes=working_set,
        engine=selected,
    )


def life_sensitivities(
    policies: PolicyModelPointSet | list[LifePolicy] | tuple[LifePolicy, ...],
    assumptions: ProjectionAssumptions,
    *,
    mortality_relative_bump: float = 0.10,
    lapse_absolute_bump: float = 0.01,
    rate_absolute_bump: float = 0.001,
    expense_relative_bump: float = 0.10,
    engine: Engine = "auto",
) -> LifeSensitivityReport:
    """Return transparent forward bump-and-revalue liability impacts."""

    bumps = (
        mortality_relative_bump,
        lapse_absolute_bump,
        rate_absolute_bump,
        expense_relative_bump,
    )
    if any(not np.isfinite(value) or value <= 0.0 for value in bumps):
        raise ValueError("sensitivity bumps must be finite and positive")
    model_points = _as_model_points(policies)
    base = project_liabilities(model_points, assumptions, engine=engine)
    mortality = project_liabilities(
        model_points,
        replace(
            assumptions,
            mortality_multiplier=assumptions.mortality_multiplier * (1.0 + mortality_relative_bump),
        ),
        engine=engine,
    )
    bumped_lapse = np.clip(cast(FloatArray, assumptions.lapse_rate) + lapse_absolute_bump, 0.0, 1.0)
    lapse = project_liabilities(
        model_points,
        replace(assumptions, lapse_rate=bumped_lapse),
        engine=engine,
    )
    shifted_curve = YieldCurve(
        assumptions.curve.times,
        assumptions.curve.zero_rates + rate_absolute_bump,
    )
    rates = project_liabilities(
        model_points,
        replace(assumptions, curve=shifted_curve),
        engine=engine,
    )
    expenses = project_liabilities(
        model_points,
        replace(
            assumptions,
            expense_per_policy=assumptions.expense_per_policy * (1.0 + expense_relative_bump),
        ),
        engine=engine,
    )
    return LifeSensitivityReport(
        base_present_value=base.present_value,
        mortality_impact=mortality.present_value - base.present_value,
        lapse_impact=lapse.present_value - base.present_value,
        rate_impact=rates.present_value - base.present_value,
        expense_impact=expenses.present_value - base.present_value,
        mortality_relative_bump=mortality_relative_bump,
        lapse_absolute_bump=lapse_absolute_bump,
        rate_absolute_bump=rate_absolute_bump,
        expense_relative_bump=expense_relative_bump,
        engine=base.engine,
    )


__all__ = [
    "LifeScenarioResult",
    "LifeSensitivityReport",
    "life_sensitivities",
    "project_liability_scenarios",
]
