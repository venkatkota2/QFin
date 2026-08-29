#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace qfin {

struct PolicyProjectionResult {
    std::vector<double> expected_premiums;
    std::vector<double> expected_benefits;
    std::vector<double> expected_expenses;
    std::vector<double> net_liability_cashflows;
    std::vector<double> in_force;
    std::vector<double> policy_present_values;
    double present_value{};
    double duration{};
};

struct LifeModelPointProjectionResult {
    std::vector<double> expected_premiums;
    std::vector<double> expected_benefits;
    std::vector<double> expected_expenses;
    std::vector<double> expected_surrenders;
    std::vector<double> net_liability_cashflows;
    std::vector<double> active;
    std::vector<double> disabled;
    std::vector<double> deaths;
    std::vector<double> model_point_present_values;
    double present_value{};
    double duration{};
};

struct LifeScenarioProjectionResult {
    std::vector<double> present_values;
    std::vector<double> expected_premiums;
    std::vector<double> expected_benefits;
    std::vector<double> expected_expenses;
    std::vector<double> expected_surrenders;
};

[[nodiscard]] PolicyProjectionResult project_term_life_policies(
    std::span<const double> attained_ages,
    std::span<const double> sums_assured,
    std::span<const double> annual_premiums,
    std::span<const std::int32_t> remaining_terms,
    std::span<const double> table_ages,
    std::span<const double> mortality_qx,
    std::span<const double> lapse_rates,
    double expense_per_policy,
    double mortality_multiplier,
    std::span<const double> curve_times,
    std::span<const double> zero_rates
);

[[nodiscard]] LifeModelPointProjectionResult project_life_model_points(
    std::span<const double> attained_ages,
    std::span<const double> sums_assured,
    std::span<const double> annual_premiums,
    std::span<const std::int32_t> remaining_terms,
    std::span<const std::int32_t> policy_durations,
    std::span<const std::int32_t> original_terms,
    std::span<const std::int32_t> product_codes,
    std::span<const double> model_point_counts,
    std::span<const double> annual_benefits,
    std::span<const double> account_values,
    std::span<const double> annual_charges,
    std::span<const double> crediting_spreads,
    std::span<const double> bonus_rates,
    std::span<const double> disability_benefits,
    std::span<const double> benefit_inflation_linkage,
    std::span<const double> table_ages,
    std::span<const double> mortality_qx,
    std::span<const double> lapse_rates,
    std::span<const double> disability_rates,
    std::span<const double> recovery_rates,
    std::span<const double> crediting_rates,
    std::span<const double> expense_inflation_rates,
    double expense_per_policy,
    double mortality_multiplier,
    double disabled_mortality_multiplier,
    double premium_multiplier,
    double benefit_multiplier,
    std::span<const double> curve_times,
    std::span<const double> zero_rates
);

[[nodiscard]] LifeScenarioProjectionResult project_life_scenarios(
    std::span<const double> attained_ages,
    std::span<const double> sums_assured,
    std::span<const double> annual_premiums,
    std::span<const std::int32_t> remaining_terms,
    std::span<const std::int32_t> policy_durations,
    std::span<const std::int32_t> original_terms,
    std::span<const std::int32_t> product_codes,
    std::span<const double> model_point_counts,
    std::span<const double> annual_benefits,
    std::span<const double> account_values,
    std::span<const double> annual_charges,
    std::span<const double> crediting_spreads,
    std::span<const double> bonus_rates,
    std::span<const double> disability_benefits,
    std::span<const double> benefit_inflation_linkage,
    std::span<const double> table_ages,
    std::span<const double> mortality_qx,
    std::span<const double> lapse_rates,
    std::span<const double> disability_rates,
    std::span<const double> recovery_rates,
    std::span<const double> crediting_rates,
    std::span<const double> base_expense_inflation_rates,
    double expense_per_policy,
    double mortality_multiplier,
    double disabled_mortality_multiplier,
    double premium_multiplier,
    double benefit_multiplier,
    std::span<const double> curve_times,
    std::span<const double> zero_rates,
    std::span<const double> scenario_rate_shocks,
    std::span<const double> scenario_mortality_multipliers,
    std::span<const double> scenario_lapse_multipliers,
    std::span<const double> scenario_inflation_rates,
    std::size_t scenario_count,
    std::size_t period_count
);

}  // namespace qfin
